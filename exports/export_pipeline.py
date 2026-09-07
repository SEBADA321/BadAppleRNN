"""
exports/export_pipeline.py
---------------------------
Automated packaging and metrics export pipeline for Bad Apple Neural Video Synthesis.
Extracts models from best_model.pt, packages:
  1. bad_apple_complete_training.pt  - Full resumable checkpoint with optimizer states, tables, and config
  2. bad_apple_inference_only.pt      - Ultra-compact standalone model (h0, c0, ctf, fd) for autonomous unroll
Aggregates metrics, generates data tables (CSV/JSON), and produces analytics visual dashboards.
"""

import os
import re
import csv
import json
import shutil
import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec


# ==============================================================================
# Model Architecture Definitions (Self-contained for packaging & verification)
# ==============================================================================

class CustomTransitionFunction(nn.Module):
    """LSTM-like recurrent transition function with separate gating weights."""
    def __init__(self, hidden_size: int = 64, init_weights: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_hf = nn.Linear(hidden_size, hidden_size)
        self.W_hi = nn.Linear(hidden_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, hidden_size)
        self.W_hg = nn.Linear(hidden_size, hidden_size)
        if init_weights:
            self.init_weights()

    def init_weights(self) -> None:
        for name, param in self.named_parameters():
            if name.endswith(".weight") and 'W_h' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        self.W_hf.bias.data.fill_(1.0)

    def forward(self, h, c):
        f = torch.sigmoid(self.W_hf(h))
        i = torch.sigmoid(self.W_hi(h))
        o = torch.sigmoid(self.W_ho(h))
        c_t = torch.tanh(self.W_hg(h))
        c_next = f * c + i * c_t
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-Separable 2D Convolution with GroupNorm and SiLU activation."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class FrameDecoder(nn.Module):
    """Bilinear upsampling decoder mapping 64-dim latent vectors to [1, 384, 512] logits."""
    def __init__(
        self,
        hidden_size: int = 64,
        output_height: int = 384,
        output_width: int = 512,
    ):
        super().__init__()
        self.output_height = output_height
        self.output_width = output_width

        self.base_channels = 8
        self.base_height = output_height // 16  # 24
        self.base_width = output_width // 16    # 32

        if output_height % 16 != 0 or output_width % 16 != 0:
            raise ValueError("output_height and output_width must be divisible by 16")

        self.projection = nn.Linear(
            hidden_size,
            self.base_channels * self.base_height * self.base_width,
        )
        self.block1 = DepthwiseSeparableConv(8, 16)
        self.block2 = DepthwiseSeparableConv(16, 16)
        self.block3 = DepthwiseSeparableConv(16, 8)
        self.output = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        original_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        x = self.projection(x)
        x = x.reshape(-1, self.base_channels, self.base_height, self.base_width)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block1(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block2(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block3(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.output(x)

        x = x.reshape(*original_shape, 1, self.output_height, self.output_width)
        return x


class TrainingTableEmbeddings(nn.Module):
    """Lookup tables holding the ground-truth hidden and cell state trajectories."""
    def __init__(self, h_dim: int = 64, c_dim: int = 64, entries: int = 6573, init_weights: bool = True):
        super().__init__()
        self.h_dim = h_dim
        self.c_dim = c_dim
        self.entries = entries
        self.h_table = nn.Embedding(num_embeddings=self.entries, embedding_dim=self.h_dim)
        self.c_table = nn.Embedding(num_embeddings=self.entries, embedding_dim=self.c_dim)
        if init_weights:
            self.init_weights()

    def init_weights(self) -> None:
        nn.init.xavier_normal_(self.h_table.weight)
        nn.init.xavier_normal_(self.c_table.weight)

    def forward(self, indices):
        return self.h_table(indices), self.c_table(indices)


# ==============================================================================
# Model Packaging Functions
# ==============================================================================

def get_shared_config(entries: int = 6573, hidden_size: int = 64, cell_size: int = 64):
    """Builds the comprehensive configuration dictionary."""
    return {
        "hidden_size": hidden_size,
        "cell_size": cell_size,
        "entries": entries,
        "output_height": 384,
        "output_width": 512,
        "fps": 30,
        "video_duration_seconds": round(entries / 30.0, 2),
        "base_channels": 8,
        "base_height": 24,
        "base_width": 32,
        "upsample_stages": 4,
        "upsample_mode": "bilinear",
        "align_corners": False,
        "architecture": {
            "transition_function": "CustomTransitionFunction (LSTM-style linear gating with orthogonal init)",
            "frame_decoder": "FrameDecoder (4-stage Bilinear + DepthwiseSeparableConv blocks)",
            "table_embeddings": "TrainingTableEmbeddings (Direct frame-wise latent table lookups)"
        },
        "decoder_layers": [
            {"layer": "projection", "type": "nn.Linear", "in_features": 64, "out_features": 6144, "comment": "Projects latent to (8, 24, 32)"},
            {"layer": "block1", "type": "DepthwiseSeparableConv", "in_channels": 8, "out_channels": 16, "kernel_size": 3, "upsample": "2x (48x64)"},
            {"layer": "block2", "type": "DepthwiseSeparableConv", "in_channels": 16, "out_channels": 16, "kernel_size": 3, "upsample": "2x (96x128)"},
            {"layer": "block3", "type": "DepthwiseSeparableConv", "in_channels": 16, "out_channels": 8, "kernel_size": 3, "upsample": "2x (192x256)"},
            {"layer": "output", "type": "nn.Conv2d", "in_channels": 8, "out_channels": 1, "kernel_size": 1, "upsample": "2x (384x512)", "activation": "None (logits)"}
        ],
        "training_hyperparameters": {
            "optimizer_adam": {"lr": 1e-5, "weight_decay": 0.001, "targets": "All weights & biases except W_h recurrence"},
            "optimizer_muon": {"lr": 0.005, "momentum": 0.95, "targets": "Transition function recurrence weights W_h*"},
            "momentum_scaling": {"factor": 0.2, "interval_epochs": 10},
            "regularization": {"state_noise_std": 0.005, "lambda_dyn_h": 0.7, "lambda_dyn_c": 0.7},
            "decoder_chunk_size": 32,
            "frame_budget": 512,
            "curriculum_schedule": "K = min(2 * 2**(epoch // 50), 512)",
            "curriculum_phase_length": 50,
            "mixed_precision": "autocast(cuda, float16)"
        },
        "data_specifications": {
            "training_dataset_frames": entries,
            "training_frame_resolution": [384, 512],
            "training_frame_channels": 1,
            "training_frame_dtype": "uint8 (0-255) / float32 (0.0-1.0)",
            "raw_dataset_bytes": entries * 384 * 512,
            "raw_dataset_size_gb": round((entries * 384 * 512) / (1024**3), 3),
            "output_frame_shape": [1, 384, 512],
            "output_video_duration": f"{round(entries / 30.0, 2)}s (3m 39.1s)"
        }
    }


def package_complete_training_model(source_ckpt_path: str, output_path: str):
    """
    Packages the complete model with all optimizer states and table embeddings
    so that training can be resumed seamlessly.
    """
    print(f"-> Packaging COMPLETE RESUMABLE model from: {source_ckpt_path}")
    raw_ckpt = torch.load(source_ckpt_path, map_location="cpu")

    tte_sd = {k: v.detach().cpu().clone() for k, v in raw_ckpt["tte"].items()}
    ctf_sd = {k: v.detach().cpu().clone() for k, v in raw_ckpt["ctf"].items()}
    fd_sd = {k: v.detach().cpu().clone() for k, v in raw_ckpt["fd"].items()}

    h0 = tte_sd["h_table.weight"][0].detach().cpu().clone()
    c0 = tte_sd["c_table.weight"][0].detach().cpu().clone()

    entries = tte_sd["h_table.weight"].shape[0]
    hidden_size = tte_sd["h_table.weight"].shape[1]
    cell_size = tte_sd["c_table.weight"].shape[1]

    tte_params = sum(p.numel() for p in tte_sd.values())
    ctf_params = sum(p.numel() for p in ctf_sd.values())
    fd_params = sum(p.numel() for p in fd_sd.values())
    total_params = tte_params + ctf_params + fd_params

    config = get_shared_config(entries=entries, hidden_size=hidden_size, cell_size=cell_size)

    # Detach optimizer states to CPU
    opt_adam = raw_ckpt.get("optimizer_adam", None)
    opt_muon = raw_ckpt.get("optimizer_muon", None)
    scaler = raw_ckpt.get("scaler", None)

    metadata = {
        "package_type": "complete_training_resumable",
        "epoch": raw_ckpt.get("epoch", None),
        "loss": raw_ckpt.get("loss", None),
        "best_loss": raw_ckpt.get("best_loss", None),
        "total_parameters": total_params,
        "parameter_breakdown": {
            "TrainingTableEmbeddings": tte_params,
            "CustomTransitionFunction": ctf_params,
            "FrameDecoder": fd_params
        },
        "model_sizes": {
            "total_parameters": total_params,
            "weights_fp32_bytes": total_params * 4,
            "weights_fp32_mb": round((total_params * 4) / (1024 * 1024), 2),
            "resumable_file_size_mb": round(os.path.getsize(source_ckpt_path) / (1024 * 1024), 2)
        },
        "training_data_size": {
            "num_frames": entries,
            "resolution": "384x512",
            "channels": 1,
            "raw_dataset_bytes": entries * 384 * 512,
            "raw_dataset_size_gb": round((entries * 384 * 512) / (1024**3), 3)
        },
        "output_size": {
            "frame_shape": [1, 384, 512],
            "fps": 30,
            "duration_seconds": round(entries / 30.0, 2),
            "video_frames": entries
        },
        "source_checkpoint": str(source_ckpt_path),
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

    payload = {
        "epoch": raw_ckpt.get("epoch", 0),
        "loss": raw_ckpt.get("loss", None),
        "best_loss": raw_ckpt.get("best_loss", None),
        "h0": h0,
        "c0": c0,
        "tte_state_dict": tte_sd,
        "ctf_state_dict": ctf_sd,
        "fd_state_dict": fd_sd,
        "optimizer_adam_state_dict": opt_adam,
        "optimizer_muon_state_dict": opt_muon,
        "scaler_state_dict": scaler,
        "config": config,
        "metadata": metadata
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  [OK] Saved Complete Resumable Package: {output_path} ({size_mb:.2f} MB)")
    return payload


def package_inference_only_model(source_ckpt_path: str, output_path: str):
    """
    Packages the compact inference-only model containing ONLY:
    h0, c0, ctf_state_dict, fd_state_dict, and config.
    """
    print(f"-> Packaging COMPACT INFERENCE-ONLY model from: {source_ckpt_path}")
    raw_ckpt = torch.load(source_ckpt_path, map_location="cpu")

    tte_sd = raw_ckpt["tte"]
    ctf_sd = {k: v.detach().cpu().clone() for k, v in raw_ckpt["ctf"].items()}
    fd_sd = {k: v.detach().cpu().clone() for k, v in raw_ckpt["fd"].items()}

    h0 = tte_sd["h_table.weight"][0].detach().cpu().clone()
    c0 = tte_sd["c_table.weight"][0].detach().cpu().clone()

    entries = tte_sd["h_table.weight"].shape[0]
    hidden_size = tte_sd["h_table.weight"].shape[1]
    cell_size = tte_sd["c_table.weight"].shape[1]

    ctf_params = sum(p.numel() for p in ctf_sd.values())
    fd_params = sum(p.numel() for p in fd_sd.values())
    inference_params = ctf_params + fd_params

    config = get_shared_config(entries=entries, hidden_size=hidden_size, cell_size=cell_size)

    metadata = {
        "package_type": "inference_only",
        "epoch": raw_ckpt.get("epoch", None),
        "loss": raw_ckpt.get("loss", None),
        "best_loss": raw_ckpt.get("best_loss", None),
        "total_inference_parameters": inference_params,
        "parameter_breakdown": {
            "CustomTransitionFunction": ctf_params,
            "FrameDecoder": fd_params
        },
        "model_sizes": {
            "inference_parameters": inference_params,
            "weights_fp32_bytes": inference_params * 4,
            "weights_fp32_mb": round((inference_params * 4) / (1024 * 1024), 2),
            "file_size_mb": 1.60
        },
        "training_data_size": {
            "num_frames": entries,
            "resolution": "384x512",
            "channels": 1,
            "raw_dataset_bytes": entries * 384 * 512,
            "raw_dataset_size_gb": round((entries * 384 * 512) / (1024**3), 3)
        },
        "output_size": {
            "frame_shape": [1, 384, 512],
            "fps": 30,
            "duration_seconds": round(entries / 30.0, 2),
            "video_frames": entries
        },
        "source_checkpoint": str(source_ckpt_path),
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

    payload = {
        "h0": h0,
        "c0": c0,
        "ctf_state_dict": ctf_sd,
        "fd_state_dict": fd_sd,
        "config": config,
        "metadata": metadata
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    payload["metadata"]["model_sizes"]["file_size_mb"] = round(size_mb, 2)
    print(f"  [OK] Saved Compact Inference Package: {output_path} ({size_mb:.2f} MB, {inference_params:,} params)")
    return payload


# ==============================================================================
# Metrics & Logs Exporter (CSV, JSON)
# ==============================================================================

def export_training_metrics(log_path: str, export_tables_dir: str):
    """
    Parses logs.txt and exports clean CSV, JSON, and aggregated curriculum summaries.
    """
    print(f"-> Parsing training logs from: {log_path}")
    pattern = re.compile(
        r"Epoch\s+(\d+)\s*\|\s*K=\s*(\d+)\s*\|\s*Batch=\s*(\d+)\s*\|\s*Loss=([0-9.]+)\s*\|\s*Recon=([0-9.]+)\s*\|\s*DynH=([0-9.]+)\s*\|\s*DynC=([0-9.]+)"
    )

    records = []
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                loss_val = float(m.group(4))
                dyn_h = float(m.group(6))
                dyn_c = float(m.group(7))
                records.append({
                    "epoch": int(m.group(1)),
                    "k": int(m.group(2)),
                    "batch_size": int(m.group(3)),
                    "loss": loss_val,
                    "recon_loss": float(m.group(5)),
                    "dyn_h_loss": dyn_h,
                    "dyn_c_loss": dyn_c,
                    "dyn_total_loss": dyn_h + dyn_c,
                    "dyn_loss_pct": ((dyn_h + dyn_c) / loss_val) * 100.0 if loss_val > 0 else 0.0
                })

    os.makedirs(export_tables_dir, exist_ok=True)

    # 1. Full Epoch-by-Epoch CSV & JSON
    csv_path = os.path.join(export_tables_dir, "training_summary.csv")
    json_path = os.path.join(export_tables_dir, "training_summary.json")

    fieldnames = [
        "epoch", "k", "batch_size", "loss", "recon_loss", 
        "dyn_h_loss", "dyn_c_loss", "dyn_total_loss", "dyn_loss_pct"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  [OK] Exported {len(records)} epochs to {csv_path} and {json_path}")

    # 2. Aggregated Curriculum Stage Breakdown
    phase_dict = {}
    for r in records:
        p = r["epoch"] // 50
        if p not in phase_dict:
            phase_dict[p] = []
        phase_dict[p].append(r)

    stage_groups = []
    k_counts = {}

    for phase_id in sorted(phase_dict.keys()):
        group = phase_dict[phase_id]
        k_val = group[0]["k"]
        k_counts[k_val] = k_counts.get(k_val, 0) + 1
        set_num = k_counts[k_val]
        stage_name = f"K={k_val} (Set {set_num})" if set_num > 1 or len(phase_dict) > 9 else f"K={k_val}"

        losses = [x["loss"] for x in group]
        recons = [x["recon_loss"] for x in group]
        dyn_hs = [x["dyn_h_loss"] for x in group]
        dyn_cs = [x["dyn_c_loss"] for x in group]

        initial_loss = losses[0]
        final_loss = losses[-1]
        min_loss = min(losses)
        mean_loss = sum(losses) / len(losses)
        drop_pct = ((initial_loss - final_loss) / initial_loss) * 100.0 if initial_loss > 0 else 0.0

        stage_groups.append({
            "stage_id": int(phase_id) + 1,
            "stage_name": stage_name,
            "horizon_k": int(k_val),
            "batch_size": int(group[0]["batch_size"]),
            "start_epoch": int(group[0]["epoch"]),
            "end_epoch": int(group[-1]["epoch"]),
            "epoch_count": len(group),
            "initial_loss": round(initial_loss, 6),
            "final_loss": round(final_loss, 6),
            "min_loss": round(min_loss, 6),
            "mean_loss": round(mean_loss, 6),
            "loss_reduction_pct": round(drop_pct, 2),
            "mean_recon_loss": round(sum(recons) / len(recons), 6),
            "mean_dyn_h_loss": round(sum(dyn_hs) / len(dyn_hs), 6),
            "mean_dyn_c_loss": round(sum(dyn_cs) / len(dyn_cs), 6)
        })

    stage_csv_path = os.path.join(export_tables_dir, "curriculum_stages.csv")
    stage_fieldnames = list(stage_groups[0].keys())
    with open(stage_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stage_fieldnames)
        writer.writeheader()
        writer.writerows(stage_groups)
    print(f"  [OK] Exported curriculum stage summary to {stage_csv_path}")

    return records, stage_groups


# ==============================================================================
# Model Sizes, Data Sizes, and Statistical Tables Exporter
# ==============================================================================

def export_model_data_sizes_and_stats(complete_payload: dict, inf_payload: dict, export_tables_dir: str):
    """
    Exports comprehensive summary of Model Size, Training Data Size, and Output Size.
    """
    print("-> Computing parameter breakdown, data sizes, and table statistics...")
    tte_sd = complete_payload["tte_state_dict"]
    ctf_sd = complete_payload["ctf_state_dict"]
    fd_sd = complete_payload["fd_state_dict"]

    h_table = tte_sd["h_table.weight"].numpy()
    c_table = tte_sd["c_table.weight"].numpy()

    entries = int(h_table.shape[0])
    h_dim = int(h_table.shape[1])

    tte_params = int(sum(p.numel() for p in tte_sd.values()))
    ctf_params = int(sum(p.numel() for p in ctf_sd.values()))
    fd_params = int(sum(p.numel() for p in fd_sd.values()))
    complete_params = tte_params + ctf_params + fd_params
    inference_params = ctf_params + fd_params

    # Summary table dictionary
    summary_data = {
        "final_models": {
            "compact_inference_model": {
                "file_name": "bad_apple_inference_only.pt",
                "contents": "h0, c0, ctf_state_dict, fd_state_dict, config, metadata",
                "purpose": "Lightweight autonomous video rollout & frame decoding",
                "total_parameters": inference_params,
                "weights_fp32_bytes": inference_params * 4,
                "weights_fp32_mb": round((inference_params * 4) / (1024 * 1024), 2),
                "file_size_on_disk_mb": round(os.path.getsize(os.path.join(export_tables_dir, "../models/bad_apple_inference_only.pt")) / (1024*1024), 2)
            },
            "complete_resumable_model": {
                "file_name": "bad_apple_complete_training.pt",
                "contents": "epoch, loss, h0, c0, tte, ctf, fd, optimizer_adam, optimizer_muon, scaler, config",
                "purpose": "Complete training checkpoint enabling seamless training resumption",
                "total_parameters": complete_params,
                "weights_fp32_bytes": complete_params * 4,
                "weights_fp32_mb": round((complete_params * 4) / (1024 * 1024), 2),
                "file_size_on_disk_mb": round(os.path.getsize(os.path.join(export_tables_dir, "../models/bad_apple_complete_training.pt")) / (1024*1024), 2)
            }
        },
        "training_data_size": {
            "num_frames": entries,
            "resolution": "384 x 512 (Height x Width)",
            "channels": 1,
            "color_space": "Grayscale uint8",
            "raw_dataset_bytes": entries * 384 * 512,
            "raw_dataset_size_mb": round((entries * 384 * 512) / (1024 * 1024), 2),
            "raw_dataset_size_gb": round((entries * 384 * 512) / (1024**3), 3),
            "compressed_zip_size_mb": 19.43,
            "source_video_size_mb": 19.54,
            "extracted_png_frames_size_mb": 70.2
        },
        "output_size": {
            "frame_height": 384,
            "frame_width": 512,
            "channels": 1,
            "output_tensor_shape": "[1, 384, 512]",
            "framerate_fps": 30,
            "total_video_frames": entries,
            "duration_seconds": round(entries / 30.0, 2),
            "duration_formatted": f"{int((entries/30)//60)}m {((entries/30)%60):.1f}s",
            "uncompressed_video_stream_bytes": entries * 384 * 512,
            "uncompressed_video_stream_gb": round((entries * 384 * 512) / (1024**3), 3),
            "rendered_mp4_file_size_mb": 18.87
        }
    }

    summary_json_path = os.path.join(export_tables_dir, "model_sizes_and_data_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  [OK] Saved Model & Data Sizes Summary: {summary_json_path}")

    # Latent table statistics
    h_norms = np.linalg.norm(h_table, axis=1)
    c_norms = np.linalg.norm(c_table, axis=1)
    h_velocities = np.linalg.norm(np.diff(h_table, axis=0), axis=1)
    c_velocities = np.linalg.norm(np.diff(c_table, axis=0), axis=1)

    ctf = CustomTransitionFunction(hidden_size=h_dim, init_weights=False)
    ctf.load_state_dict(ctf_sd)
    ctf.eval()

    with torch.no_grad():
        h_tensor = torch.from_numpy(h_table[:-1])
        c_tensor = torch.from_numpy(c_table[:-1])
        h_pred, c_pred = ctf(h_tensor, c_tensor)
        h_step_errors = torch.norm(h_pred - torch.from_numpy(h_table[1:]), dim=1).numpy()
        c_step_errors = torch.norm(c_pred - torch.from_numpy(c_table[1:]), dim=1).numpy()

    table_stats = {
        "entries_total_frames": entries,
        "latent_dimension": h_dim,
        "h_table_norm": {
            "mean": float(np.mean(h_norms)),
            "std": float(np.std(h_norms)),
            "min": float(np.min(h_norms)),
            "max": float(np.max(h_norms))
        },
        "c_table_norm": {
            "mean": float(np.mean(c_norms)),
            "std": float(np.std(c_norms)),
            "min": float(np.min(c_norms)),
            "max": float(np.max(c_norms))
        },
        "h_velocity_drift_per_frame": {
            "mean": float(np.mean(h_velocities)),
            "std": float(np.std(h_velocities)),
            "median": float(np.median(h_velocities)),
            "max": float(np.max(h_velocities))
        },
        "c_velocity_drift_per_frame": {
            "mean": float(np.mean(c_velocities)),
            "std": float(np.std(c_velocities)),
            "median": float(np.median(c_velocities)),
            "max": float(np.max(c_velocities))
        },
        "ctf_1step_transition_error": {
            "h_mean_error": float(np.mean(h_step_errors)),
            "h_std_error": float(np.std(h_step_errors)),
            "c_mean_error": float(np.mean(c_step_errors)),
            "c_std_error": float(np.std(c_step_errors))
        }
    }

    table_stats_path = os.path.join(export_tables_dir, "table_embeddings_summary.json")
    with open(table_stats_path, "w") as f:
        json.dump(table_stats, f, indent=2)
    print(f"  [OK] Saved table embeddings stats to {table_stats_path}")

    # Parameter layer breakdown
    layers_info = []
    for module_name, sd in [("TrainingTableEmbeddings", tte_sd), ("CustomTransitionFunction", ctf_sd), ("FrameDecoder", fd_sd)]:
        for k, v in sd.items():
            layers_info.append({
                "module": module_name,
                "parameter_name": k,
                "shape": list(v.shape),
                "num_elements": v.numel(),
                "dtype": str(v.dtype).replace("torch.", "")
            })

    params_json_path = os.path.join(export_tables_dir, "model_parameters.json")
    with open(params_json_path, "w") as f:
        json.dump(layers_info, f, indent=2)
    print(f"  [OK] Saved model parameter layer breakdown to {params_json_path}")


# ==============================================================================
# Plot Generation (Losses, Manifolds, Recurrence, Dynamics)
# ==============================================================================

def generate_export_plots(records: list, stage_groups: list, complete_payload: dict, export_plots_dir: str):
    """
    Generates all high-resolution publication-quality PNG analytics dashboards.
    """
    print(f"-> Generating visual dashboards in: {export_plots_dir}")
    os.makedirs(export_plots_dir, exist_ok=True)

    epochs = np.array([r["epoch"] for r in records])
    losses = np.array([r["loss"] for r in records])
    recons = np.array([r["recon_loss"] for r in records])
    dyn_hs = np.array([r["dyn_h_loss"] for r in records])
    dyn_cs = np.array([r["dyn_c_loss"] for r in records])

    # Style definitions
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#CBD5E1'
    plt.rcParams['axes.linewidth'] = 1.0

    stage_colors = [
        '#2563EB', '#0D9488', '#16A34A', '#D97706', '#DC2626',
        '#9333EA', '#0284C7', '#E11D48', '#EA580C', '#CA8A04',
        '#059669', '#4F46E5', '#DB2777', '#0891B2', '#65A30D'
    ]

    # 1. Figure 1: losses_overview.png
    fig1 = plt.figure(figsize=(19, 13.0), dpi=300)
    gs1 = fig1.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

    # Subplot 1: Total Loss & Recon
    ax1 = fig1.add_subplot(gs1[0, 0])
    ax1.plot(epochs, losses, color='#0F172A', linewidth=2.2, label='Total Combined Loss', zorder=4)
    ax1.plot(epochs, recons, color='#3B82F6', linewidth=1.6, linestyle='--', label='Reconstruction Loss (MSE)', zorder=3)

    for idx, row in enumerate(stage_groups):
        c = stage_colors[idx % len(stage_colors)]
        ax1.axvspan(row["start_epoch"] - 0.5, row["end_epoch"] + 0.5, color=c, alpha=0.08, zorder=1)
        mid_ep = (row["start_epoch"] + row["end_epoch"]) / 2
        y_pos = 0.58 if idx % 2 == 0 else 0.47
        ax1.text(mid_ep, y_pos, f"K={row['horizon_k']}\nB={row['batch_size']}",
                 ha='center', va='center', fontsize=8.5, fontweight='bold', color=c,
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=c, alpha=0.92, linewidth=1.1))
        if row["start_epoch"] > 0:
            ax1.axvline(row["start_epoch"] - 0.5, color='#94A3B8', linestyle=':', linewidth=1.2)

    ax1.set_title("Total Loss & Reconstruction Loss Progression (Curriculum K ≤ 512)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Epoch", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Loss Magnitude", fontsize=10, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#CBD5E1', fontsize=9.5)
    ax1.set_xlim(-4, max(epochs) + 8)

    # Subplot 2: Latent Dynamics Losses on Log Scale
    ax2 = fig1.add_subplot(gs1[0, 1])
    ax2.plot(epochs, dyn_hs, color='#059669', linewidth=2.0, label='DynH (Hidden Recurrence Loss)', zorder=4)
    ax2.plot(epochs, dyn_cs, color='#EA580C', linewidth=2.0, label='DynC (Cell Recurrence Loss)', zorder=4)
    for idx, row in enumerate(stage_groups):
        c = stage_colors[idx % len(stage_colors)]
        ax2.axvspan(row["start_epoch"] - 0.5, row["end_epoch"] + 0.5, color=c, alpha=0.07, zorder=1)
    ax2.set_yscale('log')
    ax2.set_title("Latent Dynamics Loss Progression [Log Scale]", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Epoch", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Dynamics Loss (Log Scale)", fontsize=10, fontweight='bold')
    ax2.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#CBD5E1', fontsize=9.5)
    ax2.set_xlim(-4, max(epochs) + 8)

    # Subplot 3: Intra-Stage Convergence
    ax3 = fig1.add_subplot(gs1[1, 0])
    for idx, row in enumerate(stage_groups):
        mask = (epochs >= row["start_epoch"]) & (epochs <= row["end_epoch"])
        st_losses = losses[mask]
        c = stage_colors[idx % len(stage_colors)]
        lbl = f"Stage {row['stage_id']}: {row['stage_name']} ({len(st_losses)} eps: {st_losses[0]:.3f} → {st_losses[-1]:.3f})"
        ax3.plot(np.arange(len(st_losses)), st_losses, marker='o', markersize=3.5, linewidth=1.8, color=c, label=lbl)

    ax3.set_yscale('log')
    ax3.set_yticks([0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70])
    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax3.set_title("Intra-Stage Convergence Comparison (Relative Epochs 0 → 49) [Log Scale]", fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel("Relative Epoch in Stage", fontsize=10, fontweight='bold')
    ax3.set_ylabel("Loss (Log Scale)", fontsize=10, fontweight='bold')
    ax3.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#CBD5E1', fontsize=8.0)
    ax3.set_xlim(-1, 52)
    ax3.set_ylim(0.055, 0.80)

    # Subplot 4: Dynamics Fraction %
    ax4 = fig1.add_subplot(gs1[1, 1])
    dyn_pct = ((dyn_hs + dyn_cs) / losses) * 100.0
    ax4.plot(epochs, dyn_pct, color='#7C3AED', linewidth=2.2, label='Total Dynamics / Loss (%)', zorder=4)
    ax4.plot(epochs, (dyn_hs / losses) * 100.0, color='#059669', linestyle='--', linewidth=1.5, label='DynH / Loss (%)')
    ax4.plot(epochs, (dyn_cs / losses) * 100.0, color='#EA580C', linestyle='--', linewidth=1.5, label='DynC / Loss (%)')
    for idx, row in enumerate(stage_groups):
        c = stage_colors[idx % len(stage_colors)]
        ax4.axvspan(row["start_epoch"] - 0.5, row["end_epoch"] + 0.5, color=c, alpha=0.07, zorder=1)
    ax4.set_title("Dynamics Loss as Percentage of Total Loss", fontsize=12, fontweight='bold', pad=10)
    ax4.set_xlabel("Epoch", fontsize=10, fontweight='bold')
    ax4.set_ylabel("Dynamics Contribution (%)", fontsize=10, fontweight='bold')
    ax4.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#CBD5E1', fontsize=9.5)
    ax4.set_xlim(-4, max(epochs) + 8)

    fig1.suptitle("Bad Apple Latent Video Dynamics & Training Loss Overview", fontsize=15, fontweight='bold', y=0.985)
    loss_overview_path = os.path.join(export_plots_dir, "losses_overview.png")
    fig1.savefig(loss_overview_path, bbox_inches='tight', dpi=300)
    plt.close(fig1)
    print(f"  [OK] Saved {loss_overview_path}")

    # 2. Figure 2: losses_by_curriculum_stage.png
    num_stages = len(stage_groups)
    fig2, axes = plt.subplots(num_stages, 3, figsize=(18, max(14.0, 2.3 * num_stages)), dpi=250)
    if num_stages == 1:
        axes = np.atleast_2d(axes)
    fig2.subplots_adjust(hspace=0.48, wspace=0.25)

    for i, row in enumerate(stage_groups):
        mask = (epochs >= row["start_epoch"]) & (epochs <= row["end_epoch"])
        ep = epochs[mask]
        l = losses[mask]
        r = recons[mask]
        dh = dyn_hs[mask]
        dc = dyn_cs[mask]
        c = stage_colors[i % len(stage_colors)]

        # Col 1: Total & Recon
        ax_lr = axes[i, 0]
        ax_lr.plot(ep, l, color=c, marker='o', markersize=3.5, linewidth=2.0, label='Loss')
        ax_lr.plot(ep, r, color='#64748B', marker='s', markersize=3.0, linestyle='--', linewidth=1.5, label='Recon')
        ax_lr.set_title(f"Stage {row['stage_id']}: {row['stage_name']} (Eps {row['start_epoch']}-{row['end_epoch']})", fontsize=10, fontweight='bold')
        ax_lr.set_ylabel("Loss", fontsize=9)
        ax_lr.legend(loc='upper right', fontsize=8)
        drop_pct = row['loss_reduction_pct']
        drop_text = f"Drop: {-drop_pct:.1f}%\n({row['initial_loss']:.4f} → {row['final_loss']:.4f})"
        ax_lr.text(0.05, 0.15, drop_text, transform=ax_lr.transAxes, fontsize=8.5,
                   bbox=dict(boxstyle='square,pad=0.2', facecolor='white', alpha=0.85))

        # Col 2: Dynamics Losses
        ax_dyn = axes[i, 1]
        ax_dyn.plot(ep, dh, color='#059669', marker='o', markersize=3.5, linewidth=1.8, label='DynH')
        ax_dyn.plot(ep, dc, color='#EA580C', marker='^', markersize=3.5, linewidth=1.8, label='DynC')
        ax_dyn.set_title(f"{row['stage_name']}: DynH vs DynC", fontsize=10, fontweight='bold')
        ax_dyn.set_ylabel("Dynamics Loss", fontsize=9)
        ax_dyn.legend(loc='upper right', fontsize=8)
        ax_dyn.ticklabel_format(style='sci', scilimits=(0,0), axis='y')

        # Col 3: Dyn %
        ax_ratio = axes[i, 2]
        ax_ratio.plot(ep, ((dh + dc) / l) * 100.0, color='#7C3AED', marker='o', markersize=3.5, linewidth=2.0)
        ax_ratio.set_title(f"{row['stage_name']}: Dynamics % of Loss", fontsize=10, fontweight='bold')
        ax_ratio.set_ylabel("Dynamics %", fontsize=9)

    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch", fontsize=10, fontweight='bold')

    fig2.suptitle(f"Curriculum Stage-by-Stage Breakdown ({num_stages} Stages, K Up to 512)", fontsize=15, fontweight='bold', y=0.992)
    stage_breakdown_path = os.path.join(export_plots_dir, "losses_by_curriculum_stage.png")
    fig2.savefig(stage_breakdown_path, bbox_inches='tight', dpi=250)
    plt.close(fig2)
    print(f"  [OK] Saved {stage_breakdown_path}")

    # 3. Figure 3: tables_overview.png
    h_table = complete_payload["tte_state_dict"]["h_table.weight"].numpy()
    c_table = complete_payload["tte_state_dict"]["c_table.weight"].numpy()
    epoch_num = complete_payload.get("epoch", 638)

    T, D = h_table.shape
    fig3 = plt.figure(figsize=(20, 15), dpi=180)
    gs3 = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1.0, 1.0], hspace=0.32, wspace=0.22)

    ax3_1 = fig3.add_subplot(gs3[0, 0])
    im3_1 = ax3_1.imshow(h_table.T, aspect='auto', cmap='magma', interpolation='nearest')
    ax3_1.set_title(f"Hidden State Table ($h$) — {D} Channels $\\times$ {T} Frames (Best Model Epoch {epoch_num})", fontsize=13, fontweight='bold')
    ax3_1.set_xlabel("Frame Index (Time $t$)", fontsize=11)
    ax3_1.set_ylabel("Channel Index", fontsize=11)
    fig3.colorbar(im3_1, ax=ax3_1, fraction=0.046, pad=0.03).set_label("Hidden Value $h$", fontsize=10)
    ax3_1.grid(False)

    ax3_2 = fig3.add_subplot(gs3[0, 1])
    im3_2 = ax3_2.imshow(c_table.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax3_2.set_title(f"Cell State Table ($c$) — {D} Channels $\\times$ {T} Frames (Best Model Epoch {epoch_num})", fontsize=13, fontweight='bold')
    ax3_2.set_xlabel("Frame Index (Time $t$)", fontsize=11)
    ax3_2.set_ylabel("Channel Index", fontsize=11)
    fig3.colorbar(im3_2, ax=ax3_2, fraction=0.046, pad=0.03).set_label("Cell Value $c$", fontsize=10)
    ax3_2.grid(False)

    vel_h = np.linalg.norm(np.diff(h_table, axis=0), axis=1)
    vel_c = np.linalg.norm(np.diff(c_table, axis=0), axis=1)
    ax3_3 = fig3.add_subplot(gs3[1, 0])
    ax3_3.plot(vel_h, color='#2563eb', alpha=0.8, linewidth=1.2, label=r'Hidden State Velocity $\|h_{t+1}-h_t\|_2$')
    ax3_3.plot(vel_c, color='#059669', alpha=0.7, linewidth=1.2, label=r'Cell State Velocity $\|c_{t+1}-c_t\|_2$')
    ax3_3.set_title("Latent Trajectory Velocity (Step-to-Step Euclidean Drift)", fontsize=13, fontweight='bold')
    ax3_3.set_xlabel("Frame Index", fontsize=11)
    ax3_3.set_ylabel("Velocity Norm", fontsize=11)
    ax3_3.legend(loc='upper right', fontsize=10)

    norm_h = np.linalg.norm(h_table, axis=1)
    norm_c = np.linalg.norm(c_table, axis=1)
    ax3_4 = fig3.add_subplot(gs3[1, 1])
    ax3_4.plot(norm_h, color='#7c3aed', alpha=0.9, linewidth=1.3, label=rf'Hidden Norm $\|h_t\|_2$ (Mean: {np.mean(norm_h):.2f})')
    ax3_4.plot(norm_c, color='#d97706', alpha=0.8, linewidth=1.3, label=rf'Cell Norm $\|c_t\|_2$ (Mean: {np.mean(norm_c):.2f})')
    ax3_4.set_title("State Norm Dynamics across Sequence", fontsize=13, fontweight='bold')
    ax3_4.set_xlabel("Frame Index", fontsize=11)
    ax3_4.set_ylabel("Euclidean Norm", fontsize=11)
    ax3_4.legend(loc='upper right', fontsize=10)

    var_h = np.var(h_table, axis=0)
    sorted_h = np.argsort(var_h)[::-1]
    ax3_5 = fig3.add_subplot(gs3[2, 0])
    ax3_5.bar(range(D), var_h[sorted_h], color='#4f46e5', width=0.75, label='Channel Variance ($h$)')
    ax3_5.set_title(f"Channel Information Capacity (Sorted Variance across {T} Frames)", fontsize=13, fontweight='bold')
    ax3_5.set_xlabel("Sorted Channel Rank", fontsize=11)
    ax3_5.set_ylabel("Variance", fontsize=11)

    ax3_6 = fig3.add_subplot(gs3[2, 1])
    ax3_6.hist(h_table.flatten(), bins=80, density=True, alpha=0.65, color='#2563eb', label=f'Hidden ($h$): std={np.std(h_table):.2f}')
    ax3_6.hist(c_table.flatten(), bins=80, density=True, alpha=0.55, color='#059669', label=f'Cell ($c$): std={np.std(c_table):.2f}')
    ax3_6.set_title("Value Distribution Histograms", fontsize=13, fontweight='bold')
    ax3_6.set_xlabel("Activation Value", fontsize=11)
    ax3_6.set_ylabel("Probability Density", fontsize=11)
    ax3_6.legend(loc='upper right', fontsize=10)

    fig3.suptitle("Bad Apple Latent Table Embeddings Overview ($h$ & $c$ Tables)", fontsize=16, fontweight='bold', y=0.99)
    out_overview = os.path.join(export_plots_dir, "tables_overview.png")
    fig3.savefig(out_overview, bbox_inches='tight', dpi=180)
    plt.close(fig3)
    print(f"  [OK] Saved {out_overview}")

    # Copy existing pre-rendered 3D/PCA & Recurrence dashboards if present
    script_dir = Path(__file__).resolve().parent
    root_path = Path(export_plots_dir).parent.parent if script_dir.name == "exports" else script_dir
    for fname in ["tables_latent_manifold.png", "tables_temporal_recurrence.png", "tables_transition_dynamics.png"]:
        src_path = root_path / fname
        dst_path = os.path.join(export_plots_dir, fname)
        if src_path.exists():
            shutil.copy2(str(src_path), dst_path)
            print(f"  [OK] Exported {dst_path}")


# ==============================================================================
# Standalone Interactive 3D Viewer Exporter
# ==============================================================================

def export_interactive_visualizer(export_interactive_dir: str, root_dir: str):
    """
    Exports the standalone Three.js 3D WebGL point cloud visualizer.
    """
    os.makedirs(export_interactive_dir, exist_ok=True)
    src_html = os.path.join(root_dir, "tables_latent_manifold_3d_interactive.html")
    dst_html = os.path.join(export_interactive_dir, "tables_latent_manifold_3d_interactive.html")
    if os.path.exists(src_html):
        shutil.copy2(src_html, dst_html)
        print(f"  [OK] Exported standalone 3D visualizer to: {dst_html}")


# ==============================================================================
# Main Export Pipeline Execution
# ==============================================================================

def run_export_pipeline(root_dir: str | None = None):
    print("=" * 70)
    print(" BAD APPLE NEURAL VIDEO SYNTHESIS - PACKAGING & EXPORT PIPELINE")
    print("=" * 70)

    if root_dir is None:
        script_dir = Path(__file__).resolve().parent
        root_dir = str(script_dir.parent if script_dir.name == "exports" else script_dir)

    exports_dir = os.path.join(root_dir, "exports")
    models_dir = os.path.join(exports_dir, "models")
    tables_dir = os.path.join(exports_dir, "tables")
    plots_dir = os.path.join(exports_dir, "plots")
    interactive_dir = os.path.join(exports_dir, "interactive")

    # Select best model checkpoint as primary source
    best_ckpt_path = os.path.join(root_dir, "runs", "best_model.pt")
    last_ckpt_path = os.path.join(root_dir, "runs", "last_model.pt")
    source_ckpt = last_ckpt_path if os.path.exists(last_ckpt_path) else best_ckpt_path

    print(f"Using primary source checkpoint: {source_ckpt}")

    # 1. Package Complete Resumable Training Checkpoint
    complete_pkg_path = os.path.join(models_dir, "bad_apple_complete_training.pt")
    complete_payload = package_complete_training_model(source_ckpt, complete_pkg_path)

    # 2. Package Compact Inference-Only Model (h0, c0, ctf, fd)
    inference_pkg_path = os.path.join(models_dir, "bad_apple_inference_only.pt")
    inf_payload = package_inference_only_model(source_ckpt, inference_pkg_path)

    # 3. Export Metrics & Stage Tables
    log_path = os.path.join(root_dir, "logs.txt")
    records, stage_groups = export_training_metrics(log_path, tables_dir)

    # 4. Export Model Sizes, Data Sizes, and Statistical Tables
    export_model_data_sizes_and_stats(complete_payload, inf_payload, tables_dir)

    # 5. Export Plots & Visual Dashboards
    generate_export_plots(records, stage_groups, complete_payload, plots_dir)

    # 6. Export Interactive 3D Viewer
    export_interactive_visualizer(interactive_dir, root_dir)

    print("\n" + "=" * 70)
    print(f" EXPORT PIPELINE COMPLETED SUCCESSFULLY! Output at: {exports_dir}")
    print("=" * 70)


if __name__ == "__main__":
    run_export_pipeline()

