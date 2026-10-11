"""
exports/load_and_rollout.py
----------------------------
Versatile video synthesis and evaluation tool for Bad Apple Neural Models.
Supports:
  1. Side-by-Side Comparison: Ground Truth vs Autonomous Rollout (with real-time MSE dashboard)
  2. Tri-Panel Comparison: Ground Truth vs Direct Table Lookup vs Autonomous Rollout
  3. Single-Stream Video: Direct autonomous rollout or table reconstruction
  4. Interactive OpenCV Playback / Real-time scrubbing

Usage:
  # Generate Side-by-Side video (Ground Truth vs Autonomous Rollout)
  python load_and_rollout.py --mode side-by-side --checkpoint models/bad_apple_inference_only.pt --dataset ../frames --frames 6573 --out rollout_side_by_side.mp4

  # Generate Tri-Panel video (Ground Truth | Table Lookup | Autonomous Rollout)
  python load_and_rollout.py --mode tri-panel --checkpoint models/bad_apple_complete_training.pt --dataset ../frames --frames 6573 --out rollout_tri_panel.mp4

  # Generate standalone autonomous rollout video (512x384)
  python load_and_rollout.py --mode single --checkpoint models/bad_apple_inference_only.pt --frames 6573 --out rollout_single.mp4
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode
import cv2


# ==============================================================================
# Model Architecture Classes
# ==============================================================================

class CustomTransitionFunction(nn.Module):
    """LSTM-like recurrent transition function with separate gating weights."""
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_hf = nn.Linear(hidden_size, hidden_size)
        self.W_hi = nn.Linear(hidden_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, hidden_size)
        self.W_hg = nn.Linear(hidden_size, hidden_size)

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
        self.base_height = output_height // 16
        self.base_width = output_width // 16

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
    def __init__(self, h_dim: int = 64, c_dim: int = 64, entries: int = 6573):
        super().__init__()
        self.h_table = nn.Embedding(num_embeddings=entries, embedding_dim=h_dim)
        self.c_table = nn.Embedding(num_embeddings=entries, embedding_dim=c_dim)

    def forward(self, indices):
        return self.h_table(indices), self.c_table(indices)


# ==============================================================================
# Dataset Loader
# ==============================================================================

class BadAppleDataset:
    """Lazy or batch reader for ground-truth frame PNGs."""
    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path).resolve()
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_path}")

        self.frame_paths = sorted(
            p for p in self.dataset_path.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        )
        if not self.frame_paths:
            raise ValueError(f"No image frames found in: {self.dataset_path}")

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        # Returns float32 tensor [1, 384, 512] in range [0.0, 1.0]
        img = read_image(str(self.frame_paths[index]), mode=ImageReadMode.GRAY)
        return img.float().div_(255.0)


# ==============================================================================
# Model Loading Helper
# ==============================================================================

def resolve_path(relative_or_abs_path: str | None, candidates: list[str]) -> str | None:
    """Finds existing file among given candidates relative to cwd or script dir."""
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent if script_dir.name == "exports" else script_dir

    if relative_or_abs_path:
        p = Path(relative_or_abs_path)
        if p.exists():
            return str(p.resolve())
        # Try relative to workspace root
        if (workspace_root / relative_or_abs_path).exists():
            return str((workspace_root / relative_or_abs_path).resolve())

    for c in candidates:
        if Path(c).exists():
            return str(Path(c).resolve())
        if (workspace_root / c).exists():
            return str((workspace_root / c).resolve())

    return None


def load_packaged_model(checkpoint_path: str, device: str = "cuda"):
    """
    Loads checkpoint (packaged complete, packaged inference-only, or raw runs/*.pt)
    and returns models + config + metadata.
    """
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    print(f"Loading model checkpoint: {checkpoint_path} onto {dev}...")

    pkg = torch.load(checkpoint_path, map_location="cpu")

    # Check if this is a raw training checkpoint (runs/best_model.pt or runs/last_model.pt)
    if "ctf" in pkg and "fd" in pkg and not isinstance(pkg["ctf"], nn.Module):
        ctf_sd = pkg["ctf"]
        fd_sd = pkg["fd"]
        tte_sd = pkg.get("tte", None)
        epoch_info = pkg.get("epoch", "N/A")
        pkg_type = "raw_training_checkpoint"

        h_dim = 64
        c_dim = 64
        entries = 6573
        if tte_sd is not None and "h_table.weight" in tte_sd:
            entries = tte_sd["h_table.weight"].shape[0]
            h_dim = tte_sd["h_table.weight"].shape[1]
            c_dim = tte_sd["c_table.weight"].shape[1]

        cfg = {
            "hidden_size": h_dim,
            "cell_size": c_dim,
            "entries": entries,
            "output_height": 384,
            "output_width": 512,
        }

        ctf = CustomTransitionFunction(hidden_size=h_dim)
        fd = FrameDecoder(hidden_size=h_dim, output_height=384, output_width=512)
        ctf.load_state_dict(ctf_sd)
        fd.load_state_dict(fd_sd)
        ctf = ctf.to(dev).eval()
        fd = fd.to(dev).eval()

        if tte_sd is not None:
            h0 = tte_sd["h_table.weight"][0].detach().to(dev)
            c0 = tte_sd["c_table.weight"][0].detach().to(dev)
            tte = TrainingTableEmbeddings(h_dim=h_dim, c_dim=c_dim, entries=entries)
            tte.load_state_dict(tte_sd)
            tte = tte.to(dev).eval()
        else:
            h0 = torch.zeros(h_dim, device=dev)
            c0 = torch.zeros(c_dim, device=dev)
            tte = None

        models = {
            "ctf": ctf,
            "fd": fd,
            "h0": h0,
            "c0": c0,
            "device": dev
        }
        if tte is not None:
            models["tte"] = tte

        print(f"Loaded successfully ({pkg_type})! Source epoch: {epoch_info}.")
        return models, cfg, {"epoch": epoch_info, "package_type": pkg_type}

    # Packaged checkpoint
    cfg = pkg.get("config", {
        "hidden_size": 64,
        "cell_size": 64,
        "entries": 6573,
        "output_height": 384,
        "output_width": 512,
    })
    h_dim = cfg["hidden_size"]
    c_dim = cfg["cell_size"]
    entries = cfg.get("entries", 6573)
    out_h = cfg.get("output_height", 384)
    out_w = cfg.get("output_width", 512)

    ctf = CustomTransitionFunction(hidden_size=h_dim)
    fd = FrameDecoder(hidden_size=h_dim, output_height=out_h, output_width=out_w)

    ctf.load_state_dict(pkg["ctf_state_dict"])
    fd.load_state_dict(pkg["fd_state_dict"])

    ctf = ctf.to(dev).eval()
    fd = fd.to(dev).eval()

    h0 = pkg["h0"].to(dev)
    c0 = pkg["c0"].to(dev)

    models = {
        "ctf": ctf,
        "fd": fd,
        "h0": h0,
        "c0": c0,
        "device": dev
    }

    if "tte_state_dict" in pkg:
        tte = TrainingTableEmbeddings(h_dim=h_dim, c_dim=c_dim, entries=entries)
        tte.load_state_dict(pkg["tte_state_dict"])
        tte = tte.to(dev).eval()
        models["tte"] = tte

    pkg_type = pkg.get("metadata", {}).get("package_type", "packaged_model")
    epoch_info = pkg.get("metadata", {}).get("epoch", pkg.get("epoch", "N/A"))
    print(f"Loaded successfully ({pkg_type})! Source epoch: {epoch_info}.")
    return models, cfg, pkg.get("metadata", {})


# ==============================================================================
# Video Synthesis & Rendering Engine
# ==============================================================================

def render_synthesis_video(
    models: dict,
    dataset: BadAppleDataset | None = None,
    mode: str = "side-by-side",
    num_frames: int = 6573,
    start_frame: int = 0,
    out_mp4: str = "exports/rollout_side_by_side.mp4",
    fps: int = 30
):
    """
    Renders video in one of modes:
      - 'side-by-side': Ground Truth (Left) vs Autonomous Rollout (Right) with dashboard header
      - 'tri-panel': Ground Truth (Left) vs Table Lookup (Center) vs Autonomous Rollout (Right)
      - 'single': Standalone Autonomous Rollout (512x384)
      - 'table': Ground Truth vs Direct Table Reconstruction
    """
    dev = models["device"]
    ctf = models["ctf"]
    fd = models["fd"]
    has_table = "tte" in models
    tte = models.get("tte", None)

    total_frames = num_frames
    if dataset is not None:
        total_frames = min(num_frames, len(dataset) - start_frame)

    # Initial state
    with torch.no_grad():
        if start_frame == 0 or not has_table:
            h_curr = models["h0"].unsqueeze(0)
            c_curr = models["c0"].unsqueeze(0)
        else:
            t0 = torch.tensor([start_frame], device=dev)
            h_curr = tte.h_table(t0)
            c_curr = tte.c_table(t0)

    # Canvas dimensions
    if mode == "single":
        canvas_width, canvas_height = 512, 384
    elif mode == "tri-panel":
        canvas_width, canvas_height = 512 * 3, 384 + 60  # 1536 x 444
    else:  # side-by-side or table
        canvas_width, canvas_height = 512 * 2, 384 + 60  # 1024 x 444

    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)
    print(f"\nRendering [{mode.upper()}] video to {out_mp4} ({total_frames} frames @ {fps} FPS, {canvas_width}x{canvas_height})...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(out_mp4, fourcc, fps, (canvas_width, canvas_height))

    mse_history = []
    t_start = time.time()

    for idx in range(total_frames):
        current_t = start_frame + idx

        # 1. Ground Truth Frame (if dataset provided)
        gt_bgr = None
        gt_tensor = None
        if dataset is not None:
            gt_tensor = dataset[current_t].to(dev)  # [1, 384, 512]
            gt_np = (gt_tensor.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
            gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_GRAY2BGR)

        # 2. Autonomous Rollout Frame: fd(h_curr)
        with torch.no_grad():
            rollout_logit = fd(h_curr)
            rollout_prob = torch.sigmoid(rollout_logit).squeeze()
            rollout_np = (np.clip(rollout_prob.cpu().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8)
            rollout_bgr = cv2.cvtColor(rollout_np, cv2.COLOR_GRAY2BGR)

            # 3. Direct Table Lookup Frame (if complete model available)
            table_bgr = None
            table_prob = None
            if has_table and mode in ["tri-panel", "table"]:
                t_tensor = torch.tensor([current_t], device=dev)
                h_table_val = tte.h_table(t_tensor)
                table_logit = fd(h_table_val)
                table_prob = torch.sigmoid(table_logit).squeeze()
                table_np = (np.clip(table_prob.cpu().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8)
                table_bgr = cv2.cvtColor(table_np, cv2.COLOR_GRAY2BGR)

            # Compute MSE errors vs Ground Truth
            mse_rollout = 0.0
            mse_table = 0.0
            if gt_tensor is not None:
                mse_rollout = torch.mean((rollout_prob - gt_tensor.squeeze()) ** 2).item()
                if table_prob is not None:
                    mse_table = torch.mean((table_prob - gt_tensor.squeeze()) ** 2).item()
                
                if mode == "table":
                    mse_history.append(mse_table)
                else:
                    mse_history.append(mse_rollout)

            # Step autonomous recurrent state
            h_curr, c_curr = ctf(h_curr, c_curr)

        # 4. Construct Canvas
        if mode == "single":
            full_frame = rollout_bgr
        else:
            # Main video panels
            if mode == "tri-panel":
                p1 = gt_bgr if gt_bgr is not None else np.zeros((384, 512, 3), dtype=np.uint8)
                p2 = table_bgr if table_bgr is not None else rollout_bgr
                p3 = rollout_bgr
                body = np.hstack([p1, p2, p3])
                # Dividers
                cv2.line(body, (512, 0), (512, 384), (60, 60, 60), 2)
                cv2.line(body, (1024, 0), (1024, 384), (60, 60, 60), 2)
                # Labels
                cv2.putText(body, "1. Ground Truth", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(body, "2. Table Lookup [fd(h_t)]", (512 + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(body, "3. Autonomous Rollout [ctf unroll]", (1024 + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            elif mode == "table":
                p1 = gt_bgr if gt_bgr is not None else np.zeros((384, 512, 3), dtype=np.uint8)
                p2 = table_bgr if table_bgr is not None else rollout_bgr
                body = np.hstack([p1, p2])
                cv2.line(body, (512, 0), (512, 384), (60, 60, 60), 2)
                cv2.putText(body, "Ground Truth", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(body, "Direct Table Reconstruction", (512 + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            else:  # side-by-side
                p1 = gt_bgr if gt_bgr is not None else np.zeros((384, 512, 3), dtype=np.uint8)
                p2 = rollout_bgr
                body = np.hstack([p1, p2])
                cv2.line(body, (512, 0), (512, 384), (60, 60, 60), 2)
                cv2.putText(body, "Ground Truth (Original)", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(body, "Autonomous Rollout (Recursive ctf)", (512 + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # Header Dashboard (60px height)
            header = np.zeros((60, canvas_width, 3), dtype=np.uint8)
            header[:] = (20, 20, 20)  # Dark charcoal

            time_str = f"{current_t // (fps * 60):02d}:{(current_t // fps) % 60:02d}.{(current_t % fps):02d}"
            info_left = f"Frame: {current_t:04d}/{total_frames}  |  Time: {time_str}  |  Resolution: 512x384 @ {fps}fps"
            cv2.putText(header, info_left, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

            if gt_tensor is not None:
                if mode == "tri-panel":
                    info_right = f"Table MSE: {mse_table:.4f}  |  Rollout MSE: {mse_rollout:.4f}"
                    color_right = (0, 255, 255)
                    cv2.putText(header, info_right, (canvas_width - 550, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color_right, 2, cv2.LINE_AA)
                elif mode == "table":
                    cum_mse = np.mean(mse_history)
                    info_right = f"Table MSE: {mse_table:.4f}  (Mean: {cum_mse:.4f})"
                    color_right = (0, 255, 0) if mse_table < 0.03 else (0, 220, 255)
                    cv2.putText(header, info_right, (canvas_width - 420, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color_right, 2, cv2.LINE_AA)
                else:
                    cum_mse = np.mean(mse_history)
                    info_right = f"Frame MSE: {mse_rollout:.4f}  (Mean: {cum_mse:.4f})"
                    color_right = (0, 255, 0) if mse_rollout < 0.03 else ((0, 220, 255) if mse_rollout < 0.08 else (0, 100, 255))
                    cv2.putText(header, info_right, (canvas_width - 420, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color_right, 2, cv2.LINE_AA)

            full_frame = np.vstack([header, body])

        video_writer.write(full_frame)

        if (idx + 1) % 500 == 0 or (idx + 1) == total_frames:
            fps_proc = (idx + 1) / max(1e-3, (time.time() - t_start))
            avg_mse_str = f" | Avg MSE: {np.mean(mse_history):.4f}" if mse_history else ""
            print(f"  Frame {idx + 1:4d}/{total_frames} [{((idx+1)/total_frames)*100.0:.1f}%]  Speed: {fps_proc:.1f} fps{avg_mse_str}")

    video_writer.release()
    total_time = time.time() - t_start
    print(f"\n[OK] Video successfully rendered in {total_time:.2f}s ({total_frames / total_time:.1f} fps)!")
    print(f"     Output: {out_mp4} ({os.path.getsize(out_mp4) / (1024*1024):.2f} MB)")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bad Apple Video Synthesis & Comparison Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to packaged .pt checkpoint or runs/*.pt")
    parser.add_argument("--mode", type=str, choices=["side-by-side", "tri-panel", "single", "table"], default="side-by-side", help="Rendering mode")
    parser.add_argument("--dataset", type=str, default=None, help="Path to ground truth dataset frames folder")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    parser.add_argument("--frames", type=int, default=6573, help="Number of frames to render")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--out", type=str, default=None, help="Output MP4 file path (defaults according to mode)")
    parser.add_argument("--fps", type=int, default=30, help="Output framerate")
    args = parser.parse_args()

    # Automatic checkpoint resolution (works out of the box with zero extra args)
    ckpt_candidates = [
        "exports/models/bad_apple_inference_only.pt",
        "models/bad_apple_inference_only.pt",
        "exports/models/bad_apple_complete_training.pt",
        "models/bad_apple_complete_training.pt",
        "runs/last_model.pt",
        "runs/best_model.pt",
    ]
    ckpt_path = resolve_path(args.checkpoint, ckpt_candidates)
    if ckpt_path is None:
        raise FileNotFoundError("Could not find model checkpoint in exports/models/ or runs/.")

    # Determine default output filename if not specified
    if args.out is None:
        script_dir = Path(__file__).resolve().parent
        exports_root = script_dir if script_dir.name == "exports" else script_dir / "exports"
        if args.mode == "side-by-side":
            args.out = str(exports_root / "rollout_side_by_side.mp4")
        elif args.mode == "tri-panel":
            args.out = str(exports_root / "rollout_tri_panel.mp4")
        elif args.mode == "table":
            args.out = str(exports_root / "table_reconstruction_side_by_side.mp4")
        else:
            args.out = str(exports_root / "rollout_single.mp4")

    # Automatic dataset detection
    dataset_candidates = ["frames", "../frames", "BadApple/frames"]
    dataset_dir = resolve_path(args.dataset, dataset_candidates)
    dataset = None
    if dataset_dir and os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        try:
            dataset = BadAppleDataset(dataset_dir)
            print(f"Loaded ground truth dataset with {len(dataset)} frames from: {dataset_dir}")
        except Exception as e:
            print(f"Warning: Could not load dataset ({e}). Rendering in standalone single mode.")
            dataset = None
    else:
        if args.mode in ["side-by-side", "tri-panel", "table"]:
            print("Note: Ground truth frames directory not found. Rendering in standalone single mode.")
            args.mode = "single"

    # If tri-panel or table is requested but model is inference-only, auto-switch to complete model
    if args.mode in ["tri-panel", "table"] and "inference_only" in ckpt_path:
        complete_candidate = resolve_path(None, ["exports/models/bad_apple_complete_training.pt", "models/bad_apple_complete_training.pt", "runs/last_model.pt", "runs/best_model.pt"])
        if complete_candidate:
            print(f"Note: '{args.mode}' requires embedding table. Switching checkpoint to: {complete_candidate}")
            ckpt_path = complete_candidate
    print(ckpt_path)
    models, cfg, meta = load_packaged_model(ckpt_path, device=args.device)
    render_synthesis_video(
        models=models,
        dataset=dataset,
        mode=args.mode,
        num_frames=args.frames,
        start_frame=args.start,
        out_mp4=args.out,
        fps=args.fps
    )

