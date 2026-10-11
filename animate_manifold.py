"""
animate_manifold.py
-------------------
Publication-grade animated visualizer and interactive 3D point cloud / trajectory 
explorer for the Bad Apple latent training tables (h_table and c_table).

Features:
1. Interactive Real-Time 3D Point Cloud & Trajectory Engine (with 360° mouse orbit, pan, zoom, auto-spin)
2. 2D & 3D Multi-Panel Latent Manifold Animation MP4 Generator
3. Synchronized Ground Truth vs Decoded Frame Video Playback & Velocity Telemetry
4. Standalone HTML5 WebGL / Three.js 3D Viewer Export
"""

import os
import sys
import time
import argparse
import subprocess
import json
import concurrent.futures as cf
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ---------------------------------------------------------------------------
# Architecture Definitions (matching notebook & training checkpoints)
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class FrameDecoder(nn.Module):
    def __init__(self, hidden_size: int = 64, output_height: int = 384, output_width: int = 512):
        super().__init__()
        self.output_height = output_height
        self.output_width = output_width
        self.base_channels = 8
        self.base_height = output_height // 16
        self.base_width = output_width // 16

        self.projection = nn.Linear(
            hidden_size, self.base_channels * self.base_height * self.base_width
        )
        self.block1 = DepthwiseSeparableConv(8, 16)
        self.block2 = DepthwiseSeparableConv(16, 16)
        self.block3 = DepthwiseSeparableConv(16, 8)
        self.output = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        orig_shape = x.shape[:-1]
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
        x = x.reshape(*orig_shape, 1, self.output_height, self.output_width)
        return x


# ---------------------------------------------------------------------------
# Dimensionality Reduction & Analytics
# ---------------------------------------------------------------------------

def compute_pca(data: torch.Tensor, n_components: int = 10):
    """Computes PCA projections and explained variance ratios using low-rank SVD."""
    mean = data.mean(dim=0, keepdim=True)
    centered = data - mean
    U, S, V = torch.pca_lowrank(centered, q=n_components, center=False)
    projected = torch.matmul(centered, V[:, :n_components])
    
    var_explained = (S ** 2) / (data.shape[0] - 1)
    total_var = torch.var(data, dim=0, unbiased=True).sum()
    explained_ratio = (var_explained / total_var).numpy()
    
    return projected.numpy(), V.numpy(), explained_ratio


def compute_manifold_metadata(h_table: np.ndarray, c_table: np.ndarray):
    """Precomputes PCA projections, state velocities, norms, and consecutive similarities."""
    T, D = h_table.shape
    h_torch = torch.from_numpy(h_table).float()
    c_torch = torch.from_numpy(c_table).float()

    pca_h, V_h, var_ratio_h = compute_pca(h_torch, n_components=6)
    pca_c, V_c, var_ratio_c = compute_pca(c_torch, n_components=6)

    # Velocities (Euclidean step deltas)
    vel_h = np.linalg.norm(np.diff(h_table, axis=0), axis=1)
    vel_h = np.pad(vel_h, (0, 1), mode='edge')
    vel_c = np.linalg.norm(np.diff(c_table, axis=0), axis=1)
    vel_c = np.pad(vel_c, (0, 1), mode='edge')

    # Consecutive Cosine Similarities
    norms_h = np.linalg.norm(h_table, axis=1, keepdims=True) + 1e-8
    h_normed = h_table / norms_h
    sim_h = np.sum(h_normed[:-1] * h_normed[1:], axis=1)
    sim_h = np.pad(sim_h, (0, 1), mode='edge')

    # State Norms
    norm_h = np.linalg.norm(h_table, axis=1)
    norm_c = np.linalg.norm(c_table, axis=1)

    # 2D & 3D Bounding limits with 8% padding
    lims_2d_h = (
        (pca_h[:, 0].min() - 0.08 * np.ptp(pca_h[:, 0]), pca_h[:, 0].max() + 0.08 * np.ptp(pca_h[:, 0])),
        (pca_h[:, 1].min() - 0.08 * np.ptp(pca_h[:, 1]), pca_h[:, 1].max() + 0.08 * np.ptp(pca_h[:, 1]))
    )
    lims_2d_c = (
        (pca_c[:, 0].min() - 0.08 * np.ptp(pca_c[:, 0]), pca_c[:, 0].max() + 0.08 * np.ptp(pca_c[:, 0])),
        (pca_c[:, 1].min() - 0.08 * np.ptp(pca_c[:, 1]), pca_c[:, 1].max() + 0.08 * np.ptp(pca_c[:, 1]))
    )
    lims_3d_h = (
        (pca_h[:, 0].min() - 0.05 * np.ptp(pca_h[:, 0]), pca_h[:, 0].max() + 0.05 * np.ptp(pca_h[:, 0])),
        (pca_h[:, 1].min() - 0.05 * np.ptp(pca_h[:, 1]), pca_h[:, 1].max() + 0.05 * np.ptp(pca_h[:, 1])),
        (pca_h[:, 2].min() - 0.05 * np.ptp(pca_h[:, 2]), pca_h[:, 2].max() + 0.05 * np.ptp(pca_h[:, 2]))
    )
    lims_3d_c = (
        (pca_c[:, 0].min() - 0.05 * np.ptp(pca_c[:, 0]), pca_c[:, 0].max() + 0.05 * np.ptp(pca_c[:, 0])),
        (pca_c[:, 1].min() - 0.05 * np.ptp(pca_c[:, 1]), pca_c[:, 1].max() + 0.05 * np.ptp(pca_c[:, 1])),
        (pca_c[:, 2].min() - 0.05 * np.ptp(pca_c[:, 2]), pca_c[:, 2].max() + 0.05 * np.ptp(pca_c[:, 2]))
    )

    return {
        "T": T,
        "D": D,
        "pca_h": pca_h,
        "pca_c": pca_c,
        "var_ratio_h": var_ratio_h,
        "var_ratio_c": var_ratio_c,
        "vel_h": vel_h,
        "vel_c": vel_c,
        "sim_h": sim_h,
        "norm_h": norm_h,
        "norm_c": norm_c,
        "lims_2d_h": lims_2d_h,
        "lims_2d_c": lims_2d_c,
        "lims_3d_h": lims_3d_h,
        "lims_3d_c": lims_3d_c
    }


# ---------------------------------------------------------------------------
# Real-Time 3D Point Cloud & Trajectory Interactive Renderer
# ---------------------------------------------------------------------------

class Interactive3DPointcloudVisualizer:
    """
    High-performance real-time 3D camera projection engine rendering the latent manifold
    as an interactive point cloud and trajectory orbit with full mouse navigation.
    """
    def __init__(
        self,
        meta: dict,
        epoch: int = 0,
        gt_frames_dir: str = "frames",
        fd_module: Optional[FrameDecoder] = None,
        h_table_tensor: Optional[torch.Tensor] = None,
        width: int = 1440,
        height: int = 900
    ):
        self.meta = meta
        self.epoch = epoch
        self.gt_frames_dir = gt_frames_dir if (gt_frames_dir and os.path.exists(gt_frames_dir)) else None
        self.fd = fd_module
        self.h_table_tensor = h_table_tensor

        self.W = width
        self.H = height
        self.T = meta["T"]

        # Camera state
        self.yaw = float(np.radians(45))
        self.pitch = float(np.radians(25))
        self.distance = 3.6
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.auto_spin = False
        self.spin_speed = 0.008

        # Interaction flags
        self.active_table = "h"  # 'h' or 'c'
        self.show_point_cloud = True
        self.show_trajectory = True
        self.show_comet_tail = True
        self.show_pip_video = True
        self.tail_length = 60
        self.point_size = 3

        # Mouse tracking
        self.mouse_dragging = False
        self.mouse_button = 0
        self.last_mx = 0
        self.last_my = 0

        # Colors precomputed
        self._precompute_colors()

    def _precompute_colors(self):
        """Precomputes timeline BGR colors for both h and c manifolds."""
        t_norm = np.linspace(0.0, 1.0, self.T)
        
        # Plasma colormap for H
        cmap_plasma = matplotlib.colormaps['plasma']
        rgba_h = cmap_plasma(t_norm)
        self.colors_h_bgr = (rgba_h[:, :3][:, ::-1] * 255).astype(np.uint8)

        # Viridis colormap for C
        cmap_vir = matplotlib.colormaps['viridis']
        rgba_c = cmap_vir(t_norm)
        self.colors_c_bgr = (rgba_c[:, :3][:, ::-1] * 255).astype(np.uint8)

        # Normalize 3D coordinates to zero-mean unit scale for uniform camera framing
        h3 = self.meta["pca_h"][:, :3].copy()
        c3 = self.meta["pca_c"][:, :3].copy()

        self.center_h = h3.mean(axis=0)
        self.scale_h = np.max(np.linalg.norm(h3 - self.center_h, axis=1)) + 1e-6
        self.pts_h_norm = (h3 - self.center_h) / self.scale_h

        self.center_c = c3.mean(axis=0)
        self.scale_c = np.max(np.linalg.norm(c3 - self.center_c, axis=1)) + 1e-6
        self.pts_c_norm = (c3 - self.center_c) / self.scale_c

    def project_3d(self, pts_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Projects 3D normalized coordinates onto 2D viewport with perspective division."""
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)

        # R = R_x(pitch) @ R_y(yaw)
        R_yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        R_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
        R = (R_pitch @ R_yaw).T

        rot = pts_norm @ R
        z_cam = rot[:, 2] + self.distance
        valid = z_cam > 0.1

        fov = self.H * 0.95
        x_2d = ((rot[:, 0] + self.pan_x) / z_cam) * fov + self.W / 2.0
        y_2d = ((-rot[:, 1] + self.pan_y) / z_cam) * fov + self.H / 2.0

        return x_2d, y_2d, z_cam, valid

    def render(self, t: int) -> np.ndarray:
        """Renders one interactive 3D frame at timeline step t."""
        t = int(np.clip(t, 0, self.T - 1))

        if self.auto_spin:
            self.yaw = (self.yaw + self.spin_speed) % (2.0 * np.pi)

        canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        canvas[:] = (12, 16, 24)  # Deep dark space background

        pts = self.pts_h_norm if self.active_table == "h" else self.pts_c_norm
        colors = self.colors_h_bgr if self.active_table == "h" else self.colors_c_bgr
        table_name = "Hidden State (h-Table)" if self.active_table == "h" else "Cell State (c-Table)"

        x2d, y2d, z_cam, valid = self.project_3d(pts)

        # -------------------------------------------------------------
        # 1. 3D Coordinate Grid Floor & Axes
        # -------------------------------------------------------------
        self._draw_grid_floor(canvas)

        # -------------------------------------------------------------
        # 2. 3D Full Point Cloud (Depth-Sorted)
        # -------------------------------------------------------------
        if self.show_point_cloud:
            order = np.argsort(z_cam)[::-1]
            for idx in order:
                if not valid[idx]:
                    continue
                xi, yi = int(x2d[idx]), int(y2d[idx])
                if 0 <= xi < self.W and 0 <= yi < self.H:
                    # Depth attenuated particle size
                    pt_r = max(1, int(self.point_size * (3.0 / z_cam[idx])))
                    col = tuple(int(c) for c in colors[idx])
                    cv2.circle(canvas, (xi, yi), pt_r, col, -1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 3. 3D Trajectory Ribbon / Path
        # -------------------------------------------------------------
        if self.show_trajectory and t > 0:
            step = 1 if t < 2000 else 2
            indices = np.arange(0, t + 1, step)
            for i in range(len(indices) - 1):
                i1, i2 = indices[i], indices[i + 1]
                if valid[i1] and valid[i2]:
                    pt1 = (int(x2d[i1]), int(y2d[i1]))
                    pt2 = (int(x2d[i2]), int(y2d[i2]))
                    col = tuple(int(c) for c in colors[i2])
                    cv2.line(canvas, pt1, pt2, col, 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 4. 3D Glowing Comet Tail (Recent N frames)
        # -------------------------------------------------------------
        tail_start = max(0, t - self.tail_length)
        if self.show_comet_tail and t > tail_start:
            tail_idx = np.arange(tail_start, t + 1)
            for i in range(len(tail_idx) - 1):
                i1, i2 = tail_idx[i], tail_idx[i + 1]
                if valid[i1] and valid[i2]:
                    pt1 = (int(x2d[i1]), int(y2d[i1]))
                    pt2 = (int(x2d[i2]), int(y2d[i2]))
                    frac = (i + 1) / len(tail_idx)
                    thick = max(1, int(1 + 4.5 * frac))
                    # Brighten color towards white/yellow
                    base_col = colors[i2].astype(np.float32)
                    glow_col = tuple(int(min(255, base_col[c] + 120 * frac)) for c in range(3))
                    cv2.line(canvas, pt1, pt2, glow_col, thick, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 5. 3D Current Head Beacon & Velocity Tangent
        # -------------------------------------------------------------
        if valid[t]:
            hx, hy = int(x2d[t]), int(y2d[t])
            if 0 <= hx < self.W and 0 <= hy < self.H:
                # Pulsing halo
                cv2.circle(canvas, (hx, hy), 14, (255, 200, 50), 2, cv2.LINE_AA)
                cv2.circle(canvas, (hx, hy), 6, (255, 255, 255), -1, cv2.LINE_AA)
                # Coordinate Crosshairs
                cv2.line(canvas, (hx - 20, hy), (hx + 20, hy), (255, 200, 50), 1, cv2.LINE_AA)
                cv2.line(canvas, (hx, hy - 20), (hx, hy + 20), (255, 200, 50), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 6. UI HUD & Telemetry Overlays
        # -------------------------------------------------------------
        self._draw_hud(canvas, t, table_name)

        # -------------------------------------------------------------
        # 7. Synchronized Video PiP Thumbnail (Top-Right)
        # -------------------------------------------------------------
        if self.show_pip_video:
            self._draw_pip_video(canvas, t)

        return canvas

    def _draw_grid_floor(self, canvas: np.ndarray):
        """Draws an orientation reference floor grid."""
        grid_pts = []
        lines = []
        grid_range = np.linspace(-1.2, 1.2, 7)
        floor_z = -1.1

        # Grid lines along X and Y
        for x in grid_range:
            p1 = np.array([x, floor_z, -1.2], dtype=np.float32)
            p2 = np.array([x, floor_z, 1.2], dtype=np.float32)
            x1, y1, _, v1 = self.project_3d(p1[None, :])
            x2, y2, _, v2 = self.project_3d(p2[None, :])
            if v1[0] and v2[0]:
                cv2.line(canvas, (int(x1[0]), int(y1[0])), (int(x2[0]), int(y2[0])), (28, 38, 55), 1, cv2.LINE_AA)

        for y in grid_range:
            p1 = np.array([-1.2, floor_z, y], dtype=np.float32)
            p2 = np.array([1.2, floor_z, y], dtype=np.float32)
            x1, y1, _, v1 = self.project_3d(p1[None, :])
            x2, y2, _, v2 = self.project_3d(p2[None, :])
            if v1[0] and v2[0]:
                cv2.line(canvas, (int(x1[0]), int(y1[0])), (int(x2[0]), int(y2[0])), (28, 38, 55), 1, cv2.LINE_AA)

        # 3D Axes at origin (RGB = PC1, PC2, PC3)
        origin = np.array([[0, 0, 0], [0.6, 0, 0], [0, 0.6, 0], [0, 0, 0.6]], dtype=np.float32)
        ox, oy, _, ov = self.project_3d(origin)
        if all(ov):
            o = (int(ox[0]), int(oy[0]))
            cv2.line(canvas, o, (int(ox[1]), int(oy[1])), (0, 0, 255), 2, cv2.LINE_AA)  # PC1 Red
            cv2.line(canvas, o, (int(ox[2]), int(oy[2])), (0, 255, 0), 2, cv2.LINE_AA)  # PC2 Green
            cv2.line(canvas, o, (int(ox[3]), int(oy[3])), (255, 100, 0), 2, cv2.LINE_AA) # PC3 Blue
            cv2.putText(canvas, "PC1", (int(ox[1]) + 5, int(oy[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            cv2.putText(canvas, "PC2", (int(ox[2]) + 5, int(oy[2])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(canvas, "PC3", (int(ox[3]) + 5, int(oy[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1)

    def _draw_hud(self, canvas: np.ndarray, t: int, table_name: str):
        """Draws high-tech HUD cards and hotkey legend."""
        fps = 30
        mins = t // (fps * 60)
        secs = (t // fps) % 60
        frames = t % fps
        timecode = f"{mins:02d}:{secs:02d}.{frames:02d}"
        pct = (t / max(1, self.T - 1)) * 100.0

        v_now = self.meta["vel_h"][t] if self.active_table == "h" else self.meta["vel_c"][t]
        sim_now = self.meta["sim_h"][t]

        # Top Header Banner
        header_text = f"Bad Apple!! 3D Latent Manifold Explorer  •  {table_name}  •  Epoch {self.epoch}"
        cv2.putText(canvas, header_text, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 255), 2, cv2.LINE_AA)

        info_sub = f"Frame: {t:04d} / {self.T-1}  [{timecode}]  ({pct:.1f}%)  |  Velocity: {v_now:.4f}  |  Cosine Sim: {sim_now:.4f}"
        cv2.putText(canvas, info_sub, (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 1, cv2.LINE_AA)

        # Camera Status Box (Top Right)
        cam_info = f"Yaw: {np.degrees(self.yaw):.0f}°  Pitch: {np.degrees(self.pitch):.0f}°  Dist: {self.distance:.1f}x"
        cv2.putText(canvas, cam_info, (self.W - 350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1, cv2.LINE_AA)

        # Controls Cheat-Sheet Overlay (Bottom-Left)
        controls = [
            "MOUSE CONTROLS:",
            " • Left Click + Drag : 360° Orbit Rotation",
            " • Right / Mid Drag   : Camera Pan (dx, dy)",
            " • Scroll Wheel       : Zoom In / Out",
            "",
            "KEYBOARD CONTROLS:",
            " • [SPACE] : Play / Pause",
            " • [A]     : Toggle Auto-Spin Orbit",
            " • [TAB]   : Switch h-Table <-> c-Table",
            " • [1] / [2] / [3] : Cloud / Trajectory / Both",
            " • [V]     : Toggle Video PiP Overlay",
            " • [C]     : Reset Camera View",
            " • [P]     : Save High-Res Screenshot",
            " • [Q/ESC] : Exit Visualizer"
        ]

        # Draw semi-transparent card at bottom left
        card_h = 240
        card_w = 340
        card_y = self.H - card_h - 20
        card_x = 20

        overlay = canvas.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (18, 24, 38), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (51, 65, 85), 1)

        y_txt = card_y + 22
        for line in controls:
            color = (56, 189, 248) if "CONTROLS" in line else (203, 213, 225)
            font_scale = 0.45 if "CONTROLS" in line else 0.40
            thick = 2 if "CONTROLS" in line else 1
            cv2.putText(canvas, line, (card_x + 12, y_txt), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick, cv2.LINE_AA)
            y_txt += 16

    def _draw_pip_video(self, canvas: np.ndarray, t: int):
        """Draws live Ground Truth vs Reconstructed video thumbnail in bottom right."""
        gt_gray = None
        if self.gt_frames_dir:
            frame_path = os.path.join(self.gt_frames_dir, f"frame_{t:06d}.png")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(self.gt_frames_dir, f"frame_{t+1:06d}.png")
            if os.path.exists(frame_path):
                img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    gt_gray = img

        rec_gray = None
        if self.fd is not None and self.h_table_tensor is not None:
            with torch.no_grad():
                h_val = self.h_table_tensor[t:t+1]
                pred_logit = self.fd(h_val)
                pred_tensor = torch.sigmoid(pred_logit).squeeze().cpu().numpy()
                rec_gray = (np.clip(pred_tensor, 0.0, 1.0) * 255).astype(np.uint8)

        if gt_gray is not None and rec_gray is not None:
            comp = np.hstack([gt_gray, rec_gray])
            # Resize thumbnail to 360x135
            pip_w = 360
            pip_h = 135
            comp_resized = cv2.resize(comp, (pip_w, pip_h), interpolation=cv2.INTER_AREA)
            comp_bgr = cv2.cvtColor(comp_resized, cv2.COLOR_GRAY2BGR)

            cv2.putText(comp_bgr, "GT", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(comp_bgr, "Decoded", (pip_w // 2 + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(comp_bgr, (pip_w // 2, 0), (pip_w // 2, pip_h), (80, 80, 80), 1)

            x_offset = self.W - pip_w - 25
            y_offset = self.H - pip_h - 25

            cv2.rectangle(canvas, (x_offset - 2, y_offset - 2), (x_offset + pip_w + 2, y_offset + pip_h + 2), (56, 189, 248), 1)
            canvas[y_offset:y_offset + pip_h, x_offset:x_offset + pip_w] = comp_bgr


def run_interactive_3d_visualizer(
    checkpoint_path: str,
    fps: int = 30,
    gt_frames_dir: str = "frames"
):
    """
    Main interactive loop for the 3D Point Cloud & Trajectory visualizer with full mouse navigation.
    """
    print(f"\n============================================================")
    print(f" Launching Interactive 3D Latent Point Cloud Visualizer")
    print(f" Checkpoint: {checkpoint_path}")
    print(f"============================================================")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    epoch = ckpt.get("epoch", 0)
    h_table_tensor = ckpt["tte"]["h_table.weight"]
    c_table_tensor = ckpt["tte"]["c_table.weight"]
    fd_state_dict = ckpt.get("fd", None)

    h_table = h_table_tensor.numpy()
    c_table = c_table_tensor.numpy()

    meta = compute_manifold_metadata(h_table, c_table)

    fd_module = None
    if fd_state_dict is not None:
        fd_module = FrameDecoder(hidden_size=meta["D"])
        fd_module.load_state_dict(fd_state_dict)
        fd_module.eval()

    viz = Interactive3DPointcloudVisualizer(
        meta=meta,
        epoch=epoch,
        gt_frames_dir=gt_frames_dir,
        fd_module=fd_module,
        h_table_tensor=h_table_tensor,
        width=1440,
        height=900
    )

    window_name = "Bad Apple!! 3D Latent Point Cloud & Manifold Visualizer [Interactive]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1440, 900)

    trackbar_state = {"frame_idx": 0, "scrubbed": False}

    def on_trackbar(val):
        trackbar_state["frame_idx"] = val
        trackbar_state["scrubbed"] = True

    cv2.createTrackbar("Timeline (0 to 6572)", window_name, 0, meta["T"] - 1, on_trackbar)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            viz.mouse_dragging = True
            viz.mouse_button = 1
            viz.last_mx = x
            viz.last_my = y
        elif event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_MBUTTONDOWN:
            viz.mouse_dragging = True
            viz.mouse_button = 2
            viz.last_mx = x
            viz.last_my = y
        elif event == cv2.EVENT_MOUSEMOVE and viz.mouse_dragging:
            dx = x - viz.last_mx
            dy = y - viz.last_my
            viz.last_mx = x
            viz.last_my = y

            if viz.mouse_button == 1:
                # Orbit rotation
                viz.yaw += dx * 0.007
                viz.pitch = float(np.clip(viz.pitch - dy * 0.007, -np.radians(88), np.radians(88)))
            elif viz.mouse_button == 2:
                # Camera Pan
                viz.pan_x += dx * (viz.distance / 900.0)
                viz.pan_y += dy * (viz.distance / 900.0)

        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP or event == cv2.EVENT_MBUTTONUP:
            viz.mouse_dragging = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            # Zoom
            if flags > 0:
                viz.distance = max(1.2, viz.distance * 0.92)
            else:
                viz.distance = min(20.0, viz.distance * 1.08)

    cv2.setMouseCallback(window_name, on_mouse)

    paused = False
    current_t = 0
    speed_mult = 1.0

    print("\nVisualizer Running! Drag with mouse to orbit around the 3D manifold.\n")

    try:
        while True:
            if trackbar_state["scrubbed"]:
                current_t = trackbar_state["frame_idx"]
                trackbar_state["scrubbed"] = False

            frame_bgr = viz.render(current_t)
            cv2.imshow(window_name, frame_bgr)
            cv2.setTrackbarPos("Timeline (0 to 6572)", window_name, current_t)

            delay = 1 if paused else max(1, int(1000 / (fps * speed_mult)))
            key = cv2.waitKey(delay) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('a') or key == ord('A'):
                viz.auto_spin = not viz.auto_spin
                print(f"Auto-Spin: {'ON' if viz.auto_spin else 'OFF'}")
            elif key == ord('4'):  # TAB key
                viz.active_table = "c" if viz.active_table == "h" else "h"
                print(f"Switched active table to: {viz.active_table.upper()}-Table")
            elif key == ord('1'):
                viz.show_point_cloud = not viz.show_point_cloud
            elif key == ord('2'):
                viz.show_trajectory = not viz.show_trajectory
            elif key == ord('3'):
                viz.show_point_cloud = True
                viz.show_trajectory = True
            elif key == ord('v') or key == ord('V'):
                viz.show_pip_video = not viz.show_pip_video
            elif key == ord('c') or key == ord('C'):
                # Reset camera
                viz.yaw = float(np.radians(45))
                viz.pitch = float(np.radians(25))
                viz.distance = 3.6
                viz.pan_x = 0.0
                viz.pan_y = 0.0
                print("Camera view reset to default.")
            elif key == ord('+') or key == ord('='):
                viz.distance = max(1.2, viz.distance * 0.9)
            elif key == ord('-') or key == ord('_'):
                viz.distance = min(20.0, viz.distance * 1.1)
            elif key == ord('d') or key == 83:  # D or Right
                if current_t < meta["T"] - 1:
                    current_t += 1
            elif key == ord('a') or key == 81:  # Left arrow
                if current_t > 0:
                    current_t -= 1
            elif key == ord('w'):
                current_t = min(meta["T"] - 1, current_t + 30)
            elif key == ord('s'):
                current_t = max(0, current_t - 30)
            elif key == ord('r') or key == ord('R'):
                current_t = 0
            elif key == ord('p') or key == ord('P'):
                snap_path = f"manifold_3d_snapshot_frame_{current_t:04d}.png"
                cv2.imwrite(snap_path, frame_bgr)
                print(f"Saved snapshot: {snap_path}")

            if not paused:
                current_t = (current_t + 1) % meta["T"]

    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)


# ---------------------------------------------------------------------------
# Standalone Interactive HTML5 / WebGL / Three.js 3D Viewer Export
# ---------------------------------------------------------------------------

def export_html_3d_viewer(
    checkpoint_path: str,
    output_html_path: str = "tables_latent_manifold_3d_interactive.html"
):
    """
    Exports a self-contained, high-performance HTML5/WebGL 3D interactive viewer with Three.js.
    Can be opened directly in any browser or notebook iframe.
    """
    print(f"\nExporting standalone 3D WebGL viewer to: {output_html_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    h_table = ckpt["tte"]["h_table.weight"].numpy()
    c_table = ckpt["tte"]["c_table.weight"].numpy()
    meta = compute_manifold_metadata(h_table, c_table)

    pts_h = meta["pca_h"][:, :3].tolist()
    pts_c = meta["pca_c"][:, :3].tolist()
    vel_h = meta["vel_h"].tolist()

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Bad Apple!! 3D Latent Manifold Visualizer</title>
    <style>
        body {{ margin: 0; overflow: hidden; background: #0a0e17; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #f8fafc; }}
        #hud {{ position: absolute; top: 15px; left: 20px; z-index: 100; pointer-events: none; }}
        #hud h1 {{ margin: 0; font-size: 20px; color: #38bdf8; text-shadow: 0 0 10px rgba(56,189,248,0.5); }}
        #hud p {{ margin: 4px 0; font-size: 13px; color: #94a3b8; font-family: monospace; }}
        #controls {{ position: absolute; bottom: 20px; left: 20px; right: 20px; z-index: 100; display: flex; align-items: center; gap: 15px; background: rgba(19,27,46,0.85); padding: 12px 20px; border-radius: 10px; backdrop-filter: blur(8px); border: 1px solid #334155; }}
        #slider {{ flex-grow: 1; accent-color: #38bdf8; cursor: pointer; }}
        button {{ background: #1e293b; border: 1px solid #475569; color: #f8fafc; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-weight: bold; transition: all 0.2s; }}
        button:hover {{ background: #38bdf8; color: #0f172a; }}
        #legend {{ position: absolute; top: 15px; right: 20px; background: rgba(19,27,46,0.85); padding: 12px; border-radius: 8px; border: 1px solid #334155; font-size: 12px; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="hud">
        <h1>Bad Apple!! 3D Latent Manifold Explorer</h1>
        <p id="time-info">Frame: 0000 / {meta['T']-1} | Velocity: 0.0000</p>
    </div>
    <div id="legend">
        <b>Navigation:</b><br>
        • Left-Drag: 360° Orbit<br>
        • Right-Drag: Pan<br>
        • Scroll: Zoom In/Out
    </div>
    <div id="controls">
        <button id="btn-play">Pause</button>
        <button id="btn-table">Switch to C-Table</button>
        <button id="btn-spin">Auto-Spin: ON</button>
        <input type="range" id="slider" min="0" max="{meta['T']-1}" value="0">
    </div>

    <script>
        const ptsH = {json.dumps(pts_h)};
        const ptsC = {json.dumps(pts_c)};
        const velH = {json.dumps(vel_h)};
        const T = ptsH.length;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0e17);
        scene.fog = new THREE.FogExp2(0x0a0e17, 0.03);

        const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);
        camera.position.set(4, 3, 5);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 1.0;

        // Grid Floor
        const grid = new THREE.GridHelper(10, 20, 0x38bdf8, 0x1e293b);
        grid.position.y = -2;
        scene.add(grid);

        // Build Particle Cloud
        let currentTable = 'H';
        let currentPts = ptsH;

        function createPointCloud(pts) {{
            const geo = new THREE.BufferGeometry();
            const positions = [];
            const colors = [];
            const color = new THREE.Color();

            for (let i = 0; i < pts.length; i++) {{
                positions.push(pts[i][0], pts[i][1], pts[i][2]);
                const frac = i / pts.length;
                color.setHSL(0.7 - frac * 0.7, 0.9, 0.55);
                colors.push(color.r, color.g, color.b);
            }}

            geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

            const mat = new THREE.PointsMaterial({{
                size: 0.08,
                vertexColors: true,
                transparent: true,
                opacity: 0.85
            }});

            return new THREE.Points(geo, mat);
        }}

        let pointCloud = createPointCloud(ptsH);
        scene.add(pointCloud);

        // Head Marker
        const beaconGeo = new THREE.SphereGeometry(0.12, 16, 16);
        const beaconMat = new THREE.MeshBasicMaterial({{ color: 0x38bdf8 }});
        const beacon = new THREE.Mesh(beaconGeo, beaconMat);
        scene.add(beacon);

        let currentFrame = 0;
        let isPlaying = true;
        const slider = document.getElementById('slider');
        const timeInfo = document.getElementById('time-info');

        document.getElementById('btn-play').onclick = function() {{
            isPlaying = !isPlaying;
            this.textContent = isPlaying ? 'Pause' : 'Play';
        }};

        document.getElementById('btn-spin').onclick = function() {{
            controls.autoRotate = !controls.autoRotate;
            this.textContent = 'Auto-Spin: ' + (controls.autoRotate ? 'ON' : 'OFF');
        }};

        document.getElementById('btn-table').onclick = function() {{
            scene.remove(pointCloud);
            if (currentTable === 'H') {{
                currentTable = 'C';
                currentPts = ptsC;
                this.textContent = 'Switch to H-Table';
            }} else {{
                currentTable = 'H';
                currentPts = ptsH;
                this.textContent = 'Switch to C-Table';
            }}
            pointCloud = createPointCloud(currentPts);
            scene.add(pointCloud);
        }};

        slider.oninput = function() {{
            currentFrame = parseInt(this.value);
            updateBeacon();
        }};

        function updateBeacon() {{
            const p = currentPts[currentFrame];
            beacon.position.set(p[0], p[1], p[2]);
            timeInfo.textContent = `Frame: ${{String(currentFrame).padStart(4, '0')}} / ${{T-1}} | Table: ${{currentTable}} | Velocity: ${{velH[currentFrame].toFixed(4)}}`;
        }}

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        function animate() {{
            requestAnimationFrame(animate);
            if (isPlaying) {{
                currentFrame = (currentFrame + 1) % T;
                slider.value = currentFrame;
                updateBeacon();
            }}
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
</body>
</html>"""

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved interactive HTML5 3D viewer: {os.path.abspath(output_html_path)}")


# ---------------------------------------------------------------------------
# High-Definition 2D/3D Multi-Panel Video Generator
# ---------------------------------------------------------------------------

class ManifoldDashboardRenderer:
    """
    Multi-panel dashboard renderer for latent trajectories.
    Uses pure Object-Oriented Matplotlib API for 100% thread safety and zero deadlocks.
    """
    def __init__(
        self,
        meta: dict,
        epoch: int = 0,
        loss: float = 0.0,
        tail_length: int = 60,
        mode: str = "combined",
        gt_frames_dir: Optional[str] = "frames",
        fd_module: Optional[FrameDecoder] = None,
        h_table_tensor: Optional[torch.Tensor] = None,
        device: str = "cpu"
    ):
        self.meta = meta
        self.epoch = epoch
        self.loss = loss
        self.tail_len = tail_length
        self.mode = mode
        self.gt_frames_dir = gt_frames_dir if (gt_frames_dir and os.path.exists(gt_frames_dir)) else None
        self.fd = fd_module
        self.h_table_tensor = h_table_tensor
        self.device = device

        self.T = meta["T"]
        self.D = meta["D"]
        self.pca_h = meta["pca_h"]
        self.pca_c = meta["pca_c"]
        self.var_h = meta["var_ratio_h"]
        self.var_c = meta["var_ratio_c"]
        self.vel_h = meta["vel_h"]
        self.sim_h = meta["sim_h"]
        self.norm_h = meta["norm_h"]

    def render_frame(self, t: int, figsize=(19.2, 10.8), dpi=100) -> np.ndarray:
        t = int(np.clip(t, 0, self.T - 1))
        tail_start = max(0, t - self.tail_len)
        tail_indices = np.arange(tail_start, t + 1)
        k_tail = len(tail_indices)

        fig = Figure(figsize=figsize, dpi=dpi, facecolor='#0A0E17')
        canvas = FigureCanvasAgg(fig)

        if self.mode == "combined":
            gs = gridspec.GridSpec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0],
                                   left=0.04, right=0.98, top=0.92, bottom=0.06, hspace=0.22, wspace=0.16)
            ax1 = fig.add_subplot(gs[0, 0], facecolor='#111827')
            ax2 = fig.add_subplot(gs[0, 1], facecolor='#111827')
            ax3 = fig.add_subplot(gs[1, 0], projection='3d', facecolor='#111827')
            
            gs_sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], height_ratios=[1.0, 0.85], hspace=0.28)
            ax4_vid = fig.add_subplot(gs_sub[0, 0], facecolor='#0A0E17')
            ax4_tel = fig.add_subplot(gs_sub[1, 0], facecolor='#111827')
        elif self.mode == "quad_manifolds":
            gs = gridspec.GridSpec(2, 2, left=0.04, right=0.98, top=0.92, bottom=0.06, hspace=0.22, wspace=0.16)
            ax1 = fig.add_subplot(gs[0, 0], facecolor='#111827')
            ax2 = fig.add_subplot(gs[0, 1], facecolor='#111827')
            ax3 = fig.add_subplot(gs[1, 0], projection='3d', facecolor='#111827')
            ax4 = fig.add_subplot(gs[1, 1], projection='3d', facecolor='#111827')

        for ax in [ax1, ax2]:
            ax.tick_params(colors='#64748B', labelsize=8.5)
            for spine in ax.spines.values():
                spine.set_color('#1E293B')

        # 1. 2D H
        ax1.set_xlim(self.meta["lims_2d_h"][0])
        ax1.set_ylim(self.meta["lims_2d_h"][1])
        ax1.plot(self.pca_h[:, 0], self.pca_h[:, 1], color='#3B285E', alpha=0.20, linewidth=0.7, zorder=1)
        ax1.scatter(self.pca_h[0, 0], self.pca_h[0, 1], color='#10B981', s=35, marker='o', label='Start ($t=0$)', zorder=2)
        ax1.scatter(self.pca_h[-1, 0], self.pca_h[-1, 1], color='#EF4444', s=35, marker='X', label=f'End ($t={self.T-1}$)', zorder=2)

        if t > 0:
            step = 1 if t < 1000 else 2
            ax1.plot(self.pca_h[:t+1:step, 0], self.pca_h[:t+1:step, 1], color='#818CF8', alpha=0.35, linewidth=1.1, zorder=3)

        if k_tail > 1:
            pts = self.pca_h[tail_indices, :2]
            segments = np.concatenate([pts[:-1, np.newaxis, :], pts[1:, np.newaxis, :]], axis=1)
            alphas = np.linspace(0.15, 0.95, len(segments))
            widths = np.linspace(1.2, 3.8, len(segments))
            cmap_plasma = matplotlib.colormaps['plasma']
            colors = cmap_plasma(np.linspace(0.2, 0.95, len(segments)))
            colors[:, 3] = alphas
            lc = LineCollection(segments, colors=colors, linewidths=widths, zorder=4)
            ax1.add_collection(lc)

        if t > 0:
            dx = (self.pca_h[t, 0] - self.pca_h[t-1, 0]) * 3.0
            dy = (self.pca_h[t, 1] - self.pca_h[t-1, 1]) * 3.0
            if np.hypot(dx, dy) > 1e-4:
                ax1.annotate('', xy=(self.pca_h[t, 0] + dx, self.pca_h[t, 1] + dy),
                             xytext=(self.pca_h[t, 0], self.pca_h[t, 1]),
                             arrowprops=dict(arrowstyle='->', color='#38BDF8', lw=1.6, mutation_scale=12),
                             zorder=6)

        ax1.scatter(self.pca_h[t, 0], self.pca_h[t, 1], color='#38BDF8', s=160, alpha=0.35, zorder=5)
        ax1.scatter(self.pca_h[t, 0], self.pca_h[t, 1], color='#FFFFFF', s=45, edgecolors='#0284C7', linewidth=1.5, zorder=7)

        ax1.set_title(f"2D Hidden State ($h$) Phase Space  |  $PC_1$ ({self.var_h[0]*100:.1f}%) vs $PC_2$ ({self.var_h[1]*100:.1f}%)",
                      fontsize=11, fontweight='bold', pad=8, color='#E2E8F0')
        ax1.set_xlabel("$PC_1$", fontsize=10, color='#94A3B8')
        ax1.set_ylabel("$PC_2$", fontsize=10, color='#94A3B8')
        ax1.grid(True, linestyle=':', alpha=0.25, color='#334155')
        ax1.legend(loc='upper right', framealpha=0.7, facecolor='#1F2937', edgecolor='#374151', fontsize=8, labelcolor='#E2E8F0')

        # 2. 2D C
        ax2.set_xlim(self.meta["lims_2d_c"][0])
        ax2.set_ylim(self.meta["lims_2d_c"][1])
        ax2.plot(self.pca_c[:, 0], self.pca_c[:, 1], color='#134E4A', alpha=0.20, linewidth=0.7, zorder=1)
        ax2.scatter(self.pca_c[0, 0], self.pca_c[0, 1], color='#10B981', s=35, marker='o', label='Start ($t=0$)', zorder=2)
        ax2.scatter(self.pca_c[-1, 0], self.pca_c[-1, 1], color='#EF4444', s=35, marker='X', label=f'End ($t={self.T-1}$)', zorder=2)

        if t > 0:
            step = 1 if t < 1000 else 2
            ax2.plot(self.pca_c[:t+1:step, 0], self.pca_c[:t+1:step, 1], color='#34D399', alpha=0.35, linewidth=1.1, zorder=3)

        if k_tail > 1:
            pts_c = self.pca_c[tail_indices, :2]
            segments_c = np.concatenate([pts_c[:-1, np.newaxis, :], pts_c[1:, np.newaxis, :]], axis=1)
            alphas_c = np.linspace(0.15, 0.95, len(segments_c))
            widths_c = np.linspace(1.2, 3.8, len(segments_c))
            cmap_vir = matplotlib.colormaps['viridis']
            colors_c = cmap_vir(np.linspace(0.25, 0.98, len(segments_c)))
            colors_c[:, 3] = alphas_c
            lc_c = LineCollection(segments_c, colors=colors_c, linewidths=widths_c, zorder=4)
            ax2.add_collection(lc_c)

        if t > 0:
            dxc = (self.pca_c[t, 0] - self.pca_c[t-1, 0]) * 3.0
            dyc = (self.pca_c[t, 1] - self.pca_c[t-1, 1]) * 3.0
            if np.hypot(dxc, dyc) > 1e-4:
                ax2.annotate('', xy=(self.pca_c[t, 0] + dxc, self.pca_c[t, 1] + dyc),
                             xytext=(self.pca_c[t, 0], self.pca_c[t, 1]),
                             arrowprops=dict(arrowstyle='->', color='#34D399', lw=1.6, mutation_scale=12),
                             zorder=6)

        ax2.scatter(self.pca_c[t, 0], self.pca_c[t, 1], color='#10B981', s=160, alpha=0.35, zorder=5)
        ax2.scatter(self.pca_c[t, 0], self.pca_c[t, 1], color='#FFFFFF', s=45, edgecolors='#059669', linewidth=1.5, zorder=7)

        ax2.set_title(f"2D Cell State ($c$) Phase Space  |  $PC_1$ ({self.var_c[0]*100:.1f}%) vs $PC_2$ ({self.var_c[1]*100:.1f}%)",
                      fontsize=11, fontweight='bold', pad=8, color='#E2E8F0')
        ax2.set_xlabel("$PC_1$", fontsize=10, color='#94A3B8')
        ax2.set_ylabel("$PC_2$", fontsize=10, color='#94A3B8')
        ax2.grid(True, linestyle=':', alpha=0.25, color='#334155')
        ax2.legend(loc='upper right', framealpha=0.7, facecolor='#1F2937', edgecolor='#374151', fontsize=8, labelcolor='#E2E8F0')

        # 3. 3D Orbit H
        azim_angle = (45.0 + (t * 360.0 / 900.0)) % 360.0
        elev_angle = 24.0 + 8.0 * np.sin(t * 2.0 * np.pi / 1800.0)
        ax3.view_init(elev=elev_angle, azim=azim_angle)

        ax3.xaxis.set_pane_color((0.067, 0.094, 0.153, 1.0))
        ax3.yaxis.set_pane_color((0.067, 0.094, 0.153, 1.0))
        ax3.zaxis.set_pane_color((0.067, 0.094, 0.153, 1.0))
        ax3.xaxis._axinfo["grid"]['color'] = (0.2, 0.25, 0.35, 0.4)
        ax3.yaxis._axinfo["grid"]['color'] = (0.2, 0.25, 0.35, 0.4)
        ax3.zaxis._axinfo["grid"]['color'] = (0.2, 0.25, 0.35, 0.4)

        ax3.set_xlim(self.meta["lims_3d_h"][0])
        ax3.set_ylim(self.meta["lims_3d_h"][1])
        ax3.set_zlim(self.meta["lims_3d_h"][2])
        ax3.plot(self.pca_h[:, 0], self.pca_h[:, 1], self.pca_h[:, 2], color='#38285E', alpha=0.18, linewidth=0.6)

        if t > 0:
            step = 1 if t < 1000 else 2
            ax3.plot(self.pca_h[:t+1:step, 0], self.pca_h[:t+1:step, 1], self.pca_h[:t+1:step, 2],
                     color='#818CF8', alpha=0.35, linewidth=1.0)

        if k_tail > 1:
            pts_3d = self.pca_h[tail_indices, :3]
            segs_3d = np.concatenate([pts_3d[:-1, np.newaxis, :], pts_3d[1:, np.newaxis, :]], axis=1)
            alphas_3d = np.linspace(0.2, 0.95, len(segs_3d))
            widths_3d = np.linspace(1.2, 3.5, len(segs_3d))
            cmap_pl = matplotlib.colormaps['plasma']
            colors_3d = cmap_pl(np.linspace(0.2, 0.95, len(segs_3d)))
            colors_3d[:, 3] = alphas_3d
            lc_3d = Line3DCollection(segs_3d, colors=colors_3d, linewidths=widths_3d)
            ax3.add_collection3d(lc_3d)

        ax3.scatter(self.pca_h[t, 0], self.pca_h[t, 1], self.pca_h[t, 2],
                    color='#38BDF8', s=90, edgecolors='#FFFFFF', linewidth=1.2, depthshade=False)

        ax3.set_title(f"3D Latent Orbit ($PC_1, PC_2, PC_3$) | Azim={azim_angle:.0f}°",
                      fontsize=11, fontweight='bold', pad=8, color='#E2E8F0')
        ax3.set_xlabel("PC 1", fontsize=9, labelpad=-2, color='#94A3B8')
        ax3.set_ylabel("PC 2", fontsize=9, labelpad=-2, color='#94A3B8')
        ax3.set_zlabel("PC 3", fontsize=9, labelpad=-2, color='#94A3B8')
        ax3.tick_params(colors='#64748B', labelsize=8, pad=-2)

        # 4. Synchronized Video & Telemetry
        if self.mode == "combined":
            comp_rgb, mse_val = self._get_video_frame_rgb(t)
            if comp_rgb is not None:
                ax4_vid.imshow(comp_rgb, aspect='auto')
                title_recon = f"Synchronized Video Frame [t={t:04d}]"
                if mse_val is not None:
                    title_recon += f"  |  MSE Recon Error: {mse_val:.4f}"
                ax4_vid.set_title(title_recon, fontsize=10, fontweight='bold', pad=5, color='#38BDF8')
            else:
                ax4_vid.text(0.5, 0.5, f"Frame Index: {t:04d}", horizontalalignment='center',
                             verticalalignment='center', fontsize=12, color='#94A3B8')
            ax4_vid.axis('off')

            time_window = 300
            w_start = max(0, t - 150)
            w_end = min(self.T, w_start + time_window)
            w_start = max(0, w_end - time_window)
            t_win = np.arange(w_start, w_end)

            ax4_tel.plot(t_win, self.vel_h[t_win], color='#38BDF8', linewidth=1.6, label='Velocity $||h_{t+1} - h_t||_2$')
            ax4_tel.plot(t_win, self.sim_h[t_win], color='#F59E0B', linewidth=1.4, linestyle='--', label='Consecutive $\\cos(h_t, h_{t+1})$')
            ax4_tel.axvline(t, color='#EF4444', linewidth=2.0, alpha=0.9, label=f'Current ($t={t}$)')
            
            ax4_tel.set_xlim(w_start, w_end)
            ax4_tel.set_ylim(-0.05, max(1.05, float(np.max(self.vel_h[t_win])) + 0.1))
            ax4_tel.set_title("Real-Time Transition Dynamics & Continuity (Time Window)",
                              fontsize=10, fontweight='bold', pad=5, color='#E2E8F0')
            ax4_tel.set_xlabel("Frame Index (Timeline $t$)", fontsize=9, color='#94A3B8')
            ax4_tel.set_ylabel("Metric Value", fontsize=9, color='#94A3B8')
            ax4_tel.tick_params(colors='#64748B', labelsize=8)
            ax4_tel.grid(True, linestyle=':', alpha=0.25, color='#334155')
            ax4_tel.legend(loc='upper right', framealpha=0.7, facecolor='#1F2937', edgecolor='#374151', fontsize=7.5, labelcolor='#E2E8F0')

            for spine in ax4_tel.spines.values():
                spine.set_color('#1E293B')

            v_now = self.vel_h[t]
            cos_now = self.sim_h[t]
            hud_text = (
                f"Velocity: {v_now:.4f}\n"
                f"Cos Sim:  {cos_now:.4f}\n"
                f"Norm:     {self.norm_h[t]:.3f}"
            )
            ax4_tel.text(0.02, 0.90, hud_text, transform=ax4_tel.transAxes, fontsize=8,
                         verticalalignment='top', family='monospace', color='#F8FAFC',
                         bbox=dict(boxstyle='round,pad=0.4', facecolor='#1F2937', edgecolor='#374151', alpha=0.85))

        elif self.mode == "quad_manifolds":
            ax4.view_init(elev=elev_angle, azim=azim_angle)
            ax4.xaxis.set_pane_color((0.067, 0.094, 0.153, 1.0))
            ax4.yaxis.set_pane_color((0.067, 0.094, 0.153, 1.0))
            ax4.zaxis.set_pane_color((0.067, 0.094, 0.153, 1.0))
            ax4.xaxis._axinfo["grid"]['color'] = (0.2, 0.25, 0.35, 0.4)
            ax4.yaxis._axinfo["grid"]['color'] = (0.2, 0.25, 0.35, 0.4)
            ax4.zaxis._axinfo["grid"]['color'] = (0.2, 0.25, 0.35, 0.4)

            ax4.set_xlim(self.meta["lims_3d_c"][0])
            ax4.set_ylim(self.meta["lims_3d_c"][1])
            ax4.set_zlim(self.meta["lims_3d_c"][2])
            ax4.plot(self.pca_c[:, 0], self.pca_c[:, 1], self.pca_c[:, 2], color='#134E4A', alpha=0.18, linewidth=0.6)
            if t > 0:
                step = 1 if t < 1000 else 2
                ax4.plot(self.pca_c[:t+1:step, 0], self.pca_c[:t+1:step, 1], self.pca_c[:t+1:step, 2],
                         color='#34D399', alpha=0.35, linewidth=1.0)
            if k_tail > 1:
                pts_3dc = self.pca_c[tail_indices, :3]
                segs_3dc = np.concatenate([pts_3dc[:-1, np.newaxis, :], pts_3dc[1:, np.newaxis, :]], axis=1)
                lc_3dc = Line3DCollection(segs_3dc, colors=colors_c, linewidths=widths_c)
                ax4.add_collection3d(lc_3dc)
            ax4.scatter(self.pca_c[t, 0], self.pca_c[t, 1], self.pca_c[t, 2], color='#10B981', s=90, edgecolors='#FFFFFF', linewidth=1.2)
            ax4.set_title(f"3D Cell State ($c$) Orbit", fontsize=11, fontweight='bold', pad=8, color='#E2E8F0')

        # Header Banner
        fps = 30
        mins = t // (fps * 60)
        secs = (t // fps) % 60
        frames = t % fps
        timecode_str = f"{mins:02d}:{secs:02d}.{frames:02d}"
        progress_pct = (t / max(1, self.T - 1)) * 100.0

        header_title = (
            f"Bad Apple!! Latent Manifold Dynamics & Phase Space Trajectory  •  "
            f"Frame {t:04d} / {self.T-1}  [{timecode_str}]  ({progress_pct:.1f}%)  •  "
            f"Epoch {self.epoch}"
        )
        fig.suptitle(header_title, fontsize=13, fontweight='bold', color='#38BDF8', y=0.975)

        canvas.draw()
        rgba_buffer = np.asarray(canvas.buffer_rgba())
        bgr_frame = cv2.cvtColor(rgba_buffer, cv2.COLOR_RGBA2BGR)
        return bgr_frame

    def _get_video_frame_rgb(self, t: int) -> Tuple[Optional[np.ndarray], Optional[float]]:
        gt_gray = None
        if self.gt_frames_dir:
            frame_path = os.path.join(self.gt_frames_dir, f"frame_{t:06d}.png")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(self.gt_frames_dir, f"frame_{t+1:06d}.png")
            if os.path.exists(frame_path):
                img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    gt_gray = img.astype(np.float32) / 255.0

        rec_gray = None
        if self.fd is not None and self.h_table_tensor is not None:
            with torch.no_grad():
                h_val = self.h_table_tensor[t:t+1].to(self.device)
                pred_logit = self.fd(h_val)
                pred_tensor = torch.sigmoid(pred_logit).squeeze().cpu().numpy()
                rec_gray = np.clip(pred_tensor, 0.0, 1.0)

        def put_shadow_text(img, text, pos, color=(0, 255, 255)):
            x, y = pos
            cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        mse_val = None
        if gt_gray is not None and rec_gray is not None:
            mse_val = float(np.mean((gt_gray - rec_gray) ** 2))
            gt_uint8 = (gt_gray * 255).astype(np.uint8)
            rec_uint8 = (rec_gray * 255).astype(np.uint8)

            gt_rgb = cv2.cvtColor(gt_uint8, cv2.COLOR_GRAY2RGB)
            rec_rgb = cv2.cvtColor(rec_uint8, cv2.COLOR_GRAY2RGB)

            w_orig = gt_rgb.shape[1]
            comp = np.hstack([gt_rgb, rec_rgb])
            put_shadow_text(comp, "Ground Truth", (15, 38), (255, 255, 0))
            put_shadow_text(comp, "Table Decoded (fd(h_t))", (w_orig + 15, 38), (0, 255, 255))
            cv2.line(comp, (w_orig, 0), (w_orig, comp.shape[0]), (80, 80, 80), 2)
            return comp, mse_val
        elif rec_gray is not None:
            rec_uint8 = (rec_gray * 255).astype(np.uint8)
            rec_rgb = cv2.cvtColor(rec_uint8, cv2.COLOR_GRAY2RGB)
            put_shadow_text(rec_rgb, "Table Decoded (fd(h_t))", (15, 38), (0, 255, 255))
            return rec_rgb, None
        elif gt_gray is not None:
            gt_uint8 = (gt_gray * 255).astype(np.uint8)
            gt_rgb = cv2.cvtColor(gt_uint8, cv2.COLOR_GRAY2RGB)
            put_shadow_text(gt_rgb, "Ground Truth", (15, 38), (255, 255, 0))
            return gt_rgb, None

        return None, None


# ---------------------------------------------------------------------------
# High-Performance ThreadPool Video Generation
# ---------------------------------------------------------------------------

def generate_manifold_video(
    checkpoint_path: str,
    output_path: str = "tables_latent_manifold_animation.mp4",
    fps: int = 30,
    tail_length: int = 60,
    start_frame: int = 0,
    num_frames: Optional[int] = None,
    stride: int = 1,
    mode: str = "combined",
    num_workers: int = 12,
    gt_frames_dir: str = "frames"
):
    """
    Renders the complete high-definition latent manifold animation video using a thread pool
    streaming directly to FFmpeg.
    """
    print(f"\n============================================================")
    print(f" Latent Manifold Trajectory Animation Generator")
    print(f" Loading checkpoint: {checkpoint_path}")
    print(f"============================================================")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    epoch = ckpt.get("epoch", 0)
    loss = ckpt.get("loss", float("nan"))
    h_table_tensor = ckpt["tte"]["h_table.weight"]
    c_table_tensor = ckpt["tte"]["c_table.weight"]
    fd_state_dict = ckpt.get("fd", None)

    h_table = h_table_tensor.numpy()
    c_table = c_table_tensor.numpy()

    T, D = h_table.shape
    if num_frames is None:
        end_frame = T
    else:
        end_frame = min(T, start_frame + num_frames)

    frame_indices = list(range(start_frame, end_frame, stride))
    total_render_frames = len(frame_indices)
    print(f" Timeline Range: Frames {start_frame} -> {end_frame} (Total {total_render_frames} frames @ stride={stride})")
    print(f" Video Output: {output_path} ({fps} FPS, {total_render_frames/fps:.1f}s)")
    print(f" Comet Tail: {tail_length} frames | Layout Mode: {mode} | Parallel Threads: {num_workers}")

    meta = compute_manifold_metadata(h_table, c_table)

    fd_module = None
    if fd_state_dict is not None:
        fd_module = FrameDecoder(hidden_size=D)
        fd_module.load_state_dict(fd_state_dict)
        fd_module.eval()

    renderer = ManifoldDashboardRenderer(
        meta=meta,
        epoch=epoch,
        loss=loss,
        tail_length=tail_length,
        mode=mode,
        gt_frames_dir=gt_frames_dir,
        fd_module=fd_module,
        h_table_tensor=h_table_tensor,
        device="cpu"
    )

    sample_frame = renderer.render_frame(frame_indices[0])
    h_vid, w_vid = sample_frame.shape[:2]

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w_vid}x{h_vid}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        '-crf', '18',
        output_path
    ]

    use_ffmpeg = True
    try:
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Warning: Could not launch ffmpeg ({e}), falling back to OpenCV VideoWriter...")
        use_ffmpeg = False
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cv2_writer = cv2.VideoWriter(output_path, fourcc, fps, (w_vid, h_vid))

    print(f"\nRendering {total_render_frames} frames in parallel batches...")
    t0 = time.time()
    frames_written = 0

    batch_size = max(num_workers * 4, 32)
    with cf.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for batch_start in range(0, total_render_frames, batch_size):
            batch_indices = frame_indices[batch_start:batch_start + batch_size]
            futures = [executor.submit(renderer.render_frame, t) for t in batch_indices]
            
            for fut in futures:
                frame_bgr = fut.result()
                if use_ffmpeg:
                    proc.stdin.write(frame_bgr.tobytes())
                else:
                    cv2_writer.write(frame_bgr)
                frames_written += 1

            pct = (frames_written / total_render_frames) * 100.0
            elapsed = time.time() - t0
            cur_fps = frames_written / max(1e-5, elapsed)
            eta = (total_render_frames - frames_written) / max(1e-5, cur_fps)
            sys.stdout.write(f"\r [{frames_written:4d}/{total_render_frames}] ({pct:5.1f}%) | Speed: {cur_fps:5.1f} fps | Elapsed: {elapsed:4.1f}s | ETA: {eta:4.1f}s")
            sys.stdout.flush()

    if use_ffmpeg:
        proc.stdin.close()
        proc.wait()
    else:
        cv2_writer.release()

    t_total = time.time() - t0
    print(f"\n\nVideo rendered successfully in {t_total:.2f}s ({total_render_frames / max(1e-5, t_total):.1f} fps)!")
    print(f"Output saved to: {os.path.abspath(output_path)}")
    if os.path.exists(output_path):
        print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Animate Bad Apple latent manifold trajectories in 2D & 3D.")
    parser.add_argument("--checkpoint", type=str, default="runs/last_model.pt", help="Path to checkpoint .pt file")
    parser.add_argument("--output", type=str, default="tables_latent_manifold_animation.mp4", help="Output MP4 video filename")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for output video")
    parser.add_argument("--tail_length", type=int, default=60, help="Length of the fading comet tail in frames")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to render (default: full video)")
    parser.add_argument("--stride", type=int, default=1, help="Temporal stride / step factor")
    parser.add_argument("--mode", type=str, default="combined", choices=["combined", "quad_manifolds"],
                        help="Dashboard layout mode for video rendering")
    parser.add_argument("--workers", type=int, default=12, help="Number of parallel rendering threads")
    parser.add_argument("--frames_dir", type=str, default="frames", help="Directory containing dataset PNG frames")
    parser.add_argument("--interactive", action="store_true", help="Launch real-time interactive 3D point cloud visualizer with 360° mouse navigation")
    parser.add_argument("--export_html", action="store_true", help="Export standalone interactive HTML5 / WebGL 3D viewer")
    parser.add_argument("--html_out", type=str, default="tables_latent_manifold_3d_interactive.html", help="HTML export filename")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        if os.path.exists("runs/last_model.pt"):
            checkpoint_path = "runs/last_model.pt"
        elif os.path.exists("runs/best_model.pt"):
            checkpoint_path = "runs/best_model.pt"
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    if args.export_html:
        export_html_3d_viewer(
            checkpoint_path=checkpoint_path,
            output_html_path=args.html_out
        )
    elif args.interactive:
        run_interactive_3d_visualizer(
            checkpoint_path=checkpoint_path,
            fps=args.fps,
            gt_frames_dir=args.frames_dir
        )
    else:
        generate_manifold_video(
            checkpoint_path=checkpoint_path,
            output_path=args.output,
            fps=args.fps,
            tail_length=args.tail_length,
            start_frame=args.start_frame,
            num_frames=args.num_frames,
            stride=args.stride,
            mode=args.mode,
            num_workers=args.workers,
            gt_frames_dir=args.frames_dir
        )


if __name__ == "__main__":
    main()
