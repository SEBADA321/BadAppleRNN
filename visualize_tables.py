"""
visualize_tables.py
-------------------
Comprehensive, publication-grade visualizer for the recurrent latent training tables 
(h_table and c_table) and transition dynamics from trained checkpoints.

Generates 4 rich visual analytics dashboards:
1. tables_overview.png             - Latent channel heatmaps, velocities, norms, channel variances, histograms
2. tables_latent_manifold.png      - 2D & 3D PCA phase space trajectories, temporal manifold flow, scree plots
3. tables_temporal_recurrence.png  - Frame-to-frame cosine self-similarity recurrence matrix, autocorrelation
4. tables_transition_dynamics.png   - CTF transition error vs table, LSTM gate activations (f, i, o, c_t)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D


plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#d1d5db'
plt.rcParams['axes.linewidth'] = 1.0


class CustomTransitionFunction(nn.Module):
    """Transition module matching the notebook architecture."""
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
        return h_next, c_next, f, i, o, c_t


def compute_pca(data: torch.Tensor, n_components: int = 10):
    """Computes PCA projections and explained variance ratios using SVD."""
    mean = data.mean(dim=0, keepdim=True)
    centered = data - mean
    U, S, V = torch.pca_lowrank(centered, q=n_components, center=False)
    projected = torch.matmul(centered, V[:, :n_components])
    
    var_explained = (S ** 2) / (data.shape[0] - 1)
    total_var = torch.var(data, dim=0, unbiased=True).sum()
    explained_ratio = (var_explained / total_var).numpy()
    
    return projected.numpy(), V.numpy(), explained_ratio


def plot_tables_overview(h_table: np.ndarray, c_table: np.ndarray, epoch: int, out_path: str):
    """
    Figure 1: Latent channel heatmaps, temporal velocity, norms, variances, and histograms.
    """
    T, D = h_table.shape
    time_steps = np.arange(T)
    
    vel_h = np.linalg.norm(np.diff(h_table, axis=0), axis=1)
    vel_c = np.linalg.norm(np.diff(c_table, axis=0), axis=1)
    
    norm_h = np.linalg.norm(h_table, axis=1)
    norm_c = np.linalg.norm(c_table, axis=1)
    
    var_h = np.var(h_table, axis=0)
    var_c = np.var(c_table, axis=0)
    sorted_var_idx_h = np.argsort(var_h)[::-1]
    
    fig = plt.figure(figsize=(20, 15), dpi=180)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1.0, 1.0], hspace=0.32, wspace=0.22)
    
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(h_table.T, aspect='auto', cmap='magma', interpolation='nearest')
    ax1.set_title(f"Hidden State Table ($h$) — {D} Channels $\\times$ {T} Frames (Epoch {epoch})", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Frame Index (Time $t$)", fontsize=11)
    ax1.set_ylabel("Channel Index", fontsize=11)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    cbar1.set_label("Hidden Value $h$", fontsize=10)
    ax1.grid(False)
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(c_table.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax2.set_title(f"Cell State Table ($c$) — {D} Channels $\\times$ {T} Frames (Epoch {epoch})", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Frame Index (Time $t$)", fontsize=11)
    ax2.set_ylabel("Channel Index", fontsize=11)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
    cbar2.set_label("Cell State Value $c$", fontsize=10)
    ax2.grid(False)
    
    ax3 = fig.add_subplot(gs[1, 0])
    def smooth(x, window=15):
        return np.convolve(x, np.ones(window)/window, mode='same')
    
    ax3.plot(time_steps[:-1], vel_h, color='#2563EB', alpha=0.3, linewidth=0.8, label='$||h_{t+1} - h_t||_2$ (Raw)')
    ax3.plot(time_steps[:-1], smooth(vel_h, 15), color='#1D4ED8', linewidth=1.8, label='$||h_{t+1} - h_t||_2$ (Smoothed)')
    ax3.plot(time_steps[:-1], smooth(vel_c, 15), color='#059669', linewidth=1.8, linestyle='--', label='$||c_{t+1} - c_t||_2$ (Smoothed)')
    ax3.set_title("Temporal State Velocity (Transition Speed across Frames)", fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel("Frame Index (Time $t$)", fontsize=11)
    ax3.set_ylabel("Velocity ($L_2$ Step Delta)", fontsize=11)
    ax3.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    ax3.set_xlim(0, T)
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(time_steps, norm_h, color='#9333EA', linewidth=1.8, label='$||h_t||_2$ (Hidden State Norm)')
    ax4.plot(time_steps, norm_c, color='#D97706', linewidth=1.8, label='$||c_t||_2$ (Cell State Norm)')
    ax4.set_title("State Norm Dynamics ($L_2$ Magnitude across Video Sequence)", fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel("Frame Index (Time $t$)", fontsize=11)
    ax4.set_ylabel("Vector Norm ($L_2$)", fontsize=11)
    ax4.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    ax4.set_xlim(0, T)
    
    ax5 = fig.add_subplot(gs[2, 0])
    channels = np.arange(D)
    ax5.bar(channels - 0.2, var_h[sorted_var_idx_h], width=0.4, color='#3B82F6', alpha=0.85, label='$h$-Table Channel Variance')
    ax5.bar(channels + 0.2, var_c[sorted_var_idx_h], width=0.4, color='#10B981', alpha=0.85, label='$c$-Table Channel Variance')
    ax5.set_title(f"Channel Utilization & Variance (Sorted by $h$-Activity, Top={var_h.max():.3f})", fontsize=13, fontweight='bold', pad=10)
    ax5.set_xlabel("Sorted Channel Rank (0 = Most Active)", fontsize=11)
    ax5.set_ylabel("Variance across All Frames", fontsize=11)
    ax5.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    ax5.set_xlim(-1, D)
    
    ax6 = fig.add_subplot(gs[2, 1])
    h_flat = h_table.flatten()
    c_flat = c_table.flatten()
    ax6.hist(h_flat, bins=120, density=True, alpha=0.65, color='#6366F1', label=f'$h$ Values (Mean={h_flat.mean():.3f}, Std={h_flat.std():.3f})')
    ax6.hist(c_flat, bins=120, density=True, alpha=0.65, color='#F59E0B', label=f'$c$ Values (Mean={c_flat.mean():.3f}, Std={c_flat.std():.3f})')
    ax6.set_title("Embedding Value Distribution (Activation Density)", fontsize=13, fontweight='bold', pad=10)
    ax6.set_xlabel("Latent Coordinate Value", fontsize=11)
    ax6.set_ylabel("Probability Density", fontsize=11)
    ax6.axvline(1.0, color='#EF4444', linestyle=':', alpha=0.7, label='Tanh / Saturation Boundary $\\pm 1.0$')
    ax6.axvline(-1.0, color='#EF4444', linestyle=':', alpha=0.7)
    ax6.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    
    plt.suptitle(f"Bad Apple Latent Training Table Overview — Checkpoint Epoch {epoch} (Dim={D}, Length={T})", fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_latent_manifold(h_table: np.ndarray, c_table: np.ndarray, epoch: int, out_path: str):
    """
    Figure 2: 2D & 3D PCA Phase Space Trajectories and Scree plot.
    """
    T, D = h_table.shape
    
    h_torch = torch.from_numpy(h_table).float()
    c_torch = torch.from_numpy(c_table).float()
    
    pca_h, _, var_ratio_h = compute_pca(h_torch, n_components=10)
    pca_c, _, var_ratio_c = compute_pca(c_torch, n_components=10)
    
    time_steps = np.arange(T)
    
    fig = plt.figure(figsize=(20, 16), dpi=180)
    
    ax1 = fig.add_subplot(2, 2, 1)
    scatter1 = ax1.scatter(pca_h[:, 0], pca_h[:, 1], c=time_steps, cmap='plasma', s=4, alpha=0.75)
    ax1.plot(pca_h[0, 0], pca_h[0, 1], marker='o', markersize=10, color='#10B981', label='Start ($t=0$)', zorder=5)
    ax1.plot(pca_h[-1, 0], pca_h[-1, 1], marker='X', markersize=10, color='#EF4444', label=f'End ($t={T-1}$)', zorder=5)
    ax1.plot(pca_h[:, 0], pca_h[:, 1], color='#64748B', alpha=0.25, linewidth=0.6, zorder=1)
    
    ax1.set_title(f"2D Phase Space Manifold: Hidden State Table ($h$)\nPC1 ({var_ratio_h[0]*100:.1f}%) vs PC2 ({var_ratio_h[1]*100:.1f}%)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Principal Component 1", fontsize=11)
    ax1.set_ylabel("Principal Component 2", fontsize=11)
    cbar1 = fig.colorbar(scatter1, ax=ax1, fraction=0.046, pad=0.03)
    cbar1.set_label("Frame Timeline (0 $\\to$ 6572)", fontsize=10)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    
    ax2 = fig.add_subplot(2, 2, 2)
    scatter2 = ax2.scatter(pca_c[:, 0], pca_c[:, 1], c=time_steps, cmap='viridis', s=4, alpha=0.75)
    ax2.plot(pca_c[0, 0], pca_c[0, 1], marker='o', markersize=10, color='#10B981', label='Start ($t=0$)', zorder=5)
    ax2.plot(pca_c[-1, 0], pca_c[-1, 1], marker='X', markersize=10, color='#EF4444', label=f'End ($t={T-1}$)', zorder=5)
    ax2.plot(pca_c[:, 0], pca_c[:, 1], color='#64748B', alpha=0.25, linewidth=0.6, zorder=1)
    
    ax2.set_title(f"2D Phase Space Manifold: Cell State Table ($c$)\nPC1 ({var_ratio_c[0]*100:.1f}%) vs PC2 ({var_ratio_c[1]*100:.1f}%)", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Principal Component 1", fontsize=11)
    ax2.set_ylabel("Principal Component 2", fontsize=11)
    cbar2 = fig.colorbar(scatter2, ax=ax2, fraction=0.046, pad=0.03)
    cbar2.set_label("Frame Timeline (0 $\\to$ 6572)", fontsize=10)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    scatter3 = ax3.scatter(pca_h[:, 0], pca_h[:, 1], pca_h[:, 2], c=time_steps, cmap='plasma', s=3, alpha=0.65)
    ax3.plot(pca_h[:, 0], pca_h[:, 1], pca_h[:, 2], color='#475569', alpha=0.2, linewidth=0.5)
    ax3.scatter(pca_h[0, 0], pca_h[0, 1], pca_h[0, 2], color='#10B981', s=80, marker='o', label='Start')
    ax3.scatter(pca_h[-1, 0], pca_h[-1, 1], pca_h[-1, 2], color='#EF4444', s=80, marker='X', label='End')
    ax3.set_title("3D Latent Orbit ($PC_1, PC_2, PC_3$) for $h$-Table", fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel("PC 1", fontsize=10)
    ax3.set_ylabel("PC 2", fontsize=10)
    ax3.set_zlabel("PC 3", fontsize=10)
    ax3.view_init(elev=28, azim=45)
    ax3.legend(loc='upper left')
    
    ax4 = fig.add_subplot(2, 2, 4)
    components = np.arange(1, 11)
    
    ax4.plot(components, np.cumsum(var_ratio_h[:10]) * 100, marker='o', color='#2563EB', linewidth=2.0, label='Hidden State ($h$) Cumulative Var %')
    ax4.plot(components, np.cumsum(var_ratio_c[:10]) * 100, marker='s', color='#059669', linewidth=2.0, label='Cell State ($c$) Cumulative Var %')
    ax4.bar(components - 0.15, var_ratio_h[:10] * 100, width=0.3, alpha=0.4, color='#3B82F6', label='Individual PC Var % ($h$)')
    ax4.bar(components + 0.15, var_ratio_c[:10] * 100, width=0.3, alpha=0.4, color='#10B981', label='Individual PC Var % ($c$)')
    
    ax4.set_title("PCA Scree Plot (Intrinsic Dimensionality & Explained Variance)", fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel("Principal Component Index", fontsize=11)
    ax4.set_ylabel("Explained Variance (%)", fontsize=11)
    ax4.set_xticks(components)
    ax4.legend(frameon=True, facecolor='white', framealpha=0.9, loc='center right')
    
    plt.suptitle(f"Latent Manifold & Dimensionality Analysis — Checkpoint Epoch {epoch}", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_temporal_recurrence(h_table: np.ndarray, epoch: int, out_path: str, stride: int = 5):
    """
    Figure 3: Full Self-Similarity Matrix (Temporal Recurrence) and Autocorrelation.
    """
    T, D = h_table.shape
    h_sub = h_table[::stride]
    T_sub = h_sub.shape[0]
    
    norms = np.linalg.norm(h_sub, axis=1, keepdims=True) + 1e-8
    h_normed = h_sub / norms
    sim_matrix = np.matmul(h_normed, h_normed.T)
    
    full_norms = np.linalg.norm(h_table, axis=1, keepdims=True) + 1e-8
    h_full_normed = h_table / full_norms
    consecutive_sim = np.sum(h_full_normed[:-1] * h_full_normed[1:], axis=1)
    
    h_torch = torch.from_numpy(h_table).float()
    pca_h, _, _ = compute_pca(h_torch, n_components=3)
    
    max_lag = 600  
    lags = np.arange(max_lag)
    acf_pc1 = np.array([np.corrcoef(pca_h[:T - lag, 0], pca_h[lag:, 0])[0, 1] if lag > 0 else 1.0 for lag in range(max_lag)])
    acf_pc2 = np.array([np.corrcoef(pca_h[:T - lag, 1], pca_h[lag:, 1])[0, 1] if lag > 0 else 1.0 for lag in range(max_lag)])
    
    fig = plt.figure(figsize=(20, 10), dpi=180)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.2)
    
    ax1 = fig.add_subplot(gs[:, 0])
    im1 = ax1.imshow(sim_matrix, aspect='equal', cmap='magma', origin='upper',
                     extent=[0, T, T, 0], vmin=max(0.0, float(np.percentile(sim_matrix, 2))), vmax=1.0)
    ax1.set_title(f"Full Sequence Cosine Self-Similarity Recurrence Matrix ($S_{{i,j}} = \\cos(h_i, h_j)$)\nReveals Scene Clusters, Chorus Loops, and Repeating Motifs", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Frame Index (Time $t_j$)", fontsize=11)
    ax1.set_ylabel("Frame Index (Time $t_i$)", fontsize=11)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    cbar1.set_label("Cosine Similarity", fontsize=10)
    ax1.grid(False)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(consecutive_sim, color='#2563EB', linewidth=1.0, alpha=0.85)
    ax2.set_title("Consecutive Frame Cosine Similarity ($\\cos(h_t, h_{t+1})$)", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Frame Index (Time $t$)", fontsize=11)
    ax2.set_ylabel("Cosine Similarity", fontsize=11)
    ax2.set_ylim(min(0.5, float(np.min(consecutive_sim)) - 0.05), 1.005)
    ax2.set_xlim(0, T)
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(lags, acf_pc1, color='#DC2626', linewidth=1.8, label='$PC_1$ Autocorrelation')
    ax3.plot(lags, acf_pc2, color='#9333EA', linewidth=1.8, label='$PC_2$ Autocorrelation')
    ax3.axhline(0.0, color='#64748B', linestyle='--', alpha=0.6)
    ax3.set_title(f"Temporal Autocorrelation Function (Lags up to {max_lag} frames)", fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel("Lag $\\tau$ (Frames)", fontsize=11)
    ax3.set_ylabel("Autocorrelation $R(\\tau)$", fontsize=11)
    ax3.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    ax3.set_xlim(0, max_lag)
    
    plt.suptitle(f"Temporal Structure & Recurrence Analysis — Checkpoint Epoch {epoch}", fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_transition_dynamics(h_table: torch.Tensor, c_table: torch.Tensor, ctf: nn.Module, epoch: int, out_path: str):
    """
    Figure 4: Transition function consistency vs table, and LSTM gate inspection (f, i, o, c_t).
    """
    T, D = h_table.shape
    time_steps = np.arange(T - 1)
    
    with torch.no_grad():
        h_pred, c_pred, f_gate, i_gate, o_gate, c_cand = ctf(h_table[:-1], c_table[:-1])
        
        err_h = torch.norm(h_pred - h_table[1:], dim=-1).cpu().numpy()
        err_c = torch.norm(c_pred - c_table[1:], dim=-1).cpu().numpy()
        
        f_mean = f_gate.mean(dim=-1).cpu().numpy()
        i_mean = i_gate.mean(dim=-1).cpu().numpy()
        o_mean = o_gate.mean(dim=-1).cpu().numpy()
        c_cand_mean = c_cand.mean(dim=-1).cpu().numpy()
        
        f_gate_np = f_gate.cpu().numpy()
        i_gate_np = i_gate.cpu().numpy()
        o_gate_np = o_gate.cpu().numpy()
    
    fig = plt.figure(figsize=(20, 14), dpi=180)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.0, 1.1, 1.1], hspace=0.35, wspace=0.22)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_steps, err_h, color='#2563EB', linewidth=1.2, alpha=0.7, label=f'$h$ Transition Error $||h_{{t+1}} - \\hat{{h}}_{{t+1}}||_2$ (Mean={err_h.mean():.4f})')
    ax1.plot(time_steps, err_c, color='#D97706', linewidth=1.2, alpha=0.7, label=f'$c$ Transition Error $||c_{{t+1}} - \\hat{{c}}_{{t+1}}||_2$ (Mean={err_c.mean():.4f})')
    ax1.set_title(f"One-Step CTF Transition Consistency Error vs Table Ground Truth (Epoch {epoch})", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Frame Index (Time $t$)", fontsize=11)
    ax1.set_ylabel("$L_2$ Transition Discrepancy", fontsize=11)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    ax1.set_xlim(0, T - 1)
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_steps, f_mean, color='#10B981', linewidth=1.5, label=f'Forget Gate $f_t$ (Mean={f_mean.mean():.3f})')
    ax2.plot(time_steps, i_mean, color='#F59E0B', linewidth=1.5, label=f'Input Gate $i_t$ (Mean={i_mean.mean():.3f})')
    ax2.plot(time_steps, o_mean, color='#8B5CF6', linewidth=1.5, label=f'Output Gate $o_t$ (Mean={o_mean.mean():.3f})')
    ax2.set_title("Average LSTM Gate Activations across Time", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Frame Index (Time $t$)", fontsize=11)
    ax2.set_ylabel("Gate Activation Value $[0, 1]$", fontsize=11)
    ax2.set_ylim(0.0, 1.05)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    ax2.set_xlim(0, T - 1)
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(f_gate_np.flatten(), bins=80, density=True, alpha=0.6, color='#10B981', label='Forget Gate ($f$)')
    ax3.hist(i_gate_np.flatten(), bins=80, density=True, alpha=0.6, color='#F59E0B', label='Input Gate ($i$)')
    ax3.hist(o_gate_np.flatten(), bins=80, density=True, alpha=0.6, color='#8B5CF6', label='Output Gate ($o$)')
    ax3.set_title("Gate Activation Distributions (Channel-wise Density)", fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel("Gate Value $[0, 1]$", fontsize=11)
    ax3.set_ylabel("Probability Density", fontsize=11)
    ax3.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper center')
    
    ax4 = fig.add_subplot(gs[2, 0])
    im4 = ax4.imshow(f_gate_np.T, aspect='auto', cmap='Greens', interpolation='nearest', vmin=0.0, vmax=1.0)
    ax4.set_title("Forget Gate $f_t$ per Channel over Time", fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel("Frame Index (Time $t$)", fontsize=11)
    ax4.set_ylabel("Channel Index", fontsize=11)
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.03)
    ax4.grid(False)
    
    ax5 = fig.add_subplot(gs[2, 1])
    im5 = ax5.imshow(i_gate_np.T, aspect='auto', cmap='YlOrBr', interpolation='nearest', vmin=0.0, vmax=1.0)
    ax5.set_title("Input Gate $i_t$ per Channel over Time", fontsize=13, fontweight='bold', pad=10)
    ax5.set_xlabel("Frame Index (Time $t$)", fontsize=11)
    ax5.set_ylabel("Channel Index", fontsize=11)
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.03)
    ax5.grid(False)
    
    plt.suptitle(f"Recurrent Transition Dynamics & Gate Diagnostics — Checkpoint Epoch {epoch}", fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Bad Apple latent training tables and transition dynamics.")
    parser.add_argument("--checkpoint", type=str, default="runs/last_model.pt", help="Path to checkpoint .pt file")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save generated plot images")
    parser.add_argument("--animate", action="store_true", help="Generate animated MP4 video of 2D & 3D latent manifold trajectories")
    parser.add_argument("--video_name", type=str, default="tables_latent_manifold_animation.mp4", help="Video output filename")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for manifold animation video")
    parser.add_argument("--tail_length", type=int, default=60, help="Comet tail length in frames")
    parser.add_argument("--stride", type=int, default=1, help="Temporal stride / step factor for animation video")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        
        if os.path.exists("runs/last_model.pt"):
            checkpoint_path = "runs/last_model.pt"
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"============================================================")
    print(f" Loading checkpoint: {checkpoint_path}")
    print(f"============================================================")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("nan"))
    best_loss = checkpoint.get("best_loss", float("nan"))
    
    h_table_weight = checkpoint["tte"]["h_table.weight"]  
    c_table_weight = checkpoint["tte"]["c_table.weight"]  
    
    T, D = h_table_weight.shape
    print(f" Epoch: {epoch} | Current Loss: {loss:.6f} | Best Loss: {best_loss:.6f}")
    print(f" Table Entries (T): {T} frames | Latent Dimension (D): {D} channels")
    print(f" H-Table Norm: {torch.norm(h_table_weight, dim=-1).mean():.4f} (Mean) | Std: {h_table_weight.std():.4f}")
    print(f" C-Table Norm: {torch.norm(c_table_weight, dim=-1).mean():.4f} (Mean) | Std: {c_table_weight.std():.4f}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    ctf = None
    if "ctf" in checkpoint:
        ctf = CustomTransitionFunction(hidden_size=D)
        ctf.load_state_dict(checkpoint["ctf"])
        ctf.eval()

    h_np = h_table_weight.numpy()
    c_np = c_table_weight.numpy()

    p1 = os.path.join(args.output_dir, "tables_overview.png")
    p2 = os.path.join(args.output_dir, "tables_latent_manifold.png")
    p3 = os.path.join(args.output_dir, "tables_temporal_recurrence.png")
    p4 = os.path.join(args.output_dir, "tables_transition_dynamics.png")

    print("\n[1/4] Generating Tables Overview Dashboard...")
    plot_tables_overview(h_np, c_np, epoch, p1)

    print("[2/4] Generating Latent Manifold & Phase Space Plots...")
    plot_latent_manifold(h_np, c_np, epoch, p2)

    print("[3/4] Generating Temporal Recurrence & Self-Similarity Plots...")
    plot_temporal_recurrence(h_np, epoch, p3, stride=5)

    if ctf is not None:
        print("[4/4] Generating Transition Dynamics & Gate Analytics...")
        plot_transition_dynamics(h_table_weight, c_table_weight, ctf, epoch, p4)
    else:
        print("[4/4] Skipping CTF transition dynamics (no 'ctf' key in checkpoint).")

    if args.animate:
        print("\n[5/5] Generating 2D & 3D Latent Manifold Trajectory Animation Video...")
        from animate_manifold import generate_manifold_video
        video_out_path = os.path.join(args.output_dir, args.video_name)
        generate_manifold_video(
            checkpoint_path=checkpoint_path,
            output_path=video_out_path,
            fps=args.fps,
            tail_length=args.tail_length,
            stride=args.stride
        )

    print("\nAll visualizations generated successfully!")
    print(f"1. {os.path.abspath(p1)}")
    print(f"2. {os.path.abspath(p2)}")
    print(f"3. {os.path.abspath(p3)}")
    if ctf is not None:
        print(f"4. {os.path.abspath(p4)}")
    if args.animate:
        print(f"5. {os.path.abspath(os.path.join(args.output_dir, args.video_name))}")


if __name__ == "__main__":
    main()
