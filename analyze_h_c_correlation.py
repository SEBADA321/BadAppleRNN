"""
analyze_h_c_correlation.py
--------------------------
Quantitative and visual analysis of the correlation, redundancy, and functional differences
between the Hidden State Table (h) and Cell State Table (c) in the trained Bad Apple model.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def calc_pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates Pearson correlation coefficient between two 1D numpy arrays."""
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    denom = np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)) + 1e-12
    return float(np.sum(vx * vy) / denom)


def analyze_h_and_c(checkpoint_path: str = "runs/best_model.pt", out_png: str = "exports/plots/h_c_correlation_analysis.png"):
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    tte = ckpt["tte"]
    ctf_sd = ckpt["ctf"]

    h = tte["h_table.weight"].numpy()  # [6573, 64]
    c = tte["c_table.weight"].numpy()  # [6573, 64]
    T, D = h.shape

    # 1. Cosine similarity per frame
    dot = np.sum(h * c, axis=1)
    norm_h = np.linalg.norm(h, axis=1)
    norm_c = np.linalg.norm(c, axis=1)
    cos_sim = dot / (norm_h * norm_c + 1e-8)

    # 2. Channel-wise Pearson r
    channel_corrs = np.array([calc_pearson_r(h[:, d], c[:, d]) for d in range(D)])

    # 3. Global correlation
    global_pearson = calc_pearson_r(h.flatten(), c.flatten())

    # 4. Gate activations
    W_hf = ctf_sd["W_hf.weight"].numpy(); b_hf = ctf_sd["W_hf.bias"].numpy()
    W_hi = ctf_sd["W_hi.weight"].numpy(); b_hi = ctf_sd["W_hi.bias"].numpy()
    W_ho = ctf_sd["W_ho.weight"].numpy(); b_ho = ctf_sd["W_ho.bias"].numpy()
    W_hg = ctf_sd["W_hg.weight"].numpy(); b_hg = ctf_sd["W_hg.bias"].numpy()

    def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    f_gate = sigmoid(h @ W_hf.T + b_hf)
    i_gate = sigmoid(h @ W_hi.T + b_hi)
    o_gate = sigmoid(h @ W_ho.T + b_ho)
    c_candidate = np.tanh(h @ W_hg.T + b_hg)

    # Algebraic coupling: h vs o * tanh(c)
    tanh_c = np.tanh(c)
    o_tanh_c = o_gate * tanh_c
    diff_coupled = np.linalg.norm(h - o_tanh_c, axis=1)
    diff_raw = np.linalg.norm(h - c, axis=1)

    # 5. Velocity / Drift comparison
    vel_h = np.linalg.norm(np.diff(h, axis=0), axis=1)
    vel_c = np.linalg.norm(np.diff(c, axis=0), axis=1)

    # Print summary statistics
    print("\n" + "=" * 65)
    print("  QUANTITATIVE CORRELATION METRICS: h vs c")
    print("=" * 65)
    print(f"• Global Pearson Correlation (all 6573x64 values): r = {global_pearson:.4f}")
    print(f"• Frame-wise Cosine Similarity (mean +/- std):      {np.mean(cos_sim):.4f} +/- {np.std(cos_sim):.4f}")
    print(f"  - Min Cosine Sim: {np.min(cos_sim):.4f} | Max: {np.max(cos_sim):.4f} | Median: {np.median(cos_sim):.4f}")
    print(f"• Channel-wise Pearson r across time:             {np.mean(channel_corrs):.4f} +/- {np.std(channel_corrs):.4f}")
    print(f"  - Channels with |r| > 0.90:                      {np.sum(np.abs(channel_corrs) > 0.90)} / 64 ({np.sum(np.abs(channel_corrs) > 0.90)/64*100:.1f}%)")
    print(f"  - Channels with |r| > 0.98:                      {np.sum(np.abs(channel_corrs) > 0.98)} / 64 ({np.sum(np.abs(channel_corrs) > 0.98)/64*100:.1f}%)")
    print(f"• Mean State Norms:                                ||h|| = {np.mean(norm_h):.2f}  vs  ||c|| = {np.mean(norm_c):.2f}  (Ratio: {np.mean(norm_c)/np.mean(norm_h):.2f}x)")
    print(f"• Algebraic Identity Error:                       ||h - o*tanh(c)|| = {np.mean(diff_coupled):.4f} (tight LSTM coupling)")
    print(f"• Mean Output Gate Activation (o):                 {np.mean(o_gate):.4f} +/- {np.std(o_gate):.4f}")
    print("=" * 65 + "\n")

    # ==============================================================================
    # Generate 6-Panel Diagnostic Dashboard
    # ==============================================================================
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#CBD5E1'

    fig = plt.figure(figsize=(20, 14), dpi=250)
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.22)

    # Subplot 1: Channel-by-Channel Correlation Bar Plot
    ax1 = fig.add_subplot(gs[0, 0])
    sorted_idx = np.argsort(channel_corrs)[::-1]
    bars = ax1.bar(range(D), channel_corrs[sorted_idx], color='#3B82F6', alpha=0.85, width=0.8, edgecolor='#1E40AF')
    ax1.axhline(0.9, color='#DC2626', linestyle='--', linewidth=1.5, label=f'High Correlation Threshold (r=0.90, {np.sum(channel_corrs > 0.9)}/64 channels)')
    ax1.axhline(np.mean(channel_corrs), color='#059669', linestyle='-', linewidth=2.0, label=f'Mean Channel r = {np.mean(channel_corrs):.3f}')
    ax1.set_title("1. Channel-Wise Pearson Correlation $r(h_d, c_d)$ Across Time (Ranked)", fontsize=11.5, fontweight='bold', pad=10)
    ax1.set_xlabel("Sorted Channel Rank (0 to 63)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Pearson Correlation $r$", fontsize=10, fontweight='bold')
    ax1.set_ylim(0.4, 1.05)
    ax1.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.95, fontsize=8.5)

    # Subplot 2: Temporal Cosine Similarity Timeline
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(cos_sim, color='#7C3AED', linewidth=1.2, label=rf'Frame Alignment $\cos(h_t, c_t)$ (Mean: {np.mean(cos_sim):.3f})')
    ax2.axhline(np.mean(cos_sim), color='#4C1D95', linestyle='--', linewidth=1.5)
    ax2.set_title(r"2. Frame-Wise Cosine Similarity Timeline $\cos(h_t, c_t) = \frac{h_t \cdot c_t}{\|h_t\| \|c_t\|}$", fontsize=11.5, fontweight='bold', pad=10)
    ax2.set_xlabel("Frame Index (Timeline $t=0 \to 6573$)", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Cosine Similarity", fontsize=10, fontweight='bold')
    ax2.set_ylim(0.2, 1.02)
    ax2.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, fontsize=8.5)

    # Subplot 3: 2D Scatter: Raw h vs Raw c (Showing Scale & Monotonicity)
    ax3 = fig.add_subplot(gs[1, 0])
    sample_indices = np.random.RandomState(42).choice(h.size, size=6000, replace=False)
    h_sample = h.flatten()[sample_indices]
    c_sample = c.flatten()[sample_indices]
    ax3.scatter(c_sample, h_sample, alpha=0.15, s=6, color='#2563EB', rasterized=True, label='Subsampled Element Pairs $(c_{t,d}, h_{t,d})$')
    
    # Fit line
    poly = np.poly1d(np.polyfit(c.flatten(), h.flatten(), 1))
    x_grid = np.linspace(c.min(), c.max(), 200)
    ax3.plot(x_grid, poly(x_grid), color='#DC2626', linewidth=2.2, label=f'Linear Fit: $h \\approx {poly[1]:.2f}c {poly[0]:+.2f}$ ($R^2 = {global_pearson**2:.3f}$)')
    ax3.plot(x_grid, np.tanh(x_grid), color='#059669', linestyle=':', linewidth=2.0, label=r'Theoretical Bound: $y = \tanh(c)$')
    ax3.set_title(f"3. Latent Value Joint Distribution ($h$ vs $c$) [Global $r = {global_pearson:.3f}$]", fontsize=11.5, fontweight='bold', pad=10)
    ax3.set_xlabel("Cell State Value ($c$)", fontsize=10, fontweight='bold')
    ax3.set_ylabel("Hidden State Value ($h$)", fontsize=10, fontweight='bold')
    ax3.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, fontsize=8.5)

    # Subplot 4: Dynamic Range & State Norm Evolution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(norm_c, color='#D97706', linewidth=1.4, alpha=0.85, label=rf'Cell Norm $\|c_t\|_2$ (Mean: {np.mean(norm_c):.2f})')
    ax4.plot(norm_h, color='#2563EB', linewidth=1.4, alpha=0.9, label=rf'Hidden Norm $\|h_t\|_2$ (Mean: {np.mean(norm_h):.2f})')
    ax4.plot(norm_c / (norm_h + 1e-6), color='#059669', linestyle='--', linewidth=1.2, alpha=0.75, label=rf'Norm Ratio $\|c\| / \|h\|$ (Mean: {np.mean(norm_c/norm_h):.2f}x)')
    ax4.set_title("4. State Energy & Dynamic Range Difference (Norm Scaling)", fontsize=11.5, fontweight='bold', pad=10)
    ax4.set_xlabel("Frame Index", fontsize=10, fontweight='bold')
    ax4.set_ylabel("Euclidean Norm", fontsize=10, fontweight='bold')
    ax4.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=8.5)

    # Subplot 5: Gate Dynamics Modulating the Coupling
    ax5 = fig.add_subplot(gs[2, 0])
    mean_f = np.mean(f_gate, axis=1)
    mean_i = np.mean(i_gate, axis=1)
    mean_o = np.mean(o_gate, axis=1)
    ax5.plot(mean_f, color='#2563EB', linewidth=1.2, alpha=0.8, label=f'Forget Gate $f$ (Mean: {np.mean(mean_f):.2f})')
    ax5.plot(mean_i, color='#EA580C', linewidth=1.2, alpha=0.8, label=f'Input Gate $i$ (Mean: {np.mean(mean_i):.2f})')
    ax5.plot(mean_o, color='#16A34A', linewidth=1.3, alpha=0.9, label=r'Output Gate $o$ (Mean: ' + f'{np.mean(mean_o):.2f}' + r' — Modulates $h = o \odot \tanh(c)$)')
    ax5.set_title("5. LSTM Gate Dynamics Across the Video Timeline", fontsize=11.5, fontweight='bold', pad=10)
    ax5.set_xlabel("Frame Index", fontsize=10, fontweight='bold')
    ax5.set_ylabel("Mean Gate Activation $[0, 1]$", fontsize=10, fontweight='bold')
    ax5.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=8.5)

    # Subplot 6: Algebraic Coupling Error ||h - o*tanh(c)|| vs ||h - c||
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(diff_raw, color='#DC2626', linewidth=1.2, alpha=0.7, label=rf'Raw Distance $\|h_t - c_t\|_2$ (Mean: {np.mean(diff_raw):.2f})')
    ax6.plot(diff_coupled, color='#059669', linewidth=1.4, alpha=0.9, label=rf'Gated Residual $\|h_t - o_t \odot \tanh(c_t)\|_2$ (Mean: {np.mean(diff_coupled):.2f})')
    ax6.set_title(r"6. Verification of Algebraic Mechanism: $h_t = o_t \odot \tanh(c_t)$", fontsize=11.5, fontweight='bold', pad=10)
    ax6.set_xlabel("Frame Index", fontsize=10, fontweight='bold')
    ax6.set_ylabel("Residual Euclidean Error", fontsize=10, fontweight='bold')
    ax6.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=8.5)

    fig.suptitle("Bad Apple Latent Representations: $h$ vs $c$ Correlation & Architectural Redundancy Analysis", fontsize=14.5, fontweight='bold', y=0.99)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_inches='tight', dpi=250)
    plt.close(fig)
    print(f"Saved diagnostic dashboard to: {out_png}")

    # Also save a JSON summary report
    stats_json = {
        "global_pearson_r": float(global_pearson),
        "mean_cosine_similarity": float(np.mean(cos_sim)),
        "std_cosine_similarity": float(np.std(cos_sim)),
        "min_cosine_similarity": float(np.min(cos_sim)),
        "max_cosine_similarity": float(np.max(cos_sim)),
        "mean_channel_pearson_r": float(np.mean(channel_corrs)),
        "channels_with_r_gt_0_90": int(np.sum(channel_corrs > 0.90)),
        "channels_with_r_gt_0_98": int(np.sum(channel_corrs > 0.98)),
        "mean_norm_h": float(np.mean(norm_h)),
        "mean_norm_c": float(np.mean(norm_c)),
        "mean_norm_ratio_c_over_h": float(np.mean(norm_c) / np.mean(norm_h)),
        "mean_algebraic_coupling_residual": float(np.mean(diff_coupled)),
        "mean_raw_euclidean_distance": float(np.mean(diff_raw)),
        "gate_statistics": {
            "forget_gate_f_mean": float(np.mean(f_gate)),
            "input_gate_i_mean": float(np.mean(i_gate)),
            "output_gate_o_mean": float(np.mean(o_gate))
        }
    }
    json_path = os.path.splitext(out_png)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(stats_json, f, indent=2)
    print(f"Saved JSON metrics to: {json_path}")


if __name__ == "__main__":
    analyze_h_and_c()
