import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0

log_path = "/home/tumi/Desktop/BadApple/logs.txt"
artifact_dirs = [
    "/home/tumi/.gemini/antigravity-ide/brain/f10a2290-d9b0-4da7-8af5-1d1169d77714",
    "/home/tumi/.gemini/antigravity-ide/brain/99d3b20d-29e9-4927-b41d-820d3163e3c1"
]
for ad in artifact_dirs:
    os.makedirs(ad, exist_ok=True)

epochs, ks, batches, losses, recons, dyn_hs, dyn_cs = [], [], [], [], [], [], []

pattern = re.compile(
    r"Epoch\s+(\d+)\s*\|\s*K=\s*(\d+)\s*\|\s*Batch=\s*(\d+)\s*\|\s*Loss=([0-9.]+)\s*\|\s*Recon=([0-9.]+)\s*\|\s*DynH=([0-9.]+)\s*\|\s*DynC=([0-9.]+)"
)

with open(log_path, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            epochs.append(int(m.group(1)))
            ks.append(int(m.group(2)))
            batches.append(int(m.group(3)))
            losses.append(float(m.group(4)))
            recons.append(float(m.group(5)))
            dyn_hs.append(float(m.group(6)))
            dyn_cs.append(float(m.group(7)))

epochs = np.array(epochs)
ks = np.array(ks)
batches = np.array(batches)
losses = np.array(losses)
recons = np.array(recons)
dyn_hs = np.array(dyn_hs)
dyn_cs = np.array(dyn_cs)

phases = epochs // 50
unique_phases = sorted(list(set(phases)))

stages = []
k_counts = {}
for p in unique_phases:
    mask = (phases == p)
    ep_p = epochs[mask]
    k_p = ks[mask][0]
    b_p = batches[mask][0]
    
    k_counts[k_p] = k_counts.get(k_p, 0) + 1
    stages.append({
        'phase': p,
        'mask': mask,
        'k': k_p,
        'batch': b_p,
        'set_idx': k_counts[k_p],
        'start_ep': ep_p[0],
        'end_ep': ep_p[-1],
        'count': len(ep_p)
    })

k_has_multiple_sets = {k: count > 1 for k, count in k_counts.items()}

stage_colors_list = [
    '#2563EB',
    '#0D9488',
    '#16A34A',
    '#D97706',
    '#DC2626',
    '#9333EA',
    '#0284C7',
    '#E11D48',
    '#EA580C',
    '#CA8A04',
    '#059669',
    '#4F46E5',
    '#DB2777',
    '#0891B2',
    '#65A30D',
    '#7C3AED',
]

for idx, s in enumerate(stages):
    s['color'] = stage_colors_list[idx % len(stage_colors_list)]
    k_val = s['k']
    set_idx = s['set_idx']
    if k_has_multiple_sets[k_val]:
        s['name'] = f"K={k_val} (Set {set_idx})"
        s['short_name'] = f"K={k_val} [S{set_idx}]"
        s['full_label'] = f"Stage {s['phase']+1}: K={k_val} [Set {set_idx}]"
    else:
        s['name'] = f"K={k_val}"
        s['short_name'] = f"K={k_val}"
        s['full_label'] = f"Stage {s['phase']+1}: K={k_val}"

max_epoch = int(np.max(epochs))
max_loss = float(np.max(losses))

fig1 = plt.figure(figsize=(19, 13.0), dpi=300)
gs = fig1.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

ax1 = fig1.add_subplot(gs[0, 0])
ax1.plot(epochs, losses, color='#0F172A', linewidth=2.2, label='Total Loss (Combined)', zorder=4)
ax1.plot(epochs, recons, color='#3B82F6', linewidth=1.6, linestyle='--', alpha=0.9, label='Recon Loss (Image MSE)', zorder=3)

for s in stages:
    start_ep = s['start_ep']
    end_ep = s['end_ep']
    color = s['color']
    ax1.axvspan(start_ep - 0.5, end_ep + 0.5, color=color, alpha=0.09, zorder=1)
    
    mid_ep = (start_ep + end_ep) / 2
    y_pos = 0.58 if s['phase'] % 2 == 0 else 0.47
    ax1.text(mid_ep, y_pos, f"{s['short_name']}\nB={s['batch']}", 
             ha='center', va='center', fontsize=8.5, fontweight='bold', color=color,
             bbox=dict(boxstyle='round,pad=0.30', facecolor='white', edgecolor=color, alpha=0.92, linewidth=1.2))
    if start_ep > 0:
        ax1.axvline(start_ep - 0.5, color='#64748B', linestyle=':', linewidth=1.3, alpha=0.8)

ax1.set_title("Total Loss & Reconstruction Loss across Curriculum Stages & Sets (Horizon K ≤ 512)", fontsize=12.5, fontweight='bold', pad=12)
ax1.set_xlabel("Epoch", fontsize=11, fontweight='bold')
ax1.set_ylabel("Loss Magnitude", fontsize=11, fontweight='bold')
ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#CBD5E1', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_xlim(-4, max_epoch + 8)
ax1.set_ylim(-0.01, max(0.74, max_loss * 1.06))

ax2 = fig1.add_subplot(gs[0, 1])
ax2.plot(epochs, dyn_hs, color='#059669', linewidth=2.0, label='DynH (Hidden State Dynamics)', zorder=4)
ax2.plot(epochs, dyn_cs, color='#EA580C', linewidth=2.0, label='DynC (Cell State Dynamics)', zorder=4)

for s in stages:
    start_ep = s['start_ep']
    end_ep = s['end_ep']
    color = s['color']
    ax2.axvspan(start_ep - 0.5, end_ep + 0.5, color=color, alpha=0.08, zorder=1)
    if start_ep > 0:
        ax2.axvline(start_ep - 0.5, color='#64748B', linestyle=':', linewidth=1.3, alpha=0.8)

ax2.set_yscale('log')
ax2.set_title("Latent Dynamics Loss Progression (DynH & DynC) [Log Scale]", fontsize=12.5, fontweight='bold', pad=12)
ax2.set_xlabel("Epoch", fontsize=11, fontweight='bold')
ax2.set_ylabel("Dynamics Loss (Log Scale)", fontsize=11, fontweight='bold')
ax2.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#CBD5E1', fontsize=10)
ax2.grid(True, which="both", linestyle='--', alpha=0.5)
ax2.set_xlim(-4, max_epoch + 8)

ax3 = fig1.add_subplot(gs[1, 0])
for s in stages:
    mask = s['mask']
    stage_loss = losses[mask]
    stage_epochs = np.arange(len(stage_loss))
    color = s['color']
    
    if len(stage_loss) == 1:
        label_text = f"{s['full_label']} (1 ep: {stage_loss[0]:.3f})"
    else:
        label_text = f"{s['full_label']} ({len(stage_loss)} eps: {stage_loss[0]:.3f} → {stage_loss[-1]:.3f})"
        
    ax3.plot(stage_epochs, stage_loss, marker='o', markersize=3.5, linewidth=2.0, 
             color=color, label=label_text)

ax3.set_yscale('log')
ax3.set_yticks([0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70])
ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax3.yaxis.set_minor_locator(ticker.NullLocator())
ax3.set_title("Intra-Stage Convergence Comparison (Relative Epochs 0 → 49 per Set) [Log Scale]", fontsize=12.5, fontweight='bold', pad=12)
ax3.set_xlabel("Relative Epoch in Stage / Set", fontsize=11, fontweight='bold')
ax3.set_ylabel("Total Loss (Log Scale)", fontsize=11, fontweight='bold')
ax3.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#CBD5E1', fontsize=8.5, ncol=1)
ax3.grid(True, which="major", linestyle='--', alpha=0.5)
ax3.set_xlim(-1, 52)
ax3.set_ylim(0.055, 0.80)

ax4 = fig1.add_subplot(gs[1, 1])
dyn_total = dyn_hs + dyn_cs
dyn_fraction_pct = (dyn_total / losses) * 100.0

ax4.plot(epochs, dyn_fraction_pct, color='#7C3AED', linewidth=2.3, label='Total Dynamics / Loss (%)', zorder=4)
ax4.plot(epochs, (dyn_hs / losses) * 100.0, color='#059669', linewidth=1.6, linestyle='--', label='DynH / Loss (%)', zorder=3)
ax4.plot(epochs, (dyn_cs / losses) * 100.0, color='#EA580C', linewidth=1.6, linestyle='--', label='DynC / Loss (%)', zorder=3)

for s in stages:
    start_ep = s['start_ep']
    end_ep = s['end_ep']
    color = s['color']
    ax4.axvspan(start_ep - 0.5, end_ep + 0.5, color=color, alpha=0.08, zorder=1)
    if start_ep > 0:
        ax4.axvline(start_ep - 0.5, color='#64748B', linestyle=':', linewidth=1.3, alpha=0.8)

ax4.set_title("Dynamics Loss as Percentage of Total Loss (Relative Impact)", fontsize=12.5, fontweight='bold', pad=12)
ax4.set_xlabel("Epoch", fontsize=11, fontweight='bold')
ax4.set_ylabel("Dynamics Contribution (%)", fontsize=11, fontweight='bold')
ax4.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#CBD5E1', fontsize=10)
ax4.grid(True, linestyle='--', alpha=0.5)
ax4.set_xlim(-4, max_epoch + 8)
ax4.set_ylim(-0.5, max(18.0, float(np.max(dyn_fraction_pct)) * 1.15))

fig1.suptitle("Bad Apple Latent Video Dynamics & Reconstruction Loss Analysis (Curriculum & Fixed-K Sets)", fontsize=15.5, fontweight='bold', y=0.985)

out_dashboard_workspace = "/home/tumi/Desktop/BadApple/losses_overview.png"
plt.savefig(out_dashboard_workspace, bbox_inches='tight', dpi=300)
for ad in artifact_dirs:
    plt.savefig(os.path.join(ad, "losses_overview.png"), bbox_inches='tight', dpi=300)
plt.close(fig1)

num_stages = len(stages)
fig2_height = max(14.0, 2.3 * num_stages)
fig2, axes = plt.subplots(num_stages, 3, figsize=(18, fig2_height), sharex=False, dpi=250)
if num_stages == 1:
    axes = np.atleast_2d(axes)
fig2.subplots_adjust(hspace=0.48, wspace=0.25)

for i, s in enumerate(stages):
    mask = s['mask']
    ep = epochs[mask]
    l = losses[mask]
    r = recons[mask]
    dh = dyn_hs[mask]
    dc = dyn_cs[mask]
    c = s['color']
    stage_title_ep = f"Epochs {ep[0]}-{ep[-1]}" if len(ep) > 1 else f"Epoch {ep[0]}"

    ax_lr = axes[i, 0]
    ax_lr.plot(ep, l, color=c, marker='o', markersize=3.5, linewidth=2.0, label='Loss')
    ax_lr.plot(ep, r, color='#64748B', marker='s', markersize=3.0, linestyle='--', linewidth=1.5, label='Recon')
    ax_lr.set_title(f"{s['full_label']} ({stage_title_ep}): Total & Recon Loss", fontsize=10, fontweight='bold')
    ax_lr.set_ylabel("Loss", fontsize=9)
    ax_lr.grid(True, linestyle='--', alpha=0.4)
    ax_lr.legend(loc='upper right', fontsize=8)
    
    if len(l) > 1:
        drop_pct = ((l[0] - l[-1]) / l[0]) * 100.0
        drop_text = f'Drop: {-drop_pct:.1f}%\n({l[0]:.4f} → {l[-1]:.4f})'
    else:
        drop_text = f'Loss: {l[0]:.4f}\n(Epoch {ep[0]})'
    ax_lr.text(0.05, 0.15, drop_text, 
               transform=ax_lr.transAxes, fontsize=8.5, 
               bbox=dict(boxstyle='square,pad=0.2', facecolor='white', alpha=0.8))

    ax_dyn = axes[i, 1]
    ax_dyn.plot(ep, dh, color='#059669', marker='o', markersize=3.5, linewidth=1.8, label='DynH')
    ax_dyn.plot(ep, dc, color='#EA580C', marker='^', markersize=3.5, linewidth=1.8, label='DynC')
    ax_dyn.set_title(f"{s['short_name']}: DynH vs DynC", fontsize=10, fontweight='bold')
    ax_dyn.set_ylabel("Dynamics Loss", fontsize=9)
    ax_dyn.grid(True, linestyle='--', alpha=0.4)
    ax_dyn.legend(loc='upper right', fontsize=8)
    ax_dyn.ticklabel_format(style='sci', scilimits=(0,0), axis='y')

    ax_ratio = axes[i, 2]
    ratio = ((dh + dc) / l) * 100.0
    ax_ratio.plot(ep, ratio, color='#7C3AED', marker='o', markersize=3.5, linewidth=2.0)
    ax_ratio.set_title(f"{s['short_name']}: Dynamics Contribution %", fontsize=10, fontweight='bold')
    ax_ratio.set_ylabel("% of Loss", fontsize=9)
    ax_ratio.grid(True, linestyle='--', alpha=0.4)

for ax in axes[-1, :]:
    ax.set_xlabel("Epoch", fontsize=10, fontweight='semibold')

fig2.suptitle(f"Detailed Stage-by-Stage Metrics Breakdown ({num_stages} Stages, K Up to 512)", fontsize=15, fontweight='bold', y=0.992)

out_stages_workspace = "/home/tumi/Desktop/BadApple/losses_by_curriculum_stage.png"
plt.savefig(out_stages_workspace, bbox_inches='tight', dpi=250)
for ad in artifact_dirs:
    plt.savefig(os.path.join(ad, "losses_by_curriculum_stage.png"), bbox_inches='tight', dpi=250)
plt.close(fig2)

print("Generated all plots successfully:")
print(f"1. {out_dashboard_workspace}")
print(f"2. {out_stages_workspace}")
