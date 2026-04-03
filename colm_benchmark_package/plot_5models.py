#!/usr/bin/env python3
"""Plot COLM benchmark: x=Per-Request TPS, y=Total TPS for 3 datasets."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = "benchmark_plots"
os.makedirs(OUT_DIR, exist_ok=True)

bs_list = [1, 2, 4, 8, 16, 32, 64]

# From PLAN.md / benchmark_results.md Table 6 & 7
job_tps = {
    "MBPP": {
        "AR":             [141, 267, 512, 908, 2007, 3545, 5037],
        "DFlash s1d16":   [313, 521, 1003, 1142, 2996, 4092, 4562],
        "EAGLE3":         [243, 444, 865, 1375, 3005, 5388, 7881],
        "Ours N=4 LoRA":  [290, 498, 992, 1334, 3012, 5049, 6340],
        "Ours N=4 b2":    [313, 555, 1115, 1457, 3344, 5559, 7069],
    },
    "MATH-500": {
        "AR":             [141, 263, 517, 903, 1918, 3520, 4597],
        "DFlash s1d16":   [361, 567, 1161, 1683, 3080, 4625, 4920],
        "EAGLE3":         [236, 417, 823, 1488, 2872, 5022, 6784],
        "Ours N=4 LoRA":  [306, 483, 1022, 1642, 3040, 5087, 6009],
        "Ours N=4 b2":    [322, 526, 1123, 1806, 3359, 5667, 6798],
    },
    "LMSYS-Chat": {
        "AR":             [141, 260, 484, 1077, 1993, 3580, 5779],
        "DFlash s1d16":   [284, 450, 623, 1574, 2614, 3630, 4133],
        "EAGLE3":         [207, 352, 521, 1135, 1890, 3275, 4818],
        "Ours N=4 LoRA":  [281, 466, 796, 1700, 2840, 4335, 5692],
        "Ours N=4 b2":    [303, 504, 872, 1805, 3088, 4796, 6185],
    },
}

per_req_tps = {
    "MBPP": {
        "AR":             [141, 134, 129, 125, 126, 112, 94],
        "DFlash s1d16":   [324, 292, 292, 224, 236, 156, 84],
        "EAGLE3":         [245, 229, 232, 205, 207, 188, 137],
        "Ours N=4 LoRA":  [295, 265, 273, 218, 210, 177, 110],
        "Ours N=4 b2":    [317, 293, 302, 233, 234, 196, 124],
    },
    "MATH-500": {
        "AR":             [142, 132, 130, 119, 123, 111, 93],
        "DFlash s1d16":   [371, 314, 344, 272, 247, 176, 96],
        "EAGLE3":         [238, 215, 220, 205, 199, 176, 132],
        "Ours N=4 LoRA":  [310, 256, 283, 235, 214, 182, 111],
        "Ours N=4 b2":    [326, 275, 305, 256, 237, 201, 125],
    },
    "LMSYS-Chat": {
        "AR":             [141, 131, 130, 135, 125, 113, 92],
        "DFlash s1d16":   [312, 287, 268, 331, 280, 181, 101],
        "EAGLE3":         [221, 212, 193, 220, 204, 184, 135],
        "Ours N=4 LoRA":  [291, 261, 249, 268, 228, 176, 111],
        "Ours N=4 b2":    [312, 289, 276, 287, 249, 197, 125],
    },
}

COLORS = {
    "AR":             "#888888",
    "DFlash s1d16":   "#e377c2",
    "EAGLE3":         "#ff7f0e",
    "Ours N=4 LoRA":  "#9467bd",
    "Ours N=4 b2":    "#1f77b4",
}
MARKERS = {
    "AR":             "s",
    "DFlash s1d16":   "D",
    "EAGLE3":         "^",
    "Ours N=4 LoRA":  "X",
    "Ours N=4 b2":    "o",
}
methods = ["AR", "DFlash s1d16", "EAGLE3", "Ours N=4 LoRA", "Ours N=4 b2"]
datasets = ["MBPP", "MATH-500", "LMSYS-Chat"]

# ── Combined 1×3 figure ──
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

for ax, ds in zip(axes, datasets):
    for method in methods:
        x = per_req_tps[ds][method]
        y = job_tps[ds][method]
        ax.plot(x, y, color=COLORS[method], marker=MARKERS[method],
                linewidth=2.5, markersize=10, label=method, zorder=3)
        # Annotate bs at select points
        for i, bs in enumerate(bs_list):
            if bs in [1, 4, 16, 64]:
                ax.annotate(str(bs), (x[i], y[i]),
                            textcoords="offset points", xytext=(7, -10),
                            fontsize=8, color=COLORS[method], alpha=0.6)

    ax.set_xlabel("Per-Request TPS (tok/s)", fontsize=13)
    ax.set_ylabel("Total Throughput (tok/s)", fontsize=13)
    ax.set_title(ds, fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.25)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=12,
           bbox_to_anchor=(0.5, 1.05), frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{OUT_DIR}/colm_tvl_all.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_DIR}/colm_tvl_all.png")

# ── Individual dataset plots ──
for ds in datasets:
    fig, ax = plt.subplots(figsize=(10, 7))
    for method in methods:
        x = per_req_tps[ds][method]
        y = job_tps[ds][method]
        ax.plot(x, y, color=COLORS[method], marker=MARKERS[method],
                linewidth=2.5, markersize=10, label=method, zorder=3)
        for i, bs in enumerate(bs_list):
            ax.annotate(str(bs), (x[i], y[i]),
                        textcoords="offset points", xytext=(7, -10),
                        fontsize=8, color=COLORS[method], alpha=0.6)

    ax.set_xlabel("Per-Request TPS (tok/s)", fontsize=14)
    ax.set_ylabel("Total Throughput (tok/s)", fontsize=14)
    ax.set_title(f"{ds} — Total vs Per-Request TPS", fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fname = f"{OUT_DIR}/colm_tvl_{ds.lower().replace('-', '_').replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

# ── Speedup over AR (Job TPS) ──
fig, axes = plt.subplots(1, 3, figsize=(24, 7))
for ax, ds in zip(axes, datasets):
    x = np.arange(len(bs_list))
    ar_vals = np.array(job_tps[ds]["AR"])
    for method in methods:
        if method == "AR":
            continue
        vals = np.array(job_tps[ds][method])
        speedup = vals / ar_vals
        ax.plot(x, speedup, color=COLORS[method], marker=MARKERS[method],
                linewidth=2.5, markersize=9, label=method, zorder=3)

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bs_list])
    ax.set_xlabel("Batch Size", fontsize=13)
    ax.set_ylabel("Speedup over AR", fontsize=13)
    ax.set_title(ds, fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0.5, 3.0)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=12,
           bbox_to_anchor=(0.5, 1.05), frameon=True)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{OUT_DIR}/colm_speedup_over_ar.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_DIR}/colm_speedup_over_ar.png")

print(f"\nAll plots in {OUT_DIR}/")
