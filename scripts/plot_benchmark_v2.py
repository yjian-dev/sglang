#!/usr/bin/env python3
"""Plot: x=Per-Request TPS, y=Total TPS, each point=batch size, each curve=model."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = "benchmark_plots"
os.makedirs(OUT_DIR, exist_ok=True)

bs_list = [1, 2, 4, 8, 16, 32, 64]

# Job-level TPS (y-axis)
job_tps = {
    "AIME 2022-2024": {
        "AR":             [148, 286, 565, 1088, 2025, 3623, 5851],
        "DFlash s1d16":   [325, 580, 1079, 1975, 3077, 4128, 4665],
        "DFlash s1d4":    [257, 463, 860, 1703, 3077, 5143, 7500],
        "EAGLE3":         [224, 409, 768, 1471, 2576, 4539, 6468],
        "Ours N=4 b2":    [293, 572, 1057, 1947, 3251, 5121, 6548],
        "Ours N=4 b3":    [302, 551, 1043, 1945, 3194, 4834, 5828],
        "Ours N=4 LoRA":  [270, 499, 917, 1791, 2901, 4431, 5346],
    },
    "MATH-500": {
        "AR":             [148, 286, 565, 1091, 2029, 3623, 5851],
        "DFlash s1d16":   [342, 616, 1123, 2053, 3378, 4601, 5074],
        "DFlash s1d4":    [253, 485, 910, 1765, 3218, 5558, 8437],
        "EAGLE3":         [229, 426, 801, 1585, 2780, 4924, 7200],
        "Ours N=4 b2":    [300, 608, 1109, 2069, 3533, 5572, 7178],
        "Ours N=4 b3":    [313, 630, 1134, 2122, 3603, 5635, 7213],
        "Ours N=4 LoRA":  [289, 569, 1019, 1928, 3219, 5016, 6329],
    },
    "ShareGPT": {
        "AR":             [148, 286, 565, 1084, 2025, 3617, 4693],
        "DFlash s1d16":   [346, 563, 913, 1467, 2351, 3465, 4249],
        "DFlash s1d4":    [240, 449, 796, 1336, 2441, 4299, 6493],
        "EAGLE3":         [236, 393, 681, 1080, 1747, 2908, 4842],
        "Ours N=4 b2":    [281, 531, 977, 1695, 2887, 3606, 4122],
        "Ours N=4 b3":    [309, 574, 975, 1803, 3197, 5076, 6306],
        "Ours N=4 LoRA":  [268, 498, 870, 1555, 2696, 3757, 4692],
    },
    "Synthetic": {
        "AR":             [137, 285, 562, 1071, 1983, 3449, 5425],
        "DFlash s1d16":   [245, 380, 711, 1321, 2155, 2863, 3112],
        "DFlash s1d4":    [195, 341, 646, 1218, 2197, 3899, 5554],
        "EAGLE3":         [211, 375, 716, 1372, 2406, 4192, 6307],
        "Ours N=4 b2":    [273, 516, 900, 1622, 2890, 4533, 6059],
        "Ours N=4 b3":    [285, 491, 1061, 1687, 2976, 4637, 5924],
        "Ours N=4 LoRA":  [264, 529, 855, 1543, 2612, 4057, 5359],
    },
}

# Per-request TPS (x-axis)
per_req_tps = {
    "AIME 2022-2024": {
        "AR":             [149, 145, 143, 137, 128, 114, 93],
        "DFlash s1d16":   [341, 337, 319, 304, 249, 166, 91],
        "DFlash s1d4":    [264, 257, 243, 239, 222, 196, 147],
        "EAGLE3":         [228, 218, 208, 203, 184, 164, 121],
        "Ours N=4 b2":    [298, 303, 291, 273, 233, 185, 118],
        "Ours N=4 b3":    [314, 318, 296, 276, 237, 189, 118],
        "Ours N=4 LoRA":  [281, 287, 265, 251, 214, 167, 104],
    },
    "MATH-500": {
        "AR":             [149, 145, 143, 137, 128, 114, 93],
        "DFlash s1d16":   [352, 347, 339, 318, 270, 176, 94],
        "DFlash s1d4":    [257, 258, 252, 246, 230, 200, 153],
        "EAGLE3":         [232, 225, 219, 216, 197, 173, 127],
        "Ours N=4 b2":    [305, 328, 309, 290, 251, 198, 126],
        "Ours N=4 b3":    [319, 336, 317, 298, 258, 203, 127],
        "Ours N=4 LoRA":  [294, 308, 285, 270, 231, 180, 112],
    },
    "ShareGPT": {
        "AR":             [149, 145, 143, 138, 128, 115, 93],
        "DFlash s1d16":   [362, 321, 323, 302, 258, 169, 95],
        "DFlash s1d4":    [247, 251, 232, 226, 215, 189, 146],
        "EAGLE3":         [248, 234, 228, 216, 200, 175, 132],
        "Ours N=4 b2":    [291, 288, 288, 260, 232, 181, 137],
        "Ours N=4 b3":    [319, 331, 297, 285, 253, 195, 127],
        "Ours N=4 LoRA":  [269, 278, 257, 246, 205, 146, 92],
    },
    "Synthetic": {
        "AR":             [148, 144, 142, 136, 126, 110, 87],
        "DFlash s1d16":   [341, 230, 249, 258, 181, 137, 75],
        "DFlash s1d4":    [202, 197, 173, 177, 167, 153, 110],
        "EAGLE3":         [211, 209, 210, 200, 176, 157, 119],
        "Ours N=4 b2":    [298, 361, 302, 277, 258, 200, 127],
        "Ours N=4 b3":    [300, 324, 343, 305, 267, 202, 129],
        "Ours N=4 LoRA":  [280, 258, 277, 262, 233, 182, 112],
    },
}

# Styling
COLORS = {
    "AR":             "#888888",
    "DFlash s1d16":   "#e377c2",
    "DFlash s1d4":    "#d62728",
    "EAGLE3":         "#ff7f0e",
    "Ours N=4 b2":    "#2ca02c",
    "Ours N=4 b3":    "#1f77b4",
    "Ours N=4 LoRA":  "#9467bd",
}
MARKERS = {
    "AR":             "s",
    "DFlash s1d16":   "D",
    "DFlash s1d4":    "d",
    "EAGLE3":         "^",
    "Ours N=4 b2":    "o",
    "Ours N=4 b3":    "P",
    "Ours N=4 LoRA":  "X",
}

methods = ["AR", "DFlash s1d16", "DFlash s1d4", "EAGLE3", "Ours N=4 b2", "Ours N=4 b3", "Ours N=4 LoRA"]

# ── Combined 2x2: x=per-req, y=total ──
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
datasets = ["AIME 2022-2024", "MATH-500", "ShareGPT", "Synthetic"]

for ax, ds in zip(axes.flat, datasets):
    for method in methods:
        x = per_req_tps[ds][method]
        y = job_tps[ds][method]
        ax.plot(x, y, color=COLORS[method], marker=MARKERS[method],
                linewidth=2.2, markersize=9, label=method, zorder=3)
        # Annotate bs at first and last point
        ax.annotate(f'bs={bs_list[0]}', (x[0], y[0]),
                    textcoords="offset points", xytext=(8, -12),
                    fontsize=7, color=COLORS[method], alpha=0.7)
        ax.annotate(f'bs={bs_list[-1]}', (x[-1], y[-1]),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=7, color=COLORS[method], alpha=0.7)

    ax.set_xlabel("Per-Request TPS (tok/s)", fontsize=12)
    ax.set_ylabel("Total Throughput (tok/s)", fontsize=12)
    ax.set_title(ds, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.25)
    # x from small to large (left=low per-req, right=high per-req)

    # Upper-right is ideal (high per-req + high total)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=12,
           bbox_to_anchor=(0.5, 1.02), frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{OUT_DIR}/throughput_vs_latency_all.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_DIR}/throughput_vs_latency_all.png")

# ── Individual dataset plots ──
for ds in datasets:
    fig, ax = plt.subplots(figsize=(10, 7))
    for method in methods:
        x = per_req_tps[ds][method]
        y = job_tps[ds][method]
        ax.plot(x, y, color=COLORS[method], marker=MARKERS[method],
                linewidth=2.5, markersize=10, label=method, zorder=3)
        # Annotate each point with bs
        for i, bs in enumerate(bs_list):
            ax.annotate(str(bs), (x[i], y[i]),
                        textcoords="offset points", xytext=(6, -10),
                        fontsize=7.5, color=COLORS[method], alpha=0.6)

    ax.set_xlabel("Per-Request TPS (tok/s)", fontsize=13)
    ax.set_ylabel("Total Throughput (tok/s)", fontsize=13)
    ax.set_title(f"{ds} — Throughput vs Latency Frontier", fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.25)
    # x from small to large

    plt.tight_layout()
    fname = f"{OUT_DIR}/tvl_{ds.lower().replace(' ', '_').replace('-', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

print(f"\nAll plots in {OUT_DIR}/")
