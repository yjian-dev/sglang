#!/usr/bin/env python3
"""Plot ultrachat: x=Per-Request TPS, y=Total TPS, each point=batch size."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = "benchmark_plots"
os.makedirs(OUT_DIR, exist_ok=True)

bs_list = [1, 2, 4, 8, 16, 32, 64]

job_tps = {
    "AR":             [269, 483, 883, 1595, 2702, 4003, 5073],
    "DFlash s1d16":   [260, 474, 846, 1489, 2429, 3313, 3702],
    "EAGLE3":         [248, 456, 865, 1576, 2837, 4997, 7349],
    "Ours N=4 LoRA":  [268, 486, 896, 1637, 2638, 4016, 5016],
    "Ours N=4 b3":    [300, 558, 1007, 1730, 2945, 4591, 5756],
    "Ours N=4 b2":    [283, 530, 950, 1717, 2847, 4477, 5606],
}

per_req_tps = {
    "AR":             [274, 262, 255, 233, 199, 155, 94],
    "DFlash s1d16":   [275, 270, 262, 251, 197, 134, 74],
    "EAGLE3":         [250, 240, 235, 223, 204, 182, 135],
    "Ours N=4 LoRA":  [274, 264, 251, 240, 198, 153, 94],
    "Ours N=4 b3":    [306, 301, 287, 268, 224, 177, 110],
    "Ours N=4 b2":    [290, 290, 273, 255, 214, 171, 105],
}

COLORS = {
    "AR":             "#888888",
    "DFlash s1d16":   "#e377c2",
    "EAGLE3":         "#ff7f0e",
    "Ours N=4 LoRA":  "#9467bd",
    "Ours N=4 b3":    "#1f77b4",
    "Ours N=4 b2":    "#2ca02c",
}
MARKERS = {
    "AR":             "s",
    "DFlash s1d16":   "D",
    "EAGLE3":         "^",
    "Ours N=4 LoRA":  "X",
    "Ours N=4 b3":    "P",
    "Ours N=4 b2":    "o",
}

methods = ["AR", "DFlash s1d16", "EAGLE3", "Ours N=4 LoRA", "Ours N=4 b3", "Ours N=4 b2"]

fig, ax = plt.subplots(figsize=(11, 8))

for method in methods:
    x = per_req_tps[method]
    y = job_tps[method]
    ax.plot(x, y, color=COLORS[method], marker=MARKERS[method],
            linewidth=2.5, markersize=10, label=method, zorder=3)
    for i, bs in enumerate(bs_list):
        ax.annotate(str(bs), (x[i], y[i]),
                    textcoords="offset points", xytext=(7, -10),
                    fontsize=8, color=COLORS[method], alpha=0.65)

ax.set_xlabel("Per-Request TPS (tok/s)", fontsize=14)
ax.set_ylabel("Total Throughput (tok/s)", fontsize=14)
ax.set_title("Ultrachat — Throughput vs Latency Frontier\n(1×H100, bf16, burst, max_tokens=2048)", fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/tvl_ultrachat.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_DIR}/tvl_ultrachat.png")
