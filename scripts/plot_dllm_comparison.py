#!/usr/bin/env python3
"""Plot: Ours N=4 b2 vs LLaDA 2.1-mini vs SDAR — Total TPS vs Per-Req TPS."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT_DIR = "benchmark_plots"
os.makedirs(OUT_DIR, exist_ok=True)

bs_list = [1, 2, 4, 8, 16, 32, 64]

job_tps = {
    "MBPP": {
        "Ours N=4 b2":    [313, 555, 1115, 1457, 3344, 5559, 7069],
        "LLaDA 2.1-mini": [455, 705, 978, 1246, 1637, 1999, 2418],
        "SDAR":            [106, 196, 357, 622, 977, 1408, 1745],
    },
    "MATH-500": {
        "Ours N=4 b2":    [322, 526, 1123, 1806, 3359, 5667, 6798],
        "LLaDA 2.1-mini": [398, 653, 908, 1185, 1497, 1835, 2333],
        "SDAR":            [121, 210, 367, 625, 999, 1424, 1781],
    },
    "LMSYS-Chat": {
        "Ours N=4 b2":    [303, 504, 872, 1805, 3088, 4796, 6185],
        "LLaDA 2.1-mini": [341, 487, 679, 961, 1288, 1614, 1916],
        "SDAR":            [117, 200, 354, 616, 956, 1396, 1737],
    },
}

per_req_tps = {
    "MBPP": {
        "Ours N=4 b2":    [317, 293, 302, 233, 234, 196, 124],
        "LLaDA 2.1-mini": [464, 360, 249, 158, 104, 63, 38],
        "SDAR":            [108, 99, 90, 78, 62, 44, 27],
    },
    "MATH-500": {
        "Ours N=4 b2":    [326, 275, 305, 256, 237, 201, 125],
        "LLaDA 2.1-mini": [467, 342, 241, 155, 96, 59, 37],
        "SDAR":            [125, 107, 93, 80, 64, 45, 28],
    },
    "LMSYS-Chat": {
        "Ours N=4 b2":    [312, 289, 276, 287, 249, 197, 125],
        "LLaDA 2.1-mini": [356, 255, 176, 125, 83, 52, 31],
        "SDAR":            [122, 102, 91, 80, 62, 45, 28],
    },
}

COLORS = {
    "Ours N=4 b2":    "#1f77b4",
    "LLaDA 2.1-mini": "#2ca02c",
    "SDAR":            "#d62728",
}
MARKERS = {
    "Ours N=4 b2":    "o",
    "LLaDA 2.1-mini": "s",
    "SDAR":            "^",
}
methods = ["Ours N=4 b2", "LLaDA 2.1-mini", "SDAR"]
datasets = ["MBPP", "MATH-500", "LMSYS-Chat"]

# ── 1×3 combined ──
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

for ax, ds in zip(axes, datasets):
    for method in methods:
        x = per_req_tps[ds][method]
        y = job_tps[ds][method]
        ax.plot(x, y, color=COLORS[method], marker=MARKERS[method],
                linewidth=2.5, markersize=10, label=method, zorder=3)
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
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=13,
           bbox_to_anchor=(0.5, 1.05), frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{OUT_DIR}/dllm_comparison_all.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_DIR}/dllm_comparison_all.png")

# ── Individual plots ──
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
    ax.set_title(f"{ds} — DLLM Comparison", fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fname = f"{OUT_DIR}/dllm_cmp_{ds.lower().replace('-', '_').replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

print(f"\nAll plots in {OUT_DIR}/")
