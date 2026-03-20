import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

batch_sizes = [32, 64]

# Per-request tok/s (1 GPU, AIME OSL=2048)
ar_per      = [114,  92]
eagle3_per  = [142,  90]
dflash_per  = [162,  89]
ours_per    = [189, 132]

# Job-level tok/s
ar_job      = [3640, 5874]
eagle3_job  = [3985, 5227]
dflash_job  = [4222, 4822]
ours_job    = [5379, 7383]

labels = ['Baseline\n(AR)', 'EAGLE3', 'DFlash', 'Ours\n(N=3)']
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

x = np.arange(len(batch_sizes))
n_groups = 4
width = 0.2
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, (title, datasets) in zip(axes, [
    ('Per-Request Throughput (tok/s)', [ar_per, eagle3_per, dflash_per, ours_per]),
    ('Job-Level Throughput (tok/s)',    [ar_job, eagle3_job, dflash_job, ours_job]),
]):
    for i, (data, label, color, offset) in enumerate(zip(datasets, labels, colors, offsets)):
        bars = ax.bar(x + offset, data, width, label=label, color=color, alpha=0.9, edgecolor='white', linewidth=0.5)
        # Add value labels on top of bars
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + max(datasets[0]) * 0.01,
                    f'{h:,.0f}' if h >= 1000 else f'{h:.0f}',
                    ha='center', va='bottom', fontsize=6.5, rotation=90)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Batch Size (Concurrency)', fontsize=11)
    ax.set_ylabel('Throughput (tok/s)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Throughput Comparison: AR vs EAGLE3 vs DFlash vs Ours (N=3)\n1 GPU H100, AIME prompts, OSL=2048',
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
out = 'bench_results/throughput_comparison.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
