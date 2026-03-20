import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

batch_sizes = [32, 64]

# Job-level tok/s on H200 (1 GPU, ShareGPT-style prompts)
ar_job      = [5650, 8845]
dflash_job  = [5720, 6966]
ours_job    = [6590, 9014]

labels = ['Baseline\n(AR)', 'DFlash', 'Ours\n(N=3)']
colors = ['#4C72B0', '#55A868', '#C44E52']

x = np.arange(len(batch_sizes))
width = 0.25
offsets = np.array([-1, 0, 1]) * width

fig, ax = plt.subplots(figsize=(8, 6))

for i, (data, label, color, offset) in enumerate(zip(
    [ar_job, dflash_job, ours_job], labels, colors, offsets
)):
    bars = ax.bar(x + offset, data, width, label=label, color=color, alpha=0.9,
                  edgecolor='white', linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 100,
                f'{h:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_title('Job-Level Throughput on B200\n(1 GPU, AIME prompts)', fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('Batch Size (Concurrency)', fontsize=11)
ax.set_ylabel('Throughput (tok/s)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels([str(b) for b in batch_sizes], fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max(ours_job) * 1.18)

# Add ratio annotations
for i, bs in enumerate(batch_sizes):
    ar = ar_job[i]
    ours = ours_job[i]
    ratio = (ours - ar) / ar * 100
    ax.annotate(f'N=3: +{ratio:.0f}%\nvs AR',
                xy=(x[i] + offsets[2], ours),
                xytext=(x[i] + offsets[2] + 0.02, ours + max(ours_job) * 0.07),
                fontsize=8.5, color='#C44E52', fontweight='bold', ha='center')

plt.tight_layout()
out = 'bench_results/throughput_h200.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
