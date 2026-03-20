import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Per-request tok/s on B200
bs_per = [1, 8, 32, 64]
ar_per      = [224, 205, 180, 143]
dflash_per  = [506, 408, 223, 133]
ours_per    = [424, 354, 228, 155]

# Job-level tok/s on B200
bs_job = [32, 64]
ar_job      = [5650, 8845]
dflash_job  = [5720, 6966]
ours_job    = [6590, 9014]

labels = ['Baseline\n(AR)', 'DFlash', 'Ours\n(N=3)']
colors = ['#4C72B0', '#55A868', '#C44E52']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left: Per-request ---
ax = axes[0]
x = np.arange(len(bs_per))
width = 0.25
offsets = np.array([-1, 0, 1]) * width

for data, label, color, offset in zip([ar_per, dflash_per, ours_per], labels, colors, offsets):
    bars = ax.bar(x + offset, data, width, label=label, color=color, alpha=0.9,
                  edgecolor='white', linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 5,
                f'{h}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax.set_title('Per-Request Throughput (tok/s)', fontsize=12, fontweight='bold', pad=8)
ax.set_xlabel('Batch Size (Concurrency)', fontsize=11)
ax.set_ylabel('Throughput (tok/s)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels([str(b) for b in bs_per], fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max(dflash_per) * 1.22)

# --- Right: Job-level ---
ax = axes[1]
x = np.arange(len(bs_job))
width = 0.25
offsets = np.array([-1, 0, 1]) * width

for data, label, color, offset in zip([ar_job, dflash_job, ours_job], labels, colors, offsets):
    bars = ax.bar(x + offset, data, width, label=label, color=color, alpha=0.9,
                  edgecolor='white', linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 80,
                f'{h:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_title('Job-Level Throughput (tok/s)', fontsize=12, fontweight='bold', pad=8)
ax.set_xlabel('Batch Size (Concurrency)', fontsize=11)
ax.set_ylabel('Throughput (tok/s)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels([str(b) for b in bs_job], fontsize=12)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max(ours_job) * 1.18)

# Add ratio annotations on job chart
for i, bs in enumerate(bs_job):
    ratio = (ours_job[i] - ar_job[i]) / ar_job[i] * 100
    axes[1].annotate(f'+{ratio:.0f}%\nvs AR',
                xy=(x[i] + offsets[2], ours_job[i]),
                xytext=(x[i] + offsets[2], ours_job[i] + max(ours_job) * 0.08),
                fontsize=8.5, color='#C44E52', fontweight='bold', ha='center')

fig.suptitle('Throughput Comparison on B200\n(1 GPU, AIME prompts, OSL=2048)',
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
out = 'bench_results/throughput_b200.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
