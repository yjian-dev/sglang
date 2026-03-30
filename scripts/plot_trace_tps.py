#!/usr/bin/env python3
"""Plot TPS time series from multiple trace replay results."""
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="JSON result files")
    parser.add_argument("--output", default="trace_tps_comparison.png")
    parser.add_argument("--title", default="Throughput Over Time (Azure Trace, peak QPS=24)")
    args = parser.parse_args()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, path in enumerate(args.inputs):
        with open(path) as f:
            data = json.load(f)
        label = data.get("label", path)
        tps = data["tps_series"]
        active = data["active_series"]

        # Compute per-req TPS in time buckets from raw results if available
        # Otherwise use job_tps
        per_req_tps = data.get("per_req_tps_series", tps)
        color = colors[i % len(colors)]

        times = [p["time"] for p in per_req_tps]
        vals = [p.get("per_req_tps", p.get("tps", 0)) for p in per_req_tps]

        # Smooth with rolling average (window=3)
        if len(vals) > 3:
            smoothed = []
            for j in range(len(vals)):
                start = max(0, j - 1)
                end = min(len(vals), j + 2)
                smoothed.append(sum(vals[start:end]) / (end - start))
            vals = smoothed

        # Compute avg per-req TPS from summary
        summary = data.get("summary", {})
        avg_label = f"avg {summary.get('per_req_tps_mean', summary.get('job_tps', 0)):.0f}"

        ax1.plot(times, vals, label=f"{label} ({avg_label})", color=color, linewidth=1.8)

        act_times = [p["time"] for p in active]
        act_vals = [p["active"] for p in active]
        ax2.plot(act_times, act_vals, label=label, color=color, linewidth=1.2, alpha=0.7)

    ax1.set_ylabel("Per-Request TPS (tok/s)", fontsize=12)
    ax1.set_title(args.title, fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Active Requests", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
