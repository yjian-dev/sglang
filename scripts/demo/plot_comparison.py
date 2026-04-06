#!/usr/bin/env python3
"""Plot throughput comparison chart from collected data.

Wrapper entry point for live_tps_chart.py. Can also read from the consolidated
throughput_comparison.json format.

Usage:
  # From per-engine files (default):
  python scripts/demo/plot_comparison.py

  # From consolidated JSON:
  python scripts/demo/plot_comparison.py --from-combined scripts/demo/results/throughput_comparison.json

  # Custom output:
  python scripts/demo/plot_comparison.py --output scripts/demo/results/throughput_comparison.png
"""

import argparse
import json
import sys
from pathlib import Path

from live_tps_chart import load_data, render_static, render_animated


RESULTS_DIR = Path(__file__).parent / "results"


def load_combined(path):
    """Load from consolidated throughput_comparison.json format."""
    with open(path) as f:
        data = json.load(f)
    ds_data = {"tps_series": data["dreamshift"]}
    sdar_data = {"tps_series": data["sdar"]}
    if "metadata" in data:
        ds_data.update(data["metadata"].get("dreamshift", {}))
        sdar_data.update(data["metadata"].get("sdar", {}))
    return ds_data, sdar_data


def main():
    parser = argparse.ArgumentParser(description="Plot DreamShift vs SDAR throughput comparison")
    parser.add_argument("--from-combined", default="",
                        help="Path to consolidated throughput_comparison.json")
    parser.add_argument("--dreamshift-data", default=str(RESULTS_DIR / "dreamshift_tps.json"))
    parser.add_argument("--sdar-data", default=str(RESULTS_DIR / "sdar_tps.json"))
    parser.add_argument("--output", default=str(RESULTS_DIR / "throughput_comparison.png"))
    parser.add_argument("--output-gif", default="")
    parser.add_argument("--title", default="Throughput Comparison")
    parser.add_argument("--batch-label", default="Batch Size: 32")
    parser.add_argument("--max-time", type=float, default=90)
    parser.add_argument("--no-animate", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    if args.from_combined:
        ds_data, sdar_data = load_combined(args.from_combined)
    else:
        ds_data, sdar_data = load_data(args.dreamshift_data, args.sdar_data)

    # Static PNG
    render_static(ds_data, sdar_data, args.output,
                  title=args.title, batch_label=args.batch_label,
                  max_time=args.max_time)

    # Animated GIF
    if args.output_gif and not args.no_animate:
        render_animated(ds_data, sdar_data, args.output_gif,
                        title=args.title, batch_label=args.batch_label,
                        fps=args.fps, max_time=args.max_time)


if __name__ == "__main__":
    main()
