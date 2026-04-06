#!/usr/bin/env python3
"""Orchestrator: run DreamShift + SDAR benchmarks and produce combined results.

This is the main entry point that coordinates data collection from both engines
and produces the consolidated throughput_comparison.json.

Usage:
  # Full run (requires DreamShift server running + GPU 1 free for JetEngine):
  python scripts/demo/bench_comparison.py

  # Just merge existing per-engine JSON files:
  python scripts/demo/bench_comparison.py --merge-only
"""

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path


RESULTS_DIR = Path(__file__).parent / "results"


def merge_results(dreamshift_path, sdar_path, output_path):
    """Merge per-engine JSON files into consolidated throughput_comparison.json."""
    with open(dreamshift_path) as f:
        ds = json.load(f)
    with open(sdar_path) as f:
        sdar = json.load(f)

    combined = {
        "dreamshift": ds["tps_series"],
        "sdar": sdar["tps_series"],
        "metadata": {
            "dreamshift": {k: v for k, v in ds.items() if k != "tps_series"},
            "sdar": {k: v for k, v in sdar.items() if k != "tps_series"},
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Combined results saved: {output_path}")
    return combined


def main():
    parser = argparse.ArgumentParser(description="DreamShift vs SDAR benchmark comparison")
    parser.add_argument("--merge-only", action="store_true",
                        help="Just merge existing JSON files without re-running benchmarks")
    parser.add_argument("--server-url", default="http://localhost:30000")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--duration", type=int, default=90)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ds_path = results_dir / "dreamshift_tps.json"
    sdar_path = results_dir / "sdar_tps.json"
    combined_path = results_dir / "throughput_comparison.json"

    if not args.merge_only:
        # Run DreamShift collection
        print("=" * 50)
        print("Running DreamShift TPS collection...")
        print("=" * 50)
        ds_cmd = [
            sys.executable, str(Path(__file__).parent / "collect_tps_timeseries.py"),
            "--server-url", args.server_url,
            "--concurrency", str(args.concurrency),
            "--max-tokens", str(args.max_tokens),
            "--duration", str(args.duration),
            "--output", str(ds_path),
        ]
        subprocess.run(ds_cmd, check=True)

        # Run SDAR/JetEngine collection (via torchrun on GPU 1)
        print()
        print("=" * 50)
        print("Running SDAR JetEngine TPS collection...")
        print("=" * 50)
        sdar_cmd = [
            "torchrun", "--nproc_per_node=1",
            str(Path(__file__).parent / "collect_jetengine_tps.py"),
            "--batch-size", str(args.batch_size),
            "--max-tokens", str(args.max_tokens),
            "--num-batches", str(args.num_batches),
            "--output", str(sdar_path),
        ]
        env = dict(__import__("os").environ)
        env["CUDA_VISIBLE_DEVICES"] = "1"
        subprocess.run(sdar_cmd, check=True, env=env)

    # Merge
    if not ds_path.exists() or not sdar_path.exists():
        print(f"ERROR: Missing data files. Need both {ds_path} and {sdar_path}")
        sys.exit(1)

    merge_results(ds_path, sdar_path, combined_path)

    # Print summary
    with open(combined_path) as f:
        data = json.load(f)
    ds_tps = [p["tps"] for p in data["dreamshift"] if p["tps"] > 0]
    sdar_tps = [p["tps"] for p in data["sdar"] if p["tps"] > 0]
    if ds_tps and sdar_tps:
        ds_avg = sum(ds_tps) / len(ds_tps)
        sdar_avg = sum(sdar_tps) / len(sdar_tps)
        print(f"\nSummary:")
        print(f"  DreamShift avg: {ds_avg:,.0f} tok/s")
        print(f"  SDAR avg:       {sdar_avg:,.0f} tok/s")
        print(f"  Speedup:        {ds_avg/sdar_avg:.2f}×")


if __name__ == "__main__":
    main()
