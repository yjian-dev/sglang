#!/usr/bin/env python3
"""Render TPS traffic monitor chart — dark theme area chart comparing DreamShift vs SDAR.

Produces both static PNG and animated MP4/GIF.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.ticker import FuncFormatter


# Style constants
BG_COLOR = "#0a0e27"
GRID_COLOR = "#1a2040"
TEXT_COLOR = "#e0e8ff"
DREAMSHIFT_COLOR = "#00d4ff"
DREAMSHIFT_FILL = "#00d4ff"
SDAR_COLOR = "#ff4444"
SDAR_FILL = "#ff4444"
FONT_FAMILY = "monospace"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.linestyle": ":",
        "grid.alpha": 0.4,
        "font.family": FONT_FAMILY,
        "font.size": 12,
    })


def load_data(dreamshift_path, sdar_path):
    with open(dreamshift_path) as f:
        ds_data = json.load(f)
    with open(sdar_path) as f:
        sdar_data = json.load(f)
    return ds_data, sdar_data


def smooth(values, window=5):
    """Simple moving average smoothing."""
    if len(values) <= window:
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        result.append(sum(values[start:end]) / (end - start))
    return result


def render_static(ds_data, sdar_data, output_path, title="Throughput Comparison",
                  batch_label="Batch Size: 32", max_time=None):
    setup_style()

    fig, ax = plt.subplots(figsize=(14, 7))

    # Extract data, optionally trimming to max_time
    ds_series = ds_data["tps_series"]
    sdar_series = sdar_data["tps_series"]
    if max_time is not None:
        ds_series = [p for p in ds_series if p["time"] <= max_time]
        sdar_series = [p for p in sdar_series if p["time"] <= max_time]

    ds_times = [p["time"] for p in ds_series]
    ds_tps = [p["tps"] for p in ds_series]
    sdar_times = [p["time"] for p in sdar_series]
    sdar_tps = [p["tps"] for p in sdar_series]

    # Smooth
    ds_tps_smooth = smooth(ds_tps, window=7)
    sdar_tps_smooth = smooth(sdar_tps, window=7)

    # Time axis range
    max_time = max(max(ds_times) if ds_times else 0, max(sdar_times) if sdar_times else 0)

    # Plot area fills
    ax.fill_between(ds_times, ds_tps_smooth, alpha=0.25, color=DREAMSHIFT_FILL, linewidth=0)
    ax.fill_between(sdar_times, sdar_tps_smooth, alpha=0.15, color=SDAR_FILL, linewidth=0)

    # Plot lines
    ax.plot(ds_times, ds_tps_smooth, color=DREAMSHIFT_COLOR, linewidth=2.5,
            label="DreamShift (Ours)", zorder=5)
    ax.plot(sdar_times, sdar_tps_smooth, color=SDAR_COLOR, linewidth=2.0,
            label="SDAR Baseline", zorder=4, linestyle="-")

    # Compute averages (skip warmup: first 10%)
    ds_warmup = max(1, len(ds_tps) // 10)
    sdar_warmup = max(1, len(sdar_tps) // 10)
    ds_avg = np.mean(ds_tps[ds_warmup:]) if len(ds_tps) > ds_warmup else 0
    sdar_avg = np.mean(sdar_tps[sdar_warmup:]) if len(sdar_tps) > sdar_warmup else 0

    # Average lines
    if ds_avg > 0:
        ax.axhline(y=ds_avg, color=DREAMSHIFT_COLOR, linestyle="--", alpha=0.5, linewidth=1)
        ax.text(max_time * 0.98, ds_avg, f"avg {ds_avg:,.0f}", color=DREAMSHIFT_COLOR,
                fontsize=10, ha="right", va="bottom", alpha=0.8)
    if sdar_avg > 0:
        ax.axhline(y=sdar_avg, color=SDAR_COLOR, linestyle="--", alpha=0.5, linewidth=1)
        ax.text(max_time * 0.98, sdar_avg, f"avg {sdar_avg:,.0f}", color=SDAR_COLOR,
                fontsize=10, ha="right", va="bottom", alpha=0.8)

    # Labels
    ax.set_xlabel("Time (seconds)", fontsize=13, labelpad=10)
    ax.set_ylabel("Tokens / Second", fontsize=13, labelpad=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, None)

    # Legend
    legend = ax.legend(loc="upper left", fontsize=12, frameon=True,
                       facecolor="#111833", edgecolor=GRID_COLOR, framealpha=0.9)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    # Title and annotations
    ax.set_title(title, fontsize=20, fontweight="bold", pad=20, color=TEXT_COLOR)

    # Big throughput overlay
    speedup = ds_avg / sdar_avg if sdar_avg > 0 else 0
    info_text = f"{batch_label}\n"
    info_text += f"DreamShift: {ds_avg:,.0f} tok/s\n"
    info_text += f"SDAR:       {sdar_avg:,.0f} tok/s\n"
    if speedup > 0:
        info_text += f"Speedup:    {speedup:.2f}×"

    ax.text(0.98, 0.95, info_text, transform=ax.transAxes,
            fontsize=14, fontweight="bold", color=TEXT_COLOR,
            verticalalignment="top", horizontalalignment="right",
            fontfamily=FONT_FAMILY,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#111833",
                      edgecolor=DREAMSHIFT_COLOR, alpha=0.9, linewidth=1.5))

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close()
    print(f"Static chart saved: {output_path}")


def render_animated(ds_data, sdar_data, output_path, title="Throughput Comparison",
                    batch_label="Batch Size: 32", fps=30, hold_frames=90, max_time=None):
    """Render animated chart that builds up over time."""
    setup_style()

    fig, ax = plt.subplots(figsize=(14, 7))

    ds_series = ds_data["tps_series"]
    sdar_series = sdar_data["tps_series"]
    if max_time is not None:
        ds_series = [p for p in ds_series if p["time"] <= max_time]
        sdar_series = [p for p in sdar_series if p["time"] <= max_time]

    ds_times = [p["time"] for p in ds_series]
    ds_tps = [p["tps"] for p in ds_series]
    sdar_times = [p["time"] for p in sdar_series]
    sdar_tps = [p["tps"] for p in sdar_series]

    ds_tps_smooth = smooth(ds_tps, window=7)
    sdar_tps_smooth = smooth(sdar_tps, window=7)

    max_time = max(max(ds_times, default=0), max(sdar_times, default=0))
    max_tps = max(max(ds_tps, default=0), max(sdar_tps, default=0)) * 1.15

    # Number of animation frames (one per data point, roughly)
    n_frames = max(len(ds_times), len(sdar_times))
    total_frames = n_frames + hold_frames

    ds_line, = ax.plot([], [], color=DREAMSHIFT_COLOR, linewidth=2.5,
                       label="DreamShift (Ours)", zorder=5)
    sdar_line, = ax.plot([], [], color=SDAR_COLOR, linewidth=2.0,
                         label="SDAR Baseline", zorder=4)
    ds_fill = None
    sdar_fill = None

    ax.set_xlim(0, max_time)
    ax.set_ylim(0, max_tps)
    ax.set_xlabel("Time (seconds)", fontsize=13, labelpad=10)
    ax.set_ylabel("Tokens / Second", fontsize=13, labelpad=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=20, fontweight="bold", pad=20, color=TEXT_COLOR)
    legend = ax.legend(loc="upper left", fontsize=12, frameon=True,
                       facecolor="#111833", edgecolor=GRID_COLOR, framealpha=0.9)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    info_box = ax.text(0.98, 0.95, "", transform=ax.transAxes,
                       fontsize=14, fontweight="bold", color=TEXT_COLOR,
                       verticalalignment="top", horizontalalignment="right",
                       fontfamily=FONT_FAMILY,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="#111833",
                                 edgecolor=DREAMSHIFT_COLOR, alpha=0.9, linewidth=1.5))

    def update(frame):
        nonlocal ds_fill, sdar_fill
        idx = min(frame, n_frames - 1)

        # DreamShift
        ds_n = min(idx + 1, len(ds_times))
        if ds_n > 0:
            ds_line.set_data(ds_times[:ds_n], ds_tps_smooth[:ds_n])
            if ds_fill:
                ds_fill.remove()
            ds_fill = ax.fill_between(ds_times[:ds_n], ds_tps_smooth[:ds_n],
                                      alpha=0.25, color=DREAMSHIFT_FILL, linewidth=0)

        # SDAR
        sdar_n = min(idx + 1, len(sdar_times))
        if sdar_n > 0:
            sdar_line.set_data(sdar_times[:sdar_n], sdar_tps_smooth[:sdar_n])
            if sdar_fill:
                sdar_fill.remove()
            sdar_fill = ax.fill_between(sdar_times[:sdar_n], sdar_tps_smooth[:sdar_n],
                                        alpha=0.15, color=SDAR_FILL, linewidth=0)

        # Update info box
        ds_current = ds_tps_smooth[ds_n - 1] if ds_n > 0 else 0
        sdar_current = sdar_tps_smooth[sdar_n - 1] if sdar_n > 0 else 0
        info = f"{batch_label}\n"
        info += f"DreamShift: {ds_current:,.0f} tok/s\n"
        info += f"SDAR:       {sdar_current:,.0f} tok/s"
        if sdar_current > 0:
            info += f"\nSpeedup:    {ds_current/sdar_current:.2f}×"
        info_box.set_text(info)

        return ds_line, sdar_line, info_box

    anim = animation.FuncAnimation(fig, update, frames=total_frames,
                                   interval=1000/fps, blit=False)

    # Save as mp4 or gif
    fig.set_facecolor(BG_COLOR)
    suffix = Path(output_path).suffix.lower()
    if suffix == ".mp4":
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(output_path, writer=writer)
    elif suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
    else:
        anim.save(output_path, fps=fps)

    plt.close()
    print(f"Animated chart saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dreamshift-data", default="scripts/demo/results/dreamshift_tps.json")
    parser.add_argument("--sdar-data", default="scripts/demo/results/sdar_tps.json")
    parser.add_argument("--output-png", default="scripts/demo/results/tps_comparison.png")
    parser.add_argument("--output-mp4", default="")
    parser.add_argument("--output-gif", default="scripts/demo/results/tps_comparison.gif")
    parser.add_argument("--title", default="Throughput Comparison")
    parser.add_argument("--batch-label", default="Batch Size: 32")
    parser.add_argument("--max-time", type=float, default=None, help="Crop data at this time (seconds)")
    parser.add_argument("--no-animate", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    ds_data, sdar_data = load_data(args.dreamshift_data, args.sdar_data)

    # Render static PNG
    if args.output_png:
        render_static(ds_data, sdar_data, args.output_png,
                      title=args.title, batch_label=args.batch_label,
                      max_time=args.max_time)

    # Optionally render animated
    if not args.no_animate:
        if args.output_mp4:
            render_animated(ds_data, sdar_data, args.output_mp4,
                            title=args.title, batch_label=args.batch_label,
                            fps=args.fps, max_time=args.max_time)
        if args.output_gif:
            render_animated(ds_data, sdar_data, args.output_gif,
                            title=args.title, batch_label=args.batch_label,
                            fps=args.fps, max_time=args.max_time)


if __name__ == "__main__":
    main()
