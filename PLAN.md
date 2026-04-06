# Plan — DreamShift vs SDAR Demo

## Status
Phase 1-3 complete. Production data collected and charts generated. Phase 4 polish in progress.

## Tasks
- [x] Phase 1: Setup & Validation
  - [x] Start DreamShift N=4 SGLang server on GPU 0 port 30000
  - [x] Verify server health and test with 1 MBPP problem (correct output)
  - [x] Write JetEngine worker script and test with 1 MBPP problem via torchrun on GPU 1
  - [x] Verify both produce correct, fluent code output
  - [x] Confirm fairness: same prompts, same sampling params, same TP=1/bfloat16

- [x] Phase 2: Data Collection
  - [x] Write collect_tps_timeseries.py — counts streaming tokens per second from DreamShift
  - [x] Write collect_jetengine_tps.py — runs SDAR via JetEngine, records per-batch TPS intervals
  - [x] Test run: DreamShift concurrency=8 → ~780 tok/s avg, JetEngine bs=8 → ~370 tok/s peak
  - [x] Run full production collection (concurrency=32, ~90s duration)
  - [x] Save final time-series data to JSON

- [x] Phase 3: Visualization
  - [x] Write live_tps_chart.py — dark theme area chart, cyan vs red
  - [x] Test with short-run data → chart renders correctly with speedup overlay
  - [x] Generate final production charts (static PNG + animated GIF)
  - [x] Review output quality, iterate on styling (added --max-time crop)

- [ ] Phase 4: Polish
  - [x] Write run_demo.sh one-click launcher
  - [x] Fix SDAR time range to match DreamShift (20 batches → 93s, matches 90s DreamShift)
  - [x] Add --max-time flag to crop chart cleanly at 90s
  - [x] Fix GIF animation (matplotlib facecolor API change)
  - [ ] Optional: stream_compare.py for single-request side-by-side
  - [ ] Test full end-to-end flow with run_demo.sh

## Scripts Created
- `scripts/demo/collect_tps_timeseries.py` — DreamShift TPS via streaming token counting
- `scripts/demo/collect_jetengine_tps.py` — JetEngine SDAR batch TPS measurement
- `scripts/demo/live_tps_chart.py` — Dark theme area chart (static PNG + animated GIF)
- `scripts/demo/run_demo.sh` — One-click launcher
- `scripts/demo/test_jetengine.py` — Quick JetEngine validation script

## Production Results (Batch Size 32, max_tokens=2048, 1 GPU H100 each)
- **DreamShift N=4 (SGLang, concurrency=32):** 1,692 avg tok/s, 1,948 peak
- **SDAR Baseline (JetEngine, batch=32):** 425 avg tok/s, 618 peak
- **Speedup: 3.99×**
- Time coverage: DreamShift 90s, SDAR 93s (well-matched)

## Key Findings
- Server metric polling (`/v1/loads` gen_throughput) was unreliable — switched to direct
  streaming token counting which works perfectly
- JetEngine output `token_ids` includes prompt tokens — script subtracts prompt length
- JetEngine must be run via `torchrun --nproc_per_node=1` (not direct python)
- ffmpeg not available — using Pillow GIF writer instead of MP4
- matplotlib 3.10+ removed facecolor kwarg from Animation.save() — use fig.set_facecolor()

## Next Steps
1. Verify GIF animation renders correctly
2. Optional: stream_compare.py for single-request side-by-side demo
3. Test run_demo.sh end-to-end (requires restarting server)

## Progress Log
### Iteration 1 (2026-04-06)
- Created all 4 scripts + test script
- Started DreamShift server, verified both engines produce correct code
- Fixed TPS collection: switched from unreliable server metric polling to direct streaming
  token counting for DreamShift
- Fixed JetEngine token counting: subtract prompt tokens from output
- Validated full pipeline: collect data → generate chart → renders correctly
- Test chart shows 2.96× speedup (780 vs 264 tok/s at bs=8)

### Iteration 2 (2026-04-06)
- Ran full production data collection: DreamShift concurrency=32 (90s), SDAR batch=32 (20 batches, 93s)
- Production results: **3.99× speedup** (1,692 vs 425 tok/s)
- Generated static PNG chart with dark theme — looks polished
- Added --max-time flag to crop chart at 90s (removes DreamShift tail-off from request draining)
- Fixed GIF animation bug (matplotlib API change for facecolor)
- Switched from MP4 (no ffmpeg) to GIF (Pillow) for animation output
- Updated run_demo.sh to use GIF + --max-time
