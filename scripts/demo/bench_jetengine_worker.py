#!/usr/bin/env python3
"""JetEngine SDAR benchmark worker.

Wrapper entry point for collect_jetengine_tps.py.
Must be run via: CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 bench_jetengine_worker.py [args]

This is a thin wrapper — all logic lives in collect_jetengine_tps.py.
"""

from collect_jetengine_tps import main

if __name__ == "__main__":
    main()
