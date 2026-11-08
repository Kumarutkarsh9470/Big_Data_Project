#!/usr/bin/env bash
# Full experiment schedule (CPU, ~4 cores). Finished runs are skipped, so it is safe to re-run.
set -e
cd "$(dirname "$0")/.."
python3 scripts/run_grid.py --grid main --workers 4
python3 scripts/run_grid.py --grid plm --workers 1 --threads 4
python3 scripts/run_grid.py --grid ablation --workers 4
python3 scripts/run_grid.py --grid plm-seeds --workers 1 --threads 4
python3 scripts/analyze.py
