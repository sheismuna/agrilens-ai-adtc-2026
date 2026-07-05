#!/usr/bin/env bash
# Local smoke test using the official ADTC profiler, mirroring what judges
# will run during Gate 2 auditing. See:
# https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "[run_profiler] Ensuring profiler is installed..."
pip install --quiet "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"

echo "[run_profiler] Ensuring model weights are present..."
bash download_model.sh

echo "[run_profiler] Ensuring retrieval index is built..."
bash scripts/build_index.sh

echo "[run_profiler] Running profiler in participant mode..."
adtc-profiler run \
  --submission . \
  --mode participant \
  --output submission.json \
  --skip-accuracy

echo "[run_profiler] Done. Report:"
cat submission.json
