#!/usr/bin/env bash
#
# download_model.sh
#
# Downloads the AgriLens AI model weights (GGUF, quantized) into model/.
# Requirements from the ADTC 2026 submission template:
#   - Idempotent: safe to run multiple times without re-downloading.
#   - No credentials required: weights must be publicly accessible.
#   - Output path must exactly match `_runtime.model_path` in metadata.json.
#
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model"
MODEL_FILE="smollm3-3b-instruct-q4_k_m.gguf"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"

# Public Hugging Face repo, no auth token needed.
MODEL_URL="https://huggingface.co/ggml-org/SmolLM3-3B-GGUF/resolve/main/SmolLM3-Q4_K_M.gguf"

# Expected file size in bytes, used only as a sanity check (approx 1.92 GB).
MIN_EXPECTED_BYTES=1800000000

mkdir -p "${MODEL_DIR}"

if [ -f "${MODEL_PATH}" ]; then
  ACTUAL_BYTES=$(stat -c%s "${MODEL_PATH}" 2>/dev/null || stat -f%z "${MODEL_PATH}")
  if [ "${ACTUAL_BYTES}" -ge "${MIN_EXPECTED_BYTES}" ]; then
    echo "[download_model] ${MODEL_FILE} already present (${ACTUAL_BYTES} bytes). Skipping download."
    exit 0
  else
    echo "[download_model] Existing file looks incomplete (${ACTUAL_BYTES} bytes). Re-downloading."
    rm -f "${MODEL_PATH}"
  fi
fi

echo "[download_model] Fetching ${MODEL_FILE} from Hugging Face..."
curl -L --fail --retry 5 --retry-delay 5 -o "${MODEL_PATH}.part" "${MODEL_URL}"
mv "${MODEL_PATH}.part" "${MODEL_PATH}"

ACTUAL_BYTES=$(stat -c%s "${MODEL_PATH}" 2>/dev/null || stat -f%z "${MODEL_PATH}")
if [ "${ACTUAL_BYTES}" -lt "${MIN_EXPECTED_BYTES}" ]; then
  echo "[download_model] ERROR: downloaded file is smaller than expected (${ACTUAL_BYTES} bytes)." >&2
  exit 1
fi

echo "[download_model] Done. Model saved to ${MODEL_PATH}"
