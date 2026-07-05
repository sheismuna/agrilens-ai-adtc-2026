#!/usr/bin/env bash
# Builds the BM25 retrieval index from data/agri_docs/.
# Run this once after adding or editing any document in data/agri_docs/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "[build_index] Chunking documents..."
python src/ingest.py

echo "[build_index] Building BM25 index..."
python src/bm25_index.py

echo "[build_index] Done."
