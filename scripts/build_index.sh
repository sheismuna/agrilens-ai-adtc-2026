#!/usr/bin/env bash
# Builds the BM25 retrieval index from data/agri_docs/.
# Idempotent: skips rebuilding if the index is already newer than every
# source document, so re-running this in a notebook doesn't waste time
# re-chunking and re-indexing unchanged documents.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

INDEX_FILE="data/bm25_index.json"

needs_rebuild=1
if [ -f "$INDEX_FILE" ]; then
  newest_doc=$(find data/agri_docs -name '*.md' -newer "$INDEX_FILE" | head -n 1)
  if [ -z "$newest_doc" ]; then
    needs_rebuild=0
  fi
fi

if [ "$needs_rebuild" -eq 0 ]; then
  echo "[build_index] Index is already up to date. Skipping rebuild."
  exit 0
fi

echo "[build_index] Chunking documents..."
python src/ingest.py

echo "[build_index] Building BM25 index..."
python src/bm25_index.py

echo "[build_index] Done."

