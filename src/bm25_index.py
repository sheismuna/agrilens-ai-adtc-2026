"""
bm25_index.py

Builds a BM25 index over the chunked agriculture documents (data/chunks.json)
and saves the tokenized corpus + chunk metadata to data/bm25_index.json so
the live app can load it in well under a second without re-tokenizing.

Why BM25 instead of dense embeddings: see REPORT.md section 2.2. In short —
zero additional model resident in RAM, which matters directly for the ADTC
Efficiency score under the 7 GB ceiling, and BM25's lexical matching performs
well here because farmer queries and the reference docs share vocabulary
(disease names, pest names, fertilizer terms).

Usage:
    python src/bm25_index.py            # build the index
"""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "chunks.json"
INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "bm25_index.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Simple, dependency-light lowercase tokenizer.

    Deliberately not using a heavier NLP pipeline — BM25 works fine on
    plain lowercased alphanumeric tokens, and this keeps startup fast and
    RAM usage minimal.
    """
    return _TOKEN_RE.findall(text.lower())


def build_index():
    if not CHUNKS_PATH.exists():
        raise SystemExit(
            f"{CHUNKS_PATH} not found. Run `python src/ingest.py` first."
        )

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]

    # BM25Okapi builds term statistics; we don't need to keep the object
    # itself on disk, just the tokenized corpus, since rebuilding BM25Okapi
    # from tokens is fast (milliseconds for a corpus this size).
    index_payload = {
        "chunks": chunks,
        "tokenized_corpus": tokenized_corpus,
    }
    INDEX_PATH.write_text(json.dumps(index_payload), encoding="utf-8")
    print(f"[bm25_index] Indexed {len(chunks)} chunks -> {INDEX_PATH}")


class Retriever:
    """Loads the prebuilt index and answers top-k retrieval queries."""

    def __init__(self, index_path: Path = INDEX_PATH):
        if not index_path.exists():
            raise FileNotFoundError(
                f"{index_path} not found. Run `python src/bm25_index.py` first."
            )
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        self.chunks = payload["chunks"]
        self.bm25 = BM25Okapi(payload["tokenized_corpus"])

    def query(self, question: str, top_k: int = 3) -> list[dict]:
        tokens = tokenize(question)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        results = []
        for idx in ranked:
            if scores[idx] <= 0:
                continue  # don't return irrelevant chunks just to fill top_k
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results


if __name__ == "__main__":
    build_index()
