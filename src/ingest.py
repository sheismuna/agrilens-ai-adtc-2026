"""
ingest.py

Reads all markdown files in data/agri_docs/, splits them into retrievable
chunks along "## " subsections (each subsection is already a self-contained
topic, e.g. one disease, one pest, one fertilizer stage), and writes a flat
list of chunk records to data/chunks.json.

This runs once, offline, ahead of time — it is NOT part of the live inference
path, so it has no effect on the tokens/sec or latency benchmarks.

Usage:
    python src/ingest.py
"""

import json
import re
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "agri_docs"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "chunks.json"


def split_into_sections(text: str, source: str) -> list[dict]:
    """Split a markdown doc into (heading, body) chunks on '## ' boundaries.

    Keeps the top-level '# Title' as context prepended to every chunk from
    that file, so retrieved passages remain understandable in isolation.
    """
    lines = text.splitlines()
    title = ""
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        lines = lines[1:]

    chunks = []
    current_heading = None
    current_body: list[str] = []

    def flush():
        if current_heading is not None and current_body:
            body_text = "\n".join(current_body).strip()
            if body_text:
                chunks.append(
                    {
                        "source": source,
                        "doc_title": title,
                        "section": current_heading,
                        "text": f"{current_heading}\n{body_text}",
                    }
                )

    for line in lines:
        match = re.match(r"^##\s+(.*)", line)
        if match:
            flush()
            current_heading = match.group(1).strip()
            current_body = []
        else:
            current_body.append(line)
    flush()

    return chunks


def main():
    if not DOCS_DIR.exists():
        raise SystemExit(f"No documents directory found at {DOCS_DIR}")

    all_chunks = []
    md_files = sorted(DOCS_DIR.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No markdown files found in {DOCS_DIR}")

    for path in md_files:
        text = path.read_text(encoding="utf-8")
        chunks = split_into_sections(text, source=path.name)
        all_chunks.extend(chunks)

    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = f"c_{i:04d}"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    print(f"[ingest] Wrote {len(all_chunks)} chunks from {len(md_files)} files "
          f"to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
