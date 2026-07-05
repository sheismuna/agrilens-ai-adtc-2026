# AgriLens AI

Offline maize advisory assistant for the Africa Deep Tech Challenge 2026,
Laptop LLM track (Agriculture domain). Runs entirely on an 8 GB RAM laptop
with no cloud dependency, no discrete GPU, using llama.cpp and a GGUF
quantized SmolLM3-3B model with a local BM25 retrieval-augmented generation
(RAG) pipeline over maize agronomy reference documents.

See `REPORT.md` for the full technical writeup and `metadata.json` for the
ADTC submission metadata.

## Quickstart

```bash
# 1. Install dependencies (CPU-only, no GPU packages needed)
pip install -r requirements.txt

# 2. Download the model weights (~1.9 GB, public Hugging Face repo)
bash download_model.sh

# 3. Build the retrieval index from data/agri_docs/
bash scripts/build_index.sh

# 4. Run the assistant
python -m src.app
```

Or run a single non-interactive prompt (useful for quick testing):

```bash
python -m src.app --prompt "My maize leaves have small round reddish-brown pustules. What is this?"
```

## Local ADTC profiler smoke test

```bash
bash scripts/run_profiler.sh
```

This installs the official `adtc-profiler`, ensures the model and index are
present, and runs it in participant mode to produce `submission.json` with
latency, throughput, and memory numbers before you submit.

## Project layout

See `STRUCTURE.txt` for the full annotated folder structure.

## Adding more agriculture knowledge

Drop additional `.md` files into `data/agri_docs/`, using `## Heading`
subsections as the retrievable unit (see existing files for the pattern),
then re-run `bash scripts/build_index.sh`.
