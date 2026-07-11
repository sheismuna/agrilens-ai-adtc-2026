"""
app.py

AgriLens AI - offline maize advisory assistant.
Run from the project root:

    python -m src.app

Everything below this point runs with zero network access: model inference
via llama.cpp, retrieval via a prebuilt local BM25 index. No servers, no
external calls.

NOTEBOOK USAGE (recommended for demos / multiple questions):
Don't re-run `!python -m src.app --prompt "..."` as a separate process for
every question â€” each invocation reloads the ~1.9GB model from scratch,
which is where the "~2 minutes before an answer" time comes from. Instead,
in a notebook cell:

    from src.rag_pipeline import AgriLensRAG
    rag = AgriLensRAG()              # loads model ONCE
    print_answer(rag.answer("..."))  # fast - reuses loaded model
    print_answer(rag.answer("..."))  # fast - reuses loaded model
"""

import argparse
import sys

import psutil

from src.rag_pipeline import AgriLensRAG

BANNER = """
========================================
  AgriLens AI - Offline Maize Assistant
  (type 'exit' to quit)
========================================
"""

MODEL_NAME = "SmolLM3-3B-Instruct"
QUANTIZATION = "GGUF Q4_K_M"


def print_metrics_panel(rag: AgriLensRAG, answer_metrics: dict):
    proc = psutil.Process()
    current_rss_mb = proc.memory_info().rss / (1024 * 1024)

    print("--- Performance metrics ---")
    print(f"  Model:              {MODEL_NAME} ({QUANTIZATION})")
    print(f"  Context window:     {rag.engine.n_ctx} tokens, {rag.engine.n_threads} threads")
    print(f"  Model load time:    {rag.engine.load_time_s:.2f} s  (one-time, paid once per session)")
    print(f"  Approx RAM in use:  {current_rss_mb:.0f} MB")
    print(f"  Retrieval time:     {answer_metrics['retrieval_time_s']:.3f} s")
    print(f"  Generation time:    {answer_metrics['generation_time_s']:.2f} s")
    print(f"  Tokens generated:   {answer_metrics['completion_tokens']}")
    print(f"  Tokens/second:      {answer_metrics['tokens_per_second']:.2f}")
    print("----------------------------\n")


def print_answer(result: dict, rag: AgriLensRAG, show_metrics: bool = True):
    print("\n" + result["answer"] + "\n")
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  - {s['source']} :: {s['section']} (relevance {s['score']})")
    else:
        print("Sources: none matched â€” answer used general knowledge only.")
    print()
    if show_metrics:
        print_metrics_panel(rag, result["metrics"])


def run_single_prompt(rag: AgriLensRAG, question: str):
    result = rag.answer(question)
    print_answer(result, rag)


def run_interactive(rag: AgriLensRAG):
    print(BANNER)
    while True:
        try:
            question = input("Ask AgriLens AI> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        result = rag.answer(question)
        print_answer(result, rag)


def main():
    parser = argparse.ArgumentParser(description="AgriLens AI CLI")
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Run a single prompt non-interactively and exit (useful for testing).",
    )
    parser.add_argument(
        "--think", action="store_true",
        help="Use SmolLM3's /think reasoning mode instead of /no_think.",
    )
    args = parser.parse_args()

    reasoning_mode = "/think" if args.think else "/no_think"
    print("Loading model and index (offline, no network calls)...", file=sys.stderr)
    rag = AgriLensRAG(reasoning_mode=reasoning_mode)

    if args.prompt:
        run_single_prompt(rag, args.prompt)
    else:
        run_interactive(rag)


if __name__ == "__main__":
    main()

