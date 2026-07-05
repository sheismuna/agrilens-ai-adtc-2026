"""
app.py

AgriLens AI — offline maize advisory assistant.
Run from the project root:

    python -m src.app

Everything below this point runs with zero network access: model inference
via llama.cpp, retrieval via a prebuilt local BM25 index. No servers, no
external calls.
"""

import argparse
import sys

from src.rag_pipeline import AgriLensRAG

BANNER = """
========================================
  AgriLens AI - Offline Maize Assistant
  (type 'exit' to quit)
========================================
"""


def print_answer(result: dict):
    print("\n" + result["answer"] + "\n")
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  - {s['source']} :: {s['section']} (relevance {s['score']})")
    else:
        print("Sources: none matched — answer used general knowledge only.")
    print()


def run_single_prompt(rag: AgriLensRAG, question: str):
    result = rag.answer(question)
    print_answer(result)


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
        print_answer(result)


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
