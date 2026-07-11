"""
llama_engine.py

Thin wrapper around llama-cpp-python that loads the GGUF model once and
exposes a single generate() call. Settings here are chosen specifically for
the ADTC 7 GB RAM ceiling and the 4 vCPU Standard Laptop profile:

  - n_ctx=2048: RAG context here is a system prompt + 3 short retrieved
    chunks + one question â€” comfortably fits in 2048 tokens. Halving n_ctx
    from 4096 roughly halves KV-cache RAM and speeds up context allocation
    at load time, with no loss of capability for this use case.
  - n_threads=4: matches the Standard Laptop's 4 vCPU spec. Over-threading
    on a 4-core machine causes contention, not speedup. Auto-detected at
    runtime (see _default_threads()) so it also behaves sensibly on smaller
    dev machines (e.g. a 2-vCPU cloud notebook).
  - n_gpu_layers=0: explicit â€” the Standard Laptop has no discrete GPU, and
    integrated graphics offload is unreliable across judge machines, so we
    keep everything on CPU for guaranteed compatibility.
  - use_mmap=True, use_mlock=False: mmap lets the OS page in weights lazily
    and share pages if the OS is asked to run it twice; avoiding mlock keeps
    us from force-pinning all weight pages in physical RAM, which matters
    right at the 7 GB edge.

Performance note: model loading (reading ~1.9GB off disk + llama.cpp init)
is the dominant cost at ~1-2 minutes depending on disk speed, NOT inference.
This cost is paid ONCE per AgriLensEngine instance. If you are re-running
`python -m src.app --prompt "..."` as a separate process every time (e.g.
one `!python ...` cell per question in a notebook), you pay this cost on
EVERY call, since each process starts with a cold model. For a demo or
multi-question session, instantiate AgriLensRAG/AgriLensEngine ONCE in a
notebook cell and call .answer() multiple times against that same object
instead of shelling out repeatedly â€” see the "Notebook usage" section in
README.md.
"""

import re
import time
from pathlib import Path

import psutil
from llama_cpp import Llama

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "smollm3-3b-instruct-q4_k_m.gguf"

SYSTEM_PROMPT = (
    "You are AgriLens AI, an offline assistant for smallholder maize farmers "
    "in Africa. Answer using ONLY the reference passages provided in the "
    "context below. If the passages do not contain enough information to "
    "answer confidently, say so plainly instead of guessing. Keep answers "
    "practical, specific, and actionable for a farmer with basic literacy â€” "
    "prefer short paragraphs or a short numbered list over long prose."
)

# Strips <think>...</think> blocks (including empty ones) from model output.
# SmolLM3 emits this wrapper even in /no_think mode; it should never reach
# the user. DOTALL so it also catches multi-line reasoning if /think is used.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _default_threads() -> int:
    """Match physical/logical CPU count, capped at 4 to mirror the ADTC
    Standard Laptop spec even on machines with more cores available."""
    return max(1, min(4, psutil.cpu_count(logical=True) or 4))


def _strip_think(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", text)
    return cleaned.strip()


class AgriLensEngine:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        reasoning_mode: str = "/no_think",
    ):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run `bash download_model.sh` first."
            )
        self.reasoning_mode = reasoning_mode
        n_threads = n_threads or _default_threads()

        proc = psutil.Process()
        rss_before_mb = proc.memory_info().rss / (1024 * 1024)
        t0 = time.perf_counter()

        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )

        self.load_time_s = time.perf_counter() - t0
        rss_after_mb = proc.memory_info().rss / (1024 * 1024)
        self.load_rss_mb = rss_after_mb
        self.load_rss_delta_mb = rss_after_mb - rss_before_mb
        self.n_ctx = n_ctx
        self.n_threads = n_threads

    def build_prompt(self, question: str, context_chunks: list[str]) -> list[dict]:
        if context_chunks:
            context_block = "\n\n---\n\n".join(context_chunks)
        else:
            context_block = "(no matching reference passages found)"

        user_content = (
            f"{self.reasoning_mode}\n\n"
            f"Reference passages:\n{context_block}\n\n"
            f"Farmer's question: {question}"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def generate(
        self,
        question: str,
        context_chunks: list[str],
        max_tokens: int = 400,
        temperature: float = 0.3,
    ) -> dict:
        """Returns {"text": str, "metrics": dict} â€” text is already cleaned
        of any <think> block and safe to print directly to the user."""
        messages = self.build_prompt(question, context_chunks)

        t0 = time.perf_counter()
        result = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        gen_time_s = time.perf_counter() - t0

        raw_text = result["choices"][0]["message"]["content"]
        clean_text = _strip_think(raw_text)

        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        tokens_per_sec = (completion_tokens / gen_time_s) if gen_time_s > 0 else 0.0

        return {
            "text": clean_text,
            "metrics": {
                "generation_time_s": round(gen_time_s, 2),
                "completion_tokens": completion_tokens,
                "tokens_per_second": round(tokens_per_sec, 2),
            },
        }

