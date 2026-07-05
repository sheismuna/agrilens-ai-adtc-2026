"""
llama_engine.py

Thin wrapper around llama-cpp-python that loads the GGUF model once and
exposes a single generate() call. Settings here are chosen specifically for
the ADTC 7 GB RAM ceiling and the 4 vCPU Standard Laptop profile:

  - n_ctx=4096: enough room for a system prompt + several retrieved chunks
    + conversation history, without letting the KV cache grow unbounded.
    Each doubling of n_ctx roughly doubles KV-cache RAM, so this is kept
    as small as the RAG use case allows rather than maxing out the model's
    native 64K context.
  - n_threads=4: matches the Standard Laptop's 4 vCPU spec. Over-threading
    on a 4-core machine causes contention, not speedup.
  - n_gpu_layers=0: explicit — the Standard Laptop has no discrete GPU, and
    integrated graphics offload is unreliable across judge machines, so we
    keep everything on CPU for guaranteed compatibility.
  - use_mmap=True, use_mlock=False: mmap lets the OS page in weights lazily
    and share pages if the OS is asked to run it twice; avoiding mlock keeps
    us from force-pinning all weight pages in physical RAM, which matters
    right at the 7 GB edge.
"""

from pathlib import Path

from llama_cpp import Llama

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "smollm3-3b-instruct-q4_k_m.gguf"

SYSTEM_PROMPT = (
    "You are AgriLens AI, an offline assistant for smallholder maize farmers "
    "in Africa. Answer using ONLY the reference passages provided in the "
    "context below. If the passages do not contain enough information to "
    "answer confidently, say so plainly instead of guessing. Keep answers "
    "practical, specific, and actionable for a farmer with basic literacy — "
    "prefer short paragraphs or a short numbered list over long prose."
)


class AgriLensEngine:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        n_ctx: int = 4096,
        n_threads: int = 4,
        reasoning_mode: str = "/no_think",
    ):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run `bash download_model.sh` first."
            )
        self.reasoning_mode = reasoning_mode
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )

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
    ) -> str:
        messages = self.build_prompt(question, context_chunks)
        result = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return result["choices"][0]["message"]["content"].strip()
