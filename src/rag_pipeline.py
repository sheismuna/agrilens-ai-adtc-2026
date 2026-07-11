"""
rag_pipeline.py

The single entry point that combines BM25 retrieval with llama.cpp
generation. This is what app.py calls for every farmer query.

Instantiate AgriLensRAG ONCE per session and reuse it for multiple
questions â€” model loading only happens in __init__, not in answer().
This matters a lot in a notebook: a persistent object across cells means
you pay the ~model-load cost once, not on every question.
"""

import time

from src.bm25_index import Retriever
from src.llama_engine import AgriLensEngine


class AgriLensRAG:
    def __init__(self, top_k: int = 3, reasoning_mode: str = "/no_think"):
        t0 = time.perf_counter()
        self.retriever = Retriever()
        self.retriever_load_time_s = round(time.perf_counter() - t0, 3)

        self.engine = AgriLensEngine(reasoning_mode=reasoning_mode)
        self.top_k = top_k

    def answer(self, question: str) -> dict:
        t0 = time.perf_counter()
        retrieved = self.retriever.query(question, top_k=self.top_k)
        retrieval_time_s = round(time.perf_counter() - t0, 3)

        context_chunks = [r["text"] for r in retrieved]
        gen_result = self.engine.generate(question, context_chunks)

        return {
            "question": question,
            "answer": gen_result["text"],
            "sources": [
                {"source": r["source"], "section": r["section"], "score": round(r["score"], 3)}
                for r in retrieved
            ],
            "metrics": {
                "retrieval_time_s": retrieval_time_s,
                **gen_result["metrics"],
            },
        }

