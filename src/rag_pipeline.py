"""
rag_pipeline.py

The single entry point that combines BM25 retrieval with llama.cpp
generation. This is what app.py calls for every farmer query.
"""

from src.bm25_index import Retriever
from src.llama_engine import AgriLensEngine


class AgriLensRAG:
    def __init__(self, top_k: int = 3, reasoning_mode: str = "/no_think"):
        self.retriever = Retriever()
        self.engine = AgriLensEngine(reasoning_mode=reasoning_mode)
        self.top_k = top_k

    def answer(self, question: str) -> dict:
        retrieved = self.retriever.query(question, top_k=self.top_k)
        context_chunks = [r["text"] for r in retrieved]

        answer_text = self.engine.generate(question, context_chunks)

        return {
            "question": question,
            "answer": answer_text,
            "sources": [
                {"source": r["source"], "section": r["section"], "score": round(r["score"], 3)}
                for r in retrieved
            ],
        }
