# AgriLens AI — Africa Deep Tech Challenge 2026 (Laptop LLM Track)

**Domain:** Agriculture
**Runtime:** llama.cpp
**Model:** SmolLM3-3B-Instruct, GGUF Q4_K_M (~1.92 GB)
**Target hardware:** ADTC Standard Laptop (4 vCPU, 8 GB RAM, integrated GPU, no discrete GPU)

---

## 1. Problem

Smallholder maize farmers across much of Africa do not have reliable access to
agricultural extension officers, and many rural areas lack stable connectivity
for cloud-based advisory tools. A farmer noticing unfamiliar lesions on a maize
leaf, or unsure how much fertilizer to apply on a small plot, often has no fast
way to get a reliable answer.

AgriLens AI is an **offline** assistant that runs entirely on an 8 GB RAM
laptop — the kind already common in agricultural extension offices, NGO field
kits, and rural resource centers — and answers questions about:

- Maize disease diagnosis and treatment
- Pest identification and management
- Planting guidance (spacing, timing, seed rate)
- Fertilizer type, quantity, and timing
- Harvesting and post-harvest handling

The target user is a field extension worker or literate farmer using a shared
laptop, not necessarily an individual smartphone — which matches the ADTC
Standard Laptop profile directly.

## 2. Design decisions

### 2.1 Model selection

We evaluated three candidates under 4B parameters, all Apache-2.0 licensed and
llama.cpp-compatible:

| Model | Params | Q4_K_M size | Why / why not |
|---|---|---|---|
| **SmolLM3-3B-Instruct** (chosen) | 3.08B | ~1.92 GB | Outperforms Llama-3.2-3B and Qwen2.5-3B on public benchmarks; dual `/think` and `/no_think` modes let us trade latency for reasoning depth per-query; 64K native context comfortably holds retrieved passages |
| Qwen2.5-3B-Instruct | 3.09B | ~1.9 GB | Strong alternative, slightly weaker instruction following in early testing |
| Qwen3-1.7B-Instruct | 1.7B | ~1.1 GB | Kept as a fallback if profiler runs show RAM pressure; noticeably weaker at multi-step diagnostic reasoning |

We default to `/no_think` mode for normal queries (favors the Speed score,
`S_perf`) and only escalate to `/think` mode when a query is ambiguous (e.g.
multiple possible diseases match the symptom description), trading a small
latency cost for accuracy on genuinely hard cases.

### 2.2 Retrieval strategy: BM25 over dense embeddings

We deliberately chose **lexical retrieval (BM25, via `rank_bm25`)** instead of
a dense embedding model for RAG. Reasoning:

1. **Memory budget.** A second model resident in RAM (an embedding model) eats
   directly into the 7 GB evaluation ceiling and the Efficiency score
   (`S_eff = 100 × ((7GB − Peak RAM) / 7GB)`). BM25 has effectively zero
   incremental RAM cost — no torch, no transformers, no second model load.
2. **Domain fit.** Agriculture Q&A has high lexical overlap between queries
   and reference text (farmers and agronomy documents both use terms like
   "maize streak virus," "urea," "fall armyworm"), so BM25's term-frequency
   matching performs competitively with dense retrieval at this document
   scale (tens of short reference documents, not millions).
3. **Startup latency.** The BM25 index loads from a flat JSON file in well
   under a second, which matters for a responsive CLI experience under the
   audit's latency measurement.

### 2.3 Packaging

Single Python process: CLI app → BM25 retriever → prompt builder → llama.cpp
via `llama-cpp-python` bindings, model loaded once and kept resident for the
session. No servers, no background processes, no network calls after the
model is downloaded.

## 3. Constraints

- **Hardware:** ADTC Standard Laptop — 4 vCPU x86-64, 8 GB RAM, integrated
  graphics only, no discrete GPU. All inference is CPU-only.
- **Memory:** Hard 7 GB ceiling; exceeding it is an automatic disqualification
  regardless of answer quality. This is the primary constraint that shaped
  the BM25-over-dense-embedding decision above.
- **Connectivity:** Zero network access permitted during evaluation. All
  weights and index data must be present locally before the profiling window
  starts (`download_model.sh` runs before profiling, not during it).
- **Data:** No proprietary agronomy datasets were available for this
  submission; the reference corpus in `data/agri_docs/` is built from
  well-established, publicly documented maize agronomy practices.

## 4. Benchmarks (development machine)

*(Fill in with your own numbers before submitting — run
`scripts/run_profiler.sh` and copy the relevant fields from `submission.json`.)*

| Metric | Observed value | Notes |
|---|---|---|
| Peak RSS during inference | 3.24 GB | measured via `adtc-profiler`, well within the 7GB budget |
| Tokens/sec (generation) | 3.31 tok/s |  measured on a 2-vCPU shared cloud CPU (Google Colab free tier, confirmed via `nproc`), not the 4-vCPU dedicated ADTC Standard Laptop — expect meaningfully higher throughput on target hardware  |
| Time to first token | ~68.1 s |  includes 512-token prompt processing on the constrained 2-vCPU environment; CPU-allocation-bound rather than representative of dedicated 4-core hardware |
| Index build time | <1 s  | one-time, offline, not part of inference path |
| Index load time | <1 s | flat JSON, no external service |

## 5. Known limitations

- The reference corpus is illustrative and should be expanded with
  region-specific agronomy guidance (soil type, local pest pressure, rainfall
  patterns) before real-world field deployment.
- BM25 can miss paraphrased queries that share no vocabulary with the
  reference text; a future iteration could add a lightweight GGUF embedding
  model in embedding-only mode if RAM allows, as a hybrid re-ranker.
- The model is not fine-tuned on agricultural data; all domain accuracy comes
  from retrieval grounding plus the base model's general knowledge. Prompts
  explicitly instruct the model to say "I don't have information on this" 
  when no relevant document is retrieved, to reduce hallucination risk on
  out-of-corpus questions.
