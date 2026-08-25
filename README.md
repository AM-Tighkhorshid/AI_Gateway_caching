# llm_cache_gateway

A gateway-level caching layer for LLM APIs, implementing every black-box
mechanism in the design document. Model-internal techniques (prefix caching,
KV reuse) are deliberately excluded — a gateway only sees requests and
responses, so those belong to the inference engine, not here.

Backed by OpenRouter's free models, with no hard dependencies beyond the
Python standard library.

---

## Coverage of the design document

| § | Mechanism | Implementation |
|---|---|---|
| 2.1 | Exact prompt caching | `exact_cache.ExactCache` — SHA-256 over normalized prompt + model + output-relevant params |
| 2.2 | Prompt normalization | `normalize.PromptNormalizer` — NFKC, casing, whitespace, punctuation, metadata stripping, JSON canonicalization, code-block protection |
| 2.3 | Keyword caching | `keyword_cache.KeywordCache` — inverted index + BM25; TF-IDF / RAKE / YAKE / KeyBERT extractors in `keywords.py` |
| 2.4 | Semantic caching | `semantic_cache.SemanticCache` — embeddings + cosine vector search + threshold, margin, and scope guards |
| 2.5 | Hierarchical caching | `gateway.CachingGateway` — exact → keyword → semantic → few-shot → LLM |
| 2.6 | Response caching | Same entry store; all layers resolve to one `CacheEntry` |
| 2.7 | Few-shot retrieval | `context_cache.FewShotCache` — neighbours below the reuse threshold become demonstrations |
| 2.8 | Conversation context | `context_cache.ConversationCache` — rolling summary + last N turns, summary itself cached |
| 2.9 | Prefix caching | **Out of scope** — inference-engine concern (vLLM), not visible to a gateway |
| 2.10 | KV cache reuse | **Out of scope** — same reason |
| 2.11 | Adaptive policies | `policy.AdaptivePolicy` — volatility, personal-data, determinism, popularity, TTL, cost-aware eviction, feedback-driven thresholds |

---

## Install and run

```bash
pip install -r requirements.txt          # optional extras only
python demo.py offline                   # full walkthrough, no API key needed
python -m pytest tests/ -q               # 27 tests, all offline

export OPENROUTER_API_KEY=sk-or-...      # from openrouter.ai/keys
python demo.py models                    # which models are free right now
python demo.py live                      # real calls through the cache
python demo.py bench                     # hit rate / latency benchmark
```

## Usage

```python
from llm_cache_gateway import CacheConfig, CachingGateway

gw = CachingGateway(CacheConfig.from_env(backend="disk"))

r1 = gw.complete("What is a vector database?")
print(r1.cached, r1.source, r1.latency_ms)     # False llm     1840.2

r2 = gw.complete("What's a vector database?")
print(r2.cached, r2.source, r2.similarity)     # True  semantic 0.94

print(gw.report())
```

Per-request overrides mirror HTTP cache-control semantics:

```python
gw.complete(prompt, no_cache=True)        # skip the read path
gw.complete(prompt, no_store=True)        # answer, but do not cache
gw.complete(prompt, ttl=300)              # custom lifetime
gw.complete(messages, conversation_id="c1", user_id="u42")
```

Feedback closes the loop on retrieval quality:

```python
gw.feedback(response.entry_key, accepted=False)   # raises the threshold,
                                                  # retires the bad entry
```

---

## A note on free OpenRouter models

Free model IDs rotate constantly — entire tiers get delisted week to week.
The client therefore **discovers** a currently-free model from
`GET /models` (pricing `prompt == completion == 0`) instead of hard-coding
one. Pin `chat_model` in the config when you need reproducible behaviour.
Free tiers are rate-limited (roughly 20 req/min), so the retry path handles
`429` with exponential backoff and honours `Retry-After`.

Embeddings go through `POST /embeddings`, which is OpenAI-compatible. The
embedder falls back automatically: **OpenRouter → sentence-transformers →
hashing**.

The hashing fallback deserves a warning. It hashes word n-grams and character
n-grams, so it catches vocabulary overlap but **is not a semantic model** — in
testing, "create a Kubernetes deployment" vs "create a Kubernetes ingress"
scores *higher* (0.85) than a true paraphrase pair (0.60). It exists so the
test suite and demo run offline. Do not ship it as your semantic layer. The
gateway logs a warning and lowers the threshold to 0.62 when it falls back to
hashing.

---

## Design decisions worth knowing about

**Keyword hits are confirmed semantically by default.** Lexical overlap is a
weak proxy for intent: "how do I *start* a container" and "how do I *stop* a
container" share nearly every keyword. `keyword_requires_semantic_confirm`
makes the keyword layer a fast *candidate generator* whose winner still has to
clear an embedding check. Turn it off only if you have measured the false-hit
rate on your own traffic.

**Scope is part of correctness.** A cached answer is only served when the
model matches, output-affecting parameters match, and — for prompts
classified as personal — the user matches. Parameters like `stream` don't
participate in the key, since they don't change the content.

**Volatile queries are not served from cache at all by default.** Anything
matching "latest / today / price / news / current" gets a short TTL and is
not reused. Serving yesterday's stock price fast is worse than serving
today's slowly.

**Non-deterministic requests are not stored.** "Write a poem", "give me
another one", `temperature > 0.5` — reusing these defeats the point of asking.

**Few-shot retrieval is the safety valve for the mid-similarity band.** When a
neighbour is related but not close enough to reuse (0.55–0.90), it becomes a
demonstration rather than an answer. The model still runs, so the answer is
fresh, but it is grounded in prior accepted answers.

**Eviction is cost-aware, not just LRU.** Utility combines recency, hit
frequency, regeneration cost, and feedback quality, so a rarely-hit but
expensive answer can outrank a cheap popular one. `lru` and `lfu` remain
available via `eviction_policy`.

---

## Configuration

Everything is in `config.py`. The knobs that matter most in production:

| Setting | Default | Notes |
|---|---|---|
| `similarity_threshold` | `0.90` | The single highest-leverage number. Below ~0.85 with a real embedding model, false hits start appearing. |
| `keyword_min_score` | `0.55` | Normalized BM25 + Jaccard. |
| `keyword_requires_semantic_confirm` | `True` | See above. |
| `default_ttl` / `volatile_ttl` | `24h` / `120s` | |
| `store_after_n_requests` | `1` | Set to 2–3 to keep one-off prompts out of the index. |
| `max_temperature_for_reuse` | `0.5` | |
| `backend` | `memory` | `disk` (SQLite) or `redis` for multi-process deployments. |
| `max_entries` | `20000` | Triggers eviction. |

---

## Scaling notes

`VectorIndex` is an exact brute-force cosine search, numpy-accelerated when
available. That is the right call up to roughly 10⁵ vectors. Past that, swap
it for FAISS, Qdrant, Milvus, or Redis vector search — the interface it has to
satisfy is only three methods: `add`, `search`, `remove`.

The keyword inverted index is rewritten on each store, which is fine at
moderate write rates but should become an incremental Redis-hash write in a
high-throughput deployment.

Everything here is synchronous. A production gateway should run the layers
async, and the embedding call for the semantic layer is the one blocking hop
worth parallelizing against the few-shot retrieval.

---

## Layout

```
llm_cache_gateway/
  config.py           all tunables
  types.py            CacheRequest / CacheEntry / GatewayResponse
  normalize.py        §2.2
  keywords.py         TF-IDF, RAKE, YAKE, KeyBERT, BM25
  keyword_cache.py    §2.3
  embedding.py        OpenRouter / sbert / hashing backends + memoization
  semantic_cache.py   §2.4 + vector index
  exact_cache.py      §2.1, §2.6
  context_cache.py    §2.7, §2.8
  policy.py           §2.11
  repository.py       shared entry store + eviction
  store.py            memory / sqlite / redis
  providers.py        OpenRouter client + offline mock
  stats.py            metrics
  gateway.py          §2.5 orchestration
demo.py               offline | live | bench | models
tests/                27 offline tests
```
