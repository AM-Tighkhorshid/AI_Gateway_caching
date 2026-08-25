"""Offline test suite - exercises every layer without network or API key."""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_cache_gateway import (CacheConfig, CachingGateway, DiskStore,
                               HashingEmbedder, KeywordExtractor, MemoryStore,
                               MockProvider, PromptNormalizer,
                               build_offline_gateway)


def make_gateway(**overrides) -> CachingGateway:
    config = CacheConfig(
        backend="memory",
        embedding_backend="hashing",
        chat_model="mock/echo",
        log_level="ERROR",
        **overrides,
    )
    return CachingGateway(config=config, provider=MockProvider(latency=0.0))


# --------------------------------------------------------------------------- #
# 2.2 normalization
# --------------------------------------------------------------------------- #
def test_normalization_collapses_cosmetic_differences():
    norm = PromptNormalizer(CacheConfig())
    a = norm.normalize("  What   is   Kubernetes?? ")
    b = norm.normalize("what is kubernetes?")
    assert a == b


def test_normalization_strips_volatile_metadata():
    norm = PromptNormalizer(CacheConfig())
    a = norm.normalize("request-id: abc123\nSummarize the doc")
    b = norm.normalize("request_id: zzz999\nSummarize the doc")
    assert a == b


def test_normalization_preserves_code_blocks():
    norm = PromptNormalizer(CacheConfig())
    out = norm.normalize("Fix this:\n```python\nX = ClassName()\n```")
    assert "ClassName" in out  # casing inside code survives


def test_normalization_canonicalizes_json():
    norm = PromptNormalizer(CacheConfig())
    a = norm.normalize('{"b": 2, "a": 1}')
    b = norm.normalize('{"a": 1, "b": 2}')
    assert a == b


# --------------------------------------------------------------------------- #
# 2.1 / 2.6 exact + response cache
# --------------------------------------------------------------------------- #
def test_exact_cache_hit():
    gw = make_gateway(semantic_enabled=False, keyword_enabled=False,
                      fewshot_enabled=False)
    first = gw.complete("What is the capital of France?")
    second = gw.complete("What is the capital of France?")
    assert not first.cached
    assert second.cached and second.source == "exact"
    assert second.text == first.text
    assert gw.provider.call_count == 1


def test_exact_cache_after_normalization():
    gw = make_gateway(semantic_enabled=False, keyword_enabled=False,
                      fewshot_enabled=False)
    gw.complete("What is the capital of France?")
    hit = gw.complete("   what is the CAPITAL of France???  ")
    assert hit.cached and hit.source == "exact"


def test_different_params_are_different_keys():
    gw = make_gateway(semantic_enabled=False, keyword_enabled=False,
                      fewshot_enabled=False)
    gw.complete("Ping", max_tokens=100)
    result = gw.complete("Ping", max_tokens=500)
    assert not result.cached


# --------------------------------------------------------------------------- #
# 2.3 keyword cache
# --------------------------------------------------------------------------- #
def test_keyword_extractors_all_run():
    text = ("Kubernetes horizontal pod autoscaler scales deployments "
            "based on observed CPU utilization metrics.")
    for method in ("tfidf", "rake", "yake"):
        extractor = KeywordExtractor(method=method, top_k=5)
        keywords = extractor.extract(text)
        assert keywords, f"{method} returned nothing"
        assert all(isinstance(k, str) for k in keywords)


def test_keybert_extractor_with_embedder():
    extractor = KeywordExtractor(method="keybert", top_k=5,
                                 embedder=HashingEmbedder(256))
    assert extractor.extract("How do I resize a persistent volume claim?")


def test_keyword_layer_matches_reordered_wording():
    gw = make_gateway(semantic_enabled=False, fewshot_enabled=False,
                      keyword_requires_semantic_confirm=False,
                      keyword_min_score=0.4)
    gw.complete("How do I restart a docker container safely?")
    result = gw.complete("Safely restart docker container - how?")
    assert result.cached and result.source == "keyword"


# --------------------------------------------------------------------------- #
# 2.4 semantic cache
# --------------------------------------------------------------------------- #
def test_semantic_cache_hits_paraphrase():
    # NOTE: 0.55 is calibrated for the offline hashing embedder, whose cosine
    # values are lower than a real embedding model's. With OpenRouter or
    # sentence-transformers the equivalent threshold is ~0.90.
    gw = make_gateway(keyword_enabled=False, fewshot_enabled=False,
                      similarity_threshold=0.55)
    gw.complete("How do I install Python packages with pip?")
    result = gw.complete("How can I install python packages using pip?")
    assert result.cached and result.source == "semantic"
    assert result.similarity >= 0.55


def test_semantic_cache_misses_unrelated_prompt():
    gw = make_gateway(keyword_enabled=False, fewshot_enabled=False,
                      similarity_threshold=0.85)
    gw.complete("How do I install Python packages with pip?")
    result = gw.complete("What is the boiling point of mercury?")
    assert not result.cached


def test_semantic_scope_respects_model():
    gw = make_gateway(keyword_enabled=False, fewshot_enabled=False,
                      similarity_threshold=0.7)
    gw.complete("Explain gradient descent", model="model-a")
    result = gw.complete("Explain gradient descent please", model="model-b")
    assert not result.cached


# --------------------------------------------------------------------------- #
# 2.5 hierarchy
# --------------------------------------------------------------------------- #
def test_hierarchy_prefers_cheapest_layer():
    gw = make_gateway(similarity_threshold=0.75, fewshot_enabled=False)
    gw.complete("What is a vector database?")
    exact_hit = gw.complete("What is a vector database?")
    assert exact_hit.source == "exact"
    layers = [step["layer"] for step in exact_hit.trace]
    # The semantic layer must not run once the exact key hits.
    assert "semantic" not in layers


# --------------------------------------------------------------------------- #
# 2.7 few-shot retrieval
# --------------------------------------------------------------------------- #
def test_fewshot_injects_related_examples():
    gw = make_gateway(similarity_threshold=0.999, keyword_enabled=False,
                      fewshot_min_similarity=0.3, fewshot_k=2)
    gw.complete("How do I create a Kubernetes deployment?")
    gw.complete("How do I create a Kubernetes service?")
    result = gw.complete("How do I create a Kubernetes ingress?")
    assert not result.cached
    assert result.fewshot_used >= 1
    injected = gw.provider.calls[-1]["messages"]
    assert any("Example 1" in str(m.get("content", "")) for m in injected)


# --------------------------------------------------------------------------- #
# 2.8 conversation context cache
# --------------------------------------------------------------------------- #
def test_conversation_compaction_shortens_history():
    gw = make_gateway(conversation_summary_trigger_chars=500,
                      conversation_keep_recent_turns=2,
                      semantic_enabled=False, keyword_enabled=False,
                      fewshot_enabled=False)
    history = []
    for i in range(12):
        history.append({"role": "user", "content": f"Message {i} " + "x" * 60})
        history.append({"role": "assistant", "content": f"Reply {i} " + "y" * 60})
    history.append({"role": "user", "content": "So what did we decide?"})

    result = gw.complete(history, conversation_id="conv-1")
    assert result.conversation_compacted
    sent = gw.provider.calls[-1]["messages"]
    assert len(sent) < len(history)
    assert any("Summary of the earlier conversation" in str(m.get("content", ""))
               for m in sent)


# --------------------------------------------------------------------------- #
# 2.11 adaptive policies
# --------------------------------------------------------------------------- #
def test_volatile_queries_are_not_served_from_cache():
    gw = make_gateway(fewshot_enabled=False)
    gw.complete("What is the latest news about AI today?")
    result = gw.complete("What is the latest news about AI today?")
    assert not result.cached


def test_high_temperature_is_not_cached():
    gw = make_gateway(fewshot_enabled=False, max_temperature_for_reuse=0.5)
    gw.complete("Write a haiku about rain", temperature=1.2)
    result = gw.complete("Write a haiku about rain", temperature=1.2)
    assert not result.cached


def test_popularity_gate_delays_storage():
    gw = make_gateway(store_after_n_requests=3, semantic_enabled=False,
                      keyword_enabled=False, fewshot_enabled=False)
    for _ in range(2):
        assert not gw.complete("Rare one-off prompt about widgets").cached
    assert gw.repo.count() == 0
    gw.complete("Rare one-off prompt about widgets")   # 3rd -> stored
    assert gw.repo.count() == 1
    assert gw.complete("Rare one-off prompt about widgets").cached


def test_no_cache_and_no_store_flags():
    gw = make_gateway(fewshot_enabled=False)
    gw.complete("Cacheable question about trees")
    bypass = gw.complete("Cacheable question about trees", no_cache=True)
    assert not bypass.cached

    gw2 = make_gateway(fewshot_enabled=False)
    gw2.complete("Secret question", no_store=True)
    assert gw2.repo.count() == 0


def test_feedback_moves_threshold_and_evicts_bad_entry():
    gw = make_gateway(fewshot_enabled=False, similarity_threshold=0.9)
    first = gw.complete("How do I rotate API keys?")
    before = gw.policy.threshold
    gw.feedback(first.entry_key, accepted=False)
    assert gw.policy.threshold > before
    gw.feedback(first.entry_key, accepted=False)
    assert gw.repo.get(first.entry_key) is None  # retired after repeated rejects


def test_ttl_expiry():
    gw = make_gateway(default_ttl=0.4, fewshot_enabled=False)
    gw.complete("Question with a short ttl")
    assert gw.complete("Question with a short ttl").cached
    time.sleep(0.5)
    assert not gw.complete("Question with a short ttl").cached


def test_cost_aware_eviction_enforces_capacity():
    gw = make_gateway(max_entries=5, fewshot_enabled=False,
                      semantic_enabled=False, keyword_enabled=False)
    for i in range(12):
        gw.complete(f"Unique prompt number {i} about topic {i}")
    assert gw.repo.count() <= 5
    assert gw.repo.evicted > 0


# --------------------------------------------------------------------------- #
# storage backends
# --------------------------------------------------------------------------- #
def test_memory_and_disk_stores_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        for store in (MemoryStore(), DiskStore(os.path.join(tmp, "c.sqlite3"))):
            store.set("a:1", {"v": 1})
            store.set("a:2", {"v": 2})
            store.set("b:1", {"v": 3})
            assert store.get("a:1") == {"v": 1}
            assert sorted(store.keys("a:")) == ["a:1", "a:2"]
            assert store.incr("ctr") == 1 and store.incr("ctr") == 2
            store.delete("a:1")
            assert store.get("a:1") is None
            assert store.clear("a:") == 1


def test_disk_backend_persists_across_gateways():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cache.sqlite3")
        cfg = dict(backend="disk", disk_path=path, embedding_backend="hashing",
                   chat_model="mock/echo", log_level="ERROR")
        gw1 = CachingGateway(config=CacheConfig(**cfg),
                             provider=MockProvider(latency=0.0))
        gw1.complete("Persisted question about storage")
        gw1.store.close()

        gw2 = CachingGateway(config=CacheConfig(**cfg),
                             provider=MockProvider(latency=0.0))
        result = gw2.complete("Persisted question about storage")
        assert result.cached
        assert gw2.provider.call_count == 0


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #
def test_warm_and_invalidate():
    gw = build_offline_gateway(log_level="ERROR")
    gw.warm([("What is HTTP?", "HyperText Transfer Protocol.")])
    result = gw.complete("What is HTTP?")
    assert result.cached and "HyperText" in result.text
    gw.invalidate(result.entry_key)
    assert not gw.complete("What is HTTP?").cached


def test_report_shape():
    gw = build_offline_gateway(log_level="ERROR")
    gw.complete("Hello there")
    gw.complete("Hello there")
    report = gw.report()
    assert report["requests"] == 2
    assert report["hit_rate"] > 0
    assert "hits_by_layer" in report and "latency_ms" in report


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
