#!/usr/bin/env python3
"""Demo for llm_cache_gateway.

    python demo.py offline     # no API key, no network - MockProvider
    python demo.py live        # real OpenRouter call, free models
    python demo.py bench       # hit-rate / latency benchmark on a workload
    python demo.py models      # list which OpenRouter models are free today

`live` needs OPENROUTER_API_KEY. Nothing else is required: the chat model and
the embedding model are discovered at runtime, because free model IDs on
OpenRouter rotate week to week.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import List

from llm_cache_gateway import (CacheConfig, CachingGateway, MockProvider,
                               OpenRouterClient)

BOLD, DIM, GREEN, YELLOW, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[0m"


def header(title: str) -> None:
    print(f"\n{BOLD}{'=' * 68}\n{title}\n{'=' * 68}{RESET}")


def show(gateway: CachingGateway, prompt: str, **kwargs) -> None:
    result = gateway.complete(prompt, **kwargs)
    marker = f"{GREEN}HIT {result.source}{RESET}" if result.cached else f"{YELLOW}MISS -> llm{RESET}"
    sim = f" sim={result.similarity:.3f}" if result.similarity is not None else ""
    print(f"\n> {prompt}")
    print(f"  [{marker}{sim} {result.latency_ms:.1f}ms] {result.text[:110]}")
    if result.fewshot_used:
        print(f"  {DIM}few-shot: {result.fewshot_used} example(s) injected{RESET}")
    for step in result.trace:
        print(f"  {DIM}. {step['layer']:<12} "
              f"{'hit ' if step['hit'] else 'miss'} {step.get('detail', '')}{RESET}")


# --------------------------------------------------------------------------- #
def demo_offline() -> None:
    """Walk through every mechanism with a deterministic mock provider."""
    config = CacheConfig(
        backend="memory",
        embedding_backend="hashing",
        chat_model="mock/echo",
        similarity_threshold=0.55,       # calibrated for the hashing fallback
        keyword_min_score=0.45,
        keyword_requires_semantic_confirm=False,
        fewshot_min_similarity=0.60,
        conversation_summary_trigger_chars=2000,
        conversation_keep_recent_turns=4,
        log_level="ERROR",
    )
    provider = MockProvider(latency=0.15, canned={
        "summarize this conversation":
            "User walked through 10 numbered points; assistant "
            "acknowledged each. No decision recorded yet.",
    })
    gw = CachingGateway(config=config, provider=provider)

    header("2.1 / 2.6  Exact prompt + response cache")
    show(gw, "What is the capital of France?")
    show(gw, "What is the capital of France?")

    header("2.2  Prompt normalization  (casing, spacing, punctuation)")
    show(gw, "   what is the CAPITAL of France???  ")

    header("2.3  Keyword cache  (same intent, reordered wording)")
    show(gw, "How do I restart a docker container safely?")
    show(gw, "Safely restart a docker container - how?")

    header("2.4  Semantic cache  (paraphrase)")
    # The keyword layer would catch this one first; turn it off so the
    # semantic layer is the one visibly doing the work.
    gw.config.keyword_enabled = False
    show(gw, "How do I install Python packages with pip?")
    show(gw, "How can I install python packages using pip?")

    header("2.4  Semantic miss  (unrelated prompt goes to the model)")
    show(gw, "What is the boiling point of mercury?")
    gw.config.keyword_enabled = True

    header("2.7  Few-shot retrieval  (related, but not close enough to reuse)")
    # Reuse thresholds raised so nothing is served from cache; the
    # neighbours are injected as demonstrations instead.
    gw.set_similarity_threshold(0.99)
    gw.config.keyword_enabled = False
    show(gw, "How do I create a Kubernetes deployment?")
    show(gw, "How do I create a Kubernetes service?")
    show(gw, "How do I create a Kubernetes ingress?")
    gw.set_similarity_threshold(0.55)
    gw.config.keyword_enabled = True

    header("2.8  Conversation context cache")
    history = []
    for i in range(10):
        history.append({"role": "user", "content": f"Point {i}: " + "detail " * 25})
        history.append({"role": "assistant", "content": f"Noted {i}: " + "reply " * 25})
    history.append({"role": "user", "content": "So what did we decide overall?"})
    result = gw.complete(history, conversation_id="demo-conv")
    sent = gw.provider.calls[-1]["messages"]
    print(f"\n  history: {len(history)} messages -> {len(sent)} sent upstream "
          f"(compacted={result.conversation_compacted})")

    header("2.11  Adaptive policies")
    show(gw, "What is the latest AI news today?")
    show(gw, "What is the latest AI news today?")   # volatile: still a miss
    show(gw, "Write a haiku about rain", temperature=1.3)
    show(gw, "Write a haiku about rain", temperature=1.3)  # high temp: no reuse

    print(f"\n  adaptive threshold before feedback: {gw.policy.threshold:.4f}")
    first = gw.complete("How do I rotate API keys?")
    gw.feedback(first.entry_key, accepted=False)
    print(f"  after one rejected hit:             {gw.policy.threshold:.4f}")

    header("Report")
    print(json.dumps(gw.report(), indent=2))


# --------------------------------------------------------------------------- #
def demo_live() -> None:
    """Same pipeline against real free models on OpenRouter."""
    if not os.getenv("OPENROUTER_API_KEY"):
        sys.exit("Set OPENROUTER_API_KEY first (get one at openrouter.ai/keys).")

    config = CacheConfig.from_env(
        backend="disk",
        disk_path="./.llm_cache/live.sqlite3",
        embedding_backend="auto",     # OpenRouter -> sbert -> hashing
        similarity_threshold=0.90,
        log_level="INFO",
    )
    gw = CachingGateway(config=config)
    print(f"chat model:      {gw._default_model()}")
    print(f"embedding stack: {gw.embedder.name}")

    header("Cold call, then exact repeat, then a paraphrase")
    show(gw, "In two sentences, what is a vector database?")
    show(gw, "In two sentences, what is a vector database?")
    show(gw, "Briefly, what is a vector database? Two sentences please.")

    header("Report")
    print(json.dumps(gw.report(), indent=2))
    print(f"\n{DIM}Cache persisted at {config.disk_path} - "
          f"re-run to see hits survive the restart.{RESET}")


# --------------------------------------------------------------------------- #
def demo_models() -> None:
    """Show which models are free right now (the list rotates constantly)."""
    if not os.getenv("OPENROUTER_API_KEY"):
        sys.exit("Set OPENROUTER_API_KEY first.")
    client = OpenRouterClient(CacheConfig.from_env())
    free = client.free_models()
    print(f"{len(free)} free chat model(s) available right now:")
    for model_id in free[:25]:
        print(f"  - {model_id}")
    print(f"\nEmbedding model that would be used: {client.pick_embedding_model()}")


# --------------------------------------------------------------------------- #
def demo_bench() -> None:
    """Measure hit rate and latency on a workload with repeats and paraphrases."""
    base: List[str] = [
        "How do I install Python packages with pip?",
        "What is the difference between a list and a tuple in Python?",
        "How do I restart a docker container?",
        "Explain what a vector database is",
        "How do I write a for loop in Rust?",
    ]
    paraphrases: List[str] = [
        "How can I install python packages using pip?",
        "In Python, what is the difference between tuple and list?",
        "How can I restart docker containers?",
        "Explain vector databases to me",
        "How do I write for loops in Rust?",
    ]
    workload = (base * 3) + paraphrases + (base * 2)

    config = CacheConfig(
        backend="memory", embedding_backend="hashing", chat_model="mock/echo",
        similarity_threshold=0.55, keyword_requires_semantic_confirm=False,
        keyword_min_score=0.45, fewshot_enabled=False, log_level="ERROR",
    )
    gw = CachingGateway(config=config, provider=MockProvider(latency=0.4))

    started = time.perf_counter()
    for prompt in workload:
        gw.complete(prompt)
    wall = time.perf_counter() - started

    report = gw.report()
    baseline = len(workload) * 0.4
    header("Benchmark")
    print(f"  requests:            {len(workload)}")
    print(f"  upstream calls:      {report['llm_calls']}")
    print(f"  hit rate:            {report['hit_rate'] * 100:.1f}%")
    print(f"  hits by layer:       {report['hits_by_layer']}")
    print(f"  mean latency (hit):  {report['latency_ms']['mean_on_hit']} ms")
    print(f"  mean latency (miss): {report['latency_ms']['mean_on_miss']} ms")
    print(f"  wall clock:          {wall:.2f}s vs {baseline:.2f}s uncached "
          f"({baseline / wall:.1f}x faster)")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "offline"
    {"offline": demo_offline, "live": demo_live,
     "bench": demo_bench, "models": demo_models}.get(mode, demo_offline)()
