"""llm_cache_gateway - gateway-level caching for LLM APIs.

Implements the black-box mechanisms from the design document: exact matching,
normalization, keyword retrieval, semantic retrieval, hierarchical layering,
response caching, few-shot retrieval, conversation compaction, and adaptive
policies. Model-internal techniques (prefix caching, KV reuse) are out of
scope by design, since a gateway only sees requests and responses.
"""

from .config import CacheConfig
from .context_cache import ConversationCache, FewShotCache
from .embedding import (BaseEmbedder, CachingEmbedder, HashingEmbedder,
                        OpenRouterEmbedder, SentenceTransformerEmbedder,
                        build_embedder)
from .exact_cache import ExactCache
from .gateway import CachingGateway, build_offline_gateway
from .keyword_cache import KeywordCache
from .keywords import CorpusStats, KeywordExtractor
from .normalize import PromptNormalizer
from .policy import AdaptivePolicy
from .providers import MockProvider, OpenRouterClient, ProviderError
from .repository import EntryRepository
from .semantic_cache import SemanticCache, VectorIndex
from .stats import GatewayStats
from .store import DiskStore, MemoryStore, RedisStore, build_store
from .types import CacheEntry, CacheRequest, GatewayResponse

__version__ = "1.0.0"

__all__ = [
    "CacheConfig", "CachingGateway", "build_offline_gateway",
    "OpenRouterClient", "MockProvider", "ProviderError",
    "PromptNormalizer", "KeywordExtractor", "CorpusStats",
    "ExactCache", "KeywordCache", "SemanticCache", "VectorIndex",
    "FewShotCache", "ConversationCache", "AdaptivePolicy",
    "EntryRepository", "GatewayStats",
    "MemoryStore", "DiskStore", "RedisStore", "build_store",
    "BaseEmbedder", "HashingEmbedder", "OpenRouterEmbedder",
    "SentenceTransformerEmbedder", "CachingEmbedder", "build_embedder",
    "CacheEntry", "CacheRequest", "GatewayResponse",
]
