from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelFeatureConfig(BaseModel):
    """
    Feature-based LLM model routing.

    Selection priority (highest to lowest):
    1. high_complexity  → use for ambiguous, deeply nested, multi-role segments
    2. standard         → default for most unknown segments
    3. fast             → simple segments with weak signals but not rule-matchable

    Complexity is determined by:
    - dom_depth > threshold
    - child_tag_counts diversity (many unique tags = complex)
    - text_density_ratio (very high or very low = complex)
    - sibling_count == 0 (one-off sections = complex)
    """
    high_complexity_model: str = "anthropic/claude-opus-4"
    standard_model: str = "anthropic/claude-sonnet-4-5"
    fast_model: str = "anthropic/claude-haiku-4-5"

    high_complexity_dom_depth_threshold: int = 6
    high_complexity_unique_tag_threshold: int = 8
    fast_model_max_dom_depth: int = 3


class CacheConfig(BaseModel):
    l1_cache_path: str = ".cache/l1_fingerprints.json"
    l2_cache_path: str = ".cache/l2_clusters.json"
    l2_embeddings_path: str = ".cache/l2_embeddings.npy"
    l2_similarity_threshold: float = 0.85
    l2_max_cluster_size: int = 50
    persist_on_update: bool = True


class ClassifierSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CLASSIFIER_")

    # LiteLLM
    litellm_api_key: str = ""
    litellm_batch_size: int = 20         # max segments per LLM batch call
    litellm_max_concurrent_batches: int = 5
    litellm_timeout_seconds: int = 60

    # Pipeline
    rule_based_confidence_threshold: float = 0.90
    l1_min_confidence: float = 0.85
    l2_min_confidence: float = 0.75

    model_routing: ModelFeatureConfig = ModelFeatureConfig()
    cache: CacheConfig = CacheConfig()
