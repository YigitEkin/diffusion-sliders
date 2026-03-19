"""Steering vector utilities shared across all models."""

from steering.vectors import (
    DTYPE_MAP,
    compute_difference_of_means,
    find_all_spans,
    load_steering_vector,
    pool_tokens,
    save_steering_outputs,
    split_style_terms,
    validate_max_pairs,
    validate_path_exists,
)
from steering.elastic_band import (
    ElasticBandConfig,
    ElasticBandRunner,
    batched,
    canonical_strength,
    cosine_step,
    elastic_band_search,
    find_effective_minimum,
    load_min_projection_value,
    summarize_valid_range,
)

__all__ = [
    # vectors
    "DTYPE_MAP",
    "compute_difference_of_means",
    "find_all_spans",
    "load_steering_vector",
    "pool_tokens",
    "save_steering_outputs",
    "split_style_terms",
    "validate_max_pairs",
    "validate_path_exists",
    # elastic_band
    "ElasticBandConfig",
    "ElasticBandRunner",
    "batched",
    "canonical_strength",
    "cosine_step",
    "elastic_band_search",
    "find_effective_minimum",
    "load_min_projection_value",
    "summarize_valid_range",
]
