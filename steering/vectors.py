#!/usr/bin/env python3
"""Core steering vector utilities shared across all models.

Provides:
- Difference-of-means computation and persistence
- Token-span alignment helpers (find_all_spans, split_style_terms, pool_tokens)
- Argparse validators reused in each model's CLI
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

EPS = 1e-8
Span = Tuple[int, int]

DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


# ---------------------------------------------------------------------------
# Token-span helpers
# ---------------------------------------------------------------------------


def find_all_spans(text: str, needle: str) -> List[Span]:
    """Return character spans for every case-insensitive occurrence of *needle* in *text*."""
    spans: List[Span] = []
    lowered = text.lower()
    target = (needle or "").lower()
    if not target:
        return spans
    start = 0
    while True:
        idx = lowered.find(target, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(target)))
        start = idx + 1
    return spans


def split_style_terms(style: str) -> List[str]:
    """Split a style identifier into searchable terms (e.g. ``clean_shaved`` → ``['clean', 'shaved']``)."""
    return [part for part in re.split(r"[_\s]+", (style or "").strip()) if part]


def pool_tokens(hidden_states: torch.Tensor, token_indices: Sequence[int]) -> Optional[torch.Tensor]:
    """Mean-pool rows of *hidden_states* at *token_indices*, returning ``None`` if indices are empty.

    Works for both 1-D index lists into a ``[seq_len, hidden]`` tensor and a leading batch dim
    (only the first batch element is used when ``hidden_states`` is 2-D and indices address
    positions directly).
    """
    if not token_indices:
        return None
    vectors = [hidden_states[idx].to(dtype=torch.float32).cpu() for idx in token_indices]
    return torch.stack(vectors).mean(0)


# ---------------------------------------------------------------------------
# Difference-of-means
# ---------------------------------------------------------------------------


def compute_difference_of_means(
    vectors: Sequence[np.ndarray],
    labels: Sequence[int],
) -> Tuple[np.ndarray, float, float]:
    """Compute a normalized difference-of-means steering vector.

    Args:
        vectors: Pooled embeddings, alternating positive/negative.
        labels:  Binary labels — 1 for positive, 0 for negative.

    Returns:
        ``(steering, max_projection, min_projection)`` where *steering* is the
        unit-norm DoM direction and the projection values are the per-class
        extrema used for elastic-band range estimation.
    """
    if not vectors or not labels:
        raise ValueError("No vectors provided for steering computation.")

    X = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Input vectors contain non-finite values.")
    y = np.asarray(labels, dtype=np.int32)
    if len(X) == 0 or np.all(y == 0) or np.all(y == 1):
        raise ValueError("Steering computation requires both positive and negative examples.")

    pos_mean = X[y == 1].mean(axis=0)
    neg_mean = X[y == 0].mean(axis=0)
    dom = (pos_mean - neg_mean).astype(np.float32)
    if not np.isfinite(dom).all():
        raise ValueError("Difference-of-means produced non-finite values.")

    norm = float(np.linalg.norm(dom))
    if not np.isfinite(norm) or norm < EPS:
        raise ValueError("Steering vector norm is too small; check that embeddings are well-conditioned.")

    steering = (dom / norm).astype(np.float32)
    projections = X @ steering  # project onto unit-norm direction for meaningful scale
    if not np.isfinite(projections).all():
        raise ValueError("Projection values are non-finite.")

    max_projection = float(projections[y == 1].max())
    min_projection = float(projections[y == 0].min())
    return steering, max_projection, min_projection


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_steering_outputs(
    out_dir: Path,
    steering: np.ndarray,
    max_projection: float,
    min_projection: float,
) -> None:
    """Save the steering vector and projection statistics to *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "steering_last_layer.npy", steering.astype(np.float32))
    np.save(out_dir / "max_projection_value.npy", np.array(max_projection, dtype=np.float32))
    np.save(out_dir / "min_projection_value.npy", np.array(min_projection, dtype=np.float32))
    print(f"Saved steering vector → {out_dir / 'steering_last_layer.npy'}")
    print(f"Saved max projection  → {out_dir / 'max_projection_value.npy'}")
    print(f"Saved min projection  → {out_dir / 'min_projection_value.npy'}")


def load_steering_vector(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Load a ``.npy`` steering vector and return it as a float32 tensor on *device*."""
    return torch.tensor(np.load(Path(path)), dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Argparse validators (reused across model CLIs)
# ---------------------------------------------------------------------------


def validate_path_exists(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Path does not exist: {path}")
    return path


def validate_max_pairs(value: str) -> int:
    max_pairs = int(value)
    if max_pairs <= 0 and max_pairs != -1:
        raise argparse.ArgumentTypeError("--max_pairs must be -1 (all) or a positive integer.")
    return max_pairs
