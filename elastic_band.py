#!/usr/bin/env python3
"""Model-agnostic elastic-band search for steering strength ranges.

The elastic-band algorithm adaptively places control points along a 1-D strength
axis [a_min, 0] so that consecutive images are roughly equidistant in perceptual
space (DreamSim distance ≈ TARGET_GAP). It then reports the largest contiguous
interval where every image stays within MAX_DREAMSIM_DISTANCE of the unsteered
reference, giving a practical operating range for each concept vector.

Usage
-----
Each model implements a runner that satisfies the ``ElasticBandRunner`` protocol
(i.e. exposes ``generate_images``, ``reference_distance``, and ``pair_distance``).
Instantiate an ``ElasticBandConfig`` with model-specific hyperparameters and call
``find_effective_minimum`` followed by ``elastic_band_search``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Hyperparameter config
# ---------------------------------------------------------------------------


@dataclass
class ElasticBandConfig:
    """All hyperparameters for one elastic-band run.

    Defaults match the Flux 2 setting; override per model as needed.
    """

    # Validity threshold: a strength is "valid" if its DreamSim distance to the
    # unsteered reference is below this value.
    max_dreamsim_distance: float = 0.2

    # How many times to double the initial minimum strength when probing for an
    # effective starting point.
    max_doubling_steps: int = 3

    # Initial and maximum number of control points on the strength axis.
    starting_number_of_points: int = 8
    maximum_number_of_points: int = 24

    # Stopping criteria.
    maximum_number_of_iterations: int = 25

    # A gap is "target-sized" when DreamSim(left, right) ≈ this value.
    target_gap: float = 0.01

    # Insert a midpoint when the largest gap exceeds target_gap * (1 + expand_threshold).
    expand_threshold: float = 0.05

    # Gap-imbalance amplification for step sizing.
    lam: float = 1.0

    # Minimum separation enforced between any two neighbouring control points.
    epsilon: float = 0.1

    # Fraction of epsilon used as a minimum meaningful move.
    move_fraction: float = 1.0

    # Step size = BASE_STEP_FRACTION * interval_width * cosine_decay.
    base_step_fraction: float = 0.02

    # A move is only accepted if it exceeds this absolute threshold.
    min_meaningful_move: float = 0.25

    # Skip moving a point if its max local gap is below this normalised value.
    min_normalized_gap_for_move: float = 0.08

    # Skip moving a point if its gap imbalance is below this value.
    min_gap_imbalance_for_move: float = 0.01

    # Images generated per forward-pass batch.
    inference_batch_size: int = 4

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ElasticBandConfig":
        """Load an ElasticBandConfig from the ``elastic_band`` section of a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data["elastic_band"])

    def to_dict(self) -> dict:
        return {
            "starting_number_of_points": self.starting_number_of_points,
            "maximum_number_of_points": self.maximum_number_of_points,
            "maximum_number_of_iterations": self.maximum_number_of_iterations,
            "target_gap": self.target_gap,
            "expand_threshold": self.expand_threshold,
            "lam": self.lam,
            "epsilon": self.epsilon,
            "move_fraction": self.move_fraction,
            "base_step_fraction": self.base_step_fraction,
            "min_meaningful_move": self.min_meaningful_move,
            "min_normalized_gap_for_move": self.min_normalized_gap_for_move,
            "min_gap_imbalance_for_move": self.min_gap_imbalance_for_move,
            "inference_batch_size": self.inference_batch_size,
        }


# ---------------------------------------------------------------------------
# Runner protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ElasticBandRunner(Protocol):
    """Interface that every model-specific runner must satisfy."""

    def generate_images(
        self,
        concept_dir: Path,
        concept_name: str,
        steering_vector: torch.Tensor,
        strengths: list[float],
    ) -> None:
        """Generate and cache images for each strength value."""
        ...

    def reference_distance(
        self,
        concept_dir: Path,
        concept_name: str,
        steering_vector: torch.Tensor,
        strength: float,
    ) -> float:
        """Return DreamSim distance between the image at *strength* and the unsteered reference."""
        ...

    def pair_distance(
        self,
        concept_dir: Path,
        concept_name: str,
        steering_vector: torch.Tensor,
        left: float,
        right: float,
    ) -> float:
        """Return DreamSim distance between images at *left* and *right* strengths."""
        ...


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def canonical_strength(value: float) -> float:
    """Round a strength value to 1 decimal place for stable cache keys."""
    return float(f"{value:.1f}")


def batched(items: list[float], batch_size: int) -> list[list[float]]:
    """Split *items* into successive chunks of at most *batch_size*."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def cosine_step(
    iteration: int,
    max_iterations: int,
    interval_width: float,
    config: ElasticBandConfig,
) -> float:
    """Cosine-annealed step size, clamped to at least *config.epsilon*."""
    if max_iterations <= 1:
        cosine = 1.0
    else:
        cosine = 0.5 * (1.0 + math.cos(math.pi * (iteration - 1) / (max_iterations - 1)))
    return max(config.epsilon, config.base_step_fraction * interval_width * cosine)


def load_min_projection_value(concept_path: Path) -> float:
    """Load the stored minimum projection value and ensure it is negative."""
    value = -abs(float(np.load(concept_path / "min_projection_value.npy")))
    if value == 0.0:
        raise ValueError(
            f"Minimum projection value must be non-zero: {concept_path / 'min_projection_value.npy'}"
        )
    return value


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def find_effective_minimum(
    runner: ElasticBandRunner,
    concept_dir: Path,
    concept_name: str,
    steering_vector: torch.Tensor,
    initial_min: float,
    config: ElasticBandConfig,
) -> dict:
    """Probe for the largest valid minimum strength by doubling from *initial_min*.

    Starting at ``initial_min`` (expected to be negative), the function checks
    whether the image at that strength is still within ``config.max_dreamsim_distance``
    of the reference. If so, it doubles the magnitude and tries again, up to
    ``config.max_doubling_steps`` times.

    Returns a dict with keys:
        - ``effective_minimum_value``: the accepted strength (0.0 if none passed)
        - ``search_minimum_value``: the value to use as ``a_min`` for the band search
        - ``doubling_attempts``: per-attempt trace list
    """
    attempts = []
    current = canonical_strength(initial_min)
    accepted = None

    for _ in range(config.max_doubling_steps + 1):
        distance = runner.reference_distance(concept_dir, concept_name, steering_vector, current)
        attempts.append({"strength": current, "dreamsim_to_reference": distance})
        if distance <= config.max_dreamsim_distance:
            accepted = current
            current = canonical_strength(current * 2.0)
        else:
            break

    search_minimum_value = accepted if accepted is not None else canonical_strength(initial_min)
    if accepted is None:
        accepted = 0.0

    return {
        "initial_min_projection_value": canonical_strength(initial_min),
        "effective_minimum_value": accepted,
        "search_minimum_value": search_minimum_value,
        "doubling_attempts": attempts,
    }


def elastic_band_search(
    runner: ElasticBandRunner,
    concept_dir: Path,
    concept_name: str,
    steering_vector: torch.Tensor,
    a_min: float,
    a_max: float,
    config: ElasticBandConfig,
) -> dict:
    """Adaptively redistribute control points on [a_min, a_max] for even perceptual spacing.

    Each iteration either:
    - **expands**: inserts a midpoint in the largest gap if it exceeds the target, or
    - **moves**: shifts interior points toward the larger of their two neighbouring gaps.

    Terminates when no point moves or ``config.maximum_number_of_iterations`` is reached.

    Returns a dict with keys:
        - ``stop_reason``: ``"no_point_moved"``, ``"max_iterations_reached"``, or
          ``"degenerate_interval"`` / ``"no_neighbor_gaps"``
        - ``final_control_points``: the last set of strength values
        - ``valid_control_points``: subset within ``config.max_dreamsim_distance``
        - ``reference_distances``: ``{strength_str: distance}`` for all final points
        - ``iterations``: per-iteration trace list
    """
    if math.isclose(a_min, a_max, abs_tol=config.epsilon):
        value = canonical_strength(a_max)
        distance = runner.reference_distance(concept_dir, concept_name, steering_vector, value)
        return {
            "stop_reason": "degenerate_interval",
            "iterations": [],
            "final_control_points": [value],
            "valid_control_points": [value],
            "reference_distances": {f"{value:+.6f}": distance},
        }
    if a_min >= a_max:
        raise ValueError(f"Expected a_min < a_max, got {a_min} >= {a_max}")

    control_points = [
        canonical_strength(v)
        for v in np.linspace(a_min, a_max, config.starting_number_of_points, dtype=np.float64).tolist()
    ]
    move_threshold = max(config.move_fraction * config.epsilon, config.min_meaningful_move)
    interval_width = abs(a_max - a_min)
    trace_iterations = []
    stop_reason = "max_iterations_reached"

    for iteration in range(1, config.maximum_number_of_iterations + 1):
        runner.generate_images(concept_dir, concept_name, steering_vector, control_points)

        reference_distances = [
            {
                "strength": value,
                "dreamsim_to_reference": runner.reference_distance(
                    concept_dir, concept_name, steering_vector, value
                ),
            }
            for value in control_points
        ]
        raw_gaps = []
        normalized_gaps = []
        for left, right in zip(control_points[:-1], control_points[1:]):
            gap = runner.pair_distance(concept_dir, concept_name, steering_vector, left, right)
            raw_gaps.append({"left": left, "right": right, "dreamsim_gap": gap})
            normalized_gaps.append(gap / config.target_gap)

        if not normalized_gaps:
            trace_iterations.append(
                {
                    "iteration": iteration,
                    "action": "stop",
                    "reason": "no_neighbor_gaps",
                    "control_points": list(control_points),
                    "reference_distances": reference_distances,
                    "neighbor_gaps": raw_gaps,
                    "normalized_neighbor_gaps": [],
                }
            )
            stop_reason = "no_neighbor_gaps"
            break

        largest_gap_index = int(np.argmax(normalized_gaps))
        if (
            normalized_gaps[largest_gap_index] > 1.0 + config.expand_threshold
            and len(control_points) < config.maximum_number_of_points
        ):
            midpoint = canonical_strength(
                0.5 * (control_points[largest_gap_index] + control_points[largest_gap_index + 1])
            )
            if midpoint not in control_points:
                trace_iterations.append(
                    {
                        "iteration": iteration,
                        "action": "expand",
                        "control_points_before": list(control_points),
                        "reference_distances": reference_distances,
                        "neighbor_gaps": raw_gaps,
                        "normalized_neighbor_gaps": normalized_gaps,
                        "largest_gap_index": largest_gap_index,
                        "inserted_midpoint": midpoint,
                    }
                )
                control_points = (
                    control_points[: largest_gap_index + 1]
                    + [midpoint]
                    + control_points[largest_gap_index + 1 :]
                )
                continue

        moved = False
        base_step = cosine_step(iteration, config.maximum_number_of_iterations, interval_width, config)
        updated_points = list(control_points)
        move_updates = []
        updated_from_left = [control_points[0]]

        for index in range(1, len(control_points) - 1):
            left_bound = updated_from_left[-1] + config.epsilon
            right_bound = control_points[index + 1] - config.epsilon
            if left_bound > right_bound:
                right_bound = left_bound

            left_gap = normalized_gaps[index - 1]
            right_gap = normalized_gaps[index]
            gap_imbalance = abs(left_gap - right_gap)
            max_local_gap = max(left_gap, right_gap)

            if (
                max_local_gap < config.min_normalized_gap_for_move
                or gap_imbalance < config.min_gap_imbalance_for_move
            ):
                current_value = canonical_strength(min(max(control_points[index], left_bound), right_bound))
                updated_points[index] = current_value
                updated_from_left.append(current_value)
                continue

            direction = -1.0 if left_gap > right_gap else 1.0
            step = base_step * (1.0 + config.lam * gap_imbalance)
            proposed_value = control_points[index] + direction * step
            new_value = canonical_strength(min(max(proposed_value, left_bound), right_bound))

            if abs(new_value - control_points[index]) >= move_threshold:
                move_updates.append(
                    {
                        "index": index,
                        "old_value": control_points[index],
                        "new_value": new_value,
                        "left_gap_normalized": left_gap,
                        "right_gap_normalized": right_gap,
                        "gap_imbalance": gap_imbalance,
                        "max_local_gap": max_local_gap,
                        "direction": "left" if direction < 0 else "right",
                        "step_size": step,
                    }
                )
                updated_points[index] = new_value
                moved = True
                updated_from_left.append(new_value)
            else:
                current_value = canonical_strength(min(max(control_points[index], left_bound), right_bound))
                updated_points[index] = current_value
                updated_from_left.append(current_value)

        updated_points[-1] = control_points[-1]
        trace_iterations.append(
            {
                "iteration": iteration,
                "action": "move" if moved else "stop",
                "reason": None if moved else "no_point_moved",
                "control_points_before": list(control_points),
                "control_points_after": list(updated_points),
                "reference_distances": reference_distances,
                "neighbor_gaps": raw_gaps,
                "normalized_neighbor_gaps": normalized_gaps,
                "base_step": base_step,
                "move_threshold": move_threshold,
                "updates": move_updates,
            }
        )
        control_points = updated_points
        if not moved:
            stop_reason = "no_point_moved"
            break

    valid_points = []
    final_reference_distances = {}
    for value in control_points:
        distance = runner.reference_distance(concept_dir, concept_name, steering_vector, value)
        final_reference_distances[f"{value:+.6f}"] = distance
        if distance <= config.max_dreamsim_distance:
            valid_points.append(canonical_strength(value))

    return {
        "stop_reason": stop_reason,
        "iterations": trace_iterations,
        "final_control_points": list(control_points),
        "valid_control_points": sorted(set(valid_points)),
        "reference_distances": final_reference_distances,
    }


def summarize_valid_range(valid_control_points: list[float]) -> dict:
    """Extract the ``[minimum, 0.0]`` valid range from the search result.

    Raises ``RuntimeError`` if the result is degenerate (no valid range found or
    parameters were too strict).
    """
    unique = sorted(set(canonical_strength(v) for v in valid_control_points))
    if 0.0 not in unique:
        raise RuntimeError("Expected 0.0 to remain a valid control point, but it was not valid.")
    if unique == [0.0]:
        raise RuntimeError(
            "No valid steering range found — parameters may be too strict. "
            "Try increasing max_dreamsim_distance or max_doubling_steps."
        )
    return {
        "minimum_valid_value": unique[0],
        "maximum_valid_value": 0.0,
    }
