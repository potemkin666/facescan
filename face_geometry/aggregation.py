"""Aggregation across pairings, consensus shapes and the change score.

The score documented here is an *engineering summary metric*, NOT a
probability:

    geometry_change_score = 100 * clip((d95 - D0) / (D1 - D0), 0, 1)

where ``d95`` is the 95th percentile of per-landmark mean displacement
computed between the two consensus shapes (in units of the stable reference
distance, after Procrustes-style normalisation), ``D0 = 0.002`` is the noise
floor (0.2% of the reference distance, below typical landmark jitter), and
``D1 = 0.10`` is the saturation point (10% of the reference distance, an
extremely large geometry difference).  Therefore:

* 0   = virtually identical detected geometry after normalisation;
* 100 = extremely large detected geometry difference.

All underlying raw measurements (percentiles, consensus displacement field)
remain available in the outputs so the score is fully auditable.
"""

from __future__ import annotations

import numpy as np

from .alignment import procrustes_align
from .geometry import displacement_stats, regional_displacements
from .models import (
    ConsensusComparison,
    DisplacementStats,
    PairwiseResult,
    RatioChange,
)

#: Engineering-score calibration constants (reference-distance units).
SCORE_NOISE_FLOOR = 0.002
SCORE_SATURATION = 0.10


def consensus_shape(shapes: list[np.ndarray]) -> np.ndarray:
    """Average landmark configuration of a folder (Fréchet mean under
    similarity, approximated by one Procrustes alignment pass to the median
    shape followed by arithmetic mean).

    Args:
        shapes: non-empty list of normalised shapes, each (N, 2).

    Returns:
        Consensus shape (N, 2).

    Raises:
        ValueError: if ``shapes`` is empty.
    """

    if not shapes:
        raise ValueError("Cannot build a consensus shape from zero shapes")
    stack = np.stack([np.asarray(s, dtype=np.float64)[:, :2] for s in shapes])
    if len(shapes) == 1:
        return stack[0].copy()
    # Align everything to the median shape (robust to outlier poses).
    median_shape = np.median(stack, axis=0)
    aligned = np.stack(
        [procrustes_align(median_shape, s, allow_scaling=False)[0] for s in stack]
    )
    return aligned.mean(axis=0)


def compare_consensus(
    baseline_shapes: list[np.ndarray], makeup_shapes: list[np.ndarray]
) -> ConsensusComparison:
    """Compare the baseline consensus shape against the makeup consensus."""

    base = consensus_shape(baseline_shapes)
    make = consensus_shape(makeup_shapes)
    # Both consensus shapes are already similarity-normalised to a unit
    # reference distance; use rigid (rotation+translation) alignment only so
    # re-scaling cannot mask a genuine size/shape change.
    aligned, residual, _, _, _ = procrustes_align(base, make, allow_scaling=False)
    disp = np.linalg.norm(aligned - base, axis=1)
    stats = displacement_stats(disp)
    return ConsensusComparison(
        mean_landmark_displacement=stats.mean,
        median_landmark_displacement=stats.median,
        max_landmark_displacement=stats.max,
        landmark_rmse=stats.rmse,
        procrustes_distance=residual,
        regional=regional_displacements(disp),
        per_landmark_displacement=disp,
        baseline_consensus_shape=base,
        makeup_consensus_shape=make,
        aligned_makeup_consensus=aligned,
    )


def summarize_pairwise(values: list[float]) -> dict[str, float]:
    """Distribution summary used for every pairwise metric."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("No pairwise results to summarise")
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def aggregate_pairwise(results: list[PairwiseResult]) -> dict[str, dict[str, float]]:
    """Aggregate scalar metrics across all baseline×makeup pairings."""

    if not results:
        raise ValueError("No pairwise results to aggregate")
    metrics = (
        "mean_landmark_displacement",
        "median_landmark_displacement",
        "max_landmark_displacement",
        "landmark_rmse",
        "procrustes_distance",
        "comparison_quality",
    )
    return {
        m: summarize_pairwise([float(getattr(r, m)) for r in results])
        for m in metrics
    }


def aggregate_regional(
    results: list[PairwiseResult],
) -> dict[str, dict[str, float]]:
    """Aggregate per-region mean/RMSE displacement across all pairings."""

    if not results:
        raise ValueError("No pairwise results to aggregate")
    regions = sorted({s.region for r in results for s in r.regional})
    out: dict[str, dict[str, float]] = {}
    for region in regions:
        means: list[float] = []
        rmses: list[float] = []
        medians: list[float] = []
        maxs: list[float] = []
        for r in results:
            for s in r.regional:
                if s.region == region:
                    means.append(s.mean)
                    rmses.append(s.rmse)
                    medians.append(s.median)
                    maxs.append(s.max)
        out[region] = {
            "mean_displacement": float(np.mean(means)),
            "median_displacement": float(np.mean(medians)),
            "max_displacement": float(np.mean(maxs)),
            "rmse": float(np.mean(rmses)),
        }
    return out


def aggregate_ratio_changes(
    baseline_ratios: list[dict[str, float]],
    makeup_ratios: list[dict[str, float]],
) -> list[RatioChange]:
    """Compute baseline_mean/makeup_mean/absolute/percentage change per ratio."""

    names = sorted({k for r in baseline_ratios + makeup_ratios for k in r})
    changes: list[RatioChange] = []
    for name in names:
        b = np.array(
            [r[name] for r in baseline_ratios if name in r], dtype=np.float64
        )
        m = np.array([r[name] for r in makeup_ratios if name in r], dtype=np.float64)
        b = b[np.isfinite(b)]
        m = m[np.isfinite(m)]
        if b.size == 0 or m.size == 0:
            continue
        b_mean = float(np.mean(b))
        m_mean = float(np.mean(m))
        absolute = m_mean - b_mean
        percentage = (
            float(100.0 * absolute / b_mean) if abs(b_mean) > 1e-12 else float("nan")
        )
        changes.append(
            RatioChange(
                name=name,
                baseline_mean=b_mean,
                makeup_mean=m_mean,
                absolute_change=absolute,
                percentage_change=percentage,
            )
        )
    return changes


def geometry_change_score(consensus: ConsensusComparison) -> float:
    """0-100 engineering summary of detected geometry change (NOT a probability).

    Definition (reference-distance units, consensus comparison):

        d95   = 95th percentile of the per-landmark displacement field
        score = 100 * clip((d95 - D0) / (D1 - D0), 0, 1)
        D0    = 0.002 (noise floor),  D1 = 0.10 (saturation)

    0 = virtually identical detected geometry after normalisation;
    100 = extremely large detected geometry difference.
    """

    disp = np.asarray(consensus.per_landmark_displacement, dtype=np.float64)
    if disp.size == 0:
        raise ValueError("Consensus comparison lacks per-landmark displacement")
    d95 = float(np.percentile(disp, 95))
    scaled = (d95 - SCORE_NOISE_FLOOR) / (SCORE_SATURATION - SCORE_NOISE_FLOOR)
    return float(100.0 * np.clip(scaled, 0.0, 1.0))


def score_details(consensus: ConsensusComparison) -> dict[str, float]:
    """Expose every intermediate quantity behind :func:`geometry_change_score`."""

    disp = np.asarray(consensus.per_landmark_displacement, dtype=np.float64)
    stats: DisplacementStats = displacement_stats(disp)
    return {
        "noise_floor": SCORE_NOISE_FLOOR,
        "saturation": SCORE_SATURATION,
        "d95": float(np.percentile(disp, 95)),
        "mean_displacement": stats.mean,
        "median_displacement": stats.median,
        "max_displacement": stats.max,
        "rmse": stats.rmse,
        "procrustes_distance": consensus.procrustes_distance,
        "geometry_change_score": geometry_change_score(consensus),
    }
