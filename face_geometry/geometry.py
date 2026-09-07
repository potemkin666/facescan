"""Pairwise geometric comparison of pose-normalised landmark shapes.

All functions operate on *normalised* coordinates produced by
:mod:`face_geometry.alignment`, so displacement values are expressed in
units of the stable reference distance (outer eye corner distance).
"""

from __future__ import annotations

import numpy as np

from .alignment import _check_shapes, procrustes_distance
from .models import DisplacementStats, RegionalStats
from .regions import REGIONS


def euclidean_displacements(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-landmark Euclidean displacement between two equal shapes.

    Args:
        a, b: arrays of shape (N, D) with corresponding points.

    Returns:
        Array of shape (N,) with ``||a_i - b_i||``.
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    _check_shapes(a, b)
    return np.linalg.norm(a - b, axis=1)


def displacement_stats(displacements: np.ndarray) -> DisplacementStats:
    """Summarise per-landmark displacements.

    Args:
        displacements: 1-D array of non-negative distances.

    Returns:
        Mean, median, max and root-mean-square displacement.

    Raises:
        ValueError: if the array is empty.
    """

    values = np.asarray(displacements, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot summarise an empty displacement array")
    return DisplacementStats(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        max=float(np.max(values)),
        rmse=float(np.sqrt(np.mean(values**2))),
    )


def regional_displacements(
    displacements: np.ndarray,
) -> tuple[RegionalStats, ...]:
    """Aggregate per-landmark displacements into per-region statistics."""

    values = np.asarray(displacements, dtype=np.float64)
    out: list[RegionalStats] = []
    for spec in REGIONS:
        idx = np.fromiter(spec.indices, dtype=np.intp)
        stats = displacement_stats(values[idx])
        out.append(
            RegionalStats(
                region=spec.name,
                mean=stats.mean,
                median=stats.median,
                max=stats.max,
                rmse=stats.rmse,
            )
        )
    return tuple(out)


def compare_shapes(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Compute the full scalar comparison between two normalised shapes.

    The shapes are assumed already similarity-normalised; an additional
    (rotation-only refinement is unnecessary) Procrustes distance is computed
    with scaling allowed, giving the residual shape difference after optimal
    translation/rotation/scale.

    Returns:
        Dictionary with mean/median/max/RMSE displacement and Procrustes
        distance.
    """

    disp = euclidean_displacements(a, b)
    stats = displacement_stats(disp)
    return {
        "mean_landmark_displacement": stats.mean,
        "median_landmark_displacement": stats.median,
        "max_landmark_displacement": stats.max,
        "landmark_rmse": stats.rmse,
        "procrustes_distance": procrustes_distance(a, b),
    }
