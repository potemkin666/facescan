"""Pose normalisation and Procrustes alignment of landmark configurations.

Two coordinate systems are kept for every observation:

* **raw** — pixel coordinates as detected;
* **normalised** — similarity-transformed so that a stable reference segment
  (outer eye corner distance) has unit length, is centred on the origin and
  lies horizontal.  This removes translation, in-plane rotation (head tilt)
  and scale (camera distance) before any geometry comparison.

Procrustes analysis then provides *optimal* similarity alignment between any
two normalised shapes, with a residual distance that vanishes for identical
shapes.
"""

from __future__ import annotations

import numpy as np

# MediaPipe outer eye corners.
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


class ShapeMismatchError(ValueError):
    """Raised when two landmark sets do not have the same cardinality."""


def _check_shapes(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ShapeMismatchError(
            f"Landmark shape mismatch: {a.shape} vs {b.shape}"
        )


def reference_distance(landmarks: np.ndarray) -> float:
    """Stable normalisation distance: outer-eye-corner (bi-canthal) distance.

    Args:
        landmarks: array of shape (N, 2) or (N, 3).

    Returns:
        Euclidean distance between the outer eye corners. Falls back to the
        bounding-box diagonal if the eye corners are degenerate.
    """

    left = landmarks[LEFT_EYE_OUTER, :2]
    right = landmarks[RIGHT_EYE_OUTER, :2]
    distance = float(np.linalg.norm(right - left))
    if distance < 1e-12:
        spread = landmarks[:, :2].max(axis=0) - landmarks[:, :2].min(axis=0)
        distance = float(np.linalg.norm(spread)) or 1.0
    return distance


def normalize_pose(
    landmarks: np.ndarray,
    ref_distance: float | None = None,
) -> tuple[np.ndarray, float]:
    """Similarity-normalise landmarks.

    The transform is *translate to the eye-corner midpoint, rotate the
    eye-corner axis horizontal, scale by the reference distance*.

    Args:
        landmarks: array of shape (N, 2) with raw pixel coordinates.
        ref_distance: optional explicit normaliser; computed when omitted.

    Returns:
        ``(normalised, reference_distance)`` where ``normalised`` has shape
        (N, 2) and the eye-corner segment is centred, horizontal, unit length.
    """

    pts = np.asarray(landmarks, dtype=np.float64)[:, :2]
    left = pts[LEFT_EYE_OUTER]
    right = pts[RIGHT_EYE_OUTER]
    centre = (left + right) / 2.0
    axis = right - left
    angle = float(np.arctan2(axis[1], axis[0]))
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    centred = pts - centre
    rotated = centred @ rotation.T
    scale = ref_distance if ref_distance is not None else reference_distance(pts)
    if scale < 1e-12:
        scale = 1.0
    return rotated / scale, float(scale)


def procrustes_align(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    allow_scaling: bool = True,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float]:
    """Optimally similarity-align ``moving`` onto ``reference``.

    Solves the orthogonal Procrustes problem (translation + rotation +
    optional uniform scaling) with the classical Kabsch/SVD solution.

    Args:
        reference: target shape, (N, 2).
        moving: shape to align, (N, 2).
        allow_scaling: include uniform scaling in the similarity transform.

    Returns:
        Tuple ``(aligned, residual, translation, rotation, scale)`` where
        ``aligned`` is ``moving`` after alignment, ``residual`` is the RMS
        Procrustes distance, ``translation``/``rotation``/``scale`` are the
        applied similarity components.

    Raises:
        ShapeMismatchError: if the two shapes have different point counts.
    """

    reference = np.asarray(reference, dtype=np.float64)[:, :2]
    moving = np.asarray(moving, dtype=np.float64)[:, :2]
    _check_shapes(reference, moving)

    mu_ref = reference.mean(axis=0)
    mu_mov = moving.mean(axis=0)
    ref_c = reference - mu_ref
    mov_c = moving - mu_mov

    norm_ref = float(np.linalg.norm(ref_c))
    norm_mov = float(np.linalg.norm(mov_c))
    if norm_ref < 1e-12 or norm_mov < 1e-12:
        raise ShapeMismatchError("Degenerate (zero-variance) landmark shape")

    ref_n = ref_c / norm_ref
    mov_n = mov_c / norm_mov

    # Optimal rotation via SVD of the cross-covariance matrix.  For
    # row-vector shapes with ``moving @ rotation ~= reference`` and
    # ``ref_n.T @ mov_n = U S Vt``, the minimiser is ``Vt.T @ U.T``.
    u, _, vt = np.linalg.svd(ref_n.T @ mov_n)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        # Reject reflections: flip the last singular vector.
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    aligned_n = mov_n @ rotation
    scale = float(np.trace(ref_n.T @ aligned_n)) if allow_scaling else 1.0
    scaled = mov_n @ rotation * scale
    disparity = float(np.sum((ref_n - scaled) ** 2))
    residual = float(np.sqrt(max(disparity, 0.0)))

    # Map the unit-norm solution back onto the reference scale/origin.
    aligned = scaled * norm_ref + mu_ref
    translation = mu_ref - mu_mov @ rotation * (scale * norm_ref / norm_mov)
    return aligned, residual, translation, rotation, scale


def procrustes_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Residual shape difference after optimal similarity alignment.

    This is the square root of the Procrustes *disparity* between the two
    centred, unit-norm shapes; 0 means identical up to similarity.

    Args:
        a, b: shapes of equal size, (N, 2).

    Returns:
        Non-negative residual distance.
    """

    _, residual, _, _, _ = procrustes_align(a, b, allow_scaling=True)
    return residual


def rms_displacement(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square per-landmark Euclidean displacement."""

    a = np.asarray(a, dtype=np.float64)[:, :2]
    b = np.asarray(b, dtype=np.float64)[:, :2]
    _check_shapes(a, b)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
