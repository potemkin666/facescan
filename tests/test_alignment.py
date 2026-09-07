"""Tests for pose normalisation and Procrustes alignment invariance."""

from __future__ import annotations

import numpy as np
import pytest

from face_geometry.alignment import (
    ShapeMismatchError,
    normalize_pose,
    procrustes_align,
    procrustes_distance,
    reference_distance,
    rms_displacement,
)


def _transform(
    points: np.ndarray, scale: float, angle_deg: float, translate: np.ndarray
) -> np.ndarray:
    """Apply a similarity transform (scale, in-plane rotation, translation)."""

    theta = np.radians(angle_deg)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return points @ rotation.T * scale + translate


class TestNormalizePose:
    def test_unit_reference_distance(self, synthetic_face: np.ndarray) -> None:
        normalised, ref = normalize_pose(synthetic_face)
        assert ref == pytest.approx(100.0)
        left = normalised[33]
        right = normalised[263]
        assert np.linalg.norm(right - left) == pytest.approx(1.0)
        assert left[1] == pytest.approx(right[1], abs=1e-9)  # horizontal axis
        midpoint = (left + right) / 2.0
        assert np.linalg.norm(midpoint) == pytest.approx(0.0, abs=1e-9)

    def test_translation_invariance(self, synthetic_face: np.ndarray) -> None:
        a, _ = normalize_pose(synthetic_face)
        b, _ = normalize_pose(synthetic_face + np.array([517.0, -233.0]))
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_scale_invariance(self, synthetic_face: np.ndarray) -> None:
        a, _ = normalize_pose(synthetic_face)
        b, _ = normalize_pose(synthetic_face * 3.7 + 11.0)
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_rotation_invariance(self, synthetic_face: np.ndarray) -> None:
        a, _ = normalize_pose(synthetic_face)
        rotated = _transform(synthetic_face, 1.0, 37.0, np.array([3.0, -9.0]))
        b, _ = normalize_pose(rotated)
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_combined_similarity_invariance(
        self, synthetic_face: np.ndarray
    ) -> None:
        a, _ = normalize_pose(synthetic_face)
        moved = _transform(synthetic_face, 2.3, -18.0, np.array([401.0, 87.0]))
        b, _ = normalize_pose(moved)
        np.testing.assert_allclose(a, b, atol=1e-9)


class TestProcrustes:
    def test_identical_shapes_zero_distance(
        self, synthetic_face: np.ndarray
    ) -> None:
        assert procrustes_distance(synthetic_face, synthetic_face) == pytest.approx(
            0.0, abs=1e-12
        )

    def test_similarity_transformed_shapes_zero_distance(
        self, synthetic_face: np.ndarray
    ) -> None:
        moved = _transform(synthetic_face, 1.8, 24.0, np.array([-55.0, 120.0]))
        assert procrustes_distance(synthetic_face, moved) == pytest.approx(
            0.0, abs=1e-8
        )

    def test_alignment_recovers_shape(self, synthetic_face: np.ndarray) -> None:
        moved = _transform(synthetic_face, 0.6, -41.0, np.array([12.0, 7.0]))
        aligned, residual, _, _, _ = procrustes_align(synthetic_face, moved)
        assert residual == pytest.approx(0.0, abs=1e-8)
        np.testing.assert_allclose(aligned, synthetic_face, atol=1e-6)

    def test_reflection_rejected(self) -> None:
        rng = np.random.default_rng(7)
        base = rng.normal(size=(40, 2))
        reflected = base * np.array([-1.0, 1.0])
        # A proper rotation cannot undo a reflection: residual must be > 0.
        assert procrustes_distance(base, reflected) > 1e-3

    def test_shape_mismatch_raises(self, synthetic_face: np.ndarray) -> None:
        with pytest.raises(ShapeMismatchError):
            procrustes_align(synthetic_face, synthetic_face[:100])

    def test_degenerate_shape_raises(self) -> None:
        flat = np.zeros((10, 2))
        with pytest.raises(ShapeMismatchError):
            procrustes_align(flat, flat)

    def test_rms_displacement(self) -> None:
        a = np.array([[0.0, 0.0], [3.0, 4.0]])
        b = np.array([[0.0, 0.0], [0.0, 0.0]])
        assert rms_displacement(a, b) == pytest.approx(np.sqrt(12.5))


class TestReferenceDistance:
    def test_eye_corner_distance(self, synthetic_face: np.ndarray) -> None:
        assert reference_distance(synthetic_face) == pytest.approx(100.0)

    def test_fallback_when_degenerate(self) -> None:
        pts = np.random.default_rng(3).normal(size=(468, 2))
        pts[33] = pts[263]  # zero eye-corner distance -> bounding-box fallback
        assert reference_distance(pts) > 0.0
