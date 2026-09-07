"""Tests for displacement metrics, regional stats and full shape comparison."""

from __future__ import annotations

import numpy as np
import pytest

from face_geometry.geometry import (
    compare_shapes,
    displacement_stats,
    euclidean_displacements,
    regional_displacements,
)
from face_geometry.regions import REGION_INDEX


class TestEuclideanDisplacements:
    def test_basic(self) -> None:
        a = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
        b = np.array([[0.0, 0.0], [1.0, 3.0], [4.0, 2.0]])
        disp = euclidean_displacements(a, b)
        np.testing.assert_allclose(disp, [0.0, 3.0, 4.0])

    def test_identical_is_zero(self, synthetic_face: np.ndarray) -> None:
        np.testing.assert_allclose(
            euclidean_displacements(synthetic_face, synthetic_face), 0.0
        )


class TestDisplacementStats:
    def test_values(self) -> None:
        stats = displacement_stats(np.array([0.0, 3.0, 4.0]))
        assert stats.mean == pytest.approx(7.0 / 3.0)
        assert stats.median == pytest.approx(3.0)
        assert stats.max == pytest.approx(4.0)
        assert stats.rmse == pytest.approx(np.sqrt(25.0 / 3.0))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            displacement_stats(np.array([]))


class TestRegionalDisplacements:
    def test_only_eyebrows_flagged(
        self, synthetic_face: np.ndarray, moved_eyebrows_face: np.ndarray
    ) -> None:
        """Moving only eyebrows must make eyebrows the dominant region."""
        disp = euclidean_displacements(synthetic_face, moved_eyebrows_face)
        regional = regional_displacements(disp)
        by_name = {r.region: r for r in regional}
        brow_mean = np.mean(
            [by_name["left_eyebrow"].mean, by_name["right_eyebrow"].mean]
        )
        other_means = [
            r.mean for r in regional if "eyebrow" not in r.region
        ]
        assert brow_mean > 0
        assert all(brow_mean > m for m in other_means)
        # Non-brow regions must be exactly unchanged in this synthetic setup.
        assert all(m == pytest.approx(0.0) for m in other_means)

    def test_all_regions_present(self, synthetic_face: np.ndarray) -> None:
        disp = np.zeros(len(synthetic_face))
        names = {r.region for r in regional_displacements(disp)}
        assert names == set(REGION_INDEX)


class TestCompareShapes:
    def test_identical_shapes(self, synthetic_face: np.ndarray) -> None:
        result = compare_shapes(synthetic_face, synthetic_face)
        assert result["mean_landmark_displacement"] == pytest.approx(0.0, abs=1e-12)
        assert result["landmark_rmse"] == pytest.approx(0.0, abs=1e-12)
        assert result["procrustes_distance"] == pytest.approx(0.0, abs=1e-12)

    def test_metric_relationships(self, synthetic_face: np.ndarray) -> None:
        moved = synthetic_face.copy()
        rng = np.random.default_rng(11)
        moved += rng.normal(scale=2.0, size=moved.shape)
        result = compare_shapes(synthetic_face, moved)
        assert 0.0 <= result["median_landmark_displacement"]
        assert result["mean_landmark_displacement"] <= result["max_landmark_displacement"]
        assert result["mean_landmark_displacement"] <= result["landmark_rmse"]
