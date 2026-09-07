"""Tests for consensus shapes, pairwise aggregation and the change score."""

from __future__ import annotations

import numpy as np
import pytest

from face_geometry.aggregation import (
    SCORE_NOISE_FLOOR,
    SCORE_SATURATION,
    aggregate_pairwise,
    aggregate_ratio_changes,
    aggregate_regional,
    compare_consensus,
    consensus_shape,
    geometry_change_score,
    score_details,
    summarize_pairwise,
)
from face_geometry.geometry import euclidean_displacements, regional_displacements
from face_geometry.models import PairwiseResult


def _make_result(mean: float, rmse: float, face: np.ndarray) -> PairwiseResult:
    disp = euclidean_displacements(face, face)
    return PairwiseResult(
        baseline_path="b.jpg",
        makeup_path="m.jpg",
        mean_landmark_displacement=mean,
        median_landmark_displacement=mean,
        max_landmark_displacement=mean,
        landmark_rmse=rmse,
        procrustes_distance=rmse,
        comparison_quality=1.0,
        quality_flags=(),
        regional=regional_displacements(disp),
    )


class TestConsensusShape:
    def test_single_shape(self, synthetic_face: np.ndarray) -> None:
        np.testing.assert_allclose(consensus_shape([synthetic_face]), synthetic_face)

    def test_identical_shapes(self, synthetic_face: np.ndarray) -> None:
        result = consensus_shape([synthetic_face] * 5)
        np.testing.assert_allclose(result, synthetic_face, atol=1e-9)

    def test_averaging_two_shapes(self) -> None:
        rng = np.random.default_rng(5)
        base = rng.normal(size=(468, 2))
        shifted = base.copy()
        shifted[:, 0] += 4.0  # pure translation, removed by Procrustes
        consensus = consensus_shape([base, shifted])
        # Consensus shape must equal base up to translation (centres differ).
        np.testing.assert_allclose(
            consensus - consensus.mean(axis=0),
            base - base.mean(axis=0),
            atol=1e-8,
        )

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            consensus_shape([])


class TestCompareConsensus:
    def test_identical_folders_zero_change(
        self, synthetic_face: np.ndarray
    ) -> None:
        comp = compare_consensus([synthetic_face], [synthetic_face])
        assert comp.landmark_rmse == pytest.approx(0.0, abs=1e-9)
        assert comp.procrustes_distance == pytest.approx(0.0, abs=1e-9)
        assert geometry_change_score(comp) == pytest.approx(0.0)

    def test_eyebrow_only_change_flags_eyebrows(
        self, synthetic_face: np.ndarray, moved_eyebrows_face: np.ndarray
    ) -> None:
        """Consensus comparison must point at eyebrows as dominant region."""
        comp = compare_consensus([synthetic_face], [moved_eyebrows_face])
        by_name = {r.region: r for r in comp.regional}
        brow_mean = np.mean(
            [by_name["left_eyebrow"].mean, by_name["right_eyebrow"].mean]
        )
        others = [r.mean for r in comp.regional if "eyebrow" not in r.region]
        assert brow_mean > max(others)
        assert brow_mean > 0.05  # 12 px on a 100 px reference


class TestSummaries:
    def test_distribution_summary(self) -> None:
        summary = summarize_pairwise([1.0, 2.0, 3.0, 4.0])
        assert summary["mean"] == pytest.approx(2.5)
        assert summary["median"] == pytest.approx(2.5)
        assert summary["min"] == 1.0
        assert summary["max"] == 4.0
        assert summary["p05"] <= summary["median"] <= summary["p95"]

    def test_aggregate_pairwise(self, synthetic_face: np.ndarray) -> None:
        results = [
            _make_result(v, v * 1.1, synthetic_face) for v in (0.01, 0.02, 0.03)
        ]
        agg = aggregate_pairwise(results)
        assert agg["mean_landmark_displacement"]["mean"] == pytest.approx(0.02)
        assert agg["mean_landmark_displacement"]["max"] == pytest.approx(0.03)

    def test_aggregate_regional(self, synthetic_face: np.ndarray) -> None:
        results = [_make_result(0.01, 0.01, synthetic_face)]
        regional = aggregate_regional(results)
        assert "left_eyebrow" in regional
        assert regional["left_eyebrow"]["mean_displacement"] == pytest.approx(0.0)


class TestRatioChangeAggregation:
    def test_percentage_change(self) -> None:
        baseline = [{"mouth_width_over_face_width": 0.40} for _ in range(3)]
        makeup = [{"mouth_width_over_face_width": 0.44} for _ in range(3)]
        (change,) = aggregate_ratio_changes(baseline, makeup)
        assert change.baseline_mean == pytest.approx(0.40)
        assert change.makeup_mean == pytest.approx(0.44)
        assert change.absolute_change == pytest.approx(0.04)
        assert change.percentage_change == pytest.approx(10.0)

    def test_zero_baseline_gives_nan_percentage(self) -> None:
        (change,) = aggregate_ratio_changes([{"x": 0.0}], [{"x": 1.0}])
        assert change.percentage_change != change.percentage_change  # NaN


class TestGeometryChangeScore:
    def test_score_bounds_and_monotonicity(self) -> None:
        rng = np.random.default_rng(2)
        base = rng.normal(size=(468, 2))
        # Make the reference distance deterministic so `delta` maps linearly
        # onto reference-distance units.
        base[33] = (-0.5, 0.0)
        base[263] = (0.5, 0.0)

        def score_for(delta: float) -> float:
            moved = base.copy()
            # Asymmetric, non-similarity displacement (can't be absorbed by
            # Procrustes alignment): only the lower half of the face moves.
            moved[base[:, 1] < 0, 1] -= delta
            return geometry_change_score(compare_consensus([base], [moved]))

        s_small = score_for(0.0)
        s_quarter = score_for(SCORE_SATURATION * 0.25)
        s_half = score_for(SCORE_SATURATION * 0.5)
        s_full = score_for(SCORE_SATURATION)
        s_huge = score_for(SCORE_SATURATION * 3)
        assert s_small == pytest.approx(0.0)
        assert 0.0 <= s_small <= s_quarter <= s_half <= s_full <= s_huge
        assert s_half > 0.0
        assert s_huge == pytest.approx(100.0)

    def test_score_details_audit_trail(self, synthetic_face: np.ndarray) -> None:
        comp = compare_consensus([synthetic_face], [synthetic_face])
        details = score_details(comp)
        assert details["noise_floor"] == SCORE_NOISE_FLOOR
        assert details["saturation"] == SCORE_SATURATION
        assert "d95" in details
        assert details["geometry_change_score"] == pytest.approx(0.0)
