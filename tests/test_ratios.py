"""Tests for facial ratios, angles and expression signals."""

from __future__ import annotations

import numpy as np
import pytest

from face_geometry.alignment import normalize_pose
from face_geometry.ratios import (
    compute_angles,
    compute_ratios,
    eye_openness,
    mouth_opening,
)


def _normalised(face: np.ndarray) -> np.ndarray:
    out, _ = normalize_pose(face)
    return out


class TestRatios:
    def test_expected_values_on_synthetic_face(
        self, synthetic_face: np.ndarray
    ) -> None:
        ratios = compute_ratios(_normalised(synthetic_face))
        # Reference distance = 100 px, so 1 px == 0.01 reference units.
        assert ratios["eye_width_over_interocular"] == pytest.approx(2.0)
        assert ratios["eye_height_over_eye_width"] == pytest.approx(0.3)
        assert ratios["inner_eyebrow_separation"] == pytest.approx(0.2)
        assert ratios["mouth_width_over_interocular"] == pytest.approx(2.8)
        assert ratios["nose_width_over_face_width"] == pytest.approx(0.3)
        assert ratios["upper_lip_height"] == pytest.approx(0.08)
        assert ratios["lower_lip_height"] == pytest.approx(0.08)
        assert ratios["mouth_height_over_mouth_width"] == pytest.approx(22 / 56)
        assert ratios["jaw_width_over_face_width"] == pytest.approx(80 / 120)
        assert ratios["facial_length_over_width"] == pytest.approx(190 / 120)
        assert ratios["eyebrow_to_eye_distance"] == pytest.approx(0.27)

    def test_ratios_scale_invariant(self, synthetic_face: np.ndarray) -> None:
        a = compute_ratios(_normalised(synthetic_face))
        b = compute_ratios(_normalised(synthetic_face * 5.0 + 3.0))
        for key in a:
            assert a[key] == pytest.approx(b[key], rel=1e-9), key

    def test_zero_denominator_yields_nan(self) -> None:
        pts = np.zeros((468, 2))
        ratios = compute_ratios(pts)
        assert np.isnan(ratios["eye_width_over_interocular"])
        assert np.isnan(ratios["mouth_height_over_mouth_width"])


class TestAngles:
    def test_symmetric_face_axis_angles(
        self, synthetic_face: np.ndarray
    ) -> None:
        angles = compute_angles(_normalised(synthetic_face))
        # Undirected axis angles live in (-90, 90]; a horizontal axis is 0.
        assert angles["left_eye_axis_deg"] == pytest.approx(0.0)
        assert angles["right_eye_axis_deg"] == pytest.approx(0.0)
        assert angles["mouth_corner_angle_deg"] == pytest.approx(0.0)
        assert angles["jaw_contour_angle_deg"] > 0.0
        assert angles["left_eyebrow_slope_deg"] == pytest.approx(
            -angles["right_eyebrow_slope_deg"]
        )

    def test_angles_rotation_invariant(self, synthetic_face: np.ndarray) -> None:
        theta = np.radians(33.0)
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        rotated = synthetic_face @ rot.T + np.array([9.0, -4.0])
        a = compute_angles(_normalised(synthetic_face))
        b = compute_angles(_normalised(rotated))
        for key in a:
            assert a[key] == pytest.approx(b[key], abs=1e-6), key


class TestExpressionSignals:
    def test_mouth_opening(self, synthetic_face: np.ndarray) -> None:
        assert mouth_opening(_normalised(synthetic_face)) == pytest.approx(0.06)

    def test_eye_openness(self, synthetic_face: np.ndarray) -> None:
        left, right = eye_openness(_normalised(synthetic_face))
        assert left == pytest.approx(0.3)
        assert right == pytest.approx(0.3)

    def test_blink_reduces_openness(self, synthetic_face: np.ndarray) -> None:
        face = synthetic_face.copy()
        face[159, 1] = 202.0  # left top lid almost meets the bottom lid
        left, right = eye_openness(_normalised(face))
        assert left < 0.1 < right
