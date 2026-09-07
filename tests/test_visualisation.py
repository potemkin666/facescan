"""Tests for the geometry change-map visualisations."""

from __future__ import annotations

import numpy as np
import pytest

from face_geometry.visualisation import (
    draw_change_map,
    draw_threshold_map,
    save_image,
)


class TestChangeMap:
    def test_canvas_shape(self, synthetic_face: np.ndarray) -> None:
        image = draw_change_map(synthetic_face, synthetic_face, size=400)
        assert image.shape == (400, 400, 3)
        assert image.dtype == np.uint8

    def test_identical_faces_no_displacement_vectors(
        self, synthetic_face: np.ndarray
    ) -> None:
        image = draw_change_map(synthetic_face, synthetic_face, size=400)
        # No red pixels: zero-length arrows for identical shapes.
        red_channel = image[:, :, 2]
        green_channel = image[:, :, 1]
        strongly_red = (red_channel > 200) & (green_channel < 60)
        assert strongly_red.sum() == 0

    def test_moved_landmarks_produce_hot_pixels(
        self, synthetic_face: np.ndarray, moved_eyebrows_face: np.ndarray
    ) -> None:
        image = draw_change_map(
            synthetic_face, moved_eyebrows_face, size=500, show_landmark_ids=True
        )
        warm = (image[:, :, 2] > 150) | (image[:, :, 1] > 180)
        assert warm.sum() > 0


class TestThresholdMap:
    def test_no_flags_below_threshold(self, synthetic_face: np.ndarray) -> None:
        image = draw_threshold_map(
            synthetic_face, synthetic_face, warn_threshold=0.02
        )
        # Only background, faint context dots and the title text.
        bright = (image > 100).any(axis=2)
        assert bright.sum() < 5000

    def test_flags_above_threshold(
        self, synthetic_face: np.ndarray, moved_eyebrows_face: np.ndarray
    ) -> None:
        image = draw_threshold_map(
            synthetic_face,
            moved_eyebrows_face,
            warn_threshold=0.02,
            alert_threshold=0.05,
        )
        bright = (image > 100).any(axis=2)
        assert bright.sum() > 100

    def test_save_image(
        self, synthetic_face: np.ndarray, tmp_path
    ) -> None:
        image = draw_change_map(synthetic_face, synthetic_face, size=200)
        out = save_image(image, tmp_path / "nested" / "map.png")
        assert out.exists()
        import cv2

        reloaded = cv2.imread(str(out))
        assert reloaded is not None
        assert reloaded.shape == (200, 200, 3)


class TestDeterminism:
    def test_same_inputs_same_image(
        self, synthetic_face: np.ndarray, moved_eyebrows_face: np.ndarray
    ) -> None:
        a = draw_change_map(synthetic_face, moved_eyebrows_face, size=300)
        b = draw_change_map(synthetic_face, moved_eyebrows_face, size=300)
        np.testing.assert_array_equal(a, b)
