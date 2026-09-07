"""Tests for comparison-quality assessment (expression/pose flagging)."""

from __future__ import annotations

import numpy as np

from face_geometry.alignment import normalize_pose
from face_geometry.models import FaceObservation, HeadPose
from face_geometry.quality import assess_observation, assess_pair


def _observation(
    face: np.ndarray, pose: HeadPose | None = None
) -> FaceObservation:
    normalised, ref = normalize_pose(face)
    return FaceObservation(
        image_path="x.jpg",
        image_size=(400, 400),
        landmarks=face,
        pose=pose or HeadPose(yaw=0.0, pitch=0.0, roll=0.0),
        detection_confidence=1.0,
        normalized=normalised,
        reference_distance=ref,
    )


class TestAssessObservation:
    def test_clean_pose_no_flags(self, synthetic_face: np.ndarray) -> None:
        assert assess_observation(_observation(synthetic_face)) == ()

    def test_extreme_yaw_flagged(self, synthetic_face: np.ndarray) -> None:
        obs = _observation(synthetic_face, HeadPose(yaw=45.0, pitch=0.0, roll=0.0))
        assert "extreme_yaw" in assess_observation(obs)

    def test_extreme_pitch_flagged(self, synthetic_face: np.ndarray) -> None:
        obs = _observation(synthetic_face, HeadPose(yaw=0.0, pitch=-40.0, roll=0.0))
        assert "extreme_pitch" in assess_observation(obs)

    def test_thresholds_respected(self, synthetic_face: np.ndarray) -> None:
        obs = _observation(synthetic_face, HeadPose(yaw=28.0, pitch=0.0, roll=0.0))
        assert "extreme_yaw" not in assess_observation(obs, max_yaw=30.0)
        assert "extreme_yaw" in assess_observation(obs, max_yaw=25.0)


class TestAssessPair:
    def test_identical_pair_full_quality(self, synthetic_face: np.ndarray) -> None:
        a = _observation(synthetic_face)
        quality = assess_pair(a, _observation(synthetic_face))
        assert quality.score == 1.0
        assert quality.flags == ()

    def test_mouth_opening_mismatch(self, synthetic_face: np.ndarray) -> None:
        opened = synthetic_face.copy()
        opened[14, 1] += 25.0  # drop the inner lower lip (mouth opens)
        quality = assess_pair(_observation(synthetic_face), _observation(opened))
        assert "mouth_opening_mismatch" in quality.flags
        assert quality.score < 1.0

    def test_blink_flagged(self, synthetic_face: np.ndarray) -> None:
        blink = synthetic_face.copy()
        blink[159, 1] = 202.0  # left eye nearly closed
        quality = assess_pair(_observation(synthetic_face), _observation(blink))
        assert "possible_blink" in quality.flags
        assert quality.min_eye_openness < 0.1

    def test_flags_reduce_score_monotonically(
        self, synthetic_face: np.ndarray
    ) -> None:
        base = _observation(synthetic_face)
        calm = assess_pair(base, _observation(synthetic_face))
        posed = assess_pair(
            base,
            _observation(synthetic_face, HeadPose(yaw=60.0, pitch=50.0, roll=0.0)),
        )
        assert posed.score < calm.score
        assert {"extreme_yaw", "extreme_pitch"} <= set(posed.flags)

    def test_unnormalised_raises(self, synthetic_face: np.ndarray) -> None:
        obs = _observation(synthetic_face)
        obs.normalized = None
        try:
            assess_pair(obs, _observation(synthetic_face))
        except ValueError:
            return
        raise AssertionError("Expected ValueError for unnormalised observation")
