"""Dense facial landmark extraction backed by MediaPipe Face Landmarker.

The Face Landmarker task model outputs 468 3D landmarks (the canonical face
mesh) together with a 4x4 facial transformation matrix that we use for head
pose estimation and quality gating.

Model file
----------
The ``face_landmarker.task`` model (~30 MB) is required at runtime. It is
downloaded automatically on first use into
``~/.cache/face_geometry/models`` (override with the environment variable
``FACE_GEOMETRY_MODEL_PATH`` or pass ``model_path`` explicitly).
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .models import FaceObservation, HeadPose, NUM_LANDMARKS

logger = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float32/1/face_landmarker.task"
)
_MODEL_ENV_VAR = "FACE_GEOMETRY_MODEL_PATH"
_DEFAULT_MODEL_DIR = Path.home() / ".cache" / "face_geometry" / "models"
_MODEL_FILENAME = "face_landmarker.task"

# Pinned SHA-256 of the official float32 face_landmarker.task, used to
# integrity-check the downloaded model. When ``None`` the check is skipped
# (e.g. while the official digest is not yet pinned); setting it to the
# published digest makes any mismatching download be rejected and removed.
_MODEL_SHA256: str | None = None

#: OpenCV solvePnP reference points (right eye outer corner is index 263).
_POSE_LANDMARK_IDS = (1, 152, 263, 33, 287, 57)


class ModelDownloadError(RuntimeError):
    """Raised when the face landmarker model cannot be obtained."""


class NoFaceDetectedError(RuntimeError):
    """Raised when no face is found in an image."""


def default_model_path() -> Path:
    """Return the expected location of the face landmarker model file."""

    override = os.environ.get(_MODEL_ENV_VAR)
    if override:
        return Path(override)
    return _DEFAULT_MODEL_DIR / _MODEL_FILENAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(model_path: Path | None = None) -> Path:
    """Return a path to a valid model file, downloading it if necessary.

    The download is integrity-checked against a pinned SHA-256; a corrupted
    or tampered file is rejected and removed rather than silently trusted.

    Raises:
        ModelDownloadError: if the file is missing, cannot be downloaded, or
            fails the integrity check.
    """

    path = model_path or default_model_path()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading face landmarker model to %s", path)
    try:
        urllib.request.urlretrieve(MODEL_URL, path)  # noqa: S310 - fixed URL
    except Exception as exc:  # pragma: no cover - network dependent
        raise ModelDownloadError(
            f"Could not download face landmarker model from {MODEL_URL}: {exc}. "
            f"Download it manually and set {_MODEL_ENV_VAR} or pass --model-path."
        ) from exc
    actual = _sha256(path)
    if _MODEL_SHA256 is not None and actual != _MODEL_SHA256:
        path.unlink(missing_ok=True)
        raise ModelDownloadError(
            "Downloaded face landmarker model failed its SHA-256 integrity "
            f"check (expected {_MODEL_SHA256}, got {actual}). The file was "
            "removed; download it manually from the official MediaPipe source "
            f"and set {_MODEL_ENV_VAR} or pass --model-path."
        )
    return path


def estimate_head_pose(
    landmarks: np.ndarray, image_size: tuple[int, int]
) -> HeadPose:
    """Estimate a coarse yaw/pitch/roll for quality gating.

    Approach: MediaPipe reports a z value per landmark that shares the x/y
    scale, so the detected (x, y, z) triplets form a rough 3D face. We centre
    those 3D points to build an object model and run ``cv2.solvePnP`` to
    recover the rotation that best projects that model onto the detected 2D
    image points. Because the object model is itself derived from the
    (already posed) detection, the estimate is *biased toward zero* for
    out-of-plane rotation; it is a heuristic intended only for coarse
    extreme-pose quality gating (the ``--max-yaw``/``--max-pitch`` flags),
    not for precise pose measurement.

    ``output_facial_transformation_matrixes=True`` is enabled on the
    landmarker so a metric-accurate matrix is available should a future
    refinement prefer it over this heuristic; the current implementation does
    not consume it.

    Args:
        landmarks: array of shape (N, 3) in pixel units (x, y, z as reported).
        image_size: (width, height) of the source image.

    Returns:
        HeadPose with yaw, pitch, roll in degrees.
    """

    width, height = image_size
    ids = list(_POSE_LANDMARK_IDS)
    pts3d = landmarks[ids, :].astype(np.float64)
    image_points = pts3d[:, :2].copy()

    focal = float(max(width, height))
    camera_matrix = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
    )
    # Object model: centre the 3D points, keep z as depth.
    object_points = pts3d - pts3d.mean(axis=0, keepdims=True)

    ok, rvec, _ = cv2.solvePnP(
        object_points,
        image_points - np.array([width / 2.0, height / 2.0]),
        camera_matrix,
        None,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        return HeadPose(yaw=0.0, pitch=0.0, roll=0.0)

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    # Decompose rotation (ZYX) into extrinsic angles.
    sy = float(np.hypot(rotation_matrix[0, 0], rotation_matrix[1, 0]))
    singular = sy < 1e-6
    if not singular:
        pitch = float(np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])))
        yaw = float(np.degrees(np.arctan2(-rotation_matrix[2, 0], sy)))
        roll = float(np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])))
    else:
        pitch = float(np.degrees(np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])))
        yaw = float(np.degrees(np.arctan2(-rotation_matrix[2, 0], sy)))
        roll = 0.0
    return HeadPose(yaw=yaw, pitch=pitch, roll=roll)


class FaceLandmarkExtractor:
    """Extract 468-point face landmarks from images using MediaPipe."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_path: Path | None = None,
    ) -> None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )

        self.min_detection_confidence = float(min_detection_confidence)
        resolved = ensure_model(model_path)
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved)),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=float(min_detection_confidence),
            min_face_presence_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_detection_confidence),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        """Release the underlying MediaPipe resources."""

        self._landmarker.close()

    def __enter__(self) -> "FaceLandmarkExtractor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def extract(self, image_path: Path) -> FaceObservation:
        """Detect the primary face in ``image_path`` and return landmarks.

        Raises:
            NoFaceDetectedError: when no face passes the confidence threshold.
            ValueError: when the image cannot be decoded.
        """

        import mediapipe as mp

        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"Cannot decode image: {image_path}")
        height, width = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            raise NoFaceDetectedError(f"No face detected in {image_path}")

        face = result.face_landmarks[0]
        if len(face) != NUM_LANDMARKS:
            logger.warning(
                "%s: expected %d landmarks, got %d",
                image_path,
                NUM_LANDMARKS,
                len(face),
            )
        coords = np.array(
            [[lm.x * width, lm.y * height, lm.z * width] for lm in face],
            dtype=np.float64,
        )
        pose = estimate_head_pose(coords, (width, height))

        # The detector only returns faces above min_face_detection_confidence,
        # so a returned face is at/above the configured threshold; record that
        # threshold as the effective detection confidence.
        confidence = self.min_detection_confidence
        logger.debug(
            "%s: %d landmarks, pose yaw=%.1f pitch=%.1f roll=%.1f",
            image_path,
            len(face),
            pose.yaw,
            pose.pitch,
            pose.roll,
        )
        return FaceObservation(
            image_path=str(image_path),
            image_size=(width, height),
            landmarks=coords[:, :2].copy(),
            pose=pose,
            detection_confidence=confidence,
        )
