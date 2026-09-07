"""Image discovery, face detection and observation loading.

Detection itself lives in :mod:`face_geometry.landmarks`; this module handles
folder traversal, decoding, quality gating and pose normalisation so the CLI
gets a ready-to-compare list of :class:`FaceObservation` objects.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .alignment import normalize_pose
from .landmarks import FaceLandmarkExtractor, NoFaceDetectedError
from .models import FaceObservation
from .quality import assess_observation

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def find_images(folder: Path, recursive: bool = False) -> list[Path]:
    """Return all supported image paths in ``folder``, sorted for determinism.

    Raises:
        NotADirectoryError: if ``folder`` does not exist or is not a directory.
    """

    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Image folder not found: {folder}")
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    images = sorted(
        p for p in iterator if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()
    )
    logger.info("Found %d image(s) in %s (recursive=%s)", len(images), folder, recursive)
    return images


def load_observations(
    folder: Path,
    extractor: FaceLandmarkExtractor,
    *,
    recursive: bool = False,
    max_yaw: float = 30.0,
    max_pitch: float = 25.0,
) -> list[FaceObservation]:
    """Detect, pose-normalise and quality-flag every usable image in a folder.

    Images where no face is detected (or which cannot be decoded) are logged
    and skipped — the caller compares every *usable* baseline against every
    *usable* makeup image.
    """

    observations: list[FaceObservation] = []
    for path in find_images(folder, recursive=recursive):
        try:
            obs = extractor.extract(path)
        except NoFaceDetectedError as exc:
            logger.warning("Skipping %s: %s", path, exc)
            continue
        except ValueError as exc:
            logger.warning("Skipping %s: %s", path, exc)
            continue

        normalised, ref = normalize_pose(np.asarray(obs.landmarks, dtype=np.float64))
        obs.normalized = normalised
        obs.reference_distance = ref
        obs.quality_flags = assess_observation(obs, max_yaw=max_yaw, max_pitch=max_pitch)
        observations.append(obs)
        logger.debug(
            "Loaded %s (ref_distance=%.3f, flags=%s)",
            path,
            ref,
            obs.quality_flags or "none",
        )

    logger.info("%s: %d usable observation(s)", folder, len(observations))
    return observations


def load_image_bgr(path: str | Path) -> np.ndarray:
    """Decode an image as BGR; raises a clear error when undecodable."""

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image
