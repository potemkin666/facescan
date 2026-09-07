"""Synthetic landmark fixtures: a deterministic face-like 468-point layout.

The fixture places points at plausible normalised locations for the landmark
indices the tool actually reads (eyes, brows, nose, lips, jaw, cheeks,
contour) and fills the remaining indices on a deterministic grid. No
photographs of real people are required.
"""

from __future__ import annotations

import numpy as np
import pytest

from face_geometry.models import NUM_LANDMARKS
from face_geometry.regions import REGIONS


def _grid_base() -> np.ndarray:
    """Deterministic base layout: a face-oval-like ring grid in [-1, 1]."""

    rng = np.random.default_rng(20240907)
    pts = rng.uniform(-1.0, 1.0, size=(NUM_LANDMARKS, 2))
    # Force bilateral symmetry so ratios/angles are meaningful.
    pts[: NUM_LANDMARKS // 2] = np.abs(pts[: NUM_LANDMARKS // 2])
    pts[NUM_LANDMARKS // 2 :] = pts[: NUM_LANDMARKS - NUM_LANDMARKS // 2]
    return pts


def _place(pts: np.ndarray, index: int, x: float, y: float) -> None:
    pts[index] = (x, y)


def build_synthetic_face() -> np.ndarray:
    """Construct a plausible synthetic face in pixel-like coordinates.

    The reference segment (outer eye corners, indices 33 and 263) is 100 px
    long and horizontal, centred at (200, 200).
    """

    pts = _grid_base() * 400.0 + 200.0

    # --- Eyes (reference distance = 100 px) -------------------------------
    _place(pts, 33, 150.0, 200.0)    # left outer corner
    _place(pts, 133, 190.0, 200.0)   # left inner corner
    _place(pts, 362, 210.0, 200.0)   # right inner corner
    _place(pts, 263, 250.0, 200.0)   # right outer corner
    _place(pts, 159, 170.0, 192.0)   # left top lid
    _place(pts, 145, 170.0, 204.0)   # left bottom lid
    _place(pts, 386, 230.0, 192.0)   # right top lid
    _place(pts, 374, 230.0, 204.0)   # right bottom lid
    for i, (x, y) in enumerate(
        [(150, 200), (158, 194), (166, 191), (174, 190), (182, 192), (188, 197)]
    ):
        _place(pts, (7, 163, 144, 153, 154, 155)[i], x, y)
    for i, (x, y) in enumerate(
        [(212, 197), (218, 192), (226, 190), (234, 191), (242, 194), (250, 200)]
    ):
        _place(pts, (398, 384, 385, 380, 381, 382)[i], x, y)

    # --- Eyebrows (exactly symmetric about x=200) --------------------------
    _place(pts, 107, 190.0, 170.0)   # left inner brow (near nose)
    _place(pts, 105, 169.0, 164.0)   # left brow mid
    _place(pts, 70, 148.0, 170.0)    # left outer brow
    _place(pts, 66, 179.0, 166.0)
    _place(pts, 336, 210.0, 170.0)   # right inner brow (near nose)
    _place(pts, 334, 231.0, 164.0)   # right brow mid
    _place(pts, 293, 221.0, 166.0)
    _place(pts, 300, 252.0, 170.0)   # right outer brow

    # --- Nose --------------------------------------------------------------
    _place(pts, 6, 200.0, 178.0)     # bridge top
    _place(pts, 4, 200.0, 216.0)     # tip
    _place(pts, 2, 200.0, 222.0)     # base
    _place(pts, 98, 182.0, 226.0)    # left alar
    _place(pts, 327, 218.0, 226.0)   # right alar

    # --- Mouth -------------------------------------------------------------
    _place(pts, 61, 172.0, 252.0)    # left corner
    _place(pts, 291, 228.0, 252.0)   # right corner
    _place(pts, 0, 200.0, 244.0)     # upper-lip top
    _place(pts, 13, 200.0, 252.0)    # inner top
    _place(pts, 14, 200.0, 258.0)    # inner bottom
    _place(pts, 17, 200.0, 266.0)    # lower-lip bottom

    # --- Face contour ------------------------------------------------------
    _place(pts, 234, 140.0, 230.0)   # left cheek (face width)
    _place(pts, 454, 260.0, 230.0)   # right cheek
    _place(pts, 172, 160.0, 280.0)   # left jaw
    _place(pts, 397, 240.0, 280.0)   # right jaw
    _place(pts, 152, 200.0, 310.0)   # chin
    _place(pts, 10, 200.0, 120.0)    # forehead

    return pts


@pytest.fixture()
def synthetic_face() -> np.ndarray:
    """A fresh copy of the synthetic face for every test."""

    return build_synthetic_face()


@pytest.fixture()
def moved_eyebrows_face(synthetic_face: np.ndarray) -> np.ndarray:
    """Synthetic face with *only* eyebrow landmarks shifted upward."""

    moved = synthetic_face.copy()
    for spec in REGIONS:
        if "eyebrow" in spec.name:
            moved[list(spec.indices), 1] -= 12.0  # upward in image coords
    return moved
