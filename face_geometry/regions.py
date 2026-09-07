"""Facial region definitions over the MediaPipe 468-landmark topology.

Each region maps a human-readable facial feature to a set of landmark
indices. Regions may overlap (e.g. ``mouth`` includes the lip landmarks) —
that is intentional: regions are analysis lenses, not a partition.
"""

from __future__ import annotations

from .models import RegionSpec

# Canonical MediaPipe Face Mesh index groups.
LEFT_EYE: tuple[int, ...] = (
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
)
RIGHT_EYE: tuple[int, ...] = (
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
)
LEFT_EYEBROW: tuple[int, ...] = (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
RIGHT_EYEBROW: tuple[int, ...] = (336, 296, 334, 293, 300, 283, 295, 282, 285, 276)
NOSE: tuple[int, ...] = (
    1, 2, 4, 5, 6, 19, 94, 97, 98, 168, 195, 197, 236, 3, 51, 48, 115, 131,
    134, 102, 49, 220, 305, 281, 363, 360, 279, 456, 420, 326, 327, 294,
)
UPPER_LIP: tuple[int, ...] = (
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 164, 167, 165, 92, 186,
)
LOWER_LIP: tuple[int, ...] = (
    146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61, 78, 95, 88, 178, 87,
    14, 317, 402, 318, 324, 308,
)
MOUTH: tuple[int, ...] = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0,
    37, 39, 40, 185, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
    310, 311, 312, 13, 82, 81, 80, 191,
)
JAWLINE: tuple[int, ...] = (
    356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
    150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)
LEFT_CHEEK: tuple[int, ...] = (
    205, 50, 101, 118, 119, 100, 36, 123, 147, 187, 207, 206, 216,
)
RIGHT_CHEEK: tuple[int, ...] = (
    425, 280, 330, 347, 348, 329, 266, 352, 376, 411, 427, 426, 436,
)

REGIONS: tuple[RegionSpec, ...] = (
    RegionSpec("left_eye", LEFT_EYE),
    RegionSpec("right_eye", RIGHT_EYE),
    RegionSpec("left_eyebrow", LEFT_EYEBROW),
    RegionSpec("right_eyebrow", RIGHT_EYEBROW),
    RegionSpec("nose", NOSE),
    RegionSpec("upper_lip", UPPER_LIP),
    RegionSpec("lower_lip", LOWER_LIP),
    RegionSpec("mouth", MOUTH),
    RegionSpec("jawline", JAWLINE),
    RegionSpec("left_cheek", LEFT_CHEEK),
    RegionSpec("right_cheek", RIGHT_CHEEK),
)

REGION_INDEX: dict[str, tuple[int, ...]] = {r.name: r.indices for r in REGIONS}

# Coarser groupings used for the report table and the change-map summary.
REPORT_REGION_GROUPS: dict[str, tuple[str, ...]] = {
    "Eyes": ("left_eye", "right_eye"),
    "Eyebrows": ("left_eyebrow", "right_eyebrow"),
    "Nose": ("nose",),
    "Mouth": ("mouth", "upper_lip", "lower_lip"),
    "Jaw": ("jawline",),
    "Cheeks": ("left_cheek", "right_cheek"),
}


def region_names() -> list[str]:
    """Return the ordered list of region names."""

    return [r.name for r in REGIONS]


def indices_for(region: str) -> tuple[int, ...]:
    """Return landmark indices for a region name.

    Raises:
        KeyError: if the region is unknown.
    """

    if region not in REGION_INDEX:
        raise KeyError(f"Unknown facial region: {region!r}")
    return REGION_INDEX[region]
