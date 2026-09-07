"""Core dataclasses shared across the face_geometry pipeline.

Every numerical container uses NumPy arrays and full type annotations so the
pipeline stays deterministic, auditable and easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# MediaPipe canonical face-mesh constants (468 landmark topology).
# These are well-known index sets published with the Face Mesh model.
# ---------------------------------------------------------------------------

NUM_LANDMARKS = 468

# 10 landmarks tracing the oval silhouette on each side, starting at the
# forehead top-centre (10) and proceeding toward the chin (152).
FACE_OVAL_LEFT: tuple[int, ...] = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323)
FACE_OVAL_RIGHT: tuple[int, ...] = (10, 109, 67, 103, 54, 21, 162, 127, 234, 93)
CHIN_INDEX = 152
FOREHEAD_INDEX = 10


@dataclass(frozen=True, slots=True)
class HeadPose:
    """Estimated head orientation.

    Angles are in degrees; positive yaw turns the face toward its left,
    positive pitch nods the chin downward.
    """

    yaw: float
    pitch: float
    roll: float


@dataclass(slots=True)
class FaceObservation:
    """All measurements extracted from a single photograph."""

    image_path: str
    image_size: tuple[int, int]  # (width, height)
    landmarks: "object"  # np.ndarray shape (468, 2), pixel coordinates
    pose: HeadPose | None
    detection_confidence: float
    normalized: "object | None" = None  # np.ndarray (468, 2), pose-normalised
    reference_distance: float = 1.0  # normaliser used to build `normalized`
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RegionSpec:
    """Definition of a facial region as a set of landmark indices."""

    name: str
    indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DisplacementStats:
    """Summary statistics of per-landmark Euclidean displacement."""

    mean: float
    median: float
    max: float
    rmse: float


@dataclass(frozen=True, slots=True)
class RatioMeasurement:
    """One scalar facial ratio measured on a single observation."""

    name: str
    value: float


@dataclass(frozen=True, slots=True)
class RatioChange:
    """Aggregated change of one ratio between the two folders."""

    name: str
    baseline_mean: float
    makeup_mean: float
    absolute_change: float
    percentage_change: float  # NaN when baseline_mean is zero


@dataclass(frozen=True, slots=True)
class RegionalStats:
    """Aggregated displacement statistics for one facial region."""

    region: str
    mean: float
    median: float
    max: float
    rmse: float


@dataclass(slots=True)
class QualityAssessment:
    """Quality flags for a single baseline-vs-makeup pairing."""

    score: float  # 1.0 = clean, 0.0 = unusable-but-kept
    flags: tuple[str, ...] = ()
    mouth_opening_delta: float = 0.0
    min_eye_openness: float = 1.0


@dataclass(slots=True)
class PairwiseResult:
    """Result of comparing one baseline image against one makeup image."""

    baseline_path: str
    makeup_path: str
    mean_landmark_displacement: float
    median_landmark_displacement: float
    max_landmark_displacement: float
    landmark_rmse: float
    procrustes_distance: float
    comparison_quality: float
    quality_flags: tuple[str, ...] = ()
    regional: tuple[RegionalStats, ...] = ()


ConsensusKind = Literal["baseline", "makeup"]


@dataclass(slots=True)
class ConsensusComparison:
    """Comparison between the two per-folder consensus shapes."""

    mean_landmark_displacement: float
    median_landmark_displacement: float
    max_landmark_displacement: float
    landmark_rmse: float
    procrustes_distance: float
    regional: tuple[RegionalStats, ...] = ()
    per_landmark_displacement: "object" = field(default=None, repr=False)
