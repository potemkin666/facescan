"""Interpretable facial ratios, angles and expression signals.

Everything is computed on *pose-normalised* coordinates (unit reference
distance, eye axis horizontal, centred on the eye-corner midpoint), so the
measurements are invariant to image scale, translation and in-plane tilt and
never depend on arbitrary pixel dimensions.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe canonical face mesh)
# ---------------------------------------------------------------------------
LEFT_EYE_OUTER, LEFT_EYE_INNER = 33, 133
RIGHT_EYE_INNER, RIGHT_EYE_OUTER = 362, 263
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374

LEFT_BROW_INNER, LEFT_BROW_MID, LEFT_BROW_OUTER = 107, 105, 70
RIGHT_BROW_INNER, RIGHT_BROW_MID, RIGHT_BROW_OUTER = 336, 334, 300

NOSE_LEFT_ALAR, NOSE_RIGHT_ALAR = 98, 327
NOSE_BRIDGE_TOP, NOSE_TIP, NOSE_BASE = 6, 4, 2

MOUTH_LEFT, MOUTH_RIGHT = 61, 291
LIP_TOP, LIP_BOTTOM = 0, 17          # upper-lip top, lower-lip bottom
LIP_INNER_TOP, LIP_INNER_BOTTOM = 13, 14

FACE_LEFT_CHEEK, FACE_RIGHT_CHEEK = 234, 454  # widest face-contour points
JAW_LEFT, JAW_RIGHT = 172, 397
CHIN, FOREHEAD = 152, 10


def _p(landmarks: np.ndarray, index: int) -> np.ndarray:
    return landmarks[index, :2]


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle of the vector a->b relative to the +x axis, in degrees (-180, 180]."""

    delta = b - a
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def _axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Undirected axis angle in degrees, mapped to (-90, 90].

    A segment has no intrinsic direction, so 178° and -178° describe the same
    axis; this canonicalisation makes near-horizontal axes comparable.
    """

    angle = _angle_deg(a, b)
    # Map to (-90, 90]: adding 90, mod 180, subtracting 90.
    return float((angle + 90.0) % 180.0 - 90.0)


def compute_ratios(landmarks: np.ndarray) -> dict[str, float]:
    """Compute interpretable facial ratios on normalised landmarks.

    All distances are in reference-distance units (outer eye corner distance
    == 1), so the ratios are pure shape descriptors.

    Args:
        landmarks: normalised array of shape (468, 2).

    Returns:
        Mapping of ratio name to dimensionless value.
    """

    pts = np.asarray(landmarks, dtype=np.float64)

    interocular = _dist(_p(pts, LEFT_EYE_INNER), _p(pts, RIGHT_EYE_INNER))
    left_eye_w = _dist(_p(pts, LEFT_EYE_OUTER), _p(pts, LEFT_EYE_INNER))
    right_eye_w = _dist(_p(pts, RIGHT_EYE_INNER), _p(pts, RIGHT_EYE_OUTER))
    eye_width = (left_eye_w + right_eye_w) / 2.0
    left_eye_h = _dist(_p(pts, LEFT_EYE_TOP), _p(pts, LEFT_EYE_BOTTOM))
    right_eye_h = _dist(_p(pts, RIGHT_EYE_TOP), _p(pts, RIGHT_EYE_BOTTOM))
    eye_height = (left_eye_h + right_eye_h) / 2.0

    # Eyebrow-to-eye vertical distance (normalised coords have +y down).
    left_brow_eye = float(
        np.mean(pts[[LEFT_BROW_MID, 66], 1]) - np.mean(pts[[LEFT_EYE_TOP], 1])
    )
    right_brow_eye = float(
        np.mean(pts[[RIGHT_BROW_MID, 293], 1]) - np.mean(pts[[RIGHT_EYE_TOP], 1])
    )
    brow_to_eye = (abs(left_brow_eye) + abs(right_brow_eye)) / 2.0
    inner_brow_sep = _dist(_p(pts, LEFT_BROW_INNER), _p(pts, RIGHT_BROW_INNER))

    nose_width = _dist(_p(pts, NOSE_LEFT_ALAR), _p(pts, NOSE_RIGHT_ALAR))
    nose_length = _dist(_p(pts, NOSE_BRIDGE_TOP), _p(pts, NOSE_BASE))

    mouth_width = _dist(_p(pts, MOUTH_LEFT), _p(pts, MOUTH_RIGHT))
    upper_lip_h = abs(_p(pts, LIP_INNER_TOP)[1] - _p(pts, LIP_TOP)[1])
    lower_lip_h = abs(_p(pts, LIP_BOTTOM)[1] - _p(pts, LIP_INNER_BOTTOM)[1])
    mouth_height = _dist(_p(pts, LIP_TOP), _p(pts, LIP_BOTTOM))

    face_width = _dist(_p(pts, FACE_LEFT_CHEEK), _p(pts, FACE_RIGHT_CHEEK))
    jaw_width = _dist(_p(pts, JAW_LEFT), _p(pts, JAW_RIGHT))
    face_height = _dist(_p(pts, FOREHEAD), _p(pts, CHIN))

    def safe_div(num: float, den: float) -> float:
        return num / den if den > 1e-12 else float("nan")

    return {
        "eye_width_over_interocular": safe_div(eye_width, interocular),
        "eye_height_over_eye_width": safe_div(eye_height, eye_width),
        "eyebrow_to_eye_distance": brow_to_eye,
        "inner_eyebrow_separation": inner_brow_sep,
        "nose_width_over_face_width": safe_div(nose_width, face_width),
        "nose_length_over_face_height": safe_div(nose_length, face_height),
        "mouth_width_over_face_width": safe_div(mouth_width, face_width),
        "upper_lip_height": upper_lip_h,
        "lower_lip_height": lower_lip_h,
        "mouth_height_over_mouth_width": safe_div(mouth_height, mouth_width),
        "jaw_width_over_face_width": safe_div(jaw_width, face_width),
        "face_width_over_interocular": safe_div(face_width, interocular),
        "facial_length_over_width": safe_div(face_height, face_width),
        "mouth_width_over_interocular": safe_div(mouth_width, interocular),
    }


def compute_angles(landmarks: np.ndarray) -> dict[str, float]:
    """Compute angular measurements (degrees) on normalised landmarks.

    Angles are measured after pose normalisation, i.e. with the eye axis
    horizontal, so they describe facial geometry rather than head tilt.

    Returns:
        Mapping of angle name to degrees.
    """

    pts = np.asarray(landmarks, dtype=np.float64)

    left_brow_slope = _axis_angle_deg(_p(pts, LEFT_BROW_INNER), _p(pts, LEFT_BROW_OUTER))
    right_brow_slope = _axis_angle_deg(_p(pts, RIGHT_BROW_INNER), _p(pts, RIGHT_BROW_OUTER))
    left_eye_axis = _axis_angle_deg(_p(pts, LEFT_EYE_INNER), _p(pts, LEFT_EYE_OUTER))
    right_eye_axis = _axis_angle_deg(_p(pts, RIGHT_EYE_INNER), _p(pts, RIGHT_EYE_OUTER))
    mouth_corner_angle = _axis_angle_deg(_p(pts, MOUTH_LEFT), _p(pts, MOUTH_RIGHT))

    # Eye-corner openness: angle at the outer corner between inner corner and
    # the eyelid mid-height point; smaller means more narrowed/squinting.
    def corner_angle(outer: int, inner: int, top: int, bottom: int) -> float:
        lid_mid = (_p(pts, top) + _p(pts, bottom)) / 2.0
        v1 = _p(pts, inner) - _p(pts, outer)
        v2 = lid_mid - _p(pts, outer)
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    left_corner = corner_angle(
        LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM
    )
    right_corner = corner_angle(
        RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM
    )

    # Jaw contour angle at the chin.
    v1 = _p(pts, JAW_LEFT) - _p(pts, CHIN)
    v2 = _p(pts, JAW_RIGHT) - _p(pts, CHIN)
    cos_jaw = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    jaw_contour_angle = float(np.degrees(np.arccos(np.clip(cos_jaw, -1.0, 1.0))))

    return {
        "left_eyebrow_slope_deg": left_brow_slope,
        "right_eyebrow_slope_deg": right_brow_slope,
        "left_eye_axis_deg": left_eye_axis,
        "right_eye_axis_deg": right_eye_axis,
        "left_eye_corner_angle_deg": left_corner,
        "right_eye_corner_angle_deg": right_corner,
        "mouth_corner_angle_deg": mouth_corner_angle,
        "jaw_contour_angle_deg": jaw_contour_angle,
    }


def mouth_opening(landmarks: np.ndarray) -> float:
    """Inner-lip vertical separation in reference units (expression signal)."""

    pts = np.asarray(landmarks, dtype=np.float64)
    return abs(float(_p(pts, LIP_INNER_TOP)[1] - _p(pts, LIP_INNER_BOTTOM)[1]))


def eye_openness(landmarks: np.ndarray) -> tuple[float, float]:
    """Per-eye openness (height / width), a blink/narrowing signal."""

    pts = np.asarray(landmarks, dtype=np.float64)
    left = _dist(_p(pts, LEFT_EYE_TOP), _p(pts, LEFT_EYE_BOTTOM)) / max(
        _dist(_p(pts, LEFT_EYE_OUTER), _p(pts, LEFT_EYE_INNER)), 1e-12
    )
    right = _dist(_p(pts, RIGHT_EYE_TOP), _p(pts, RIGHT_EYE_BOTTOM)) / max(
        _dist(_p(pts, RIGHT_EYE_INNER), _p(pts, RIGHT_EYE_OUTER)), 1e-12
    )
    return left, right
