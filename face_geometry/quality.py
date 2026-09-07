"""Comparison-quality assessment: separating expression/pose noise.

Questionable comparisons are *flagged, never discarded*: every pairing keeps
its ``comparison_quality`` score (1.0 clean → 0.0 severely degraded) plus
human-readable flags, so downstream consumers can weight or filter as they
see fit while the raw numbers stay auditable.
"""

from __future__ import annotations

import numpy as np

from .models import FaceObservation, QualityAssessment
from .ratios import eye_openness, mouth_opening

#: Mouth-opening ratio between pair members above this flags expression change.
MOUTH_OPENING_RATIO_THRESHOLD = 2.5
MOUTH_OPENING_ABS_THRESHOLD = 0.10  # reference-distance units
#: Eye-openness below this fraction of the typical value suggests a blink.
BLINK_RATIO_THRESHOLD = 0.55
BLINK_ABS_THRESHOLD = 0.05
#: Default pose limits (overridable from the CLI).
DEFAULT_MAX_YAW = 30.0
DEFAULT_MAX_PITCH = 25.0
DEFAULT_MAX_ROLL = 45.0

#: Penalty applied per flag category when computing the quality score.
_PENALTY = {
    "mouth_opening_mismatch": 0.25,
    "possible_blink": 0.20,
    "extreme_yaw": 0.30,
    "extreme_pitch": 0.30,
    "extreme_roll": 0.20,
}


def assess_observation(
    obs: FaceObservation,
    max_yaw: float = DEFAULT_MAX_YAW,
    max_pitch: float = DEFAULT_MAX_PITCH,
    max_roll: float = DEFAULT_MAX_ROLL,
) -> tuple[str, ...]:
    """Flag a single observation with an extreme head pose."""

    if obs.pose is None:
        return ()
    flags: list[str] = []
    if abs(obs.pose.yaw) > max_yaw:
        flags.append("extreme_yaw")
    if abs(obs.pose.pitch) > max_pitch:
        flags.append("extreme_pitch")
    if abs(obs.pose.roll) > max_roll:
        flags.append("extreme_roll")
    return tuple(flags)


def assess_pair(
    baseline: FaceObservation,
    makeup: FaceObservation,
    max_yaw: float = DEFAULT_MAX_YAW,
    max_pitch: float = DEFAULT_MAX_PITCH,
    max_roll: float = DEFAULT_MAX_ROLL,
) -> QualityAssessment:
    """Assess the quality of one baseline-vs-makeup pairing.

    Detects unusually large mouth-opening differences, strongly narrowed
    eyes (blinks) and extreme head poses in either image. The pairing is
    kept with a reduced score rather than discarded.

    Returns:
        QualityAssessment with score in [0, 1] and descriptive flags.
    """

    if baseline.normalized is None or makeup.normalized is None:
        raise ValueError("Observations must be pose-normalised before pairing")

    flags: list[str] = []
    for obs in (baseline, makeup):
        flags.extend(assess_observation(obs, max_yaw, max_pitch, max_roll))
    flags = sorted(set(flags))

    base_mouth = mouth_opening(baseline.normalized)
    make_mouth = mouth_opening(makeup.normalized)
    delta = abs(base_mouth - make_mouth)
    smaller = max(min(base_mouth, make_mouth), 1e-9)
    if delta > MOUTH_OPENING_ABS_THRESHOLD or (
        smaller > 1e-6 and max(base_mouth, make_mouth) / smaller > MOUTH_OPENING_RATIO_THRESHOLD
    ):
        flags.append("mouth_opening_mismatch")

    base_eyes = eye_openness(baseline.normalized)
    make_eyes = eye_openness(makeup.normalized)
    min_open = min(base_eyes + make_eyes)
    pairs = zip(base_eyes, make_eyes)
    blink = any(
        (max(a, b) > 1e-6 and min(a, b) / max(a, b) < BLINK_RATIO_THRESHOLD)
        or (min(a, b) < BLINK_ABS_THRESHOLD and abs(a - b) > BLINK_ABS_THRESHOLD)
        for a, b in pairs
    )
    if blink:
        flags.append("possible_blink")

    penalty = sum(_PENALTY.get(f, 0.2) for f in flags)
    score = max(0.0, 1.0 - penalty)
    return QualityAssessment(
        score=score,
        flags=tuple(flags),
        mouth_opening_delta=float(delta),
        min_eye_openness=float(min_open),
    )
