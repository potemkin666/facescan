"""Landmark overlay and geometry change-map visualisation.

Two kinds of artefacts are produced:

* **aligned overlay** — baseline landmarks, makeup landmarks after alignment,
  and per-landmark displacement vectors colour-coded by magnitude;
* **threshold map** — only landmarks whose displacement exceeds configurable
  thresholds, to spotlight the dominant changes.

Everything is drawn with OpenCV on a neutral canvas using normalised
coordinates, so the output is deterministic and does not require the source
photographs (the visualisation is a geometry map, not an identity image).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .alignment import procrustes_align

_BASELINE_COLOR = (80, 200, 80)   # green (BGR)
_MAKEUP_COLOR = (220, 120, 40)    # blue (BGR)
_TEXT_COLOR = (230, 230, 230)
_BACKGROUND = (18, 18, 24)


def _canvas_and_transform(
    shapes: list[np.ndarray], size: int, margin: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Create a canvas plus the similarity transform mapping coords to pixels."""

    all_pts = np.concatenate([np.asarray(s)[:, :2] for s in shapes], axis=0)
    lo = all_pts.min(axis=0)
    hi = all_pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    scale = (size * (1.0 - 2 * margin)) / float(span.max())
    offset = (size - span * scale) / 2.0 - lo * scale
    canvas = np.full((size, size, 3), _BACKGROUND, dtype=np.uint8)
    return canvas, offset, scale


def _to_px(point: np.ndarray, offset: np.ndarray, scale: float) -> tuple[int, int]:
    px = point * scale + offset
    return int(round(px[0])), int(round(px[1]))


def _magnitude_color(value: float, vmax: float) -> tuple[int, int, int]:
    """Map 0..vmax to a green→yellow→red BGR ramp."""

    t = 0.0 if vmax <= 0 else float(np.clip(value / vmax, 0.0, 1.0))
    # BGR: interpolate green (0,200,0) -> yellow (0,200,255) -> red (0,0,255)
    if t < 0.5:
        b, g, r = 0, 200, int(255 * (t / 0.5))
    else:
        b, g, r = 0, int(200 * (1 - (t - 0.5) / 0.5)), 255
    return (b, g, r)


def draw_change_map(
    baseline: np.ndarray,
    makeup: np.ndarray,
    *,
    size: int = 900,
    margin: float = 0.08,
    show_landmark_ids: bool = False,
    magnitude_vmax: float | None = None,
    title: str = "Geometry change map",
) -> np.ndarray:
    """Full annotated change map: both shapes + displacement vectors.

    ``makeup`` is optimally aligned to ``baseline`` before drawing.  Every
    landmark is drawn (baseline green, makeup blue) with a displacement
    vector whose colour encodes magnitude; optionally landmark indices are
    annotated.

    Returns:
        BGR image array.
    """

    base = np.asarray(baseline, dtype=np.float64)[:, :2]
    make = np.asarray(makeup, dtype=np.float64)[:, :2]
    aligned, _, _, _, _ = procrustes_align(base, make, allow_scaling=False)
    disp = np.linalg.norm(aligned - base, axis=1)
    if magnitude_vmax is not None:
        vmax = float(magnitude_vmax)
    else:
        vmax = max(float(disp.max()), 1e-9)  # avoid div-by-zero for identical shapes

    canvas, offset, scale = _canvas_and_transform([base, aligned], size, margin)
    for i, (b_pt, m_pt) in enumerate(zip(base, aligned)):
        b_px = _to_px(b_pt, offset, scale)
        m_px = _to_px(m_pt, offset, scale)
        color = _magnitude_color(disp[i], vmax)
        cv2.arrowedLine(canvas, b_px, m_px, color, 1, tipLength=0.25)
        cv2.circle(canvas, b_px, 2, _BASELINE_COLOR, -1)
        cv2.circle(canvas, m_px, 2, _MAKEUP_COLOR, -1)
        if show_landmark_ids and (i % 2 == 0 or disp[i] > 0.5 * vmax):
            cv2.putText(
                canvas, str(i), (b_px[0] + 3, b_px[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, _TEXT_COLOR, 1, cv2.LINE_AA,
            )

    cv2.putText(canvas, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                _TEXT_COLOR, 2, cv2.LINE_AA)
    legend = (
        f"green=baseline  blue=makeup  max disp={vmax:.4f} ref units"
    )
    cv2.putText(canvas, legend, (16, size - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                _TEXT_COLOR, 1, cv2.LINE_AA)
    return canvas


def draw_threshold_map(
    baseline: np.ndarray,
    makeup: np.ndarray,
    *,
    warn_threshold: float = 0.02,
    alert_threshold: float = 0.05,
    size: int = 900,
    margin: float = 0.08,
    show_landmark_ids: bool = True,
) -> np.ndarray:
    """Simplified map of landmarks exceeding displacement thresholds.

    Landmarks below ``warn_threshold`` are omitted entirely; landmarks above
    it are drawn in yellow, and landmarks above ``alert_threshold`` in red,
    with displacement vectors to scale.

    Returns:
        BGR image array plus the count of flagged landmarks in the title.
    """

    base = np.asarray(baseline, dtype=np.float64)[:, :2]
    make = np.asarray(makeup, dtype=np.float64)[:, :2]
    aligned, _, _, _, _ = procrustes_align(base, make, allow_scaling=False)
    disp = np.linalg.norm(aligned - base, axis=1)

    canvas, offset, scale = _canvas_and_transform([base, aligned], size, margin)
    # Faint context: all landmarks in dark grey.
    for pt in base:
        cv2.circle(canvas, _to_px(pt, offset, scale), 1, (70, 70, 70), -1)

    flagged = 0
    for i in np.argsort(disp):  # draw strongest last (on top)
        if disp[i] < warn_threshold:
            continue
        flagged += 1
        color = (0, 80, 255) if disp[i] >= alert_threshold else (0, 220, 255)
        b_px = _to_px(base[i], offset, scale)
        m_px = _to_px(aligned[i], offset, scale)
        cv2.arrowedLine(canvas, b_px, m_px, color, 2, tipLength=0.3)
        cv2.circle(canvas, b_px, 3, _BASELINE_COLOR, -1)
        cv2.circle(canvas, m_px, 3, color, -1)
        if show_landmark_ids:
            cv2.putText(
                canvas, str(i), (b_px[0] + 4, b_px[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
            )

    title = (
        f"Threshold map: {flagged} landmark(s) >= {warn_threshold:.3f} "
        f"(red >= {alert_threshold:.3f})"
    )
    cv2.putText(canvas, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                _TEXT_COLOR, 2, cv2.LINE_AA)
    return canvas


def save_image(image: np.ndarray, path: Path) -> Path:
    """Write a BGR image to ``path`` (PNG), creating parent directories."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Failed to write image: {path}")
    return path
