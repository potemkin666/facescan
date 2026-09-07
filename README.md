# facescan — Facial Landmark Geometry Change Measurement

A Python tool that measures **how much facial landmark geometry appears to
change** between two folders of photographs: a `baseline/` set (before) and a
`makeup/` set (after a treatment such as makeup).

> **This tool performs no identity recognition.** It quantifies changes in
> detected facial landmark geometry and produces interpretable measurements.

Built on Python 3.12+, OpenCV and MediaPipe's dense **Face Landmarker /
Face Mesh** (468 landmarks covering eyebrows, eyelids, eye corners, nose
bridge and nostrils, lips, jawline, cheeks and the forehead/face contour).

---

## Installation

```bash
pip install -e .            # package
pip install -e '.[test]'    # with test dependencies
```

The MediaPipe `face_landmarker.task` model (~30 MB) is downloaded
automatically on first use into `~/.cache/face_geometry/models`. To use a
local copy instead, set `FACE_GEOMETRY_MODEL_PATH=/path/to/face_landmarker.task`
or pass `--model-path`.

## Usage

```bash
python -m face_geometry --baseline ./baseline --makeup ./makeup --output ./results
```

Both folders are scanned for `.jpg`, `.jpeg`, `.png` and `.webp` files. The
primary face is detected in every image; images with no usable face are
skipped (with a log message).

### Options

| Option | Meaning |
| --- | --- |
| `--recursive` | Search the input folders recursively. |
| `--min-detection-confidence 0.7` | Minimum face detection confidence. |
| `--max-yaw 25` | Flag comparisons where either face has `|yaw|` above this. |
| `--max-pitch 20` | Flag comparisons where either face has `|pitch|` above this. |
| `--save-overlays` | Save the annotated change-map images. |
| `--show-landmark-ids` | Annotate overlays with landmark indices. |
| `--workers 8` | Parallel workers for the pairwise comparisons. |
| `--model-path PATH` | Local path to the MediaPipe model file. |
| `--warn-threshold 0.02` | Change-map 'warn' displacement threshold. |
| `--alert-threshold 0.05` | Change-map 'alert' displacement threshold. |
| `--verbose` | Debug logging. |

---

## How it works

1. **Load images** from both folders.
2. **Detect the primary face** in each image with MediaPipe Face Landmarker.
3. **Extract 468 dense landmarks.**
4. **Pose normalisation (critical).** Each face is similarity-normalised —
   translated to the eye-corner midpoint, rotated so the eye axis is
   horizontal, and scaled by the stable reference distance (the outer
   eye-corner / bi-canthal distance). Differences caused only by image scale,
   translation, head tilt, camera distance or moderate yaw/pitch are therefore
   *not* interpreted as shape changes. Both raw and normalised coordinates
   are kept.
5. **Pairwise comparison.** Every usable baseline image is compared against
   every usable makeup image (N×M). For each pair the tool computes:
   `mean_landmark_displacement`, `median_landmark_displacement`,
   `max_landmark_displacement`, `landmark_rmse` and `procrustes_distance`
   (residual shape difference after optimal translation, rotation and scale).
6. **Regional displacement** is computed separately for `left_eye`,
   `right_eye`, `left_eyebrow`, `right_eyebrow`, `nose`, `upper_lip`,
   `lower_lip`, `mouth`, `jawline`, `left_cheek` and `right_cheek`
   (mean / median / max / RMSE each), because a treatment may strongly alter
   one feature while leaving the rest of the face almost unchanged.
7. **Facial ratios & angles** (eye width / interocular distance, eye height /
   width, eyebrow-to-eye distance, eyebrow angle, nose width / face width,
   mouth width / face width, lip heights, jaw width / face width, facial
   length / width, eyebrow slope, eye-axis and jaw contour angles, …) are
   computed on normalised coordinates and reported as
   `baseline_mean`, `makeup_mean`, `absolute_change`, `percentage_change`.
8. **Geometry change map.** Per-landmark average displacement is visualised
   as an annotated overlay (baseline landmarks, aligned makeup landmarks and
   displacement vectors colour-coded by magnitude), plus a simplified map
   showing only landmarks above configurable thresholds.
9. **Multiple photographs.** Every baseline is compared with every makeup
   image, then mean / median / std / 5th / 95th percentile / min / max are
   reported. A per-folder **consensus shape** (Procrustes-aligned average) is
   also computed and compared — this is a primary measurement because
   individual photographs contain pose/expression noise.
10. **Expression vs geometry.** The tool flags (but never discards)
    comparisons with unusually large mouth-opening differences, possible
    blinks / strongly narrowed eyes, or extreme head poses, storing a
    `comparison_quality` score with each pairing.

## Outputs

| File | Contents |
| --- | --- |
| `pairwise_landmarks.csv` | One row per baseline×makeup image pair. |
| `feature_changes.csv` | Human-readable ratio changes. |
| `regional_changes.csv` | Displacement aggregated by facial region. |
| `summary.json` | Full machine-readable results. |
| `report.html` | Well-formatted visual report. |
| `overlays/` | Change-map images (with `--save-overlays`). |

The HTML report prominently shows **FACIAL LANDMARK GEOMETRY CHANGE**, the
overall Procrustes distance, normalised landmark RMSE, average landmark
displacement, the strongest- and weakest-changing regions, a regional table,
feature-ratio changes and the aligned landmark overlays.

---

## Geometry change score (0–100)

The `geometry_change_score` is an **engineering summary metric, NOT a
probability**. It is defined mathematically from the consensus comparison as

```
d95   = 95th percentile of the per-landmark displacement field
        between the two consensus shapes (reference-distance units)
score = 100 * clip( (d95 - D0) / (D1 - D0), 0, 1 )
D0    = 0.002   (noise floor, 0.2% of the reference distance)
D1    = 0.10    (saturation, 10% of the reference distance)
```

* **0** = virtually identical detected geometry after normalisation.
* **100** = extremely large detected geometry difference.

The calibration constants and every intermediate quantity (`d95`, mean/median/
max displacement, RMSE, Procrustes distance) are stored in `summary.json`
under `geometry_change_score_definition` and the consensus section, so the
score is fully auditable against the raw measurements.

---

## Repository layout

```
face_geometry/
    __init__.py
    __main__.py
    cli.py            # command-line interface & pipeline orchestration
    detection.py      # image discovery, observation loading
    landmarks.py      # MediaPipe Face Landmarker extraction + head pose
    alignment.py      # pose normalisation & Procrustes alignment
    regions.py        # facial-region -> landmark-index definitions
    geometry.py       # displacement metrics, regional stats
    ratios.py         # facial ratios, angles, expression signals
    aggregation.py    # pairwise aggregation, consensus, change score
    quality.py        # expression/pose quality flagging
    visualisation.py  # change-map overlays
    reporting.py      # CSV / JSON / HTML writers
    models.py         # shared dataclasses
tests/
README.md
pyproject.toml
.gitignore
```

## Testing

```bash
pytest
```

The tests use **synthetic landmark fixtures** (a deterministic face-like
468-point layout), so no photographs of real people are required. Covered:
Euclidean displacement, normalisation, Procrustes alignment, scale /
translation / rotation invariance, facial ratios, regional aggregation and
consensus-shape computation — including the two key mathematical guarantees:

* translating, scaling and rotating an identical landmark configuration
  yields a post-alignment geometry difference that approaches zero;
* moving only the eyebrow landmarks makes the tool report the eyebrows as the
  dominant changed region.
