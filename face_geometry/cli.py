"""Command-line interface for the facial geometry change tool.

Usage::

    python -m face_geometry --baseline ./baseline --makeup ./makeup --output ./results
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from .aggregation import (
    aggregate_ratio_changes,
    aggregate_regional,
    compare_consensus,
)
from .alignment import procrustes_align
from .detection import load_observations
from .geometry import compare_shapes, euclidean_displacements, regional_displacements
from .landmarks import FaceLandmarkExtractor, ModelDownloadError
from .models import FaceObservation, PairwiseResult
from .quality import DEFAULT_MAX_PITCH, DEFAULT_MAX_YAW, assess_pair
from .ratios import compute_angles, compute_ratios
from .reporting import (
    build_summary,
    write_feature_changes_csv,
    write_pairwise_csv,
    write_regional_changes_csv,
    write_report_html,
    write_summary_json,
)
from .visualisation import draw_change_map, draw_threshold_map, save_image

logger = logging.getLogger("face_geometry")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(
        prog="face_geometry",
        description=(
            "Measure how much facial landmark geometry changes between two "
            "folders of photographs (no identity recognition)."
        ),
    )
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Folder with baseline (no-treatment) photographs")
    parser.add_argument("--makeup", type=Path, required=True,
                        help="Folder with post-treatment (makeup) photographs")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output folder for CSV/JSON/HTML results")
    parser.add_argument("--recursive", action="store_true",
                        help="Search input folders recursively")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5,
                        help="Minimum face detection confidence (default: 0.5)")
    parser.add_argument("--max-yaw", type=float, default=DEFAULT_MAX_YAW,
                        help="Flag comparisons with |yaw| above this (degrees)")
    parser.add_argument("--max-pitch", type=float, default=DEFAULT_MAX_PITCH,
                        help="Flag comparisons with |pitch| above this (degrees)")
    parser.add_argument("--save-overlays", action="store_true",
                        help="Save annotated geometry change-map images")
    parser.add_argument("--show-landmark-ids", action="store_true",
                        help="Annotate overlays with landmark indices")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for detection (default: 1)")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Path to the MediaPipe face_landmarker.task model")
    parser.add_argument("--warn-threshold", type=float, default=0.02,
                        help="Change-map 'warn' displacement threshold")
    parser.add_argument("--alert-threshold", type=float, default=0.05,
                        help="Change-map 'alert' displacement threshold")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    return parser


def _compare_pair(
    pair: tuple[FaceObservation, FaceObservation], args: argparse.Namespace
) -> PairwiseResult:
    """Module-level worker (picklable) comparing one baseline/makeup pair."""

    return _compare_one(pair[0], pair[1], args)


def _compare_one(
    baseline: FaceObservation, makeup: FaceObservation, args: argparse.Namespace
) -> PairwiseResult:
    """Compare one baseline observation against one makeup observation."""

    assert baseline.normalized is not None and makeup.normalized is not None
    # Shapes are already similarity-normalised to a unit reference distance,
    # so only a rigid (rotation+translation) refinement is applied; re-scaling
    # here would artificially shrink the measured displacement.
    aligned, _, _, _, _ = procrustes_align(
        baseline.normalized, makeup.normalized, allow_scaling=False
    )
    metrics = compare_shapes(baseline.normalized, aligned)
    quality = assess_pair(
        baseline, makeup, max_yaw=args.max_yaw, max_pitch=args.max_pitch
    )
    disp = euclidean_displacements(baseline.normalized, aligned)
    return PairwiseResult(
        baseline_path=baseline.image_path,
        makeup_path=makeup.image_path,
        mean_landmark_displacement=metrics["mean_landmark_displacement"],
        median_landmark_displacement=metrics["median_landmark_displacement"],
        max_landmark_displacement=metrics["max_landmark_displacement"],
        landmark_rmse=metrics["landmark_rmse"],
        procrustes_distance=metrics["procrustes_distance"],
        comparison_quality=quality.score,
        quality_flags=quality.flags,
        regional=regional_displacements(disp),
    )


def run(args: argparse.Namespace) -> dict:
    """Execute the full pipeline and return the summary document."""

    args.output.mkdir(parents=True, exist_ok=True)

    with FaceLandmarkExtractor(
        min_detection_confidence=args.min_detection_confidence,
        model_path=args.model_path,
    ) as extractor:
        baseline_obs = load_observations(
            args.baseline, extractor,
            recursive=args.recursive, max_yaw=args.max_yaw, max_pitch=args.max_pitch,
        )
        makeup_obs = load_observations(
            args.makeup, extractor,
            recursive=args.recursive, max_yaw=args.max_yaw, max_pitch=args.max_pitch,
        )

    if not baseline_obs:
        raise RuntimeError(f"No usable baseline images in {args.baseline}")
    if not makeup_obs:
        raise RuntimeError(f"No usable makeup images in {args.makeup}")

    # Every baseline against every makeup image (N x M comparisons).
    pairs = [(b, m) for b in baseline_obs for m in makeup_obs]
    logger.info("Running %d pairwise comparisons", len(pairs))
    if args.workers > 1:
        worker = partial(_compare_pair, args=args)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pairwise = list(pool.map(worker, pairs))
    else:
        pairwise = [_compare_one(b, m, args) for b, m in pairs]

    base_shapes = [np.asarray(o.normalized) for o in baseline_obs]
    make_shapes = [np.asarray(o.normalized) for o in makeup_obs]
    consensus = compare_consensus(base_shapes, make_shapes)

    base_ratios = [compute_ratios(o.normalized) for o in baseline_obs]
    make_ratios = [compute_ratios(o.normalized) for o in makeup_obs]
    ratio_changes = aggregate_ratio_changes(base_ratios, make_ratios)
    base_angles = [compute_angles(o.normalized) for o in baseline_obs]
    make_angles = [compute_angles(o.normalized) for o in makeup_obs]
    angle_changes = aggregate_ratio_changes(base_angles, make_angles)

    summary = build_summary(
        pairwise, consensus, ratio_changes, len(baseline_obs), len(makeup_obs)
    )
    summary["angle_changes"] = [
        {
            "angle": c.name,
            "baseline_mean": c.baseline_mean,
            "makeup_mean": c.makeup_mean,
            "absolute_change": c.absolute_change,
            "percentage_change": (
                c.percentage_change if c.percentage_change == c.percentage_change else None
            ),
        }
        for c in angle_changes
    ]

    regional = aggregate_regional(pairwise)
    write_pairwise_csv(pairwise, args.output / "pairwise_landmarks.csv")
    write_feature_changes_csv(ratio_changes, args.output / "feature_changes.csv")
    write_regional_changes_csv(regional, args.output / "regional_changes.csv")
    write_summary_json(summary, args.output / "summary.json")

    overlays: list[tuple[str, np.ndarray]] = []
    base_consensus = np.mean(np.stack(base_shapes), axis=0)
    make_consensus = np.mean(np.stack(make_shapes), axis=0)
    change_map = draw_change_map(
        base_consensus, make_consensus, show_landmark_ids=args.show_landmark_ids
    )
    threshold_map = draw_threshold_map(
        base_consensus, make_consensus,
        warn_threshold=args.warn_threshold, alert_threshold=args.alert_threshold,
        show_landmark_ids=args.show_landmark_ids,
    )
    overlays.append(("Consensus geometry change map", change_map))
    overlays.append(("Threshold change map", threshold_map))
    if args.save_overlays:
        save_image(change_map, args.output / "overlays" / "change_map.png")
        save_image(threshold_map, args.output / "overlays" / "threshold_map.png")

    write_report_html(summary, args.output / "report.html", overlays=overlays)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        summary = run(args)
    except ModelDownloadError as exc:
        logger.error("%s", exc)
        return 2
    except (RuntimeError, NotADirectoryError) as exc:
        logger.error("%s", exc)
        return 1

    score = summary["geometry_change_score"]
    consensus = summary["consensus_comparison"]
    print("\n=== FACIAL LANDMARK GEOMETRY CHANGE ===")
    print(f"Geometry change score (engineering metric, not a probability): {score:.1f}/100")
    print(f"Overall Procrustes distance:      {consensus['procrustes_distance']:.5f}")
    print(f"Overall normalised landmark RMSE: {consensus['landmark_rmse']:.5f}")
    print(f"Strongest-changing region:        {summary['strongest_region']['region']}")
    print(f"Weakest-changing region:          {summary['weakest_region']['region']}")
    print(f"Results written to: {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
