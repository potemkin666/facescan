"""Output writers: CSVs, machine-readable JSON and the HTML report.

Four artefacts plus optional overlay images are produced:

* ``pairwise_landmarks.csv`` — one row per baseline×makeup image pair;
* ``feature_changes.csv``   — human-readable ratio changes;
* ``regional_changes.csv``  — displacement aggregated by facial region;
* ``summary.json``          — full machine-readable results;
* ``report.html``           — well-formatted visual report.
"""

from __future__ import annotations

import base64
import csv
import io
import json
from dataclasses import asdict
from html import escape
from pathlib import Path

import cv2
import numpy as np

from .aggregation import (
    SCORE_NOISE_FLOOR,
    SCORE_SATURATION,
    aggregate_pairwise,
    aggregate_regional,
    geometry_change_score,
)
from .models import ConsensusComparison, PairwiseResult, RatioChange
from .regions import REPORT_REGION_GROUPS


def _png_data_url(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise IOError("Could not encode overlay image as PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode()


def _fmt(value: float, digits: int = 5) -> str:
    if value != value:  # NaN
        return "n/a"
    return f"{value:.{digits}f}"


def write_pairwise_csv(results: list[PairwiseResult], path: Path) -> Path:
    """Write one row per baseline-vs-makeup image pair."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "baseline_image",
        "makeup_image",
        "mean_landmark_displacement",
        "median_landmark_displacement",
        "max_landmark_displacement",
        "landmark_rmse",
        "procrustes_distance",
        "comparison_quality",
        "quality_flags",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "baseline_image": r.baseline_path,
                    "makeup_image": r.makeup_path,
                    "mean_landmark_displacement": f"{r.mean_landmark_displacement:.6f}",
                    "median_landmark_displacement": f"{r.median_landmark_displacement:.6f}",
                    "max_landmark_displacement": f"{r.max_landmark_displacement:.6f}",
                    "landmark_rmse": f"{r.landmark_rmse:.6f}",
                    "procrustes_distance": f"{r.procrustes_distance:.6f}",
                    "comparison_quality": f"{r.comparison_quality:.3f}",
                    "quality_flags": ";".join(r.quality_flags),
                }
            )
    return path


def write_feature_changes_csv(changes: list[RatioChange], path: Path) -> Path:
    """Write human-readable feature-ratio changes."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature",
        "baseline_mean",
        "makeup_mean",
        "absolute_change",
        "percentage_change",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for c in changes:
            writer.writerow(
                {
                    "feature": c.name,
                    "baseline_mean": f"{c.baseline_mean:.6f}",
                    "makeup_mean": f"{c.makeup_mean:.6f}",
                    "absolute_change": f"{c.absolute_change:.6f}",
                    "percentage_change": (
                        f"{c.percentage_change:.3f}"
                        if c.percentage_change == c.percentage_change
                        else "n/a"
                    ),
                }
            )
    return path


def write_regional_changes_csv(
    regional: dict[str, dict[str, float]], path: Path
) -> Path:
    """Write displacement aggregated by facial region."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "region",
        "mean_displacement",
        "median_displacement",
        "max_displacement",
        "rmse",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for region, stats in sorted(regional.items()):
            writer.writerow(
                {
                    "region": region,
                    "mean_displacement": f"{stats['mean_displacement']:.6f}",
                    "median_displacement": f"{stats['median_displacement']:.6f}",
                    "max_displacement": f"{stats['max_displacement']:.6f}",
                    "rmse": f"{stats['rmse']:.6f}",
                }
            )
    return path


def write_summary_json(summary: dict, path: Path) -> Path:
    """Write the full machine-readable summary."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(obj: object) -> object:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        if isinstance(obj, float) and obj != obj:
            return None
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=default, sort_keys=True)
    return path


def build_summary(
    pairwise: list[PairwiseResult],
    consensus: ConsensusComparison,
    ratio_changes: list[RatioChange],
    n_baseline: int,
    n_makeup: int,
) -> dict:
    """Assemble the machine-readable summary document."""

    score = geometry_change_score(consensus)
    regional = aggregate_regional(pairwise)
    grouped = _grouped_region_table(regional)
    strongest = max(grouped, key=lambda g: g["mean"])
    weakest = min(grouped, key=lambda g: g["mean"])
    return {
        "tool": "face_geometry",
        "description": (
            "Facial landmark geometry change measurement (no identity recognition)"
        ),
        "inputs": {
            "n_baseline_images": n_baseline,
            "n_makeup_images": n_makeup,
            "n_pairwise_comparisons": len(pairwise),
        },
        "geometry_change_score": score,
        "geometry_change_score_definition": {
            "kind": "engineering summary metric, NOT a probability",
            "formula": (
                "100 * clip((d95 - D0) / (D1 - D0), 0, 1) over the per-landmark "
                "displacement field between the two consensus shapes"
            ),
            "D0_noise_floor": SCORE_NOISE_FLOOR,
            "D1_saturation": SCORE_SATURATION,
            "meaning": (
                "0 = virtually identical detected geometry after normalisation; "
                "100 = extremely large detected geometry difference"
            ),
        },
        "pairwise_aggregates": aggregate_pairwise(pairwise),
        "consensus_comparison": {
            "mean_landmark_displacement": consensus.mean_landmark_displacement,
            "median_landmark_displacement": consensus.median_landmark_displacement,
            "max_landmark_displacement": consensus.max_landmark_displacement,
            "landmark_rmse": consensus.landmark_rmse,
            "procrustes_distance": consensus.procrustes_distance,
        },
        "strongest_region": strongest,
        "weakest_region": weakest,
        "regional_changes": regional,
        "region_groups": grouped,
        "feature_changes": [
            {
                "feature": c.name,
                "baseline_mean": c.baseline_mean,
                "makeup_mean": c.makeup_mean,
                "absolute_change": c.absolute_change,
                "percentage_change": (
                    c.percentage_change
                    if c.percentage_change == c.percentage_change
                    else None
                ),
            }
            for c in ratio_changes
        ],
        "pairwise": [
            {
                "baseline_image": r.baseline_path,
                "makeup_image": r.makeup_path,
                "mean_landmark_displacement": r.mean_landmark_displacement,
                "median_landmark_displacement": r.median_landmark_displacement,
                "max_landmark_displacement": r.max_landmark_displacement,
                "landmark_rmse": r.landmark_rmse,
                "procrustes_distance": r.procrustes_distance,
                "comparison_quality": r.comparison_quality,
                "quality_flags": list(r.quality_flags),
            }
            for r in pairwise
        ],
    }


def _grouped_region_table(regional: dict[str, dict[str, float]]) -> list[dict]:
    """Combine fine-grained regions into the coarse report groups."""

    rows: list[dict] = []
    for label, members in REPORT_REGION_GROUPS.items():
        present = [regional[m] for m in members if m in regional]
        if not present:
            continue
        mean = float(np.mean([p["mean_displacement"] for p in present]))
        rmse = float(np.mean([p["rmse"] for p in present]))
        rows.append({"region": label, "mean": mean, "rmse": rmse})
    grand = float(np.mean([r["mean"] for r in rows])) if rows else 0.0
    for row in rows:
        row["relative_change"] = row["mean"] / grand if grand > 1e-12 else 0.0
    return rows


def write_report_html(
    summary: dict,
    path: Path,
    overlays: list[tuple[str, np.ndarray]] | None = None,
) -> Path:
    """Write the well-formatted HTML report.

    Prominently shows FACIAL LANDMARK GEOMETRY CHANGE, the headline metrics,
    the region table, feature-ratio changes and aligned landmark overlays.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    score = summary["geometry_change_score"]
    pairwise_agg = summary["pairwise_aggregates"]
    consensus = summary["consensus_comparison"]
    strongest = summary["strongest_region"]
    weakest = summary["weakest_region"]

    region_rows = "\n".join(
        "<tr><td>{region}</td><td>{mean}</td><td>{rmse}</td><td>{rel}x</td></tr>".format(
            region=escape(str(r["region"])),
            mean=escape(_fmt(r["mean"])),
            rmse=escape(_fmt(r["rmse"])),
            rel=escape(f"{float(r['relative_change']):.2f}"),
        )
        for r in summary["region_groups"]
    )

    ratio_rows = "\n".join(
        "<tr><td>{name}</td><td>{b}</td><td>{m}</td><td>{d}</td><td>{p}</td></tr>".format(
            name=escape(c["feature"]),
            b=_fmt(c["baseline_mean"]),
            m=_fmt(c["makeup_mean"]),
            d=_fmt(c["absolute_change"]),
            p=(
                f"{c['percentage_change']:+.1f}%"
                if c["percentage_change"] is not None
                else "n/a"
            ),
        )
        for c in summary["feature_changes"]
    )

    overlay_html = ""
    if overlays:
        figures = "\n".join(
            '<figure><img src="{data}" alt="{name}"/>'
            "<figcaption>{name}</figcaption></figure>".format(
                data=_png_data_url(img), name=escape(name)
            )
            for name, img in overlays
        )
        overlay_html = f"""
    <h2>Aligned landmark overlays</h2>
    <div class="overlays">{figures}</div>"""

    css = """
    body{font-family:system-ui,Arial,sans-serif;margin:2rem auto;max-width:1000px;
         color:#1c1e21;background:#f7f8fa;line-height:1.45}
    h1{background:#0b3050;color:#fff;padding:1rem 1.4rem;border-radius:8px;
       letter-spacing:.05em}
    h2{border-bottom:2px solid #0b3050;padding-bottom:.2rem;margin-top:2.2rem}
    .cards{display:flex;flex-wrap:wrap;gap:1rem;margin:1rem 0}
    .card{background:#fff;border:1px solid #d7dce2;border-radius:8px;
          padding:.8rem 1.1rem;min-width:170px;flex:1}
    .card .value{font-size:1.5rem;font-weight:700;color:#0b3050}
    .card .label{font-size:.82rem;color:#5a636e;text-transform:uppercase}
    table{border-collapse:collapse;width:100%;margin:.8rem 0;background:#fff}
    th,td{border:1px solid #d7dce2;padding:.45rem .7rem;text-align:right}
    th:first-child,td:first-child{text-align:left}
    th{background:#eef2f6}
    .overlays{display:flex;flex-wrap:wrap;gap:1.2rem}
    .overlays figure{margin:0;background:#fff;border:1px solid #d7dce2;
        border-radius:8px;padding:.6rem;max-width:460px}
    .overlays img{max-width:100%;height:auto;display:block}
    figcaption{font-size:.85rem;color:#5a636e;padding-top:.4rem}
    .note{background:#fff8e1;border:1px solid #f0d98c;border-radius:8px;
          padding:.7rem 1rem;font-size:.9rem}
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Facial Landmark Geometry Change Report</title>
<style>{css}</style>
</head>
<body>
<h1>FACIAL LANDMARK GEOMETRY CHANGE</h1>
<p class="note">This report quantifies changes in <em>detected facial landmark
geometry</em> between the baseline and makeup photograph sets. It performs
<strong>no identity recognition</strong>. All distances are expressed in units
of the stable reference distance (outer eye corner distance) after
similarity/Procrustes pose normalisation.</p>

<div class="cards">
  <div class="card"><div class="value">{score:.1f} / 100</div>
    <div class="label">Geometry change score</div></div>
  <div class="card"><div class="value">{_fmt(consensus["procrustes_distance"])}</div>
    <div class="label">Overall Procrustes distance</div></div>
  <div class="card"><div class="value">{_fmt(consensus["landmark_rmse"])}</div>
    <div class="label">Overall normalised landmark RMSE</div></div>
  <div class="card"><div class="value">{_fmt(pairwise_agg["mean_landmark_displacement"]["mean"])}</div>
    <div class="label">Average landmark displacement</div></div>
  <div class="card"><div class="value">{escape(str(strongest["region"]))}</div>
    <div class="label">Strongest-changing region</div></div>
  <div class="card"><div class="value">{escape(str(weakest["region"]))}</div>
    <div class="label">Weakest-changing region</div></div>
</div>

<h2>Geometry change score (0-100)</h2>
<p>Engineering summary metric, <strong>not a probability</strong>. Defined as
<code>100 * clip((d95 &minus; D0) / (D1 &minus; D0), 0, 1)</code> where
<code>d95</code> is the 95th percentile of per-landmark displacement between
the two consensus shapes, <code>D0 = {SCORE_NOISE_FLOOR}</code> (noise floor)
and <code>D1 = {SCORE_SATURATION}</code> (saturation). 0 = virtually identical
detected geometry after normalisation; 100 = extremely large detected geometry
difference. Raw measurements are preserved in <code>summary.json</code> for
auditability.</p>

<h2>Regional displacement</h2>
<table>
<thead><tr><th>Region</th><th>Mean displacement</th><th>RMSE</th>
<th>Relative change</th></tr></thead>
<tbody>
{region_rows}
</tbody>
</table>

<h2>Feature-ratio changes</h2>
<table>
<thead><tr><th>Feature ratio</th><th>Baseline mean</th><th>Makeup mean</th>
<th>Absolute change</th><th>Percentage change</th></tr></thead>
<tbody>
{ratio_rows}
</tbody>
</table>

<h2>Consensus comparison (primary measurement)</h2>
<table>
<thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Mean landmark displacement</td><td>{_fmt(consensus["mean_landmark_displacement"])}</td></tr>
<tr><td>Median landmark displacement</td><td>{_fmt(consensus["median_landmark_displacement"])}</td></tr>
<tr><td>Max landmark displacement</td><td>{_fmt(consensus["max_landmark_displacement"])}</td></tr>
<tr><td>Landmark RMSE</td><td>{_fmt(consensus["landmark_rmse"])}</td></tr>
<tr><td>Procrustes distance</td><td>{_fmt(consensus["procrustes_distance"])}</td></tr>
</tbody>
</table>

<h2>Pairwise statistics</h2>
<table>
<thead><tr><th>Metric</th><th>Mean</th><th>Median</th><th>Std</th>
<th>5th pct</th><th>95th pct</th><th>Min</th><th>Max</th></tr></thead>
<tbody>
{"".join(
    "<tr><td>{m}</td><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td><td>{e}</td><td>{f}</td><td>{g}</td></tr>".format(
        m=escape(metric), a=_fmt(s["mean"]), b=_fmt(s["median"]), c=_fmt(s["std"]),
        d=_fmt(s["p05"]), e=_fmt(s["p95"]), f=_fmt(s["min"]), g=_fmt(s["max"]),
    )
    for metric, s in pairwise_agg.items()
)}
</tbody>
</table>
{overlay_html}
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path
