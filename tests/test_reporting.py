"""Tests for CSV/JSON/HTML output writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from face_geometry.aggregation import (
    aggregate_ratio_changes,
    aggregate_regional,
    compare_consensus,
    geometry_change_score,
)
from face_geometry.geometry import euclidean_displacements, regional_displacements
from face_geometry.models import PairwiseResult
from face_geometry.reporting import (
    build_summary,
    write_feature_changes_csv,
    write_pairwise_csv,
    write_regional_changes_csv,
    write_report_html,
    write_summary_json,
)


@pytest.fixture()
def example_results(
    synthetic_face: np.ndarray, moved_eyebrows_face: np.ndarray
) -> tuple[list[PairwiseResult], object, list]:
    disp = euclidean_displacements(synthetic_face, moved_eyebrows_face)
    pairwise = [
        PairwiseResult(
            baseline_path=f"baseline_{i}.jpg",
            makeup_path=f"makeup_{j}.jpg",
            mean_landmark_displacement=float(disp.mean()),
            median_landmark_displacement=float(np.median(disp)),
            max_landmark_displacement=float(disp.max()),
            landmark_rmse=float(np.sqrt(np.mean(disp**2))),
            procrustes_distance=0.01,
            comparison_quality=0.8,
            quality_flags=("mouth_opening_mismatch",),
            regional=regional_displacements(disp),
        )
        for i in range(2)
        for j in range(2)
    ]
    consensus = compare_consensus([synthetic_face], [moved_eyebrows_face])
    changes = aggregate_ratio_changes(
        [{"mouth_width_over_face_width": 0.4}], [{"mouth_width_over_face_width": 0.42}]
    )
    return pairwise, consensus, changes


class TestCsvWriters:
    def test_pairwise_csv(self, example_results, tmp_path: Path) -> None:
        pairwise, _, _ = example_results
        out = write_pairwise_csv(pairwise, tmp_path / "pairwise_landmarks.csv")
        rows = list(csv.DictReader(out.open()))
        assert len(rows) == 4
        assert rows[0]["baseline_image"] == "baseline_0.jpg"
        assert "mean_landmark_displacement" in rows[0]
        assert "comparison_quality" in rows[0]
        assert rows[0]["quality_flags"] == "mouth_opening_mismatch"

    def test_feature_changes_csv(self, example_results, tmp_path: Path) -> None:
        _, _, changes = example_results
        out = write_feature_changes_csv(changes, tmp_path / "feature_changes.csv")
        rows = list(csv.DictReader(out.open()))
        assert rows[0]["feature"] == "mouth_width_over_face_width"
        assert float(rows[0]["absolute_change"]) == pytest.approx(0.02)
        assert float(rows[0]["percentage_change"]) == pytest.approx(5.0)

    def test_regional_changes_csv(self, example_results, tmp_path: Path) -> None:
        pairwise, _, _ = example_results
        regional = aggregate_regional(pairwise)
        out = write_regional_changes_csv(regional, tmp_path / "regional_changes.csv")
        rows = {r["region"]: r for r in csv.DictReader(out.open())}
        assert "left_eyebrow" in rows
        brow = float(rows["left_eyebrow"]["mean_displacement"])
        jaw = float(rows["jawline"]["mean_displacement"])
        assert brow > jaw


class TestSummaryJson:
    def test_roundtrip(self, example_results, tmp_path: Path) -> None:
        pairwise, consensus, changes = example_results
        summary = build_summary(pairwise, consensus, changes, 2, 2)
        out = write_summary_json(summary, tmp_path / "summary.json")
        data = json.loads(out.read_text())
        assert data["inputs"]["n_pairwise_comparisons"] == 4
        assert data["strongest_region"]["region"] == "Eyebrows"
        assert "geometry_change_score_definition" in data
        assert 0.0 <= data["geometry_change_score"] <= 100.0

    def test_score_matches_consensus(self, example_results) -> None:
        pairwise, consensus, changes = example_results
        summary = build_summary(pairwise, consensus, changes, 2, 2)
        assert summary["geometry_change_score"] == pytest.approx(
            geometry_change_score(consensus)
        )


class TestHtmlReport:
    def test_report_contains_required_sections(
        self, example_results, tmp_path: Path, synthetic_face: np.ndarray,
        moved_eyebrows_face: np.ndarray
    ) -> None:
        pairwise, consensus, changes = example_results
        summary = build_summary(pairwise, consensus, changes, 2, 2)
        from face_geometry.visualisation import draw_change_map

        overlays = [("map", draw_change_map(synthetic_face, moved_eyebrows_face, size=300))]
        out = write_report_html(summary, tmp_path / "report.html", overlays=overlays)
        html = out.read_text()
        assert "FACIAL LANDMARK GEOMETRY CHANGE" in html
        assert "Procrustes distance" in html
        assert "landmark RMSE" in html.lower() or "Landmark RMSE" in html
        assert "Strongest-changing region" in html
        assert "Weakest-changing region" in html
        assert "Eyebrows" in html
        assert "<table" in html
        assert "data:image/png;base64," in html
        assert "not a probability" in html.lower()
