"""Tests for image discovery and observation loading (no model needed)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from face_geometry.detection import SUPPORTED_EXTENSIONS, find_images, load_image_bgr


def _write_image(path: Path) -> None:
    image = np.full((20, 20, 3), 127, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


class TestFindImages:
    def test_supported_extensions(self, tmp_path: Path) -> None:
        for name in ("a.jpg", "b.jpeg", "c.png", "d.webp"):
            _write_image(tmp_path / name)
        (tmp_path / "notes.txt").write_text("not an image")
        found = find_images(tmp_path)
        assert [p.name for p in found] == ["a.jpg", "b.jpeg", "c.png", "d.webp"]

    def test_recursive(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "top.jpg")
        _write_image(tmp_path / "nested" / "deep" / "inner.png")
        assert len(find_images(tmp_path, recursive=False)) == 1
        assert len(find_images(tmp_path, recursive=True)) == 2

    def test_sorted_output(self, tmp_path: Path) -> None:
        for name in ("z.png", "m.png", "a.png"):
            _write_image(tmp_path / name)
        assert [p.name for p in find_images(tmp_path)] == ["a.png", "m.png", "z.png"]

    def test_missing_folder_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError):
            find_images(tmp_path / "nope")

    def test_supported_extensions_constant(self) -> None:
        assert SUPPORTED_EXTENSIONS == {".jpg", ".jpeg", ".png", ".webp"}


class TestLoadImage:
    def test_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "x.png"
        _write_image(path)
        image = load_image_bgr(path)
        assert image.shape == (20, 20, 3)

    def test_undecodable_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.png"
        path.write_bytes(b"this is not a png")
        with pytest.raises(ValueError, match="Cannot decode"):
            load_image_bgr(path)
