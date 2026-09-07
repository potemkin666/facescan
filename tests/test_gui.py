"""Tests for the non-UI logic in the desktop GUI module."""

from __future__ import annotations

import logging
import queue

import pytest

pytest.importorskip("tkinter")

from face_geometry.gui import _LogHandler, _cli_default  # noqa: E402


def test_log_handler_pushes_formatted_records_onto_queue() -> None:
    log_queue: "queue.Queue[str]" = queue.Queue()
    handler = _LogHandler(log_queue)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger = logging.getLogger("face_geometry.gui.test")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        logger.info("hello %s", "world")
    finally:
        logger.removeHandler(handler)

    assert log_queue.get_nowait() == "INFO: hello world"
    with pytest.raises(queue.Empty):
        log_queue.get_nowait()


def test_cli_default_matches_argparse_defaults() -> None:
    from face_geometry.cli import build_parser

    parser = build_parser()
    for name in (
        "min_detection_confidence",
        "workers",
        "warn_threshold",
        "alert_threshold",
        "max_yaw",
        "max_pitch",
        "show_landmark_ids",
        "model_path",
        "verbose",
    ):
        assert _cli_default(name) == parser.get_default(name)
