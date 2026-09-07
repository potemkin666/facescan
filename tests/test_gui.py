"""Tests for the non-UI logic in the desktop GUI module."""

from __future__ import annotations

import logging
import queue
from pathlib import Path

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


@pytest.fixture()
def app():
    from tkinter import TclError, Tk

    from face_geometry.gui import FaceGeometryApp

    try:
        root = Tk()
    except TclError as exc:  # pragma: no cover - no display available
        pytest.skip(f"no display available for tkinter: {exc}")
    try:
        yield FaceGeometryApp(root)
    finally:
        root.destroy()


def test_run_pipeline_success_updates_state_and_logs_score(app, monkeypatch) -> None:
    import argparse

    # ``run`` is imported lazily inside ``_run_pipeline`` from ``face_geometry.cli``.
    monkeypatch.setattr(
        "face_geometry.cli.run", lambda args: {"geometry_change_score": 42.0}
    )

    args = argparse.Namespace(output=Path("/tmp/does-not-matter"), verbose=False)
    app._run_pipeline(args)
    app._drain_log_queue()

    assert app._result_output == args.output
    assert "Done" in app.status_var.get()
    log_contents = app.log_text.get("1.0", "end")
    assert "Geometry change score: 42.0" in log_contents


def test_run_pipeline_failure_logs_traceback(app, monkeypatch) -> None:
    import argparse

    def _boom(_args):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("face_geometry.cli.run", _boom)

    args = argparse.Namespace(output=Path("/tmp/does-not-matter"), verbose=False)
    app._run_pipeline(args)
    app._drain_log_queue()

    assert "Failed" in app.status_var.get()
    log_contents = app.log_text.get("1.0", "end")
    assert "RuntimeError: kaboom" in log_contents
