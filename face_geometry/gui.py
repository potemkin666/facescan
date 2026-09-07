"""Desktop GUI launcher for the facial geometry change tool.

A small, dependency-free (stdlib ``tkinter`` only) desktop application that
wraps :func:`face_geometry.cli.run`. It lets a user pick the ``baseline`` and
``makeup`` folders, an output folder, tweak the most common options, run the
pipeline, watch progress in a log pane, and open the finished HTML report —
without touching a terminal.

Visual design ("Moonlit Shore"): a deep indigo/violet night-sky palette with
a pale moonglow accent and a silvery moonlit-water highlight, matching the
reference aesthetic supplied for this tool. The same palette is reused for
the generated ``report.html`` (see :mod:`face_geometry.reporting`).

Run with::

    python -m face_geometry.gui
    face-geometry-gui          # after ``pip install -e .``

Package as a standalone double-clickable desktop app with PyInstaller::

    pyinstaller packaging/face-geometry-gui.spec
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import webbrowser
from pathlib import Path
from tkinter import BooleanVar, StringVar, Text, Tk, filedialog, ttk

from .quality import DEFAULT_MAX_PITCH, DEFAULT_MAX_YAW

# Shared with the CLI so the GUI's defaults never silently drift out of sync
# (see ``face_geometry.cli.build_parser``).
_CLI_DEFAULTS = None


def _cli_default(name: str):
    """Look up an argparse default from ``face_geometry.cli.build_parser``."""

    global _CLI_DEFAULTS
    if _CLI_DEFAULTS is None:
        from .cli import build_parser

        _CLI_DEFAULTS = build_parser()
    return _CLI_DEFAULTS.get_default(name)

# --- "Moonlit Shore" palette -------------------------------------------------
NIGHT_SKY = "#181233"          # deep indigo background
NIGHT_SKY_ALT = "#241a4d"      # secondary panel background
HORIZON = "#392e63"            # distant mountains / mid-tone panels
MOONGLOW = "#f6efd9"           # warm pale moon / primary text on dark
MOONLIGHT = "#cfe3f7"          # cool moonlit-water highlight / accents
MOONLIGHT_DIM = "#9aa8c9"      # secondary / muted text
SILHOUETTE = "#0d0a1f"         # near-black palm silhouette
ACCENT = "#8b7fd1"             # lavender accent (buttons, borders)
ACCENT_ACTIVE = "#a89bef"      # hover/active accent
ERROR = "#e08a8a"

FONT_FAMILY = "Georgia"


class _LogHandler(logging.Handler):
    """Logging handler that pushes formatted records onto a queue."""

    def __init__(self, log_queue: "queue.Queue[str]") -> None:
        super().__init__()
        self._queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - UI glue
        self._queue.put(self.format(record))


class FaceGeometryApp:
    """Main application window."""

    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Face Geometry \u2014 Moonlit Shore")
        self.root.configure(bg=NIGHT_SKY)
        self.root.geometry("760x620")
        self.root.minsize(680, 560)

        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._worker: threading.Thread | None = None
        self._result_output: Path | None = None

        self.baseline_var = StringVar()
        self.makeup_var = StringVar()
        self.output_var = StringVar()
        self.recursive_var = BooleanVar(value=False)
        self.save_overlays_var = BooleanVar(value=True)
        self.status_var = StringVar(value="Ready.")

        self._build_style()
        self._build_layout()
        self.root.after(150, self._drain_log_queue)

    # -- UI construction ----------------------------------------------------
    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:  # pragma: no cover - platform dependent
            pass

        style.configure("Moon.TFrame", background=NIGHT_SKY)
        style.configure("Panel.TFrame", background=NIGHT_SKY_ALT)
        style.configure(
            "Moon.TLabel", background=NIGHT_SKY, foreground=MOONGLOW,
            font=(FONT_FAMILY, 11),
        )
        style.configure(
            "Title.TLabel", background=NIGHT_SKY, foreground=MOONGLOW,
            font=(FONT_FAMILY, 20, "bold"),
        )
        style.configure(
            "Subtitle.TLabel", background=NIGHT_SKY, foreground=MOONLIGHT,
            font=(FONT_FAMILY, 11, "italic"),
        )
        style.configure(
            "Status.TLabel", background=NIGHT_SKY, foreground=MOONLIGHT_DIM,
            font=(FONT_FAMILY, 10),
        )
        style.configure(
            "Moon.TEntry", fieldbackground=HORIZON, foreground=MOONGLOW,
            insertcolor=MOONGLOW, borderwidth=0,
        )
        style.configure(
            "Moon.TCheckbutton", background=NIGHT_SKY, foreground=MOONLIGHT,
            font=(FONT_FAMILY, 10),
        )
        style.map("Moon.TCheckbutton", background=[("active", NIGHT_SKY)])
        style.configure(
            "Moon.TButton", background=ACCENT, foreground=SILHOUETTE,
            borderwidth=0, font=(FONT_FAMILY, 10, "bold"), padding=8,
        )
        style.map(
            "Moon.TButton",
            background=[("active", ACCENT_ACTIVE), ("disabled", HORIZON)],
            foreground=[("disabled", MOONLIGHT_DIM)],
        )
        style.configure(
            "Horizon.Horizontal.TProgressbar",
            troughcolor=HORIZON, background=MOONLIGHT, bordercolor=NIGHT_SKY,
            lightcolor=MOONLIGHT, darkcolor=MOONLIGHT,
        )

    def _build_layout(self) -> None:
        header = ttk.Frame(self.root, style="Moon.TFrame")
        header.pack(fill="x", padx=24, pady=(20, 8))
        ttk.Label(header, text="FACE GEOMETRY", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Measure facial landmark geometry change \u2014 under a moonlit shore.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        form = ttk.Frame(self.root, style="Moon.TFrame")
        form.pack(fill="x", padx=24, pady=8)
        form.columnconfigure(1, weight=1)

        self._folder_row(form, 0, "Baseline folder", self.baseline_var, self._pick_baseline)
        self._folder_row(form, 1, "Makeup folder", self.makeup_var, self._pick_makeup)
        self._folder_row(form, 2, "Output folder", self.output_var, self._pick_output)

        options = ttk.Frame(self.root, style="Moon.TFrame")
        options.pack(fill="x", padx=24, pady=(4, 8))
        ttk.Checkbutton(
            options, text="Search subfolders recursively",
            variable=self.recursive_var, style="Moon.TCheckbutton",
        ).pack(side="left", padx=(0, 20))
        ttk.Checkbutton(
            options, text="Save change-map overlay images",
            variable=self.save_overlays_var, style="Moon.TCheckbutton",
        ).pack(side="left")

        actions = ttk.Frame(self.root, style="Moon.TFrame")
        actions.pack(fill="x", padx=24, pady=(4, 4))
        self.run_button = ttk.Button(
            actions, text="\u2728 Run comparison", style="Moon.TButton",
            command=self._on_run,
        )
        self.run_button.pack(side="left")
        self.open_button = ttk.Button(
            actions, text="Open report", style="Moon.TButton",
            command=self._open_report, state="disabled",
        )
        self.open_button.pack(side="left", padx=(10, 0))

        self.progress = ttk.Progressbar(
            self.root, mode="indeterminate", style="Horizon.Horizontal.TProgressbar",
        )
        self.progress.pack(fill="x", padx=24, pady=(10, 4))

        ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel").pack(
            anchor="w", padx=26,
        )

        log_frame = ttk.Frame(self.root, style="Panel.TFrame")
        log_frame.pack(fill="both", expand=True, padx=24, pady=(8, 20))

        self.log_text = Text(
            log_frame, bg=NIGHT_SKY_ALT, fg=MOONLIGHT, insertbackground=MOONGLOW,
            borderwidth=0, highlightthickness=0, font=("Consolas", 9), wrap="word",
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_text.configure(state="disabled")

    def _folder_row(self, parent, row: int, label: str, var: StringVar, command) -> None:
        ttk.Label(parent, text=label, style="Moon.TLabel").grid(
            row=row, column=0, sticky="w", pady=6,
        )
        entry = ttk.Entry(parent, textvariable=var, style="Moon.TEntry")
        entry.grid(row=row, column=1, sticky="ew", padx=10)
        ttk.Button(parent, text="Browse\u2026", style="Moon.TButton", command=command).grid(
            row=row, column=2,
        )

    # -- folder pickers -------------------------------------------------------
    def _pick_baseline(self) -> None:
        path = filedialog.askdirectory(title="Select baseline (before) folder")
        if path:
            self.baseline_var.set(path)

    def _pick_makeup(self) -> None:
        path = filedialog.askdirectory(title="Select makeup (after) folder")
        if path:
            self.makeup_var.set(path)

    def _pick_output(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_var.set(path)

    # -- run pipeline ---------------------------------------------------------
    def _on_run(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        if not self.baseline_var.get() or not self.makeup_var.get() or not self.output_var.get():
            self.status_var.set("Please choose baseline, makeup and output folders.")
            return

        self._append_log("Starting comparison\u2026")
        self.status_var.set("Running\u2026")
        self.run_button.configure(state="disabled")
        self.open_button.configure(state="disabled")
        self.progress.start(12)

        args = argparse.Namespace(
            baseline=Path(self.baseline_var.get()),
            makeup=Path(self.makeup_var.get()),
            output=Path(self.output_var.get()),
            recursive=self.recursive_var.get(),
            min_detection_confidence=_cli_default("min_detection_confidence"),
            max_yaw=DEFAULT_MAX_YAW,
            max_pitch=DEFAULT_MAX_PITCH,
            save_overlays=self.save_overlays_var.get(),
            show_landmark_ids=False,
            workers=_cli_default("workers"),
            model_path=None,
            warn_threshold=_cli_default("warn_threshold"),
            alert_threshold=_cli_default("alert_threshold"),
            verbose=False,
        )

        self._worker = threading.Thread(target=self._run_pipeline, args=(args,), daemon=True)
        self._worker.start()

    def _run_pipeline(self, args: argparse.Namespace) -> None:
        # Local import to avoid pulling heavy dependencies (OpenCV/MediaPipe)
        # until the GUI actually needs to run a comparison.
        from .cli import run

        handler = _LogHandler(self._log_queue)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger = logging.getLogger("face_geometry")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            summary = run(args)
        except Exception as exc:  # noqa: BLE001 - surfaced to the log pane
            self._log_queue.put(f"ERROR: {exc}")
            self.root.after(0, self._on_finished, None, str(exc))
        else:
            self.root.after(0, self._on_finished, args.output, None)
            score = summary.get("geometry_change_score")
            if score is not None:
                self._log_queue.put(f"Geometry change score: {score:.1f} / 100")
        finally:
            logger.removeHandler(handler)

    def _on_finished(self, output: Path | None, error: str | None) -> None:
        self.progress.stop()
        self.run_button.configure(state="normal")
        if error:
            self.status_var.set(f"Failed: {error}")
        else:
            self._result_output = output
            self.open_button.configure(state="normal")
            self.status_var.set(f"Done. Report written to {output}")
            self._append_log("Comparison complete.")

    def _open_report(self) -> None:
        if self._result_output is None:
            return
        report = self._result_output / "report.html"
        if report.exists():
            webbrowser.open(report.resolve().as_uri())

    # -- logging --------------------------------------------------------------
    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _drain_log_queue(self) -> None:
        try:
            while True:
                message = self._log_queue.get_nowait()
                self._append_log(message)
        except queue.Empty:
            pass
        self.root.after(150, self._drain_log_queue)


def main() -> int:
    """GUI entry point (``face-geometry-gui``)."""

    root = Tk()
    FaceGeometryApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
