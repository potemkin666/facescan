"""Entry point so the tool runs as ``python -m face_geometry``."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
