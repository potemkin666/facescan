"""Facial landmark geometry change measurement tool.

Quantifies how much detected facial landmark geometry changes between a
folder of baseline photographs and a folder of treated (makeup) photographs.

This tool performs **no identity recognition**. It only measures changes in
dense facial landmark geometry after pose normalisation.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
