"""Sponsor server package for autonomous sponsor simulation.

Provides FastAPI-based endpoints and agent logic for the 24-hour
autonomous sponsor simulation in physical-AI oncology trials.
Works with or without FastAPI/Pydantic installed (falls back to
dataclasses and plain dicts).
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
