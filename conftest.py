"""Pytest configuration: make the repository root importable so tests can do
``from lunar_data_pipeline...`` regardless of where pytest is invoked."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
