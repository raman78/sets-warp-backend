"""Shared pytest configuration.

Lives at the repo root rather than in `tests/` on purpose: the backend is a
flat set of modules (`main.py`, `admin_*.py`), not an installed package, so
the tests import `main` directly and the root has to be on `sys.path`. Pytest
only adds the directory holding the test file, which is why a run from
anywhere other than the repo root used to fail collection with
`ModuleNotFoundError: No module named 'main'`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
