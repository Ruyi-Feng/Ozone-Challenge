"""Shared pytest bootstrap.

Import order workaround (Windows): pyarrow and torch bundle conflicting
runtime DLLs in this environment; importing pyarrow AFTER torch's DLLs are
loaded intermittently dies with an access violation inside
``pyarrow.lib`` (observed with torch 2.3.1 + pyarrow 24 on win64).
Loading pyarrow first — before any test module pulls in torch — makes the
resolution order deterministic.  Keep this at the very top.
"""
try:  # pragma: no cover
    import pyarrow  # noqa: F401
except ImportError:
    pass

import sys
from pathlib import Path

# repo root importable for all tests (they also do this individually)
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
