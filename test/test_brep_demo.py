"""Smoke test of the B-rep demo in examples/brep/: it runs without an error."""
import os
import subprocess
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent / "examples" / "brep" / "stitch_faces.py"


def test_stitch_faces_runs():
    result = subprocess.run([sys.executable, str(DEMO)], env={**os.environ, "MPLBACKEND": "Agg"}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
