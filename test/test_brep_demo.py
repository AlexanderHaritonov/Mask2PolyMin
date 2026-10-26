"""Smoke test of the B-rep demo in examples/brep/: its scripts run without an error. draw_frames.py writes its frames to the ignored examples/brep/frames/."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).resolve().parent.parent / "examples" / "brep"


@pytest.mark.parametrize("script", ["stitch_faces.py", "draw_frames.py"])
def test_demo_runs(script):
    result = subprocess.run([sys.executable, str(DEMO_DIR / script)], env={**os.environ, "MPLBACKEND": "Agg"}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
