import os
import shutil
import subprocess
from pathlib import Path

import pytest

BLENDER = os.environ.get("BLENDER_EXE", "blender")


@pytest.mark.blender
def test_blender_smoke_produces_blend(tmp_path: Path):
    if shutil.which(BLENDER) is None:
        pytest.skip(f"Blender executable not found: {BLENDER}")

    xyz = tmp_path / "traj.xyz"
    # Minimal 2-frame XYZ (H2 stretch) — bypass full NEB for smoke test
    xyz.write_text(
        "2\nFrame 0\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n"
        "2\nFrame 1\nH 0.0 0.0 0.0\nH 0.0 0.0 1.10\n"
    )
    out_blend = tmp_path / "scene.blend"
    script = Path(__file__).resolve().parent.parent / "blender" / "render.py"
    result = subprocess.run(
        [BLENDER, "--background", "--python", str(script),
         "--", str(xyz), str(out_blend)],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert out_blend.exists()
    assert out_blend.stat().st_size > 0
