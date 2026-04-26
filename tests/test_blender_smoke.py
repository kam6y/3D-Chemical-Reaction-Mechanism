import os
import re
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
        "2\nFrame 0\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n2\nFrame 1\nH 0.0 0.0 0.0\nH 0.0 0.0 1.10\n"
    )
    out_blend = tmp_path / "scene.blend"
    script = Path(__file__).resolve().parent.parent / "blender" / "render.py"
    result = subprocess.run(
        [BLENDER, "--background", "--python", str(script), "--", str(xyz), str(out_blend)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert out_blend.exists()
    assert out_blend.stat().st_size > 0


@pytest.mark.blender
def test_blender_vdw_rescale(tmp_path: Path):
    if shutil.which(BLENDER) is None:
        pytest.skip(f"Blender executable not found: {BLENDER}")

    xyz = tmp_path / "traj.xyz"
    # 1-frame minimal CH (H + C) — exercises both Hydrogen_ball and Carbon_ball
    xyz.write_text("2\nFrame 0\nH 0.0 0.0 0.0\nC 0.0 0.0 1.10\n")
    out_blend = tmp_path / "scene.blend"
    script = Path(__file__).resolve().parent.parent / "blender" / "render.py"
    result = subprocess.run(
        [BLENDER, "--background", "--python", str(script), "--", str(xyz), str(out_blend)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr

    pattern = re.compile(r"\[reactx\] vdw-rescale: (\S+) scale=(\S+)")
    scales: dict[str, float] = {}
    for line in result.stdout.splitlines():
        m = pattern.search(line)
        if m:
            scales[m.group(1)] = float(m.group(2))

    assert "Hydrogen" in scales, f"no Hydrogen rescale log; stdout was:\n{result.stdout}"
    assert "Carbon" in scales, f"no Carbon rescale log; stdout was:\n{result.stdout}"
    # Alvarez 2013 vdW: H=1.20, C=1.77; default scale=0.25.
    assert abs(scales["Hydrogen"] - 1.20 * 0.25) < 1e-3
    assert abs(scales["Carbon"] - 1.77 * 0.25) < 1e-3


@pytest.mark.blender
def test_blender_bonds_detected(tmp_path: Path):
    if shutil.which(BLENDER) is None:
        pytest.skip(f"Blender executable not found: {BLENDER}")

    xyz = tmp_path / "traj.xyz"
    # Frame 0: C-H at 1.05 A (bonded; threshold = (0.76+0.31)*1.1 = 1.177 A).
    # Frame 1: C-H at 5.00 A (well past any reasonable threshold; bond breaks).
    # Union still reports 1 bond, but per-frame visibility hides it at frame 1.
    xyz.write_text(
        "2\nFrame 0\nH 0.0 0.0 0.0\nC 0.0 0.0 1.05\n2\nFrame 1\nH 0.0 0.0 0.0\nC 0.0 0.0 5.00\n"
    )
    out_blend = tmp_path / "scene.blend"
    script = Path(__file__).resolve().parent.parent / "blender" / "render.py"
    result = subprocess.run(
        [BLENDER, "--background", "--python", str(script), "--", str(xyz), str(out_blend)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert re.search(r"bonds:\s*1\s+bond", result.stdout), (
        f"expected bond log line; stdout was:\n{result.stdout}"
    )


@pytest.mark.blender
@pytest.mark.parametrize(
    "rxn_name",
    [
        "sn2",
        "sn1_step1",
        "e2",
        "e1_step2",
        "proton_transfer",
    ],
)
def test_blender_renders_all_phase1_reactions(
    tmp_path: Path,
    examples_dir: Path,
    rxn_name: str,
):
    """End-to-end: each Phase 1 reaction generates trajectory.xyz + scene.blend.

    Uses LJ backend so the test is fast (no UMA). Only checks that Blender
    successfully produces a non-empty .blend file. Chemistry accuracy is
    verified by the slow UMA tests (test_neb_*).
    """
    if shutil.which(BLENDER) is None:
        pytest.skip(f"Blender executable not found: {BLENDER}")

    from reactx import cli

    out = tmp_path / rxn_name
    rc = cli.main(
        [
            "run",
            str(examples_dir / f"{rxn_name}.rxn"),
            "-o",
            str(out),
            "--backend",
            "lj",
            "--images",
            "5",
            "--fmax",
            "0.5",
            "--max-steps",
            "10",
        ]
    )
    assert rc == 0, f"reactx CLI failed for {rxn_name}"
    xyz = out / "trajectory.xyz"
    assert xyz.exists()

    blend = out / "scene.blend"
    script = Path(__file__).resolve().parent.parent / "blender" / "render.py"
    result = subprocess.run(
        [BLENDER, "--background", "--python", str(script), "--", str(xyz), str(blend)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"blender failed for {rxn_name}: {result.stderr}"
    assert blend.exists() and blend.stat().st_size > 0
