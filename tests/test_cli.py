import json
from pathlib import Path

import pytest

from reactx import cli


def test_cli_run_end_to_end_with_lj_backend(tmp_path: Path, sn2_rxn_path: Path):
    out = tmp_path / "out"
    rc = cli.main([
        "run", str(sn2_rxn_path), "-o", str(out),
        "--images", "5", "--fmax", "0.5", "--max-steps", "20",
        "--backend", "lj",
    ])
    assert rc == 0
    assert (out / "trajectory.xyz").exists()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["n_images"] == 5
    energies = json.loads((out / "energies.json").read_text())
    assert len(energies) == 5


def test_cli_missing_rxn_returns_nonzero(tmp_path: Path):
    rc = cli.main([
        "run", str(tmp_path / "nope.rxn"), "-o", str(tmp_path / "out"),
        "--backend", "lj",
    ])
    assert rc != 0


def test_cli_render_flag_invokes_blender_when_successful(
    tmp_path: Path, sn2_rxn_path: Path, monkeypatch
):
    """--render path success: mock subprocess.run to return rc=0."""
    from reactx import cli as cli_mod

    calls = []

    class FakeResult:
        returncode = 0

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return FakeResult()

    monkeypatch.setattr("subprocess.run", fake_run)

    out = tmp_path / "out"
    rc = cli_mod.main([
        "run", str(sn2_rxn_path), "-o", str(out),
        "--images", "5", "--fmax", "0.5", "--max-steps", "10",
        "--backend", "lj", "--render", "--blender-exe", "mock-blender",
    ])
    assert rc == 0
    # Blender was invoked with the right positional args after "--"
    assert any("mock-blender" in c[0] for c in calls), calls
    assert any("--background" in c for c in calls)


def test_cli_render_propagates_blender_failure(
    tmp_path: Path, sn2_rxn_path: Path, monkeypatch
):
    """--render path failure: subprocess returns rc=1, CLI returns nonzero."""
    class FakeResult:
        returncode = 1

    monkeypatch.setattr("subprocess.run", lambda *a, **k: FakeResult())

    out = tmp_path / "out"
    rc = cli.main([
        "run", str(sn2_rxn_path), "-o", str(out),
        "--images", "5", "--fmax", "0.5", "--max-steps", "10",
        "--backend", "lj", "--render", "--blender-exe", "mock-blender",
    ])
    assert rc != 0
