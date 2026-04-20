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
