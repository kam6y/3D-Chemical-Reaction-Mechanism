"""CLI argument parsing and Phase 11 meta.json tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import build_parser, main

_FAST_TOML = """\
description = "sn2 quick"

[endpoint_relax]
fmax = 100.0
max_steps = 1

[neb]
n_images = 3
fmax = 100.0
max_steps = 1
"""


def test_default_flags_parse():
    p = build_parser()
    a = p.parse_args(["run", "examples/sn2.rxn", "-o", "out/"])
    assert a.cmd == "run"
    assert a.backend == "uma"
    assert a.model == "uma-m-1p1"
    assert a.render is False
    assert a.blender_exe == "blender"
    for removed in ("seed", "relax_fmax", "traj_stride", "neb_refine", "neb_images"):
        assert not hasattr(a, removed)


def test_removed_flags_rejected():
    p = build_parser()
    for flag in [
        "--neb-refine",
        "--neb-images",
        "--seed",
        "--traj-stride",
        "--relax-fmax",
        "--reaction-type",
        "--k-form",
        "--k-broken",
        "--r-form",
        "--r-broken",
        "--max-relax-steps",
        "--n-angles",
        "--cone-half-deg",
        "--no-mmff-prescreen",
        "--prescreen-keep",
        "--prescreen-steps",
    ]:
        with pytest.raises(SystemExit):
            args = ["run", "examples/sn2.rxn", "-o", "out/", flag]
            if flag not in ("--neb-refine", "--no-mmff-prescreen"):
                args.append("0")
            p.parse_args(args)


def test_cli_lj_writes_phase11_outputs(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn2", toml_body=_FAST_TOML)
    out = tmp_path / "out"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "lj"])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["description"] == "sn2 quick"
    assert "trials" not in meta
    assert "selected_trial" not in meta
    assert "placement" not in meta
    assert meta["endpoint_relax_r"]["converged"] in (True, False)
    assert meta["endpoint_relax_p"]["converged"] in (True, False)
    assert meta["neb"]["n_images"] == 3
    assert len(meta["neb"]["image_energies"]) == 3
    assert meta["effective_params"]["endpoint_relax"]["max_steps"] == 1
    assert meta["effective_params"]["neb"]["max_steps"] == 1

    energies = json.loads((out / "energies.json").read_text())
    assert len(energies) == 3
    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) == 3


def test_missing_sidecar_returns_config_error(tmp_path: Path, sn2_rxn_path: Path):
    rxn = tmp_path / "sn2.rxn"
    rxn.write_bytes(sn2_rxn_path.read_bytes())
    out = tmp_path / "out"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "lj"])
    assert rc == 2
