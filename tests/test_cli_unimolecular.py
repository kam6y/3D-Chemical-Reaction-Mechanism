"""CLI sanity checks for unimolecular endpoints in Phase 11."""
import json
from pathlib import Path

from ase.io import read

from reactx.cli import main

_SN1_DISSOC_FAST = """\
description = "sn1_dissoc fast"

[endpoint_relax]
fmax = 100.0
max_steps = 1

[neb]
n_images = 3
fmax = 100.0
max_steps = 1
"""


def test_unimolecular_cli_lj_runs_without_trial_metadata(
    tmp_path: Path,
    tmp_rxn_with_toml,
):
    rxn = tmp_rxn_with_toml("sn1_dissoc", toml_body=_SN1_DISSOC_FAST)
    out = tmp_path / "out"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "lj"])
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text())
    assert "trials" not in meta
    assert "placement" not in meta
    assert meta["neb"]["n_images"] == 3
    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) == 3
