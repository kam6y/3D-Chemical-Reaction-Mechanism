"""Diels-Alder simple integration test (Phase 11)."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main

_DA_SIMPLE_TOML = """\
description = "DA simple test"

[placement]
orientation = "default"
"""


@pytest.mark.slow
def test_diels_alder_simple_neb(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("diels_alder_simple", toml_body=_DA_SIMPLE_TOML)
    out = tmp_path / "da_simple"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text())
    assert meta["endpoint_relax_r"]["converged"] is True
    assert meta["endpoint_relax_p"]["converged"] is True
    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) == 11
