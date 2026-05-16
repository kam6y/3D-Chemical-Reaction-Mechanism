"""SN1 dissociation integration test (Phase 11)."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main

_SN1_DISSOC_TOML = """\
description = "SN1 dissociation test"

[placement]
initial_separation = 6.0
"""


@pytest.mark.slow
def test_sn1_dissoc_neb(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn1_dissoc", toml_body=_SN1_DISSOC_TOML)
    out = tmp_path / "sn1_dissoc"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text())
    assert meta["endpoint_relax_r"]["converged"] is True
    assert meta["endpoint_relax_p"]["converged"] is True
    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) == 11
