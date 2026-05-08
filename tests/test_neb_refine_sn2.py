"""--neb-refine flag exercises the optional NEB refinement on the best trial."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main

_SN2_NEB = """\
description = "SN2 neb-refine fast"
formed = [[1, 3]]
broken = [[1, 2]]

[afir]
alpha_formed = 0.5
alpha_broken = 1.0
max_relax_steps = 30

[scoring]
r_broken_threshold = 4.0

[sampling]
n_candidates = 4
"""


@pytest.mark.slow
def test_neb_refine_writes_refined_trajectory(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn2", toml_body=_SN2_NEB)
    out = tmp_path / "sn2_neb"
    rc = main([
        "run", str(rxn), "-o", str(out),
        "--backend", "uma",
        "--neb-refine",
        "--neb-images", "5",
    ])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["neb_refined"] is True

    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) == 5
