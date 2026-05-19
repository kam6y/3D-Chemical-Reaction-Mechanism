"""SN2 integration test (Phase 11 Pure CI-NEB)."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from reactx.cli import main

_SN2_TOML = """\
description = "SN2 anion test"
reactant_structure = "sn2.reactant.xyz"
product_structure = "sn2.product.xyz"
"""


@pytest.mark.slow
def test_sn2_neb_trajectory_and_walden(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn2", toml_body=_SN2_TOML)
    out = tmp_path / "sn2"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text())
    assert meta["neb"]["n_images"] == 11
    assert meta["endpoint_relax_r"]["converged"] is True
    assert meta["endpoint_relax_p"]["converged"] is True
    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) == 11

    r_frame = frames[0]
    c, cl, o = r_frame.positions[0], r_frame.positions[1], r_frame.positions[2]
    v_ccl = cl - c
    v_co = o - c
    cos_theta = (v_ccl @ v_co) / (np.linalg.norm(v_ccl) * np.linalg.norm(v_co))
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))
    assert angle_deg >= 150.0, f"Walden angle {angle_deg:.1f} < 150 threshold"

    p_frame = frames[-1]
    c_p, cl_p, o_p = p_frame.positions[0], p_frame.positions[1], p_frame.positions[2]
    assert float(np.linalg.norm(o_p - c_p)) < 1.7
    assert float(np.linalg.norm(cl_p - c_p)) > 3.0
