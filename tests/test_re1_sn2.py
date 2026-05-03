"""End-to-end SN2 test using UMA. Marked slow, requires HF auth + GPU."""
import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from reactx.cli import main

_SN2_FAST = """\
description = "SN2 fast"
formed = [[1, 3]]
broken = [[1, 2]]
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 50
[sampling]
n_angles = 4
"""


@pytest.mark.slow
def test_re1_sn2_end_to_end(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn2", toml_body=_SN2_FAST)
    out = tmp_path / "sn2"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["description"] == "SN2 fast"
    assert meta["selected_trial"] >= 0
    pre = meta["prescreen"]
    assert pre["enabled"] is True
    if not pre["mmff_failed"]:
        assert len(pre["kept"]) == 3
        assert len(meta["trials"]) == 3
    assert any(t["reached_product"] for t in meta["trials"])

    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) >= 3

    syms = frames[0].get_chemical_symbols()
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    o_idx = syms.index("O")

    angles = []
    for f in frames:
        v_co = f.positions[o_idx] - f.positions[c_idx]
        v_ccl = f.positions[cl_idx] - f.positions[c_idx]
        cos_t = float(np.dot(v_co, v_ccl) / (
            np.linalg.norm(v_co) * np.linalg.norm(v_ccl)
        ))
        angles.append(float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))))
    assert max(angles) >= 120.0

    d_co_first = frames[0].get_distance(c_idx, o_idx)
    d_co_last = frames[-1].get_distance(c_idx, o_idx)
    d_ccl_first = frames[0].get_distance(c_idx, cl_idx)
    d_ccl_last = frames[-1].get_distance(c_idx, cl_idx)
    assert d_co_last < d_co_first - 0.5
    assert d_ccl_last > d_ccl_first + 0.5
