"""SN2 end-to-end with CI-NEB. Slow, requires UMA."""
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
k_form = 1.0
k_broken = 3.0
r_broken = 4.0
max_relax_steps = 100
[sampling]
n_candidates = 16
[neb]
top_k = 2
n_images = 5
max_steps = 30
[parallel]
screening_workers = 2
neb_workers = 2
"""


@pytest.mark.slow
def test_re1_sn2_end_to_end_with_cineb(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn2", toml_body=_SN2_FAST)
    out = tmp_path / "sn2"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["description"] == "SN2 fast"
    assert meta["selected_trial"] >= 0
    pl = meta["placement"]
    assert pl["n_candidates"] >= 1
    assert pl["n_valid"] >= 1
    assert pl["n_blocked"] == pl["n_candidates"] - pl["n_valid"]
    assert len(meta["screening_trials"]) == pl["n_valid"]
    assert any(t["reached_product"] for t in meta["screening_trials"])
    assert len(meta["top_k_indices"]) >= 1
    assert len(meta["neb_results"]) >= 1
    assert all("peak_energy" in r for r in meta["neb_results"])
    assert meta["selected_trial"] in [r["trial_idx"] for r in meta["neb_results"]]
    assert "wall_clock_breakdown" in meta
    assert meta["effective_params"]["screening_model"]
    assert meta["effective_params"]["neb_model"]

    frames = read(str(out / "trajectory.xyz"), index=":")
    # Phase 8 trajectory.xyz is the NEB images of the selected trial:
    # length == cfg.neb.n_images + 2 * cfg.neb.pad_frames (here 5 + 0).
    assert len(frames) == 5

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
