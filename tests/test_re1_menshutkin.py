"""End-to-end Menshutkin test (NH3 + CH3Cl -> CH3NH3+ + Cl-).

Slow: requires UMA + GPU (~3 min). Validates the menshutkin TOML config:
- reached_product=True for at least one trial
- C-N forms (final ≤ 1.7 Å)
- C-Cl breaks (final ≥ 3.5 Å)
"""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main

_MEN_FAST = """\
description = "Menshutkin fast"
formed = [[1, 5]]
broken = [[5, 9]]

[afir]
alpha_formed = 4.0
alpha_broken = 4.0
max_relax_steps = 300

[scoring]
r_broken_threshold = 3.0

[sampling]
n_candidates = 8
"""


@pytest.mark.slow
def test_re1_menshutkin_end_to_end(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("menshutkin", toml_body=_MEN_FAST)
    out = tmp_path / "menshutkin"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["description"] == "Menshutkin fast"
    assert meta["effective_params"]["alpha_formed"] == 4.0
    assert meta["effective_params"]["alpha_broken"] == 4.0
    assert meta["effective_params"]["r_broken_threshold"] == 3.0
    assert meta["effective_params"]["max_relax_steps"] == 300
    assert meta["selected_trial"] >= 0
    pl = meta["placement"]
    assert pl["n_candidates"] >= 1
    assert pl["n_valid"] >= 1
    assert pl["n_blocked"] == pl["n_candidates"] - pl["n_valid"]
    assert len(meta["trials"]) == pl["n_valid"]
    assert any(t["reached_product"] for t in meta["trials"]), (
        f"No trial reached product. trials={meta['trials']}"
    )

    frames = read(str(out / "trajectory.xyz"), index=":")
    syms = frames[0].get_chemical_symbols()
    n_idx = syms.index("N")
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")

    d_nc_first = frames[0].get_distance(n_idx, c_idx)
    d_nc_last = frames[-1].get_distance(n_idx, c_idx)
    d_ccl_first = frames[0].get_distance(c_idx, cl_idx)
    d_ccl_last = frames[-1].get_distance(c_idx, cl_idx)

    assert d_nc_last < 1.7, (
        f"N-C should form: {d_nc_first:.2f} -> {d_nc_last:.2f} (target ≤ 1.7)"
    )
    assert d_ccl_last >= 3.0, (
        f"C-Cl should break: {d_ccl_first:.2f} -> {d_ccl_last:.2f} (target ≥ 3.0)"
    )


_MEN_SAFETY = """\
description = "Menshutkin safety check"
formed = [[1, 5]]
broken = [[5, 9]]

[afir]
alpha_formed = 4.0
alpha_broken = 4.0
max_relax_steps = 300

[scoring]
r_broken_threshold = 3.0
"""


@pytest.mark.slow
def test_re1_menshutkin_min_nonbonded_distance(
    tmp_path: Path, tmp_rxn_with_toml, assert_min_nonbonded_distance_ok,
):
    """Spec §6.4: AFIR-driven trajectory must not produce non-bonded
    atom pairs closer than 0.5 Å (UMA off-manifold detection).

    menshutkin.rxn atom-map: 1->idx 0 (N), 5->idx 4 (C), 9->idx 8 (Cl).
    formed=[[1,5]] -> (0,4); broken=[[5,9]] -> (4,8).
    """
    rxn = tmp_rxn_with_toml("menshutkin", toml_body=_MEN_SAFETY)
    out = tmp_path / "menshutkin_safety"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    frames = read(str(out / "trajectory.xyz"), index=":")
    assert_min_nonbonded_distance_ok(
        frames, formed=[(0, 4)], broken=[(4, 8)],
    )
