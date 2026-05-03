"""End-to-end Menshutkin test (NH3 + CH3Cl -> CH3NH3+ + Cl-).

Slow: requires UMA + GPU (~3 min). Validates the menshutkin preset:
- reached_product=True for at least one trial
- C-N forms (final ≤ 1.7 Å)
- C-Cl breaks (final ≥ 3.5 Å; full r_broken=5.0 may be unreachable due to ion-pair Coulomb attraction)
"""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main


@pytest.mark.slow
def test_re1_menshutkin_end_to_end(tmp_path: Path, menshutkin_rxn_path: Path):
    out = tmp_path / "menshutkin"
    rc = main([
        "run", str(menshutkin_rxn_path), "-o", str(out),
        "--backend", "uma",
        "--reaction-type", "menshutkin",
        "--n-angles", "4",  # smaller for test speed
    ])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["reaction_type"] == "menshutkin"
    assert meta["effective_params"]["k_form"] == 2.0
    assert meta["effective_params"]["k_broken"] == 2.0
    assert meta["effective_params"]["r_broken"] == 5.0
    assert meta["effective_params"]["max_relax_steps"] == 200
    assert meta["selected_trial"] >= 0
    # Menshutkin (neutral -> ion pair) sometimes outruns MMFF94 parameters; both
    # outcomes (mmff_failed=True with all 4 trials kept, or success with K=3)
    # should still produce at least one product-reaching trial via UMA.
    pre = meta["prescreen"]
    assert pre["enabled"] is True
    if pre["mmff_failed"]:
        assert len(meta["trials"]) == 4
    else:
        assert len(pre["kept"]) == 3
        assert len(meta["trials"]) == 3
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
    assert d_ccl_last >= 3.5, (
        f"C-Cl should break: {d_ccl_first:.2f} -> {d_ccl_last:.2f} (target ≥ 3.5)"
    )
