"""End-to-end E2 elimination test using UMA. Marked slow."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main

_E2_FAST = """\
description = "E2 fast"
formed = [[4, 5]]
broken = [[2, 5], [1, 3]]
[restraints]
k_form = 1.0
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
[sampling]
n_candidates = 8
"""


@pytest.mark.slow
def test_re3_e2_end_to_end(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("e2", toml_body=_E2_FAST)
    out = tmp_path / "e2"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["selected_trial"] >= 0, f"no trial selected: {meta}"
    assert any(t["reached_product"] for t in meta["trials"]), (
        f"no E2 trial reached product: {meta['trials']}"
    )
    assert isinstance(meta["effective_params"]["r_form_targets"], list)
    assert len(meta["effective_params"]["r_form_targets"]) == 1
    assert meta["description"] == "E2 fast"

    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) >= 3

    syms = frames[0].get_chemical_symbols()
    cl_idx = syms.index("Cl")
    o_idx = syms.index("O")
    c_atoms = [i for i, s in enumerate(syms) if s == "C"]
    pos0 = frames[0].positions
    d_to_cl = [(i, float(((pos0[i] - pos0[cl_idx]) ** 2).sum() ** 0.5)) for i in c_atoms]
    c_alpha = min(d_to_cl, key=lambda kv: kv[1])[0]
    c_beta = next(i for i in c_atoms if i != c_alpha)

    d_ccl_first = frames[0].get_distance(c_alpha, cl_idx)
    d_ccl_last = frames[-1].get_distance(c_alpha, cl_idx)
    assert d_ccl_last > d_ccl_first + 0.5, (
        f"C-Cl should grow: {d_ccl_first:.2f} -> {d_ccl_last:.2f}"
    )

    h_atoms = [i for i, s in enumerate(syms) if s == "H"]
    d_to_cbeta = [(i, frames[0].get_distance(c_beta, i)) for i in h_atoms]
    h_beta = min(d_to_cbeta, key=lambda kv: kv[1])[0]
    d_oh_first = frames[0].get_distance(o_idx, h_beta)
    d_oh_last = frames[-1].get_distance(o_idx, h_beta)
    assert d_oh_last < d_oh_first - 1.0, (
        f"O-H_β should shrink: {d_oh_first:.2f} -> {d_oh_last:.2f}"
    )
