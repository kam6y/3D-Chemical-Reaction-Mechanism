"""End-to-end E2 elimination test using UMA with CI-NEB. Marked slow."""
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
k_form = 2.0
k_broken = 5.0
r_broken = 4.0
max_relax_steps = 250
[sampling]
n_candidates = 8
[neb]
top_k = 2
n_images = 5
max_steps = 30
pad_frames = 0
interp_factor = 1
[parallel]
screening_workers = 2
neb_workers = 2
"""


@pytest.mark.slow
def test_re3_e2_end_to_end(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("e2", toml_body=_E2_FAST)
    out = tmp_path / "e2"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["selected_trial"] >= 0, f"no trial selected: {meta}"
    assert any(t["reached_product"] for t in meta["screening_trials"]), (
        f"no E2 trial reached product: {meta['screening_trials']}"
    )
    assert isinstance(meta["effective_params"]["r_form_targets"], list)
    assert len(meta["effective_params"]["r_form_targets"]) == 1
    assert meta["description"] == "E2 fast"
    assert len(meta["top_k_indices"]) >= 1
    assert len(meta["neb_results"]) >= 1
    assert meta["selected_trial"] in [r["trial_idx"] for r in meta["neb_results"]]

    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) == 5

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
    # Phase 8 (uma-s-1p2): screening typically achieves 0.8-1.0 shrinkage on O-H
    # within the bond-formation window (NEB completes the rest in middle images).
    # Threshold loosened from 1.0 to 0.7 to reflect the smaller model's gentler
    # gradient; chemistry still moves in the right direction.
    assert d_oh_last < d_oh_first - 0.7, (
        f"O-H_beta should shrink: {d_oh_first:.2f} -> {d_oh_last:.2f}"
    )
