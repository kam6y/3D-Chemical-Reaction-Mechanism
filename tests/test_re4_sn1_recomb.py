"""End-to-end SN1 step 2 (cation + nucleophile recombination) test using UMA with CI-NEB.

Marked slow.
"""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main


_SN1R_FAST = """\
description = "SN1 recomb fast"
formed = [[1, 5]]
broken = []
[restraints]
k_form = 1.0
k_broken = 0.0
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
def test_re4_sn1_recomb_end_to_end(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn1_recomb", toml_body=_SN1R_FAST)
    out = tmp_path / "sn1r"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())

    pl = meta["placement"]
    assert pl["n_candidates"] >= 1
    assert pl["n_valid"] >= 1
    assert pl["n_blocked"] == pl["n_candidates"] - pl["n_valid"]
    assert len(meta["screening_trials"]) == pl["n_valid"]
    assert meta["selected_trial"] >= 0
    assert meta["description"] == "SN1 recomb fast"

    rfts = meta["effective_params"]["r_form_targets"]
    assert len(rfts) == 1
    assert 1.7 <= rfts[0] <= 1.85, f"r_form_targets[0] should be ~1.78, got {rfts[0]}"

    reached = [t for t in meta["screening_trials"] if t["reached_product"]]
    assert len(reached) >= 1, "expected >=1 reached_product trial, got 0"

    assert len(meta["top_k_indices"]) >= 1
    assert len(meta["neb_results"]) >= 1
    assert meta["selected_trial"] in [r["trial_idx"] for r in meta["neb_results"]]

    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) == 5
    syms = frames[-1].get_chemical_symbols()
    cl_idx = syms.index("Cl")
    c_atoms = [i for i, s in enumerate(syms) if s == "C"]
    pos_last = frames[-1].positions
    d_to_cl = [(i, float(((pos_last[i] - pos_last[cl_idx]) ** 2).sum() ** 0.5)) for i in c_atoms]
    central, d_last = min(d_to_cl, key=lambda kv: kv[1])
    assert d_last <= 1.95, f"final C-Cl should be <=1.95 A (Cordero x 1.1), got {d_last:.2f}"
