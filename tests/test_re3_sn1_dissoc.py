"""End-to-end SN1 step 1 (heterolytic dissociation) test using UMA with CI-NEB. Marked slow."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main


_SN1D_FAST = """\
description = "SN1 dissoc fast"
formed = []
broken = [[1, 5]]
[restraints]
k_form = 0.0
k_broken = 2.0
r_broken = 6.0
max_relax_steps = 100
[sampling]
n_candidates = 1
[neb]
top_k = 1
n_images = 5
max_steps = 30
[parallel]
screening_workers = 1
neb_workers = 1
"""


@pytest.mark.slow
def test_re3_sn1_dissoc_end_to_end(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn1_dissoc", toml_body=_SN1D_FAST)
    out = tmp_path / "sn1d"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert len(meta["screening_trials"]) == 1, (
        f"unimolecular auto-clamp expected 1 trial; got {len(meta['screening_trials'])}"
    )
    assert meta["selected_trial"] == 0
    assert meta["effective_params"]["r_form_targets"] == []
    assert meta["description"] == "SN1 dissoc fast"

    pl = meta["placement"]
    assert pl["n_candidates"] == 1
    assert pl["n_valid"] == 1
    assert pl["n_blocked"] == 0

    assert meta["top_k_indices"] == [0]
    assert len(meta["neb_results"]) == 1
    assert meta["neb_results"][0]["trial_idx"] == 0

    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) == 5

    syms = frames[0].get_chemical_symbols()
    br_idx = syms.index("Br")
    c_atoms = [i for i, s in enumerate(syms) if s == "C"]
    pos0 = frames[0].positions
    d_to_br = [(i, float(((pos0[i] - pos0[br_idx]) ** 2).sum() ** 0.5)) for i in c_atoms]
    c_central = min(d_to_br, key=lambda kv: kv[1])[0]

    d_first = frames[0].get_distance(c_central, br_idx)
    d_last = frames[-1].get_distance(c_central, br_idx)
    assert d_last > d_first + 1.5, (
        f"C-Br should grow appreciably: {d_first:.2f} -> {d_last:.2f}"
    )
    assert d_last >= 4.5, f"final C-Br should be at least 4.5 A, got {d_last:.2f}"
