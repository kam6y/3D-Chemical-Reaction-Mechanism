"""End-to-end SN1 step 2 (cation + nucleophile recombination) test using UMA. Marked slow."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main

_SN1R_FAST = """\
description = "SN1 recomb fast"
formed = [[1, 5]]
broken = []

[afir]
alpha_formed = 1.5
max_relax_steps = 100

[sampling]
n_candidates = 16
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
    assert len(meta["trials"]) == pl["n_valid"]
    assert meta["selected_trial"] >= 0
    assert meta["description"] == "SN1 recomb fast"

    # Phase 9: per-trial formed_thresholds replaces effective_params.r_form_targets
    fts = meta["trials"][0]["formed_thresholds"]
    assert len(fts) == 1
    # 1.15 × (Cordero C + Cordero Cl) = 1.15 × (0.76 + 1.02) ≈ 2.047
    assert 1.95 <= fts[0] <= 2.10, f"formed_thresholds[0] should be ≈2.05, got {fts[0]}"

    reached = [t for t in meta["trials"] if t["reached_product"]]
    assert len(reached) >= 1, "expected ≥1 reached_product trial, got 0"

    frames = read(str(out / "trajectory.xyz"), index=":")
    syms = frames[-1].get_chemical_symbols()
    cl_idx = syms.index("Cl")
    c_atoms = [i for i, s in enumerate(syms) if s == "C"]
    pos_last = frames[-1].positions
    d_to_cl = [(i, float(((pos_last[i] - pos_last[cl_idx]) ** 2).sum() ** 0.5)) for i in c_atoms]
    central, d_last = min(d_to_cl, key=lambda kv: kv[1])
    assert d_last <= 1.95, f"final C-Cl should be ≤1.95 A (Cordero × 1.1), got {d_last:.2f}"


_SN1R_SAFETY = """\
description = "SN1 recomb safety check"
formed = [[1, 5]]
broken = []

[afir]
alpha_formed = 1.5
max_relax_steps = 200
"""


@pytest.mark.slow
def test_re4_sn1_recomb_min_nonbonded_distance(
    tmp_path: Path, tmp_rxn_with_toml, assert_min_nonbonded_distance_ok,
):
    """Spec §6.4: AFIR-driven trajectory must not produce non-bonded
    atom pairs closer than 0.5 Å (UMA off-manifold detection).

    sn1_recomb.rxn atom-map: 1->idx 0 (C+), 5->idx 4 (Cl-).
    formed=[[1,5]] -> (0,4); broken=[].
    """
    rxn = tmp_rxn_with_toml("sn1_recomb", toml_body=_SN1R_SAFETY)
    out = tmp_path / "sn1r_safety"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    frames = read(str(out / "trajectory.xyz"), index=":")
    assert_min_nonbonded_distance_ok(
        frames, formed=[(0, 4)], broken=[],
    )
