"""End-to-end SN1 step 1 (heterolytic dissociation) test using UMA. Marked slow."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main

_SN1D_FAST = """\
description = "SN1 dissoc fast"
formed = []
broken = [[1, 5]]

[afir]
alpha_broken = 2.5
max_relax_steps = 200

[scoring]
r_broken_threshold = 6.0

[sampling]
n_candidates = 1
"""


@pytest.mark.slow
def test_re3_sn1_dissoc_end_to_end(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn1_dissoc", toml_body=_SN1D_FAST)
    out = tmp_path / "sn1d"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert len(meta["trials"]) == 1, (
        f"unimolecular auto-clamp expected 1 trial; got {len(meta['trials'])}"
    )
    assert meta["selected_trial"] == 0
    # Phase 9: per-trial formed_thresholds (empty for SN1 dissoc) replaces r_form_targets
    assert meta["trials"][0]["formed_thresholds"] == []
    assert meta["description"] == "SN1 dissoc fast"

    pl = meta["placement"]
    assert pl["n_candidates"] == 1
    assert pl["n_valid"] == 1
    assert pl["n_blocked"] == 0

    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) >= 3

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


@pytest.mark.slow
def test_re3_sn1_dissoc_min_nonbonded_distance(
    tmp_path: Path, tmp_rxn_with_toml, assert_min_nonbonded_distance_ok,
):
    """Spec §6.4 safety: AFIR-driven trajectory must not produce non-bonded
    atom pairs closer than 0.5 Å (UMA off-manifold detection).

    Uses the same TOML as the end-to-end test. Atom-map 1->idx 0 (C central),
    atom-map 5->idx 4 (Br) per examples/sn1_dissoc.rxn, so broken=[(0, 4)].
    """
    rxn = tmp_rxn_with_toml("sn1_dissoc", toml_body=_SN1D_FAST)
    out = tmp_path / "sn1d_safety"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    frames = read(str(out / "trajectory.xyz"), index=":")
    # SN1 dissoc: formed=[], broken=[[1, 5]] in atom-map -> (0, 4) in 0-based.
    assert_min_nonbonded_distance_ok(
        frames, formed=[], broken=[(0, 4)],
    )
