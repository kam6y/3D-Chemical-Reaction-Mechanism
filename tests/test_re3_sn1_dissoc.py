"""End-to-end SN1 step 1 (heterolytic dissociation) test using UMA. Marked slow."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main


@pytest.mark.slow
def test_re3_sn1_dissoc_end_to_end(tmp_path: Path, sn1_dissoc_rxn_path: Path):
    out = tmp_path / "sn1d"
    rc = main([
        "run", str(sn1_dissoc_rxn_path), "-o", str(out),
        "--backend", "uma",
        "--reaction-type", "sn1_dissoc",
        "--n-angles", "8",   # will be auto-clamped to 1
    ])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert len(meta["trials"]) == 1, (
        f"unimolecular auto-clamp expected 1 trial; got {len(meta['trials'])}"
    )
    assert meta["selected_trial"] == 0
    assert meta["effective_params"]["r_form_targets"] == []
    assert meta["reaction_type"] == "sn1_dissoc"

    pre = meta["prescreen"]
    assert pre["enabled"] is False or pre["kept"] is None

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
