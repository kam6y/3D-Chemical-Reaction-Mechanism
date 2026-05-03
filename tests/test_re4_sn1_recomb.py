"""End-to-end SN1 step 2 (cation + nucleophile recombination) test using UMA. Marked slow."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main


@pytest.mark.slow
def test_re4_sn1_recomb_end_to_end(tmp_path: Path, sn1_recomb_rxn_path: Path):
    out = tmp_path / "sn1r"
    rc = main([
        "run", str(sn1_recomb_rxn_path), "-o", str(out),
        "--backend", "uma",
        "--reaction-type", "sn1_recomb",
        "--n-angles", "8",
    ])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())

    # bimolecular なので unimolecular auto-clamp は発動せず prescreen が走る。
    # n_angles=8 のうち top-3 (--prescreen-keep default) が UMA に進み
    # meta.trials には UMA 実行分のみ並ぶ (prescreen-rejected は記録されない)。
    pre = meta["prescreen"]
    assert pre["enabled"] is True
    if not pre["mmff_failed"]:
        assert len(pre["kept"]) == 3
        assert len(meta["trials"]) == 3
    assert meta["selected_trial"] >= 0
    assert meta["reaction_type"] == "sn1_recomb"

    # formed=1 で r_form_targets が 1 要素 list (Cordero C-Cl ≈ 1.78 Å)
    rfts = meta["effective_params"]["r_form_targets"]
    assert len(rfts) == 1
    assert 1.7 <= rfts[0] <= 1.85, f"r_form_targets[0] should be ≈1.78, got {rfts[0]}"

    # ≥1 trial が reached_product
    reached = [t for t in meta["trials"] if t["reached_product"]]
    assert len(reached) >= 1, "expected ≥1 reached_product trial, got 0"

    # 最終フレームで C-Cl 距離 ≤ 1.95 Å (Cordero × 1.1)
    frames = read(str(out / "trajectory.xyz"), index=":")
    syms = frames[-1].get_chemical_symbols()
    cl_idx = syms.index("Cl")
    c_atoms = [i for i, s in enumerate(syms) if s == "C"]
    pos_last = frames[-1].positions
    d_to_cl = [(i, float(((pos_last[i] - pos_last[cl_idx]) ** 2).sum() ** 0.5)) for i in c_atoms]
    central, d_last = min(d_to_cl, key=lambda kv: kv[1])
    assert d_last <= 1.95, f"final C-Cl should be ≤1.95 A (Cordero × 1.1), got {d_last:.2f}"
