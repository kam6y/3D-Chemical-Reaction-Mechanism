"""End-to-end 4-center metathesis test using UMA. Marked slow."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main


@pytest.mark.slow
def test_re5_metathesis_end_to_end(tmp_path: Path, metathesis_rxn_path: Path):
    out = tmp_path / "m4c"
    rc = main([
        "run", str(metathesis_rxn_path), "-o", str(out),
        "--backend", "uma",
        "--reaction-type", "metathesis_4center",
        "--n-angles", "8",
    ])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())

    pre = meta["prescreen"]
    assert pre["enabled"] is True
    if not pre["mmff_failed"]:
        # MMFF parameterize 成功時は top-3 (--prescreen-keep default) のみ UMA に
        assert len(pre["kept"]) == 3
        assert len(meta["trials"]) == 3
    # MMFF 失敗時 (Li / Br を MMFF94 が扱えない可能性) は全 trial が UMA
    # n_angles=8 のうち prescreen-rejected を除いた数が meta.trials に並ぶ
    assert meta["selected_trial"] >= 0
    assert meta["reaction_type"] == "metathesis_4center"

    # formed=2 で r_form_targets が 2 要素 list (Cordero C-Br ≈ 1.94, Li-Cl ≈ 2.02)
    rfts = meta["effective_params"]["r_form_targets"]
    assert len(rfts) == 2
    # 順序は formed bonds の順序と一致する想定だが、symbol 集合で検証
    sorted_targets = sorted(rfts)
    # C-Br (1.94) と Li-Cl (2.02) のいずれかに近い 2 値
    assert 1.85 <= sorted_targets[0] <= 2.10, (
        f"r_form_targets[0] should be ~Cordero C-Br/Li-Cl, got {sorted_targets[0]}"
    )
    assert 1.85 <= sorted_targets[1] <= 2.10, (
        f"r_form_targets[1] should be ~Cordero C-Br/Li-Cl, got {sorted_targets[1]}"
    )

    # ≥1 trial が reached_product
    reached = [t for t in meta["trials"] if t["reached_product"]]
    assert len(reached) >= 1, "expected ≥1 reached_product trial, got 0"

    # 最終フレームで C-Br ≤ 2.13 Å かつ Li-Cl ≤ 2.22 Å (Cordero × 1.1)
    frames = read(str(out / "trajectory.xyz"), index=":")
    syms = frames[-1].get_chemical_symbols()
    pos_last = frames[-1].positions
    c_atoms = [i for i, s in enumerate(syms) if s == "C"]
    cl_idx = syms.index("Cl")
    li_idx = syms.index("Li")
    br_idx = syms.index("Br")

    # 中心 C は Br に最も近い C (CH3Br になっている想定)
    d_c_to_br = [(i, float(((pos_last[i] - pos_last[br_idx]) ** 2).sum() ** 0.5)) for i in c_atoms]
    central_c, d_c_br = min(d_c_to_br, key=lambda kv: kv[1])
    d_li_cl = float(((pos_last[li_idx] - pos_last[cl_idx]) ** 2).sum() ** 0.5)

    assert d_c_br <= 2.13, f"final C-Br should be ≤2.13 A (Cordero × 1.1), got {d_c_br:.2f}"
    assert d_li_cl <= 2.22, f"final Li-Cl should be ≤2.22 A (Cordero × 1.1), got {d_li_cl:.2f}"
