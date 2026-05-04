"""vdW radii lookup: Alvarez (2013) Dalton Trans. 42, 8617."""
import logging

import pytest

from reactx.vdw_radii import FALLBACK_RADIUS, VDW_RADII_ANGSTROM, vdw_radius


def test_vdw_radii_alvarez_z1_to_z83_exact_set():
    # 元素記号は Z=1..83 (H..Bi) **だけ**。OMol25 / UMA omol 訓練範囲と一致。
    # 将来 Z>83 を追加すると test_vdw_radius_fallback_for_unknown_symbol
    # ("Po") が壊れるので両側等号でロック。
    expected_symbols = {
        "H", "He",
        "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
        "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr",
        "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "In", "Sn", "Sb", "Te", "I", "Xe",
        "Cs", "Ba",
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
        "Tm", "Yb", "Lu",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    }
    assert set(VDW_RADII_ANGSTROM.keys()) == expected_symbols
    assert len(VDW_RADII_ANGSTROM) == 83


def test_vdw_radius_known_values():
    # 既知値 (Alvarez 2013 from blender/render.py)
    assert vdw_radius("H") == pytest.approx(1.20)
    assert vdw_radius("C") == pytest.approx(1.77)
    assert vdw_radius("N") == pytest.approx(1.66)
    assert vdw_radius("O") == pytest.approx(1.50)
    assert vdw_radius("Cl") == pytest.approx(1.82)


def test_vdw_radius_fallback_for_unknown_symbol(caplog):
    # Z > 83 (例: "Po", "U") はテーブル外 → fallback + warning
    # cli._configure_reactx_logging が `reactx` parent で propagate=False を
    # 設定するため (test_cli_* 後)、caplog の root handler が捕捉できない。
    # 当該テスト中だけ propagation を一時的に有効化する。
    reactx_log = logging.getLogger("reactx")
    saved = reactx_log.propagate
    reactx_log.propagate = True
    try:
        with caplog.at_level("WARNING"):
            r = vdw_radius("Po")
    finally:
        reactx_log.propagate = saved
    assert r == FALLBACK_RADIUS
    assert "Po" in caplog.text


def test_vdw_radius_fallback_value_is_lower_quartile():
    # FALLBACK_RADIUS = 1.50 Å は O の値 (= Alvarez Z=1..83 の lower quartile 付近)。
    # 「unknown 元素では小さめに見積もって blocking を緩める」方針の固定値。
    # 実用上 OMol25 は Z=1..83 しか出ないので発火しない安全網。
    assert pytest.approx(1.50) == FALLBACK_RADIUS
