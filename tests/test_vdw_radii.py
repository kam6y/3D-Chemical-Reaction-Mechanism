"""vdW radii lookup: Alvarez (2013) Dalton Trans. 42, 8617."""
import pytest

from reactx.vdw_radii import VDW_RADII_ANGSTROM, FALLBACK_RADIUS, vdw_radius


def test_vdw_radii_alvarez_z1_to_z83_present():
    # 元素記号は Z=1..83 (H..Bi)。OMol25 / UMA omol 訓練範囲と一致。
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
    assert expected_symbols.issubset(VDW_RADII_ANGSTROM.keys())


def test_vdw_radius_known_values():
    # 既知値 (Alvarez 2013 from blender/render.py)
    assert vdw_radius("H") == pytest.approx(1.20)
    assert vdw_radius("C") == pytest.approx(1.77)
    assert vdw_radius("N") == pytest.approx(1.66)
    assert vdw_radius("O") == pytest.approx(1.50)
    assert vdw_radius("Cl") == pytest.approx(1.82)


def test_vdw_radius_fallback_for_unknown_symbol(caplog):
    # Z > 83 (例: "Po", "U") はテーブル外 → fallback + warning
    with caplog.at_level("WARNING"):
        r = vdw_radius("Po")
    assert r == FALLBACK_RADIUS
    assert "Po" in caplog.text


def test_vdw_radius_fallback_value_is_alvarez_median_neighborhood():
    # FALLBACK_RADIUS は 1.50 (Alvarez Z=1..83 median 近傍、O と一致するのは偶然)
    assert FALLBACK_RADIUS == pytest.approx(1.50)
