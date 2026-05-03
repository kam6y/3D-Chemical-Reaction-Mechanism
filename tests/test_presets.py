"""Unit tests for reaction-type presets."""
import pytest

from reactx.presets import PRESETS, ReactionPreset, get_preset


def test_sn2_anion_values():
    p = get_preset("sn2_anion")
    assert p.name == "sn2_anion"
    assert p.k_form == 0.5
    assert p.k_broken == 1.0
    assert p.r_broken == 4.0
    assert p.max_relax_steps == 100
    assert p.r_form is None  # 元素表 fallback


def test_proton_transfer_values():
    p = get_preset("proton_transfer")
    assert p.name == "proton_transfer"
    assert p.k_form == 0.5
    assert p.k_broken == 1.0
    assert p.r_broken == 4.0
    assert p.max_relax_steps == 100
    assert p.r_form == 1.05


def test_menshutkin_values():
    p = get_preset("menshutkin")
    assert p.name == "menshutkin"
    assert p.k_form == 2.0
    assert p.k_broken == 2.0
    assert p.r_broken == 5.0
    assert p.max_relax_steps == 200
    assert p.r_form is None


def test_get_preset_unknown_raises():
    with pytest.raises(ValueError) as exc:
        get_preset("not_a_preset")
    msg = str(exc.value)
    assert "not_a_preset" in msg
    assert "sn2_anion" in msg
    assert "menshutkin" in msg
    assert "proton_transfer" in msg


def test_presets_dict_keys():
    assert set(PRESETS) == {
        "sn2_anion",
        "proton_transfer",
        "menshutkin",
        "e2",
        "sn1_dissoc",
        "sn1_recomb",
    }


def test_preset_is_frozen():
    p = get_preset("sn2_anion")
    with pytest.raises((AttributeError, Exception)):
        p.k_form = 99.0  # frozen dataclass should reject mutation


def test_reaction_preset_dataclass_signature():
    # Default r_form should be None when not provided
    p = ReactionPreset(
        name="ad_hoc", k_form=1.0, k_broken=2.0,
        r_broken=4.5, max_relax_steps=120,
    )
    assert p.r_form is None


def test_e2_preset_values():
    p = get_preset("e2")
    assert p.name == "e2"
    assert p.k_form == 1.0
    assert p.k_broken == 1.0
    assert p.r_broken == 4.0
    assert p.max_relax_steps == 200
    assert p.r_form is None  # element-pair table


def test_sn1_dissoc_preset_values():
    p = get_preset("sn1_dissoc")
    assert p.name == "sn1_dissoc"
    assert p.k_form == 0.0
    assert p.k_broken == 2.0
    assert p.r_broken == 6.0
    assert p.max_relax_steps == 200
    assert p.r_form is None


def test_sn1_recomb_preset_values():
    p = get_preset("sn1_recomb")
    assert p.name == "sn1_recomb"
    assert p.k_form == 1.0
    assert p.k_broken == 0.0
    assert p.r_broken == 4.0
    assert p.max_relax_steps == 200
    assert p.r_form is None
