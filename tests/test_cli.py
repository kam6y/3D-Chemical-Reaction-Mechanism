"""CLI argument parsing smoke tests (no UMA invocation)."""
import pytest

from reactx.cli import build_parser


def test_default_flags_parse():
    p = build_parser()
    a = p.parse_args(["run", "examples/sn2.rxn", "-o", "out/"])
    assert a.cmd == "run"
    assert a.n_angles == 8
    assert a.cone_half_deg == 30.0
    assert a.seed == 0
    # Preset-overrideable flags now default to None (sentinel).
    assert a.r_form is None
    assert a.r_broken is None
    assert a.k_form is None
    assert a.k_broken is None
    assert a.max_relax_steps is None
    # Default reaction type:
    assert a.reaction_type == "sn2_anion"
    # Unchanged:
    assert a.relax_fmax == 0.1
    assert a.traj_stride == 5
    assert a.neb_refine is False
    assert a.neb_images == 7


def test_neb_refine_flag():
    p = build_parser()
    a = p.parse_args(["run", "examples/sn2.rxn", "-o", "out/", "--neb-refine"])
    assert a.neb_refine is True


def test_n_angles_override():
    p = build_parser()
    a = p.parse_args(["run", "examples/sn2.rxn", "-o", "out/", "--n-angles", "1"])
    assert a.n_angles == 1


def test_r_form_override():
    p = build_parser()
    a = p.parse_args(["run", "examples/sn2.rxn", "-o", "out/", "--r-form", "1.05"])
    assert a.r_form == 1.05


def test_reaction_type_explicit():
    p = build_parser()
    a = p.parse_args([
        "run", "examples/sn2.rxn", "-o", "out/",
        "--reaction-type", "menshutkin",
    ])
    assert a.reaction_type == "menshutkin"


def test_reaction_type_unknown_rejected():
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args([
            "run", "examples/sn2.rxn", "-o", "out/",
            "--reaction-type", "not_a_preset",
        ])
