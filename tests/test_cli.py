"""CLI argument parsing smoke tests (no UMA invocation)."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from reactx.cli import _resolve_effective_params, _write_outputs_and_exit, build_parser
from reactx.scoring import TrialResult


def test_resolve_effective_params_returns_list_for_r_form_targets():
    """_resolve_effective_params returns r_form_targets as list[float] (Phase 3)."""
    import argparse

    from reactx.cli import _resolve_effective_params
    args = argparse.Namespace(
        reaction_type="sn2_anion",
        k_form=None, k_broken=None, r_broken=None, r_form=None,
        max_relax_steps=None,
    )
    syms = ["C", "Cl", "O"]
    formed_pairs = [(0, 2)]  # C-O formed
    eff = _resolve_effective_params(args, syms, formed_pairs)
    assert isinstance(eff["r_form_targets"], list)
    assert len(eff["r_form_targets"]) == 1
    assert abs(eff["r_form_targets"][0] - 1.43) < 1e-3  # C-O table value


def test_resolve_effective_params_empty_formed_returns_empty_list():
    import argparse

    from reactx.cli import _resolve_effective_params
    args = argparse.Namespace(
        reaction_type="sn1_dissoc",
        k_form=None, k_broken=None, r_broken=None, r_form=None,
        max_relax_steps=None,
    )
    eff = _resolve_effective_params(args, ["C", "Br"], [])
    assert eff["r_form_targets"] == []


def test_resolve_effective_params_scalar_r_form_broadcasts():
    import argparse

    from reactx.cli import _resolve_effective_params
    args = argparse.Namespace(
        reaction_type="sn2_anion",
        k_form=None, k_broken=None, r_broken=None, r_form=1.10,
        max_relax_steps=None,
    )
    eff = _resolve_effective_params(args, ["C", "Cl", "O", "F"], [(0, 2), (0, 3)])
    assert eff["r_form_targets"] == [1.10, 1.10]


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
    # Prescreen defaults:
    assert a.no_mmff_prescreen is False
    assert a.prescreen_keep == 3
    assert a.prescreen_steps == 30
    # Unchanged:
    assert a.relax_fmax == 0.1
    assert a.traj_stride == 5
    assert a.neb_refine is False
    assert a.neb_images == 7


def test_no_mmff_prescreen_flag():
    p = build_parser()
    a = p.parse_args([
        "run", "examples/sn2.rxn", "-o", "out/", "--no-mmff-prescreen",
    ])
    assert a.no_mmff_prescreen is True


def test_prescreen_keep_override():
    p = build_parser()
    a = p.parse_args([
        "run", "examples/sn2.rxn", "-o", "out/", "--prescreen-keep", "1",
    ])
    assert a.prescreen_keep == 1


def test_prescreen_steps_override():
    p = build_parser()
    a = p.parse_args([
        "run", "examples/sn2.rxn", "-o", "out/", "--prescreen-steps", "10",
    ])
    assert a.prescreen_steps == 10


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


def _make_args(**overrides):
    """Build argparse Namespace for tests by parsing then overriding."""
    p = build_parser()
    a = p.parse_args(["run", "examples/sn2.rxn", "-o", "out/"])
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def test_resolve_defaults_to_sn2_anion_preset():
    args = _make_args()
    syms = ["C", "Cl", "H", "H", "H", "F"]  # formed = (0, 5) -> C-F
    eff = _resolve_effective_params(args, syms, formed_pairs=[(0, 5)])
    assert eff["reaction_type"] == "sn2_anion"
    assert eff["k_form"] == 0.5
    assert eff["k_broken"] == 1.0
    assert eff["r_broken"] == 4.0
    assert eff["max_relax_steps"] == 100
    # sn2_anion has no r_form override -> element-table lookup C-F = 1.39
    assert eff["r_form_targets"][0] == pytest.approx(1.39)


def test_resolve_menshutkin_preset():
    args = _make_args(reaction_type="menshutkin")
    syms = ["N", "C", "Cl", "H", "H", "H", "H", "H", "H"]
    eff = _resolve_effective_params(args, syms, formed_pairs=[(0, 1)])  # N-C
    assert eff["reaction_type"] == "menshutkin"
    assert eff["k_form"] == 2.0
    assert eff["k_broken"] == 2.0
    assert eff["r_broken"] == 5.0
    assert eff["max_relax_steps"] == 200
    # menshutkin r_form = None -> element table N-C = 1.47
    assert eff["r_form_targets"][0] == pytest.approx(1.47)


def test_resolve_proton_transfer_preset_uses_r_form_1_05():
    args = _make_args(reaction_type="proton_transfer")
    syms = ["H", "Cl", "N", "H", "H"]
    eff = _resolve_effective_params(args, syms, formed_pairs=[(2, 0)])  # N-H
    assert eff["r_form_targets"][0] == pytest.approx(1.05)


def test_individual_flag_overrides_preset():
    args = _make_args(reaction_type="menshutkin", k_form=3.5, r_broken=6.0)
    syms = ["N", "C", "Cl"]
    eff = _resolve_effective_params(args, syms, formed_pairs=[(0, 1)])
    assert eff["k_form"] == 3.5  # overridden
    assert eff["r_broken"] == 6.0  # overridden
    assert eff["k_broken"] == 2.0  # from preset
    assert eff["max_relax_steps"] == 200  # from preset


def test_r_form_individual_flag_overrides_preset_r_form():
    args = _make_args(reaction_type="proton_transfer", r_form=1.10)
    syms = ["H", "Cl", "N"]
    eff = _resolve_effective_params(args, syms, formed_pairs=[(2, 0)])
    assert eff["r_form_targets"][0] == pytest.approx(1.10)  # individual flag wins over preset 1.05


def test_meta_json_includes_reaction_type_and_effective_params(tmp_path: Path):
    args = argparse.Namespace(
        backend="lj",
        output=tmp_path,
        reaction_type="menshutkin",
    )
    eff = {
        "reaction_type": "menshutkin",
        "k_form": 2.0, "k_broken": 2.0,
        "r_broken": 5.0, "max_relax_steps": 200,
        "r_form_targets": [1.47],
    }
    trials = [TrialResult(
        trial_idx=0, rotation_deg=0.0, frames=[], energies=[1.0, 2.0],
        reached_product=True, peak_energy=2.0, n_steps=2,
    )]
    # Patch score_trials to avoid relying on its internals here.
    with patch("reactx.cli.score_trials", return_value=trials[0]):
        rc = _write_outputs_and_exit(
            args, trials, t_start=0.0,
            neb_refined=False, rc=0, effective=eff,
        )
    assert rc == 0
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["reaction_type"] == "menshutkin"
    assert meta["effective_params"] == {
        "k_form": 2.0, "k_broken": 2.0,
        "r_broken": 5.0, "max_relax_steps": 200,
        "r_form_targets": [1.47],
    }


def test_meta_json_includes_prescreen_section_when_enabled(tmp_path: Path):
    args = argparse.Namespace(
        backend="lj", output=tmp_path, reaction_type="sn2_anion",
    )
    prescreen_meta = {
        "enabled": True,
        "kept": [0, 3, 5],
        "skipped": [1, 2, 4, 6, 7],
        "mmff_failed": False,
        "wall_clock_seconds": 1.8,
    }
    trials = [TrialResult(
        trial_idx=0, rotation_deg=0.0, frames=[], energies=[1.0, 2.0],
        reached_product=True, peak_energy=2.0, n_steps=2,
    )]
    with patch("reactx.cli.score_trials", return_value=trials[0]):
        rc = _write_outputs_and_exit(
            args, trials, t_start=0.0,
            neb_refined=False, rc=0,
            effective=None, prescreen_meta=prescreen_meta,
        )
    assert rc == 0
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["prescreen"] == prescreen_meta


def test_meta_json_prescreen_disabled(tmp_path: Path):
    args = argparse.Namespace(
        backend="lj", output=tmp_path, reaction_type="sn2_anion",
    )
    prescreen_meta = {
        "enabled": False, "kept": None, "skipped": None,
        "mmff_failed": None, "wall_clock_seconds": 0.0,
    }
    trials = [TrialResult(
        trial_idx=0, rotation_deg=0.0, frames=[], energies=[1.0, 2.0],
        reached_product=True, peak_energy=2.0, n_steps=2,
    )]
    with patch("reactx.cli.score_trials", return_value=trials[0]):
        rc = _write_outputs_and_exit(
            args, trials, t_start=0.0,
            neb_refined=False, rc=0,
            effective=None, prescreen_meta=prescreen_meta,
        )
    assert rc == 0
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["prescreen"]["enabled"] is False
    assert meta["prescreen"]["kept"] is None
    assert meta["prescreen"]["mmff_failed"] is None
