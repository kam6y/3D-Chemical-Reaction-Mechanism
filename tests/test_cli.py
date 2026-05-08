"""CLI argument parsing + meta.json writer tests (no UMA invocation)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from reactx.cli import _write_outputs_and_exit, build_parser, main
from reactx.config import (
    AFIRSection,
    ReactionConfigV9,
    SamplingConfig,
    ScoringSection,
)
from reactx.placement import PlacementResult, PlacementTrial
from reactx.scoring import TrialResultV9 as TrialResult


def _sample_cfg() -> ReactionConfigV9:
    return ReactionConfigV9(
        description="sample",
        formed=((1, 2),),
        broken=((1, 3),),
        afir=AFIRSection(
            alpha_formed=2.0,
            alpha_broken=2.0,
            max_relax_steps=200,
        ),
        scoring=ScoringSection(
            r_broken_threshold=5.0,
            r_formed_threshold=None,
        ),
        sampling=SamplingConfig(),
    )


def _sample_placement() -> PlacementResult:
    return PlacementResult(
        trials=[PlacementTrial(
            direction=np.array([0.0, 0.0, 1.0]),
            d_min=3.5,
            positions=np.zeros((3, 3)),
        )],
        n_candidates=64,
        n_blocked=63,
        blocked_reasons=[None] + ["angle_shadow:atom_index=0"] * 63,
    )


def _trial(
    trial_idx: int = 0,
    direction=None,
    *,
    reached_product: bool = True,
    peak_energy: float = 2.0,
    n_steps: int = 2,
    formed_thresholds=None,
    broken_thresholds=None,
    energies=None,
) -> TrialResult:
    if direction is None:
        direction = np.array([0.0, 0.0, 1.0])
    if formed_thresholds is None:
        formed_thresholds = [1.6]
    if broken_thresholds is None:
        broken_thresholds = [4.0]
    if energies is None:
        energies = [1.0, 2.0]
    return TrialResult(
        trial_idx=trial_idx,
        direction=direction,
        frames=[],
        energies=energies,
        reached_product=reached_product,
        peak_energy=peak_energy,
        n_steps=n_steps,
        formed_thresholds=formed_thresholds,
        broken_thresholds=broken_thresholds,
        product_distance_residual=0.0,
        formed_latch_count=1,
        broken_latch_count=1,
        initial_latched_formed=0,
        initial_latched_broken=0,
    )


def test_default_flags_parse():
    p = build_parser()
    a = p.parse_args(["run", "examples/sn2.rxn", "-o", "out/"])
    assert a.cmd == "run"
    assert a.backend == "uma"
    assert a.model == "uma-m-1p1"
    assert a.seed == 0
    assert a.relax_fmax == 0.1
    assert a.traj_stride == 5
    assert a.neb_refine is False
    assert a.neb_images == 7
    assert a.render is False
    assert a.blender_exe == "blender"


def test_dropped_flags_now_rejected():
    p = build_parser()
    for flag in [
        "--reaction-type", "--k-form", "--k-broken", "--r-form", "--r-broken",
        "--max-relax-steps", "--n-angles", "--cone-half-deg",
        "--no-mmff-prescreen", "--prescreen-keep", "--prescreen-steps",
    ]:
        with pytest.raises(SystemExit):
            args = ["run", "examples/sn2.rxn", "-o", "out/", flag]
            if flag != "--no-mmff-prescreen":
                args.append("0")
            p.parse_args(args)


def test_neb_refine_flag():
    p = build_parser()
    a = p.parse_args(["run", "examples/sn2.rxn", "-o", "out/", "--neb-refine"])
    assert a.neb_refine is True


def test_meta_json_includes_description_and_effective_params(tmp_path: Path):
    args = argparse.Namespace(backend="lj", output=tmp_path)
    cfg = _sample_cfg()
    trials = [_trial()]
    with patch("reactx.cli.score_trials", return_value=trials[0]):
        rc = _write_outputs_and_exit(
            args, trials, t_start=0.0,
            neb_refined=False, rc=0,
            cfg=cfg, placement=None,
        )
    assert rc == 0
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["description"] == "sample"
    assert "reaction_type" not in meta
    assert meta["effective_params"] == {
        "alpha_formed": 2.0,
        "alpha_broken": 2.0,
        "max_relax_steps": 200,
        "r_broken_threshold": 5.0,
        "r_formed_threshold": None,
        "n_candidates": 64,
    }


def test_meta_json_placement_block(tmp_path: Path):
    args = argparse.Namespace(backend="lj", output=tmp_path)
    cfg = _sample_cfg()
    placement = _sample_placement()
    trials = [_trial()]
    with patch("reactx.cli.score_trials", return_value=trials[0]):
        rc = _write_outputs_and_exit(
            args, trials, t_start=0.0,
            neb_refined=False, rc=0,
            cfg=cfg, placement=placement,
        )
    assert rc == 0
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["placement"] == {
        "n_candidates": 64,
        "n_blocked": 63,
        "n_valid": 1,
        "placement_kind": "single_anchor",
    }
    assert "prescreen" not in meta


def test_meta_json_trials_have_direction_field(tmp_path: Path):
    args = argparse.Namespace(backend="lj", output=tmp_path)
    cfg = _sample_cfg()
    trials = [_trial(direction=np.array([0.5, -0.5, 0.7071]))]
    with patch("reactx.cli.score_trials", return_value=trials[0]):
        rc = _write_outputs_and_exit(
            args, trials, t_start=0.0,
            neb_refined=False, rc=0,
            cfg=cfg, placement=_sample_placement(),
        )
    assert rc == 0
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert isinstance(meta["trials"][0]["direction"], list)
    assert len(meta["trials"][0]["direction"]) == 3
    assert meta["trials"][0]["direction"][0] == pytest.approx(0.5)
    assert "rotation_deg" not in meta["trials"][0]


def test_meta_json_when_cfg_missing_writes_null_effective(tmp_path: Path):
    """rc != 0 path may pass cfg=None (e.g. when load_config failed early)."""
    args = argparse.Namespace(backend="lj", output=tmp_path)
    rc = _write_outputs_and_exit(
        args, trials=[], t_start=0.0,
        neb_refined=False, rc=1,
        cfg=None, placement=None,
    )
    assert rc == 1
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["description"] is None
    assert meta["effective_params"] is None
    assert meta["placement"] == {
        "n_candidates": 0, "n_blocked": 0, "n_valid": 0, "placement_kind": None,
    }


def test_meta_includes_per_bond_lists_when_toml_uses_lists(tmp_path, tmp_rxn_with_toml):
    """TOML で list 指定したら meta.json も list を保持する。"""
    body = """\
description = "sn2 with list alpha_formed"
formed = [[1, 3]]
broken = [[1, 2]]

[afir]
alpha_formed = [0.5]
alpha_broken = 1.0
max_relax_steps = 5

[scoring]
r_broken_threshold = 4.0

[sampling]
n_candidates = 2
"""
    rxn = tmp_rxn_with_toml("sn2", toml_body=body)
    out = tmp_path / "out"
    main(["run", str(rxn), "-o", str(out), "--backend", "lj"])
    meta_path = out / "meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    ep = meta["effective_params"]
    assert ep["alpha_formed"] == [0.5]   # list passes through
    assert ep["alpha_broken"] == 1.0     # scalar stays scalar
    assert ep["r_broken_threshold"] == 4.0


def test_meta_includes_placement_kind_and_orientation_phase_7_compat(tmp_path, tmp_rxn_with_toml):
    """既存 SN2 (Phase 7 single-anchor) で placement_kind と orientation が出力される。"""
    body = """\
description = "sn2 quick"
formed = [[1, 3]]
broken = [[1, 2]]

[afir]
alpha_formed = 0.5
alpha_broken = 1.0
max_relax_steps = 5

[scoring]
r_broken_threshold = 4.0

[sampling]
n_candidates = 2
"""
    rxn = tmp_rxn_with_toml("sn2", toml_body=body)
    out = tmp_path / "out"
    main(["run", str(rxn), "-o", str(out), "--backend", "lj"])
    meta = json.loads((out / "meta.json").read_text())
    assert meta["placement"]["placement_kind"] == "single_anchor"
    assert all(t["orientation"] == "single" for t in meta["trials"])
