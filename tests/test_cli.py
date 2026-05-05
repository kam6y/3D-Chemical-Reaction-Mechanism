"""CLI argument parsing + meta.json writer tests (no UMA invocation)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from reactx.cli import _write_outputs, build_parser, main
from reactx.config import (
    ModelConfig,
    NebConfig,
    ParallelConfig,
    ReactionConfig,
    RestraintConfig,
    SamplingConfig,
)
from reactx.placement import PlacementResult, PlacementTrial
from reactx.scoring import ScreeningTrialResult


def _sample_cfg() -> ReactionConfig:
    return ReactionConfig(
        description="sample",
        formed=((1, 2),),
        broken=((1, 3),),
        restraints=RestraintConfig(
            k_form=2.0, k_broken=2.0, r_broken=5.0,
            max_relax_steps=200, r_form=None,
        ),
        sampling=SamplingConfig(),
        model=ModelConfig(),
        neb=NebConfig(),
        parallel=ParallelConfig(),
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


def test_default_flags_parse():
    p = build_parser()
    a = p.parse_args(["run", "examples/sn2.rxn", "-o", "out/"])
    assert a.cmd == "run"
    assert a.backend == "uma"
    assert a.seed == 0
    assert a.relax_fmax == 0.1
    assert a.traj_stride == 5
    assert a.render is False
    assert a.blender_exe == "blender"


def test_dropped_flags_now_rejected():
    p = build_parser()
    for flag in [
        "--reaction-type", "--k-form", "--k-broken", "--r-form", "--r-broken",
        "--max-relax-steps", "--n-angles", "--cone-half-deg",
        "--no-mmff-prescreen", "--prescreen-keep", "--prescreen-steps",
        "--neb-refine", "--neb-images", "--model",
    ]:
        with pytest.raises(SystemExit):
            args = ["run", "examples/sn2.rxn", "-o", "out/", flag]
            if flag != "--no-mmff-prescreen" and flag != "--neb-refine":
                args.append("0")
            p.parse_args(args)


def test_meta_json_includes_description_and_effective_params(tmp_path: Path):
    args = argparse.Namespace(backend="lj", output=tmp_path)
    cfg = _sample_cfg()
    r_form_targets = [1.47]
    screen = [ScreeningTrialResult(
        trial_idx=0, direction=np.array([0.0, 0.0, 1.0]),
        frames=[], energies=[1.0, 2.0],
        reached_product=True, peak_energy=2.0, n_steps=2,
    )]
    rc = _write_outputs(
        args, cfg, screen,
        top_k_indices=[0], neb_results=[],
        placement=None, t_start=0.0, breakdown={},
        r_form_targets=r_form_targets, rc=0, selected_trial=0,
    )
    assert rc == 0
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["description"] == "sample"
    assert "reaction_type" not in meta
    eff = meta["effective_params"]
    assert eff["k_form"] == 2.0
    assert eff["k_broken"] == 2.0
    assert eff["r_broken"] == 5.0
    assert eff["max_relax_steps"] == 200
    assert eff["r_form_targets"] == [1.47]
    assert eff["n_candidates"] == 64
    assert eff["screening_model"] == "uma-s-1p2"
    assert eff["neb_model"] == "uma-s-1p2"
    assert eff["top_k"] == 4
    assert eff["n_images"] == 7


def test_meta_json_placement_block(tmp_path: Path):
    args = argparse.Namespace(backend="lj", output=tmp_path)
    cfg = _sample_cfg()
    placement = _sample_placement()
    screen = [ScreeningTrialResult(
        trial_idx=0, direction=np.array([0.0, 0.0, 1.0]),
        frames=[], energies=[1.0, 2.0],
        reached_product=True, peak_energy=2.0, n_steps=2,
    )]
    rc = _write_outputs(
        args, cfg, screen,
        top_k_indices=[0], neb_results=[],
        placement=placement, t_start=0.0, breakdown={},
        r_form_targets=[1.47], rc=0, selected_trial=0,
    )
    assert rc == 0
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["placement"] == {
        "n_candidates": 64,
        "n_blocked": 63,
        "n_valid": 1,
    }
    assert "prescreen" not in meta


def test_meta_json_screening_trials_have_direction_field(tmp_path: Path):
    args = argparse.Namespace(backend="lj", output=tmp_path)
    cfg = _sample_cfg()
    screen = [ScreeningTrialResult(
        trial_idx=0, direction=np.array([0.5, -0.5, 0.7071]),
        frames=[], energies=[1.0, 2.0],
        reached_product=True, peak_energy=2.0, n_steps=2,
    )]
    rc = _write_outputs(
        args, cfg, screen,
        top_k_indices=[0], neb_results=[],
        placement=_sample_placement(), t_start=0.0, breakdown={},
        r_form_targets=[1.47], rc=0, selected_trial=0,
    )
    assert rc == 0
    meta = json.loads((tmp_path / "meta.json").read_text())
    trials = meta["screening_trials"]
    assert isinstance(trials[0]["direction"], list)
    assert len(trials[0]["direction"]) == 3
    assert trials[0]["direction"][0] == pytest.approx(0.5)
    assert "rotation_deg" not in trials[0]


def test_cli_phase8_meta_json_schema(tmp_path: Path, tmp_rxn_with_toml):
    """meta.json contains screening_trials, top_k_indices, neb_results."""
    rxn = tmp_rxn_with_toml("sn2", toml_body=
        'description = "sn2 lj fast"\n'
        'formed = [[1, 3]]\n'
        'broken = [[1, 2]]\n'
        '[restraints]\nk_form=0.1\nk_broken=0.1\nr_broken=4.0\nmax_relax_steps=3\n'
        '[sampling]\nn_candidates=2\n'
        '[neb]\ntop_k=1\nn_images=3\nmax_steps=2\npad_frames=0\ninterp_factor=1\n'
        '[parallel]\nscreening_workers=1\nneb_workers=1\n'
    )
    out = tmp_path / "sn2"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "lj"])
    # LJ backend で UMA-quality NEB は出ないが、cli 経路全体が完走することを確認
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text())
    assert "screening_trials" in meta
    assert "top_k_indices" in meta
    assert "neb_results" in meta
    assert meta["effective_params"]["screening_model"]
    assert meta["effective_params"]["neb_model"]
    assert isinstance(meta["wall_clock_breakdown"], dict)
    assert {"placement", "screening", "neb"} <= set(meta["wall_clock_breakdown"])
