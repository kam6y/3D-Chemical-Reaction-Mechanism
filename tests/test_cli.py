"""CLI argument parsing + meta.json writer tests (no UMA invocation)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from reactx.cli import _write_outputs_and_exit, build_parser
from reactx.config import (
    PrescreenConfig,
    ReactionConfig,
    RestraintConfig,
    SamplingConfig,
)
from reactx.scoring import TrialResult


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
        prescreen=PrescreenConfig(),
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
    r_form_targets = [1.47]
    trials = [TrialResult(
        trial_idx=0, rotation_deg=0.0, frames=[], energies=[1.0, 2.0],
        reached_product=True, peak_energy=2.0, n_steps=2,
    )]
    with patch("reactx.cli.score_trials", return_value=trials[0]):
        rc = _write_outputs_and_exit(
            args, trials, t_start=0.0,
            neb_refined=False, rc=0,
            cfg=cfg, r_form_targets=r_form_targets, prescreen_meta=None,
        )
    assert rc == 0
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["description"] == "sample"
    assert "reaction_type" not in meta
    assert meta["effective_params"] == {
        "k_form": 2.0, "k_broken": 2.0,
        "r_broken": 5.0, "max_relax_steps": 200,
        "r_form_targets": [1.47],
    }


def test_meta_json_prescreen_block(tmp_path: Path):
    args = argparse.Namespace(backend="lj", output=tmp_path)
    cfg = _sample_cfg()
    pre = {
        "enabled": True, "kept": [0, 3, 5], "skipped": [1, 2, 4, 6, 7],
        "mmff_failed": False, "wall_clock_seconds": 1.8,
    }
    trials = [TrialResult(
        trial_idx=0, rotation_deg=0.0, frames=[], energies=[1.0, 2.0],
        reached_product=True, peak_energy=2.0, n_steps=2,
    )]
    with patch("reactx.cli.score_trials", return_value=trials[0]):
        rc = _write_outputs_and_exit(
            args, trials, t_start=0.0,
            neb_refined=False, rc=0,
            cfg=cfg, r_form_targets=[1.47], prescreen_meta=pre,
        )
    assert rc == 0
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["prescreen"] == pre


def test_meta_json_when_cfg_missing_writes_null_effective(tmp_path: Path):
    """rc != 0 path may pass cfg=None (e.g. when load_config failed early)."""
    args = argparse.Namespace(backend="lj", output=tmp_path)
    rc = _write_outputs_and_exit(
        args, trials=[], t_start=0.0,
        neb_refined=False, rc=1,
        cfg=None, r_form_targets=None, prescreen_meta=None,
    )
    assert rc == 1
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["description"] is None
    assert meta["effective_params"] is None
