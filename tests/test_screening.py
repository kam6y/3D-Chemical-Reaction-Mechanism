"""Stage 1 screening: parallel artificial-force relax of all placement survivors."""
from __future__ import annotations

import numpy as np
from rdkit import Chem

from reactx.bond_changes import BondChanges
from reactx.config import ReactionConfig, RestraintConfig, SamplingConfig
from reactx.placement import PlacementResult, PlacementTrial
from reactx.screening import screen_all_trials


def _trivial_placement_two_trials() -> tuple[Chem.Mol, BondChanges, PlacementResult]:
    # H + H system (covers vdw_radius lookup + Atoms build), 2 placement trials.
    mol = Chem.MolFromSmiles("[H].[H]")
    mol_h = Chem.AddHs(mol)
    Chem.SanitizeMol(mol_h)
    pos1 = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    pos2 = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    trials = [
        PlacementTrial(direction=np.array([1.0, 0, 0]), d_min=3.0, positions=pos1),
        PlacementTrial(direction=np.array([-1.0, 0, 0]), d_min=4.0, positions=pos2),
    ]
    pl = PlacementResult(trials=trials, n_candidates=2, n_blocked=0,
                         blocked_reasons=[None, None])
    bc = BondChanges(formed=((0, 1),), broken=())
    return mol_h, bc, pl


def _make_cfg() -> ReactionConfig:
    return ReactionConfig(
        description="t",
        formed=((1, 2),),
        broken=tuple(),
        restraints=RestraintConfig(
            k_form=0.5, k_broken=0.0, r_broken=4.0, max_relax_steps=10,
        ),
        sampling=SamplingConfig(n_candidates=2),
    )


def test_screen_all_trials_workers_1_returns_results_in_order():
    mol_h, bc, pl = _trivial_placement_two_trials()
    cfg = _make_cfg()

    results = screen_all_trials(
        pl, mol_h, bc, cfg,
        backend="lj",
        screening_model="",
        workers=1,
        relax_fmax=0.5,
        traj_stride=5,
        seed=0,
    )
    assert [r.trial_idx for r in results] == [0, 1]
    assert all(len(r.frames) >= 1 for r in results)


def test_screen_all_trials_records_error_for_failed_trial(monkeypatch):
    mol_h, bc, pl = _trivial_placement_two_trials()
    cfg = _make_cfg()

    from reactx import screening

    real_relax = screening.relax_with_restraints
    call_count = {"n": 0}

    def flaky_relax(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("boom")
        return real_relax(*args, **kwargs)

    monkeypatch.setattr(screening, "relax_with_restraints", flaky_relax)

    results = screen_all_trials(
        pl, mol_h, bc, cfg,
        backend="lj",
        screening_model="",
        workers=1,
        relax_fmax=0.5,
        traj_stride=5,
        seed=0,
    )
    assert results[0].error is not None
    assert "boom" in results[0].error
    assert results[0].frames == []
    assert results[1].error is None


def test_screen_all_trials_empty_placement_returns_empty():
    mol = Chem.AddHs(Chem.MolFromSmiles("[H].[H]"))
    pl = PlacementResult(trials=[], n_candidates=0, n_blocked=0, blocked_reasons=[])
    bc = BondChanges(formed=((0, 1),), broken=())
    cfg = _make_cfg()
    out = screen_all_trials(
        pl, mol, bc, cfg,
        backend="lj", screening_model="", workers=1,
        relax_fmax=0.5, traj_stride=5, seed=0,
    )
    assert out == []
