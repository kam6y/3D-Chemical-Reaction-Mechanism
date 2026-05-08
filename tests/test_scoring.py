"""Unit tests for reactx.scoring (Phase 9 API)."""
import numpy as np
import pytest
from ase import Atoms

from reactx.config import ConfigError
from reactx.scoring import (
    TrialResult,
    count_initial_latched,
    product_distance_residual,
    reached_product,
    resolve_broken_thresholds,
    resolve_formed_thresholds,
    score_trials,
    select_best_trial,
)


def _atoms_cc(d: float):
    return Atoms("CC", positions=[[0.0, 0.0, 0.0], [d, 0.0, 0.0]])


def test_resolve_formed_thresholds_default_is_1_15_times_rsum():
    a = _atoms_cc(1.5)
    thrs = resolve_formed_thresholds(a, formed=[(0, 1)], override=None)
    assert thrs == pytest.approx([1.748], abs=1e-6)


def test_resolve_broken_thresholds_required_when_nonempty():
    a = _atoms_cc(1.0)
    with pytest.raises(ConfigError, match="r_broken_threshold"):
        resolve_broken_thresholds(a, broken=[(0, 1)], override=None)


def test_reached_product_per_pair():
    a = _atoms_cc(1.5)
    assert reached_product(a, [(0, 1)], [], [1.6], [])
    assert not reached_product(_atoms_cc(2.0), [(0, 1)], [], [1.6], [])


def test_residual_zero_when_reached():
    assert product_distance_residual(
        _atoms_cc(1.4), [(0, 1)], [], [1.6], []
    ) == 0.0


def test_residual_positive_when_formed_overshoot():
    r = product_distance_residual(_atoms_cc(2.0), [(0, 1)], [], [1.6], [])
    assert r == pytest.approx(0.16, abs=1e-12)


def test_count_initial_latched_both():
    counts = count_initial_latched(
        _atoms_cc(1.5), [(0, 1)], [(0, 1)], [1.6], [1.4],
    )
    assert counts == {"formed": 1, "broken": 1}


def _trial(idx, *, reached, peak, residual=0.0):
    return TrialResult(
        trial_idx=idx,
        direction=np.array([0.0, 0.0, 1.0]),
        frames=[], energies=[],
        reached_product=reached, peak_energy=peak,
        n_steps=10,
        formed_thresholds=[1.6],
        broken_thresholds=[],
        product_distance_residual=residual,
        formed_latch_count=0, broken_latch_count=0,
        initial_latched_formed=0, initial_latched_broken=0,
    )


def test_score_trials_prefers_reached_with_lowest_peak():
    best = score_trials([
        _trial(0, reached=True, peak=10.0),
        _trial(1, reached=True, peak=5.0),
        _trial(2, reached=False, peak=1.0),
    ])
    assert best.trial_idx == 1


def test_score_trials_fallback_uses_residual_then_peak():
    best = score_trials([
        _trial(0, reached=False, peak=1.0, residual=10.0),
        _trial(1, reached=False, peak=5.0, residual=2.0),
        _trial(2, reached=False, peak=2.0, residual=2.0),
    ])
    assert best.trial_idx == 2


def test_score_trials_empty_raises():
    with pytest.raises(ValueError):
        score_trials([])


def test_select_best_trial_prefers_reached_product():
    trials = [
        _trial(0, reached=False, peak=-5.0),
        _trial(1, reached=True, peak=-3.0),
        _trial(2, reached=True, peak=-4.0),
    ]
    assert select_best_trial(trials) == 2


def test_select_best_trial_falls_back_when_none_reached():
    trials = [
        _trial(0, reached=False, peak=-5.0, residual=10.0),
        _trial(1, reached=False, peak=-7.0, residual=5.0),
        _trial(2, reached=False, peak=-3.0, residual=20.0),
    ]
    assert select_best_trial(trials) == 1


def test_select_best_trial_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        select_best_trial([])
