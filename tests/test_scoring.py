"""Unit tests for reactx.scoring."""
import pytest
from ase import Atoms

from reactx.scoring import TrialResult, reached_product, score_trials


def _atoms_with_distances(d_form: float, d_broken: float) -> Atoms:
    """3 atoms: 0=anchor, 1=incoming (distance d_form), 2=leaving (distance d_broken)."""
    return Atoms(
        "CFC",
        positions=[[0, 0, 0], [d_form, 0, 0], [-d_broken, 0, 0]],
    )


def test_reached_product_true_when_formed_close_and_broken_far():
    a = _atoms_with_distances(d_form=1.5, d_broken=4.5)
    assert reached_product(
        a, formed=[(0, 1)], broken=[(0, 2)],
        r_form_targets=[1.5], r_broken_target=4.0,
    ) is True


def test_reached_product_false_when_formed_too_far():
    a = _atoms_with_distances(d_form=2.5, d_broken=4.5)  # formed exceeded tol
    assert reached_product(
        a, formed=[(0, 1)], broken=[(0, 2)],
        r_form_targets=[1.5], r_broken_target=4.0,
        form_tol=0.3,
    ) is False


def test_reached_product_false_when_broken_too_close():
    a = _atoms_with_distances(d_form=1.5, d_broken=3.0)  # broken still close
    assert reached_product(
        a, formed=[(0, 1)], broken=[(0, 2)],
        r_form_targets=[1.5], r_broken_target=4.0,
        broken_tol=0.5,
    ) is False


def test_score_trials_picks_lowest_peak_among_reached():
    t1 = TrialResult(trial_idx=0, rotation_deg=0.0, frames=[], energies=[],
                     reached_product=True, peak_energy=-100.0, n_steps=50)
    t2 = TrialResult(trial_idx=1, rotation_deg=15.0, frames=[], energies=[],
                     reached_product=True, peak_energy=-105.0, n_steps=60)
    t3 = TrialResult(trial_idx=2, rotation_deg=20.0, frames=[], energies=[],
                     reached_product=False, peak_energy=-200.0, n_steps=30)
    best = score_trials([t1, t2, t3])
    assert best.trial_idx == 1  # lowest peak among reached


def test_score_trials_falls_back_to_least_bad_when_all_failed():
    t1 = TrialResult(trial_idx=0, rotation_deg=0.0, frames=[], energies=[10.0, 12.0],
                     reached_product=False, peak_energy=12.0, n_steps=50)
    t2 = TrialResult(trial_idx=1, rotation_deg=15.0, frames=[], energies=[10.0, 11.0],
                     reached_product=False, peak_energy=11.0, n_steps=60)
    best = score_trials([t1, t2])
    # When all failed, prefer lowest peak
    assert best.trial_idx == 1


def test_score_trials_empty_raises():
    with pytest.raises(ValueError):
        score_trials([])
