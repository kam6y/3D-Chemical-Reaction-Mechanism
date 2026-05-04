"""Unit tests for reactx.scoring."""
import numpy as np
import pytest
from ase import Atoms

from reactx.scoring import ScreeningTrialResult, reached_product, score_trials, select_best_trial


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
    t1 = ScreeningTrialResult(trial_idx=0, direction=np.array([0.0, 0.0, 1.0]), frames=[], energies=[],
                     reached_product=True, peak_energy=-100.0, n_steps=50)
    t2 = ScreeningTrialResult(trial_idx=1, direction=np.array([1.0, 0.0, 0.0]), frames=[], energies=[],
                     reached_product=True, peak_energy=-105.0, n_steps=60)
    t3 = ScreeningTrialResult(trial_idx=2, direction=np.array([0.0, 1.0, 0.0]), frames=[], energies=[],
                     reached_product=False, peak_energy=-200.0, n_steps=30)
    best = score_trials([t1, t2, t3])
    assert best.trial_idx == 1  # lowest peak among reached


def test_score_trials_falls_back_to_least_bad_when_all_failed():
    t1 = ScreeningTrialResult(trial_idx=0, direction=np.array([0.0, 0.0, 1.0]), frames=[], energies=[10.0, 12.0],
                     reached_product=False, peak_energy=12.0, n_steps=50)
    t2 = ScreeningTrialResult(trial_idx=1, direction=np.array([1.0, 0.0, 0.0]), frames=[], energies=[10.0, 11.0],
                     reached_product=False, peak_energy=11.0, n_steps=60)
    best = score_trials([t1, t2])
    # When all failed, prefer lowest peak
    assert best.trial_idx == 1


def test_score_trials_empty_raises():
    with pytest.raises(ValueError):
        score_trials([])


def test_reached_product_per_bond_r_form_targets():
    """Phase 3: r_form_targets list で formed bond ごとに別個に判定する。"""
    a = Atoms("OFCN", positions=[
        [0, 0, 0], [1.5, 0, 0], [-3.0, 0, 0], [-3.0, 1.05, 0],
    ])
    # Two formed bonds with different targets (O-F: 1.5, C-N: 1.05).
    assert reached_product(
        a, formed=[(0, 1), (2, 3)], broken=[],
        r_form_targets=[1.5, 1.05], r_broken_target=4.0,
    ) is True

    a.set_positions([[0, 0, 0], [3.0, 0, 0], [-3.0, 0, 0], [-3.0, 1.05, 0]])
    assert reached_product(
        a, formed=[(0, 1), (2, 3)], broken=[],
        r_form_targets=[1.5, 1.05], r_broken_target=4.0,
    ) is False


def test_reached_product_mismatched_targets_length_raises():
    a = Atoms("OF", positions=[[0, 0, 0], [1.5, 0, 0]])
    with pytest.raises(ValueError, match="r_form_targets"):
        reached_product(
            a, formed=[(0, 1)], broken=[],
            r_form_targets=[1.5, 1.05],
            r_broken_target=4.0,
        )


def _stub(idx: int, *, reached: bool, peak: float) -> ScreeningTrialResult:
    return ScreeningTrialResult(
        trial_idx=idx,
        direction=np.array([0.0, 0.0, 1.0]),
        frames=[Atoms(symbols=["H"], positions=[[0.0, 0.0, 0.0]])],
        energies=[peak],
        reached_product=reached,
        peak_energy=peak,
        n_steps=1,
    )


def test_select_best_trial_prefers_reached_product():
    trials = [
        _stub(0, reached=False, peak=-5.0),  # reached=False, lower peak
        _stub(1, reached=True, peak=-3.0),   # reached=True, higher peak
        _stub(2, reached=True, peak=-4.0),   # reached=True, lowest peak among reached
    ]
    best = select_best_trial(trials)
    assert best == 2


def test_select_best_trial_falls_back_when_none_reached():
    trials = [
        _stub(0, reached=False, peak=-5.0),
        _stub(1, reached=False, peak=-7.0),  # lowest peak overall
        _stub(2, reached=False, peak=-3.0),
    ]
    best = select_best_trial(trials)
    assert best == 1


def test_select_best_trial_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        select_best_trial([])


def test_trial_result_direction_field_replaces_rotation_deg():
    t = ScreeningTrialResult(
        trial_idx=0,
        direction=np.array([1.0, 0.0, 0.0]),
        frames=[],
        energies=[],
        reached_product=False,
        peak_energy=float("inf"),
        n_steps=0,
    )
    assert t.direction.shape == (3,)
    np.testing.assert_array_equal(t.direction, [1.0, 0.0, 0.0])
    assert not hasattr(t, "rotation_deg")


def _make_trial(idx: int, *, reached: bool, peak: float):
    from reactx.scoring import ScreeningTrialResult
    return ScreeningTrialResult(
        trial_idx=idx, direction=np.array([0.0, 0.0, 1.0]),
        frames=[], energies=[], reached_product=reached,
        peak_energy=peak, n_steps=0,
    )


def test_top_k_trials_returns_k_when_enough_reached():
    from reactx.scoring import top_k_trials
    trials = [
        _make_trial(0, reached=True, peak=2.0),
        _make_trial(1, reached=True, peak=1.0),
        _make_trial(2, reached=True, peak=3.0),
        _make_trial(3, reached=False, peak=0.5),
    ]
    out = top_k_trials(trials, 2)
    assert [t.trial_idx for t in out] == [1, 0]


def test_top_k_trials_falls_back_to_unreached_to_fill_k():
    from reactx.scoring import top_k_trials
    trials = [
        _make_trial(0, reached=True, peak=2.0),
        _make_trial(1, reached=False, peak=1.0),
        _make_trial(2, reached=False, peak=0.5),
    ]
    out = top_k_trials(trials, 3)
    # reached=True 群 (1 件) を最優先 -> reached=False 群を peak 昇順で補完
    assert [t.trial_idx for t in out] == [0, 2, 1]


def test_top_k_trials_returns_all_when_fewer_than_k():
    from reactx.scoring import top_k_trials
    trials = [
        _make_trial(0, reached=True, peak=1.0),
        _make_trial(1, reached=False, peak=2.0),
    ]
    out = top_k_trials(trials, 5)
    assert len(out) == 2


def test_top_k_trials_empty_raises():
    from reactx.scoring import top_k_trials
    with pytest.raises(ValueError):
        top_k_trials([], 1)


def test_top_k_trials_zero_k_raises():
    from reactx.scoring import top_k_trials
    with pytest.raises(ValueError, match="k must be"):
        top_k_trials([_make_trial(0, reached=True, peak=1.0)], 0)
