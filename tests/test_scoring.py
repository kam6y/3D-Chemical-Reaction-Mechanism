"""Unit tests for reactx.scoring."""
import numpy as np
import pytest
from ase import Atoms

from reactx.scoring import TrialResult, reached_product, score_trials, select_best_trial


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
    t1 = TrialResult(trial_idx=0, direction=np.array([0.0, 0.0, 1.0]), frames=[], energies=[],
                     reached_product=True, peak_energy=-100.0, n_steps=50)
    t2 = TrialResult(trial_idx=1, direction=np.array([1.0, 0.0, 0.0]), frames=[], energies=[],
                     reached_product=True, peak_energy=-105.0, n_steps=60)
    t3 = TrialResult(trial_idx=2, direction=np.array([0.0, 1.0, 0.0]), frames=[], energies=[],
                     reached_product=False, peak_energy=-200.0, n_steps=30)
    best = score_trials([t1, t2, t3])
    assert best.trial_idx == 1  # lowest peak among reached


def test_score_trials_falls_back_to_least_bad_when_all_failed():
    t1 = TrialResult(trial_idx=0, direction=np.array([0.0, 0.0, 1.0]), frames=[], energies=[10.0, 12.0],
                     reached_product=False, peak_energy=12.0, n_steps=50)
    t2 = TrialResult(trial_idx=1, direction=np.array([1.0, 0.0, 0.0]), frames=[], energies=[10.0, 11.0],
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


def _stub(idx: int, *, reached: bool, peak: float) -> TrialResult:
    return TrialResult(
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
    t = TrialResult(
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


def test_reached_product_per_bond_r_broken_list():
    from ase import Atoms

    from reactx.scoring import reached_product

    atoms = Atoms(
        "CHCH",
        positions=[
            [0, 0, 0],     # C0
            [0, 0, 5],     # H1 (5 Å from C0)
            [10, 0, 0],    # C2
            [10, 0, 3],    # H3 (3 Å from C2)
        ],
    )
    # broken=[(0,1), (2,3)]:
    #   |C0-H1|=5, |C2-H3|=3
    # r_broken_target=[3.0, 5.0] (per-bond):
    #   bond 0: 5 >= 3-0.5=2.5 OK
    #   bond 1: 3 >= 5-0.5=4.5 NG -> False
    assert not reached_product(
        atoms,
        formed=[],
        broken=[(0, 1), (2, 3)],
        r_form_targets=[],
        r_broken_target=[3.0, 5.0],
    )
    # r_broken_target=[3.0, 2.0]:
    #   bond 0: 5 >= 2.5 OK
    #   bond 1: 3 >= 1.5 OK -> True
    assert reached_product(
        atoms,
        formed=[],
        broken=[(0, 1), (2, 3)],
        r_form_targets=[],
        r_broken_target=[3.0, 2.0],
    )


def test_reached_product_scalar_r_broken_still_works():
    """Backward path: scalar r_broken_target broadcasts to all broken bonds."""
    from ase import Atoms

    from reactx.scoring import reached_product

    atoms = Atoms(
        "CHCH",
        positions=[[0, 0, 0], [0, 0, 5], [10, 0, 0], [10, 0, 5]],
    )
    assert reached_product(
        atoms, formed=[], broken=[(0, 1), (2, 3)],
        r_form_targets=[], r_broken_target=4.0,
    )
    # Both bonds at 5 A, with r_broken=4.0-0.5=3.5 -> both pass


def test_reached_product_r_broken_list_length_mismatch_raises():
    import pytest
    from ase import Atoms

    from reactx.scoring import reached_product

    atoms = Atoms("CC", positions=[[0, 0, 0], [5, 0, 0]])
    with pytest.raises(ValueError, match="broken"):
        reached_product(
            atoms, formed=[], broken=[(0, 1)],
            r_form_targets=[], r_broken_target=[1.0, 2.0],
        )


# ===== Phase 9 additions (coexist with v8 TrialResult / reached_product) =====
from reactx.config import ConfigError
from reactx.scoring import (
    TrialResultV9,
    count_initial_latched,
    product_distance_residual,
    reached_product_v9,
    resolve_broken_thresholds,
    resolve_formed_thresholds,
    score_trials_v9,
)


def _atoms_cc_v9(d: float):
    from ase import Atoms as A
    return A("CC", positions=[[0.0, 0.0, 0.0], [d, 0.0, 0.0]])


def test_resolve_formed_thresholds_default_is_1_15_times_rsum():
    a = _atoms_cc_v9(1.5)
    thrs = resolve_formed_thresholds(a, formed=[(0, 1)], override=None)
    assert thrs == pytest.approx([1.748], abs=1e-6)


def test_resolve_broken_thresholds_required_when_nonempty():
    a = _atoms_cc_v9(1.0)
    with pytest.raises(ConfigError, match="r_broken_threshold"):
        resolve_broken_thresholds(a, broken=[(0, 1)], override=None)


def test_reached_product_v9_per_pair():
    a = _atoms_cc_v9(1.5)
    assert reached_product_v9(a, [(0, 1)], [], [1.6], [])
    assert not reached_product_v9(_atoms_cc_v9(2.0), [(0, 1)], [], [1.6], [])


def test_residual_zero_when_reached():
    assert product_distance_residual(
        _atoms_cc_v9(1.4), [(0, 1)], [], [1.6], []
    ) == 0.0


def test_residual_positive_when_formed_overshoot():
    r = product_distance_residual(_atoms_cc_v9(2.0), [(0, 1)], [], [1.6], [])
    assert r == pytest.approx(0.16, abs=1e-12)


def test_count_initial_latched_both():
    counts = count_initial_latched(
        _atoms_cc_v9(1.5), [(0, 1)], [(0, 1)], [1.6], [1.4],
    )
    assert counts == {"formed": 1, "broken": 1}


def _v9_trial(idx, *, reached, peak, residual=0.0):
    return TrialResultV9(
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


def test_score_trials_v9_prefers_reached_with_lowest_peak():
    best = score_trials_v9([
        _v9_trial(0, reached=True, peak=10.0),
        _v9_trial(1, reached=True, peak=5.0),
        _v9_trial(2, reached=False, peak=1.0),
    ])
    assert best.trial_idx == 1


def test_score_trials_v9_fallback_uses_residual_then_peak():
    best = score_trials_v9([
        _v9_trial(0, reached=False, peak=1.0, residual=10.0),
        _v9_trial(1, reached=False, peak=5.0, residual=2.0),
        _v9_trial(2, reached=False, peak=2.0, residual=2.0),
    ])
    assert best.trial_idx == 2
