"""Unit tests for BondChanges + from_atom_map_pairs (Tier B)."""
from pathlib import Path

import pytest

from reactx.bond_changes import BondChanges
from reactx.rxn_parser import parse_rxn

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"


def test_self_loop_rejected():
    with pytest.raises(ValueError, match="self-loop"):
        BondChanges(formed=((0, 0),), broken=((1, 2),))


def test_duplicate_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        BondChanges(formed=((0, 1), (1, 0)), broken=())


def test_empty_total_rejected():
    with pytest.raises(ValueError, match="at least one"):
        BondChanges(formed=(), broken=())


def test_formed_broken_overlap_rejected():
    with pytest.raises(ValueError, match="formed and broken collide"):
        BondChanges(formed=((0, 1),), broken=((1, 0),))


def test_multi_bond_topologies_accepted():
    bc = BondChanges(formed=((4, 5),), broken=((0, 2), (1, 4)))
    assert len(bc.formed) == 1
    assert len(bc.broken) == 2
    bc = BondChanges(formed=(), broken=((0, 1),))
    assert len(bc.formed) == 0
    assert len(bc.broken) == 1


def test_from_atom_map_pairs_translates_indices():
    # atom-map 1->idx 0, 5->idx 4
    m2i = {1: 0, 5: 4}
    bc = BondChanges.from_atom_map_pairs(
        formed_map=[(1, 5)],
        broken_map=[],
        atom_map_to_idx=m2i,
    )
    assert bc.formed == ((0, 4),)
    assert bc.broken == ()


def test_from_atom_map_pairs_unknown_map_number_raises_key_error():
    m2i = {1: 0, 2: 1}
    with pytest.raises(KeyError, match="unknown atom-map number 99"):
        BondChanges.from_atom_map_pairs(
            formed_map=[(1, 99)],
            broken_map=[],
            atom_map_to_idx=m2i,
        )


def test_from_atom_map_pairs_overlap_via_translation_raises():
    """formed=(1,5) broken=(5,1) collapse to the same canonical pair after translation."""
    m2i = {1: 0, 5: 4}
    with pytest.raises(ValueError, match="formed and broken collide"):
        BondChanges.from_atom_map_pairs(
            formed_map=[(1, 5)],
            broken_map=[(5, 1)],
            atom_map_to_idx=m2i,
        )


@pytest.mark.parametrize(
    "rxn_name,expected_formed_count,expected_broken_count",
    [
        ("sn2", 1, 1),
        ("proton_transfer", 1, 1),
        ("menshutkin", 1, 1),
        ("e2", 1, 2),
        ("sn1_dissoc", 0, 1),
        ("sn1_recomb", 1, 0),
        ("diels_alder_simple", 2, 0),
        ("diels_alder_endo", 2, 0),
    ],
)
def test_from_reaction_diff_counts(
    rxn_name, expected_formed_count, expected_broken_count
):
    r_mol, p_mol, _ = parse_rxn(EXAMPLES / f"{rxn_name}.rxn")
    bc = BondChanges.from_reaction_diff(r_mol, p_mol)
    assert len(bc.formed) == expected_formed_count, (
        f"{rxn_name}: expected {expected_formed_count} formed, got {len(bc.formed)}"
    )
    assert len(bc.broken) == expected_broken_count, (
        f"{rxn_name}: expected {expected_broken_count} broken, got {len(bc.broken)}"
    )


def test_from_reaction_diff_sn2_matches_atom_indices():
    r_mol, p_mol, _ = parse_rxn(EXAMPLES / "sn2.rxn")
    bc = BondChanges.from_reaction_diff(r_mol, p_mol)
    # SN2: C(1) + Cl(2) + OH-(3,4) -> C(1)-OH(3,4) + Cl-(2)
    # atom map 1 -> C (reactant idx 0), 2 -> Cl (idx 1), 3 -> O (idx 2)
    # formed: C-O (0,2) or (2,0)
    # broken: C-Cl (0,1) or (1,0)
    formed_canon = {tuple(sorted(p)) for p in bc.formed}
    broken_canon = {tuple(sorted(p)) for p in bc.broken}
    assert formed_canon == {(0, 2)}
    assert broken_canon == {(0, 1)}
