"""Unit tests for BondChanges + from_atom_map_pairs (Tier B)."""
import pytest

from reactx.bond_changes import BondChanges


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
