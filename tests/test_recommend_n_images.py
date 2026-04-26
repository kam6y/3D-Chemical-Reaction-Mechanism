"""Tests for recommend_n_images pure function."""

from reactx.cli import recommend_n_images
from reactx.reaction_topology import BondChange, BondChanges


def _bc(n: int) -> list[BondChange]:
    return [BondChange(a=0, b=1, order_before=1.0, order_after=0.0) for _ in range(n)]


def test_recommend_n_images_single_bond_change():
    changes = BondChanges(broken=_bc(1), formed=_bc(1))
    assert recommend_n_images(changes) == 13


def test_recommend_n_images_dissociation():
    changes = BondChanges(broken=_bc(1), formed=_bc(0))
    assert recommend_n_images(changes) == 11


def test_recommend_n_images_e2():
    changes = BondChanges(broken=_bc(2), formed=_bc(2))
    assert recommend_n_images(changes) == 17


def test_recommend_n_images_floor():
    assert recommend_n_images(BondChanges(broken=[], formed=[])) == 11
