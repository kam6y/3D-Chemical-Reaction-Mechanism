"""Tests for reactx.reaction_topology module."""
from reactx.reaction_topology import BondChange, BondChanges


def test_bondchange_dataclass_construction():
    bc = BondChange(a=0, b=1, order_before=1.0, order_after=0.0)
    assert bc.a == 0
    assert bc.b == 1
    assert bc.order_before == 1.0
    assert bc.order_after == 0.0


def test_bondchanges_default_lists():
    changes = BondChanges(broken=[], formed=[])
    assert changes.broken == []
    assert changes.formed == []
