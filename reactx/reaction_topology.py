"""Bond-change extraction from .rxn atom mapping.

Compares reactant and product bond graphs (under a common atom-mapping
projection) and returns the set of broken / formed bonds. Phase 1 uses these
to drive generic fragment placement (see reactx.placement).
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True)
class BondChange:
    """A bond whose order changed between reactant and product.

    Indices a and b are atom indices in the *reactant* mol_h coordinate system
    (i.e. positions in Chem.AddHs(reactant_mol).GetAtoms()).
    """

    a: int
    b: int
    order_before: float
    order_after: float


@dataclass(frozen=True)
class BondChanges:
    broken: list[BondChange]
    formed: list[BondChange]


def compute_bond_changes(
    reactant_mol_h: Chem.Mol,
    product_mol_h: Chem.Mol,
    atom_mapping: dict[int, int],
) -> BondChanges:
    raise NotImplementedError("filled in next task")
