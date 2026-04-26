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
    """Compare bond graphs and return broken/formed bonds (reactant-indexed).

    atom_mapping is the dict returned by parse_rxn — it covers every atom that
    has a non-zero AtomMapNum in the .rxn (typically all heavy atoms plus any
    explicitly tracked migrating Hs). Implicit/unmapped Hs added by Chem.AddHs
    are paired positionally per heavy-atom group.
    """
    expanded = _build_expanded_mapping(reactant_mol_h, product_mol_h, atom_mapping)
    inv_expanded = {p: r for r, p in expanded.items()}

    r_bonds = _bond_orders(reactant_mol_h)
    p_bonds_in_r: dict[tuple[int, int], float] = {}
    for (pa, pb), order in _bond_orders(product_mol_h).items():
        if pa not in inv_expanded or pb not in inv_expanded:
            raise ValueError(
                f"Product bond {pa}-{pb} cannot be projected to reactant indexing "
                "(atom mapping is incomplete; check explicit-H map numbers in .rxn)"
            )
        ra, rb = inv_expanded[pa], inv_expanded[pb]
        p_bonds_in_r[(min(ra, rb), max(ra, rb))] = order

    broken: list[BondChange] = []
    formed: list[BondChange] = []
    for key in sorted(set(r_bonds) | set(p_bonds_in_r)):
        before = r_bonds.get(key, 0.0)
        after = p_bonds_in_r.get(key, 0.0)
        if before == after:
            continue
        bc = BondChange(a=key[0], b=key[1], order_before=before, order_after=after)
        if after < before:
            broken.append(bc)
        else:
            formed.append(bc)
    return BondChanges(broken=broken, formed=formed)


def _bond_orders(mol_h: Chem.Mol) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for bond in mol_h.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        order = bond.GetBondTypeAsDouble()
        if order == 1.5:
            raise NotImplementedError(
                "Aromatic bonds are out of scope in Phase 1; kekulize the input"
            )
        out[(min(a, b), max(a, b))] = order
    return out


def expanded_atom_mapping(
    r_mol_h: Chem.Mol,
    p_mol_h: Chem.Mol,
    atom_mapping: dict[int, int],
) -> dict[int, int]:
    """Public: full reactant->product index map including implicit Hs.

    Wrapper around the same internal pairing used by compute_bond_changes.
    Useful for callers that need to project reactant-indexed BondChanges into
    product indexing (see reactx.placement.place_fragments_generic).
    """
    return _build_expanded_mapping(r_mol_h, p_mol_h, atom_mapping)


def _build_expanded_mapping(
    r_mol_h: Chem.Mol,
    p_mol_h: Chem.Mol,
    atom_mapping: dict[int, int],
) -> dict[int, int]:
    """Pair every reactant atom to its product counterpart.

    Three-stage pairing:
      1. Atoms with explicit AtomMapNum from the .rxn (heavy atoms and any
         explicitly-tracked migrating Hs). Already supplied via atom_mapping.
      2. Hs whose AtomMapNum is set on both sides but which were not in the
         heavy_mapping subset. Re-derived directly from the mol_h objects.
      3. Implicit Hs added by AddHs (no map number): paired positionally per
         heavy atom by walking the H neighbors of each mapped heavy in
         encounter order.
    """
    expanded: dict[int, int] = dict(atom_mapping)

    r_h_by_map = {
        a.GetAtomMapNum(): a.GetIdx()
        for a in r_mol_h.GetAtoms()
        if a.GetAtomicNum() == 1 and a.GetAtomMapNum() != 0
    }
    p_h_by_map = {
        a.GetAtomMapNum(): a.GetIdx()
        for a in p_mol_h.GetAtoms()
        if a.GetAtomicNum() == 1 and a.GetAtomMapNum() != 0
    }
    for mn, r_idx in r_h_by_map.items():
        if mn in p_h_by_map:
            expanded[r_idx] = p_h_by_map[mn]

    paired_r = set(expanded.keys())
    paired_p = set(expanded.values())
    # iterate only original heavy mappings; H entries in atom_mapping have no
    # H neighbors of their own so they would just no-op
    for r_idx, p_idx in atom_mapping.items():
        if r_mol_h.GetAtomWithIdx(r_idx).GetAtomicNum() == 1:
            continue
        r_atom = r_mol_h.GetAtomWithIdx(r_idx)
        p_atom = p_mol_h.GetAtomWithIdx(p_idx)
        r_unpaired = [
            n.GetIdx()
            for n in r_atom.GetNeighbors()
            if n.GetAtomicNum() == 1 and n.GetIdx() not in paired_r
        ]
        p_unpaired = [
            n.GetIdx()
            for n in p_atom.GetNeighbors()
            if n.GetAtomicNum() == 1 and n.GetIdx() not in paired_p
        ]
        for rh, ph in zip(r_unpaired, p_unpaired, strict=False):
            expanded[rh] = ph
            paired_r.add(rh)
            paired_p.add(ph)
    return expanded
