"""Compute formed/broken bonds for elementary steps (Phase 3: multi-bond).

σ-only connectivity diff. Bond order changes (single↔double) are NOT
detected; π formation is left to the QM calculator. See the Phase 3 spec
section "σ-only connectivity diff".
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True)
class BondChanges:
    """Multi-bond elementary step: lists of formed and broken bonds.

    Atom indices are in the reactant_mol_h (Chem.AddHs(reactant_mol)) ordering.
    Tuples (not lists) for frozen-dataclass hashability.
    """

    formed: tuple[tuple[int, int], ...]
    broken: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        for label, bonds in (("formed", self.formed), ("broken", self.broken)):
            seen: set[tuple[int, int]] = set()
            for a, b in bonds:
                if a == b:
                    raise ValueError(f"{label} bond {(a, b)} is a self-loop")
                key = (a, b) if a <= b else (b, a)
                if key in seen:
                    raise ValueError(
                        f"{label} contains duplicate bond {(a, b)} "
                        f"(canonical form {key} already seen)"
                    )
                seen.add(key)
        if len(self.formed) + len(self.broken) == 0:
            raise ValueError("BondChanges must have at least one formed or broken bond")


def compute_bond_changes(
    reactant_mol_h: Chem.Mol,
    product_mol_h: Chem.Mol,
    heavy_mapping: dict[int, int],
) -> BondChanges:
    """Diff bonds between reactant and product mol_h. σ-only connectivity.

    Atom indices in the result use reactant_mol_h's ordering.
    """
    if reactant_mol_h.GetNumAtoms() != product_mol_h.GetNumAtoms():
        raise ValueError(
            f"reactant and product mol_h must have same atom count: "
            f"{reactant_mol_h.GetNumAtoms()} vs {product_mol_h.GetNumAtoms()}"
        )

    full_mapping = _build_full_atom_mapping(reactant_mol_h, product_mol_h, heavy_mapping)
    inv_mapping = {p: r for r, p in full_mapping.items()}

    r_bonds = _bond_set_in_self_idx(reactant_mol_h)
    p_bonds_in_r_space = {
        _ordered(inv_mapping[a], inv_mapping[b]) for a, b in _bond_set_in_self_idx(product_mol_h)
    }

    formed = tuple(sorted(p_bonds_in_r_space - r_bonds))
    broken = tuple(sorted(r_bonds - p_bonds_in_r_space))
    return BondChanges(formed=formed, broken=broken)


def _bond_set_in_self_idx(mol: Chem.Mol) -> set[tuple[int, int]]:
    return {_ordered(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()}


def _ordered(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _build_full_atom_mapping(
    reactant_mol_h: Chem.Mol,
    product_mol_h: Chem.Mol,
    heavy_mapping: dict[int, int],
) -> dict[int, int]:
    """Extend heavy_mapping with implicit-H pairings.

    heavy_mapping covers all atoms with explicit atom map numbers (heavy + any
    explicit-mapped H). Remaining unmapped Hs are paired by their bonded heavy
    atom group: reactant Hs of heavy_r <-> product Hs of heavy_p where
    heavy_r -> heavy_p in heavy_mapping.

    Index-space assumption: heavy_mapping uses indices that are valid in BOTH
    the pre-AddHs Mol and the post-AddHs Mol. This holds because Chem.AddHs
    appends implicit Hs at indices >= original atom count, preserving every
    pre-existing atom's index. Callers must pass a heavy_mapping derived from
    parse_rxn (= pre-AddHs atom-map-number lookup) together with the AddHs'd
    Mols.
    """
    full: dict[int, int] = dict(heavy_mapping)
    used_p: set[int] = set(full.values())

    for r_heavy, p_heavy in heavy_mapping.items():
        r_atom = reactant_mol_h.GetAtomWithIdx(r_heavy)
        p_atom = product_mol_h.GetAtomWithIdx(p_heavy)
        if r_atom.GetSymbol() == "H" or p_atom.GetSymbol() == "H":
            continue

        r_implicit_hs = [
            n.GetIdx()
            for n in r_atom.GetNeighbors()
            if n.GetSymbol() == "H" and n.GetIdx() not in full
        ]
        p_implicit_hs = [
            n.GetIdx()
            for n in p_atom.GetNeighbors()
            if n.GetSymbol() == "H" and n.GetIdx() not in used_p
        ]
        if len(r_implicit_hs) != len(p_implicit_hs):
            raise ValueError(
                f"implicit H count mismatch for heavy atom {r_heavy}->{p_heavy}: "
                f"reactant has {len(r_implicit_hs)}, product has {len(p_implicit_hs)}"
            )
        for r_h, p_h in zip(sorted(r_implicit_hs), sorted(p_implicit_hs), strict=True):
            full[r_h] = p_h
            used_p.add(p_h)

    return full
