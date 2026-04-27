"""Compute formed/broken bonds for a single elementary step (1 formed + 1 broken).

Phase Re1 supports only this minimal topology. Generic multi-bond reactions
(E2, dissociation, etc.) raise NotImplementedError and are deferred to Phase 2.
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True)
class SimpleBondChanges:
    """Single elementary step: exactly one bond formed and one bond broken.

    Atom indices are in the **reactant_mol_h** coordinate system
    (Chem.AddHs(reactant_mol).GetAtoms() ordering).
    """
    formed: tuple[int, int]
    broken: tuple[int, int]

    @property
    def shared_atom(self) -> int:
        """The atom common to both formed and broken bonds (= 'central anchor').

        For SN2 this is the substrate C; for proton transfer this is the H.
        """
        f = set(self.formed)
        b = set(self.broken)
        common = f & b
        if len(common) != 1:
            raise ValueError(
                f"formed {self.formed} and broken {self.broken} must share exactly "
                f"one atom; got {common}"
            )
        return next(iter(common))


def compute_simple_bond_changes(
    reactant_mol_h: Chem.Mol,
    product_mol_h: Chem.Mol,
    heavy_mapping: dict[int, int],
) -> SimpleBondChanges:
    """Diff bonds between reactant and product mol_h, return formed + broken.

    Raises NotImplementedError if not exactly 1 formed + 1 broken bond.
    Atom indices in the result use reactant_mol_h's ordering.
    """
    if reactant_mol_h.GetNumAtoms() != product_mol_h.GetNumAtoms():
        raise ValueError(
            f"reactant and product mol_h must have same atom count: "
            f"{reactant_mol_h.GetNumAtoms()} vs {product_mol_h.GetNumAtoms()}"
        )

    full_mapping = _build_full_atom_mapping(
        reactant_mol_h, product_mol_h, heavy_mapping
    )
    inv_mapping = {p: r for r, p in full_mapping.items()}

    r_bonds = _bond_set_in_self_idx(reactant_mol_h)
    p_bonds_in_r_space = {
        _ordered(inv_mapping[a], inv_mapping[b])
        for a, b in _bond_set_in_self_idx(product_mol_h)
    }

    formed = sorted(p_bonds_in_r_space - r_bonds)
    broken = sorted(r_bonds - p_bonds_in_r_space)

    if len(formed) != 1 or len(broken) != 1:
        raise NotImplementedError(
            f"Phase Re1 supports exactly 1 formed + 1 broken bond. "
            f"Got formed={formed} ({len(formed)}), broken={broken} ({len(broken)}). "
            f"Generic bond-change support is Phase 2."
        )
    return SimpleBondChanges(formed=formed[0], broken=broken[0])


def _bond_set_in_self_idx(mol: Chem.Mol) -> set[tuple[int, int]]:
    return {
        _ordered(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
    }


def _ordered(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _build_full_atom_mapping(
    reactant_mol_h: Chem.Mol,
    product_mol_h: Chem.Mol,
    heavy_mapping: dict[int, int],
) -> dict[int, int]:
    """Extend heavy_mapping with implicit-H pairings.

    heavy_mapping covers all atoms with explicit atom map numbers (heavy + any
    explicit-mapped H). Remaining unmapped Hs (= implicit Hs added by AddHs)
    are paired by their bonded heavy atom group: reactant Hs of heavy_r <->
    product Hs of heavy_p where heavy_r -> heavy_p in heavy_mapping.
    """
    full: dict[int, int] = dict(heavy_mapping)
    used_p: set[int] = set(full.values())

    for r_heavy, p_heavy in heavy_mapping.items():
        r_atom = reactant_mol_h.GetAtomWithIdx(r_heavy)
        p_atom = product_mol_h.GetAtomWithIdx(p_heavy)
        if r_atom.GetSymbol() == "H" or p_atom.GetSymbol() == "H":
            continue  # explicit-mapped H itself, not a heavy atom group

        r_implicit_hs = [
            n.GetIdx() for n in r_atom.GetNeighbors()
            if n.GetSymbol() == "H" and n.GetIdx() not in full
        ]
        p_implicit_hs = [
            n.GetIdx() for n in p_atom.GetNeighbors()
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
