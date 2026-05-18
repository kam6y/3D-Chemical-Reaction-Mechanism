"""BondChanges container + atom-map -> 0-based index converter."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BondChanges:
    """Multi-bond elementary step: tuples of formed and broken bonds.

    Atom indices are 0-based heavy-atom indices in the reactant Mol's ordering
    (which matches the post-AddHs Mol because Chem.AddHs preserves indices).
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

        formed_canonical = {
            (a, b) if a <= b else (b, a) for a, b in self.formed
        }
        broken_canonical = {
            (a, b) if a <= b else (b, a) for a, b in self.broken
        }
        overlap = formed_canonical & broken_canonical
        if overlap:
            raise ValueError(
                f"formed and broken collide on bond(s) {sorted(overlap)}"
            )

    @classmethod
    def from_atom_map_pairs(
        cls,
        formed_map: Sequence[tuple[int, int]],
        broken_map: Sequence[tuple[int, int]],
        atom_map_to_idx: dict[int, int],
    ) -> BondChanges:
        """Build a BondChanges from TOML atom-map pairs and a reactant lookup.

        Raises:
            KeyError: when an atom-map number is not present in atom_map_to_idx.
        """
        return cls(
            formed=tuple(_translate(p, atom_map_to_idx, "formed") for p in formed_map),
            broken=tuple(_translate(p, atom_map_to_idx, "broken") for p in broken_map),
        )

    @classmethod
    def from_reaction_diff(
        cls,
        r_mol,
        p_mol,
    ) -> BondChanges:
        """Build BondChanges by diffing R/P bond sets in atom-map space.

        Returns BondChanges with 0-based atom indices in the reactant Mol's
        ordering. Bond order changes are ignored; only connectivity diffs are
        considered.
        """
        r_map_to_idx: dict[int, int] = {}
        for atom in r_mol.GetAtoms():
            mapnum = atom.GetAtomMapNum()
            if mapnum == 0:
                raise ValueError(f"reactant atom {atom.GetIdx()} has no atom map number")
            r_map_to_idx[mapnum] = atom.GetIdx()

        p_map_to_idx: dict[int, int] = {}
        for atom in p_mol.GetAtoms():
            mapnum = atom.GetAtomMapNum()
            if mapnum == 0:
                raise ValueError(f"product atom {atom.GetIdx()} has no atom map number")
            p_map_to_idx[mapnum] = atom.GetIdx()

        if set(r_map_to_idx.keys()) != set(p_map_to_idx.keys()):
            raise ValueError(
                f"atom map mismatch: reactant {sorted(r_map_to_idx)} "
                f"vs product {sorted(p_map_to_idx)}"
            )

        def _bond_set_in_mapnums(mol, map_to_idx: dict[int, int]) -> set[tuple[int, int]]:
            idx_to_map = {idx: mn for mn, idx in map_to_idx.items()}
            out: set[tuple[int, int]] = set()
            for bond in mol.GetBonds():
                a = idx_to_map[bond.GetBeginAtomIdx()]
                b = idx_to_map[bond.GetEndAtomIdx()]
                out.add((a, b) if a <= b else (b, a))
            return out

        r_bonds = _bond_set_in_mapnums(r_mol, r_map_to_idx)
        p_bonds = _bond_set_in_mapnums(p_mol, p_map_to_idx)

        formed_mapnum_pairs = sorted(p_bonds - r_bonds)
        broken_mapnum_pairs = sorted(r_bonds - p_bonds)

        formed = tuple(
            (r_map_to_idx[a], r_map_to_idx[b]) for a, b in formed_mapnum_pairs
        )
        broken = tuple(
            (r_map_to_idx[a], r_map_to_idx[b]) for a, b in broken_mapnum_pairs
        )

        if not formed and not broken:
            raise ValueError(
                "from_reaction_diff: no bond changes detected between R and P "
                "(both have identical connectivity in atom-map space)"
            )
        return cls(formed=formed, broken=broken)


def _translate(
    pair: tuple[int, int],
    atom_map_to_idx: dict[int, int],
    label: str,
) -> tuple[int, int]:
    a, b = pair
    if a not in atom_map_to_idx:
        raise KeyError(f"{label}: unknown atom-map number {a}")
    if b not in atom_map_to_idx:
        raise KeyError(f"{label}: unknown atom-map number {b}")
    return (atom_map_to_idx[a], atom_map_to_idx[b])
