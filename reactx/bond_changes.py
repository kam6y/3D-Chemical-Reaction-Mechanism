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
