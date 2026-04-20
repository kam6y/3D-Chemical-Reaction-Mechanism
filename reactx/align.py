"""Reorder product atoms to match reactant atom ordering using atom mapping."""
from __future__ import annotations

from ase import Atoms


def align_product_to_reactant(
    reactant: Atoms,
    product: Atoms,
    heavy_mapping: dict[int, int],
    reactant_h_groups: dict[int, list[int]],
    product_h_groups: dict[int, list[int]],
) -> Atoms:
    """Return product Atoms reordered so that atom i corresponds to reactant atom i.

    heavy_mapping maps reactant heavy-atom index -> product heavy-atom index.
    *_h_groups maps heavy-atom index -> list of bonded hydrogen indices on that side.
    """
    if len(reactant) != len(product):
        raise ValueError(
            f"Atom count mismatch: reactant={len(reactant)} product={len(product)}"
        )

    permutation: list[int] = [-1] * len(reactant)

    for r_idx in range(len(reactant)):
        sym_r = reactant.get_chemical_symbols()[r_idx]
        if sym_r == "H":
            continue
        if r_idx not in heavy_mapping:
            raise ValueError(f"Reactant heavy atom {r_idx} ({sym_r}) not in mapping")
        p_idx = heavy_mapping[r_idx]
        permutation[r_idx] = p_idx

        r_hs = reactant_h_groups.get(r_idx, [])
        p_hs = product_h_groups.get(p_idx, [])
        if len(r_hs) != len(p_hs):
            raise ValueError(
                f"hydrogen count mismatch for heavy atom {r_idx}->{p_idx}: "
                f"{len(r_hs)} vs {len(p_hs)}"
            )
        for r_h, p_h in zip(r_hs, p_hs):
            permutation[r_h] = p_h

    if any(x < 0 for x in permutation):
        missing = [i for i, x in enumerate(permutation) if x < 0]
        raise ValueError(f"Unmapped reactant atoms: {missing}")

    aligned = product[permutation]
    return aligned
