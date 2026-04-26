"""Reorder product atoms to match reactant atom ordering using atom mapping."""
from __future__ import annotations

from itertools import permutations

import numpy as np
from ase import Atoms
from ase.build.rotate import minimize_rotation_and_translation


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
    reactant_symbols = reactant.get_chemical_symbols()

    for r_idx in range(len(reactant)):
        sym_r = reactant_symbols[r_idx]
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

    # Rigid-body rotate+translate product onto reactant. Without this step,
    # embed3d generates R and P in independent orientations, which causes the
    # NEB interpolated path to pass atoms through each other (e.g. F/Cl
    # swapping sides across C in SN2). Alignment is driven by atom-index
    # correspondence so the RMSD-minimization is chemically meaningful.
    minimize_rotation_and_translation(reactant, aligned)

    # ETKDG seeds R and P independently, so hydrogens bonded to the same heavy
    # atom may be in different relative orientations. Reassign P's Hs to R's
    # Hs within each group by minimum sum of pairwise distances (≤3 Hs in
    # Phase 0 substrates, so brute-force enumeration is fine).
    h_perm = list(range(len(reactant)))
    for r_hs in reactant_h_groups.values():
        n = len(r_hs)
        if n <= 1:
            continue
        r_positions = reactant.positions[r_hs]
        p_positions = aligned.positions[r_hs]
        dist = np.linalg.norm(
            p_positions[:, None, :] - r_positions[None, :, :], axis=-1
        )
        identity = tuple(range(n))
        best_perm = min(
            permutations(identity),
            key=lambda perm: dist[perm, identity].sum(),
        )
        if best_perm == identity:
            continue
        for dst_i, src_i in enumerate(best_perm):
            h_perm[r_hs[dst_i]] = r_hs[src_i]
    if h_perm != list(range(len(reactant))):
        aligned = aligned[h_perm]
    return aligned
