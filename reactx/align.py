"""Reorder product atoms to match reactant atom ordering using atom mapping.

Phase 1 update: takes a *full* atom-index mapping (every reactant atom in
AddHs order to its product counterpart) so that migrating Hs (which switch
heavy atoms between R and P) are handled directly via their atom map number.

The previous Phase 0 contract (heavy_mapping + per-heavy H groups) couldn't
express migration because it assumed each H stays on the same heavy atom.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
from ase import Atoms
from ase.build.rotate import minimize_rotation_and_translation
from rdkit import Chem


def align_product_to_reactant(
    reactant: Atoms,
    product: Atoms,
    atom_index_mapping: dict[int, int],
    swappable_h_groups: list[list[int]] | None = None,
) -> Atoms:
    """Return product reordered so atom i corresponds to reactant atom i.

    atom_index_mapping: r_idx -> p_idx for *every* atom in AddHs(reactant)
        ordering. Build via ``reactx.reaction_topology.expanded_atom_mapping``.
    swappable_h_groups: optional list of reactant H index groups whose members
        can be permuted within the group to minimize H-H RMSD (typically the
        implicit Hs of a single heavy atom). Migrating Hs (those with explicit
        atom map numbers in the .rxn) MUST NOT appear here — they are pinned
        by atom_index_mapping.
    """
    if len(reactant) != len(product):
        raise ValueError(f"Atom count mismatch: reactant={len(reactant)} product={len(product)}")
    if len(atom_index_mapping) != len(reactant):
        missing = [i for i in range(len(reactant)) if i not in atom_index_mapping]
        raise ValueError(
            f"atom_index_mapping must cover all reactant atoms; missing: {missing[:8]}"
        )

    permutation = [atom_index_mapping[i] for i in range(len(reactant))]
    aligned = product[permutation]

    # Rigid-body rotate+translate product onto reactant. Without this step,
    # embed3d generates R and P in independent orientations, which causes the
    # NEB interpolated path to pass atoms through each other.
    minimize_rotation_and_translation(reactant, aligned)

    if not swappable_h_groups:
        return aligned

    # ETKDG seeds R and P independently; within a swappable H group (implicit
    # Hs of one heavy atom), find the permutation that minimizes Σ|p_i - r_i|.
    h_perm = list(range(len(reactant)))
    for group in swappable_h_groups:
        n = len(group)
        if n <= 1:
            continue
        r_positions = reactant.positions[group]
        p_positions = aligned.positions[group]
        dist = np.linalg.norm(p_positions[:, None, :] - r_positions[None, :, :], axis=-1)
        identity = tuple(range(n))
        best_perm = min(
            permutations(identity),
            key=lambda perm: dist[perm, identity].sum(),
        )
        if best_perm == identity:
            continue
        for dst_i, src_i in enumerate(best_perm):
            h_perm[group[dst_i]] = group[src_i]
    if h_perm != list(range(len(reactant))):
        aligned = aligned[h_perm]
    return aligned


def build_swappable_h_groups(
    r_mol_h: Chem.Mol,
) -> list[list[int]]:
    """List groups of reactant H indices that may be permuted in alignment.

    A group is the H children of one heavy atom EXCLUDING any H that has an
    atom map number (those are pinned by the .rxn mapping — moving them would
    misalign migrating-H trajectories).
    """
    groups: list[list[int]] = []
    for atom in r_mol_h.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        hs = [
            n.GetIdx()
            for n in atom.GetNeighbors()
            if n.GetAtomicNum() == 1 and n.GetAtomMapNum() == 0
        ]
        if len(hs) >= 2:
            groups.append(hs)
    return groups
