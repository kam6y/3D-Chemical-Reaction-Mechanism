"""Convert 2D RDKit Mol to 3D ase.Atoms via RDKit ETKDG + MMFF (+ optional UMA).

Preserves the atom ordering of Chem.AddHs(mol) so that downstream consumers
(align, NEB, restraints) can correlate atom indices with the same AddHs(mol)
result.

Multi-fragment placement (e.g. SN2 substrate + nucleophile) is driven by the
caller-supplied BondChanges:
- The substrate fragment is identified as the one containing both atoms of
  the broken bond.
- The nucleophile fragment(s) are placed along the backside direction
  (-unit(anchor->leaving)) at FRAGMENT_SEPARATION distance.
- An optional rotation_perturbation rotates the backside direction within
  the cone of multi-angle trials.
"""
from __future__ import annotations

import logging

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS
from rdkit import Chem
from rdkit.Chem import AllChem

from reactx.bond_changes import BondChanges

log = logging.getLogger(__name__)

MAX_EMBED_RETRIES = 5
FRAGMENT_SEPARATION = 3.5  # Å — attack distance for multi-fragment placement


def embed_mol_to_atoms(
    mol: Chem.Mol,
    *,
    calculator: Calculator | None = None,
    seed: int = 0xC0FFEE,
    fmax: float = 0.01,
    max_opt_steps: int = 300,
    bond_changes: BondChanges | None = None,
    rotation_perturbation: np.ndarray | None = None,
) -> Atoms:
    """Embed a 2D Mol into 3D and return an ase.Atoms with implicit Hs added.

    For multi-fragment Mols, `bond_changes` MUST be provided; the substrate
    fragment is identified as the one containing both broken-bond atoms, and
    the remaining fragment(s) are placed along the backside direction.

    `rotation_perturbation` (optional 3×3 numpy array) is left-multiplied
    onto the computed backside direction before placement; identity = no
    perturbation = Phase 0 baseline.
    """
    mol_h = Chem.AddHs(mol)
    n_atoms = mol_h.GetNumAtoms()

    frag_indices = Chem.GetMolFrags(mol_h)
    frag_mols = Chem.GetMolFrags(mol_h, asMols=True, sanitizeFrags=True)

    positions = np.zeros((n_atoms, 3))
    for i, (indices, frag) in enumerate(zip(frag_indices, frag_mols, strict=True)):
        _embed_in_place(frag, seed=seed + i * MAX_EMBED_RETRIES)
        conf = frag.GetConformer()
        for j, orig_idx in enumerate(indices):
            p = conf.GetAtomPosition(j)
            positions[orig_idx] = (p.x, p.y, p.z)

    if len(frag_indices) > 1:
        if bond_changes is None:
            raise ValueError(
                "Multi-fragment Mol requires bond_changes to determine placement; "
                "got None. Compute via reactx.bond_changes.compute_bond_changes."
            )
        positions = _place_nucleophile_backside(
            mol_h, frag_indices, positions, bond_changes,
            rotation_perturbation=rotation_perturbation,
        )

    symbols = [a.GetSymbol() for a in mol_h.GetAtoms()]
    charges = [a.GetFormalCharge() for a in mol_h.GetAtoms()]
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_initial_charges(charges)

    atoms.info["charge"] = int(sum(charges))
    atoms.info["spin"] = 1  # Phase Re1: assume closed-shell singlet

    if calculator is not None:
        atoms.calc = calculator
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=max_opt_steps)

    return atoms


def _place_nucleophile_backside(
    mol_h: Chem.Mol,
    frag_indices: tuple[tuple[int, ...], ...],
    positions: np.ndarray,
    bond_changes: BondChanges,
    *,
    rotation_perturbation: np.ndarray | None = None,
) -> np.ndarray:
    """Place non-substrate fragments along the rotated backside direction.

    Substrate fragment = the one containing both atoms of the broken bond.
    Anchor (= shared atom of formed and broken) and leaving (= other end of
    broken) live in the substrate. Incoming (= other end of formed) lives in
    the nucleophile fragment, which is shifted so its centroid aligns with
    `anchor + R @ (-unit(anchor->leaving)) * FRAGMENT_SEPARATION`.
    """
    # Phase Re1 (1 formed + 1 broken) shape; Task 2 dispatcher will replace this
    # with multi-bond _directional_placement.
    a_form, b_form = bond_changes.formed[0]
    a_brk, b_brk = bond_changes.broken[0]
    common = (set((a_form, b_form)) & set((a_brk, b_brk)))
    if len(common) != 1:
        raise ValueError(
            f"formed {bond_changes.formed[0]} and broken {bond_changes.broken[0]} "
            f"must share exactly one atom; got {common}"
        )
    shared = next(iter(common))
    anchor = shared
    leaving = b_brk if a_brk == shared else a_brk
    incoming = b_form if a_form == shared else a_form

    substrate_frag = next(
        (fi for fi in frag_indices if anchor in fi and leaving in fi), None
    )
    if substrate_frag is None:
        raise ValueError(
            f"Anchor {anchor} and leaving {leaving} are not in the same fragment; "
            f"Phase Re1 expects the broken bond's two atoms to be co-fragmented."
        )

    nuc_fragments = [fi for fi in frag_indices if fi is not substrate_frag]
    if not nuc_fragments:
        raise ValueError(
            "Only one fragment found, but multi-fragment placement was invoked. "
            "(Internal inconsistency: did embed_mol_to_atoms call this for n_frags=1?)"
        )

    a_pos = positions[anchor]
    c_pos = positions[leaving]
    a_c = c_pos - a_pos
    a_c_norm = float(np.linalg.norm(a_c))
    if a_c_norm < 1e-6:
        raise RuntimeError(
            "Anchor and leaving atoms coincide after MMFF — embedding is broken."
        )
    backside = -a_c / a_c_norm
    if rotation_perturbation is not None:
        backside = rotation_perturbation @ backside

    target = a_pos + backside * FRAGMENT_SEPARATION
    for nuc in nuc_fragments:
        if incoming in nuc:
            nuc_centroid_anchor = positions[incoming]
        else:
            nuc_centroid_anchor = positions[list(nuc)].mean(axis=0)
        positions[list(nuc)] += target - nuc_centroid_anchor
    return positions


def _embed_in_place(frag: Chem.Mol, *, seed: int) -> None:
    params = AllChem.ETKDGv3()
    for attempt in range(MAX_EMBED_RETRIES):
        params.randomSeed = seed + attempt
        status = AllChem.EmbedMolecule(frag, params)
        if status == 0:
            break
    else:
        elems = ", ".join(sorted({a.GetSymbol() for a in frag.GetAtoms()}))
        raise RuntimeError(
            f"RDKit failed to embed fragment ({frag.GetNumAtoms()} atoms: {elems}) "
            f"after {MAX_EMBED_RETRIES} attempts. Check input structure and RDKit version."
        )

    if frag.GetNumHeavyAtoms() > 1:
        result = AllChem.MMFFOptimizeMolecule(frag, maxIters=500)
        if result == -1:
            raise RuntimeError(
                f"MMFF94 force field could not be constructed for fragment "
                f"({frag.GetNumAtoms()} atoms). Check element coverage."
            )
