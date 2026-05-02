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
        positions = _place_fragments(
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


def _place_fragments(
    mol_h: Chem.Mol,
    frag_indices: tuple[tuple[int, ...], ...],
    positions: np.ndarray,
    bond_changes: BondChanges,
    *,
    rotation_perturbation: np.ndarray | None,
) -> np.ndarray:
    """Dispatch to the appropriate placement strategy.

    Tier 1 (directional, Phase 3): broken bond の方向情報がある反応 (E2 / SN2 / PT)。
    Tier 2 (centroid, future):     broken=0 / multi-substrate metathesis。Phase 4+。
    """
    substrate = _find_substrate_fragment(frag_indices, bond_changes.broken)
    if substrate is not None and bond_changes.broken:
        return _directional_placement(
            frag_indices, positions, bond_changes, substrate,
            rotation_perturbation=rotation_perturbation,
        )
    raise NotImplementedError(
        "centroid-based placement (broken=0 / multi-substrate metathesis) is "
        f"Phase 4+. Got formed={bond_changes.formed}, broken={bond_changes.broken}, "
        f"frags={len(frag_indices)}."
    )


def _find_substrate_fragment(
    frag_indices: tuple[tuple[int, ...], ...],
    broken: tuple[tuple[int, int], ...],
) -> tuple[int, ...] | None:
    """Return the unique fragment containing both atoms of every broken bond.

    None when broken is empty, or when broken bonds span multiple fragments.
    """
    if not broken:
        return None
    candidates: list[tuple[int, ...]] = []
    for frag in frag_indices:
        frag_set = set(frag)
        if all(a in frag_set and b in frag_set for a, b in broken):
            candidates.append(frag)
    if len(candidates) != 1:
        return None
    return candidates[0]


def _directional_placement(
    frag_indices: tuple[tuple[int, ...], ...],
    positions: np.ndarray,
    bond_changes: BondChanges,
    substrate: tuple[int, ...],
    *,
    rotation_perturbation: np.ndarray | None,
) -> np.ndarray:
    """Tier 1: anchor + leaving direction for each non-substrate fragment."""
    substrate_set = set(substrate)
    non_substrate = [f for f in frag_indices if f is not substrate]
    if len(non_substrate) >= 2:
        log.warning(
            "termolecular placement (%d non-substrate fragments); geometric "
            "quality may be reduced", len(non_substrate),
        )

    bridging_by_anchor: dict[int, list[tuple[int, ...]]] = {}
    for f_idx, f in enumerate(non_substrate):
        f_set = set(f)
        bridging = [
            (a, b) for a, b in bond_changes.formed
            if (a in substrate_set and b in f_set) or (b in substrate_set and a in f_set)
        ]
        if not bridging:
            raise ValueError(
                f"fragment {f_idx} has no formed bond bridging to substrate; "
                f"check input atom mapping (formed={bond_changes.formed}, "
                f"substrate atoms={sorted(substrate_set)})"
            )
        for bond in bridging:
            anchor = bond[0] if bond[0] in substrate_set else bond[1]
            bridging_by_anchor.setdefault(anchor, []).append((f, bond))

    for anchor, hits in bridging_by_anchor.items():
        unique_frags = {id(f) for f, _ in hits}
        if len(unique_frags) > 1:
            raise NotImplementedError(
                f"multi-base attack on single anchor {anchor} not supported "
                f"(Phase 3 supports at most one fragment per anchor)"
            )

    for fragment in non_substrate:
        f_set = set(fragment)
        anchor: int | None = None
        bridging_bond: tuple[int, int] | None = None
        for a, hits in bridging_by_anchor.items():
            for f, bond in hits:
                if f is fragment:
                    anchor, bridging_bond = a, bond
                    break
            if anchor is not None:
                break
        if anchor is None or bridging_bond is None:
            raise RuntimeError(
                f"fragment {fragment} has no entry in bridging_by_anchor — "
                "this is a logic error in _directional_placement"
            )

        relevant_broken = sorted(
            [(a, b) for a, b in bond_changes.broken if anchor in (a, b)],
            key=lambda ab: ab[1] if ab[0] == anchor else ab[0],
        )
        if relevant_broken:
            chosen = relevant_broken[0]
            leaving = chosen[1] if chosen[0] == anchor else chosen[0]
        else:
            substrate_centroid = positions[list(substrate)].mean(axis=0)
            distances = [
                float(np.linalg.norm(positions[i] - substrate_centroid))
                for i in substrate if i != anchor
            ]
            leaving_candidates = [i for i in substrate if i != anchor]
            leaving = leaving_candidates[int(np.argmax(distances))]

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

        incoming = bridging_bond[1] if bridging_bond[0] == anchor else bridging_bond[0]
        if incoming in f_set:
            nuc_centroid_anchor = positions[incoming]
        else:
            nuc_centroid_anchor = positions[list(fragment)].mean(axis=0)
        positions[list(fragment)] += target - nuc_centroid_anchor

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
