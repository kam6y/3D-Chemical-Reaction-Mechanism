"""Simplified fragment placement for Phase 11 Pure CI-NEB pipeline."""
from __future__ import annotations

from typing import Literal

import numpy as np
from ase import Atoms
from rdkit import Chem

from reactx.bond_changes import BondChanges


def build_atoms_from_positions(mol_h: Chem.Mol, positions: np.ndarray) -> Atoms:
    """Build ASE Atoms from a hydrogen-expanded RDKit molecule and positions."""
    if positions.shape != (mol_h.GetNumAtoms(), 3):
        raise ValueError(
            f"positions shape {positions.shape} does not match "
            f"mol_h atom count {mol_h.GetNumAtoms()}"
        )
    symbols = [atom.GetSymbol() for atom in mol_h.GetAtoms()]
    charges = [atom.GetFormalCharge() for atom in mol_h.GetAtoms()]
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_initial_charges(charges)
    atoms.info["charge"] = int(sum(charges))
    atoms.info["spin"] = 1
    return atoms


def simple_placement(
    mol_h: Chem.Mol,
    base_positions: np.ndarray,
    bond_changes: BondChanges,
    *,
    initial_separation: float,
    side: Literal["reactant", "product"],
    orientation: str = "default",
    heavy_mapping: dict[int, int] | None = None,
) -> np.ndarray:
    """Return positions with separated fragments for CI-NEB endpoints.

    Reactant placement uses formed bonds as anchors. Product placement uses
    broken bonds translated from reactant atom indices into product atom
    indices via `heavy_mapping`.
    """
    if side not in ("reactant", "product"):
        raise ValueError(f"side must be 'reactant' or 'product', got {side!r}")
    if side == "product" and heavy_mapping is None:
        raise ValueError("heavy_mapping is required when side='product'")

    frag_indices = Chem.GetMolFrags(mol_h)
    positions = base_positions.copy()
    if len(frag_indices) == 1:
        return positions

    anchors_r = bond_changes.formed if side == "reactant" else bond_changes.broken
    if not anchors_r:
        label = "formed" if side == "reactant" else "broken"
        raise ValueError(
            f"simple_placement(side={side!r}): no anchor pairs available "
            f"({label} is empty)"
        )

    if side == "product":
        assert heavy_mapping is not None
        anchors = [(heavy_mapping[a], heavy_mapping[b]) for a, b in anchors_r]
    else:
        anchors = list(anchors_r)

    bridges = _bridge_pairs(anchors, frag_indices)
    if len(bridges) == 1:
        return _place_bridges_one(
            positions,
            bridges[0],
            frag_indices,
            initial_separation,
        )
    if len(bridges) == 2:
        return _place_bridges_two(
            positions,
            bridges,
            frag_indices,
            initial_separation,
            orientation,
        )
    raise NotImplementedError(
        f"simple_placement supports bridges in {{1, 2}}, got {len(bridges)}"
    )


def _bridge_pairs(
    anchors: list[tuple[int, int]],
    frag_indices: tuple[tuple[int, ...], ...],
) -> list[tuple[int, int]]:
    atom_to_frag = {a: fi for fi, atoms in enumerate(frag_indices) for a in atoms}
    bridges = []
    for i, j in anchors:
        if atom_to_frag[i] != atom_to_frag[j]:
            bridges.append((i, j))
    return bridges


def _place_bridges_one(
    positions: np.ndarray,
    anchor: tuple[int, int],
    frag_indices: tuple[tuple[int, ...], ...],
    initial_separation: float,
) -> np.ndarray:
    if len(frag_indices) != 2:
        raise NotImplementedError(
            f"bridges=1 placement supports exactly 2 fragments, got {len(frag_indices)}"
        )

    atom_to_frag = {a: fi for fi, atoms in enumerate(frag_indices) for a in atoms}
    i, j = anchor
    substrate_frag = max(
        range(len(frag_indices)),
        key=lambda idx: (len(frag_indices[idx]), -min(frag_indices[idx])),
    )
    incoming_frag = 1 - substrate_frag

    if atom_to_frag[i] == substrate_frag:
        sub_anchor, inc_anchor = i, j
    else:
        sub_anchor, inc_anchor = j, i

    sub_atoms = list(frag_indices[substrate_frag])
    positions[sub_atoms] -= positions[sub_anchor].copy()

    inc_atoms = list(frag_indices[incoming_frag])
    target = np.array([0.0, 0.0, initial_separation])
    positions[inc_atoms] += target - positions[inc_anchor]
    return positions


def _place_bridges_two(
    positions: np.ndarray,
    anchors: list[tuple[int, int]],
    frag_indices: tuple[tuple[int, ...], ...],
    initial_separation: float,
    orientation: str,
) -> np.ndarray:
    if len(frag_indices) != 2:
        raise NotImplementedError(
            f"bridges=2 placement supports exactly 2 fragments, got {len(frag_indices)}"
        )
    if orientation not in ("default", "endo", "exo"):
        raise ValueError(
            f"orientation must be 'default', 'endo', or 'exo', got {orientation!r}"
        )

    atom_to_frag = {a: fi for fi, atoms in enumerate(frag_indices) for a in atoms}
    anchors_by_frag = {0: [], 1: []}
    for a, b in anchors:
        anchors_by_frag[atom_to_frag[a]].append(a)
        anchors_by_frag[atom_to_frag[b]].append(b)

    if len(anchors_by_frag[0]) != 2 or len(anchors_by_frag[1]) != 2:
        raise ValueError("bridges=2 requires two anchor atoms on each fragment")

    for frag_idx in (0, 1):
        frag_atoms = list(frag_indices[frag_idx])
        _orient_fragment_to_x_axis(
            positions,
            frag_atoms=frag_atoms,
            anchor_atoms=anchors_by_frag[frag_idx],
            flip=(frag_idx == 1 and orientation == "exo"),
        )

    positions[list(frag_indices[1])] += np.array([0.0, 0.0, initial_separation])
    return positions


def _orient_fragment_to_x_axis(
    positions: np.ndarray,
    *,
    frag_atoms: list[int],
    anchor_atoms: list[int],
    flip: bool,
) -> None:
    centroid = positions[anchor_atoms].mean(axis=0)
    positions[frag_atoms] -= centroid

    v = positions[anchor_atoms[1]] - positions[anchor_atoms[0]]
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-6:
        return

    v_unit = v / v_norm
    target = np.array([1.0, 0.0, 0.0])
    axis = np.cross(v_unit, target)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm > 1e-6:
        angle = float(np.arccos(np.clip(v_unit @ target, -1.0, 1.0)))
        rotation = _rotation_matrix(axis / axis_norm, angle)
        positions[frag_atoms] = positions[frag_atoms] @ rotation.T
    elif float(np.dot(v_unit, target)) < 0.0:
        rotation = _rotation_matrix(np.array([0.0, 0.0, 1.0]), np.pi)
        positions[frag_atoms] = positions[frag_atoms] @ rotation.T

    if flip:
        rotation = _rotation_matrix(np.array([0.0, 0.0, 1.0]), np.pi)
        positions[frag_atoms] = positions[frag_atoms] @ rotation.T


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    return np.array(
        [
            [
                a * a + b * b - c * c - d * d,
                2 * (b * c - a * d),
                2 * (b * d + a * c),
            ],
            [
                2 * (b * c + a * d),
                a * a + c * c - b * b - d * d,
                2 * (c * d - a * b),
            ],
            [
                2 * (b * d - a * c),
                2 * (c * d + a * b),
                a * a + d * d - b * b - c * c,
            ],
        ]
    )
