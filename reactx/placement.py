"""Simplified fragment placement for Phase 11 Pure CI-NEB pipeline."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from ase import Atoms
from rdkit import Chem

from reactx.bond_changes import BondChanges
from reactx.vdw_radii import vdw_radius

log = logging.getLogger(__name__)


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


def sample_sphere_directions(n: int, seed: int = 0) -> list[np.ndarray]:
    """Fibonacci sphere over full 4pi sr; return n unit vectors."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    directions = [np.array([0.0, 0.0, 1.0])]
    if n == 1:
        return directions

    rng = np.random.default_rng(seed)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(n - 1):
        t = (i + 0.5) / (n - 1)
        z = 1.0 - 2.0 * t
        r_xy = float(np.sqrt(max(0.0, 1.0 - z * z)))
        theta = phase + i * golden_angle
        v = np.array([
            r_xy * float(np.cos(theta)),
            r_xy * float(np.sin(theta)),
            z,
        ])
        directions.append(v / float(np.linalg.norm(v)))
    return directions


def compute_d_min(
    d: np.ndarray,
    anchor_pos: np.ndarray,
    substrate_positions: np.ndarray,
    substrate_vdw: np.ndarray,
    incoming_positions: np.ndarray,
    incoming_vdw: np.ndarray,
    incoming_anchor_pos: np.ndarray,
    *,
    gap: float = 0.5,
) -> float:
    """Shortest sterically safe distance between fragments along direction d."""
    sub_proj = (substrate_positions - anchor_pos) @ d + substrate_vdw
    inc_proj = (incoming_positions - incoming_anchor_pos) @ (-d) + incoming_vdw
    max_substrate_fwd = float(sub_proj.max()) if sub_proj.size else 0.0
    max_incoming_back = float(inc_proj.max()) if inc_proj.size else 0.0
    return max_substrate_fwd + max_incoming_back + float(gap)


def evaluate_direction(
    d: np.ndarray,
    anchor_pos: np.ndarray,
    substrate_positions: np.ndarray,
    substrate_vdw: np.ndarray,
    incoming_positions: np.ndarray,
    incoming_vdw: np.ndarray,
    incoming_anchor_pos: np.ndarray,
    *,
    gap: float = 0.5,
    d_min_ceiling: float = 8.0,
) -> tuple[bool, float, str | None]:
    """Return whether a sampled direction is blocked by sterics."""
    rel = substrate_positions - anchor_pos
    r = np.linalg.norm(rel, axis=1)
    safe = r > 1e-9
    if np.any(safe):
        cos_to_d = (rel[safe] @ d) / r[safe]
        vdw_safe = substrate_vdw[safe]
        cos_thresh = r[safe] / np.sqrt(r[safe] ** 2 + vdw_safe**2)
        hits = np.where(cos_to_d > cos_thresh)[0]
        if hits.size:
            orig_indices = np.where(safe)[0]
            blocked_atom = int(orig_indices[hits[0]])
            return True, float("nan"), f"angle_shadow:atom_index={blocked_atom}"

    d_min = compute_d_min(
        d,
        anchor_pos,
        substrate_positions,
        substrate_vdw,
        incoming_positions,
        incoming_vdw,
        incoming_anchor_pos,
        gap=gap,
    )
    if d_min > d_min_ceiling:
        return True, d_min, f"d_min_ceiling:value={d_min:.3f}"
    return False, d_min, None


def _rotate_atoms(
    positions: np.ndarray,
    *,
    indices: tuple[int, ...],
    axis: np.ndarray,
    center: np.ndarray,
    angle: float,
) -> np.ndarray:
    new_pos = positions.copy()
    if angle == 0.0:
        return new_pos
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        raise ValueError("rotation axis has zero length")
    k = axis / n
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    for i in indices:
        v = positions[i] - center
        v_rot = (
            v * cos_a
            + np.cross(k, v) * sin_a
            + k * float(np.dot(k, v)) * (1.0 - cos_a)
        )
        new_pos[i] = center + v_rot
    return new_pos


def _permutation_aware_rmsd(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    syms: list[str],
) -> float:
    if len(pos_a) != len(pos_b) or len(pos_a) != len(syms):
        raise ValueError("shape mismatch")
    if len(pos_a) == 0:
        return 0.0
    used_b: set[int] = set()
    sq_sum = 0.0
    for i, sym_i in enumerate(syms):
        best_j = -1
        best_d2 = float("inf")
        for j, sym_j in enumerate(syms):
            if j in used_b or sym_j != sym_i:
                continue
            d2 = float(np.sum((pos_a[i] - pos_b[j]) ** 2))
            if d2 < best_d2:
                best_d2 = d2
                best_j = j
        if best_j >= 0:
            used_b.add(best_j)
        else:
            best_d2 = float(np.sum((pos_a[i] - pos_b[i]) ** 2))
        sq_sum += best_d2
    return float(np.sqrt(sq_sum / len(pos_a)))


@dataclass(frozen=True)
class PlacementTrial:
    direction: np.ndarray
    d_min: float
    positions: np.ndarray
    orientation: Literal["single", "endo", "exo", "achiral"] = "single"


@dataclass(frozen=True)
class PlacementResult:
    trials: list[PlacementTrial]
    n_candidates: int
    n_blocked: int
    blocked_reasons: list[str | None]
    placement_kind: Literal["single_anchor", "multi_anchor"] = "single_anchor"


def _identify_substrate(
    frag_indices: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    if not frag_indices:
        raise ValueError("frag_indices is empty")
    return max(frag_indices, key=lambda f: (len(f), -min(f)))


def _find_bridging_formed(
    formed: tuple[tuple[int, int], ...],
    substrate: set[int],
    fragment: set[int],
) -> list[tuple[int, int]]:
    bridges = [
        (a, b)
        for a, b in formed
        if (a in substrate and b in fragment) or (b in substrate and a in fragment)
    ]
    if not bridges:
        raise ValueError(
            f"fragment has no formed bond bridging to substrate; "
            f"check input atom mapping (formed={list(formed)})"
        )
    if len(bridges) >= 3:
        raise NotImplementedError(
            f"multi-anchor placement with N>=3 bridges is out of scope; "
            f"got {len(bridges)} formed bonds bridging this fragment"
        )
    return [(a, b) if a in substrate else (b, a) for a, b in bridges]


def _broken_bridges_fragments(
    broken: tuple[tuple[int, int], ...],
    frag_indices: tuple[tuple[int, ...], ...],
) -> bool:
    if not broken:
        return False
    atom_to_frag = {a: k for k, frag in enumerate(frag_indices) for a in frag}
    return any(atom_to_frag.get(a) != atom_to_frag.get(b) for a, b in broken)


def _single_anchor_placement(
    positions: np.ndarray,
    syms: list[str],
    substrate: tuple[int, ...],
    fragment_set: set[int],
    bridge: tuple[int, int],
    *,
    n_candidates: int,
    seed: int,
    gap: float,
    d_min_ceiling: float,
) -> PlacementResult:
    anchor, incoming_anchor = bridge
    vdw_all = np.array([vdw_radius(s) for s in syms], dtype=float)
    out_positions = positions.copy()
    directions = sample_sphere_directions(n_candidates, seed=seed)

    fragment = sorted(fragment_set)
    substrate_atoms = [i for i in substrate if i != anchor]
    sub_pos = out_positions[substrate_atoms]
    sub_vdw = vdw_all[substrate_atoms]
    inc_pos = out_positions[fragment]
    inc_vdw = vdw_all[fragment]

    anchor_pos = out_positions[anchor]
    incoming_anchor_pos = out_positions[incoming_anchor]

    survivors: list[PlacementTrial] = []
    blocked_reasons: list[str | None] = []
    for d in directions:
        blocked, d_min, reason = evaluate_direction(
            d,
            anchor_pos,
            sub_pos,
            sub_vdw,
            inc_pos,
            inc_vdw,
            incoming_anchor_pos,
            gap=gap,
            d_min_ceiling=d_min_ceiling,
        )
        if blocked:
            blocked_reasons.append(reason)
            continue
        target = anchor_pos + d * d_min
        new_positions = out_positions.copy()
        new_positions[fragment] += target - incoming_anchor_pos
        survivors.append(
            PlacementTrial(
                direction=d,
                d_min=float(d_min),
                positions=new_positions,
                orientation="single",
            )
        )
        blocked_reasons.append(None)

    n_blocked_total = sum(1 for r in blocked_reasons if r is not None)
    if not survivors:
        raise RuntimeError(
            f"anchor at atom {anchor} has no valid placement direction "
            f"(all {n_candidates} candidates blocked); substrate may be fully enclosed"
        )
    if len(survivors) < max(1, n_candidates // 4):
        log.warning(
            "only %d/%d candidates survived blocking at anchor %d",
            len(survivors),
            n_candidates,
            anchor,
        )
    return PlacementResult(
        trials=survivors,
        n_candidates=n_candidates,
        n_blocked=n_blocked_total,
        blocked_reasons=blocked_reasons,
        placement_kind="single_anchor",
    )


DUAL_ANCHOR_ASYMMETRY_THRESHOLD = 0.40


def _multi_anchor_placement(
    positions: np.ndarray,
    syms: list[str],
    substrate_set: set[int],
    fragment_set: set[int],
    bridges: list[tuple[int, int]],
    *,
    n_candidates: int,
    seed: int,
    gap: float,
    d_min_ceiling: float,
) -> tuple[list[PlacementTrial], list[str | None]]:
    a1, i1 = bridges[0]
    a2, i2 = bridges[1]

    m_sub = (positions[a1] + positions[a2]) / 2.0
    v_sub = positions[a2] - positions[a1]
    l_sub = float(np.linalg.norm(v_sub))
    if l_sub < 1e-9:
        raise ValueError("substrate anchor pair is coincident")
    u_sub = v_sub / l_sub

    m_inc = (positions[i1] + positions[i2]) / 2.0
    v_inc = positions[i2] - positions[i1]
    l_inc = float(np.linalg.norm(v_inc))
    if l_inc < 1e-9:
        raise ValueError("incoming anchor pair is coincident")
    u_inc = v_inc / l_inc

    vdw_all = np.array([vdw_radius(s) for s in syms], dtype=float)
    substrate_atoms = sorted(substrate_set - {a1, a2})
    fragment = sorted(fragment_set)
    sub_pos = positions[substrate_atoms]
    sub_vdw = vdw_all[substrate_atoms]
    inc_pos = positions[fragment]
    inc_vdw = vdw_all[fragment]

    survivors: list[PlacementTrial] = []
    blocked_reasons: list[str | None] = []
    for d in sample_sphere_directions(n_candidates, seed=seed):
        d_min = compute_d_min(d, m_sub, sub_pos, sub_vdw, inc_pos, inc_vdw, m_inc, gap=gap)
        translation = (m_sub + d * d_min) - m_inc
        new_positions = positions.copy()
        for idx in fragment:
            new_positions[idx] = positions[idx] + translation

        cos_uu = float(np.clip(np.dot(u_inc, u_sub), -1.0, 1.0))
        if cos_uu < 1.0 - 1e-9:
            if cos_uu < -1.0 + 1e-9:
                ortho = np.array([1.0, 0.0, 0.0])
                if abs(float(np.dot(u_sub, ortho))) > 0.9:
                    ortho = np.array([0.0, 1.0, 0.0])
                axis = ortho - u_sub * float(np.dot(u_sub, ortho))
                axis /= float(np.linalg.norm(axis))
                angle = np.pi
            else:
                axis = np.cross(u_inc, u_sub)
                axis /= float(np.linalg.norm(axis))
                angle = float(np.arccos(cos_uu))
            new_positions = _rotate_atoms(
                new_positions,
                indices=tuple(fragment),
                axis=axis,
                center=m_sub + d * d_min,
                angle=angle,
            )

        b1 = float(np.linalg.norm(new_positions[i1] - new_positions[a1]))
        b2 = float(np.linalg.norm(new_positions[i2] - new_positions[a2]))
        if max(b1, b2) > d_min_ceiling:
            blocked_reasons.append(f"unreachable_dual_anchor:b1={b1:.2f},b2={b2:.2f}")
            continue
        if abs(b1 - b2) / max(b1, b2) > DUAL_ANCHOR_ASYMMETRY_THRESHOLD:
            blocked_reasons.append(f"asymmetric_dual_anchor:b1={b1:.2f},b2={b2:.2f}")
            continue

        endo_pos = new_positions
        exo_pos = _rotate_atoms(
            new_positions,
            indices=tuple(fragment),
            axis=u_sub,
            center=m_sub + d * d_min,
            angle=np.pi,
        )
        fragment_syms = [syms[i] for i in fragment]
        rmsd = _permutation_aware_rmsd(endo_pos[fragment], exo_pos[fragment], fragment_syms)
        if rmsd < 0.01:
            survivors.append(
                PlacementTrial(d, float(d_min), endo_pos, orientation="achiral")
            )
        else:
            survivors.append(PlacementTrial(d, float(d_min), endo_pos, orientation="endo"))
            survivors.append(PlacementTrial(d, float(d_min), exo_pos, orientation="exo"))
        blocked_reasons.append(None)
    return survivors, blocked_reasons


def valid_placements(
    mol_h: Chem.Mol,
    frag_indices: tuple[tuple[int, ...], ...],
    positions: np.ndarray,
    bond_changes: BondChanges,
    *,
    n_candidates: int = 64,
    seed: int = 0,
    gap: float = 0.5,
    d_min_ceiling: float = 8.0,
) -> PlacementResult:
    """Sample 4pi directions, reject steric blockers, and return candidates."""
    if len(frag_indices) == 1:
        return PlacementResult(
            trials=[
                PlacementTrial(
                    direction=np.array([0.0, 0.0, 1.0]),
                    d_min=0.0,
                    positions=positions.copy(),
                    orientation="single",
                )
            ],
            n_candidates=1,
            n_blocked=0,
            blocked_reasons=[None],
            placement_kind="single_anchor",
        )

    if _broken_bridges_fragments(bond_changes.broken, frag_indices):
        raise NotImplementedError("multi-substrate metathesis is out of scope")

    substrate = _identify_substrate(frag_indices)
    substrate_set = set(substrate)
    non_substrate = [f for f in frag_indices if f != substrate]
    if len(non_substrate) > 1:
        raise NotImplementedError(
            f"termolecular placement ({len(non_substrate)} non-substrate fragments) "
            "is out of scope"
        )

    fragment_set = set(non_substrate[0])
    bridges = _find_bridging_formed(bond_changes.formed, substrate_set, fragment_set)
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]

    if len(bridges) == 1:
        return _single_anchor_placement(
            positions,
            syms,
            substrate,
            fragment_set,
            bridges[0],
            n_candidates=n_candidates,
            seed=seed,
            gap=gap,
            d_min_ceiling=d_min_ceiling,
        )

    survivors, blocked_reasons = _multi_anchor_placement(
        positions,
        syms,
        substrate_set,
        fragment_set,
        bridges,
        n_candidates=n_candidates,
        seed=seed,
        gap=gap,
        d_min_ceiling=d_min_ceiling,
    )
    n_blocked_total = sum(1 for r in blocked_reasons if r is not None)
    if not survivors:
        raise RuntimeError(
            f"anchor pair ({bridges[0][0]}, {bridges[1][0]}) has no valid "
            f"placement direction (all {n_candidates} candidates blocked)"
        )
    return PlacementResult(
        trials=survivors,
        n_candidates=n_candidates,
        n_blocked=n_blocked_total,
        blocked_reasons=blocked_reasons,
        placement_kind="multi_anchor",
    )
