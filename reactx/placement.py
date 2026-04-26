"""Generic bond-change-driven fragment placement.

Replaces Phase 0's SN2-hardcoded backside-attack placement
(_find_c_lg_bond / _place_nucleophile_backside in embed3d).
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from rdkit import Chem

from reactx.reaction_topology import BondChange, BondChanges

log = logging.getLogger(__name__)

DEFAULT_D_FORM = 3.5  # Å — Phase 0 SN2 で使われた値と一致 (FRAGMENT_SEPARATION)
DEFAULT_D_DISSOC = 4.0  # Å — broken bond による product 側分離距離


def place_fragments_generic(
    mol_h: Chem.Mol,
    frag_indices: tuple[tuple[int, ...], ...],
    positions: np.ndarray,
    bond_changes: BondChanges,
    *,
    side: Literal["reactant", "product"],
    d_form: float = DEFAULT_D_FORM,
    d_dissoc: float = DEFAULT_D_DISSOC,
    index_translation: dict[int, int] | None = None,
) -> np.ndarray:
    """Position fragments (i != primary) relative to the primary fragment.

    primary fragment = the heavy-atom-richest fragment. For each non-primary
    fragment, find anchor atom pairs across the boundary that participate in
    formed (reactant side) or broken (product side) bond changes, compute each
    anchor's ideal position, then rigid-transform the fragment via Kabsch
    alignment. Single-anchor fragments are translated only.

    bond_changes uses reactant indexing by convention. When mol_h is the
    *product* mol_h, pass index_translation (typically expanded_atom_mapping
    from reaction_topology) so that BondChange.{a,b} are projected to
    product-side indices.
    """
    if len(frag_indices) <= 1:
        return positions

    bc = _translate_bond_changes(bond_changes, index_translation)

    primary_idx = _pick_primary_fragment(mol_h, frag_indices)
    primary = frag_indices[primary_idx]
    primary_set = set(primary)

    relevant = bc.formed if side == "reactant" else bc.broken
    distance = d_form if side == "reactant" else d_dissoc

    out = positions.copy()
    for i, frag in enumerate(frag_indices):
        if i == primary_idx:
            continue
        anchors = _collect_anchors(
            primary_set,
            set(frag),
            relevant,
            out,
            bc,
            side=side,
            distance=distance,
        )
        if not anchors:
            log.warning(
                "Fragment %d has no anchor bond change to primary fragment; "
                "placing naively along +x",
                i,
            )
            out = _place_naively(out, primary, frag, distance)
            continue
        out = _apply_kabsch(out, frag, anchors)
    return out


def _translate_bond_changes(
    bc: BondChanges,
    index_map: dict[int, int] | None,
) -> BondChanges:
    if index_map is None:
        return bc
    return BondChanges(
        broken=[
            BondChange(
                a=index_map[c.a],
                b=index_map[c.b],
                order_before=c.order_before,
                order_after=c.order_after,
            )
            for c in bc.broken
        ],
        formed=[
            BondChange(
                a=index_map[c.a],
                b=index_map[c.b],
                order_before=c.order_before,
                order_after=c.order_after,
            )
            for c in bc.formed
        ],
    )


def _pick_primary_fragment(mol_h: Chem.Mol, frag_indices: tuple[tuple[int, ...], ...]) -> int:
    """Largest fragment by heavy-atom count; ties broken by lowest index."""
    best_size = -1
    best_idx = 0
    for i, frag in enumerate(frag_indices):
        n_heavy = sum(1 for idx in frag if mol_h.GetAtomWithIdx(idx).GetAtomicNum() > 1)
        if n_heavy > best_size:
            best_size = n_heavy
            best_idx = i
    return best_idx


def _collect_anchors(
    primary_set: set[int],
    frag_set: set[int],
    relevant: list[BondChange],
    positions: np.ndarray,
    bond_changes: BondChanges,
    *,
    side: str,
    distance: float,
) -> list[tuple[int, np.ndarray]]:
    """Return [(b_idx_in_fragment, ideal_position), ...]."""
    anchors: list[tuple[int, np.ndarray]] = []
    for bc in relevant:
        a, b = _split_bond_across_boundary(bc, primary_set, frag_set)
        if a is None or b is None:
            continue
        ideal_b = _compute_ideal_position(
            a=a,
            b=b,
            positions=positions,
            primary_set=primary_set,
            bond_changes=bond_changes,
            side=side,
            distance=distance,
        )
        anchors.append((b, ideal_b))
    return anchors


def _split_bond_across_boundary(
    bc: BondChange, primary_set: set[int], frag_set: set[int]
) -> tuple[int | None, int | None]:
    if bc.a in primary_set and bc.b in frag_set:
        return bc.a, bc.b
    if bc.b in primary_set and bc.a in frag_set:
        return bc.b, bc.a
    return None, None


def _compute_ideal_position(
    *,
    a: int,
    b: int,
    positions: np.ndarray,
    primary_set: set[int],
    bond_changes: BondChanges,
    side: str,
    distance: float,
) -> np.ndarray:
    """Find anchor atom b's ideal position relative to primary atom a.

    Reactant side: b is the "incoming" atom (formed bond endpoint outside
    primary). If a also has a broken bond to a partner in primary, place b on
    the backside of (a, partner) — reproduces SN2 backside attack.

    Product side: b is the "leaving" atom (broken bond endpoint outside
    primary). If a also has a formed bond to a partner in primary, place b on
    the backside of (a, partner) — reproduces Walden-inverted product (e.g.
    Cl⁻ opposite from the new C–F).

    Without a partner-in-primary anchor, fall back to a direction that
    pushes b away from the primary fragment centroid.
    """
    p_a = positions[a]
    candidate_changes = bond_changes.broken if side == "reactant" else bond_changes.formed
    for change in candidate_changes:
        partner: int | None = None
        if change.a == a and change.b in primary_set:
            partner = change.b
        elif change.b == a and change.a in primary_set:
            partner = change.a
        if partner is not None:
            direction = p_a - positions[partner]
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                break
            return p_a + direction / norm * distance
    return _away_from_centroid(p_a, positions, primary_set, distance)


def _away_from_centroid(
    p_a: np.ndarray, positions: np.ndarray, primary_set: set[int], distance: float
) -> np.ndarray:
    centroid = positions[list(primary_set)].mean(axis=0)
    direction = p_a - centroid
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.array([1.0, 0.0, 0.0])
        norm = 1.0
    return p_a + direction / norm * distance


def _apply_kabsch(
    positions: np.ndarray,
    frag: tuple[int, ...],
    anchors: list[tuple[int, np.ndarray]],
) -> np.ndarray:
    if len(anchors) == 1:
        b_idx, ideal = anchors[0]
        delta = ideal - positions[b_idx]
        out = positions.copy()
        for idx in frag:
            out[idx] = positions[idx] + delta
        return out

    src = np.array([positions[b] for b, _ in anchors])
    dst = np.array([ideal for _, ideal in anchors])
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    out = positions.copy()
    for idx in frag:
        out[idx] = (positions[idx] - src_c) @ R.T + dst_c
    return out


def _place_naively(
    positions: np.ndarray,
    primary: tuple[int, ...],
    frag: tuple[int, ...],
    distance: float,
) -> np.ndarray:
    primary_pos = positions[list(primary)]
    frag_pos = positions[list(frag)]
    primary_c = primary_pos.mean(axis=0)
    frag_c = frag_pos.mean(axis=0)
    primary_radius = float(np.linalg.norm(primary_pos - primary_c, axis=1).max())
    frag_radius = float(np.linalg.norm(frag_pos - frag_c, axis=1).max()) if len(frag) > 1 else 0.0
    target = primary_c + np.array([primary_radius + frag_radius + distance, 0.0, 0.0])
    delta = target - frag_c
    out = positions.copy()
    for idx in frag:
        out[idx] = positions[idx] + delta
    return out
