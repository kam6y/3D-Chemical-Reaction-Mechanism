"""Generic steric-aware fragment placement (Phase 7).

Replaces the Tier 1 / Tier 2 dispatch of embed3d.py with a single algorithm:
Fibonacci-sphere sample of N candidate directions from the substrate-side
anchor, two-stage blocking filter (angular shadow + d_min ceiling), and
per-direction d_min based on each fragment's projection onto d.
"""
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


def sample_sphere_directions(n: int, seed: int = 0) -> list[np.ndarray]:
    """Fibonacci sphere over full 4π sr; return n unit vectors.

    Index 0 is deterministically +z (re-producibility / phase origin).
    Remaining n-1 points are placed on the unit sphere by golden-angle spiral
    in z ∈ [-1, 1]; seed deterministically rotates the spiral phase.

    Raises:
        ValueError: when n < 1.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rotations: list[np.ndarray] = [np.array([0.0, 0.0, 1.0])]
    if n == 1:
        return rotations

    rng = np.random.default_rng(seed)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(n - 1):
        # i ∈ [0, n-2] → t ∈ (0, 1) → z ∈ (1 - 2*t) ∈ (-1, 1)
        t = (i + 0.5) / (n - 1)
        z = 1.0 - 2.0 * t
        r_xy = float(np.sqrt(max(0.0, 1.0 - z * z)))
        theta = phase + i * golden_angle
        x = r_xy * float(np.cos(theta))
        y = r_xy * float(np.sin(theta))
        v = np.array([x, y, z])
        # 数値誤差対策で再正規化
        rotations.append(v / float(np.linalg.norm(v)))
    return rotations


def compute_d_min(
    d: np.ndarray,
    anchor_pos: np.ndarray,
    substrate_positions: np.ndarray,    # (M, 3) anchor 自身を含めても anchor の vdW=0 なら無害
    substrate_vdw: np.ndarray,          # (M,)
    incoming_positions: np.ndarray,     # (K, 3) incoming fragment 全原子
    incoming_vdw: np.ndarray,           # (K,)
    incoming_anchor_pos: np.ndarray,    # (3,) incoming 内 bridging bond 端の現在位置
    *,
    gap: float = 0.5,
) -> float:
    """Per-direction shortest safe distance between fragments along d.

    max_substrate_fwd = max((p - anchor_pos) · d + r_vdW for p, r_vdW in substrate)
    max_incoming_back = max((p - incoming_anchor_pos) · (-d) + r_vdW for p, r_vdW in incoming)
    d_min = max_substrate_fwd + max_incoming_back + gap

    Note: max_substrate_fwd は全原子が anchor の後ろ (-d 側) にあるとき負になりうる。
    その値は clamp せずそのまま d_min に算入する (test_compute_d_min_substrate_behind_anchor_does_not_inflate
    で挙動を固定済み)。
    """
    sub_proj = (substrate_positions - anchor_pos) @ d + substrate_vdw
    inc_proj = (incoming_positions - incoming_anchor_pos) @ (-d) + incoming_vdw
    max_substrate_fwd = float(sub_proj.max()) if sub_proj.size else 0.0
    max_incoming_back = float(inc_proj.max()) if inc_proj.size else 0.0
    return max_substrate_fwd + max_incoming_back + float(gap)


def evaluate_direction(
    d: np.ndarray,
    anchor_pos: np.ndarray,
    substrate_positions: np.ndarray,    # (M, 3) anchor を除外済み
    substrate_vdw: np.ndarray,          # (M,)
    incoming_positions: np.ndarray,
    incoming_vdw: np.ndarray,
    incoming_anchor_pos: np.ndarray,
    *,
    gap: float = 0.5,
    d_min_ceiling: float = 8.0,
) -> tuple[bool, float, str | None]:
    """Two-stage blocking check; returns (blocked, d_min, reason).

    Stage 1 — angular shadow:
      ∃i s.t. angle(d, substrate_positions[i] - anchor_pos)
             < atan(substrate_vdw[i] / ||substrate_positions[i] - anchor_pos||)
      → blocked=True, d_min=NaN, reason="angle_shadow:atom_index=K".
    Stage 2 — d_min ceiling:
      compute_d_min(...) > d_min_ceiling
      → blocked=True, reason="d_min_ceiling:value=V".
    Otherwise: blocked=False, d_min=value, reason=None.
    """
    # Stage 1: angular shadow
    rel = substrate_positions - anchor_pos
    r = np.linalg.norm(rel, axis=1)
    # 数値安定化: 距離 0 (anchor 自身が紛れ込んだ等) は無視
    safe = r > 1e-9
    if np.any(safe):
        cos_to_d = (rel[safe] @ d) / r[safe]
        # Angle test: angle(d, rel) < atan(vdw / r)
        # ⇔ (d は単位、r > 0 で) cos(angle) > r / sqrt(r² + vdw²)
        # 形式変形により per-atom arctan を回避している。
        vdw_safe = substrate_vdw[safe]
        cos_thresh = r[safe] / np.sqrt(r[safe] ** 2 + vdw_safe ** 2)
        hits = np.where(cos_to_d > cos_thresh)[0]
        if hits.size:
            # safe-mask 上の index → 元 index へ戻す
            orig_indices = np.where(safe)[0]
            blocked_atom = int(orig_indices[hits[0]])
            return True, float("nan"), f"angle_shadow:atom_index={blocked_atom}"

    # Stage 2: d_min ceiling
    d_min = compute_d_min(
        d, anchor_pos, substrate_positions, substrate_vdw,
        incoming_positions, incoming_vdw, incoming_anchor_pos,
        gap=gap,
    )
    if d_min > d_min_ceiling:
        return True, d_min, f"d_min_ceiling:value={d_min:.3f}"
    return False, d_min, None


def _permutation_aware_rmsd(
    pos_a: np.ndarray,            # (K, 3) fragment atoms in pose A
    pos_b: np.ndarray,            # (K, 3) fragment atoms in pose B
    syms: list[str],              # length K, element symbols
) -> float:
    """RMSD between pose A and pose B over atom indices, but each atom in A
    matches the nearest unmatched atom in B of the same element.

    Used for achiral collapse: if 180° rotation maps the fragment to a
    permutation of itself (e.g., ethylene's C2 symmetry permutes H atoms),
    this RMSD will be ~0 even though index-wise RMSD is large.

    Greedy matching by element: for each atom in A in order, match to the
    nearest unmatched atom in B of the same element. For chemistry-grade
    symmetry detection (small fragments) this is accurate enough.
    """
    if len(pos_a) != len(pos_b) or len(pos_a) != len(syms):
        raise ValueError(
            f"shape mismatch: pos_a={pos_a.shape}, pos_b={pos_b.shape}, "
            f"len(syms)={len(syms)}"
        )
    K = len(pos_a)
    if K == 0:
        return 0.0
    used_b: set[int] = set()
    sq_sum = 0.0
    for i in range(K):
        sym_i = syms[i]
        # Find nearest unmatched atom in B with same element
        best_j = -1
        best_d2 = float("inf")
        for j in range(K):
            if j in used_b or syms[j] != sym_i:
                continue
            d2 = float(np.sum((pos_a[i] - pos_b[j]) ** 2))
            if d2 < best_d2:
                best_d2 = d2
                best_j = j
        if best_j < 0:
            # No matching element atom available — fall back to index-wise
            best_d2 = float(np.sum((pos_a[i] - pos_b[i]) ** 2))
        else:
            used_b.add(best_j)
        sq_sum += best_d2
    return float(np.sqrt(sq_sum / K))


def _rotate_atoms(
    positions: np.ndarray,
    *,
    indices: tuple[int, ...],
    axis: np.ndarray,
    center: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Rotate selected atoms around `axis` (passing through `center`) by `angle` rad.

    Uses Rodrigues' rotation formula. Returns a new positions array; input is
    not modified. `axis` is normalized internally; `angle == 0` returns a copy
    of `positions` unchanged.

    Raises:
        ValueError: when `axis` has zero length.
    """
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
        # Rodrigues: v_rot = v cos + (k×v) sin + k (k·v) (1 - cos)
        v_rot = (
            v * cos_a
            + np.cross(k, v) * sin_a
            + k * float(np.dot(k, v)) * (1.0 - cos_a)
        )
        new_pos[i] = center + v_rot
    return new_pos


@dataclass(frozen=True)
class PlacementTrial:
    """A single surviving placement candidate.

    Invariants:
    - `direction` is a unit vector (||direction||₂ == 1 to float tolerance).
    - `d_min` is the placement distance applied along `direction` from the
      anchor; for the unimolecular passthrough trial it is 0.0; for
      bimolecular trials it is `compute_d_min(...)` and may be negative
      when all substrate atoms lie behind the anchor relative to direction.
    - `positions` shape is `(N, 3)` matching `mol_h.GetNumAtoms()`. Substrate
      atoms are unchanged from input; the placed fragment(s) have been
      translated so `incoming_anchor` lands at `anchor_pos + direction * d_min`.
    - `orientation` records cycloaddition trial flavor:
      "single" for single-anchor (Phase 7) or unimolecular passthrough,
      "endo" / "exo" for cycloaddition with asymmetric incoming fragment,
      "achiral" for cycloaddition where endo/exo collapsed by RMSD.
    """
    direction: np.ndarray
    d_min: float
    positions: np.ndarray
    orientation: Literal["single", "endo", "exo", "achiral"] = "single"


@dataclass(frozen=True)
class PlacementResult:
    """Aggregated outcome of `valid_placements` on a single bond_changes.

    Invariants (single_anchor):
    - `len(trials) + n_blocked == n_candidates`
    - `len(blocked_reasons) == n_candidates`; element is None for survivors,
      a string like "angle_shadow:atom_index=K" or "d_min_ceiling:value=V"
      for blocked candidates.

    Invariants (multi_anchor):
    - `len(blocked_reasons) == n_candidates`
    - `len(trials) ∈ [n_candidates - n_blocked, 2 * (n_candidates - n_blocked)]`
    - exactly one None per surviving direction; trial-to-direction is 1-or-2
      (1 for achiral collapse, 2 for endo + exo on asymmetric fragments).

    Invariants (unimolecular passthrough, single_anchor):
    - `n_candidates == 1`, `n_blocked == 0`, `len(trials) == 1`,
      `blocked_reasons == [None]`.

    `placement_kind` records which dispatch path was taken:
    - "single_anchor": Phase 7 path (1 bridging formed bond, or unimolecular
      passthrough).
    - "multi_anchor": Phase 8 path (2 bridging formed bonds, cycloaddition).
    """
    trials: list[PlacementTrial]
    n_candidates: int
    n_blocked: int
    blocked_reasons: list[str | None]
    placement_kind: Literal["single_anchor", "multi_anchor"] = "single_anchor"


def _identify_substrate(
    frag_indices: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    """Largest fragment by atom count; tie-break by smallest minimum atom index."""
    if not frag_indices:
        raise ValueError("frag_indices is empty")
    return max(frag_indices, key=lambda f: (len(f), -min(f)))


def _find_bridging_formed(
    formed: tuple[tuple[int, int], ...],
    substrate: set[int],
    fragment: set[int],
) -> list[tuple[int, int]]:
    """Return all formed bonds bridging substrate ↔ fragment, normalized so
    the substrate-side atom is first.

    bridges == 0  → ValueError
    bridges == 1 or 2 → list of normalized tuples
    bridges >= 3 → NotImplementedError (general cycloaddition is Phase 9+)
    """
    bridges = [
        (a, b) for a, b in formed
        if (a in substrate and b in fragment) or (b in substrate and a in fragment)
    ]
    if not bridges:
        raise ValueError(
            f"fragment has no formed bond bridging to substrate; "
            f"check input atom mapping (formed={list(formed)})"
        )
    if len(bridges) >= 3:
        raise NotImplementedError(
            f"multi-anchor placement with N>=3 bridges is out of scope for Phase 8; "
            f"got {len(bridges)} formed bonds bridging this fragment"
        )
    # 正規化: substrate 側 atom が前
    return [
        (a, b) if a in substrate else (b, a)
        for (a, b) in bridges
    ]


def build_atoms_from_positions(
    mol_h: Chem.Mol,
    positions: np.ndarray,
) -> Atoms:
    """Build an ase.Atoms with symbols, formal charges, and given positions.

    Mirrors the Atoms construction path that used to live inside
    embed3d.embed_mol_to_atoms (symbols + initial_charges + info[charge/spin]).
    """
    if positions.shape != (mol_h.GetNumAtoms(), 3):
        raise ValueError(
            f"positions shape {positions.shape} does not match "
            f"mol_h atom count {mol_h.GetNumAtoms()}"
        )
    symbols = [a.GetSymbol() for a in mol_h.GetAtoms()]
    charges = [a.GetFormalCharge() for a in mol_h.GetAtoms()]
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_initial_charges(charges)
    atoms.info["charge"] = int(sum(charges))
    atoms.info["spin"] = 1   # Phase Re1 baseline: closed-shell singlet
    return atoms


def _broken_bridges_fragments(
    broken: tuple[tuple[int, int], ...],
    frag_indices: tuple[tuple[int, ...], ...],
) -> bool:
    """True iff at least one broken bond has its two atoms in different fragments."""
    if not broken:
        return False
    atom_to_frag: dict[int, int] = {}
    for k, frag in enumerate(frag_indices):
        for a in frag:
            atom_to_frag[a] = k
    return any(atom_to_frag.get(a) != atom_to_frag.get(b) for a, b in broken)


DUAL_ANCHOR_ASYMMETRY_THRESHOLD = 0.40   # see spec §5.4 (used in Task 4.4)


def _multi_anchor_placement(
    positions: np.ndarray,
    syms: list[str],
    substrate_set: set[int],
    fragment_set: set[int],
    bridges: list[tuple[int, int]],   # length 2; substrate-side first (normalized)
    *,
    n_candidates: int = 64,
    seed: int = 0,
    gap: float = 0.5,
    d_min_ceiling: float = 8.0,
) -> tuple[list[PlacementTrial], list[str | None]]:
    """Cycloaddition (bridges == 2) 用の rigid-body multi-anchor 配置 (skeleton).

    Phase 8 step 12/N (Task 4.3): translation + 2-point Kabsch alignment.
    Reachability blocking (Task 4.4) and endo/exo expansion (Task 4.5) are
    added incrementally in subsequent tasks.

    Returns:
        (survivors, blocked_reasons) where:
        - survivors: list of PlacementTrial for surviving directions.
        - blocked_reasons: list of length n_candidates with None for survivors
          and a string for blocked directions. (No blocking yet — all None.)
    """
    A1, I1 = bridges[0]
    A2, I2 = bridges[1]

    M_sub = (positions[A1] + positions[A2]) / 2.0
    v_sub = positions[A2] - positions[A1]
    L_sub = float(np.linalg.norm(v_sub))
    if L_sub < 1e-9:
        raise ValueError(
            f"substrate anchor pair (A1={A1}, A2={A2}) are coincident; "
            f"check input atom mapping"
        )
    u_sub = v_sub / L_sub

    M_inc = (positions[I1] + positions[I2]) / 2.0
    v_inc = positions[I2] - positions[I1]
    L_inc = float(np.linalg.norm(v_inc))
    if L_inc < 1e-9:
        raise ValueError(
            f"incoming anchor pair (I1={I1}, I2={I2}) are coincident; "
            f"check formed bond atom-mapping"
        )
    u_inc = v_inc / L_inc

    vdw_all = np.array([vdw_radius(s) for s in syms], dtype=float)
    substrate_atoms = sorted(substrate_set - {A1, A2})
    fragment_list = sorted(fragment_set)
    sub_pos = positions[substrate_atoms]
    sub_vdw = vdw_all[substrate_atoms]
    inc_pos = positions[fragment_list]
    inc_vdw = vdw_all[fragment_list]

    directions = sample_sphere_directions(n_candidates, seed=seed)
    survivors: list[PlacementTrial] = []
    blocked_reasons: list[str | None] = []

    for d in directions:
        # Step A: placement distance via existing compute_d_min, anchored at M_sub
        d_min = compute_d_min(
            d, M_sub, sub_pos, sub_vdw, inc_pos, inc_vdw, M_inc, gap=gap,
        )

        # Step B: translate fragment so M_inc lands at M_sub + d * d_min
        translation = (M_sub + d * d_min) - M_inc
        new_positions = positions.copy()
        for idx in fragment_list:
            new_positions[idx] = positions[idx] + translation

        # Step C: 2-point Kabsch rotation u_inc → u_sub
        cos_uu = float(np.clip(np.dot(u_inc, u_sub), -1.0, 1.0))
        if cos_uu > 1.0 - 1e-9:
            # Already aligned: identity
            pass
        else:
            if cos_uu < -1.0 + 1e-9:
                # Antiparallel: pick any axis ⊥ u_sub
                ortho = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(u_sub, ortho)) > 0.9:
                    ortho = np.array([0.0, 1.0, 0.0])
                R_axis = ortho - u_sub * float(np.dot(u_sub, ortho))
                R_axis /= float(np.linalg.norm(R_axis))
                R_angle = np.pi
            else:
                R_axis = np.cross(u_inc, u_sub)
                R_axis /= float(np.linalg.norm(R_axis))
                R_angle = float(np.arccos(cos_uu))

            new_positions = _rotate_atoms(
                new_positions,
                indices=tuple(fragment_list),
                axis=R_axis,
                center=M_sub + d * d_min,
                angle=R_angle,
            )

        # Reachability check (Task 4.4):
        b1 = float(np.linalg.norm(new_positions[I1] - new_positions[A1]))
        b2 = float(np.linalg.norm(new_positions[I2] - new_positions[A2]))

        if max(b1, b2) > d_min_ceiling:
            blocked_reasons.append(f"unreachable_dual_anchor:b1={b1:.2f},b2={b2:.2f}")
            continue
        if abs(b1 - b2) / max(b1, b2) > DUAL_ANCHOR_ASYMMETRY_THRESHOLD:
            blocked_reasons.append(f"asymmetric_dual_anchor:b1={b1:.2f},b2={b2:.2f}")
            continue

        # direction survived blocking; expand into endo/exo trials (Task 4.5)
        endo_pos = new_positions
        exo_pos = _rotate_atoms(
            new_positions,
            indices=tuple(fragment_list),
            axis=u_sub,
            center=M_sub + d * d_min,
            angle=np.pi,
        )
        # Symmetry detection via permutation-aware RMSD on fragment atoms
        endo_frag = endo_pos[fragment_list]
        exo_frag = exo_pos[fragment_list]
        fragment_syms = [syms[i] for i in fragment_list]
        rmsd = _permutation_aware_rmsd(endo_frag, exo_frag, fragment_syms)
        if rmsd < 0.01:
            # symmetric fragment → 1 achiral trial
            survivors.append(PlacementTrial(
                direction=d, d_min=float(d_min),
                positions=endo_pos, orientation="achiral",
            ))
        else:
            # asymmetric → 2 trials (endo + exo)
            survivors.append(PlacementTrial(
                direction=d, d_min=float(d_min),
                positions=endo_pos, orientation="endo",
            ))
            survivors.append(PlacementTrial(
                direction=d, d_min=float(d_min),
                positions=exo_pos, orientation="exo",
            ))
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
    """Sphere-sample directions, filter by angular shadow + d_min ceiling, place.

    Unimolecular (1 fragment): returns single passthrough trial (direction=+z, d_min=0).
    Bimolecular: identifies substrate (largest fragment) and one non-substrate fragment;
    samples n_candidates directions; for each, evaluates blocking and applies translation.

    Raises:
        NotImplementedError: when broken bond bridges fragments (metathesis), or when
            a non-substrate fragment is connected to the substrate by ≥2 formed bonds
            (cycloaddition).
        RuntimeError: when 0 candidates survive blocking.
        ValueError: when a non-substrate fragment has no bridging formed bond.
    """
    # Unimolecular: passthrough.
    if len(frag_indices) == 1:
        return PlacementResult(
            trials=[PlacementTrial(
                direction=np.array([0.0, 0.0, 1.0]),
                d_min=0.0,
                positions=positions.copy(),
                orientation="single",
            )],
            n_candidates=1,
            n_blocked=0,
            blocked_reasons=[None],
            placement_kind="single_anchor",
        )

    # Bimolecular: refuse metathesis up front.
    if _broken_bridges_fragments(bond_changes.broken, frag_indices):
        raise NotImplementedError(
            "multi-substrate metathesis (broken bonds spanning fragments) "
            "is out of scope for Phase 7"
        )

    substrate = _identify_substrate(frag_indices)
    substrate_set = set(substrate)
    non_substrate = [f for f in frag_indices if f is not substrate]

    # Phase 7 supports a single non-substrate fragment per call.
    if len(non_substrate) > 1:
        raise NotImplementedError(
            f"termolecular placement ({len(non_substrate)} non-substrate "
            f"fragments) is out of scope for Phase 7; only one non-substrate "
            f"fragment is supported"
        )

    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    vdw_all = np.array([vdw_radius(s) for s in syms], dtype=float)

    out_positions = positions.copy()
    directions = sample_sphere_directions(n_candidates, seed=seed)

    fragment = non_substrate[0]
    fragment_set = set(fragment)
    bridges = _find_bridging_formed(bond_changes.formed, substrate_set, fragment_set)
    if len(bridges) == 2:
        raise NotImplementedError(
            "multi-anchor placement (bridges==2) is wired in Task 4.3"
        )
    # bridges == 1 (single-anchor path)
    anchor, incoming_anchor = bridges[0]   # already normalized: substrate first, fragment second

    substrate_atoms = [i for i in substrate if i != anchor]
    sub_pos = out_positions[substrate_atoms]
    sub_vdw = vdw_all[substrate_atoms]
    inc_pos = out_positions[list(fragment)]
    inc_vdw = vdw_all[list(fragment)]

    anchor_pos = out_positions[anchor]
    incoming_anchor_pos = out_positions[incoming_anchor]

    survivors: list[PlacementTrial] = []
    blocked_reasons: list[str | None] = []
    for d in directions:
        blocked, d_min, reason = evaluate_direction(
            d, anchor_pos, sub_pos, sub_vdw,
            inc_pos, inc_vdw, incoming_anchor_pos,
            gap=gap, d_min_ceiling=d_min_ceiling,
        )
        if blocked:
            blocked_reasons.append(reason)
            continue
        target = anchor_pos + d * d_min
        new_positions = out_positions.copy()
        new_positions[list(fragment)] += target - incoming_anchor_pos
        survivors.append(PlacementTrial(
            direction=d, d_min=d_min, positions=new_positions,
            orientation="single",
        ))
        blocked_reasons.append(None)

    n_blocked_total = sum(1 for r in blocked_reasons if r is not None)
    if not survivors:
        raise RuntimeError(
            f"anchor at atom {anchor} has no valid placement direction "
            f"(all {n_candidates} candidates blocked); substrate may be "
            f"fully enclosed"
        )
    if len(survivors) < max(1, n_candidates // 4):
        log.warning(
            "only %d/%d candidates survived blocking at anchor %d; "
            "consider larger n_candidates or check substrate geometry",
            len(survivors), n_candidates, anchor,
        )

    return PlacementResult(
        trials=survivors,
        n_candidates=n_candidates,
        n_blocked=n_blocked_total,
        blocked_reasons=blocked_reasons,
        placement_kind="single_anchor",
    )
