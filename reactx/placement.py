"""Generic steric-aware fragment placement (Phase 7).

Replaces the Tier 1 / Tier 2 dispatch of embed3d.py with a single algorithm:
Fibonacci-sphere sample of N candidate directions from the substrate-side
anchor, two-stage blocking filter (angular shadow + d_min ceiling), and
per-direction d_min based on each fragment's projection onto d.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

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


@dataclass(frozen=True)
class PlacementTrial:
    direction: np.ndarray   # (3,) unit vector
    d_min: float            # placement distance applied along direction
    positions: np.ndarray   # (N, 3) full-system coordinates after translation


@dataclass(frozen=True)
class PlacementResult:
    trials: list[PlacementTrial]
    n_candidates: int
    n_blocked: int
    blocked_reasons: list[str | None]


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
) -> tuple[int, int]:
    """Return the unique formed bond bridging substrate ↔ fragment.

    Multiple → NotImplementedError (cycloaddition is Phase 8+).
    None → ValueError.
    """
    bridges = [
        (a, b) for a, b in formed
        if (a in substrate and b in fragment) or (b in substrate and a in fragment)
    ]
    if len(bridges) > 1:
        raise NotImplementedError(
            f"multi-anchor placement (cycloaddition) is out of scope for Phase 7; "
            f"got {len(bridges)} formed bonds bridging this fragment"
        )
    if not bridges:
        raise ValueError(
            f"fragment has no formed bond bridging to substrate; "
            f"check input atom mapping (formed={list(formed)})"
        )
    return bridges[0]


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
    for a, b in broken:
        if atom_to_frag.get(a) != atom_to_frag.get(b):
            return True
    return False


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
            )],
            n_candidates=1,
            n_blocked=0,
            blocked_reasons=[None],
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

    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    vdw_all = np.array([vdw_radius(s) for s in syms])

    # Phase 7 supports a single non-substrate fragment per call.
    if len(non_substrate) > 1:
        log.warning(
            "Phase 7: %d non-substrate fragments detected; placing each "
            "independently with the same sphere sample (geometric quality "
            "may degrade for termolecular)", len(non_substrate),
        )

    out_positions = positions.copy()
    survivors: list[PlacementTrial] | None = None
    blocked_reasons_final: list[str | None] = []
    n_blocked_total = 0

    directions = sample_sphere_directions(n_candidates, seed=seed)

    for fragment in non_substrate:
        fragment_set = set(fragment)
        bridge = _find_bridging_formed(bond_changes.formed, substrate_set, fragment_set)
        anchor = bridge[0] if bridge[0] in substrate_set else bridge[1]
        incoming_anchor = bridge[1] if bridge[0] == anchor else bridge[0]

        substrate_atoms = [i for i in substrate if i != anchor]
        sub_pos = out_positions[substrate_atoms]
        sub_vdw = vdw_all[substrate_atoms]
        inc_pos = out_positions[list(fragment)]
        inc_vdw = vdw_all[list(fragment)]

        anchor_pos = out_positions[anchor]
        incoming_anchor_pos = out_positions[incoming_anchor]

        per_direction_trials: list[PlacementTrial] = []
        per_direction_reasons: list[str | None] = []
        for d in directions:
            blocked, d_min, reason = evaluate_direction(
                d, anchor_pos, sub_pos, sub_vdw,
                inc_pos, inc_vdw, incoming_anchor_pos,
                gap=gap, d_min_ceiling=d_min_ceiling,
            )
            if blocked:
                per_direction_reasons.append(reason)
                continue
            target = anchor_pos + d * d_min
            new_positions = out_positions.copy()
            new_positions[list(fragment)] += target - incoming_anchor_pos
            per_direction_trials.append(PlacementTrial(
                direction=d, d_min=d_min, positions=new_positions,
            ))
            per_direction_reasons.append(None)

        n_blocked_this = sum(1 for r in per_direction_reasons if r is not None)
        if not per_direction_trials:
            raise RuntimeError(
                f"anchor at atom {anchor} has no valid placement direction "
                f"(all {n_candidates} candidates blocked); substrate may be "
                f"fully enclosed"
            )
        if len(per_direction_trials) < max(1, n_candidates // 4):
            log.warning(
                "only %d/%d candidates survived blocking at anchor %d; "
                "consider larger n_candidates or check substrate geometry",
                len(per_direction_trials), n_candidates, anchor,
            )

        if survivors is None:
            survivors = per_direction_trials
            blocked_reasons_final = per_direction_reasons
            n_blocked_total = n_blocked_this
        else:
            # 2 つ目以降の non-substrate fragment は 1 つ目で生存した direction だけ
            # で配置を続ける (termolecular は警告済み)。実用上 Phase 7 では発火しない。
            log.warning(
                "termolecular placement: applying first-fragment survivors to "
                "subsequent fragment without re-filtering (best-effort)"
            )

    assert survivors is not None
    return PlacementResult(
        trials=survivors,
        n_candidates=n_candidates,
        n_blocked=n_blocked_total,
        blocked_reasons=blocked_reasons_final,
    )
