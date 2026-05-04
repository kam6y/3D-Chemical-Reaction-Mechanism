"""Generic steric-aware fragment placement (Phase 7).

Replaces the Tier 1 / Tier 2 dispatch of embed3d.py with a single algorithm:
Fibonacci-sphere sample of N candidate directions from the substrate-side
anchor, two-stage blocking filter (angular shadow + d_min ceiling), and
per-direction d_min based on each fragment's projection onto d.
"""
from __future__ import annotations

import logging

import numpy as np

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
        # angle threshold: atan(vdw / r) → cos threshold: r / sqrt(r^2 + vdw^2)
        vdw_safe = substrate_vdw[safe]
        cos_thresh = r[safe] / np.sqrt(r[safe] ** 2 + vdw_safe ** 2)
        # angle < threshold ⇔ cos > cos_thresh
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
