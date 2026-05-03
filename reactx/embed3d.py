"""Convert 2D RDKit Mol to 3D ase.Atoms via RDKit ETKDG + MMFF (+ optional UMA).

Preserves the atom ordering of Chem.AddHs(mol) so that downstream consumers
(align, NEB, restraints) can correlate atom indices with the same AddHs(mol)
result.

Multi-fragment placement is driven by the caller-supplied BondChanges via the
_place_fragments dispatcher, which selects one of three placement strategies:

- Tier 1 (Phase 3, _directional_placement): broken bond の方向情報がある反応
  (E2 / SN2 / proton transfer / Menshutkin / SN1 dissoc) で、anchor の backside
  に nucleophile を置く。
- Tier 2 (Phase 4, _planar_face_placement): broken=() の bimolecular (SN1 step 2
  recombination) で、anchor の sp²-like 平面の法線方向に nucleophile を置く。
- Tier 3 (Phase 5, _kabsch_alignment): 2-fragment 4-center metathesis (formed=2,
  broken=2, broken bonds 各 fragment 内で完結) で、anchor pair に垂直な face に
  target を立て Kabsch (orthogonal Procrustes) で moving fragment を剛体整列する。

An optional rotation_perturbation generates the cone of multi-angle trials
(applied per-tier in the strategy that fits the geometry).
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
PLANE_FIT_TOLERANCE = 0.3  # Å — SVD residual (smallest singular value) threshold


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
    Tier 2 (planar face, Phase 4): broken=() かつ formed=1 の bimolecular (SN1 step 2)。
    Tier 3 (Kabsch, Phase 5):     2-fragment 4-center metathesis (formed=2, broken=2,
                                   broken bonds 各 fragment 内で完結)。
    Phase 6+ (未実装):            cycloaddition (formed>=2, broken=0) /
                                   3+ fragment ionic salt metathesis /
                                   非対称 metathesis (formed_count != broken_count) /
                                   broken bonds が両 fragment を跨ぐケース。
    """
    substrate = _find_substrate_fragment(frag_indices, bond_changes.broken)
    if substrate is not None and bond_changes.broken:
        return _directional_placement(
            frag_indices, positions, bond_changes, substrate,
            rotation_perturbation=rotation_perturbation,
        )

    if not bond_changes.broken and bond_changes.formed:
        if len(bond_changes.formed) > 1:
            raise NotImplementedError(
                "cycloaddition (broken=0, formed>=2) is Phase 6+. "
                f"Got formed={bond_changes.formed}, frags={len(frag_indices)}."
            )
        substrate = _find_substrate_by_size(frag_indices)
        return _planar_face_placement(
            mol_h, frag_indices, positions, bond_changes, substrate,
            rotation_perturbation=rotation_perturbation,
        )

    # Tier 3: Phase 5 metathesis (broken bonds 各 fragment 内に閉じ、formed が両 frag を跨ぐ)
    if (
        substrate is None
        and bond_changes.broken
        and bond_changes.formed
        and len(frag_indices) == 2
        and len(bond_changes.formed) == 2
        and len(bond_changes.broken) == 2
    ):
        # Tier 3 内で broken_within_reference / broken_within_moving の本数を再検証
        # (1+1 でない場合は _kabsch_alignment が NotImplementedError を投げる)
        return _kabsch_alignment(
            mol_h, frag_indices, positions, bond_changes,
            rotation_perturbation=rotation_perturbation,
        )

    raise NotImplementedError(
        "multi-substrate placement only supports 2-fragment 4-center metathesis "
        "(formed=2, broken=2) in Phase 5; other shapes are Phase 6+. "
        f"Got formed={bond_changes.formed}, broken={bond_changes.broken}, "
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


def _find_substrate_by_size(
    frag_indices: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    """Tier 2 substrate identification: largest fragment by total atom count.

    Heavy-atom-count proxy: len(f) counts heavy + H, not heavy-only. For SN1
    step 2 (tBu⁺ 13 atoms vs Cl⁻ 1 atom) this is exact. Tie の場合は最小 atom
    index を含む方を選ぶ (deterministic)。
    """
    if not frag_indices:
        raise ValueError("frag_indices is empty")
    # NOTE: len(f) is a heavy-count proxy. If a future caller needs true heavy-
    # count ranking (e.g. H-rich substrate vs halide), pass mol_h and filter
    # atoms with GetAtomicNum() > 1.
    return max(
        frag_indices,
        key=lambda f: (len(f), -min(f)),
    )


def _plane_normal_at_anchor(
    positions: np.ndarray,
    anchor: int,
    mol_h: Chem.Mol,
    substrate: tuple[int, ...],
) -> np.ndarray:
    """Tier 2: anchor の sp²-like 平面の法線方向を返す (unit vector)。

    Strategy (priority order):
      1. anchor の substrate 内隣接 (heavy + H) を集める。
      2. 隣接 ≥3 かつ平面 fit 残差 < PLANE_FIT_TOLERANCE: SVD 法線。
         符号 disambiguation: direction[2] < 0 なら反転 (常に +z 寄り)。
      3. 隣接 = 1 or 2、または平面 fit 残差が大きい:
         direction = -unit(mean_neighbor - anchor)。norm < 1e-6 なら次へ。
      4. degenerate: direction = [0, 0, 1] + warning ログ。
    """
    substrate_set = set(substrate)
    neighbors_in_substrate = [
        n.GetIdx() for n in mol_h.GetAtomWithIdx(anchor).GetNeighbors()
        if n.GetIdx() in substrate_set
    ]

    if len(neighbors_in_substrate) >= 3:
        coords = np.array([positions[i] for i in neighbors_in_substrate])
        centered = coords - positions[anchor]
        # SVD: 最小特異値方向が plane normal
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        residual = float(S[-1])
        if residual < PLANE_FIT_TOLERANCE:
            normal = Vt[-1]
            if normal[2] < 0:
                normal = -normal
            # Vt rows are orthonormal; division is defensive (norm == 1 by construction).
            return normal / np.linalg.norm(normal)

    # Fallback: -unit(mean_neighbor - anchor)
    if neighbors_in_substrate:
        coords = np.array([positions[i] for i in neighbors_in_substrate])
        mean_neighbor = coords.mean(axis=0)
        direction = positions[anchor] - mean_neighbor
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            return direction / norm

    log.warning(
        "anchor %d has no usable substrate neighbors for plane-normal "
        "computation; using +z fallback direction", anchor,
    )
    return np.array([0.0, 0.0, 1.0])


def _directional_placement(
    frag_indices: tuple[tuple[int, ...], ...],
    positions: np.ndarray,
    bond_changes: BondChanges,
    substrate: tuple[int, ...],
    *,
    rotation_perturbation: np.ndarray | None,
) -> np.ndarray:
    """Tier 1: anchor + leaving direction for each non-substrate fragment.

    同一 anchor に複数 broken bond が掛かる場合 (例: retro-cycloaddition で
    将来現れうる)、leaving 原子は spec §9 (#3) に従い「相手側 atom index 最小」
    で決定論的に選ぶ。
    """
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

        # bridging_bond は line 161-164 で (substrate↔fragment) と filter 済み、
        # anchor は substrate 側端なので incoming (反対端) は構造的に必ず f_set ∋。
        incoming = bridging_bond[1] if bridging_bond[0] == anchor else bridging_bond[0]
        positions[list(fragment)] += target - positions[incoming]

    return positions


def _planar_face_placement(
    mol_h: Chem.Mol,
    frag_indices: tuple[tuple[int, ...], ...],
    positions: np.ndarray,
    bond_changes: BondChanges,
    substrate: tuple[int, ...],
    *,
    rotation_perturbation: np.ndarray | None,
) -> np.ndarray:
    """Tier 2 placement: anchor の sp²-like 平面の法線方向に nucleophile を置く。

    For each non-substrate fragment F:
      bridging = formed bond で substrate↔F を跨ぐもの (空なら ValueError)。
      複数あれば canonical-ordered の最初の 1 本を deterministic に採用。
      anchor   = bridging の substrate 側端。
      direction = _plane_normal_at_anchor(...) を rotation_perturbation で回転。
      F の incoming 原子を anchor + direction * FRAGMENT_SEPARATION に置く。

    複数の non-substrate fragment が同じ anchor を共有する場合は
    NotImplementedError("multi-base attack on single anchor not supported")。
    """
    substrate_set = set(substrate)
    non_substrate = [f for f in frag_indices if f is not substrate]
    if len(non_substrate) >= 2:
        log.warning(
            "Tier 2 termolecular placement (%d non-substrate fragments); "
            "geometric quality may be reduced", len(non_substrate),
        )

    bridging_by_anchor: dict[int, list[tuple[tuple[int, ...], tuple[int, int]]]] = {}
    for f_idx, f in enumerate(non_substrate):
        f_set = set(f)
        bridging = sorted([
            (a, b) if a <= b else (b, a)
            for a, b in bond_changes.formed
            if (a in substrate_set and b in f_set) or (b in substrate_set and a in f_set)
        ])
        if not bridging:
            raise ValueError(
                f"fragment {f_idx} has no formed bond bridging to substrate; "
                f"check input atom mapping (formed={bond_changes.formed}, "
                f"substrate atoms={sorted(substrate_set)})"
            )
        chosen_bond = bridging[0]
        anchor = chosen_bond[0] if chosen_bond[0] in substrate_set else chosen_bond[1]
        bridging_by_anchor.setdefault(anchor, []).append((f, chosen_bond))

    for anchor, hits in bridging_by_anchor.items():
        unique_frags = {id(f) for f, _ in hits}
        if len(unique_frags) > 1:
            raise NotImplementedError(
                f"multi-base attack on single anchor {anchor} not supported "
                f"(Tier 2 supports at most one fragment per anchor)"
            )

    for fragment in non_substrate:
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
                "logic error in _planar_face_placement"
            )

        direction = _plane_normal_at_anchor(positions, anchor, mol_h, substrate)
        if rotation_perturbation is not None:
            direction = rotation_perturbation @ direction
        target = positions[anchor] + direction * FRAGMENT_SEPARATION

        incoming = bridging_bond[1] if bridging_bond[0] == anchor else bridging_bond[0]
        positions[list(fragment)] += target - positions[incoming]

    return positions


# --- Phase 5 Tier 3 helpers ---


def _perpendicular_face_dir(
    axis: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """Return a unit vector perpendicular to axis, biased by offset.

    Strategy:
      1. axis を正規化。
      2. offset の axis-平行成分を除去 → offset_perp。
      3. ||offset_perp|| > 1e-6 なら正規化して返す。
      4. Fallback: 世界基底 [+z, +y, +x] を順に試し、axis と直交成分を持つ
         最初のものを正規化して返す。axis は unit vector なので最低 2 つは
         必ず非ゼロ垂直成分を持つ → fallback は必ず一意に決まる。

    Special case: if offset is the zero vector, its perpendicular component
    is also zero; falls through to the fallback chain (returns +z when axis
    is not parallel to +z, otherwise +y).
    """
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        raise ValueError("axis must be a non-zero vector")
    axis_unit = axis / axis_norm

    offset_perp = offset - float(np.dot(offset, axis_unit)) * axis_unit
    norm = float(np.linalg.norm(offset_perp))
    if norm > 1e-6:
        return offset_perp / norm

    for basis in (
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ):
        proj = float(np.dot(basis, axis_unit)) * axis_unit
        candidate = basis - proj
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm > 1e-6:
            return candidate / candidate_norm

    raise RuntimeError("could not find a perpendicular direction; axis is not a unit vector?")


def _kabsch_rigid_transform(
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal Procrustes (Kabsch) solve: src を dst に合わせる剛体変換 (R, t)。

    Centroid 中心化 → cross-covariance H = src_c.T @ dst_c → SVD → R = Vt.T @ D @ U.T
    where D = diag(1, 1, sign(det(Vt.T @ U.T))) で reflection を防ぐ。

    Returns (R: (3,3) rotation matrix with det>=0, t: (3,) translation vector).
    Raises ValueError if shapes mismatch or N < 2.

    Note: R is unique only when N >= 4 with points in general position.
    With N < 4 or near-collinear points, multiple rotations achieve the
    same minimum RMSD; the function returns one of them (chosen by
    np.linalg.svd's null-space convention). Phase 5 metathesis uses N=2
    and relies on cone perturbation in the caller to break the residual
    rotational ambiguity.
    """
    if src.shape != dst.shape:
        raise ValueError(f"src and dst must have same shape; got {src.shape} vs {dst.shape}")
    if src.ndim != 2 or src.shape[1] != 3 or src.shape[0] < 2:
        raise ValueError(f"expected (N>=2, 3) arrays; got {src.shape}")

    centroid_src = src.mean(axis=0)
    centroid_dst = dst.mean(axis=0)
    src_c = src - centroid_src
    dst_c = dst - centroid_dst

    H = src_c.T @ dst_c
    # Singular values (_S) are not needed; only the orthonormal U/Vt factors enter R.
    U, _S, Vt = np.linalg.svd(H)
    # Vt.T @ U.T is orthogonal so det is exactly +/-1; no zero-guard needed.
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = centroid_dst - R @ centroid_src
    return R, t


def _kabsch_alignment(
    mol_h: Chem.Mol,
    frag_indices: tuple[tuple[int, ...], ...],
    positions: np.ndarray,
    bond_changes: BondChanges,
    *,
    rotation_perturbation: np.ndarray | None,
) -> np.ndarray:
    """Tier 3 placement: 2-fragment 4-center metathesis を Kabsch alignment で配置。

    Algorithm (詳細は spec §3 参照):
      1. reference = _find_substrate_by_size(frag_indices), moving = もう一方
      2. broken_within_reference / broken_within_moving に分類。各 1 本ずつ前提。
      3. anchor pair = broken_within_reference の 2 endpoint。
      4. formed bonds 各本から (anchor_in_reference, incoming_in_moving) を抽出。
      5. perp_dir = anchor 軸の垂直方向 (moving 重心 offset から決定、fallback あり)。
      6. target_a/b = positions[anchor_a/b] + FRAGMENT_SEPARATION/2 × perp_dir
      7. _kabsch_rigid_transform(incoming_positions, target_positions) → (R, t)
      8. moving fragment 全体に (R, t) を適用 (positions = (R @ p.T).T + t)。
      9. rotation_perturbation が指定されていれば、target_midpoint 周りに moving fragment
         全体を追加回転 (cone trial 生成、Kabsch 後に独立適用)。

    Edge cases (Task 6 で error テストとして実装):
      - broken_within_reference または broken_within_moving が 1 本でない →
        NotImplementedError ("Phase 5 only supports 1+1 within-fragment broken bonds")
      - 2 incoming atoms が同一 (multi-bond from single anchor) →
        NotImplementedError ("multi-bond from single anchor in metathesis")

    Note: mol_h is accepted for API symmetry with Tier 1 (_directional_placement)
    and Tier 2 (_planar_face_placement); Tier 3 operates on atom-index sets only
    and does not consult the RDKit molecule.
    """
    reference = _find_substrate_by_size(frag_indices)
    moving = next(f for f in frag_indices if f is not reference)
    reference_set = set(reference)
    moving_set = set(moving)

    broken_within_reference = [
        (a, b) for a, b in bond_changes.broken
        if a in reference_set and b in reference_set
    ]
    broken_within_moving = [
        (a, b) for a, b in bond_changes.broken
        if a in moving_set and b in moving_set
    ]
    if len(broken_within_reference) != 1 or len(broken_within_moving) != 1:
        raise NotImplementedError(
            "Phase 5 only supports 1+1 within-fragment broken bonds; "
            f"got broken_within_reference={broken_within_reference}, "
            f"broken_within_moving={broken_within_moving}"
        )

    anchor_a, anchor_b = sorted(broken_within_reference[0])

    anchor_to_incoming: dict[int, int] = {}
    for a, b in bond_changes.formed:
        if a in reference_set and b in moving_set:
            anchor, incoming = a, b
        elif b in reference_set and a in moving_set:
            anchor, incoming = b, a
        else:
            raise ValueError(
                f"formed bond ({a}, {b}) does not bridge reference↔moving; "
                f"check bond_changes (formed={bond_changes.formed})"
            )
        if anchor not in (anchor_a, anchor_b):
            raise ValueError(
                f"formed bond anchor {anchor} is not in anchor pair "
                f"({anchor_a}, {anchor_b}) from broken_within_reference"
            )
        if anchor in anchor_to_incoming:
            raise NotImplementedError(
                f"multi-bond from single anchor in metathesis (anchor={anchor})"
            )
        anchor_to_incoming[anchor] = incoming

    if anchor_a not in anchor_to_incoming or anchor_b not in anchor_to_incoming:
        raise ValueError(
            f"each anchor in pair ({anchor_a}, {anchor_b}) must have one formed bond; "
            f"got {anchor_to_incoming}"
        )
    incoming_a = anchor_to_incoming[anchor_a]
    incoming_b = anchor_to_incoming[anchor_b]

    if incoming_a == incoming_b:
        raise NotImplementedError(
            f"multi-bond from single anchor in metathesis "
            f"(two reference anchors target the same moving atom {incoming_a})"
        )

    axis = positions[anchor_a] - positions[anchor_b]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        raise RuntimeError(
            "anchor pair coincident after MMFF — broken-within-reference bond is broken"
        )
    axis_unit = axis / axis_norm

    moving_indices = list(moving)
    moving_centroid = positions[moving_indices].mean(axis=0)
    anchor_midpoint = (positions[anchor_a] + positions[anchor_b]) / 2.0
    offset = moving_centroid - anchor_midpoint
    perp_dir = _perpendicular_face_dir(axis_unit, offset)

    target_a = positions[anchor_a] + (FRAGMENT_SEPARATION / 2.0) * perp_dir
    target_b = positions[anchor_b] + (FRAGMENT_SEPARATION / 2.0) * perp_dir

    src = np.stack([positions[incoming_a], positions[incoming_b]])
    dst = np.stack([target_a, target_b])
    R, t = _kabsch_rigid_transform(src, dst)

    # Apply Kabsch (R, t) to the moving fragment as a rigid body.
    moving_pos = positions[moving_indices]
    moving_pos = (R @ moving_pos.T).T + t

    # Cone perturbation: rotate the just-placed moving fragment around the
    # target midpoint. This samples orientations of the same 4-center geometry
    # without disturbing target_a, target_b (which stay fixed in reference frame).
    if rotation_perturbation is not None:
        target_midpoint = (target_a + target_b) / 2.0
        moving_pos = (
            (rotation_perturbation @ (moving_pos - target_midpoint).T).T
            + target_midpoint
        )

    positions[moving_indices] = moving_pos
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
            elems = ", ".join(sorted({a.GetSymbol() for a in frag.GetAtoms()}))
            log.warning(
                "MMFF94 cannot parameterize fragment (atoms: %s); using ETKDG "
                "geometry without MMFF refinement", elems,
            )
            # ETKDG already produced a 3D conformation; downstream UMA relaxation
            # will provide energetic refinement.
