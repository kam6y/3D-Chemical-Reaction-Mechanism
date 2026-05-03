# Phase 5 Multi-Substrate Metathesis (4-Center Tier 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 4 で `embed3d._place_fragments` の最終分岐に残された `NotImplementedError("multi-substrate metathesis ...")` を、2 fragments / formed=2 / broken=2 の 4-center metathesis に限定して実装する。例反応 `CH₃Cl + LiBr → CH₃Br + LiCl`。

**Architecture:** `embed3d._place_fragments` の dispatcher に Tier 3 分岐を追加。新規ヘルパは `_kabsch_rigid_transform` (orthogonal Procrustes via SVD)、`_perpendicular_face_dir` (anchor 軸の垂直 face 方向計算 + 世界基底 fallback)、`_kabsch_alignment` (本体)。target 配置は anchor 軸 (broken_within_reference の 2 endpoint) に垂直な face direction に `FRAGMENT_SEPARATION/2 = 1.75 Å` 持ち上げ、moving fragment の 2 incoming atoms を Kabsch で対応点 fit。`metathesis_4center` preset を新設、CLI / NEB refine guard / unimolecular auto-clamp は既存挙動を維持。

**Tech Stack:** Python 3.11, RDKit, ASE, NumPy (SVD via `np.linalg.svd`), pytest, UMA-m-1p1 (slow integration test only)

**Spec:** `docs/superpowers/specs/2026-05-03-phase-5-metathesis-design.md`

**Branch:** `phase-5` (already created from `develop`, spec already committed at `11a406e`)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `reactx/embed3d.py` | Modify | Tier 3 dispatch + 3 new helpers (`_kabsch_rigid_transform`, `_perpendicular_face_dir`, `_kabsch_alignment`) |
| `reactx/presets.py` | Modify | Add `metathesis_4center` preset |
| `examples/metathesis_4center.rxn` | Create | CH₃Cl + LiBr → CH₃Br + LiCl の MOL V2000 |
| `tests/conftest.py` | Modify | Add `metathesis_rxn_path` and `metathesis_atoms_setup` fixtures |
| `tests/test_embed3d_placement.py` | Modify | Tier 3 ユニットテスト 9 ケース追加 (Kabsch / perp_dir / alignment / dispatch / errors) |
| `tests/test_presets.py` | Modify | `metathesis_4center` preset assertion + `PRESETS` キー集合更新 |
| `tests/test_rxn_parser.py` | Modify | metathesis .rxn parse smoke test |
| `tests/test_re5_metathesis.py` | Create | slow E2E 統合テスト |
| `tests/test_cli_neb_refine_guard.py` | Modify | metathesis ケース追加 |
| `README.md` | Modify | preset 表に `metathesis_4center` 行 + Phase 5 DoD セクション + wall-clock 表に metathesis 行 + 対応反応一覧 + Phase tag header |

---

## Task 1: `_kabsch_rigid_transform` core helper

**Files:**
- Modify: `reactx/embed3d.py`
- Modify: `tests/test_embed3d_placement.py`

- [ ] **Step 1: Write failing test for forward → inverse identity**

`tests/test_embed3d_placement.py` の末尾に追加:

```python
def test_kabsch_rigid_transform_recovers_known_rotation_translation():
    """既知の (R, t) を src に適用した dst から、Kabsch が同じ (R, t) を回復する。"""
    from reactx.embed3d import _kabsch_rigid_transform

    rng = np.random.default_rng(42)
    src = rng.normal(size=(4, 3))
    # 既知の回転 (z 軸周り 30°) と translation
    theta = np.deg2rad(30.0)
    R_true = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ])
    t_true = np.array([1.5, -2.0, 0.5])
    dst = (R_true @ src.T).T + t_true

    R, t = _kabsch_rigid_transform(src, dst)
    np.testing.assert_allclose(R, R_true, atol=1e-9)
    np.testing.assert_allclose(t, t_true, atol=1e-9)


def test_kabsch_rigid_transform_rejects_reflection():
    """rotoinversion を解にしてしまう対応点でも det(R) >= 0 の rotation を返す。"""
    from reactx.embed3d import _kabsch_rigid_transform

    # 鏡面反転を要求する明示的な対応点 (xy 平面で z 反転を要求)
    src = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    # dst は src の x, y は同じだが z を反転 (= 鏡面反転)
    dst = np.array([
        [1.0, 0.0,  0.0],
        [0.0, 1.0,  0.0],
        [0.0, 0.0, -1.0],
    ])
    R, _t = _kabsch_rigid_transform(src, dst)
    det = float(np.linalg.det(R))
    assert det >= 0, f"expected proper rotation (det>=0), got det={det}"
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_embed3d_placement.py::test_kabsch_rigid_transform_recovers_known_rotation_translation -v
```

Expected: FAIL with `ImportError: cannot import name '_kabsch_rigid_transform'`

- [ ] **Step 3: Write minimal implementation**

`reactx/embed3d.py` の末尾の `_embed_in_place` 関数の手前 (= ファイル末尾の `def _embed_in_place(...)` 定義の前) に追加:

```python
def _kabsch_rigid_transform(
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal Procrustes (Kabsch) solve: src を dst に合わせる剛体変換 (R, t)。

    Centroid 中心化 → cross-covariance H = src_c.T @ dst_c → SVD → R = Vt.T @ D @ U.T
    where D = diag(1, 1, sign(det(Vt.T @ U.T))) で reflection を防ぐ。

    Returns (R: (3,3) rotation matrix with det>=0, t: (3,) translation vector).
    Raises ValueError if shapes mismatch or N < 2.
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
    U, _S, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    if d == 0.0:
        d = 1.0
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = centroid_dst - R @ centroid_src
    return R, t
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_embed3d_placement.py::test_kabsch_rigid_transform_recovers_known_rotation_translation tests/test_embed3d_placement.py::test_kabsch_rigid_transform_rejects_reflection -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add reactx/embed3d.py tests/test_embed3d_placement.py
git commit -m "$(cat <<'EOF'
feat(embed3d): add _kabsch_rigid_transform Procrustes solver

Phase 5 Tier 3 placement の核心: 2 fragments の対応点ペアから最小二乗で
剛体変換 (R, t) を求める。SVD via np.linalg.svd, sign correction で
reflection を防止 (det(R) >= 0)。N=2 で 4-center metathesis、N>=3 で
将来の cycloaddition 等にも流用可能。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `_perpendicular_face_dir` helper

**Files:**
- Modify: `reactx/embed3d.py`
- Modify: `tests/test_embed3d_placement.py`

- [ ] **Step 1: Write failing tests**

`tests/test_embed3d_placement.py` の末尾に追加:

```python
def test_perpendicular_face_dir_normal_case():
    """offset が axis に垂直成分を持つとき、その方向に正規化された unit vector を返す。"""
    from reactx.embed3d import _perpendicular_face_dir
    axis = np.array([1.0, 0.0, 0.0])
    offset = np.array([0.0, 2.0, 0.0])
    perp = _perpendicular_face_dir(axis, offset)
    np.testing.assert_allclose(perp, [0.0, 1.0, 0.0], atol=1e-9)


def test_perpendicular_face_dir_axis_aligned_offset_falls_back_to_z():
    """offset が axis と平行 (垂直成分なし) のとき +z fallback (axis が +x のとき)。"""
    from reactx.embed3d import _perpendicular_face_dir
    axis = np.array([1.0, 0.0, 0.0])
    offset = np.array([2.0, 0.0, 0.0])  # axis と平行
    perp = _perpendicular_face_dir(axis, offset)
    # +z は axis (=+x) と直交するので採用される
    np.testing.assert_allclose(perp, [0.0, 0.0, 1.0], atol=1e-9)


def test_perpendicular_face_dir_axis_z_skips_z_uses_y_fallback():
    """axis が +z のとき、世界基底 +z は使えないので +y にフォールバック。"""
    from reactx.embed3d import _perpendicular_face_dir
    axis = np.array([0.0, 0.0, 1.0])
    offset = np.array([0.0, 0.0, 2.0])  # axis と平行
    perp = _perpendicular_face_dir(axis, offset)
    # +z は axis と平行 → スキップ、+y は axis と直交 → 採用
    np.testing.assert_allclose(perp, [0.0, 1.0, 0.0], atol=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_embed3d_placement.py -k _perpendicular_face_dir -v
```

Expected: FAIL with `ImportError: cannot import name '_perpendicular_face_dir'`

- [ ] **Step 3: Write minimal implementation**

`reactx/embed3d.py` の `_kabsch_rigid_transform` の **前** (Task 1 で追加した位置の手前) に追加:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_embed3d_placement.py -k _perpendicular_face_dir -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add reactx/embed3d.py tests/test_embed3d_placement.py
git commit -m "$(cat <<'EOF'
feat(embed3d): add _perpendicular_face_dir helper for Tier 3

axis 軸と垂直な face direction を offset の垂直成分から計算し、degenerate
ケース (offset が axis と平行) は世界基底 [+z, +y, +x] の順で fallback
する deterministic ヘルパ。Phase 5 Kabsch alignment が anchor 軸からの
target 持ち上げ方向を決めるのに使う。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `metathesis_atoms_setup` fixture + smoke test

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_embed3d_placement.py`

- [ ] **Step 1: Add fixture to conftest.py**

`tests/conftest.py` の `sn1_recomb_atoms_setup` fixture (line 88-123) の末尾に追加:

```python
@pytest.fixture()
def metathesis_atoms_setup():
    """4-center metathesis setup: CH3Cl + LiBr.

    formed = ((C, Br), (Li, Cl))
    broken = ((C, Cl), (Li, Br))

    Layout:
      - Cl at (0, 0, 0), C at (1.78, 0, 0) — anchor pair on x axis
      - 3 H around C tetrahedrally on +x side
      - LiBr centroid offset to (0.89, 2.0, 0) so perp_dir resolves to +y
        (Br at (1.99, 2.0, 0), Li at (-0.21, 2.0, 0))
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("CCl.[Li]Br"))
    Chem.SanitizeMol(mol)
    frag_indices = Chem.GetMolFrags(mol)
    n = mol.GetNumAtoms()
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    li_idx = syms.index("Li")
    br_idx = syms.index("Br")

    positions = np.zeros((n, 3))
    positions[cl_idx] = (0.0, 0.0, 0.0)
    positions[c_idx] = (1.78, 0.0, 0.0)
    h_atoms = [
        a.GetIdx() for a in mol.GetAtomWithIdx(c_idx).GetNeighbors()
        if a.GetSymbol() == "H"
    ]
    for k, h in enumerate(h_atoms):
        theta = 2 * np.pi * k / 3
        positions[h] = (1.78 + 0.5, np.cos(theta) * 0.9, np.sin(theta) * 0.9)
    # LiBr (Br-Li ~2.2 Å) above CCl axis midpoint (0.89, 0, 0), offset +y
    positions[br_idx] = (1.99, 2.0, 0.0)
    positions[li_idx] = (-0.21, 2.0, 0.0)

    bc = BondChanges(
        formed=((c_idx, br_idx), (li_idx, cl_idx)),
        broken=((c_idx, cl_idx), (li_idx, br_idx)),
    )
    return mol, frag_indices, positions, bc
```

- [ ] **Step 2: Smoke test the fixture**

`tests/test_embed3d_placement.py` の末尾に追加:

```python
def test_metathesis_fixture_shape(metathesis_atoms_setup):
    """fixture の形状確認: 2 fragments, formed=2, broken=2, anchor pair on x axis."""
    mol_h, frag_indices, positions, bc = metathesis_atoms_setup
    assert len(frag_indices) == 2
    assert len(bc.formed) == 2
    assert len(bc.broken) == 2
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    li_idx = syms.index("Li")
    br_idx = syms.index("Br")
    np.testing.assert_allclose(positions[cl_idx], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(positions[c_idx], [1.78, 0.0, 0.0], atol=1e-9)
    # broken bonds 各 fragment 内に閉じる
    cccl = next(i for i, b in enumerate(bc.broken) if c_idx in b and cl_idx in b)
    libr = next(i for i, b in enumerate(bc.broken) if li_idx in b and br_idx in b)
    assert cccl != libr  # 別々の broken bond
```

- [ ] **Step 3: Run smoke test**

```
pytest tests/test_embed3d_placement.py::test_metathesis_fixture_shape -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_embed3d_placement.py
git commit -m "$(cat <<'EOF'
test(conftest): add metathesis_atoms_setup fixture

Phase 5 Tier 3 placement テスト用の CH3Cl + LiBr fixture。Cl を原点, C を
+x に置き anchor pair を x 軸に並べる。LiBr は +y 方向の (0.89, 2.0, 0)
centroid 周辺に配置 (perp_dir が +y に確実に resolve する初期条件)。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `_kabsch_alignment` main path (happy case)

**Files:**
- Modify: `reactx/embed3d.py`
- Modify: `tests/test_embed3d_placement.py`

- [ ] **Step 1: Write failing test**

`tests/test_embed3d_placement.py` の末尾に追加:

```python
def test_kabsch_alignment_creates_4center_geometry(metathesis_atoms_setup):
    """Tier 3 主路: CH3Cl + LiBr fixture で 4-center geometry を達成。

    Assertions (spec §7 (q)):
      - C-Br ≤ 2.5 Å (formed bond)
      - Li-Cl ≤ 2.5 Å (formed bond)
      - anchor 軸 (C-Cl) と incoming 軸 (Br-Li) が概並行 (cos angle > 0.7)
      - moving 重心が anchor 軸から FRAGMENT_SEPARATION/2 ± 0.5 Å 離れる
    """
    from reactx.embed3d import FRAGMENT_SEPARATION, _kabsch_alignment

    mol_h, frag_indices, positions, bc = metathesis_atoms_setup
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    li_idx = syms.index("Li")
    br_idx = syms.index("Br")

    out = _kabsch_alignment(
        mol_h, frag_indices, positions.copy(), bc,
        rotation_perturbation=None,
    )

    d_c_br = float(np.linalg.norm(out[c_idx] - out[br_idx]))
    d_li_cl = float(np.linalg.norm(out[li_idx] - out[cl_idx]))
    assert d_c_br <= 2.5, f"C-Br should be <=2.5 A (formed bond), got {d_c_br:.2f}"
    assert d_li_cl <= 2.5, f"Li-Cl should be <=2.5 A (formed bond), got {d_li_cl:.2f}"

    axis_anchor = out[c_idx] - out[cl_idx]
    axis_anchor /= np.linalg.norm(axis_anchor)
    axis_incoming = out[br_idx] - out[li_idx]
    axis_incoming /= np.linalg.norm(axis_incoming)
    cos_angle = abs(float(np.dot(axis_anchor, axis_incoming)))
    assert cos_angle > 0.7, f"anchor and incoming axes should be ~parallel, got cos={cos_angle:.3f}"

    moving_frag = next(f for f in frag_indices if li_idx in f)
    moving_centroid = out[list(moving_frag)].mean(axis=0)
    anchor_midpoint = (out[c_idx] + out[cl_idx]) / 2
    perp_dist = float(np.linalg.norm(
        (moving_centroid - anchor_midpoint)
        - np.dot(moving_centroid - anchor_midpoint, axis_anchor) * axis_anchor
    ))
    expected = FRAGMENT_SEPARATION / 2
    assert abs(perp_dist - expected) <= 0.5, (
        f"moving centroid should be {expected:.2f} A above anchor axis (perp), got {perp_dist:.2f}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_embed3d_placement.py::test_kabsch_alignment_creates_4center_geometry -v
```

Expected: FAIL with `ImportError: cannot import name '_kabsch_alignment'`

- [ ] **Step 3: Write minimal implementation**

`reactx/embed3d.py` の `_planar_face_placement` 関数 (line 329-408 付近、Tier 2) の **直後** に追加:

```python
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
      8. R = rotation_perturbation @ R (cone 散らし)
      9. moving fragment 全体に R, t を centroid 経由で適用。

    Edge cases (Task 6 で error テストとして実装):
      - broken_within_reference または broken_within_moving が 1 本でない →
        NotImplementedError ("Phase 5 only supports 1+1 within-fragment broken bonds")
      - 2 incoming atoms が同一 (multi-bond from single anchor) →
        NotImplementedError ("multi-bond from single anchor in metathesis")
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
            f"(both formed bonds share moving atom {incoming_a})"
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
```

注: Kabsch の `(R, t)` を moving fragment 全体に `R @ p + t` で適用 (centroid 経由の冗長な中間ステップは省略)。`rotation_perturbation` は Kabsch 後の moving fragment を target midpoint 周りに回転して cone of trials を生成する (Tier 1/2 のように direction vector に乗算するのではなく、配置済み fragment を独立に回転)。

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_embed3d_placement.py::test_kabsch_alignment_creates_4center_geometry -v
```

Expected: PASS

- [ ] **Step 5: Run all placement regression tests to ensure Tier 1/2 unchanged**

```
pytest tests/test_embed3d_placement.py -v
```

Expected: All previously-passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add reactx/embed3d.py tests/test_embed3d_placement.py
git commit -m "$(cat <<'EOF'
feat(embed3d): add _kabsch_alignment Tier 3 placement (happy path)

Phase 5 metathesis 配置: 2-fragment 4-center で reference (重原子最大)
固定、anchor pair = broken_within_reference の 2 endpoint、target を
anchor pair の垂直方向に FRAGMENT_SEPARATION/2 持ち上げて Kabsch で
moving 側 incoming 2 atoms を fit。エラー分岐 (1+1 broken topology
violation, multi-bond from single anchor) は本 commit で raise 済み、
専用テストは Task 6 で追加。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `_kabsch_alignment` rotation_perturbation handling

**Files:**
- Modify: `tests/test_embed3d_placement.py`

- [ ] **Step 1: Write failing test**

`tests/test_embed3d_placement.py` の末尾に追加:

```python
def test_kabsch_alignment_respects_rotation_perturbation(metathesis_atoms_setup):
    """rotation_perturbation で moving fragment が回転される (相対距離は不変)。"""
    from reactx.embed3d import _kabsch_alignment

    mol_h, frag_indices, positions, bc = metathesis_atoms_setup
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    li_idx = syms.index("Li")
    br_idx = syms.index("Br")

    out_id = _kabsch_alignment(
        mol_h, frag_indices, positions.copy(), bc,
        rotation_perturbation=None,
    )

    # 90° rotation around z axis (incoming axis Br-Li is along x in this fixture,
    # so x rotation would not move the atoms — must rotate around y or z).
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    out_rot = _kabsch_alignment(
        mol_h, frag_indices, positions.copy(), bc,
        rotation_perturbation=R,
    )

    # Br と Li の絶対位置が変わる
    assert not np.allclose(out_id[br_idx], out_rot[br_idx], atol=0.1), (
        f"rotation_perturbation should change Br position; "
        f"identity={out_id[br_idx]}, rotated={out_rot[br_idx]}"
    )
    # 相対距離 (Br-Li) は剛体変換で不変
    d_id = float(np.linalg.norm(out_id[br_idx] - out_id[li_idx]))
    d_rot = float(np.linalg.norm(out_rot[br_idx] - out_rot[li_idx]))
    np.testing.assert_allclose(d_id, d_rot, atol=0.05)
```

- [ ] **Step 2: Run test to verify it passes (already implemented in Task 4)**

```
pytest tests/test_embed3d_placement.py::test_kabsch_alignment_respects_rotation_perturbation -v
```

Expected: PASS — Task 4 の `rotation_perturbation @ R` ロジックがそのままカバー。

- [ ] **Step 3: Commit**

```bash
git add tests/test_embed3d_placement.py
git commit -m "$(cat <<'EOF'
test(embed3d): cover rotation_perturbation in _kabsch_alignment

Task 4 で既に実装済みの rotation_perturbation @ R 経路を、識別ベクトルと
90° 回転で Br/Li 位置が動き相対距離が保存されることで検証。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `_kabsch_alignment` error branches

**Files:**
- Modify: `tests/test_embed3d_placement.py`

- [ ] **Step 1: Write failing tests**

`tests/test_embed3d_placement.py` の末尾に追加:

```python
def test_kabsch_alignment_rejects_broken_spanning_fragments():
    """broken bonds が両 fragment を跨ぐ (= 純粋 metathesis 以外) で NotImplementedError。"""
    from reactx.embed3d import _kabsch_alignment

    # 2 fragments で broken bond が両 frag を跨ぐ (cross-fragment broken)
    mol = Chem.AddHs(Chem.MolFromSmiles("CC.OO"))
    frags = Chem.GetMolFrags(mol)
    a = frags[0][0]  # 左 fragment の C
    b = frags[1][0]  # 右 fragment の O
    # broken=2 だが両方 cross-fragment → broken_within_* がいずれも 0
    bc = BondChanges(
        formed=((frags[0][1], frags[1][1]), (a, b)),
        broken=((a, b), (frags[0][1], frags[1][1])),
    )
    with pytest.raises(NotImplementedError, match="1\\+1 within-fragment"):
        _kabsch_alignment(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )


def test_kabsch_alignment_rejects_multi_bond_from_single_anchor(metathesis_atoms_setup):
    """同 anchor から 2 formed bond で NotImplementedError。"""
    from reactx.embed3d import _kabsch_alignment

    mol_h, frag_indices, positions, bc = metathesis_atoms_setup
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    li_idx = syms.index("Li")
    br_idx = syms.index("Br")
    # 両 formed bond の reference 端を C にする (anchor=C 共有)
    bad_bc = BondChanges(
        formed=((c_idx, br_idx), (c_idx, li_idx)),
        broken=((c_idx, cl_idx), (li_idx, br_idx)),
    )
    with pytest.raises(NotImplementedError, match="multi-bond from single anchor"):
        _kabsch_alignment(
            mol_h, frag_indices, positions.copy(), bad_bc,
            rotation_perturbation=None,
        )
```

- [ ] **Step 2: Run tests to verify they pass (already implemented in Task 4)**

```
pytest tests/test_embed3d_placement.py -k "rejects_broken_spanning or rejects_multi_bond_from_single_anchor" -v
```

Expected: 2 passed — Task 4 の error branch ロジックがそのままカバー。

- [ ] **Step 3: Commit**

```bash
git add tests/test_embed3d_placement.py
git commit -m "$(cat <<'EOF'
test(embed3d): cover _kabsch_alignment error branches

(1) broken bonds が両 fragment を跨ぐ (broken_within_* が 1+1 でない)
    で NotImplementedError("Phase 5 only supports 1+1 within-fragment ...")
(2) 同 anchor から 2 formed bond (multi-bond from single anchor) で
    NotImplementedError("multi-bond from single anchor in metathesis")

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `_place_fragments` Tier 3 dispatch

**Files:**
- Modify: `reactx/embed3d.py`
- Modify: `tests/test_embed3d_placement.py`

- [ ] **Step 1: Write failing tests**

`tests/test_embed3d_placement.py` の末尾に追加:

```python
def test_place_fragments_dispatches_tier3_for_metathesis(metathesis_atoms_setup):
    """Tier 3 dispatch: 2-fragment formed=2/broken=2/multi-substrate で _kabsch_alignment に流れる。"""
    from reactx.embed3d import _place_fragments

    mol_h, frag_indices, positions, bc = metathesis_atoms_setup
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    c_idx = syms.index("C")
    br_idx = syms.index("Br")

    out = _place_fragments(
        mol_h, frag_indices, positions.copy(), bc,
        rotation_perturbation=None,
    )
    # Tier 3 を経由して 4-center 配置になっていれば C-Br が中程度の距離
    d_c_br = float(np.linalg.norm(out[c_idx] - out[br_idx]))
    assert 0.5 < d_c_br < 3.0, (
        f"expected Tier 3 placement to put Br near C (0.5-3.0 A), got {d_c_br:.2f}"
    )


def test_place_fragments_rejects_3_fragment_metathesis():
    """3 fragments + broken bond が fragments を跨ぐ場合は Phase 6+ reject。

    Tier 1 dispatch には乗らず (substrate=None: broken が単一 fragment に閉じない)、
    Tier 3 にも乗らない (frag_count==2 が条件) ので最終 raise に到達。
    """
    from reactx.embed3d import _place_fragments

    # 3 fragments: CC, OO, NN。broken bond が CC ↔ OO を跨ぐ → substrate=None。
    # 3 fragments なので Tier 3 dispatch (frag_count==2) も満たさず final raise。
    mol = Chem.AddHs(Chem.MolFromSmiles("CC.OO.NN"))
    frags = Chem.GetMolFrags(mol)
    c0 = frags[0][0]
    o0 = frags[1][0]
    n0 = frags[2][0]
    bc = BondChanges(
        formed=((c0, n0),),
        broken=((c0, o0),),
    )
    with pytest.raises(NotImplementedError, match="Phase 5|Phase 6|multi-substrate"):
        _place_fragments(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )


def test_place_fragments_rejects_asymmetric_metathesis():
    """非対称 metathesis (formed=2, broken=1, 両 fragment 跨ぎ) で Phase 6+ reject。"""
    from reactx.embed3d import _place_fragments

    # 2 fragments で broken=1 が両 frag を跨ぐ (substrate=None かつ broken count != formed count)
    mol = Chem.AddHs(Chem.MolFromSmiles("CC.OO"))
    frags = Chem.GetMolFrags(mol)
    a = frags[0][0]
    b = frags[1][0]
    bc = BondChanges(
        formed=((frags[0][1], frags[1][1]), (a, b)),
        broken=((a, b),),
    )
    with pytest.raises(NotImplementedError, match="Phase 5|Phase 6"):
        _place_fragments(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )
```

- [ ] **Step 2: Run first test to verify it fails**

```
pytest tests/test_embed3d_placement.py::test_place_fragments_dispatches_tier3_for_metathesis -v
```

Expected: FAIL — まだ `_place_fragments` の Tier 3 分岐が無いので最終 raise (`NotImplementedError("multi-substrate metathesis ... Phase 5+")`) で拒絶される。

- [ ] **Step 3: Wire Tier 3 into dispatcher**

`reactx/embed3d.py` の `_place_fragments` 関数 (line 96-134) を以下で置き換え:

```python
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
```

- [ ] **Step 4: Update existing test that asserted `_place_fragments` rejects metathesis**

既存 `tests/test_embed3d_placement.py::test_place_fragments_still_raises_for_multi_substrate_metathesis_after_tier2` (line 105-117) と `test_place_fragments_raises_for_multi_substrate_metathesis` (line 120-130) を以下で置き換え:

```python
def test_place_fragments_still_rejects_3frag_or_asymmetric_after_tier3():
    """Tier 3 後も 3 fragments や 非対称 metathesis は依然として reject される。

    Phase 4 までは「broken bonds が複数 frag を跨ぐ」全ケースを reject していたが、
    Phase 5 で 2-frag formed=2/broken=2 のみ Tier 3 が拾うようになった。残るは
    Tier 3 条件を満たさない multi-substrate ケース。
    """
    from reactx.embed3d import _place_fragments
    mol = Chem.AddHs(Chem.MolFromSmiles("CC.OO"))
    frags = Chem.GetMolFrags(mol)
    a = frags[0][0]
    b = frags[1][0]
    # formed=1 + broken=1 (両方 cross-fragment) — Tier 3 は formed=2/broken=2 のみ
    bc = BondChanges(formed=((frags[0][1], frags[1][1]),), broken=((a, b),))
    with pytest.raises(NotImplementedError, match="Phase 5|Phase 6|multi-substrate"):
        _place_fragments(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )
```

- [ ] **Step 5: Run all placement tests**

```
pytest tests/test_embed3d_placement.py -v
```

Expected: All passed — 既存 Tier 1/2 テスト + Task 4-6 + Task 7 の新規 dispatch テスト。

- [ ] **Step 6: Run Tier 1/2 regression tests explicitly**

```
pytest tests/test_embed3d_placement.py::test_place_fragments_dispatch_directional_for_sn2_shape tests/test_embed3d_placement.py::test_place_fragments_e2_shape_directional_anchors_on_h tests/test_embed3d_placement.py::test_place_fragments_dispatches_tier2_for_broken_zero_bimolecular -v
```

Expected: 3 passed (Tier 1/2 dispatch 不変)

- [ ] **Step 7: Commit**

```bash
git add reactx/embed3d.py tests/test_embed3d_placement.py
git commit -m "$(cat <<'EOF'
feat(embed3d): dispatch Tier 3 for 2-fragment 4-center metathesis

Phase 4 で NotImplementedError を投げていた最終分岐に、formed=2 / broken=2 /
2 fragments の条件で _kabsch_alignment に流す Tier 3 dispatch を追加。
Tier 1/2 (E2 / SN2 / PT / Menshutkin / SN1 step1/2) の挙動は不変。
3 fragments や 非対称 metathesis は引き続き Phase 6+ として reject。
既存の "multi-substrate metathesis ... Phase 5+" を主張するテストを
新しい契約 (Tier 3 が拾う / 残りは Phase 6+) に合わせて更新。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `metathesis_4center` preset

**Files:**
- Modify: `reactx/presets.py`
- Modify: `tests/test_presets.py`

- [ ] **Step 1: Write failing test**

`tests/test_presets.py` の末尾に追加:

```python
def test_metathesis_4center_preset_values():
    p = get_preset("metathesis_4center")
    assert p.name == "metathesis_4center"
    assert p.k_form == 1.0
    assert p.k_broken == 1.0
    assert p.r_broken == 4.0
    assert p.max_relax_steps == 200
    assert p.r_form is None  # 元素表 (Cordero: C-Br ≈ 1.94, Li-Cl ≈ 2.02)
```

そして既存の `test_presets_dict_keys` (line 47-55) を以下で置き換え:

```python
def test_presets_dict_keys():
    assert set(PRESETS) == {
        "sn2_anion",
        "proton_transfer",
        "menshutkin",
        "e2",
        "sn1_dissoc",
        "sn1_recomb",
        "metathesis_4center",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_presets.py::test_metathesis_4center_preset_values tests/test_presets.py::test_presets_dict_keys -v
```

Expected: FAIL — `Unknown reaction-type preset 'metathesis_4center'` and キー集合不一致

- [ ] **Step 3: Add preset**

`reactx/presets.py` の `PRESETS["sn1_recomb"] = ...` (line 62-69) の **直後** に追加:

```python
PRESETS["metathesis_4center"] = ReactionPreset(
    name="metathesis_4center",
    k_form=1.0,
    k_broken=1.0,
    r_broken=4.0,
    max_relax_steps=200,
    r_form=None,     # 元素表 (Cordero: C-Br ≈ 1.94, Li-Cl ≈ 2.02)
)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_presets.py -v
```

Expected: All passed (既存 + 新規 1 件 + キー集合更新)

- [ ] **Step 5: Commit**

```bash
git add reactx/presets.py tests/test_presets.py
git commit -m "$(cat <<'EOF'
feat(presets): add metathesis_4center preset for Phase 5

4-center metathesis (formed=2 / broken=2) 向け: k_form=k_broken=1.0
(E2 と同等、4 本同時に効くので過大にしない)、r_broken=4.0 (default)、
max_relax_steps=200 (multi-bond は収束遅め)、r_form=None で元素表
fallback (Cordero C-Br ≈ 1.94, Li-Cl ≈ 2.02)。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `examples/metathesis_4center.rxn` + parse smoke test

**Files:**
- Create: `examples/metathesis_4center.rxn`
- Modify: `tests/conftest.py`
- Modify: `tests/test_rxn_parser.py`

- [ ] **Step 1: Add fixture to conftest.py**

`tests/conftest.py` の `sn1_recomb_rxn_path` fixture (line 36-38) の **直後** に追加:

```python
@pytest.fixture()
def metathesis_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "metathesis_4center.rxn"
```

- [ ] **Step 2: Write failing parse smoke test**

`tests/test_rxn_parser.py` の末尾に追加:

```python
def test_parse_metathesis_4center_rxn(metathesis_rxn_path):
    """examples/metathesis_4center.rxn が parse でき、formed=2 / broken=2 が抽出される。"""
    from rdkit import Chem

    from reactx.bond_changes import compute_bond_changes
    from reactx.rxn_parser import parse_rxn

    r_mol, p_mol, mapping = parse_rxn(metathesis_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_bond_changes(r_h, p_h, mapping)

    assert len(bc.formed) == 2, f"expected 2 formed bonds, got {bc.formed}"
    assert len(bc.broken) == 2, f"expected 2 broken bonds, got {bc.broken}"
    assert len(Chem.GetMolFrags(r_h)) == 2, "reactant should have 2 fragments"
    assert len(Chem.GetMolFrags(p_h)) == 2, "product should have 2 fragments"

    syms = [a.GetSymbol() for a in r_h.GetAtoms()]
    formed_pair_syms = {frozenset({syms[a], syms[b]}) for a, b in bc.formed}
    broken_pair_syms = {frozenset({syms[a], syms[b]}) for a, b in bc.broken}
    assert frozenset({"C", "Br"}) in formed_pair_syms, f"expected C-Br formed; got {formed_pair_syms}"
    assert frozenset({"Li", "Cl"}) in formed_pair_syms, f"expected Li-Cl formed; got {formed_pair_syms}"
    assert frozenset({"C", "Cl"}) in broken_pair_syms, f"expected C-Cl broken; got {broken_pair_syms}"
    assert frozenset({"Li", "Br"}) in broken_pair_syms, f"expected Li-Br broken; got {broken_pair_syms}"
```

- [ ] **Step 3: Run test to verify it fails**

```
pytest tests/test_rxn_parser.py::test_parse_metathesis_4center_rxn -v
```

Expected: FAIL — `examples/metathesis_4center.rxn` がまだ無い

- [ ] **Step 4: Create the example file**

`examples/metathesis_4center.rxn` を以下の内容で作成 (4 ブロック: reactant 1 = CH3Cl / reactant 2 = LiBr / product 1 = CH3Br / product 2 = LiCl, atom map 1=C / 2=Cl / 3=Li / 4=Br):

```
$RXN

      RDKit

  2  2
$MOL

     RDKit          2D

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  1  0  0
    1.7800    0.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  2  0  0
  1  2  1  0
M  END
$MOL

     RDKit          2D

  2  1  0  0  0  0  0  0  0  0999 V2000
    3.0000    2.0000    0.0000 Li  0  0  0  0  0  0  0  0  0  3  0  0
    5.2000    2.0000    0.0000 Br  0  0  0  0  0  0  0  0  0  4  0  0
  1  2  1  0
M  END
$MOL

     RDKit          2D

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  1  0  0
    1.9400    0.0000    0.0000 Br  0  0  0  0  0  0  0  0  0  4  0  0
  1  2  1  0
M  END
$MOL

     RDKit          2D

  2  1  0  0  0  0  0  0  0  0999 V2000
    3.0000    2.0000    0.0000 Li  0  0  0  0  0  0  0  0  0  3  0  0
    5.0200    2.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  2  0  0
  1  2  1  0
M  END
```

注: H は implicit (CH3 の 3H は AddHs で補う)。電荷 M CHG は無し (LiBr / LiCl も covalent 扱い)。

- [ ] **Step 5: Run test to verify it passes**

```
pytest tests/test_rxn_parser.py::test_parse_metathesis_4center_rxn -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add examples/metathesis_4center.rxn tests/conftest.py tests/test_rxn_parser.py
git commit -m "$(cat <<'EOF'
feat(examples): add metathesis_4center.rxn (CH3Cl + LiBr -> CH3Br + LiCl)

reactant 2 fragments (CH3Cl + LiBr), product 2 fragments (CH3Br + LiCl)。
formed=(C-Br, Li-Cl), broken=(C-Cl, Li-Br)。Atom map 1=C, 2=Cl, 3=Li, 4=Br。
LiBr / LiCl は M CHG なしの contact ion pair として covalent 扱い、UMA omol
で性能評価。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: NEB refine guard rejection for metathesis

**Files:**
- Modify: `tests/test_cli_neb_refine_guard.py`

- [ ] **Step 1: Write failing test**

`tests/test_cli_neb_refine_guard.py` の `test_neb_refine_rejected_for_sn1_recomb_reaction` (line 43-65) の **直下** に追加:

```python
def test_neb_refine_rejected_for_metathesis_reaction(tmp_path, monkeypatch, caplog):
    """metathesis (formed=2, broken=2) も 1+1 以外なので reject される。"""
    rxn = Path("examples/metathesis_4center.rxn")
    if not rxn.exists():
        pytest.skip("examples/metathesis_4center.rxn not yet created (Task 9)")

    out = tmp_path / "out"
    monkeypatch.setattr(cli, "_check_hf_auth", lambda: 0)
    monkeypatch.setattr(cli, "_configure_reactx_logging", lambda: None)
    cli.log.propagate = True
    caplog.set_level("INFO", logger="reactx")

    rc = cli.main([
        "run", str(rxn), "-o", str(out),
        "--backend", "lj", "--n-angles", "1", "--max-relax-steps", "5",
        "--neb-refine",
    ])
    assert rc == 2, (
        f"expected exit code 2 for metathesis + --neb-refine, got {rc}"
    )
    msgs = " ".join(rec.getMessage() for rec in caplog.records)
    assert "Phase 3" in msgs
    assert "1 formed" in msgs and "1 broken" in msgs
```

- [ ] **Step 2: Run test to verify it passes**

```
pytest tests/test_cli_neb_refine_guard.py::test_neb_refine_rejected_for_metathesis_reaction -v
```

Expected: PASS — `cli._cmd_run` の guard `len(formed) != 1 or len(broken) != 1` は formed=2, broken=2 でも True なので reject される。新規実装不要。

- [ ] **Step 3: Commit**

```bash
git add tests/test_cli_neb_refine_guard.py
git commit -m "$(cat <<'EOF'
test(cli): cover metathesis NEB refine guard rejection

Phase 3 で導入された len==1 and len==1 guard が metathesis (formed=2,
broken=2) でも reject することを確認 (実装変更なし、既存 guard が
そのまま機能)。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: E2E integration test (slow, UMA)

**Files:**
- Create: `tests/test_re5_metathesis.py`

- [ ] **Step 1: Write failing test**

`tests/test_re5_metathesis.py` を以下の内容で作成:

```python
"""End-to-end 4-center metathesis test using UMA. Marked slow."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main


@pytest.mark.slow
def test_re5_metathesis_end_to_end(tmp_path: Path, metathesis_rxn_path: Path):
    out = tmp_path / "m4c"
    rc = main([
        "run", str(metathesis_rxn_path), "-o", str(out),
        "--backend", "uma",
        "--reaction-type", "metathesis_4center",
        "--n-angles", "8",
    ])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())

    pre = meta["prescreen"]
    assert pre["enabled"] is True
    if not pre["mmff_failed"]:
        # MMFF parameterize 成功時は top-3 (--prescreen-keep default) のみ UMA に
        assert len(pre["kept"]) == 3
        assert len(meta["trials"]) == 3
    # MMFF 失敗時 (Li / Br を MMFF94 が扱えない可能性) は全 trial が UMA
    # n_angles=8 のうち prescreen-rejected を除いた数が meta.trials に並ぶ
    assert meta["selected_trial"] >= 0
    assert meta["reaction_type"] == "metathesis_4center"

    # formed=2 で r_form_targets が 2 要素 list (Cordero C-Br ≈ 1.94, Li-Cl ≈ 2.02)
    rfts = meta["effective_params"]["r_form_targets"]
    assert len(rfts) == 2
    # 順序は formed bonds の順序と一致する想定だが、symbol 集合で検証
    sorted_targets = sorted(rfts)
    # C-Br (1.94) と Li-Cl (2.02) のいずれかに近い 2 値
    assert 1.85 <= sorted_targets[0] <= 2.10, (
        f"r_form_targets[0] should be ~Cordero C-Br/Li-Cl, got {sorted_targets[0]}"
    )
    assert 1.85 <= sorted_targets[1] <= 2.10, (
        f"r_form_targets[1] should be ~Cordero C-Br/Li-Cl, got {sorted_targets[1]}"
    )

    # ≥1 trial が reached_product
    reached = [t for t in meta["trials"] if t["reached_product"]]
    assert len(reached) >= 1, "expected ≥1 reached_product trial, got 0"

    # 最終フレームで C-Br ≤ 2.13 Å かつ Li-Cl ≤ 2.22 Å (Cordero × 1.1)
    frames = read(str(out / "trajectory.xyz"), index=":")
    syms = frames[-1].get_chemical_symbols()
    pos_last = frames[-1].positions
    c_atoms = [i for i, s in enumerate(syms) if s == "C"]
    cl_idx = syms.index("Cl")
    li_idx = syms.index("Li")
    br_idx = syms.index("Br")

    # 中心 C は Br に最も近い C (CH3Br になっている想定)
    d_c_to_br = [(i, float(((pos_last[i] - pos_last[br_idx]) ** 2).sum() ** 0.5)) for i in c_atoms]
    central_c, d_c_br = min(d_c_to_br, key=lambda kv: kv[1])
    d_li_cl = float(((pos_last[li_idx] - pos_last[cl_idx]) ** 2).sum() ** 0.5)

    assert d_c_br <= 2.13, f"final C-Br should be ≤2.13 A (Cordero × 1.1), got {d_c_br:.2f}"
    assert d_li_cl <= 2.22, f"final Li-Cl should be ≤2.22 A (Cordero × 1.1), got {d_li_cl:.2f}"
```

- [ ] **Step 2: Run test (requires UMA + HF auth)**

```
pytest tests/test_re5_metathesis.py -v -m slow
```

Expected: PASS — wall-clock ~30-90 s on RTX 5070 Ti per Phase 4 wall-clock 表からの推定 (multi-bond で MMFF 失敗時は ~1.5x)。

注: PASS しない場合は spec §10 リスクを参照し、`out/m4c/meta.json` と `out/m4c/trajectory.xyz` を直接確認:
1. `reached_product` が 0 件なら preset の `k_form` を 0.5 に下げる調整を検討。
2. 最終 C-Br が遠すぎる (>2.13 Å) なら `max_relax_steps` を 300 に増やすか `k_form` を上げる。
3. UMA が Li を不安定に扱う場合は preset 値の再チューニングが必要 (spec §10)。

- [ ] **Step 3: Commit**

```bash
git add tests/test_re5_metathesis.py
git commit -m "$(cat <<'EOF'
test(re5): add 4-center metathesis end-to-end UMA test

assert: prescreen 動作 (kept top-3 / mmff_failed fallback) /
selected_trial >= 0 / r_form_targets が 2 要素 list で Cordero
C-Br/Li-Cl 範囲 / >=1 reached_product / 最終 C-Br <= 2.13 A かつ
Li-Cl <= 2.22 A (Cordero × 1.1)。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update Phase tag header**

`README.md` の line 9 (Phase 4 tag の直下) に追加:

```markdown
**Phase 5** で 2-fragment 4-center metathesis (formed=2 + broken=2、broken bonds 各 fragment 内) を Tier 3 Kabsch alignment で追加。
```

- [ ] **Step 2: Update preset table**

`README.md` の `## Reaction-type presets` セクション (line 54-) の preset 表の末尾 (`sn1_recomb` 行の直後) に追加:

```markdown
| `metathesis_4center` | 1.0 | 1.0 | 4.0 | 200 | 元素表 (典型: C–Br 1.94, Li–Cl 2.02) | 2-fragment 4-center metathesis (例: CH₃Cl + LiBr → CH₃Br + LiCl) |
```

- [ ] **Step 3: Update DoD section header and add new steps**

`README.md` の `## Phase Re1 + Phase 3 + Phase 4 動作確認` を `## Phase Re1 + Phase 3 + Phase 4 + Phase 5 動作確認` にリネーム。手順 9 の後ろに以下を追加:

```markdown
10. `reactx run examples/metathesis_4center.rxn -o out/m4c/ --reaction-type metathesis_4center --backend uma --render` を実行
    → `meta.json` で `selected_trial >= 0`, `trials[].reached_product` ≥1 件 True、`out/m4c/scene.blend` で 4-center TS → CH₃Br + LiCl への乗り換えを視認 (CH₃ が Cl から Br へ、Li が Br から Cl へ同時に乗り換え)
11. `pytest -m slow` で `test_re5_metathesis` + 既存 `test_re1_*` / `test_re3_*` / `test_re4_*` が全 pass
```

- [ ] **Step 4: Update wall-clock table**

`README.md` の `## Wall-clock (実測)` セクションの表に metathesis 行を追加 (`SN1 recomb` の直後):

```markdown
| Metathesis (`examples/metathesis_4center.rxn`) | — (新規) | **~30-90 s** | Phase 5, 2 formed + 2 broken、bimolecular で 8 trials → prescreen で top-3 (Li/Br MMFF 失敗時は全 trial が UMA に fallback) |
```

注: 実測値は Task 11 の DoD 段階で書き換える。

- [ ] **Step 5: Update limits section header and reaction list**

`README.md` の `## Phase Re1 + Phase 3 + Phase 4 の方針と限界` を `## Phase Re1 + Phase 3 + Phase 4 + Phase 5 の方針と限界` にリネーム。第 3 項 (対応反応一覧) を以下で置き換え:

```markdown
- 対応反応 (Phase 5 時点): SN2 / proton transfer / Menshutkin (1 formed + 1 broken) + **E2 elimination (1 formed + 2 broken)** + **SN1 step 1 解離 (0 formed + 1 broken)** + **SN1 step 2 recombination (1 formed + 0 broken)** + **2-fragment 4-center metathesis (2 formed + 2 broken, broken 各 frag 内)**。中性 addition / cycloaddition / 4+ fragment ionic salt metathesis / Diels–Alder などは Phase 6+
```

詳細仕様の参照も同様に拡張:

```markdown
- 詳細仕様: `docs/superpowers/specs/2026-04-27-reactx-phase-Re1-design.md` (Phase Re1) / `docs/superpowers/specs/2026-05-03-phase-3-multibond-design.md` (Phase 3) / `docs/superpowers/specs/2026-05-03-phase-4-sn1-recomb-design.md` (Phase 4) / `docs/superpowers/specs/2026-05-03-phase-5-metathesis-design.md` (Phase 5)
```

- [ ] **Step 6: Run docs sanity check (no test, just visual review)**

```
pytest tests/ --collect-only -q 2>&1 | head -20
```

(README が tests に影響しないことを確認)

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs(readme): document Phase 5 metathesis_4center preset

preset 表 + DoD 手順 + wall-clock 表 + 制限事項に metathesis_4center 行を
追加。Phase tag header に Phase 5 を追記、対応反応一覧に
"2-fragment 4-center metathesis" を追加し、Phase 6+ に積み残し項目
(中性 addition / cycloaddition / 4+ fragment salt metathesis / Diels-Alder)
を整理。詳細仕様への参照も Phase 5 design doc を追記。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Final regression suite run

**Files:** none (verification only)

- [ ] **Step 1: Run all non-slow / non-blender tests**

```
pytest -m "not slow and not blender" -v
```

Expected: All passed. 新規 + 既存テストが全 pass。

- [ ] **Step 2: Run Tier 1/2 regression integration tests**

```
pytest tests/test_re1_sn2.py tests/test_re1_proton_transfer.py tests/test_re1_menshutkin.py tests/test_re3_e2.py tests/test_re3_sn1_dissoc.py tests/test_re4_sn1_recomb.py -v -m slow
```

Expected: All passed (Tier 1/2 挙動が数値的に不変であることを確認)。

- [ ] **Step 3: Run Phase 5 integration test**

```
pytest tests/test_re5_metathesis.py -v -m slow
```

Expected: PASS

- [ ] **Step 4: Verify branch state**

```
git log --oneline phase-5 ^develop
```

Expected: 約 14 commits — 1 spec commit + 12-13 implementation task commits (TDD で test-only commit と impl commit が混在するので前後する)。

- [ ] **Step 5: Open PR (manual; user-triggered)**

```bash
gh pr create --base develop --head phase-5 --title "Phase 5: 2-fragment 4-center metathesis (Tier 3 Kabsch alignment)" --body "$(cat <<'EOF'
## Summary
- Phase 4 で `_place_fragments` の最終 NotImplementedError を、2-fragment / formed=2 / broken=2 / broken_within_* が 1+1 の 4-center metathesis に限定して実装
- 例反応: `CH₃Cl + LiBr → CH₃Br + LiCl`
- Tier 3 placement: anchor 軸 (broken_within_reference の 2 endpoint) の垂直方向に target を置き、Kabsch (orthogonal Procrustes) で moving fragment の incoming pair を fit
- 新規 preset `metathesis_4center`、新規 example `examples/metathesis_4center.rxn`、新規 slow test `test_re5_metathesis.py`

## Test plan
- [ ] `pytest -m "not slow and not blender"` 全 pass
- [ ] `pytest -m slow` で Tier 1/2 (SN2 / PT / Menshutkin / E2 / SN1 dissoc / SN1 recomb) + Phase 5 (metathesis) 全 pass
- [ ] `reactx run examples/metathesis_4center.rxn -o out/m4c/ --reaction-type metathesis_4center --backend uma --render` を実行 → Blender で CH₃ が Cl→Br、Li が Br→Cl に同時に乗り換える 4-center metathesis を視認

## References
- Spec: `docs/superpowers/specs/2026-05-03-phase-5-metathesis-design.md`
- Plan: `docs/superpowers/plans/2026-05-03-phase-5-metathesis.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

注: PR 作成は user の判断で行う。auto モードでも PR 作成は外部システム影響なので手動 trigger。

---

## Self-Review Notes (skip in execution; for plan author only)

**Spec coverage check:**
- §1 目的 → Task 1-12 全体
- §2 Non-goals → Task 10 (NEB guard 維持確認)
- §3 設計方針 (3.1-3.3) → Task 2 (perp_dir), Task 3 (fixture geometry), Task 4 (Kabsch + alignment)
- §4 パイプライン全体像 → Task 7 (dispatcher) + Task 11 (E2E)
- §5.1 embed3d.py → Task 1, 2, 4, 5, 6, 7
- §5.2 presets.py → Task 8
- §5.3 cli.py 不変確認 → Task 10
- §5.4 不変モジュール → 暗黙 (テスト変更なし)
- §6 例 .rxn → Task 9
- §6.2 統合テスト → Task 11
- §7 ユニットテスト (p)-(y) → Task 1 (s)(t), Task 2 (y), Task 4 (q), Task 5 (r), Task 6 (u)(v), Task 7 (p)(w)(x)
- §8 DoD → Task 12 (README) + Task 13 (regression)
- §9 適用限界 → docstring + Task 7 multi-substrate test
- §10 リスク → Task 11 注書きで再チューニング指針
- §11 実装順序 → Task 番号通り

**Type consistency:**
- `_kabsch_rigid_transform(src, dst) -> tuple[np.ndarray, np.ndarray]` — Task 1, 4
- `_perpendicular_face_dir(axis, offset) -> np.ndarray` — Task 2, 4
- `_kabsch_alignment(mol_h, frag_indices, positions, bond_changes, *, rotation_perturbation) -> np.ndarray` — Task 4, 5, 6, 7
- `FRAGMENT_SEPARATION = 3.5` (既存定数を流用) — Task 4
- `BondChanges`, `compute_bond_changes` — 既存 API、変更なし
- `metathesis_4center` preset name — Task 8 で導入、Task 9-12 で参照

**Placeholder scan:** 全タスクで実コード/コマンド/期待出力を記載済み、TBD/TODO なし。
