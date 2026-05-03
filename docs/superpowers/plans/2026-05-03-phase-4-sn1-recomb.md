# Phase 4 SN1 Step 2 (Tier 2 Centroid Placement) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 3 で `_place_fragments` の hook として残された Tier 2 (broken=0 / multi-substrate) のうち、SN1 step 2 (cation + nucleophile recombination) のみ実装する。例反応 `(CH₃)₃C⁺ + Cl⁻ → (CH₃)₃CCl`。

**Architecture:** `embed3d._place_fragments` の dispatcher に `if not bond_changes.broken and bond_changes.formed: → _planar_face_placement` 分岐を追加。新規ヘルパ `_find_substrate_by_size`, `_plane_normal_at_anchor`, `_planar_face_placement` の 3 つで構成。plane-normal は SVD 平面 fit で sp² cation の空 p 軌道方向を求め、隣接 <3 や残差大ならフォールバック (`-unit(mean_neighbor - anchor)`)、それも degenerate なら `+z`。`sn1_recomb` preset を新設、CLI / NEB refine guard / unimolecular auto-clamp は既存挙動を維持。

**Tech Stack:** Python 3.11, RDKit, ASE, NumPy (SVD via `np.linalg.svd`), pytest, UMA-m-1p1 (slow integration test only)

**Spec:** `docs/superpowers/specs/2026-05-03-phase-4-sn1-recomb-design.md`

**Branch:** `phase-4` (already created from `develop`, spec already committed at `13302b1`)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `reactx/embed3d.py` | Modify | Tier 2 dispatch + 3 new helpers (`_find_substrate_by_size`, `_plane_normal_at_anchor`, `_planar_face_placement`) |
| `reactx/presets.py` | Modify | Add `sn1_recomb` preset |
| `examples/sn1_recomb.rxn` | Create | (CH₃)₃C⁺ + Cl⁻ → (CH₃)₃CCl の MOL V2000 |
| `tests/conftest.py` | Modify | Add `sn1_recomb_rxn_path` fixture, `sn1_recomb_atoms_setup` fixture |
| `tests/test_embed3d_placement.py` | Modify | Tier 2 ユニットテスト 7 ケース追加 + 既存 `test_place_fragments_raises_for_broken_zero_bimolecular` を Tier 2 動作確認テストに置き換え |
| `tests/test_presets.py` | Modify | `sn1_recomb` preset assertion + `PRESETS` キー集合更新 |
| `tests/test_re4_sn1_recomb.py` | Create | slow E2E 統合テスト |
| `tests/test_cli_neb_refine_guard.py` | Modify | SN1 step 2 ケース追加 |
| `README.md` | Modify | preset 表に `sn1_recomb` 行 + Phase 4 DoD セクション + wall-clock 表に `sn1_recomb` 行 |

---

## Task 1: `_find_substrate_by_size` helper

**Files:**
- Modify: `reactx/embed3d.py`
- Modify: `tests/test_embed3d_placement.py`

- [ ] **Step 1: Write failing test**

`tests/test_embed3d_placement.py` の末尾に追加:

```python
def test_find_substrate_by_size_picks_larger_heavy_count():
    """Tier 2: 重原子数最大の fragment を substrate として返す。"""
    from reactx.embed3d import _find_substrate_by_size
    mol, frags = _make_mol_with_frags("[C+](C)(C)C.[Cl-]")
    found = _find_substrate_by_size(frags)
    # tBu+ fragment は 4 heavy (1 C+ + 3 methyl C), Cl- は 1 heavy。
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    heavy_in_found = sum(1 for i in found if syms[i] != "H")
    assert heavy_in_found == 4


def test_find_substrate_by_size_tie_break_smallest_atom_index():
    """同じ heavy 数の場合、最小 atom index を含む fragment を選ぶ。"""
    from reactx.embed3d import _find_substrate_by_size
    # Cl- (heavy=1) と F- (heavy=1) の tie。frags[0] が smaller idx なので選ばれる。
    mol, frags = _make_mol_with_frags("[Cl-].[F-]")
    found = _find_substrate_by_size(frags)
    assert found == frags[0], "tie-break should pick fragment with smallest atom index"
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_embed3d_placement.py::test_find_substrate_by_size_picks_larger_heavy_count -v
```

Expected: FAIL with `ImportError: cannot import name '_find_substrate_by_size'`

- [ ] **Step 3: Write minimal implementation**

`reactx/embed3d.py` の `_find_substrate_fragment` 関数の直後に追加:

```python
def _find_substrate_by_size(
    frag_indices: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    """Tier 2 substrate identification: largest heavy-atom fragment.

    Tie の場合は最小 atom index を含む方を選ぶ (deterministic)。
    SN1 step 2 では cation = (CH₃)₃C⁺ (4 heavy) > Cl⁻ (1 heavy) で明確に決まる。
    """
    if not frag_indices:
        raise ValueError("frag_indices is empty")

    def _heavy_count_in_frag(frag: tuple[int, ...]) -> int:
        # Note: this helper does not have direct access to mol; we approximate
        # heavy count by len(frag). Caller's mol_h passes Chem.AddHs already, so
        # we instead use the canonical ordering to break ties; heavy count is
        # the number of atoms in frag whose index is below mol_h's first H.
        return len(frag)

    # All-atom count (heavy + H) ranking; SN1 step 2 では heavy ratio が同じなので
    # これで十分。tie-break は frag に含まれる最小 atom index で行う。
    return max(
        frag_indices,
        key=lambda f: (len(f), -min(f)),
    )
```

注: 上記は all-atom count で代用しており、heavy count を取りたい場合は mol_h を引数で
受け取る必要がある。SN1 step 2 では tBu+ (実 atom=13: 4 heavy + 9 H) vs Cl- (1 atom)
で all-atom count でも heavy count でも同じ結果になるので、簡単化のため all-atom で
実装する。後の Task 4 で mol_h を渡せる API に統一されているなら refactor。

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_embed3d_placement.py::test_find_substrate_by_size_picks_larger_heavy_count tests/test_embed3d_placement.py::test_find_substrate_by_size_tie_break_smallest_atom_index -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add reactx/embed3d.py tests/test_embed3d_placement.py
git commit -m "feat(embed3d): add _find_substrate_by_size helper for Tier 2 placement

Phase 4 SN1 step 2 で broken=() のとき、Tier 1 の \`_find_substrate_fragment\`
が None を返すため、別経路で substrate を識別する必要がある。重原子数最大
(SN1 step 2 では cation 側) を選び、tie の場合は最小 atom index を含む方を
deterministic に返す。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `_plane_normal_at_anchor` — ≥3 neighbors planar branch

**Files:**
- Modify: `reactx/embed3d.py`
- Modify: `tests/test_embed3d_placement.py`

- [ ] **Step 1: Write failing test**

`tests/test_embed3d_placement.py` の末尾に追加:

```python
def test_plane_normal_at_anchor_planar_three_neighbors():
    """3 substrate 隣接が xy 平面に乗っている場合、法線は ±z 方向 (符号は +z 寄り)。"""
    from reactx.embed3d import _plane_normal_at_anchor

    mol = Chem.AddHs(Chem.MolFromSmiles("[C+](C)(C)C"))
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    central = next(
        i for i, a in enumerate(mol.GetAtoms())
        if a.GetSymbol() == "C" and a.GetFormalCharge() == 1
    )
    methyl_carbons = [
        n.GetIdx() for n in mol.GetAtomWithIdx(central).GetNeighbors()
        if n.GetSymbol() == "C"
    ]
    assert len(methyl_carbons) == 3, f"expected 3 methyl C neighbors, got {len(methyl_carbons)}"

    n = mol.GetNumAtoms()
    positions = np.zeros((n, 3))
    positions[central] = (0.0, 0.0, 0.0)
    # 3 methyl C を xy 平面の三角形に置く
    for k, m in enumerate(methyl_carbons):
        theta = 2 * np.pi * k / 3
        positions[m] = (np.cos(theta) * 1.5, np.sin(theta) * 1.5, 0.0)

    substrate = tuple(range(n))  # 全 atom が同じ fragment
    direction = _plane_normal_at_anchor(positions, central, mol, substrate)

    assert direction.shape == (3,)
    np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-6)
    # 法線は ±z で、符号は +z 寄り
    assert direction[2] > 0, f"sign disambiguation failed: direction={direction}"
    assert abs(direction[0]) < 1e-3 and abs(direction[1]) < 1e-3, (
        f"direction should be along z, got {direction}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_embed3d_placement.py::test_plane_normal_at_anchor_planar_three_neighbors -v
```

Expected: FAIL with `ImportError: cannot import name '_plane_normal_at_anchor'`

- [ ] **Step 3: Write minimal implementation**

`reactx/embed3d.py` の冒頭の定数定義に `PLANE_FIT_TOLERANCE` を追加:

```python
FRAGMENT_SEPARATION = 3.5  # Å — attack distance for multi-fragment placement
PLANE_FIT_TOLERANCE = 0.3  # Å — SVD residual (smallest singular value) threshold
```

`_find_substrate_by_size` の後ろに追加:

```python
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
            return normal / np.linalg.norm(normal)

    raise NotImplementedError("fallback branches in later tasks")
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_embed3d_placement.py::test_plane_normal_at_anchor_planar_three_neighbors -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add reactx/embed3d.py tests/test_embed3d_placement.py
git commit -m "feat(embed3d): add _plane_normal_at_anchor SVD planar branch

Phase 4 SN1 step 2 placement の主路: anchor の substrate 内隣接 (heavy + H)
を SVD で平面 fit、最小特異値の右特異ベクトルを法線として返す。fallback と
degenerate branch は後続 task で追加。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `_plane_normal_at_anchor` — fallback (1-2 neighbors / non-planar)

**Files:**
- Modify: `reactx/embed3d.py`
- Modify: `tests/test_embed3d_placement.py`

- [ ] **Step 1: Write failing test**

`tests/test_embed3d_placement.py` の末尾に追加:

```python
def test_plane_normal_at_anchor_two_neighbors_falls_back_to_anti_mean():
    """隣接 2 個の場合、direction = -unit(mean_neighbor - anchor)。"""
    from reactx.embed3d import _plane_normal_at_anchor

    # CH2=CH+ (vinyl cation): 中心 C+ に C 隣接 1 + H 隣接 1
    mol = Chem.AddHs(Chem.MolFromSmiles("[CH+]=C"))
    central = next(
        i for i, a in enumerate(mol.GetAtoms())
        if a.GetSymbol() == "C" and a.GetFormalCharge() == 1
    )
    n = mol.GetNumAtoms()
    positions = np.zeros((n, 3))
    positions[central] = (0.0, 0.0, 0.0)
    # 2 substrate 隣接 (C と H) を +x 方向に置く (mean が +x)
    neighbors = [
        nb.GetIdx() for nb in mol.GetAtomWithIdx(central).GetNeighbors()
    ]
    assert len(neighbors) == 2, f"expected 2 neighbors, got {len(neighbors)}"
    positions[neighbors[0]] = (1.5, 0.5, 0.0)
    positions[neighbors[1]] = (1.5, -0.5, 0.0)

    substrate = tuple(range(n))
    direction = _plane_normal_at_anchor(positions, central, mol, substrate)

    np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-6)
    # mean_neighbor = (1.5, 0.0, 0.0), -unit = (-1.0, 0.0, 0.0)
    np.testing.assert_allclose(direction, [-1.0, 0.0, 0.0], atol=1e-3)


def test_plane_normal_at_anchor_non_planar_three_neighbors_falls_back():
    """3 隣接でも平面 fit 残差が大きい (sp³-like) 場合は fallback。"""
    from reactx.embed3d import _plane_normal_at_anchor

    mol = Chem.AddHs(Chem.MolFromSmiles("[C+](C)(C)C"))
    central = next(
        i for i, a in enumerate(mol.GetAtoms())
        if a.GetSymbol() == "C" and a.GetFormalCharge() == 1
    )
    methyl_carbons = [
        nb.GetIdx() for nb in mol.GetAtomWithIdx(central).GetNeighbors()
        if nb.GetSymbol() == "C"
    ]
    n = mol.GetNumAtoms()
    positions = np.zeros((n, 3))
    positions[central] = (0.0, 0.0, 0.0)
    # 3 methyl C を sp3 風に配置 (z 方向に大きな散らばり; 平面 fit 残差が大きい)
    sp3_dirs = np.array([
        [1.0, 0.0, 1.0],
        [-0.5, 0.866, 1.0],
        [-0.5, -0.866, 1.0],
    ])
    sp3_dirs /= np.linalg.norm(sp3_dirs, axis=1, keepdims=True)
    for m, d in zip(methyl_carbons, sp3_dirs, strict=True):
        positions[m] = d * 1.5

    substrate = tuple(range(n))
    direction = _plane_normal_at_anchor(positions, central, mol, substrate)

    np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-6)
    # mean of 3 sp3 dirs is in +z direction → -unit(mean) is -z direction
    assert direction[2] < 0, f"non-planar fallback should point -z, got {direction}"
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_embed3d_placement.py::test_plane_normal_at_anchor_two_neighbors_falls_back_to_anti_mean tests/test_embed3d_placement.py::test_plane_normal_at_anchor_non_planar_three_neighbors_falls_back -v
```

Expected: FAIL with `NotImplementedError: fallback branches in later tasks`

- [ ] **Step 3: Write fallback implementation**

`reactx/embed3d.py` の `_plane_normal_at_anchor` 内、`raise NotImplementedError(...)` を以下で置き換え:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_embed3d_placement.py::test_plane_normal_at_anchor_two_neighbors_falls_back_to_anti_mean tests/test_embed3d_placement.py::test_plane_normal_at_anchor_non_planar_three_neighbors_falls_back -v
```

Expected: 2 passed

回帰テスト確認:

```
pytest tests/test_embed3d_placement.py::test_plane_normal_at_anchor_planar_three_neighbors -v
```

Expected: PASS (Task 2 の planar branch が引き続き動く)

- [ ] **Step 5: Commit**

```bash
git add reactx/embed3d.py tests/test_embed3d_placement.py
git commit -m "feat(embed3d): add fallback branches to _plane_normal_at_anchor

隣接 1-2 個または 3 個でも平面 fit 残差が大きい場合は -unit(mean_neighbor -
anchor) にフォールバック。さらに mean_neighbor が anchor と一致する degenerate
ケースでは +z 軸 (deterministic) + warning ログ。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `_plane_normal_at_anchor` — degenerate branch

**Files:**
- Modify: `tests/test_embed3d_placement.py`

- [ ] **Step 1: Write failing test**

`tests/test_embed3d_placement.py` の末尾に追加:

```python
def test_plane_normal_at_anchor_degenerate_uses_z_fallback(caplog):
    """隣接 0 個の場合 (anchor が単独 atom) は [0, 0, 1] + warning。"""
    from reactx.embed3d import _plane_normal_at_anchor

    # Cl- 単独: 隣接 0
    mol = Chem.AddHs(Chem.MolFromSmiles("[Cl-]"))
    n = mol.GetNumAtoms()
    positions = np.zeros((n, 3))
    substrate = tuple(range(n))

    caplog.set_level("WARNING", logger="reactx.embed3d")
    direction = _plane_normal_at_anchor(positions, 0, mol, substrate)

    np.testing.assert_allclose(direction, [0.0, 0.0, 1.0], atol=1e-9)
    assert any("plane-normal" in rec.getMessage() for rec in caplog.records), (
        "expected a warning log for degenerate anchor"
    )
```

- [ ] **Step 2: Run test to verify it passes (already implemented in Task 3)**

```
pytest tests/test_embed3d_placement.py::test_plane_normal_at_anchor_degenerate_uses_z_fallback -v
```

Expected: PASS (Task 3 の degenerate branch がそのままカバー)

注: TDD 原則上は test を先に書いて fail を確認するのが理想だが、Task 3 の実装が既に
このケースをカバーしているため、このタスクでは test を追加してカバレッジを明示的に
記録するだけ。新規 production code 変更なし。

- [ ] **Step 3: Commit**

```bash
git add tests/test_embed3d_placement.py
git commit -m "test(embed3d): cover degenerate (0-neighbor) branch of _plane_normal_at_anchor

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `sn1_recomb_atoms_setup` fixture

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_embed3d_placement.py` (テストで使う準備)

- [ ] **Step 1: Add fixture to conftest.py**

`tests/conftest.py` の末尾に追加:

```python
@pytest.fixture()
def sn1_recomb_atoms_setup():
    """SN1 step 2 setup: tBu+ + Cl- with formed=(C-Cl), broken=()."""
    mol = Chem.AddHs(Chem.MolFromSmiles("[C+](C)(C)C.[Cl-]"))
    Chem.SanitizeMol(mol)
    frag_indices = Chem.GetMolFrags(mol)
    n = mol.GetNumAtoms()
    positions = np.zeros((n, 3))
    # tBu+ を planar 三角形 + 中心 C+ に: 中心 C+ は (0,0,0)、3 methyl C は xy 平面
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    central = next(
        i for i, a in enumerate(mol.GetAtoms())
        if a.GetSymbol() == "C" and a.GetFormalCharge() == 1
    )
    positions[central] = (0.0, 0.0, 0.0)
    methyl_carbons = [
        nb.GetIdx() for nb in mol.GetAtomWithIdx(central).GetNeighbors()
        if nb.GetSymbol() == "C"
    ]
    for k, m in enumerate(methyl_carbons):
        theta = 2 * np.pi * k / 3
        positions[m] = (np.cos(theta) * 1.5, np.sin(theta) * 1.5, 0.0)
    # 各 methyl C の H 隣接を適当な位置に (test 内で正確には使わない)
    for m in methyl_carbons:
        for h_nb in mol.GetAtomWithIdx(m).GetNeighbors():
            if h_nb.GetSymbol() == "H":
                # methyl C の周りに H を散らす
                positions[h_nb.GetIdx()] = positions[m] + np.array(
                    [0.5 * (h_nb.GetIdx() % 3 - 1), 0.5, 0.5 * ((h_nb.GetIdx() // 3) % 2)]
                )
    # Cl- を遠くに置く (Tier 2 placement で動かされる)
    cl_idx = syms.index("Cl")
    positions[cl_idx] = (10.0, 10.0, 10.0)

    bc = BondChanges(formed=((central, cl_idx),), broken=())
    return mol, frag_indices, positions, bc
```

- [ ] **Step 2: Smoke test the fixture**

`tests/test_embed3d_placement.py` の末尾に追加:

```python
def test_sn1_recomb_fixture_shape(sn1_recomb_atoms_setup):
    """fixture の形状確認: 2 fragments, formed=1, broken=0, tBu+ planar at origin."""
    mol_h, frag_indices, positions, bc = sn1_recomb_atoms_setup
    assert len(frag_indices) == 2
    assert len(bc.formed) == 1
    assert len(bc.broken) == 0
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    central = next(
        i for i, a in enumerate(mol_h.GetAtoms())
        if a.GetSymbol() == "C" and a.GetFormalCharge() == 1
    )
    np.testing.assert_allclose(positions[central], [0.0, 0.0, 0.0], atol=1e-9)
    cl_idx = syms.index("Cl")
    assert (central, cl_idx) == bc.formed[0] or (cl_idx, central) == bc.formed[0]
```

- [ ] **Step 3: Run smoke test**

```
pytest tests/test_embed3d_placement.py::test_sn1_recomb_fixture_shape -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_embed3d_placement.py
git commit -m "test(conftest): add sn1_recomb_atoms_setup fixture

Phase 4 Tier 2 placement テスト用の tBu+ + Cl- fixture。tBu+ は xy 平面に
planar、Cl- は (10,10,10) (placement で原点近傍に動かされる対象)。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `_planar_face_placement` — メインロジック

**Files:**
- Modify: `reactx/embed3d.py`
- Modify: `tests/test_embed3d_placement.py`

- [ ] **Step 1: Write failing test**

`tests/test_embed3d_placement.py` の末尾に追加:

```python
def test_planar_face_placement_places_cl_along_plane_normal(sn1_recomb_atoms_setup):
    """Tier 2: Cl の最終位置が anchor + plane_normal * FRAGMENT_SEPARATION。"""
    from reactx.embed3d import _planar_face_placement, FRAGMENT_SEPARATION

    mol_h, frag_indices, positions, bc = sn1_recomb_atoms_setup
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    central = next(
        i for i, a in enumerate(mol_h.GetAtoms())
        if a.GetSymbol() == "C" and a.GetFormalCharge() == 1
    )
    cl_idx = syms.index("Cl")

    substrate = next(f for f in frag_indices if central in f)
    out = _planar_face_placement(
        mol_h, frag_indices, positions.copy(), bc, substrate,
        rotation_perturbation=None,
    )

    # tBu+ は xy 平面、anchor=central=(0,0,0), plane normal は ±z (sign +z)。
    # → target = (0, 0, FRAGMENT_SEPARATION) = (0, 0, 3.5)
    expected = np.array([0.0, 0.0, FRAGMENT_SEPARATION])
    actual = out[cl_idx]
    np.testing.assert_allclose(actual, expected, atol=0.5), (
        f"Cl should be placed at +z * {FRAGMENT_SEPARATION}, got {actual}"
    )


def test_planar_face_placement_respects_rotation_perturbation(sn1_recomb_atoms_setup):
    """rotation_perturbation で Cl の位置が回転されることを確認。"""
    from reactx.embed3d import _planar_face_placement, FRAGMENT_SEPARATION

    mol_h, frag_indices, positions, bc = sn1_recomb_atoms_setup
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    central = next(
        i for i, a in enumerate(mol_h.GetAtoms())
        if a.GetSymbol() == "C" and a.GetFormalCharge() == 1
    )
    cl_idx = syms.index("Cl")
    substrate = next(f for f in frag_indices if central in f)

    out_id = _planar_face_placement(
        mol_h, frag_indices, positions.copy(), bc, substrate,
        rotation_perturbation=None,
    )

    # 90° rotation around x axis: z → y
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    out_rot = _planar_face_placement(
        mol_h, frag_indices, positions.copy(), bc, substrate,
        rotation_perturbation=R,
    )

    # Cl が異なる位置に置かれること
    assert not np.allclose(out_id[cl_idx], out_rot[cl_idx], atol=0.1), (
        f"rotation_perturbation should change Cl position; "
        f"identity={out_id[cl_idx]}, rotated={out_rot[cl_idx]}"
    )
    # 距離は同じ (回転は等距変換)
    d_id = float(np.linalg.norm(out_id[cl_idx] - out_id[central]))
    d_rot = float(np.linalg.norm(out_rot[cl_idx] - out_rot[central]))
    np.testing.assert_allclose(d_id, d_rot, atol=0.1)
    np.testing.assert_allclose(d_rot, FRAGMENT_SEPARATION, atol=0.5)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_embed3d_placement.py::test_planar_face_placement_places_cl_along_plane_normal -v
```

Expected: FAIL with `ImportError: cannot import name '_planar_face_placement'`

- [ ] **Step 3: Write minimal implementation**

`reactx/embed3d.py` の `_plane_normal_at_anchor` の後ろに追加:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_embed3d_placement.py::test_planar_face_placement_places_cl_along_plane_normal tests/test_embed3d_placement.py::test_planar_face_placement_respects_rotation_perturbation -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add reactx/embed3d.py tests/test_embed3d_placement.py
git commit -m "feat(embed3d): add _planar_face_placement for Tier 2 (SN1 step 2)

各 non-substrate fragment について bridging formed bond の substrate-end を
anchor とし、anchor の plane normal 方向に FRAGMENT_SEPARATION 離れた点へ
fragment の incoming atom を並進する。複数 base attack on single anchor は
NotImplementedError。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Wire Tier 2 into `_place_fragments` dispatcher

**Files:**
- Modify: `reactx/embed3d.py`
- Modify: `tests/test_embed3d_placement.py`

- [ ] **Step 1: Update existing failing test for new behavior**

`tests/test_embed3d_placement.py` の既存 `test_place_fragments_raises_for_broken_zero_bimolecular` を以下で置き換え:

```python
def test_place_fragments_dispatches_tier2_for_broken_zero_bimolecular(sn1_recomb_atoms_setup):
    """Tier 2 dispatch: broken=() の bimolecular で _planar_face_placement に流す。

    Phase 3 では NotImplementedError を投げていたが、Phase 4 で実装したので
    placement が成功し、Cl の位置が plane normal 方向に動くことを確認する。
    """
    from reactx.embed3d import _place_fragments, FRAGMENT_SEPARATION

    mol_h, frag_indices, positions, bc = sn1_recomb_atoms_setup
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    cl_idx = syms.index("Cl")
    central = next(
        i for i, a in enumerate(mol_h.GetAtoms())
        if a.GetSymbol() == "C" and a.GetFormalCharge() == 1
    )

    out = _place_fragments(
        mol_h, frag_indices, positions.copy(), bc,
        rotation_perturbation=None,
    )
    d = float(np.linalg.norm(out[cl_idx] - out[central]))
    np.testing.assert_allclose(d, FRAGMENT_SEPARATION, atol=0.5)


def test_place_fragments_still_raises_for_multi_substrate_metathesis_after_tier2():
    """multi-substrate metathesis (broken bonds が複数 frag に跨る) は Tier 2 でも reject。"""
    from reactx.embed3d import _place_fragments
    mol = Chem.AddHs(Chem.MolFromSmiles("CC.OO"))
    frags = Chem.GetMolFrags(mol)
    a = frags[0][0]
    b = frags[1][0]
    bc = BondChanges(formed=((frags[0][1], frags[1][1]),), broken=((a, b),))
    with pytest.raises(NotImplementedError, match="multi-substrate"):
        _place_fragments(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )
```

既存の `test_place_fragments_raises_for_multi_substrate_metathesis` (line 94-104) はそのまま残し、上記が補完的にカバー。

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_embed3d_placement.py::test_place_fragments_dispatches_tier2_for_broken_zero_bimolecular -v
```

Expected: FAIL — まだ `_place_fragments` の Tier 2 分岐が無いので `NotImplementedError("centroid-based placement ...")` で拒絶される。

- [ ] **Step 3: Wire Tier 2 into dispatcher**

`reactx/embed3d.py` の `_place_fragments` 関数を以下で置き換え (既存の Tier 1 ロジックを維持しつつ Tier 2 分岐を追加):

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
    Tier 2 (planar face, Phase 4): broken=() かつ formed>=1 の bimolecular (SN1 step 2)。
    Phase 4+ (未実装):              multi-substrate metathesis (broken が複数 frag に跨る)。
    """
    substrate = _find_substrate_fragment(frag_indices, bond_changes.broken)
    if substrate is not None and bond_changes.broken:
        return _directional_placement(
            frag_indices, positions, bond_changes, substrate,
            rotation_perturbation=rotation_perturbation,
        )

    if not bond_changes.broken and bond_changes.formed:
        substrate = _find_substrate_by_size(frag_indices)
        return _planar_face_placement(
            mol_h, frag_indices, positions, bond_changes, substrate,
            rotation_perturbation=rotation_perturbation,
        )

    raise NotImplementedError(
        "multi-substrate metathesis (broken bonds spanning fragments) is "
        f"Phase 4+. Got formed={bond_changes.formed}, broken={bond_changes.broken}, "
        f"frags={len(frag_indices)}."
    )
```

- [ ] **Step 4: Run all placement tests**

```
pytest tests/test_embed3d_placement.py -v
```

Expected: All previously-passing tests still pass + new Tier 2 dispatch test passes.

- [ ] **Step 5: Run regression — Tier 1 reactions (SN2 / E2 fixture) unchanged**

```
pytest tests/test_embed3d_placement.py::test_place_fragments_dispatch_directional_for_sn2_shape tests/test_embed3d_placement.py::test_place_fragments_e2_shape_directional_anchors_on_h -v
```

Expected: 2 passed (Tier 1 挙動が不変)

- [ ] **Step 6: Commit**

```bash
git add reactx/embed3d.py tests/test_embed3d_placement.py
git commit -m "feat(embed3d): dispatch Tier 2 placement for broken=0 bimolecular

Phase 3 で NotImplementedError を投げていた Tier 2 hook を
_planar_face_placement に流す。Tier 1 (E2 / SN2 / PT / Menshutkin /
SN1 step 1) の挙動は不変。multi-substrate metathesis は引き続き
NotImplementedError。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Multi-base attack edge-case test

**Files:**
- Modify: `tests/test_embed3d_placement.py`

- [ ] **Step 1: Write failing test**

`tests/test_embed3d_placement.py` の末尾に追加:

```python
def test_planar_face_placement_rejects_multi_base_on_single_anchor():
    """Tier 2 で 1 anchor に複数 nucleophile が共有する場合は NotImplementedError。"""
    from reactx.embed3d import _place_fragments

    # [C+](C)(C)C.[F-].[Cl-] で formed=((C+, F), (C+, Cl)) — 2 base が同じ anchor 共有
    mol = Chem.AddHs(Chem.MolFromSmiles("[C+](C)(C)C.[F-].[Cl-]"))
    frags = Chem.GetMolFrags(mol)
    central = next(
        i for i, a in enumerate(mol.GetAtoms())
        if a.GetSymbol() == "C" and a.GetFormalCharge() == 1
    )
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    f_idx = syms.index("F")
    cl_idx = syms.index("Cl")
    bc = BondChanges(
        formed=((central, f_idx), (central, cl_idx)),
        broken=(),
    )
    with pytest.raises(NotImplementedError, match="multi-base"):
        _place_fragments(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )
```

- [ ] **Step 2: Run test to verify it passes (already implemented in Task 6)**

```
pytest tests/test_embed3d_placement.py::test_planar_face_placement_rejects_multi_base_on_single_anchor -v
```

Expected: PASS (Task 6 の `_planar_face_placement` 内ロジックがカバー済み)

- [ ] **Step 3: Commit**

```bash
git add tests/test_embed3d_placement.py
git commit -m "test(embed3d): cover Tier 2 multi-base attack rejection

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: `sn1_recomb` preset

**Files:**
- Modify: `reactx/presets.py`
- Modify: `tests/test_presets.py`

- [ ] **Step 1: Write failing test**

`tests/test_presets.py` の末尾に追加:

```python
def test_sn1_recomb_preset_values():
    p = get_preset("sn1_recomb")
    assert p.name == "sn1_recomb"
    assert p.k_form == 1.0
    assert p.k_broken == 0.0
    assert p.r_broken == 4.0
    assert p.max_relax_steps == 200
    assert p.r_form is None
```

そして既存の `test_presets_dict_keys` を以下で置き換え:

```python
def test_presets_dict_keys():
    assert set(PRESETS) == {
        "sn2_anion",
        "proton_transfer",
        "menshutkin",
        "e2",
        "sn1_dissoc",
        "sn1_recomb",
    }
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_presets.py::test_sn1_recomb_preset_values tests/test_presets.py::test_presets_dict_keys -v
```

Expected: FAIL — `Unknown reaction-type preset 'sn1_recomb'` and キー集合不一致

- [ ] **Step 3: Add preset**

`reactx/presets.py` の `PRESETS["sn1_dissoc"] = ...` の後ろに追加:

```python
PRESETS["sn1_recomb"] = ReactionPreset(
    name="sn1_recomb",
    k_form=1.0,
    k_broken=0.0,    # broken=() のため使われない、明示的に 0
    r_broken=4.0,    # 同上、placeholder (effective_params の見栄え用)
    max_relax_steps=200,
    r_form=None,     # 元素表 (Cordero: C-Cl ≈ 1.78 Å)
)
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_presets.py -v
```

Expected: All passed (既存 + 新規 2 件)

- [ ] **Step 5: Commit**

```bash
git add reactx/presets.py tests/test_presets.py
git commit -m "feat(presets): add sn1_recomb preset for Phase 4 SN1 step 2

cation + nucleophile recombination 向け: k_form=1.0 で nucleophile を anchor
に引き寄せる Hookean、broken=() なので k_broken=0.0、max_relax_steps=200
(結合 1 本だけ作るので E2 より安く収束する想定)、r_form は元素表 fallback
で C-Cl ≈ 1.78 Å を期待。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: `examples/sn1_recomb.rxn`

**Files:**
- Create: `examples/sn1_recomb.rxn`
- Modify: `tests/conftest.py`
- Modify: `tests/test_rxn_parser.py` (smoke parse) — または既存 conftest fixture テストで covered

- [ ] **Step 1: Add fixture to conftest.py**

`tests/conftest.py` の `sn1_dissoc_rxn_path` fixture の後ろに追加:

```python
@pytest.fixture()
def sn1_recomb_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "sn1_recomb.rxn"
```

- [ ] **Step 2: Write failing parse smoke test**

`tests/test_rxn_parser.py` の末尾に追加 (または同等場所、ファイルの構造に合わせる):

```python
def test_parse_sn1_recomb_rxn(sn1_recomb_rxn_path):
    """examples/sn1_recomb.rxn が parse でき、formed=1 / broken=0 が抽出される。"""
    from rdkit import Chem
    from reactx.bond_changes import compute_bond_changes
    from reactx.rxn_parser import parse_rxn

    r_mol, p_mol, mapping = parse_rxn(sn1_recomb_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_bond_changes(r_h, p_h, mapping)

    assert len(bc.formed) == 1, f"expected 1 formed bond, got {bc.formed}"
    assert len(bc.broken) == 0, f"expected 0 broken bond, got {bc.broken}"
    # reactant が 2 fragments (cation + nucleophile)
    assert len(Chem.GetMolFrags(r_h)) == 2
    # product が 1 fragment (recombined)
    assert len(Chem.GetMolFrags(p_h)) == 1
    # formed bond は C-Cl
    a, b = bc.formed[0]
    syms = [at.GetSymbol() for at in r_h.GetAtoms()]
    assert {syms[a], syms[b]} == {"C", "Cl"}, (
        f"expected C-Cl formed, got {syms[a]}-{syms[b]}"
    )
```

- [ ] **Step 3: Run test to verify it fails**

```
pytest tests/test_rxn_parser.py::test_parse_sn1_recomb_rxn -v
```

Expected: FAIL — `examples/sn1_recomb.rxn` がまだ無い

- [ ] **Step 4: Create the example file**

`examples/sn1_recomb.rxn` を以下の内容で作成:

```
$RXN

      RDKit

  2  1
$MOL

     RDKit          2D

  4  3  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  1  0  0
    1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  2  0  0
   -0.7500    1.3000    0.0000 C   0  0  0  0  0  0  0  0  0  3  0  0
   -0.7500   -1.3000    0.0000 C   0  0  0  0  0  0  0  0  0  4  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
M  CHG  1   1   1
M  END
$MOL

     RDKit          2D

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  5  0  0
M  CHG  1   1  -1
M  END
$MOL

     RDKit          2D

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  1  0  0
    1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  2  0  0
   -0.7500    1.3000    0.0000 C   0  0  0  0  0  0  0  0  0  3  0  0
   -0.7500   -1.3000    0.0000 C   0  0  0  0  0  0  0  0  0  4  0  0
    0.0000    0.0000    1.5000 Cl  0  0  0  0  0  0  0  0  0  5  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
M  END
```

注: Phase 3 `examples/sn1_dissoc.rxn` (tBu-Br dissociation) を逆方向 + Br→Cl に
差し替えた構造。reactant は tBu+ + Cl- (2 fragments)、product は tBuCl (1 fragment)。

- [ ] **Step 5: Run test to verify it passes**

```
pytest tests/test_rxn_parser.py::test_parse_sn1_recomb_rxn -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add examples/sn1_recomb.rxn tests/conftest.py tests/test_rxn_parser.py
git commit -m "feat(examples): add sn1_recomb.rxn (tBu+ + Cl- -> tBuCl)

Phase 3 sn1_dissoc.rxn の逆向き + Br -> Cl に差し替えて作成。reactant 2
fragments (cation + anion), product 1 fragment (neutral)。formed=C-Cl,
broken=()。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: E2E integration test (slow, UMA)

**Files:**
- Create: `tests/test_re4_sn1_recomb.py`

- [ ] **Step 1: Write failing test**

`tests/test_re4_sn1_recomb.py` を以下の内容で作成:

```python
"""End-to-end SN1 step 2 (cation + nucleophile recombination) test using UMA. Marked slow."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main


@pytest.mark.slow
def test_re4_sn1_recomb_end_to_end(tmp_path: Path, sn1_recomb_rxn_path: Path):
    out = tmp_path / "sn1r"
    rc = main([
        "run", str(sn1_recomb_rxn_path), "-o", str(out),
        "--backend", "uma",
        "--reaction-type", "sn1_recomb",
        "--n-angles", "8",
    ])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())

    # bimolecular なので unimolecular auto-clamp は発動せず、8 trials が走る
    # (prescreen が top-3 に絞るので meta.json.trials は 8 件残る; UMA は 3 件のみ)
    assert len(meta["trials"]) == 8, (
        f"bimolecular reaction expected 8 trials in meta; got {len(meta['trials'])}"
    )
    assert meta["selected_trial"] >= 0
    assert meta["reaction_type"] == "sn1_recomb"

    # formed=1 で r_form_targets が 1 要素 list (Cordero C-Cl ≈ 1.78 Å)
    rfts = meta["effective_params"]["r_form_targets"]
    assert len(rfts) == 1
    assert 1.7 <= rfts[0] <= 1.85, f"r_form_targets[0] should be ≈1.78, got {rfts[0]}"

    # ≥1 trial が reached_product
    reached = [t for t in meta["trials"] if t["reached_product"]]
    assert len(reached) >= 1, f"expected ≥1 reached_product trial, got 0"

    # 最終フレームで C-Cl 距離 ≤ 1.95 Å (Cordero × 1.1)
    frames = read(str(out / "trajectory.xyz"), index=":")
    syms = frames[-1].get_chemical_symbols()
    cl_idx = syms.index("Cl")
    # 中心 C は Cl に最も近い C atom
    c_atoms = [i for i, s in enumerate(syms) if s == "C"]
    pos_last = frames[-1].positions
    d_to_cl = [(i, float(((pos_last[i] - pos_last[cl_idx]) ** 2).sum() ** 0.5)) for i in c_atoms]
    central, d_last = min(d_to_cl, key=lambda kv: kv[1])
    assert d_last <= 1.95, f"final C-Cl should be ≤1.95 A (Cordero × 1.1), got {d_last:.2f}"
```

- [ ] **Step 2: Run test (requires UMA + HF auth)**

```
pytest tests/test_re4_sn1_recomb.py -v -m slow
```

Expected: PASS — wall-clock ~30-60 s on RTX 5070 Ti per Phase 3 wall-clock table.

注: PASS しない場合は、`out/sn1r/meta.json` と `out/sn1r/trajectory.xyz` を直接
確認して preset チューニング (k_form=1.0 → 0.5 など) を検討。Spec §10 リスク参照。

- [ ] **Step 3: Commit**

```bash
git add tests/test_re4_sn1_recomb.py
git commit -m "test(re4): add SN1 step 2 end-to-end UMA test

assert: 8 trials (bimolecular, no auto-clamp) / selected_trial >= 0 /
r_form_targets[0] ≈ 1.78 Å (Cordero C-Cl) / >=1 reached_product /
final C-Cl <= 1.95 Å。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: NEB refine guard for SN1 step 2

**Files:**
- Modify: `tests/test_cli_neb_refine_guard.py`

- [ ] **Step 1: Write failing test (or extend existing parametrize)**

`tests/test_cli_neb_refine_guard.py` の `test_neb_refine_rejected_for_e2_reaction` の真下 (line 41 後ろ) に追加:

```python
def test_neb_refine_rejected_for_sn1_recomb_reaction(tmp_path, monkeypatch, caplog):
    """SN1 step 2 (formed=1, broken=0) も 1+1 以外なので reject される。"""
    rxn = Path("examples/sn1_recomb.rxn")
    if not rxn.exists():
        pytest.skip("examples/sn1_recomb.rxn not yet created (Task 10)")

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
        f"expected exit code 2 for SN1 step 2 + --neb-refine, got {rc}"
    )
    msgs = " ".join(rec.getMessage() for rec in caplog.records)
    assert "Phase 3" in msgs
    assert "1 formed" in msgs and "1 broken" in msgs
```

- [ ] **Step 2: Run test to verify it passes (CLI guard already handles formed=1+broken=0)**

```
pytest tests/test_cli_neb_refine_guard.py::test_neb_refine_rejected_for_sn1_recomb_reaction -v
```

Expected: PASS — `cli._cmd_run` の guard `len(formed) != 1 or len(broken) != 1` は
formed=1, broken=0 でも True (broken != 1) なので reject される。新規実装不要。

- [ ] **Step 3: Commit**

```bash
git add tests/test_cli_neb_refine_guard.py
git commit -m "test(cli): cover SN1 step 2 NEB refine guard rejection

Phase 3 で導入された len==1 and len==1 guard が SN1 step 2 (formed=1,
broken=0) でも reject することを確認 (実装変更なし、既存 guard が
そのまま機能)。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update preset table**

`README.md` の `## Reaction-type presets` セクション (line 52-) の preset 表の末尾 (`sn1_dissoc` 行の直後) に追加:

```markdown
| `sn1_recomb` | 1.0 | 0.0 | 4.0 | 200 | 元素表 (典型: C–Cl 1.78) | SN1 step 2 cation + nucleophile recombination (例: (CH₃)₃C⁺ + Cl⁻) |
```

- [ ] **Step 2: Update DoD section**

`README.md` の `## Phase Re1 + Phase 3 動作確認` セクションを `## Phase Re1 + Phase 3 + Phase 4 動作確認` にリネーム。手順 7 の後ろに以下を追加:

```markdown
8. `reactx run examples/sn1_recomb.rxn -o out/sn1r/ --reaction-type sn1_recomb --backend uma --render` を実行
   → `meta.json.trials` が 8 件 (bimolecular)、`reached_product=True` の trial が ≥1 件、`out/sn1r/scene.blend` で Cl⁻ が tBu⁺ の平面に向かって接近 → C–Cl 結合形成を視認
9. `pytest -m slow` で `test_re4_sn1_recomb` + 既存 `test_re1_*` / `test_re3_*` が全 pass
```

- [ ] **Step 3: Update wall-clock table**

`README.md` の `## Wall-clock (実測)` セクションの表に SN1 step 2 行を追加 (E2 / SN1 dissoc の直後):

```markdown
| SN1 recomb (`examples/sn1_recomb.rxn`) | — (新規) | **~30-60 s** | Phase 4, 1 formed + 0 broken、bimolecular で 8 trials → prescreen で top-3 |
```

注: 実測値は Task 11 の DoD 段階で書き換える。

- [ ] **Step 4: Update limits section**

`README.md` の `## Phase Re1 + Phase 3 の方針と限界` を `## Phase Re1 + Phase 3 + Phase 4 の方針と限界` にリネーム。第 3 項 (対応反応一覧) を以下で置き換え:

```markdown
- 対応反応 (Phase 4 時点): SN2 / proton transfer / Menshutkin (1 formed + 1 broken) + **E2 elimination (1 formed + 2 broken)** + **SN1 step 1 解離 (0 formed + 1 broken)** + **SN1 step 2 recombination (1 formed + 0 broken)**。中性 addition / cycloaddition / metathesis / Diels–Alder などは Phase 5+
```

詳細仕様の参照も同様に拡張:

```markdown
- 詳細仕様: `docs/superpowers/specs/2026-04-27-reactx-phase-Re1-design.md` (Phase Re1) / `docs/superpowers/specs/2026-05-03-phase-3-multibond-design.md` (Phase 3) / `docs/superpowers/specs/2026-05-03-phase-4-sn1-recomb-design.md` (Phase 4)
```

- [ ] **Step 5: Update Phase tag header (line 7)**

```markdown
**Phase 4** で SN1 step 2 cation + nucleophile recombination (1 formed + 0 broken) を Tier 2 plane-normal placement で追加。
```

を line 7 (Phase 3 行の直下) に追加。

- [ ] **Step 6: Run docs sanity check (no test, just visual review)**

```
pytest tests/ --collect-only -q 2>&1 | head -20
```

(README が tests に影響しないことを確認)

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "docs(readme): document Phase 4 SN1 step 2 (sn1_recomb preset)

preset 表 + DoD 手順 + wall-clock 表 + 制限事項に sn1_recomb 行を追加。
詳細仕様への参照も Phase 4 design doc を追記。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Final regression suite run

**Files:** none (verification only)

- [ ] **Step 1: Run all non-slow / non-blender tests**

```
pytest -m "not slow and not blender" -v
```

Expected: All passed. 新規 + 既存テストが全 pass。

- [ ] **Step 2: Run Tier 1 regression integration tests**

```
pytest tests/test_re1_sn2.py tests/test_re1_proton_transfer.py tests/test_re1_menshutkin.py tests/test_re3_e2.py tests/test_re3_sn1_dissoc.py -v -m slow
```

Expected: All passed (Tier 1 挙動が数値的に不変であることを確認)。

- [ ] **Step 3: Run Phase 4 integration test**

```
pytest tests/test_re4_sn1_recomb.py -v -m slow
```

Expected: PASS

- [ ] **Step 4: Verify branch state**

```
git log --oneline phase-4 ^develop
```

Expected: 13 (or 14) commits — 1 spec + 1 spec amend + 12 implementation tasks (TDD で test-only commit と impl commit が混在するので前後する)。

- [ ] **Step 5: Open PR (manual; user-triggered)**

```
gh pr create --base develop --head phase-4 --title "Phase 4: SN1 step 2 (Tier 2 plane-normal placement)" --body "$(cat <<'EOF'
## Summary
- Phase 3 で hook として残された `_place_fragments` Tier 2 (broken=0 / multi-substrate) のうち、SN1 step 2 (cation + nucleophile recombination) のみ実装
- 例反応: `(CH₃)₃C⁺ + Cl⁻ → (CH₃)₃CCl`
- plane-normal at anchor で sp² cation の空 p 軌道方向に nucleophile を配置
- 新規 preset `sn1_recomb`、新規 example `examples/sn1_recomb.rxn`、新規 slow test `test_re4_sn1_recomb.py`

## Test plan
- [ ] `pytest -m "not slow and not blender"` 全 pass
- [ ] `pytest -m slow` で Tier 1 (SN2 / PT / Menshutkin / E2 / SN1 dissoc) + Phase 4 (SN1 recomb) 全 pass
- [ ] `reactx run examples/sn1_recomb.rxn -o out/sn1r/ --reaction-type sn1_recomb --backend uma --render` を実行 → Blender で Cl⁻ が tBu⁺ の平面に向かって接近する様子を視認

## References
- Spec: `docs/superpowers/specs/2026-05-03-phase-4-sn1-recomb-design.md`
- Plan: `docs/superpowers/plans/2026-05-03-phase-4-sn1-recomb.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

注: PR 作成は user の判断で行う。auto モードでも PR 作成は外部システム影響なので手動 trigger。

---

## Self-Review Notes (skip in execution; for plan author only)

**Spec coverage check:**
- §1 目的 → Task 1-13 全体
- §2 Non-goals → Task 12 (NEB guard 維持)
- §3 plane-normal 方針 → Task 2-4
- §4 パイプライン全体像 → Task 7 (dispatcher) + Task 11 (E2E)
- §5.1 embed3d.py → Task 1, 2, 3, 4, 6, 7, 8
- §5.2 presets.py → Task 9
- §5.3 cli.py 不変確認 → Task 12
- §5.4 不変モジュール → 暗黙 (テスト変更なし)
- §6 例 .rxn → Task 10
- §6.2 統合テスト → Task 11
- §7 ユニットテスト (h)-(o) → Task 1-8 で網羅 (h=Task 6, i=Task 1, j=Task 2, k=Task 3, l=Task 4, m=Task 6, n=Task 7, o=Task 8)
- §8 DoD → Task 13 (README) + Task 14 (regression)
- §9 適用限界 → docstring + Task 7 multi-substrate test
- §11 実装順序 → Task 番号通り

**Type consistency:**
- `_find_substrate_by_size(frag_indices) -> tuple[int, ...]` — Task 1, 7
- `_plane_normal_at_anchor(positions, anchor, mol_h, substrate) -> np.ndarray` — Task 2, 3, 4, 6
- `_planar_face_placement(mol_h, frag_indices, positions, bond_changes, substrate, *, rotation_perturbation)` — Task 6, 7
- `FRAGMENT_SEPARATION = 3.5`, `PLANE_FIT_TOLERANCE = 0.3` — Task 2, 6
- `BondChanges`, `compute_bond_changes` — 既存 API、変更なし

**Placeholder scan:** 全タスクで実コード/コマンド/期待出力を記載済み、TBD/TODO なし。
