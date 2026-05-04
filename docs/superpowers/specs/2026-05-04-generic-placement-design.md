# reactx Phase 7 — Generic Steric-Aware Placement Design

- Status: Draft (awaiting user spec review)
- Date: 2026-05-04
- Owner: @kam6y
- Branch: `phase-7` (from `develop` after Phase 6 merge)
- 前提仕様: `docs/superpowers/specs/2026-05-03-rxn-config-sidecar-design.md`
- 後方互換性: **完全に放棄する**。TOML schema・`meta.json`・内部 API すべて breaking change を許容。

## 1. 目的

現状の初期配置パイプライン (Phase 3/4 で確立した Tier 1 directional / Tier 2 planar face dispatch + MMFF prescreen) は、対応反応ごとに「裏側」「平面法線」とハードコードした方向に対し ±30° の cone で 8 通り振っているだけで、汎用性に欠ける。本フェーズで:

- Tier 1 / Tier 2 dispatch を **完全に廃止** し、anchor を中心とした **全球面 (4π sr) サンプリング + steric blocking filter** という単一アルゴリズムに統合する。
- 配置距離の固定値 `FRAGMENT_SEPARATION = 3.5 Å` を撤廃し、**fragment 形状に応じた per-direction `d_min`** に置換する。これにより SN1 step 2 の tBu⁺ + Cl⁻ のような大小差のあるペアでも、SN2 のような小規模ペアでも、同じコードで「ぶつからない最短距離」に置ける。
- **MMFF prescreen を全廃** する。`reactx/prescreen.py` ごと削除し、blocking filter で生き残った候補をすべて UMA full relax に流す。MMFF parameterize 失敗時の fallback 経路 (proton transfer の HCl ケース等) もまるごと消え、wall-clock のばらつきが減る。
- `formed` / `broken` 情報の役割を「anchor 特定 + 拘束力構築」に絞る。chemistry prior としての方向決定はしない。

## 2. Non-goals

明示的に **このフェーズではやらない** こと:

- **Cycloaddition / metathesis 対応**: 1 つの formed bond で 2 fragment を bridge するケース (= 現在 6 反応すべて) のみスコープ。formed≥2 の同時 multi-anchor alignment、broken が fragment 間を bridge する metathesis は引き続き `NotImplementedError`。
- **Fragment の内部回転**: 配置は **平行移動のみ**。incoming fragment の orientation は ETKDG が決めたまま。relax で吸収する (現状と同じ方針)。
- **動的 N チューニング**: anchor 周辺の混雑度に応じた N_candidates 自動調整は将来課題。
- **NEB refine の対象拡大**: 現状の「`len(formed)==1 and len(broken)==1` のみ」guard はそのまま。
- **`.rxn.toml` のセクション構造大改造**: `[sampling]` セクション内の rename と `[prescreen]` セクション削除のみ。それ以外は不変。
- **vdW テーブルの再選定**: Alvarez (2013) Dalton Trans. 42, 8617 を継続採用 (Phase 4 で `blender/render.py` が選んだものと統一)。

## 3. アーキテクチャ

### 3.1 新パイプライン

```
.rxn + .rxn.toml
  → rxn_parser + load_config → ReactionConfig
  → BondChanges (formed / broken, 0-indexed)
            │
            ▼
[各 fragment を個別 3D 化 (現状維持)]
   ETKDGv3 + MMFFOptimizeMolecule (重原子 ≥2 のとき)
            │
            ▼
[Substrate / non-substrate 判定 (placement.py)]
   formed が fragment 間を bridge → 大きい方が substrate
   bridging 無し / 1 fragment → unimolecular: 配置スキップ
            │
            ▼
[Generic placement (placement.py)]
   1. Fibonacci 球面で N=n_candidates の単位ベクトル候補を生成
   2. blocking filter (角度シャドウ + d_min ceiling)
   3. 生存候補ごとに per-direction d_min を計算
   4. fragment を anchor + d × d_min(d) - incoming_anchor だけ平行移動
            │
            ▼
[全生存候補を UMA full relax (path_relax.relax_with_restraints)]
            │
            ▼
[scoring.select_best_trial → best trial]
   trajectory.xyz / energies.json / meta.json
```

### 3.2 既存からの主要変更

| 項目 | 現状 (Phase ≤ 6) | 新 (Phase 7) |
|---|---|---|
| 配置 dispatch | Tier 1 directional / Tier 2 planar face / Phase 5+ NotImplemented | 単一の sphere sampling + blocking |
| 方向決定 | `formed`/`broken` から chemistry prior (裏側 / 平面法線) | 全球面均等サンプル、prior なし |
| Sampling 範囲 | 30° cone (Fibonacci) | 4π sr (Fibonacci) |
| 配置距離 | `FRAGMENT_SEPARATION = 3.5 Å` 固定 | per-direction `d_min`、`max_proj_substrate(+d) + max_proj_incoming(-d) + GAP(0.5)` |
| Steric 判定 | 無し (盲目的に置く) | 角度シャドウ + d_min ceiling の二段 |
| MMFF prescreen | 8→3 trial 絞り込み | 完全削除 |
| 全 trial UMA relax | top-K = 3 のみ | blocking 生存全件 |
| Substrate 判定 | broken atoms 含む / 最大 fragment (Tier で分岐) | 最大 fragment に統一 (tie 時は最小 atom index 含む方) |

## 4. コンポーネント詳細

### 4.1 新規モジュール: `reactx/placement.py`

embed3d.py から placement ロジックを切り出して独立モジュール化する。

```python
def sample_sphere_directions(n: int, seed: int = 0) -> list[np.ndarray]:
    """Fibonacci 球面で全 4π sr を均等カバーする n 個の単位ベクトルを返す。

    Index 0 は決定論的に +z (再現性のため、phase 起点)。残り n-1 は黄金比螺旋で
    球面上に配置。seed で螺旋の位相 (phase) を回す。
    """

def compute_d_min(
    d: np.ndarray,
    anchor_pos: np.ndarray,
    substrate_positions: np.ndarray,    # anchor 含む / 含まないどちらでも (anchor の自己寄与は max に勝てない範囲で OK)
    substrate_vdw: np.ndarray,
    incoming_positions: np.ndarray,
    incoming_vdw: np.ndarray,
    incoming_anchor_pos: np.ndarray,
    *,
    gap: float = 0.5,
) -> float:
    """方向 d 上の最短安全距離。

    max_substrate_fwd = max((p - anchor_pos) · d + r_vdW for p, r_vdW in substrate)
    max_incoming_back = max((p - incoming_anchor_pos) · (-d) + r_vdW for p, r_vdW in incoming)
    d_min = max_substrate_fwd + max_incoming_back + gap
    """

def evaluate_direction(
    d: np.ndarray,                          # 単位ベクトル
    anchor_pos: np.ndarray,                 # (3,)
    substrate_positions: np.ndarray,        # anchor を除外した substrate atoms (M, 3)
    substrate_vdw: np.ndarray,              # (M,)
    incoming_positions: np.ndarray,
    incoming_vdw: np.ndarray,
    incoming_anchor_pos: np.ndarray,
    *,
    gap: float = 0.5,
    d_min_ceiling: float = 8.0,
) -> tuple[bool, float, str | None]:
    """角度シャドウ + d_min ceiling を順に判定し (blocked, d_min, reason) を返す。

    Step 1: 角度シャドウ
      ∃i s.t. angle(d, substrate_positions[i] - anchor_pos)
             < atan(substrate_vdw[i] / ||substrate_positions[i] - anchor_pos||)
      該当すれば blocked=True, d_min=NaN, reason="angle_shadow:atom_index=K"
    Step 2: 角度シャドウを抜けたら compute_d_min を実行
      d_min > d_min_ceiling → blocked=True, reason="d_min_ceiling:value=V"
      それ以外               → blocked=False, reason=None

    順序にしたがって早期 return することで角度シャドウ時の compute_d_min 計算を省く。
    """

@dataclass(frozen=True)
class PlacementTrial:
    """1 つの生存方向と、その方向で配置された positions を一緒に持つ。"""
    direction: np.ndarray   # (3,) 単位ベクトル
    d_min: float            # この方向で適用した距離
    positions: np.ndarray   # (N, 3) 平行移動後の全原子座標

@dataclass(frozen=True)
class PlacementResult:
    trials: list[PlacementTrial]   # 生存候補のみ
    n_candidates: int              # 入力 N (meta.json 用)
    n_blocked: int                 # blocked された数
    blocked_reasons: list[str]     # 長さ n_candidates、生存は None、ブロックは reason
                                   # (debug / log 用、meta.json には出さない)

def valid_placements(
    mol_h: Chem.Mol,
    frag_indices: tuple[tuple[int, ...], ...],
    positions: np.ndarray,             # 各 fragment 個別 embed 直後の (N, 3)
    bond_changes: BondChanges,
    *,
    n_candidates: int = 64,
    seed: int = 0,
    gap: float = 0.5,
    d_min_ceiling: float = 8.0,
) -> PlacementResult:
    """生存方向それぞれの positions を含む PlacementResult を返す。

    - 1 fragment のみ: trials=[PlacementTrial(direction=+z, d_min=0, positions=positions)]
      の単一要素 (unimolecular passthrough)
    - bridging formed が存在しない (formed = 0 で 2+ fragment): NotImplementedError
      (現状の SN1 dissoc は 1 fragment なので一致)
    - bridging formed あり: substrate 判定 → anchor 特定 → 球面サンプル → evaluate_direction
      → per-direction d_min → fragment 平行移動

    Survivors == 0 → RuntimeError。
    Survivors < n_candidates // 4 → log.warning、続行。
    """

def build_atoms_from_positions(
    mol_h: Chem.Mol,
    positions: np.ndarray,
) -> Atoms:
    """mol_h の symbols + formal charges + 与えられた positions から ase.Atoms を組む。

    embed3d.embed_mol_to_atoms 内で行っていた Atoms 構築ロジックを切り出した helper。
    placement の各 PlacementTrial.positions から最終 Atoms を作るときに使う。
    """

def _identify_substrate(
    frag_indices: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    """最大 fragment を substrate とする。tie 時は最小 atom index を含む方。"""

def _find_bridging_formed(
    formed: tuple[tuple[int, int], ...],
    substrate: set[int],
    fragment: set[int],
) -> tuple[int, int]:
    """fragment と substrate を bridge する formed bond を返す。
    複数なら NotImplementedError("multi-anchor placement / cycloaddition")。
    なければ ValueError。"""
```

### 4.2 新規モジュール: `reactx/vdw_radii.py`

Alvarez (2013) Dalton Trans. 42, 8617 の vdW 半径テーブルを `placement.py` と `blender/render.py` の双方から参照する shared モジュール。現状は `blender/render.py` 内にハードコード。

```python
# reactx/vdw_radii.py (新規)

_ALVAREZ_2013_VDW: dict[int, float] = {
    1: 1.20,  # H
    2: 1.43,  # He
    # ... Z=1..83 全部 (現状の render.py 値をそのまま移行)
    83: 2.07, # Bi
}

_FALLBACK_RADIUS = 1.50  # Z>83 のときのフォールバック (Alvarez 中央値近傍)

def vdw_radius(atomic_number: int) -> float:
    """Alvarez 2013 vdW 半径 (Å)。Z > 83 は warning + 1.50 Å fallback。"""

def vdw_radii_array(atomic_numbers: np.ndarray | list[int]) -> np.ndarray:
    """ベクトル化版。配列を返す。"""
```

### 4.3 改変: `reactx/embed3d.py`

責務を「単分子 / 単 fragment の 3D 座標生成」に絞り、placement 決定と Atoms 組み立てを `placement.py` に渡す。

```python
def embed_fragments_to_positions(
    mol: Chem.Mol,
    *,
    seed: int = 0xC0FFEE,
) -> tuple[Chem.Mol, tuple[tuple[int, ...], ...], np.ndarray]:
    """各 fragment を独立に ETKDGv3 + MMFF で 3D 化。

    Returns:
      mol_h         — Chem.AddHs(mol)
      frag_indices  — Chem.GetMolFrags(mol_h)
      positions     — (N, 3) 各 fragment の世界座標。fragment 同士は重なっている可能性あり
                      (placement.valid_placements が後段で平行移動を担当する)。
    """

# embed_mol_to_atoms は廃止。caller は以下の 3 関数を直接組み合わせる:
#   1. mol_h, frag_indices, positions = embed_fragments_to_positions(mol)
#   2. result = placement.valid_placements(mol_h, frag_indices, positions, bond_changes, ...)
#   3. atoms = placement.build_atoms_from_positions(mol_h, result.trials[i].positions)
#
# 旧 embed_mol_to_atoms にあった BFGS pre-relax (calculator + fmax) は廃止
# (UMA model load を embed 段階で誘発しないため、かつ trial-level relax と二重になるため)。
```

### 4.4 改変: `reactx/scoring.py`

`prescreen.py:select_top_k_indices` を `scoring.py` に移動 + 「最終 best 1 件選択」用の wrapper を追加。

```python
def select_best_trial(trials: list[TrialResult]) -> int:
    """reached_product=True 群の最低 peak_energy → 無ければ全体の最低 peak_energy。
    trials が空のとき ValueError。"""
```

### 4.5 改変: `reactx/cli.py`

```python
# 旧
rotations = sample_attack_rotations(cfg.sampling.n_angles, cfg.sampling.cone_half_deg, seed)
trials_atoms = [embed_mol_to_atoms(mol, rotation_perturbation=R, ...) for R in rotations]
prescreen_result = prescreen_trials(trials_atoms, mol_h, ...)
kept_indices = prescreen_result.kept
trials_to_uma = [trials_atoms[i] for i in kept_indices]
# UMA relax on trials_to_uma...

# 新
mol_h, frag_indices, base_positions = embed_fragments_to_positions(mol, seed=seed)
result = valid_placements(
    mol_h, frag_indices, base_positions, bond_changes,
    n_candidates=cfg.sampling.n_candidates, seed=seed,
)
trials_atoms = [build_atoms_from_positions(mol_h, t.positions) for t in result.trials]
trials_directions = [t.direction for t in result.trials]
# UMA relax on ALL trials_atoms (no prescreen)
trial_results = [relax_with_restraints(a, ...) for a in trials_atoms]
best_idx = select_best_trial(trial_results)
# meta.json に result.n_candidates / n_blocked / trials[].direction を書き込み
```

### 4.6 削除されるファイル / 関数

- `reactx/prescreen.py` — **ファイルごと削除** (`RDKitMMFFCalculator`, `MMFFParameterizationError`, `make_mol_from_atoms`, `prescreen_trials`, `select_top_k_indices`, `PrescreenResult` dataclass を含む)
- `reactx/trials.py:sample_attack_rotations` — 削除 (`placement.sample_sphere_directions` が後継)。`reactx/trials.py` がこの関数だけのファイルだった場合はファイルごと削除。
- `reactx/embed3d.py` 内:
  - `embed_mol_to_atoms` (廃止 — `embed_fragments_to_positions` + `placement.valid_placements` + `placement.build_atoms_from_positions` の組み合わせに置換)
  - `_place_fragments`
  - `_directional_placement`
  - `_planar_face_placement`
  - `_plane_normal_at_anchor`
  - `_find_substrate_fragment`
  - `_find_substrate_by_size` → `placement._identify_substrate` に **移動 + 改名**
  - `FRAGMENT_SEPARATION`, `PLANE_FIT_TOLERANCE` 定数
  - `MAX_EMBED_RETRIES` は `embed_fragments_to_positions` 内に残置 (現状のまま)
  - `_embed_in_place` は `embed_fragments_to_positions` の private helper として残置

## 5. データフロー (SN2 を例に)

```
入力: examples/sn2.rxn + examples/sn2.rxn.toml
  formed=[[1,3]], broken=[[1,2]] (atom-map 番号)

→ ReactionConfig.bond_changes = BondChanges(formed=((0,2),), broken=((0,1),))
  (0-indexed: AddHs 後の heavy ordering)

→ mol_h = Chem.AddHs(mol)
  N_atoms = 8 (CH3Cl=6, OH=2)
  frag_indices = ((0,1,2,3,4,5), (6,7))

→ per-fragment ETKDGv3 + MMFF
  positions: (8, 3) 各 fragment は世界座標で重なっている可能性

→ placement.valid_placements(...)
  substrate = (0,1,2,3,4,5) (6 atoms vs 2 atoms)
  bridging formed = (0, 6)  # C ∈ substrate, O ∈ incoming
  anchor = 0 (C), incoming_anchor = 6 (O)

  directions = sample_sphere_directions(64, seed=0)

  for each d in directions:
    blocked, d_min, reason = evaluate_direction(d, anchor_pos, ..., gap=0.5, d_min_ceiling=8.0)
    if blocked:
      blocked_reasons.append(reason)
      continue
    new_positions = positions.copy()
    target = positions[anchor] + d * d_min
    new_positions[list(incoming_frag)] += target - positions[incoming_anchor]
    trials.append(PlacementTrial(direction=d, d_min=d_min, positions=new_positions))

  return PlacementResult(trials, n_candidates=64, n_blocked=len(blocked_reasons), ...)

  期待: SN2 では back hemisphere の ~28-36 directions が生存
        (front-side: Cl が cone 半径 ~46° で覆う → blocked)
        (側面: 3 H が ~48° の cones で覆う → blocked)
        (back: 3 H の cone の隙間 → 生存)

→ ~30 trials 全件を UMA full relax (path_relax.relax_with_restraints)

→ select_best_trial → trajectory.xyz, energies.json, meta.json 出力
```

## 6. エラーハンドリング

| 失敗モード | 検知箇所 | 挙動 |
|---|---|---|
| 全候補 blocked (anchor 完全埋設) | `valid_placements` 末尾 | `RuntimeError("anchor at atom X has no valid placement direction (all N_CANDIDATES blocked); substrate may be fully enclosed")` |
| 生存 < `n_candidates // 4` | 同上 | `log.warning("only K/N candidates survived blocking at anchor X; consider larger n_candidates or check substrate geometry")` 続行 |
| 非 substrate fragment に bridging formed bond なし | `_find_bridging_formed` | `ValueError("fragment F has no formed bond bridging to substrate; check atom mapping")` |
| 同一 fragment が複数 formed で bridge (cycloaddition) | 同上 | `NotImplementedError("multi-anchor placement (cycloaddition) is out of scope; got K formed bonds bridging fragment F")` |
| broken bond が fragment 間を bridge (metathesis) | `valid_placements` 入口 | `NotImplementedError("multi-substrate metathesis (broken bonds spanning fragments) is out of scope")` |
| ETKDG / MMFF fragment embed 失敗 | `_embed_in_place` (現状) | 現状と同じ `RuntimeError` |
| 元素が vdW テーブル外 (Z > 83) | `vdw_radius(Z)` | `log.warning + 1.50 Å fallback` |
| `n_candidates < 1` | TOML validation (`config.py`) | `ConfigError` |
| `n_candidates > 1` で unimolecular | TOML validation | auto-clamp to 1 + `log.warning` (現状の n_angles 動作を継承) |

UMA の trial 全件 fail (全部 reached_product=False) は `select_best_trial` が peak_energy 最小を選んで返す (現状と同じ挙動)。

## 7. テスト計画

### 7.1 新規テストファイル

**`tests/test_placement.py`**:

- `test_sample_sphere_returns_unit_vectors` — 全要素 `||d|| == 1 ± 1e-9`
- `test_sample_sphere_uniform_coverage` — 最近接ペアの最大角度が Fibonacci 球面の理論値以下
- `test_sample_sphere_seed_determinism` — 同 seed → 全要素一致
- `test_sample_sphere_index_zero_is_z` — `directions[0] == [0, 0, 1]`
- `test_direction_blocked_atom_in_line` — anchor=(0,0,0), d=+x, 障害物原子 (0.5, 0, 0) で blocked
- `test_direction_unblocked_clear_path` — substrate atoms を −x 半球に固める、d=+x → unblocked
- `test_direction_blocked_by_d_min_ceiling` — 計算値 d_min=10, ceiling=8 → blocked、理由文字列確認
- `test_compute_d_min_sn2_geometry` — CH3Cl + OH backside で d_min が 3.4-3.7 Å の範囲
- `test_compute_d_min_planar_anchor_normal_vs_inplane` — sp² 中心で plane normal 方向 < in-plane 方向
- `test_valid_placements_sn2_backside_present` — SN2 で −Cl 方向 (cos > 0.85) が生存リストに含まれる
- `test_valid_placements_sn2_frontside_rejected` — Cl 方向 (+Cl 単位ベクトルとの cos > 0.7) が生存リストに含まれない
- `test_valid_placements_buried_anchor_raises_runtime` — 全方向 blocked 構造で `RuntimeError`
- `test_valid_placements_unimolecular_passthrough` — 1 fragment → 入力 positions そのまま 1 要素
- `test_valid_placements_cycloaddition_not_implemented` — formed が同じペアを 2 本 → `NotImplementedError`
- `test_valid_placements_metathesis_not_implemented` — broken が fragment 間を bridge → `NotImplementedError`

**`tests/test_vdw_radii.py`**:

- `test_vdw_radii_alvarez_z1_to_z83_present` — Z=1..83 全部キーあり
- `test_vdw_radius_fallback_for_z_above_83` — `vdw_radius(84)` が warn + 1.50 を返す
- `test_vdw_radii_array_vectorized` — `vdw_radii_array([1, 6, 8])` が `[1.20, 1.77, 1.50]`

### 7.2 変更テストファイル

**`tests/test_embed3d.py`**:

- 削除: `test_directional_placement_*` / `test_planar_face_placement_*` / `test_plane_normal_*` (関数自体が消える)
- 削除: `test_embed_mol_to_atoms_*` 全件 (`embed_mol_to_atoms` 廃止)
- 残置 (改名): `embed_fragments_to_positions` の単体テストとして:
  - `test_embed_fragments_unimolecular_returns_single_frag_positions`
  - `test_embed_fragments_multifrag_returns_separate_positions` — 各 fragment 独立に embed されること
  - `test_embed_fragments_seed_determinism`

**`tests/test_scoring.py`** (`select_best_trial` の単体テスト追加):

- `test_select_best_trial_prefers_reached_product` — reached=True 群を優先
- `test_select_best_trial_falls_back_to_lowest_peak_when_none_reached` — 全 reached=False の場合
- `test_select_best_trial_empty_raises` — `ValueError`

### 7.3 削除テストファイル

- `tests/test_prescreen.py` — **ファイルごと削除**
- `tests/test_trials.py` の `sample_attack_rotations` 系テスト → `tests/test_placement.py` の Fibonacci 球面テストに統合 (関数自体が消えるため)

### 7.4 Slow integration tests (`pytest -m slow`)

既存の slow integration tests (`@pytest.mark.slow` がつくテスト関数群、`tests/` 配下のどこに置かれているかは実装時に確認) — 6 反応すべて pass を維持:

- `test_sn2`, `test_proton_transfer`, `test_menshutkin`, `test_e2`, `test_sn1_dissoc`, `test_sn1_recomb`
- `meta.json.prescreen.*` を assert している箇所は削除
- `meta.json.trials[].rotation_deg` を assert している箇所は `direction` (3-vec) に書き換え
- wall-clock の数値 assert (現状あれば) は緩和または削除
- 各反応で「期待される配置 (例: SN2 で OH⁻ が C-Cl の裏側) が selected_trial の trajectory[0] に出ている」を検証する geometric assert を 1 本ずつ追加する (現在の cone 30° 制約で暗黙に保証されていたものを明示化)

### 7.5 Blender smoke test (`pytest -m blender`)

- `vdw_radii` の参照先が `reactx.vdw_radii` に切り替わるが、半径テーブルの値は変えないので render は数値的に同一になる。smoke test は無変更で pass を維持する。

## 8. TOML schema 変更 + examples 修正

### 8.1 schema breaking changes

```toml
# 旧 (Phase 6)
[sampling]
n_angles = 8
cone_half_deg = 30.0

[prescreen]
enabled = true
keep = 3
steps = 30

# 新 (Phase 7)
[sampling]
n_candidates = 64
# cone_half_deg は削除 (常に 4π sr)
# [prescreen] セクションごと削除
```

### 8.2 修正対象 (examples 全 6 件)

| ファイル | 修正内容 |
|---|---|
| `examples/sn2.rxn.toml` | `n_angles` → `n_candidates`、`cone_half_deg` 削除、`[prescreen]` 削除 |
| `examples/proton_transfer.rxn.toml` | 同上 |
| `examples/menshutkin.rxn.toml` | 同上 |
| `examples/e2.rxn.toml` | 同上 |
| `examples/sn1_dissoc.rxn.toml` | 同上 (`n_candidates = 1` を残す: unimolecular auto-clamp) |
| `examples/sn1_recomb.rxn.toml` | 同上 |

### 8.3 `reactx/config.py` 改変

- `SamplingConfig`: `n_angles: int` → `n_candidates: int`、`cone_half_deg: float` → **削除**
- `PrescreenConfig` データクラス → **削除**
- `ReactionConfig.prescreen` フィールド → **削除**
- バリデーション:
  - `n_candidates >= 1` (現 `n_angles` と同じ)
  - unimolecular で `n_candidates > 1` → 1 に auto-clamp + `log.warning` (現状継承)
  - `[prescreen]` キーが存在 → `ConfigError("[prescreen] section is removed in Phase 7; delete it from <path>")` (誤った旧 TOML を黙って受けない)

## 9. `meta.json` schema 変更

```jsonc
// 旧 (Phase 6)
{
  "selected_trial": 0,
  "prescreen": {
    "enabled": true,
    "kept": [0, 3, 5],
    "skipped": [1, 2, 4, 6, 7],
    "mmff_failed": false,
    "wall_clock_seconds": 0.8
  },
  "trials": [
    {"trial_idx": 0, "rotation_deg": 0.0, "reached_product": true, ...}
  ]
}

// 新 (Phase 7)
{
  "selected_trial": 0,
  "n_candidates": 64,
  "n_blocked": 33,
  "n_valid": 31,
  "trials": [
    {"trial_idx": 0, "direction": [0.0, 0.0, 1.0], "reached_product": true, ...}
  ]
}
```

- `prescreen` キー削除
- `n_candidates` / `n_blocked` / `n_valid` を追加
- `trials[].rotation_deg` (scalar) → `trials[].direction` (3-vec)
- それ以外のキー (`peak_energy`, `n_steps`, `wall_clock_seconds` 等) は不変

## 10. `blender/render.py` 改変

- ハードコードされた Alvarez 2013 vdW テーブル → `from reactx.vdw_radii import vdw_radius`
- 数値は同一なので render は bit-exact ではなくとも視覚的に同じ
- 結合棒判定の Cordero 共有結合半径テーブルはこのファイルでは触らない (別の `bond_changes.py` 等にあれば、テーブル自体は phase 7 では touch しない)
- `REACTX_VDW_SCALE` 環境変数の挙動は不変

## 11. README 更新

- 「対応反応」表の `n_angles` 列を `n_candidates` に rename、値を `64` に (SN1 dissoc は `1`)
- 「Per-reaction `.rxn.toml` config」例から `cone_half_deg` / `[prescreen]` 例を削除
- 「Wall-clock (実測)」表を Phase 7 で再計測した値に更新 (実装後)
- 「MMFF94 prescreen は実測上 ...」段落 → 削除
- 「アーキテクチャ」ASCII 図の `prescreen` ノード → 削除
- 「動作確認」項目の「`meta.json.prescreen.mmff_failed=true`」記述 → 削除
- 「方針と限界」セクションに「初期配置は全球面サンプル + steric blocking で決定。anchor が完全に埋まった場合は RuntimeError」を追加

## 12. パラメータ default 値 + 根拠

| パラメータ | default | 根拠 |
|---|---|---|
| `n_candidates` | `64` | 全球面 (4π sr) を Fibonacci で 64 等分 → solid angle ≈ 0.196 sr/point。SN2 backside cone (~0.84 sr) に概ね 4 candidate 入る粒度で、cycloaddition / 表側 attack のような未知方向まで網羅的に探索できる。blocking で ~50% 生存と仮定し UMA full relax ~30 件 ≒ 300 s + UMA load 25 s ≒ 325 s。現状 SN2 (~54 s, prescreen 経由) より大幅に長いが、汎用性 + 全方向検証を優先する設計判断。wall-clock を下げたければ TOML で 32 / 16 等に減らせる。 |
| `cone_half_deg` (削除) | — | 全球サンプルなので不要 |
| `gap` | `0.5 Å` | RDKit MMFF 出力の典型的な座標誤差 + relax 初期に拘束力で吸収できる距離。Phase 0 の Phase 0 baseline で `FRAGMENT_SEPARATION = 3.5 Å` だったが、このうち vdW 部分を `d_min` に分離すると残差として 0.3-0.5 Å が経験的に妥当。 |
| `d_min_ceiling` | `8.0 Å` | 8 Å を超えると Hookean restraint で結合形成距離 (~1.5 Å) まで引き寄せるのに > 200 step かかる経験則。anchor が壁に隠れているとほぼ確実にこれを超える。 |
| vdW テーブル fallback | `1.50 Å` | Alvarez 2013 の中央値 (Z=1..83 の median ~1.5)。Z>83 (Po, Fr, Ra, アクチノイド) は UMA omol25 訓練外なので実用上発火しない。 |

## 13. 実装順 (writing-plans への申し送り)

おおむね以下の依存順:

1. `reactx/vdw_radii.py` 新規 + `tests/test_vdw_radii.py` (依存なし、最初)
2. `reactx/placement.py` 新規 (`sample_sphere_directions` / `compute_d_min` / `evaluate_direction` / `PlacementTrial` / `PlacementResult` / `valid_placements` / `build_atoms_from_positions` / `_identify_substrate` / `_find_bridging_formed`) + `tests/test_placement.py` (`vdw_radii` に依存)
3. `reactx/scoring.py` に `select_best_trial` 追加 (`prescreen.py` の `select_top_k_indices` ロジック移植 → 1 件選択 wrapper) + `tests/test_scoring.py` 追加
4. `reactx/embed3d.py` 改変:
   - `embed_fragments_to_positions` 抽出 (現 `embed_mol_to_atoms` の per-fragment embed 部分のみ)
   - `embed_mol_to_atoms`, `_place_fragments`, `_directional_placement`, `_planar_face_placement`, `_plane_normal_at_anchor`, `_find_substrate_fragment`, `_find_substrate_by_size`, `FRAGMENT_SEPARATION`, `PLANE_FIT_TOLERANCE` 削除
5. `reactx/cli.py` 改変:
   - `embed_fragments_to_positions` + `valid_placements` + `build_atoms_from_positions` の orchestration
   - prescreen 呼び出し削除
   - meta.json schema を新形式 (`n_candidates` / `n_blocked` / `n_valid` / `trials[].direction`) に更新
6. `reactx/config.py` 改変: `n_angles` → `n_candidates` rename、`cone_half_deg` 削除、`PrescreenConfig` 削除、`[prescreen]` キー検知時に `ConfigError`
7. `reactx/prescreen.py` 削除 + `tests/test_prescreen.py` 削除
8. `reactx/trials.py` の `sample_attack_rotations` 削除 (ファイル全体が空ならファイルごと削除)
9. `examples/*.rxn.toml` 全 6 件修正 (`n_angles` → `n_candidates`、`cone_half_deg` 削除、`[prescreen]` 削除)
10. `blender/render.py` の vdw テーブル参照差し替え (`from reactx.vdw_radii import vdw_radius`)
11. `tests/test_embed3d.py` 改変 (`embed_fragments_to_positions` の単体テストに置換)
12. Slow integration tests 修正 + 新規 geometric assert 追加 (各反応 1 本ずつ)
13. README 更新 (アーキテクチャ図、対応反応表、MMFF prescreen 段落削除、方針と限界の追記)
14. Wall-clock 再実測 → README 数値更新

各ステップで unit test pass を確認しつつ進める。
