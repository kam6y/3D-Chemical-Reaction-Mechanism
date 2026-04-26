# reactx — Phase 1 Design (Generic Bond-Change Geometry Engine)

- Status: Draft (approved for spec review)
- Date: 2026-04-26
- Owner: @kam6y
- Branch: `phase-1`
- 前提仕様: `docs/superpowers/specs/2026-04-20-reactx-phase-0-design.md`

## 1. 目的

Phase 0 で SN2 1 件に hardcode した embedding ロジック (`embed3d._find_c_lg_bond` / `_place_nucleophile_backside`) を、`.rxn` から幾何学的に決まる **Generic Bond-Change Geometry Engine** に置き換える。これによって SN1 / E2 / E1 / プロトン移動などの極性二分子反応 (および解離型 elementary step) を、反応分類の hardcode なしに同じパイプラインで処理できるようにする。

汎用化の方針は **Approach 2 (Generic Bond-Change Geometry Engine)**。`.rxn` の atom mapping から「破壊される結合 (broken bonds)」と「形成される結合 (formed bonds)」を抽出し、それを満たす幾何配置を最適化する。反応タイプの自動分類や strategy registry は Phase 1 では導入しない。

## 2. 対象反応

NEB は 1 saddle point (= 1 elementary step) を扱う前提なので、multi-step な SN1/E1 全体は対象外。各 elementary step を別 `.rxn` ファイルとして表現する。

| 反応 | 例 | broken | formed |
|---|---|---|---|
| SN2 (Phase 0 既存) | CH₃Cl + F⁻ → CH₃F + Cl⁻ | C–Cl | C–F |
| Heterolytic dissociation (SN1/E1 step 1) | (CH₃)₃C–Br → (CH₃)₃C⁺ + Br⁻ | C–Br | (なし) |
| E2 elimination (concerted) | CH₃CH₂Br + OH⁻ → CH₂=CH₂ + H₂O + Br⁻ | Cα–Br, Cβ–H | O–H, Cα–Cβ の bond order 1→2 |
| β-H elimination from cation (E1 step 2) | (CH₃)₃C⁺ → (CH₃)₂C=CH₂ + H⁺ | Cβ–H | Cα–Cβ の bond order 1→2 |
| Proton transfer | HCl + NH₃ → Cl⁻ + NH₄⁺ | H–Cl | H–N |

bond order 変化 (例: Cα–Cβ の 1→2) も "formed" として扱う (新規 π 結合が立ち上がる遷移と等価)。

## 3. アーキテクチャ変更

### 3.1 新規モジュール

#### 3.1.1 `reactx/reaction_topology.py`

```python
@dataclass(frozen=True)
class BondChange:
    a: int           # expanded-Hs (mol_h) index, reactant 側座標系
    b: int           # expanded-Hs (mol_h) index, reactant 側座標系
    order_before: float  # 0.0 if bond does not exist
    order_after: float   # 0.0 if bond is fully broken

@dataclass(frozen=True)
class BondChanges:
    broken: list[BondChange]   # order_after < order_before
    formed: list[BondChange]   # order_after > order_before

def compute_bond_changes(
    reactant_mol_h: Chem.Mol,   # AddHs 済み
    product_mol_h: Chem.Mol,    # AddHs 済み
    heavy_mapping: dict[int, int],   # reactant_heavy_idx -> product_heavy_idx (parse_rxn の戻り値)
) -> BondChanges: ...
```

実装方針:
- 引数は AddHs 後の `mol_h` を両側で受け取る (atom 数は両側で一致)。
- `heavy_mapping` から H を含む完全な expanded mapping を内部で構築 (`heavy_to_hydrogen_groups` を Phase 0 から再利用、H の対応は `align.align_product_to_reactant` の brute-force permutation と整合させる)。
- 結果の `BondChange.a` / `BondChange.b` は **reactant 側の expanded index** (`Chem.AddHs(reactant_mol)` の atom 順序) で返す。product 側の bond は expanded mapping を逆引きして reactant index に投影してから比較する。これによって placement / NEB / align の他モジュールと indexing が一貫する。
- bond order は RDKit の `BondType` を float で返す (SINGLE=1, DOUBLE=2, TRIPLE=3, AROMATIC=1.5)。Phase 1 では aromatic 系は対象外なので integer 値のみが現実的に出現する。`AROMATIC` を含む bond change が来た場合は明示的に `NotImplementedError`。

#### 3.1.2 `reactx/placement.py`

```python
def place_fragments_generic(
    mol_h: Chem.Mol,
    frag_indices: tuple[tuple[int, ...], ...],
    positions: np.ndarray,
    bond_changes: BondChanges,
    *,
    side: Literal["reactant", "product"],
    d_form: float = 3.0,    # Å — formed bond でのアプローチ距離
    d_dissoc: float = 4.0,  # Å — product 側で broken bond により分離するときの距離
) -> np.ndarray: ...
```

`embed3d` から呼び出される。fragment 0 (最大の substrate fragment) を固定し、他の fragment を bond_changes に基づいて配置する。

### 3.2 既存モジュール変更

#### 3.2.1 `reactx/embed3d.py`

- `_find_c_lg_bond` を削除。
- `_place_nucleophile_backside` を削除。
- `embed_mol_to_atoms` のシグネチャに `bond_changes: BondChanges` と `side: Literal["reactant", "product"]` を追加。fragment 数 ≥ 2 のとき `placement.place_fragments_generic` を呼び出す。
- `atoms.info["spin"]` は引き続き 1 (closed-shell) を default とする。Phase 1 では radical 推定はしない (Phase 2 のフック)。

#### 3.2.2 `reactx/cli.py`

- `--images` の default を bond change 数に応じて auto 計算: `max(11, 9 + 2 * len(broken) + 2 * len(formed))`。CLI で明示指定された場合は override。
- `parse_rxn` の戻り値から `compute_bond_changes` を呼び、`embed_mol_to_atoms` に渡す。
- 反応タイプ識別の CLI フラグは追加しない (Approach 2 を維持)。

#### 3.2.3 `reactx/neb.py`

- 変更なし (parameter は cli が計算して渡す)。

## 4. Generic Placement アルゴリズム詳細

### 4.1 Reactant 側

各 fragment i (i ≥ 1) について:

1. fragment i の原子と fragment 0 の原子を結ぶ **formed bond** をリストアップ → `formed_anchors_i = [(a ∈ frag_0, b ∈ frag_i, order_after), ...]`。
2. 各 (a, b) について b の理想位置を決定:
   - もし a が **broken bond** (a, c) を持ち、c ∈ frag_0 ならば: 背面攻撃方向を採用 → `ideal_b = p_a + d_form * (-unit(p_c - p_a))` (Phase 0 SN2 ロジックを一般化したもの)。
   - そうでなければ: fragment 0 重心から離れる方向 → `ideal_b = p_a + d_form * unit(p_a - centroid(frag_0))`。
3. fragment i 内に anchor (= b) が複数あれば、b の理想位置と現位置の対応から **Kabsch alignment (rotation + translation)** で fragment 全体を整列。anchor が 1 つなら translation のみ。
4. anchor が 1 つもなければ (例えば溶媒分子)、centroid を fragment 0 の重心から `+x` 方向に `d_form + max(box_radius_frag_0, box_radius_frag_i)` だけ離して配置し、警告ログを出す (`box_radius` = fragment の重心からの最大原子距離)。Phase 1 のサンプル反応では出現しないが、ロジックを定義しておく。

### 4.2 Product 側

各 fragment i (i ≥ 1) について:

1. fragment i の原子と fragment 0 の原子を結ぶ **broken bond** をリストアップ → `broken_anchors_i = [(a ∈ frag_0, b ∈ frag_i), ...]`。 (product 側の indexing は align モジュールでこの後 reactant に揃えられる)
2. 各 (a, b) について b の理想位置: `ideal_b = p_a + d_dissoc * unit(p_b_orig - p_a)`。元の embedding で b が a の近くにある (= MMFF が結合として配置した) はずなので、その方向を保ったまま距離だけ伸ばす。
3. Kabsch alignment は reactant 側と同じ。

### 4.3 H 原子の扱い

H 原子の扱いは Phase 0 と同じ (`align.align_product_to_reactant` 内の brute-force permutation)。bond_changes に H 関与の bond (例: O–H, H–Cl) があっても、H は `mol_h` の expand 後 index で reaction_topology に含まれる。

### 4.4 fragment 0 の選択

fragment 0 は **最大の (重原子数最多の) fragment** とする。同数の場合は最初に現れた fragment。Phase 0 と同じ慣習。

## 5. NEB パラメータの自動調整

| 反応 | broken | formed | auto images |
|---|---|---|---|
| SN2 | 1 | 1 | 13 |
| Heterolytic dissociation | 1 | 0 | 11 (= max(11, ...)) |
| E2 (concerted) | 2 | 2 | 17 |
| β-H elimination from cation | 1 | 1 | 13 |
| Proton transfer | 1 | 1 | 13 |

`--images` を CLI で明示すれば override。fmax / max-steps は default (0.05 / 500) のまま。

## 6. テスト戦略

### 6.1 ユニットテスト (高速、CI で回す)

- `tests/test_reaction_topology.py`: 5 反応で `compute_bond_changes` の結果が期待値通り (broken/formed の atom indices と order)。
- `tests/test_placement.py`: 5 反応で `place_fragments_generic` 後に:
  - formed bond の anchor 距離が `d_form ± 1.0 Å` (Kabsch alignment 後の補正で多少ずれる余地を許容)
  - broken bond の anchor 距離 (product 側) が `d_dissoc ± 1.0 Å`
  - SN2 のような背面攻撃 (broken-anchor あり) の場合、incoming 原子が `-unit(C→LG)` 方向に配置されている (内積 < -0.7)。
- `tests/test_cli.py`: `--images` 自動計算は cli から純粋関数として切り出した `recommend_n_images(bond_changes) -> int` を直接テストする (UMA や NEB は呼ばない)。

### 6.2 統合テスト (`@pytest.mark.slow`、UMA 必須)

- `tests/test_neb_sn2.py` (既存): Phase 0 の TS 角度 ≥ 120° テストが Phase 1 でも passing (regression 保護)。
- `tests/test_neb_dissociation.py` (新規): heterolytic dissociation で energy が saddle を経て product に至る (TS 周辺で energy がピークを示す)。
- `tests/test_neb_e2.py` (新規): E2 で TS で Cα–Br と Cβ–H の bond length が両方 1.3〜1.7× 平衡値の中間長を取る。
- `tests/test_neb_e1_step2.py` (新規): β-H elimination で TS で Cβ–H が伸び、Cα–Cβ が短くなる。
- `tests/test_neb_proton_transfer.py` (新規): proton transfer で TS で H が両 heavy atom の中点付近 (両距離が 1.0〜1.5 Å の中間)。

### 6.3 Blender smoke テスト

`tests/test_blender_smoke.py` (既存) を 5 反応の trajectory に対して順に実行できるよう拡張。`@pytest.mark.blender` でローカルのみ。

## 7. Examples

- `examples/sn2.rxn` (Phase 0 既存)
- `examples/sn1_step1.rxn` (新規) — `(CH3)3C-Br → (CH3)3C+ + Br-`
- `examples/e2.rxn` (新規) — `CH3CH2Br + OH- → CH2=CH2 + H2O + Br-`
- `examples/e1_step2.rxn` (新規) — `(CH3)3C+ → (CH3)2C=CH2 + H+`
- `examples/proton_transfer.rxn` (新規) — `HCl + NH3 → Cl- + NH4+`

各 `.rxn` には atom map 番号を手動付与 (Phase 0 の SN2 と同じ方針、`rxnmapper` 連携は Phase 2)。

## 8. README 更新

以下を追記:
- Phase 1 で対応する 5 反応の一覧と各 example のコマンド。
- Multi-step 反応 (SN1, E1) は elementary step ごとに `.rxn` を分けるという慣習の説明。
- bond change と generic placement の概要 (1 段落程度)。

## 9. Phase 1 完了条件 (Definition of Done)

1. **5 反応すべて**で `reactx run examples/<X>.rxn -o out/<X>/ --backend uma` が NEB `converged=True` で完了し、trajectory.xyz / energies.json / meta.json が生成される (heterolytic dissociation はピーク=平坦の可能性があるため、saddle が見つからない場合は §11 のリスク緩和に従い別判定: energy が単調増加 → 平衡 → 緩和してもよい、その場合は明示ログ + DoD 緩和判定)。
2. **5 反応すべて**で `blender --background --python blender/render.py -- out/<X>/trajectory.xyz out/<X>/scene.blend` が exit 0 で完了し、Blender GUI で開いて反応の本質的な動き (SN2 の Walden 反転 / SN1 の Br⁻ 解離 / E2 の Cα–Br 切断と Cβ–H 引き抜きと π 形成 / E1 step 2 の β-H 脱離 / proton transfer の H 移動) が視認できる。
3. `pytest` (高速ユニット) と `pytest -m slow` (統合) と `pytest -m blender` (smoke) が全件 passing。
4. Phase 0 SN2 テスト (`test_neb_sn2`) が regression なし。
5. README に Phase 1 の使い方と慣習が記載されている。

## 10. 非スコープ

- 反応タイプの自動分類 / strategy registry (Approach 1)。
- Multi-step 反応の自動連結 (1 つの `.rxn` に複数 elementary step を表現する拡張)。
- ラジカル / open-shell (UMA omol task は closed-shell 前提)。
- E2 の anti-periplanar geometry の能動的強制 (MMFF と UMA に任せる)。
- 溶媒効果、PCM、エントロピー補正。
- Curly arrow 表示。
- `rxnmapper` 統合 (atom map 自動補完)。
- xTB / GRRM23 アダプタ。

## 11. リスクと緩和

| リスク | 影響 | 緩和 |
|---|---|---|
| Heterolytic dissociation の TS が UMA で発見できない (NEB が saddle を見つけられない) | DoD §1, §2 が達成不能 | NEB 失敗時は best-effort 軌跡を保存、`meta.json` に `converged: false` を記録。Blender で「滑らかに離れる」軌跡が見えれば視覚的目的は達成 (実際の TS は IRC 次第)。 |
| E2 で MMFF が anti-periplanar 配置をしない場合、NEB が局所最小に落ちる | E2 demo 失敗 | Phase 1 では再 embed seed 変更を許容するフォールバックを実装 (ETKDG seed を 5 種試して MMFF energy 最低を採用)。 |
| `compute_bond_changes` で aromatic ring など bond order が non-integer のケース | 想定外の bond change が出る | Phase 1 のサンプル反応には aromatic を含めない。ただし aromatic を扱う `.rxn` が来たら明示的に `NotImplementedError` を投げる。 |
| Generic placement で fragment が衝突する (近すぎる初期配置) | UMA SCF 不収束 | placement 後に最近接重原子間距離をチェックし、`< 1.5 Å` ならフラグメントを更に離す再配置を 1 回試みる。 |
| 反応によっては 17 images でも TS 解像度が不足 | NEB 収束しない | `--images` で override 可。README に「収束しない場合は images 増やす」FAQ を記載。 |

## 12. Phase 2 以降のロードマップメモ

- 反応分類の自動推定 (Strategy registry を Approach 1 として導入し、generic engine と切り替え可能にする)。
- Multi-step 反応の自動連結 (1 つの `.rxn` から複数 elementary step を抽出 → NEB を連結)。
- ラジカル / open-shell サポート (`atoms.info["spin"]` の自動推定)。
- `rxnmapper` 統合。
- xTB / GRRM23 アダプタ。
- TS 付近のスローダウン再生 / bond fade (Phase 0 で先送りした項目)。
