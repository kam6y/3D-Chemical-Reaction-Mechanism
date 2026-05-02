# reactx — Phase Re1 Design (Multi-Angle + Artificial Force Path Engine)

- Status: Draft (approved sections 1–7, awaiting user spec review)
- Date: 2026-04-27
- Owner: @kam6y
- Branch: `phase-Re1` (from `phase-0-spike` 78ab68f)
- 前提仕様: `docs/superpowers/specs/2026-04-20-reactx-phase-0-design.md`
- 旧 Phase 1 (`phase-1` ブランチ) は **破棄**。本 spec が Phase 1 の代替

## 1. 目的

「**正確な TS エネルギー**」ではなく「**妥当なアニメーション**」を目的とすると明示的に再定義したうえで、以下の 3 つの未確定論点を SN2 反応に絞って固める:

1. **初期構造**: `.rxn` から MMFF 単発で 1 つ生成するだけで足りるか、複数試行が必要か
2. **人工力**: 形成結合に引力 / 切断結合に斥力をかけて relaxation を駆動するアプローチが成立するか
3. **多角度試行**: bimolecular 反応で求核剤の入射角を複数試して best を選ぶアプローチが成立するか

これらを 1 つのパイプラインに統合した **Multi-Angle + Artificial Force Path Engine** を Phase Re1 のコアとする。NEB は default で外し、wall-clock 短縮を主目的とする (旧 Phase 0 の 1 回検証に時間がかかりすぎた問題への直接の回答)。

評価のため SN2 の他に **proton transfer (HCl + NH₃ → Cl⁻ + NH₄⁺)** を 2 反応目として通す。両方とも bimolecular で形成 1 / 切断 1 の同型反応のため、同じ pipeline で対応可能。

## 2. 対象反応

| 反応 | 例 | broken | formed | 備考 |
|---|---|---|---|---|
| SN2 (Phase 0 既存) | CH₃Cl + F⁻ → CH₃F + Cl⁻ | C–Cl | C–F | Walden 反転 |
| Proton transfer (新規) | HCl + NH₃ → Cl⁻ + NH₄⁺ | H–Cl | N–H | 評価用、`r_form=1.05` |

任意反応への汎用化 (E2 / SN1 step1 / E1 step2 / generic bond-change engine) は **非スコープ** (Phase 2 以降)。

## 3. パイプライン全体像

```
.rxn ──> rxn_parser ──> embed3d (rotation_offset 引数追加, reactant 側のみ N 回)
                                        │
                                        ▼
                          trials.py: cone 内で N 角度をサンプリング
                                        │
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                       trial_1       trial_2  ...   trial_N
                          │             │             │
                  artificial_force.py: Hookean (引力) + PullApart (斥力)
                          │             │             │
                  path_relax.py: FIRE で制約付き relaxation, frame を毎 stride 保存
                          │             │             │
                          └─────────────┼─────────────┘
                                        ▼
                          scoring.py: 到達判定 + peak energy で best 1 を選択
                                        │
                                        ▼
                          (optional) neb.py: 5–7 images で smoothing  ← --neb-refine
                                        │
                                        ▼
                                 trajectory.xyz ──> blender/render.py
```

## 4. モジュール構成

### 4.1 新規モジュール

#### 4.1.0 `reactx/bond_changes.py`

```python
@dataclass(frozen=True)
class SimpleBondChanges:
    formed: tuple[int, int]    # (a, b) 形成される結合の expanded mol_h index (reactant 座標系)
    broken: tuple[int, int]    # (a, c) 切断される結合の expanded mol_h index (reactant 座標系)

def compute_simple_bond_changes(
    reactant_mol_h: Chem.Mol,
    product_mol_h: Chem.Mol,
    heavy_mapping: dict[int, int],
) -> SimpleBondChanges:
    """形成 1 + 切断 1 の elementary step 限定で bond changes を返す。
    それ以外の topology (formed/broken の数が 1 でない) は NotImplementedError。"""
```

**Phase Re1 のスコープ限定**: 旧 Phase 1 の `compute_bond_changes` (任意反応) ではなく、**形成 1・切断 1 のみ** を扱う最小版。SN2 と proton transfer は両方ともこの形 (SN2: 形成 C-Nu, 切断 C-LG / proton transfer: 形成 N-H, 切断 H-Cl)。それ以外 (E2 の 2 形成 2 切断、SN1 の 0 形成 1 切断 など) は `NotImplementedError` を投げて Phase 2 に持ち越す。

実装方針:

- AddHs 後の `mol_h` を両側で受け取る (atom 数は両側で一致)
- H を含む完全な expanded mapping を内部で構築 (`heavy_to_hydrogen_groups` を Phase 0 から再利用)
- 結果の `a, b, c` は **reactant 側の expanded index** で返す (placement と整合)
- bond order 変化 (例: 1→2) は本 spec ではサポートしない (SN2 / proton transfer ともに必要なし)

これにより `embed3d._find_c_lg_bond` の SN2 ハードコードを置き換える。proton transfer は `formed=(N, H), broken=(H, Cl)` で同じ pipeline を通る。

#### 4.1.1 `reactx/trials.py`

```python
from scipy.spatial.transform import Rotation

def sample_attack_angles(
    n: int,
    cone_half_deg: float = 30.0,
    seed: int = 0,
) -> list[Rotation]:
    """ideal direction まわりの cone 内に擬均等にサンプリングされた n 個の rotation を返す。
    n=1 のとき rotation は identity (= ideal direction そのもの、Phase 0 と同一挙動)。"""
```

実装:

- Fibonacci spiral で球面上に擬均等点を配置 → `cone_half_deg` 以内の点のみ採用
- 各点について「z 軸 → その点」へ向ける rotation を返す
- `seed` で再現性を担保

#### 4.1.2 `reactx/artificial_force.py`

```python
from ase.constraints import Hookean

def build_restraints(
    atoms: ase.Atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    *,
    r_form: float | dict[tuple[str, str], float] = 1.6,
    r_broken: float = 4.0,
    k_form: float = 5.0,
    k_broken: float = 3.0,
) -> list:
    """Hookean (引力) + カスタム PullApart (斥力) constraint のリストを返す。
    r_form が dict のときは元素ペアで lookup、scalar のときは全 formed bond に同値適用。"""
```

カスタム `PullApart` constraint (~30 行):

- ASE の `FixConstraint` を継承
- `adjust_forces` で距離 < `rt` のとき `-k * (r - rt)` の反発力を出す
- `adjust_positions` は no-op

内部 `r_form` 推定 dict (8 ペアで初期化、不足分は 1.6 fallback):

```python
DEFAULT_R_FORM = {
    ("C", "F"): 1.39, ("C", "Cl"): 1.78,
    ("C", "N"): 1.47, ("C", "O"): 1.43,
    ("C", "C"): 1.54, ("C", "H"): 1.09,
    ("N", "H"): 1.01, ("O", "H"): 0.97,
}
```

#### 4.1.3 `reactx/path_relax.py`

```python
def relax_with_restraints(
    atoms: ase.Atoms,
    restraints: list,
    calc: Calculator,
    *,
    max_steps: int = 100,
    fmax: float = 0.1,
    traj_stride: int = 5,
) -> tuple[list[ase.Atoms], list[float]]:
    """FIRE で制約付き relaxation。frames + energies を返す。"""
```

- ASE `FIRE` (BFGS より制約系で安定、`dt=0.05, a=0.1` で慣性低め)
- 毎 `traj_stride` step で frame snapshot、最後のフレームは必ず含める
- calc は呼び出し側で生成・共有する (`calculators.get_calculator` を pipeline 全体で 1 個)

#### 4.1.4 `reactx/scoring.py`

```python
@dataclass
class TrialResult:
    trial_idx: int
    rotation_deg: float       # ideal direction からの偏角
    frames: list[ase.Atoms]
    energies: list[float]
    reached_product: bool
    peak_energy: float
    n_steps: int

def reached_product(
    final_atoms: ase.Atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    *,
    form_tol: float = 0.3,
    broken_tol: float = 0.5,
) -> bool: ...

def score_trials(results: list[TrialResult]) -> TrialResult:
    """到達 trial 中で peak_energy 最小を選択。全滅時は『最も product 寄り』を返す
    (= broken/formed 距離の合計偏差が最小)。"""
```

### 4.2 変更モジュール

#### 4.2.1 `reactx/embed3d.py`

変更:

- `_place_nucleophile_backside(...)` に `rotation_offset: Rotation = Rotation.identity()` を追加し、既存の `incoming_direction` 計算後に `rotation_offset.apply(incoming_direction)` を 1 行挿入
- `embed_mol_to_atoms(...)` のシグネチャに `bond_changes: SimpleBondChanges`, `rotation_offset: Rotation = Rotation.identity()` を追加
- **substrate fragment の同定方法を変更**: Phase 0 は「fragment 0 = 最大 fragment = substrate」と仮定していたが、これは proton transfer (NH3 が最大だが substrate は HCl) で破綻するため、**「`bond_changes.broken` の両端原子を両方とも含む fragment」を substrate として動的に決定**する。残りの fragment が nucleophile 側
- `_find_c_lg_bond` は **削除** (`bond_changes.py` に役割移譲)
- Phase 0 の SN2 ケースでは substrate = CH3Cl (broken=C-Cl が両方入る fragment) で従来通りの挙動を維持。proton transfer では substrate = HCl (broken=H-Cl 両方を含む) と判定される
- `test_embed3d.py` は `bond_changes` 引数追加 + `rotation_offset=Rotation.identity()` で SN2 が同一出力 (regression 維持)

#### 4.2.2 `reactx/cli.py`

主要書き換え。pipeline は §3 のフローに張り替え。

新 CLI:

```bash
reactx run examples/sn2.rxn -o out/ \
  [--backend uma|mace] \
  [--n-angles 8] \
  [--cone-half-deg 30.0] \
  [--seed 0] \
  [--r-form 1.6] [--r-broken 4.0] \
  [--k-form 5.0] [--k-broken 3.0] \
  [--max-relax-steps 100] \
  [--relax-fmax 0.1] \
  [--traj-stride 5] \
  [--neb-refine] \
  [--neb-images 7] \
  [--render] [--blender-exe /path/to/blender]
```

`--r-form` が指定されていなければ §4.1.2 の dict から元素ペアで自動推定。`--r-form` が scalar 指定されたら全 formed bond に適用 (debug 用)。

### 4.3 default 値の根拠表

| flag | default | 根拠 |
|---|---|---|
| `--n-angles` | 8 | Fibonacci spiral で cone 30° を覆うのに十分。N=4 sparse、N=16 wall-clock 倍 |
| `--cone-half-deg` | 30 | SN2 backside attack の経験的許容偏角 |
| `--r-form` | 内部 dict (元素ペア依存) | C-F=1.39 など平衡距離 |
| `--r-broken` | 4.0 Å | Phase 0 の `d_form=3.5 Å` と整合 |
| `--k-form` | 5.0 eV/Å² | 結合形成方向の駆動力、ASE 標準で「強め」だが過大ではない |
| `--k-broken` | 3.0 eV/Å² | 形成より弱め。先に「引っ張り合い」で硬直するのを避ける |
| `--max-relax-steps` | 100 | 8 trials × 100 step ≈ Phase 0 NEB (1500 single-point) の半分強 |
| `--relax-fmax` | 0.1 eV/Å | NEB 0.05 より緩い (path 生成目的) |
| `--traj-stride` | 5 | 100 step / 5 = 20 frames/trial |
| `--neb-refine` | off | アニメーション目的では不要 |
| `--neb-images` | 7 | refine 用、Phase 0 の 15 より少ない |

## 5. データフロー詳細

| step | 入力 | 処理 | 出力 |
|---|---|---|---|
| 1. parse | `.rxn` | `rxn_parser.parse_rxn` | `mol_h_reactant`, `mol_h_product`, `heavy_mapping` |
| 2. bond change 抽出 | mol_h + mapping | `bond_changes.compute_simple_bond_changes` (形成 1 / 切断 1 限定) | `SimpleBondChanges(formed=(a, b), broken=(a, c))` |
| 3. r_form 推定 | `formed`, mol_h_reactant | 内部 dict lookup | `r_form_per_bond` |
| 4. angle sampling | n, cone, seed | `trials.sample_attack_angles` | `rotations: list[Rotation]` |
| 5. embed (×N) | mol_h + rotation | `embed3d.embed_mol_to_atoms` | `atoms_init_list` |
| 6. relax (×N) | atoms + restraints + calc | `path_relax.relax_with_restraints` | `frames_list`, `energies_list` |
| 7. score | trial 全件 | `scoring.score_trials` | best `TrialResult` |
| 8. (optional) NEB | best frames[0], frames[-1] | `neb.run_neb` | refined frames |
| 9. 出力 | best frames + meta | xyz writer | `trajectory.xyz`, `energies.json`, `meta.json` |

`meta.json` 拡張:

```json
{
  "backend": "uma",
  "converged": true,
  "selected_trial": 3,
  "trials": [
    {"trial": 0, "reached_product": true,  "peak_energy": -1234.56, "n_steps": 87, "rotation_deg": 0.0},
    {"trial": 1, "reached_product": true,  "peak_energy": -1230.12, "n_steps": 95, "rotation_deg": 12.5}
  ],
  "wall_clock_seconds": 28.4,
  "neb_refined": false
}
```

並列性: 各 trial は独立だが UMA calculator が GPU を占有するため **逐次実行**。MMFF prescreen による絞り込みは Phase 2。

エラー伝播:

- step 5 で 1 trial の embed 失敗 → trial を skip、`meta.json` に `embed_error`
- step 6 で SCF 不収束 → `reached_product=False, peak_energy=inf` で記録
- step 7 で全 trial 失敗 → CLI exit 1、`meta.json` を残してデバッグ可能に
- 部分成功 → 成立分の中で best 採用、warning ログ

## 6. テスト戦略

### 6.1 ユニット (高速、CI 対象)

| ファイル | 内容 |
|---|---|
| `tests/test_trials.py` | (1) `n=8, cone=30°` の rotation がすべて 30° 以内 (2) `n=1` で identity (3) seed 固定で再現性 |
| `tests/test_artificial_force.py` | (1) Hookean で形成結合方向に力 (距離 > rt) (2) PullApart で broken bond に反発 (距離 < rt) (3) restraint 0 個で力ゼロ |
| `tests/test_path_relax.py` | UMA を使わず toy LJ calc で: (1) 2 原子系で目標距離に収束 (2) `traj_stride=5` で正しい数の frame (3) `max_steps` 到達で停止 |
| `tests/test_scoring.py` | (1) `reached_product` の判定 (2) 全成立中 peak 最小が選ばれる (3) 全滅時の fallback |
| `tests/test_default_r_form.py` | element pair → r_form dict の table-driven テスト |
| `tests/test_bond_changes.py` | (1) SN2 で `formed=(C, F), broken=(C, Cl)` (2) proton transfer で `formed=(N, H), broken=(H, Cl)` (3) E2 など formed/broken が 1 でない reaction で `NotImplementedError` |
| `tests/test_embed3d.py` (既存変更) | `rotation_offset=Rotation.identity()` で Phase 0 と bit-exact 同一出力 |

### 6.2 統合 (`@pytest.mark.slow`、UMA 必須)

| ファイル | 内容 |
|---|---|
| `tests/test_re1_sn2.py` | (1) `--n-angles 8` で SN2 が `reached_product=True` の trial を 1 つ以上含む (2) selected trial の trajectory で C-F が単調短縮、C-Cl が単調伸長 (3) F-C-Cl 角度の最大値 ≥ 120° |
| `tests/test_re1_proton_transfer.py` | (1) `r_form=1.05` で実行成功 (2) trajectory で H が Cl→N に移動 |
| `tests/test_wallclock_sn2.py` | SN2 default の `wall_clock_seconds` ≤ 60 (ローカル基準、CI 除外) |
| `tests/test_neb_refine_sn2.py` | `--neb-refine` on で NEB 経路が走り `meta.json` の `neb_refined=true` |

### 6.3 Blender smoke (`@pytest.mark.blender`)

`tests/test_blender_smoke.py` を SN2 + proton_transfer の 2 反応で parametrize。

### 6.4 廃止するテスト

- `tests/test_neb_sn2.py` → `test_re1_sn2.py` に置換 (NEB が default 経路から外れたため)

## 7. Examples

- `examples/sn2.rxn` (Phase 0 既存、無変更)
- `examples/proton_transfer.rxn` (新規) — `HCl + NH₃ → Cl⁻ + NH₄⁺`

旧 Phase 1 で生成した sn1/e2/e1_step2 の `.rxn` および `examples_out/` は **取り込まない**。

## 8. README 更新

- Phase Re1 の使い方 (CLI 例 2 反応分)
- Multi-angle + artificial force pipeline の概要 (1 段落)
- Phase 0 NEB 経路との wall-clock 比較 (実測値を記載)
- `--neb-refine` の使いどころ (アニメーションの「保険」として on にする場合)

## 9. Phase Re1 完了条件 (Definition of Done)

1. **SN2 default 実行** が wall-clock 短縮を実現 (Phase 0 baseline 比 ≤ 50% を実測、README に記録)
2. **SN2 アニメーション** が Blender GUI で「F⁻ が backside から接近 → C 反転 → Cl⁻ 脱離」と視認可能
3. **proton transfer** が同じ pipeline で完了し、H が Cl→N に移動するアニメーションが視認可能
4. **テスト全件 passing**: fast unit + `-m slow` (UMA) + `-m blender` (ローカル)
5. **CLI flag が機能**: `--n-angles` / `--cone-half-deg` / `--neb-refine` を切り替え可能、`meta.json` に trial 全件の score
6. **README 更新済み**

## 10. 非スコープ (Phase 2 以降)

- generic bond-change engine (任意反応への汎用化)
- E2 / SN1 step1 / β-H elimination / E1 — 旧 Phase 1 の 5 反応セット
- 反応タイプの自動分類 / strategy registry
- MMFF prescreen による trial 削減
- 並列 UMA (multi-GPU)
- AFIR (Maeda 2010) の真の関数形 (今回は Hookean で代用)
- TS の正確なエネルギー / IRC / 振動解析
- 溶媒効果 / open-shell / radical
- xTB / GRRM23 アダプタ

## 11. リスクと緩和

| リスク | 影響 | 緩和 |
|---|---|---|
| restraint だけでは backside 経路を取らず横から attack する trial が多発 | アニメ不自然 | cone 30° で 8 角度 → 1 つでも backside 寄りが当たれば OK。全滅時 cone 45° に拡げて 1 回 retry |
| FIRE が restraint で振動・発散 | trial 失敗 | `dt=0.05, a=0.1` で慣性低め |
| 全 trial が `reached_product=False` | DoD 1, 3 失敗 | (a) `r_form` 狭めて retry (b) `k_form` 倍化 retry。1 回ずつフォールバック |
| `--neb-refine` を on にすると wall-clock が Phase 0 並みに戻る | 鈍重 | default off + README で明記 |
| Hookean がすでに r > rt の状態で十分な力を出さず動かない | path 生成停滞 | `k_form=5.0` で十分な経験値、ローカル実測でチューニング |
| 旧 Phase 1 ブランチからのチェリーピック誘惑 | スコープ汚染 | **取り込まない**、SN2 + proton_transfer のみ |
| Phase 0 SN2 の TS 角度 120° 閾値が relaxation trajectory ベースで達成できない | DoD test 失敗 | `test_re1_sn2.py` で「trajectory のいずれかのフレームで F-C-Cl 角度 ≥ 120°」と緩める |
| substrate fragment の動的同定 (broken bond 両端を含む fragment) が edge case で複数 fragment にまたがる | embed3d 失敗 | Phase Re1 のサンプル 2 反応では起こりえない (broken bond の両端は必ず同一 fragment 内)。それ以外が来たら明示的 `ValueError` を投げて Phase 2 への移譲を示す |

## 12. Phase 2 以降のロードマップメモ

- **generic bond-change engine** (旧 Phase 1 のリベンジ): SN2 ハードコード判定を `compute_bond_changes` に置き換える。Phase Re1 で固めた multi-angle + artificial force の上に乗せる
- **MMFF prescreen**: trial を MMFF で 8→3 に絞ってから UMA に渡す
- **並列 UMA**: multi-GPU 環境で trial 並列化
- **AFIR**: Hookean を Maeda の真の関数形に置き換える
- **multi-step 反応**: SN1/E1 全体を elementary step 連結で扱う
