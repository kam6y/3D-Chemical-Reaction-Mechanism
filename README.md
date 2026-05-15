# reactx — Generic Steric-Aware Reaction Path Engine (per-pair AFIR with sticky latch)

2D 反応機構 (`.rxn`) と sidecar TOML config から、反応点 (anchor) を中心とした全球面 (4π sr) Fibonacci サンプリング + steric blocking filter (角度シャドウ + d_min ceiling) + **per-pair AFIR force (sticky latch)** + FIRE 緩和で 3D 反応経路を探索し、Blender で ball-and-stick アニメーションを生成するパイプライン。

目的は妥当なアニメーション (正確な TS エネルギーは目標としない)。NEB refine はオプション。対応反応:

- **SN2** (`O⁻ + CH₃Cl` 等、1 formed + 1 broken)
- **Proton transfer** (`HCl + NH₃` 等、1 formed + 1 broken)
- **Menshutkin** (`NH₃ + CH₃Cl`、1 formed + 1 broken、中性 → イオン対)
- **E2 elimination** (1 formed + 2 broken)
- **SN1 step 1 解離** (0 formed + 1 broken、unimolecular)
- **SN1 step 2 recombination** (1 formed + 0 broken)
- **Diels–Alder** ([4+2] cycloaddition、`bridges == 2`、2 formed + 0 broken)
  - butadiene + ethylene → cyclohexene
  - cyclopentadiene + maleic anhydride → norbornene-2,3-dicarboxylic anhydride (endo/exo 両 trial 自動展開)

## Phase 9 changes

Phase 8 の経験的 `Hookean` (引力) + `PullApart` (斥力) 力場を **per-pair AFIR with sticky latch** に置き換えた:

- 力は per-pair に `F = α · d̂` (一定 magnitude、距離不変)。各 pair が threshold を満たした時点で **sticky latch** が ON になり、以降そのペアの AFIR 力は 0 (relax 終了まで OFF にならない)。全 pair latch 後は AFIR 力が完全消失し、自然に unbiased FIRE relax に移行する。
- ハイパラは反応あたり 4 個 (`k_form` / `k_broken` / `r_broken` / `r_form`) → **2 個** (`alpha_formed` / `alpha_broken`) に簡約。
- threshold (`r_formed_threshold` / `r_broken_threshold`) は AFIR latch 判定と `reached_product` 判定で **共有**。同じ閾値 = 同じ判定で論理的不整合がなくなる。
- TOML schema は `[restraints]` → `[afir]` + `[scoring]` に再構成。旧 `[restraints]` を含む config は `ConfigError` で reject される。
- `meta.json.trials[k]` に新フィールド: `formed_thresholds`, `broken_thresholds`, `product_distance_residual` (least-bad fallback の tiebreaker), `formed_latch_count` / `broken_latch_count`, `initial_latched_formed` / `initial_latched_broken` (debug 情報)。
- `reached_product` (最終 frame で全 pair が threshold 達成) を **唯一の成功判定** とし、latch state は debug 情報に降格 (latch ON でも UMA force が pair を threshold の外に押し戻して `reached_product=False` になりうるため)。

## Phase 10 changes

Phase 9 の SN2 実行で求核剤 OH⁻ が CH3Cl の裏側 (Walden 軸、Cl-C-O = 180°) ではなく 133° という不自然な角度から接近する path が選ばれていた問題を、**「AFIR 力を加える前に短時間 unbiased FIRE 緩和を挟む」** 物理的補正で解決した:

- `relax_with_restraints` を **2-stage 化**: Stage A (`pre_relax_steps` 回 constraints OFF で FIRE 緩和) → Stage B (既存の AFIR + sticky latch 緩和)。Stage A の最終 geometry は `final_state["frame_after_pre_relax"]` に保存され、`count_initial_latched` の評価基準として使われる。
- `[afir]` schema に `pre_relax_steps: int` を追加。**default は 30**。`0` で機能オフ (個別反応で副作用が出た場合の opt-out 用)。
- Stage B 開始時に FIRE optimizer を **再生成** (velocity リセット) し最適化の不連続を防ぐ。Stage A 末尾と Stage B 先頭の frame は同一 geometry で重複するが、可視化価値を優先して許容。
- 効果: SN2 で Cl-C-O 角度 ≥ 150° の Walden inversion 配置が安定して選ばれるようになった (Phase 9: 133°, Phase 10: 159–169°)。CH3Cl の Cl δ-/C δ+ 双極子が OH⁻ を裏側に引き寄せる ion-dipole 相互作用が、placement 由来の任意な開始方向を物理的に正しい pre-reaction complex に補正する。
- `meta.json.effective_params` に `pre_relax_steps` フィールドを追加。

詳細仕様: `docs/superpowers/specs/2026-05-15-phase-10-pre-relax-design.md`

## セットアップ

```bash
python -m pip install -e .[dev]
hf auth login    # UMA モデル取得のため
```

Blender 4.x と `atomic-blender-pdb-xyz` アドオンを別途インストールしておく。

## 使い方

```bash
# SN2 (description / formed / broken / params are all in examples/sn2.rxn.toml)
reactx run examples/sn2.rxn -o out/sn2/ --backend uma --render
# Proton transfer
reactx run examples/proton_transfer.rxn -o out/pt/ --backend uma --render
# Diels-Alder
reactx run examples/diels_alder_simple.rxn -o out/da/ --backend uma --render
reactx run examples/diels_alder_endo.rxn -o out/da_endo/ --backend uma --render
```

反応クラスごとの全パラメータは `examples/<name>.rxn.toml` に集約されている (下節 [Per-reaction .rxn.toml config](#per-reaction-rxntoml-config) 参照)。

主要フラグ (環境/出力依存のみ):

- `--backend {uma,lj}` (default: `uma`)
- `--model uma-m-1p1` (UMA model name)
- `--seed 0` (sampling 再現性デバッグ)
- `--relax-fmax 0.1`, `--traj-stride 5` (出力品質)
- `--neb-refine` + `--neb-images 7`
  - 1 formed + 1 broken 反応のみ対応 — 他は exit 2
- `--render` + `--blender-exe blender`

生成物:

- `out/<rxn>/trajectory.xyz` — best trial trajectory
- `out/<rxn>/energies.json` — best trial エネルギー列
- `out/<rxn>/meta.json` — trial 全件のスコア、placement 結果、wall_clock_seconds、neb_refined フラグ、Phase 9 で追加された `formed_thresholds` / `broken_thresholds` / `product_distance_residual` / `formed_latch_count` / `broken_latch_count` / `initial_latched_formed` / `initial_latched_broken`
- `out/<rxn>/scene.blend` — Blender シーン

## Per-reaction `.rxn.toml` config

各 `examples/<name>.rxn` には同階層に同名 stem の sidecar TOML (`<name>.rxn.toml`) を **必須で** 配置する。CLI は `<rxn_path>.toml` を機械的にロードし、結合変化情報 (`formed` / `broken`, atom-map 番号) と AFIR ハイパラ (`alpha_formed` / `alpha_broken` / `max_relax_steps` / `pre_relax_steps`) と scoring 閾値 (`r_broken_threshold` / `r_formed_threshold`) と sampling 設定をすべてここから取る。

`pre_relax_steps` は省略可 (default 30、Phase 10 で追加)。0 を指定すると pre-relax を skip して Phase 9 互換動作になる。

最小例 (`examples/sn2.rxn.toml`):

```toml
description = "SN2 anion: CH3Cl + OH- -> CH3OH + Cl-"
formed = [[1, 3]]
broken = [[1, 2]]

[afir]
alpha_formed = 4.0
alpha_broken = 2.5
max_relax_steps = 300

[scoring]
r_broken_threshold = 3.0
```

| `.rxn` | description | formed (map) | broken (map) | alpha_formed | alpha_broken | max_relax_steps | r_broken_threshold | r_formed_threshold | n_candidates |
|---|---|---|---|---|---|---|---|---|---|
| sn2.rxn | SN2 anion (`O⁻ + CH₃Cl`) | `[[1,3]]` | `[[1,2]]` | 4.0 | 2.5 | 300 | 3.0 | (default) | 64 |
| proton_transfer.rxn | Proton transfer (`HCl + NH₃`) | `[[1,3]]` | `[[1,2]]` | 2.0 | 5.0 | 300 | 3.0 | 1.5 | 64 |
| menshutkin.rxn | Menshutkin (`NH₃ + CH₃Cl`) | `[[1,5]]` | `[[5,9]]` | 4.0 | 4.0 | 300 | 3.0 | (default) | 64 |
| e2.rxn | E2 elimination | `[[4,5]]` | `[[2,5],[1,3]]` | 2.0 | `[1.5, 2.0]` | 200 | `[3.0, 4.0]` | (default) | 64 |
| sn1_dissoc.rxn | SN1 step 1 解離 | `[]` | `[[1,5]]` | (omitted) | 2.5 | 200 | 6.0 | (n/a) | **1** |
| sn1_recomb.rxn | SN1 step 2 recombination | `[[1,5]]` | `[]` | 1.5 | (omitted) | 200 | (n/a) | (default) | 64 |
| diels_alder_simple.rxn | DA: butadiene + ethylene | `[[1,5],[4,6]]` | `[]` | `[2.5, 2.5]` | (omitted) | 200 | (n/a) | (default) | 64 |
| diels_alder_endo.rxn | DA endo: CP + MA | `[[1,5],[4,6]]` | `[]` | `[2.5, 2.5]` | (omitted) | 250 | (n/a) | (default) | **16** |

`alpha_formed` / `alpha_broken` は scalar で全 pair 同値、list で per-pair 指定 (list の長さは対応する `formed` / `broken` の長さと一致を要求)。**Phase 9 では対応する pair set が空 (`formed = []` または `broken = []`) の場合、その α は省略可** (例: `sn1_dissoc` は `alpha_formed` 不要、`sn1_recomb` / DA は `alpha_broken` 不要)。空でない pair set に対して `α = 0` を指定するのは validation で禁止される (force ゼロを表現したい場合は pair から削除)。

`r_broken_threshold` は `broken` が非空のときに **必須** (default は提供されない)。`r_formed_threshold` は省略時 **`1.15 × Rsum_Cordero`** (Cordero 2008 共有結合半径の和) を per-bond でルックアップする。両方とも scalar | list[float] 両対応。

threshold は AFIR latch ON 判定と `reached_product` 判定の両方で同一値が使われる。AFIR force は per-pair の現在距離が threshold を満たした時点 (`r ≤ r_formed_threshold` または `r ≥ r_broken_threshold`) で **sticky latch** ON になり、その pair の force は relax 終了まで 0 になる。全 pair latch 後は AFIR 力が完全消失する。

`[sampling]` は省略可能で `n_candidates=64` がデフォルト。各反応とも反応点 (anchor) を中心とした全球面 (4π sr) Fibonacci サンプリングで初期方向を生成し、ステリック blocking (角度シャドウ + d_min ceiling) を通過した方向すべてを UMA で full relax する。混雑した anchor で生存数が少ない場合は `n_candidates` を上げる (例: ステリック制約の少ない DA simple では 64、12 原子の DA endo では trial 1 件 ~30s かかるため 16 に絞る)。

```bash
reactx run examples/sn2.rxn -o out/sn2/ --backend uma --render
reactx run examples/menshutkin.rxn -o out/men/ --backend uma --render
```

## 動作確認

各反応の最小確認手順:

1. `reactx run examples/sn2.rxn -o out/sn2/ --backend uma --render`
   → `meta.json` で `selected_trial >= 0`, `trials[].reached_product` ≥1 件 True, `trials[].formed_latch_count == 1`, `out/sn2/scene.blend` を Blender GUI で開いて Walden 反転を視認
2. `reactx run examples/proton_transfer.rxn -o out/pt/ --backend uma --render` → 同様に視認、`r_formed_threshold=1.5` で短い O-H 結合判定
3. `reactx run examples/menshutkin.rxn -o out/men/ --backend uma --render`
   → C–N 形成 ≤ 1.7 Å、C–Cl 切断 ≥ 3.0 Å を視認 (gas-phase で `alpha_formed=4.0` まで強くする必要あり)
4. `reactx run examples/e2.rxn -o out/e2/ --backend uma --render`
   → C–H と C–Cl の同時切断 + base (OH⁻) 接近を視認、`broken_latch_count` で per-bond の進行を確認
5. `reactx run examples/sn1_dissoc.rxn -o out/sn1d/ --backend uma`
   → unimolecular auto-clamp で `meta.json.trials` が 1 件、`trajectory.xyz` で C–Br 距離 ≥ 6.0 Å (= `r_broken_threshold`)
6. `reactx run examples/sn1_recomb.rxn -o out/sn1r/ --backend uma --render`
   → bimolecular で 64 候補、blocking 後の生存全件を UMA で full relax、Cl⁻ が tBu⁺ の平面方向から接近 → C–Cl 結合形成を視認 (default `r_formed_threshold` ≈ 1.15 × (0.75 + 1.02) Å)
7. `reactx run examples/diels_alder_simple.rxn -o out/da/ --backend uma --render`
   → meta.json で `placement_kind == "multi_anchor"`, `orientation` に "achiral" (ethylene C2 対称で縮約) が出ることを確認、最終フレームで C1-C5 ≤ 1.8 Å、C4-C6 ≤ 1.8 Å を視認
8. `reactx run examples/diels_alder_endo.rxn -o out/da_endo/ --backend uma --render`
   → meta.json で endo, exo 両 trial が出力、selected_trial の orientation を確認 (UMA の挙動次第で endo/exo どちらか) + 6-membered ring + bicyclic 構造の形成を視認
9. `pytest -m slow` で 8 反応すべての統合テストが pass (E2 の strict `reached_product` は現状 xfail、Phase 10 で再 tune 予定)

## レンダリング: 原子球サイズと結合棒

`blender/render.py` は `atomic-blender-pdb-xyz` アドオンで XYZ を読み込んだ後、`render.py` 側で 2 つの後処理を行う:

1. **原子球サイズ**: 各元素ボールの半径を **Alvarez (2013) *Dalton Trans.* 42, 8617 の van der Waals 半径 × 0.25** で上書き (ball-and-stick 風スケール)。比率は実際の vdW 半径比と一致する。`REACTX_VDW_SCALE` で全体倍率を上書き可 (例: `REACTX_VDW_SCALE=0.4 reactx run ...` で CPK 寄り)。
2. **結合棒**: アドオンの XYZ importer は結合を描画しないため、`render.py` が独自に Cordero (2008) 共有結合半径 (Phase 9 で `reactx/covalent_radii.py` に切り出し、Blender bundled Python から import 失敗時は vendored fallback table を使う) を用いた距離判定 (閾値 = `(rcov_A + rcov_B) × BOND_TOLERANCE`、デフォルト 1.1) で結合を抽出。各候補結合 1 本につきシリンダオブジェクトを 1 個生成し、フレームごとに transform (両端原子に追従) と表示/非表示 (距離が閾値以内かを毎フレーム再判定) を keyframe する。これにより SN2 のように形成・切断する結合は途中で出現・消滅する。

対応元素は UMA `omol` タスクの訓練範囲 = OMol25 = **Z=1 (H) 〜 Z=83 (Bi)** の連続 83 元素。Po (84) 以降、Fr/Ra および全アクチノイドは UMA 訓練外 → 入力 XYZ に出現しない想定。テーブル外の元素はリスケール対象外 (アドオン既定半径のまま) かつ結合棒も生成されない。

詳細仕様: `docs/superpowers/specs/2026-04-26-vdw-radii-design.md`

## 方針と限界

- 目的は妥当なアニメーション。TS エネルギーの正確さは保証しない
- 力場は per-pair AFIR (`F = α · d̂`、constant magnitude) + sticky latch。各 pair が threshold を満たした時点で latch ON、以降 AFIR 力 0。全 pair latch 後は完全に unbiased FIRE relax に移行する。trajectory は AFIR-biased path であり真の MEP/TS ではない (peak_energy は unbiased UMA energy だが approximate)。
- NEB refine は default off。`--neb-refine` は **1 formed + 1 broken 反応のみ対応** (E2 / SN1 dissoc / SN1 recomb / DA では CLI が exit code 2 で reject)。1+1 NEB の入口配置は引き続き `align_product_to_reactant` を使う (Phase 10 で AFIR endpoints との統合を検討)。
- E2 の strict `reached_product` (3 simultaneous bond changes) は現状 xfail。slow test では `selected_trial >= 0` までを assert し、reached_product の retune は Phase 10 待ち。
- 対応反応は上節「対応反応」を参照。中性 addition / metathesis などは未対応
- ラジカル / open-shell / 溶媒効果は対象外
- σ-only connectivity diff のため、結合次数変化 (single↔double) は明示的に追跡しない (π 形成は QM calculator に委ねる)
- 初期配置は反応点 (anchor) を中心とした全球面 (4π sr) Fibonacci サンプル + ステリック blocking (角度シャドウ + d_min ceiling) で決定する。anchor が完全に埋まった (全候補が blocked) 場合は `RuntimeError` で停止する
- 旧 `[restraints]` schema (`k_form` / `k_broken` / `r_broken` / `r_form`) は **`ConfigError` で reject** される。Phase 8 以前の TOML は新 schema に書き直す必要がある。
- `[prescreen]` セクション、`n_angles` キー、`cone_half_deg` キーは Phase 7 で廃止
- multi-anchor placement は `bridges == 2` (Diels-Alder) までを対応。`bridges >= 3` (一般 cycloaddition、1,3-dipolar 等) と cheletropic は未対応
- DA の endo/exo は配置時に 0°/180° の 2 値で離散化。連続 rotation を sample しない (kinetic vs thermodynamic の精密判定には不十分)

詳細仕様: `docs/superpowers/specs/2026-05-08-afir-force-design.md` (v3.1)

## Wall-clock (実測)

RTX 5070 Ti + UMA-m-1p1 で実測 (Blender 4.5 LTS で `--render` まで含む)。Phase 9 Stage 3 時点の値:

| Reaction | wall-clock | 備考 |
|---|---|---|
| SN1 dissoc (`examples/sn1_dissoc.rxn`) | ~40 s | unimolecular auto-clamp で n_candidates=1 |
| SN2 (`examples/sn2.rxn`) | (Phase 8: 232 s; Phase 9 retest pending) | `alpha_formed=4.0`, `alpha_broken=2.5`, `r_broken_threshold=3.0` |
| Proton transfer (`examples/proton_transfer.rxn`) | (Phase 8: 447 s; Phase 9 retest pending) | `alpha_formed=2.0`, `alpha_broken=5.0`, `r_formed_threshold=1.5` (短い O-H 結合判定) |
| Menshutkin (`examples/menshutkin.rxn`) | (Phase 8: 417 s; Phase 9 retest pending) | gas-phase で `alpha_formed=4.0` / `alpha_broken=4.0` |
| SN1 recomb (`examples/sn1_recomb.rxn`) | ~460 s | bimolecular、`alpha_formed=1.5` |
| E2 (`examples/e2.rxn`) | 829 s (~14 min) | 3 simultaneous bond changes、per-bond `alpha_broken=[1.5, 2.0]` / `r_broken_threshold=[3.0, 4.0]`、strict reached_product は xfail |
| Diels-Alder simple (`examples/diels_alder_simple.rxn`) | 1132 s (~19 min) | 64 候補、ethylene C2 対称で achiral 縮約、`alpha_formed=[2.5, 2.5]` |
| Diels-Alder endo (`examples/diels_alder_endo.rxn`) | 581 s (~10 min) | n_candidates=16 × 2 endo/exo、`alpha_formed=[2.5, 2.5]`, `max_relax_steps=250` |

UMA model load (~25–30 s) が固定コスト。blocking で生存した候補数 (`n_valid`) が wall-clock を直接決める。`n_candidates=64` で各反応の anchor 周辺ステリックにより 20–42 程度が生存し、それぞれ UMA で full relax する。混雑した anchor で生存が少ないとログ警告が出る (`only K/N candidates survived blocking ...`)。生存ゼロは `RuntimeError` で停止する。

`reached_product=True` 数が valid 数より少ないのは正常で、`scoring.score_trials` は (1) `reached_product=True` の中から peak_energy 最小、(2) 全 trial が `reached_product=False` の場合は `product_distance_residual` 最小の trial を least-bad fallback として選ぶ。

`--neb-refine` を on にすると NEB の収束に追加で 5–10 分かかる。アニメーション目的なら off 推奨。

各実行の trial 全件スコアと placement 結果 (`n_candidates` / `n_blocked` / `n_valid`)、Phase 9 新フィールド (`formed_thresholds` / `broken_thresholds` / `product_distance_residual` / `formed_latch_count` / `broken_latch_count` / `initial_latched_formed` / `initial_latched_broken`) と wall_clock_seconds は `out/<rxn>/meta.json` に残る。

## テスト

```bash
pytest                   # 高速ユニットテストのみ (slow / blender マーカーは除外)
pytest -m slow           # UMA 依存の 8 反応統合テスト (Phase 9 AFIR + sticky latch)
pytest -m blender        # Blender smoke test (ローカル環境のみ)
```

## アーキテクチャ

```
.rxn + .rxn.toml ─> rxn_parser + load_config ─> ReactionConfig
                                                       │
                                                       ▼
                                               BondChanges (formed/broken)
                                                       │
                                                       ▼
                                       embed_fragments_to_positions
                                       (per-fragment ETKDG + MMFF)
                                                       │
                                                       ▼
                       placement.valid_placements
                       ├─ 1 fragment      → unimolecular passthrough
                       ├─ bridges == 1     → single-anchor (Phase 7)
                       │      └─ Fibonacci on direction → blocking → trials
                       └─ bridges == 2     → multi-anchor (Phase 8)
                              ├─ Fibonacci on face direction
                              ├─ rigid-body alignment (translation + 2-point Kabsch)
                              ├─ endo/exo 0°/180° 展開、対称分子は achiral 縮約
                              └─ blocking: angular shadow + d_min ceiling + unreachable / asymmetric
                                                       │
                                                       ▼
              resolve formed_thresholds / broken_thresholds  (Cordero 2008 default for formed)
                                                       │
                                                       ▼
                              ┌── trial 1 ──┐
                              ├── trial 2 ──┤  build_afir_constraint(formed, broken,
                              ├── ...        │                       alpha_*, *_thresholds)
                              └── trial K ──┘
                                                       │
                                                       ▼
                              ┌─── 1-stage relax_with_restraints ─────────┐
                              │  FIRE + AFIRConstraint (sticky per-pair)  │
                              │   - r が threshold 達成 → latch ON、F=0   │
                              │   - 全 pair latched → unbiased UMA only   │
                              │  returns (frames, energies, final_state)  │
                              └───────────────────────────────────────────┘
                                                       │
                                                       ▼
                              reached_product (final frame, sole criterion)
                              product_distance_residual (least-bad tiebreaker)
                                                       │
                                                       ▼
                                              scoring → best trial
                                                       │
                                          (optional) neb refinement (1+1 only)
                                                       │
                                                       ▼
                                trajectory.xyz → blender/render.py → .blend
```

主要ファイル:

- `reactx/artificial_force.py` — `AFIRConstraint` (per-pair sticky latch) + `build_afir_constraint`
- `reactx/path_relax.py` — `relax_with_restraints` (1-stage、`(frames, energies, final_constraint_state)` の 3-tuple を返す)
- `reactx/scoring.py` — `reached_product` / `product_distance_residual` / `resolve_formed_thresholds` / `resolve_broken_thresholds` / `count_initial_latched`
- `reactx/covalent_radii.py` — Cordero (2008) 共有結合半径 (formed threshold default と Blender 結合棒で共用)
- `reactx/config.py` — `[afir]` + `[scoring]` schema、`[restraints]` 検出時 `ConfigError`

詳細設計: `docs/superpowers/specs/2026-05-04-generic-placement-design.md`
詳細設計: `docs/superpowers/specs/2026-05-05-phase-8-cycloaddition-design.md`
詳細設計: `docs/superpowers/specs/2026-05-08-afir-force-design.md` (Phase 9, v3.1)
実装計画: `docs/superpowers/plans/2026-05-08-afir-force-replacement.md` (Phase 9, v3)
