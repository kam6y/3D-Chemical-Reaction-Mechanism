# reactx — Generic Steric-Aware Reaction Path Engine

2D 反応機構 (`.rxn`) と sidecar TOML config から、反応点 (anchor) を中心とした全球面 (4π sr) Fibonacci サンプリング + steric blocking filter (角度シャドウ + d_min ceiling) + Hookean/PullApart 拘束 + FIRE 緩和で 3D 反応経路を探索し、Blender で ball-and-stick アニメーションを生成するパイプライン。

目的は妥当なアニメーション (正確な TS エネルギーは目標としない)。NEB refine はオプション。対応反応:

- **SN2** (`O⁻ + CH₃Cl` 等、1 formed + 1 broken)
- **Proton transfer** (`HCl + NH₃` 等、1 formed + 1 broken)
- **Menshutkin** (`NH₃ + CH₃Cl`、1 formed + 1 broken、中性 → イオン対)
- **E2 elimination** (1 formed + 2 broken)
- **SN1 step 1 解離** (0 formed + 1 broken、unimolecular)
- **SN1 step 2 recombination** (1 formed + 0 broken、Tier 2 plane-normal placement)

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
- `out/<rxn>/meta.json` — trial 全件の score, wall_clock_seconds, neb_refined フラグ
- `out/<rxn>/scene.blend` — Blender シーン

## Per-reaction `.rxn.toml` config

各 `examples/<name>.rxn` には同階層に同名 stem の sidecar TOML (`<name>.rxn.toml`) を **必須で** 配置する。CLI は `<rxn_path>.toml` を機械的にロードし、結合変化情報 (`formed` / `broken`, atom-map 番号) と物理パラメータ (`k_form` / `k_broken` / `r_broken` / `max_relax_steps` / 任意 `r_form`) と sampling/prescreen 設定をすべてここから取る。

最小例 (`examples/sn2.rxn.toml`):

```toml
description = "SN2 anion: CH3Cl + OH- -> CH3OH + Cl-"
formed = [[1, 3]]
broken = [[1, 2]]

[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
```

| `.rxn` | description | formed (map) | broken (map) | k_form | k_broken | r_broken (Å) | max_relax_steps | r_form | n_candidates |
|---|---|---|---|---|---|---|---|---|---|
| sn2.rxn | SN2 anion (`O⁻ + CH₃Cl`) | `[[1,3]]` | `[[1,2]]` | 0.5 | 1.0 | 4.0 | 100 | 元素表 | 64 |
| proton_transfer.rxn | Proton transfer (`HCl + NH₃`) | `[[1,3]]` | `[[1,2]]` | 0.5 | 1.0 | 4.0 | 100 | 1.05 | 64 |
| menshutkin.rxn | Menshutkin (`NH₃ + CH₃Cl`) | `[[1,5]]` | `[[5,9]]` | 2.0 | 2.0 | 5.0 | 200 | 元素表 | 64 |
| e2.rxn | E2 elimination | `[[4,5]]` | `[[2,5],[1,3]]` | 1.0 | 1.0 | 4.0 | 200 | 元素表 | 64 |
| sn1_dissoc.rxn | SN1 step 1 解離 | `[]` | `[[1,5]]` | 0.0 | 2.0 | 6.0 | 200 | — | **1** |
| sn1_recomb.rxn | SN1 step 2 recombination | `[[1,5]]` | `[]` | 1.0 | 0.0 | 4.0 | 200 | 元素表 (C-Cl 1.78) | 64 |

`r_form` は省略時に Cordero (2008) 共有結合半径表で per-bond ルックアップ、scalar で全 formed 同値、list で per-bond 指定。`[sampling]` は省略可能で `n_candidates=64` がデフォルト。各反応とも反応点 (anchor) を中心とした全球面 (4π sr) Fibonacci サンプリングで初期方向を生成し、ステリック blocking (角度シャドウ + d_min ceiling) を通過した方向すべてを UMA で full relax する。`n_candidates` を `[sampling]` で調整可能 (例: ステリックに混雑した anchor で生存数が少ない場合は大きく)。

```bash
reactx run examples/sn2.rxn -o out/sn2/ --backend uma --render
reactx run examples/menshutkin.rxn -o out/men/ --backend uma --render
```

## 動作確認

各反応の最小確認手順:

1. `reactx run examples/sn2.rxn -o out/sn2/ --backend uma --render`
   → `meta.json` で `selected_trial >= 0`, `trials[].reached_product` ≥1 件 True、`out/sn2/scene.blend` を Blender GUI で開いて Walden 反転を視認
2. `reactx run examples/proton_transfer.rxn -o out/pt/ --backend uma --render` → 同様に視認
3. `reactx run examples/menshutkin.rxn -o out/men/ --backend uma --render`
   → C–N 形成 ≤ 1.7 Å、C–Cl 切断 ≥ 3.5 Å を視認
4. `reactx run examples/e2.rxn -o out/e2/ --backend uma --render`
   → C–H と C–Cl の同時切断 + base (OH⁻) 接近を視認
5. `reactx run examples/sn1_dissoc.rxn -o out/sn1d/ --backend uma`
   → unimolecular auto-clamp で `meta.json.trials` が 1 件、`trajectory.xyz` で C–Br 距離 ≥ 4.5 Å
6. `reactx run examples/sn1_recomb.rxn -o out/sn1r/ --backend uma --render`
   → bimolecular で 64 候補、blocking 後の生存全件を UMA で full relax、Cl⁻ が tBu⁺ の平面方向から接近 → C–Cl 結合形成を視認
7. `pytest -m slow` で 6 反応すべての統合テストが pass

## レンダリング: 原子球サイズと結合棒

`blender/render.py` は `atomic-blender-pdb-xyz` アドオンで XYZ を読み込んだ後、`render.py` 側で 2 つの後処理を行う:

1. **原子球サイズ**: 各元素ボールの半径を **Alvarez (2013) *Dalton Trans.* 42, 8617 の van der Waals 半径 × 0.25** で上書き (ball-and-stick 風スケール)。比率は実際の vdW 半径比と一致する。`REACTX_VDW_SCALE` で全体倍率を上書き可 (例: `REACTX_VDW_SCALE=0.4 reactx run ...` で CPK 寄り)。
2. **結合棒**: アドオンの XYZ importer は結合を描画しないため、`render.py` が独自に Cordero (2008) 共有結合半径を用いた距離判定 (閾値 = `(rcov_A + rcov_B) × BOND_TOLERANCE`、デフォルト 1.1) で結合を抽出。各候補結合 1 本につきシリンダオブジェクトを 1 個生成し、フレームごとに transform (両端原子に追従) と表示/非表示 (距離が閾値以内かを毎フレーム再判定) を keyframe する。これにより SN2 のように形成・切断する結合は途中で出現・消滅する。

対応元素は UMA `omol` タスクの訓練範囲 = OMol25 = **Z=1 (H) 〜 Z=83 (Bi)** の連続 83 元素。Po (84) 以降、Fr/Ra および全アクチノイドは UMA 訓練外 → 入力 XYZ に出現しない想定。テーブル外の元素はリスケール対象外 (アドオン既定半径のまま) かつ結合棒も生成されない。

詳細仕様: `docs/superpowers/specs/2026-04-26-vdw-radii-design.md`

## 方針と限界

- 目的は妥当なアニメーション。TS エネルギーの正確さは保証しない
- NEB refine は default off。`--neb-refine` は **1 formed + 1 broken 反応のみ対応** (E2 / SN1 dissoc / SN1 recomb では CLI が exit code 2 で reject)
- 対応反応は上節「対応反応」を参照。中性 addition / cycloaddition / metathesis / Diels–Alder などは未対応
- ラジカル / open-shell / 溶媒効果は対象外
- σ-only connectivity diff のため、結合次数変化 (single↔double) は明示的に追跡しない (π 形成は QM calculator に委ねる)
- 初期配置は反応点 (anchor) を中心とした全球面 (4π sr) Fibonacci サンプル + ステリック blocking (角度シャドウ + d_min ceiling) で決定する。anchor が完全に埋まった (全候補が blocked) 場合は `RuntimeError` で停止する
- `[prescreen]` セクション、`n_angles` キー、`cone_half_deg` キーは Phase 7 で廃止。古い TOML を読ませると `ConfigError` で reject される

詳細仕様: `docs/superpowers/specs/2026-05-03-rxn-config-sidecar-design.md`

## Wall-clock (実測)

| Reaction | wall-clock | 備考 |
|---|---|---|
| (Phase 7 で再実測予定) | — | n_candidates=64 default、prescreen 撤廃で全 blocking 生存を UMA full relax |

UMA model load (~25–30 s) が固定コストとして wall-clock を支配する。Phase 7 では MMFF prescreen を廃止したため、blocking で生存した候補の数 (n_valid) が wall-clock を直接決める。`n_candidates=64` で各反応の anchor 周辺ステリックにより 8–32 程度が生存し、それぞれ UMA で full relax する想定。混雑した anchor で生存が少ないとログ警告が出る (`only K/N candidates survived blocking ...`)。生存ゼロは `RuntimeError` で停止する。

`--neb-refine` を on にすると NEB の収束に追加で 5–10 分かかる (`test_neb_refine_sn2` で実測 ~7 分)。アニメーション目的なら off 推奨。

各実行の trial 全件スコアと prescreen 結果と wall_clock_seconds は `out/<rxn>/meta.json` に残る。

## テスト

```bash
pytest                   # 高速ユニットテストのみ (slow / blender マーカーは除外)
pytest -m slow           # UMA 依存の 6 反応統合テスト
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
                                       (4π sr Fibonacci + 角度シャドウ + d_min ceiling)
                                                       │
                                                       ▼
                              ┌── trial 1 ──┐
                              ├── trial 2 ──┤  FIRE + Hookean/PullApart restraints
                              ├── ...        │  (生存全件を UMA full relax)
                              └── trial K ──┘
                                                       │
                                                       ▼
                                              scoring → best trial
                                                       │
                                          (optional) neb refinement
                                                       │
                                                       ▼
                                trajectory.xyz → blender/render.py → .blend
```

詳細設計: `docs/superpowers/specs/2026-05-04-generic-placement-design.md`
実装計画: `docs/superpowers/plans/2026-05-04-generic-placement.md`
