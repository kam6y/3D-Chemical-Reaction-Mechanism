# reactx — Phase Re1 Multi-Angle Path Engine

2D 反応機構 (`.rxn`) から多角度サンプリング + Hookean/PullApart 拘束 + FIRE 緩和で 3D 反応経路を探索し、Blender で ball-and-stick アニメーションを生成するパイプライン。

**Phase Re1 の目標**: 正確な TS エネルギーではなく、妥当なアニメーション。NEB はオプション。対応反応: SN2 / proton transfer。

**Phase 3** で multi-bond 反応 (E2 elimination = 1 formed + 2 broken / SN1 step 1 解離 = 0 formed + 1 broken) に拡張。

**Phase 4** で SN1 step 2 cation + nucleophile recombination (1 formed + 0 broken) を Tier 2 plane-normal placement で追加。

> **Breaking changes (Phase 6, develop ← phase-6):**
> - CLI フラグ `--reaction-type` および `--k-* / --r-* / --max-relax-steps / --n-angles / --cone-half-deg / --no-mmff-prescreen / --prescreen-keep / --prescreen-steps` は **全削除**。各反応の設定は `<rxn_path>.toml` (sidecar TOML) に書く。
> - `meta.json.reaction_type` キー → 削除、代わりに `meta.json.description` (TOML の `description` 値そのまま) を書く。
> - `reactx.presets` モジュール削除、`reactx.bond_changes.compute_bond_changes` 削除。`BondChanges.from_atom_map_pairs(formed_map, broken_map, atom_map_to_idx)` を使用。

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
- `--neb-refine` + `--neb-images 7` (1 formed + 1 broken のみ対応)
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

| `.rxn` | description | formed (map) | broken (map) | k_form | k_broken | r_broken (Å) | max_relax_steps | r_form | n_angles |
|---|---|---|---|---|---|---|---|---|---|
| sn2.rxn | SN2 anion (`O⁻ + CH₃Cl`) | `[[1,3]]` | `[[1,2]]` | 0.5 | 1.0 | 4.0 | 100 | 元素表 | 8 |
| proton_transfer.rxn | Proton transfer (`HCl + NH₃`) | `[[1,3]]` | `[[1,2]]` | 0.5 | 1.0 | 4.0 | 100 | 1.05 | 8 |
| menshutkin.rxn | Menshutkin (`NH₃ + CH₃Cl`) | `[[1,5]]` | `[[5,9]]` | 2.0 | 2.0 | 5.0 | 200 | 元素表 | 8 |
| e2.rxn | E2 elimination | `[[4,5]]` | `[[2,5],[1,3]]` | 1.0 | 1.0 | 4.0 | 200 | 元素表 | 8 |
| sn1_dissoc.rxn | SN1 step 1 解離 | `[]` | `[[1,5]]` | 0.0 | 2.0 | 6.0 | 200 | — | **1** |
| sn1_recomb.rxn | SN1 step 2 recombination | `[[1,5]]` | `[]` | 1.0 | 0.0 | 4.0 | 200 | 元素表 (C-Cl 1.78) | 8 |

`r_form` は省略時に Cordero (2008) 共有結合半径表で per-bond ルックアップ、scalar で全 formed 同値、list で per-bond 指定。`[sampling]` / `[prescreen]` は省略可能で、それぞれ `n_angles=8 cone_half_deg=30.0`、`enabled=true keep=3 steps=30` がデフォルト。

```bash
reactx run examples/sn2.rxn -o out/sn2/ --backend uma --render
reactx run examples/menshutkin.rxn -o out/men/ --backend uma --render
```

> **Phase 6 で削除されたフラグ:** `--reaction-type`, `--k-form`, `--k-broken`, `--r-form`, `--r-broken`, `--max-relax-steps`, `--n-angles`, `--cone-half-deg`, `--no-mmff-prescreen`, `--prescreen-keep`, `--prescreen-steps`。これらはすべて `.rxn.toml` 側で指定する。

## Phase Re1 + Phase 3 + Phase 4 動作確認

DoD は以下の手順で確認する:

1. `reactx run examples/sn2.rxn -o out/sn2/ --backend uma --render` を実行 → `meta.json` の `selected_trial >= 0`, `trials[].reached_product` で少なくとも 1 件 True を確認
2. `out/sn2/scene.blend` を Blender GUI で開いて Walden 反転を視認
3. `reactx run examples/proton_transfer.rxn -o out/pt/ --backend uma --render` を実行 → 同様に視認
4. `pytest -m slow` で SN2 + proton_transfer 統合テストが pass
5. `reactx run examples/e2.rxn -o out/e2/ --backend uma --render` を実行
   → `meta.json` で `selected_trial >= 0`, `reached_product=True` の trial が ≥1 件、`out/e2/scene.blend` で C–H と C–Cl の同時切断 + base (OH⁻) 接近を視認
6. `reactx run examples/sn1_dissoc.rxn -o out/sn1d/ --backend uma` を実行
   → `meta.json.trials` が 1 件 (unimolecular auto-clamp), `trajectory.xyz` で C–Br 距離が ≥4.5 Å まで伸びる
7. `pytest -m slow` で `test_re3_e2` + `test_re3_sn1_dissoc` + 既存 SN2/PT/Menshutkin が全 pass
8. `reactx run examples/sn1_recomb.rxn -o out/sn1r/ --backend uma --render` を実行
   → `meta.json.trials` が 8 件 (bimolecular)、`reached_product=True` の trial が ≥1 件、`out/sn1r/scene.blend` で Cl⁻ が tBu⁺ の平面に向かって接近 → C–Cl 結合形成を視認
9. `pytest -m slow` で `test_re4_sn1_recomb` + 既存 `test_re1_*` / `test_re3_*` が全 pass

## レンダリング: 原子球サイズと結合棒

`blender/render.py` は `atomic-blender-pdb-xyz` アドオンで XYZ を読み込んだ後、`render.py` 側で 2 つの後処理を行う:

1. **原子球サイズ**: 各元素ボールの半径を **Alvarez (2013) *Dalton Trans.* 42, 8617 の van der Waals 半径 × 0.25** で上書き (ball-and-stick 風スケール)。比率は実際の vdW 半径比と一致する。`REACTX_VDW_SCALE` で全体倍率を上書き可 (例: `REACTX_VDW_SCALE=0.4 reactx run ...` で CPK 寄り)。
2. **結合棒**: アドオンの XYZ importer は結合を描画しないため、`render.py` が独自に Cordero (2008) 共有結合半径を用いた距離判定 (閾値 = `(rcov_A + rcov_B) × BOND_TOLERANCE`、デフォルト 1.1) で結合を抽出。各候補結合 1 本につきシリンダオブジェクトを 1 個生成し、フレームごとに transform (両端原子に追従) と表示/非表示 (距離が閾値以内かを毎フレーム再判定) を keyframe する。これにより SN2 のように形成・切断する結合は途中で出現・消滅する。

対応元素は UMA `omol` タスクの訓練範囲 = OMol25 = **Z=1 (H) 〜 Z=83 (Bi)** の連続 83 元素。Po (84) 以降、Fr/Ra および全アクチノイドは UMA 訓練外 → 入力 XYZ に出現しない想定。テーブル外の元素はリスケール対象外 (アドオン既定半径のまま) かつ結合棒も生成されない。

詳細仕様: `docs/superpowers/specs/2026-04-26-vdw-radii-design.md`

## Phase Re1 + Phase 3 + Phase 4 の方針と限界

- 目的は妥当なアニメーション (TS エネルギーの正確さは目標としない)
- NEB は default off。`--neb-refine` は **1 formed + 1 broken 反応のみ対応** (E2 / SN1 dissoc / SN1 recomb では CLI が exit code 2 で reject)
- 対応反応 (Phase 4 時点): SN2 / proton transfer / Menshutkin (1 formed + 1 broken) + **E2 elimination (1 formed + 2 broken)** + **SN1 step 1 解離 (0 formed + 1 broken)** + **SN1 step 2 recombination (1 formed + 0 broken)**。中性 addition / cycloaddition / metathesis / Diels–Alder などは Phase 5+
- ラジカル / open-shell / 溶媒効果は対象外
- multi-bond NEB endpoint construction は Phase 5+
- 詳細仕様: `docs/superpowers/specs/2026-04-27-reactx-phase-Re1-design.md` (Phase Re1) / `docs/superpowers/specs/2026-05-03-phase-3-multibond-design.md` (Phase 3) / `docs/superpowers/specs/2026-05-03-phase-4-sn1-recomb-design.md` (Phase 4)

## Wall-clock (実測)

RTX 5070 Ti + UMA-m-1p1 で実測 (default `--n-angles 8 --prescreen-keep 3`):

| Reaction | wall-clock (旧, 8/8 UMA) | wall-clock (新, prescreen + 3/8 UMA) | 備考 |
|---|---|---|---|
| SN2 (`examples/sn2.rxn`) | ~67 s | **~54 s** | MMFF 動作、3/3 reached_product |
| Proton transfer (`examples/proton_transfer.rxn`) | ~116 s | **~117 s** | HCl で MMFF parameterize 失敗 → 8/8 UMA に fallback |
| Menshutkin (`examples/menshutkin.rxn`) | ~3 min | **~41 s** | MMFF 動作、3/3 reached_product |
| E2 (`examples/e2.rxn`) | — (新規) | **~54 s** | Phase 3, 1 formed + 2 broken、3/3 reached_product |
| SN1 dissoc (`examples/sn1_dissoc.rxn`) | — (新規) | **~21 s** | Phase 3, unimolecular → n_angles=1 強制、prescreen skipped |
| SN1 recomb (`examples/sn1_recomb.rxn`) | — (新規) | **~30-60 s** | Phase 4, 1 formed + 0 broken、bimolecular で 8 trials → prescreen で top-3 |

MMFF94 prescreen は実測上 **HCl のように小さく原子タイプを取りにくい fragment** を含む系 (proton transfer 等) では parameterize に失敗してフォールバック (= 旧挙動と同一の wall-clock) する。SN2 (anion 含む) と Menshutkin (中性) ではいずれも MMFF が成功する。失敗は `meta.json.prescreen.mmff_failed=true` で確認でき、`--no-mmff-prescreen` で明示的に旧挙動を再現することも可能。

UMA model load (~25-30 s) が固定コストとして wall-clock を支配するため、prescreen による短縮幅は SN2 で ~20 %、Menshutkin で ~75 % など反応や trial 当たりの relax コストに依存する。

`--neb-refine` を on にすると NEB の収束に追加で 5–10 分かかる (DoD 用テスト `test_neb_refine_sn2` で実測 ~7 分)。アニメーション目的なら off 推奨。

各実行の trial 全件スコアと prescreen 結果と wall_clock_seconds は `out/<rxn>/meta.json` に残る。

## テスト

```bash
pytest                   # 高速ユニットテストのみ (slow / blender マーカーは除外)
pytest -m slow           # UMA 依存の SN2 統合テスト
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
                                              embed3d (rotation perturb)
                                                ├ trial 1
                                                ├ trial 2  ─┐
                                                ├ ...        │ FIRE + Hookean/PullApart restraints
                                                └ trial N  ─┘
                                                       │
                                                       ▼
                                                scoring → best trial
                                                       │
                                          (optional) neb refinement
                                                       │
                                                       ▼
                                               trajectory.xyz → blender/render.py → .blend
```

詳細設計: `docs/superpowers/specs/2026-04-27-reactx-phase-Re1-design.md`
実装計画: `docs/superpowers/plans/2026-04-27-reactx-phase-Re1-implementation.md`
