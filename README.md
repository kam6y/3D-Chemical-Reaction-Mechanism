# reactx — Phase Re1 Multi-Angle Path Engine

2D 反応機構 (`.rxn`) から多角度サンプリング + Hookean/PullApart 拘束 + FIRE 緩和で 3D 反応経路を探索し、Blender で ball-and-stick アニメーションを生成するパイプライン。

**Phase Re1 の目標**: 正確な TS エネルギーではなく、妥当なアニメーション。NEB はオプション。対応反応: SN2 / proton transfer。

## セットアップ

```bash
python -m pip install -e .[dev]
hf auth login    # UMA モデル取得のため
```

Blender 4.x と `atomic-blender-pdb-xyz` アドオンを別途インストールしておく。

## 使い方

```bash
# SN2 (default settings)
reactx run examples/sn2.rxn -o out/sn2/ --backend uma --render
# Proton transfer (HCl + NH3 -> Cl- + NH4+)
reactx run examples/proton_transfer.rxn -o out/pt/ \
  --backend uma --r-form 1.05 --render
```

主要フラグ:

- `--n-angles 8` (default): 多角度試行数
- `--cone-half-deg 30.0`: 多角度試行の cone 半角
- `--r-form` (default: 元素ペアから自動): 形成結合の目標距離 (Å)
- `--neb-refine` (default off): best trajectory を NEB で refinement (実行時間延長)

生成物:

- `out/<rxn>/trajectory.xyz` — best trial trajectory
- `out/<rxn>/energies.json` — best trial エネルギー列
- `out/<rxn>/meta.json` — trial 全件の score, wall_clock_seconds, neb_refined フラグ
- `out/<rxn>/scene.blend` — Blender シーン

## Phase Re1 動作確認

DoD は以下の手順で確認する:

1. `reactx run examples/sn2.rxn -o out/sn2/ --backend uma --render` を実行 → `meta.json` の `selected_trial >= 0`, `trials[].reached_product` で少なくとも 1 件 True を確認
2. `out/sn2/scene.blend` を Blender GUI で開いて Walden 反転を視認
3. `reactx run examples/proton_transfer.rxn -o out/pt/ --backend uma --r-form 1.05 --render` を実行 → 同様に視認
4. `pytest -m slow` で SN2 + proton_transfer 統合テストが pass

## レンダリング: 原子球サイズと結合棒

`blender/render.py` は `atomic-blender-pdb-xyz` アドオンで XYZ を読み込んだ後、`render.py` 側で 2 つの後処理を行う:

1. **原子球サイズ**: 各元素ボールの半径を **Alvarez (2013) *Dalton Trans.* 42, 8617 の van der Waals 半径 × 0.25** で上書き (ball-and-stick 風スケール)。比率は実際の vdW 半径比と一致する。`REACTX_VDW_SCALE` で全体倍率を上書き可 (例: `REACTX_VDW_SCALE=0.4 reactx run ...` で CPK 寄り)。
2. **結合棒**: アドオンの XYZ importer は結合を描画しないため、`render.py` が独自に Cordero (2008) 共有結合半径を用いた距離判定 (閾値 = `(rcov_A + rcov_B) × BOND_TOLERANCE`、デフォルト 1.1) で結合を抽出。各候補結合 1 本につきシリンダオブジェクトを 1 個生成し、フレームごとに transform (両端原子に追従) と表示/非表示 (距離が閾値以内かを毎フレーム再判定) を keyframe する。これにより SN2 のように形成・切断する結合は途中で出現・消滅する。

対応元素は UMA `omol` タスクの訓練範囲 = OMol25 = **Z=1 (H) 〜 Z=83 (Bi)** の連続 83 元素。Po (84) 以降、Fr/Ra および全アクチノイドは UMA 訓練外 → 入力 XYZ に出現しない想定。テーブル外の元素はリスケール対象外 (アドオン既定半径のまま) かつ結合棒も生成されない。

詳細仕様: `docs/superpowers/specs/2026-04-26-vdw-radii-design.md`

## Phase Re1 の方針と限界

- 目的は妥当なアニメーション (TS エネルギーの正確さは目標としない)
- NEB は default off。`--neb-refine` で smoothing 可能だが wall-clock が大幅に伸びる
- 対応反応は形成 1 + 切断 1 の elementary step に限定 (SN2 / proton transfer 等)。E2 / SN1 step 1 / β-H elimination などは Phase 2 へ持ち越し
- ラジカル / open-shell / 溶媒効果は対象外
- 詳細仕様: `docs/superpowers/specs/2026-04-27-reactx-phase-Re1-design.md`

## Wall-clock (SN2 default)

SN2 reaction with default settings (--n-angles 8) on a typical dev GPU:
- Phase Re1 multi-angle pipeline: TBD (run `pytest -m slow tests/test_wallclock_sn2.py -s` to measure)
- Phase 0 NEB baseline (legacy): N/A on this branch (use `git checkout phase-0-spike` to compare)

## テスト

```bash
pytest                   # 高速ユニットテストのみ (slow / blender マーカーは除外)
pytest -m slow           # UMA 依存の SN2 統合テスト
pytest -m blender        # Blender smoke test (ローカル環境のみ)
```

## アーキテクチャ

```
.rxn → rxn_parser → bond_changes (formed/broken) → embed3d (rotation perturb)
                                                   ├ trial 1
                                                   ├ trial 2  ─┐
                                                   ├ ...        │ FIRE + Hookean/PullApart restraints
                                                   └ trial N  ─┘
                                                          ↓
                                                   scoring → best trial
                                                          ↓
                                          (optional) neb refinement
                                                          ↓
                                                  trajectory.xyz → blender/render.py → .blend
```

詳細設計: `docs/superpowers/specs/2026-04-27-reactx-phase-Re1-design.md`
実装計画: `docs/superpowers/plans/2026-04-27-reactx-phase-Re1-implementation.md`
