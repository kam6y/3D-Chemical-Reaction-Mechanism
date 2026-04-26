# reactx — Phase 0 Spike

2D 反応機構 (`.rxn`) から UMA + ASE NEB で 3D MEP を探索し、Blender で ball-and-stick アニメーションを再生するパイプラインのフェーズ 0 実装。

Phase 0 の範囲: SN2 反応 (CH₃Cl + F⁻ → CH₃F + Cl⁻) 1 件を end-to-end で貫通することに限定。

## セットアップ

```bash
python -m pip install -e .[dev]
hf auth login    # UMA モデル取得のため
```

Blender 4.x と `atomic-blender-pdb-xyz` アドオンを別途インストールしておく。

## 使い方

```bash
reactx run examples/sn2.rxn -o out/ --images 15 --fmax 0.05 --backend uma
blender --background --python blender/render.py -- out/trajectory.xyz out/scene.blend
```

生成物:

- `out/trajectory.xyz` — 15+ フレームの NEB 軌跡 (default `--images 15`、padding 含む)
- `out/energies.json` — 各 image のエネルギー
- `out/meta.json` — 収束情報など
- `out/scene.blend` — Blender シーン（GUI で開いて再生）

## Phase 0 動作確認

DoD は以下の手順で確認する:

1. `reactx run examples/sn2.rxn -o out/ --backend uma --render` を実行 → `out/trajectory.xyz`, `out/scene.blend` が生成される。
2. `out/scene.blend` を Blender GUI で開き、再生して **F⁻ が CH₃Cl の背面から接近 → C 中心の sp³ 反転 → Cl⁻ が脱離** する Walden 反転シーケンスが視認できることを確認する。
3. `pytest -m slow` で `test_neb_sn2` を実行し、TS の C–F–Cl 角度が一定値以上であることを確認する。

## レンダリング: 原子球サイズ

`blender/render.py` は `atomic-blender-pdb-xyz` アドオンで XYZ を読み込んだ後、各元素ボールの半径を **Alvarez (2013) *Dalton Trans.* 42, 8617 の van der Waals 半径 × 0.25** で上書きする (ball-and-stick 風スケール)。比率は実際の vdW 半径比と一致する。

- 環境変数 `REACTX_VDW_SCALE` で全体倍率を上書き可 (例: `REACTX_VDW_SCALE=0.4 reactx run ...` で CPK 寄り)
- 対応元素は UMA `omol` タスクの訓練範囲 = OMol25 = **Z=1 (H) 〜 Z=83 (Bi)** の連続 83 元素。Po (84) 以降、Fr/Ra および全アクチノイドは UMA 訓練外 → 入力 XYZ に出現しない想定。万一現れた場合はリスケール対象外 (アドオン既定半径のまま) で警告ログを出す。
- 詳細仕様: `docs/superpowers/specs/2026-04-26-vdw-radii-design.md`

## Phase 0 の既知の制約

- **TS の線形性閾値が 170° → 120° に緩和**。`uma-m-1p1` (omol task) では SN2 の TS が安定して 170° (ほぼ線形) に到達しないため、Phase 0 では背面攻撃の方向性確認に留める。詳細は `docs/superpowers/specs/2026-04-20-reactx-phase-0-design.md` §9。Phase 1 でモデル/最適化チューニング後に再引き締め予定。
- **TS 付近のスローダウン再生** は Phase 1 で実装予定。Phase 0 の `blender/render.py` は軌跡を均一速度で再生する。
- **Bond 形成・切断の動的 fade** は `atomic-blender-pdb-xyz` のデフォルト挙動（距離ベースの自動生成）に委ねる。
- **`.mp4` 最終レンダリング** は Phase 0 スコープ外。Blender GUI で `.blend` を開いて再生する。

`--render` フラグを付けると CLI から Blender を直接呼び出す:

```bash
reactx run examples/sn2.rxn -o out/ --render --blender-exe /path/to/blender
```

## テスト

```bash
pytest                   # 高速ユニットテストのみ (slow / blender マーカーは除外)
pytest -m slow           # UMA 依存の SN2 統合テスト
pytest -m blender        # Blender smoke test (ローカル環境のみ)
```

## アーキテクチャ

```
.rxn → rxn_parser → embed3d → align → neb → trajectory.xyz → blender/render.py → .blend
```

詳細設計: `docs/superpowers/specs/2026-04-20-reactx-phase-0-design.md`
実装計画: `docs/superpowers/plans/2026-04-20-reactx-phase-0-implementation.md`
