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

## Phase 1: 多反応対応

Phase 1 では `.rxn` の atom mapping から **bond change (broken / formed)** を抽出し、それに応じた幾何配置で fragment を初期化する generic engine を導入した。これにより SN2 以外の極性二分子反応 (および解離型 elementary step) を反応分類なしで処理できる。

### 対応反応

| 反応 | 例 | example |
|---|---|---|
| SN2 | CH₃Cl + F⁻ → CH₃F + Cl⁻ | `examples/sn2.rxn` |
| Heterolytic dissociation (SN1/E1 step 1) | (CH₃)₃C–Br → (CH₃)₃C⁺ + Br⁻ | `examples/sn1_step1.rxn` |
| E2 elimination | CH₃CH₂Br + OH⁻ → CH₂=CH₂ + H₂O + Br⁻ | `examples/e2.rxn` |
| β-H elimination from cation (E1 step 2) | (CH₃)₃C⁺ → (CH₃)₂C=CH₂ + H⁺ | `examples/e1_step2.rxn` |
| Proton transfer | HCl + NH₃ → Cl⁻ + NH₄⁺ | `examples/proton_transfer.rxn` |

### 使用方法

```bash
reactx run examples/proton_transfer.rxn -o out/proton/ --backend uma --render
reactx run examples/e2.rxn               -o out/e2/     --backend uma --render
reactx run examples/sn1_step1.rxn        -o out/sn1/    --backend uma --render
reactx run examples/e1_step2.rxn         -o out/e1/     --backend uma --render
```

`--images` を省略すると bond change の数から自動計算される: `max(11, 9 + 2 * n_broken + 2 * n_formed)`。SN2 → 13、解離 → 11、E2 → 17。

### 慣習: multi-step 反応の表現

NEB は 1 つの elementary step (= 1 saddle point) を扱う前提のため、SN1 や E1 のような multi-step 反応はステップごとに別 `.rxn` ファイルとして表現する。例えば SN1 全体は `sn1_step1.rxn` (heterolytic dissociation) と既存 `sn2.rxn` 相当の置換ステップを別個に走らせる。

migrating H (proton transfer の H、E2 の β-H、E1 step 2 の β-H 等) は両側で **explicit な map number 付き H** として `.rxn` に書く必要がある。Implicit Hs は AddHs で位置的に対応付けされる。

### Phase 1 の制約

- **反応タイプの自動分類は行わない**: `.rxn` の bond change のみで配置を決める (Approach 2)。
- ラジカル / open-shell 反応は対象外 (UMA omol task は closed-shell 前提)。
- aromatic 結合の bond change は `NotImplementedError` で停止する。

詳細仕様: `docs/superpowers/specs/2026-04-26-reactx-phase-1-design.md`
実装計画: `docs/superpowers/plans/2026-04-26-reactx-phase-1-implementation.md`

## レンダリング: 原子球サイズと結合棒

`blender/render.py` は `atomic-blender-pdb-xyz` アドオンで XYZ を読み込んだ後、`render.py` 側で 2 つの後処理を行う:

1. **原子球サイズ**: 各元素ボールの半径を **Alvarez (2013) *Dalton Trans.* 42, 8617 の van der Waals 半径 × 0.25** で上書き (ball-and-stick 風スケール)。比率は実際の vdW 半径比と一致する。`REACTX_VDW_SCALE` で全体倍率を上書き可 (例: `REACTX_VDW_SCALE=0.4 reactx run ...` で CPK 寄り)。
2. **結合棒**: アドオンの XYZ importer は結合を描画しないため、`render.py` が独自に Cordero (2008) 共有結合半径を用いた距離判定 (閾値 = `(rcov_A + rcov_B) × BOND_TOLERANCE`、デフォルト 1.1) で結合を抽出。各候補結合 1 本につきシリンダオブジェクトを 1 個生成し、フレームごとに transform (両端原子に追従) と表示/非表示 (距離が閾値以内かを毎フレーム再判定) を keyframe する。これにより SN2 のように形成・切断する結合は途中で出現・消滅する。

対応元素は UMA `omol` タスクの訓練範囲 = OMol25 = **Z=1 (H) 〜 Z=83 (Bi)** の連続 83 元素。Po (84) 以降、Fr/Ra および全アクチノイドは UMA 訓練外 → 入力 XYZ に出現しない想定。テーブル外の元素はリスケール対象外 (アドオン既定半径のまま) かつ結合棒も生成されない。

詳細仕様: `docs/superpowers/specs/2026-04-26-vdw-radii-design.md`

## Phase 0 の既知の制約

- **TS の線形性閾値が 170° → 120° に緩和**。`uma-m-1p1` (omol task) では SN2 の TS が安定して 170° (ほぼ線形) に到達しないため、Phase 0 では背面攻撃の方向性確認に留める。詳細は `docs/superpowers/specs/2026-04-20-reactx-phase-0-design.md` §9。Phase 1 でモデル/最適化チューニング後に再引き締め予定。
- **TS 付近のスローダウン再生** は Phase 1 で実装予定。Phase 0 の `blender/render.py` は軌跡を均一速度で再生する。
- **Bond 形成・切断の dynamic fade (滑らかなフェード)** は Phase 1 で実装予定。Phase 0 の `render.py` はフレームごとに表示/非表示を切り替える (constant interpolation、ハードカット)。
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
