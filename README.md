# reactx — Phase 0 Spike

2D 反応機構 (`.rxn`) から UMA + ASE NEB で 3D MEP を探索し、Blender で ball-and-stick アニメーションを再生するパイプラインのフェーズ 0 実装。

Phase 0 の範囲: SN2 反応 (CH₃Cl + F⁻ → CH₃F + Cl⁻) 1 件を end-to-end で貫通することに限定。

## セットアップ

```bash
python -m pip install -e .[dev]
huggingface-cli login    # UMA モデル取得のため
```

Blender 4.x と `atomic-blender-pdb-xyz` アドオンを別途インストールしておく。

## 使い方

```bash
reactx run examples/sn2.rxn -o out/ --images 11 --fmax 0.05 --backend uma
blender --background --python blender/render.py -- out/trajectory.xyz out/scene.blend
```

生成物:

- `out/trajectory.xyz` — 11+ フレームの NEB 軌跡
- `out/energies.json` — 各 image のエネルギー
- `out/meta.json` — 収束情報など
- `out/scene.blend` — Blender シーン（GUI で開いて再生）

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
