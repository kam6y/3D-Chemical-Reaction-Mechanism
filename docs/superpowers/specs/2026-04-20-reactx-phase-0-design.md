# reactx — Phase 0 Spike Design

- Status: Draft (approved for spec review)
- Date: 2026-04-20
- Owner: @kam6y
- Branch: `phase-0-spike`

## 1. Purpose

ChemDraw 等で作図した 2D 反応機構（`.rxn` 形式）を入力に、UMA (Universal ML Interatomic Potential) + ASE NEB で反応経路 (MEP) を自動探索し、生成された 3D 軌跡を Blender 上で ball-and-stick アニメーションとして再生するまでを end-to-end で貫通させる Phase 0 スパイクを構築する。

Phase 0 の目的は「特定 1 反応で全パイプラインが動作する」ことの実証であり、汎用化や高度演出は意図的にスコープ外とする。

## 2. Phase 0 の固定反応

- **SN2 反応**: CH₃Cl + F⁻ → CH₃F + Cl⁻（Walden 反転）
- 選定理由: 原子数が少なく (6 原子)、TS 構造が明確、背面攻撃の視覚的インパクトが強くアニメ映えする、UMA 学習分布内で確実に通る。

## 3. 想定用途

研究探索 + 短尺動画（教材・プレゼン用途）。論文 Figure 級の最終品質は Phase 1 以降で拡張する。

## 4. Pipeline 全体像

```
.rxn (ChemDraw export)
  └─▶ [rxn_parser]   RDKit で reactant/product/atom-mapping 抽出
       └─▶ [embed3d]  RDKit EmbedMolecule → MMFF94 → UMA 単点再最適化
            └─▶ [align]  atom mapping に基づく原子順の一致化
                 └─▶ [neb]   ASE NEB (IDPP 初期路 → CI-NEB) × UMA
                      └─▶ trajectory.xyz (multi-frame, N≈11)
                           └─▶ [blender] bpy script + atomic-blender-pdb-xyz
                                └─▶ .blend シーン (keyframe 化された ball-and-stick + カメラ/照明)
```

## 5. リポジトリ構成

```
reactx/
  __init__.py
  cli.py              # エントリポイント (`reactx run examples/sn2.rxn -o out/`)
  rxn_parser.py       # .rxn → (reactant Mol, product Mol, atom-mapping dict)
  embed3d.py          # 2D Mol → 3D 座標 (RDKit + UMA 微調整)
  align.py            # atom-mapping に基づき原子順を揃える
  neb.py              # ASE NEB (IDPP → CI-NEB) driver、trajectory.xyz を出力
  calculators.py      # ASE Calculator ファクトリ (Phase 0: UMA)
blender/
  render.py           # `blender --background --python render.py -- trajectory.xyz out.blend`
examples/
  sn2.rxn             # CH3Cl + F⁻ → CH3F + Cl⁻
tests/
  test_rxn_parser.py
  test_embed3d.py
  test_neb_sn2.py
  test_blender_smoke.py
pyproject.toml
README.md
docs/
  superpowers/
    specs/
      2026-04-20-reactx-phase-0-design.md   # 本ファイル
```

## 6. コンポーネント仕様

### 6.1 `rxn_parser`

- 入力: `.rxn` ファイルパス
- 出力: `(reactant_mol: rdkit.Chem.Mol, product_mol: rdkit.Chem.Mol, mapping: dict[int, int])`
- 実装: `rdkit.Chem.rdChemReactions.ReactionFromRxnFile` で読込、`GetAtomMapNum` から mapping を抽出。
- Phase 0 前提: `.rxn` に atom map 番号が付与されている前提。SN2 サンプル `.rxn` はリポジトリ同梱で手動付与。
- Phase 1 候補: `rxnmapper` 統合で map 番号欠損時も自動付与。

### 6.2 `embed3d`

- 入力: `Mol`（2D）
- 出力: `ase.Atoms`（3D 座標付き）
- 実装手順:
  1. `AllChem.EmbedMolecule(mol, ETKDGv3)` で初期 3D 埋め込み
  2. `AllChem.MMFFOptimizeMolecule(mol)` で粗適化
  3. `ase.Atoms` に変換し UMA Calculator で `BFGS` 単点最適化 (Fmax < 0.05)
- 失敗時: seed を変えて最大 5 回リトライ、全滅なら `RuntimeError` と手順ヒント。

### 6.3 `align`

- 入力: reactant `Atoms`, product `Atoms`, mapping
- 出力: atom mapping に基づき product 側の原子順を reactant に合わせた `Atoms`
- 重要: NEB は images 間で原子 index が対応している必要がある。mapping を使って両端の座標配列を同じ index 順に並べ替える。

### 6.4 `neb`

- 入力: reactant `Atoms`, product `Atoms` (aligned)
- 出力: `trajectory.xyz`（multi-frame ASE XYZ）
- 実装:
  - `ase.neb.NEB` で 11 images (両端 + 9 中間) 生成
  - `IDPP` (`neb.idpp_interpolate()`) で初期パスを生成
  - 全 images に UMA Calculator を装着
  - `NEB.climb = True` (CI-NEB)
  - `BFGS` で Fmax < 0.05 eV/Å, maxiter 200
  - 収束後、各 image の `ase.Atoms` を `ase.io.write` で連結 XYZ に出力
  - 前後に reactant/product の静止フレームを数枚パディング（動画の「間」を作る）
- 非収束時: ベストエフォートで保存、`converged=False` をログと JSON メタデータに記録。

### 6.5 `calculators`

- `make_calculator(name: str = "uma") -> ase.Calculator`
- Phase 0 は `name="uma"` のみサポート。内部で `fairchem.core.FAIRChemCalculator` をモデル名指定で返す。
- デバイス選択: `torch.backends.mps.is_available()` → `mps`、無ければ `cpu`。
- Phase 1 の拡張穴: `name="xtb"` で `tblite` 経由、`name="grrm23"` で外部プロセス呼出しアダプタ。

### 6.6 `cli`

- コマンド例: `reactx run examples/sn2.rxn -o out/ --images 11 --fmax 0.05`
- 実行ステップ: parse → embed3d (reactant, product) → align → neb → Blender 呼出し（`--render` フラグ時）
- 出力: `out/trajectory.xyz`, `out/energies.json`, `out/meta.json`, (`out/scene.blend`)

### 6.7 `blender/render.py`

- `blender --background --python blender/render.py -- <trajectory.xyz> <output.blend>`
- 実装:
  1. `bpy` で新規シーン
  2. `atomic-blender-pdb-xyz` アドオン経由で XYZ 軌跡をインポート（keyframe 化される）
  3. 3 点照明 (key + fill + rim) を配置
  4. カメラを反応中心に向けて軽く orbit（24 fps × 10 秒 = 240 frame）
  5. TS 付近 (軌跡中央 image ± 1) の再生速度をスローダウン（frame offset でフレーム長を伸ばす簡易実装）
  6. Bond 表現: Phase 0 は atomic-blender-pdb-xyz のデフォルト挙動（距離ベースの bond 自動生成）に任せる。結合形成・切断の動的 fade は add-on の bond オブジェクト API を実装時に調査し、**困難なら Phase 1 に繰り延べ**（Phase 0 DoD には含めない）。
  7. `bpy.ops.wm.save_as_mainfile(filepath=<output.blend>)`
- レンダリング自体（`.mp4` 出力）は Phase 0 スコープ外。ユーザが Blender GUI で開いて確認する。

## 7. データフローと中間成果物

| ステージ | 入力 | 出力 |
|---|---|---|
| rxn_parser | `examples/sn2.rxn` | reactant Mol, product Mol, mapping |
| embed3d | Mol × 2 | reactant.xyz, product.xyz |
| align | 2 Atoms + mapping | aligned product.xyz |
| neb | reactant + aligned product | trajectory.xyz (11+ frames), energies.json |
| blender | trajectory.xyz | scene.blend |

## 8. エラーハンドリング方針

- **RDKit embed 失敗**: seed を変えて最大 5 回リトライ → `RuntimeError` + 「RDKit version / 入力構造確認」ヒント。
- **UMA HF 認証失敗**: `cli` 起動時に `ReadyCheck` で事前検出し、`huggingface-cli login` 手順を案内。
- **NEB 非収束**: warning + ベストエフォート軌跡保存。`meta.json` に `converged: false` と `final_fmax` を記録。
- **Blender 起動失敗**: `atomic-blender-pdb-xyz` 未導入を検知、インストール手順 URL を提示。
- **`.rxn` に atom mapping 欠損**: 明確なエラーで停止（Phase 0 では自動補完しない）。

## 9. テスト戦略

- `test_rxn_parser`: SN2 `.rxn` → 反応物 2 分子 + 生成物 2 分子 + mapping dict が期待値通り。
- `test_embed3d`: CH₃Cl の C–Cl 結合長が 1.7–1.9 Å、H–C–H 角が 105–115°。
- `test_neb_sn2`: 11 frame XYZ、エネルギーが単峰、TS エネルギーが両端より高い、TS で C–F–Cl が 170° 以上（ほぼ線形）。
- `test_blender_smoke`: `blender --background --python blender/render.py -- out/trajectory.xyz /tmp/scene.blend` が exit code 0、`/tmp/scene.blend` が生成される（Blender と add-on は CI では skip、ローカルのみ）。

## 10. 環境前提

- OS: macOS (Apple Silicon 推奨、MPS 経由で UMA 加速) / Linux
- Python 3.10+
- Blender 4.x
- `atomic-blender-pdb-xyz` アドオンがユーザ環境にインストール済み
- Hugging Face アカウントで UMA 利用承認済み & `huggingface-cli login` 完了
- 主要依存: `rdkit`, `ase`, `fairchem-core`, `torch`, `numpy`

## 11. Phase 0 で意図的にやらないこと（非スコープ）

- 複数反応への汎用化（SN2 ハードコード）
- Curly arrow / 電子流の 3D 可視化
- 自作 Blender add-on 開発（既存 `atomic-blender-pdb-xyz` を利用）
- 高度な映像演出（DoF、motion blur、GPU 最終レンダー、`.mp4` 出力）
- Web UI / GUI ラッパ
- xTB / GRRM23 アダプタ（`calculators.py` に抽象化フックのみ用意、実装は Phase 1+）
- ベンチマーク、並列化、バッチ処理
- atom mapping 自動補完（`rxnmapper` 連携は Phase 1）
- 周期境界条件、溶媒効果、反応物の複数コンフォマー探索

## 12. Phase 0 完了条件 (Definition of Done)

1. `reactx run examples/sn2.rxn -o out/` で `trajectory.xyz` が生成される。
2. `test_neb_sn2` が passing（TS 構造が Walden 反転の特徴を満たす）。
3. `blender --background --python blender/render.py -- out/trajectory.xyz out/scene.blend` が成功する。
4. `out/scene.blend` を Blender GUI で開き、原子が滑らかに動きながら SN2 反転が視認できる（bond 自動生成は atomic-blender-pdb-xyz のデフォルト挙動で可）。
5. README に再現手順が記載されている。

## 13. 次段階 (Phase 1 以降) のロードマップメモ

- **Phase 1**: 反応の一般化（任意の小分子 SN/ED/Claisen に対応）、`rxnmapper` 統合、xTB バックエンド追加、簡易 Blender add-on（UI パネル化）。
- **Phase 2**: Curly arrow 3D 表示、GRRM23 アダプタ、論文 Figure 品質のレンダリング (EEVEE/Cycles 切替)。
- **Phase 3**: Web UI、ChemDraw `.cdxml` 直接パース、画像 (PNG) からの構造認識 (DECIMER 連携)。
