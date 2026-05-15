# reactx Phase 11 — Pure CI-NEB pipeline (AFIR 全廃)

- Status: Draft
- Date: 2026-05-15
- Owner: @kam6y
- Branch: `phase-11` (from `phase-10` HEAD, target merge to `develop`)
- 前提仕様: なし (Phase 9/10 AFIR インフラを **全廃** する破壊的変更)
- 後方互換性: develop ブランチ運用方針 (`[[feedback_no_backwards_compat]]`) により API breaking 許容。`.rxn.toml` schema を完全リプレースし、Phase 9/10 の `[afir]` / `[scoring]` を含む TOML は `ConfigError` で reject する。

## 1. 目的

`.rxn` ファイルには reactant / product の両構造が atom-mapped で完全に記述されている。Phase 9/10 の AFIR + sticky latch は本質的に「product 構造を *知らない* ものとして探索する」道具であり、与えられた product 構造を使えば不要。本フェーズでは:

1. AFIR インフラ (artificial_force / path_relax / scoring / sticky latch / 4π Fibonacci sampling / vdW shadow blocking) を **完全削除**
2. R / P 両 endpoint を直接 embed → UMA で厳密に minimum まで緩和 → CI-NEB で経路生成、という素直なパイプラインに置換
3. AFIR-biased な可視 trajectory を物理的に意味のある MEP 近似 (CI-NEB 結果) で置き換える

ユーザー指摘の核心: **「endpoint は .rxn ファイルで与えられている。それより NEB 両端の構造を事前に十分緩和することの方が大切」**。Phase 11 の精度は endpoint relax の収束品質で決まる。

### 1.1 `.rxn` の 2D 座標は使わない

本パイプラインは `.rxn` ファイルから取り出すのは **connectivity (bond list, bond order)・atom map number・元素** の 3 点のみで、`.rxn` 内の 2D 座標は無視する。3D 座標は `embed_fragments_to_positions` が **ETKDG + MMFF94** で fragment ごとに新規生成し、fragment 間相対配置は `simple_placement`、最終的な精密 minimum は `endpoint_relax` (UMA) が担う。したがって `examples/*.rxn` の座標が雑でも結果に影響しない (Phase 9/10 と同様)。

## 2. Non-goals

- **複数 trial / 多候補 sampling**: 1 反応につき R / P endpoint は 1 通り。Fibonacci sphere / blocking / endo-exo 2 trial 等は全廃。endo/exo を比較したい場合は別 `.rxn` ファイルを用意する (`diels_alder_endo.rxn` / `diels_alder_exo.rxn` のように)。
- **AFIR の opt-in 残置**: 完全削除。「AFIR でも path 描きたい」要件は出てきたら Phase 12 で別途検討。
- **NEB hyperparameter の反応別 tune**: default 値 1 セットで 8 反応すべてを通す方針。個別 `.rxn.toml` で override 可能だが、example TOML には書かない。
- **TS エネルギーの定量精度**: 引き続き「妥当なアニメーション」を目的とし、TS の絶対エネルギーは保証しない。CI-NEB により Phase 9/10 より MEP 近似は良くなるが、UMA 自体の精度が支配的。
- **multi-step elementary reaction の連結**: 1 `.rxn` = 1 elementary step を維持。

## 3. アーキテクチャ

### 3.1 パイプライン

```
.rxn (R/P 両構造 + atom map)
        │
        ▼
parse_rxn  →  r_mol, p_mol, heavy_mapping
        │
        ▼
BondChanges.from_reaction_diff(r_mol, p_mol)   ★新規 (atom-map 差分で自動推論)
        │   formed / broken は internal 用途のみ (placement 方向決め)
        ▼
embed_fragments_to_positions(r_mol)    embed_fragments_to_positions(p_mol)
        │                                       │
        ▼                                       ▼
simple_placement (R)                   simple_placement (P)
  - unimolecular: passthrough            - 同左
  - bimolecular:  反応点 anchor を         - 反応点 anchor は P 側の formed 末端
                  initial_separation Å 離す
        │                                       │
        ▼                                       ▼
build_atoms_from_positions             build_atoms_from_positions
        │                                       │
        ▼                                       ▼
relax_endpoint(atoms_r, calc=UMA,       relax_endpoint(atoms_p, calc=UMA, ...)
              fmax=0.01, max_steps=500)
        │                                       │
        └────────────────┬──────────────────────┘
                         ▼
            align_product_to_reactant
              (heavy_mapping ベース、
               1+1 制限なし)
                         │
                         ▼
            run_neb (IDPP + 2-phase NEB)
                         │
                         ▼
            trajectory.xyz (NEB image 列)
            energies.json  (image_energies)
            meta.json
```

### 3.2 削除するモジュール

| File | 理由 |
|---|---|
| `reactx/artificial_force.py` | AFIRConstraint 完全削除 |
| `reactx/path_relax.py` | constraint relax 不要 |
| `reactx/scoring.py` | trial sweep が無いので score_trials / TrialResult / reached_product 不要 |
| 対応 test (`test_afir_constraint.py`, `test_artificial_force.py`, `test_path_relax.py`, `test_scoring.py`, `test_neb_refine_sn2.py`, `test_cli_neb_refine_guard.py`, `test_config_afir.py`, `test_re*.py` の AFIR assertion 部分) | 機能消滅 |

### 3.3 縮小・改修するモジュール

| File | 改修内容 |
|---|---|
| `reactx/placement.py` | Fibonacci / blocking / multi-anchor / endo-exo 展開を全廃し、`simple_placement(mol_h, base_positions, bond_changes, *, initial_separation, side, orientation) -> np.ndarray` 1 関数のみに縮小。bridges>=2 (DA) は固定 orientation 1 通り (configurable) |
| `reactx/bond_changes.py` | `from_atom_map_pairs` 削除、`from_reaction_diff(r_mol, p_mol) -> BondChanges` 追加 (R/P の bond set を atom-map で正規化して差分)。R-side 0-based atom indices で保持。P 側 placement では `heavy_mapping` で R→P index 変換する |
| `reactx/align.py` | `bond_changes_product` 引数を取らず heavy_mapping のみで一般化、H permutation 機構はそのまま流用 |
| `reactx/neb.py` | API 変更: 戻り値に `image_atoms: list[Atoms]` を追加 (file 読み戻し不要に)、内部 `padded` 構成は維持 |
| `reactx/config.py` | schema 完全リプレース ([afir] / [scoring] / [sampling] → [placement] / [endpoint_relax] / [neb])、Phase 9/10 の旧 schema 検出時 `ConfigError` |
| `reactx/cli.py` | trial loop 全廃、AFIR import 削除、NEB を常時実行に、`--neb-refine` / `--neb-images` / `--seed` / `--traj-stride` / `--relax-fmax` を撤廃、`meta.json` 構造をフラットに刷新 |
| `examples/*.rxn.toml` | 全 8 ファイルを新 schema で書き直し |
| 統合 test (`test_re1_sn2.py`, `test_re1_proton_transfer.py`, `test_re1_menshutkin.py`, `test_re3_e2.py`, `test_re3_sn1_dissoc.py`, `test_re4_sn1_recomb.py`, `test_diels_alder_simple.py`, `test_diels_alder_endo.py`, `test_examples_menshutkin.py`) | NEB trajectory 前提に書き直し: `len(frames) == n_images + 2*pad_frames`、SN2 Walden 角度 ≥ 150° は endpoint R 上で評価 |

### 3.4 新規モジュール

| File | 内容 |
|---|---|
| `reactx/endpoint_relax.py` | `relax_endpoint(atoms, calc, *, fmax, max_steps, optimizer) -> tuple[Atoms, dict]` (FIRE/BFGS 切替、収束フラグと最終 fmax を返す) |

### 3.5 維持する (変更なし) モジュール

`reactx/calculators.py`, `reactx/embed3d.py`, `reactx/vdw_radii.py`, `reactx/covalent_radii.py`, `reactx/rxn_parser.py`, `blender/render.py`。

## 4. TOML schema (新)

```toml
description = "SN2 anion: CH3Cl + OH- -> CH3OH + Cl-"

[placement]
initial_separation = 4.0      # bimolecular の R/P 反応点間距離 [Å]、default 4.0
orientation        = "default" # bridges>=2 (DA) のみ参照、"default" | "endo" | "exo"、default "default"

[endpoint_relax]
fmax       = 0.01             # default 0.01 eV/Å、十分な minimum 収束
max_steps  = 500              # default 500
optimizer  = "FIRE"           # "FIRE" | "BFGS"、default "FIRE"

[neb]
n_images   = 11               # default 11
fmax       = 0.05             # default 0.05
max_steps  = 200              # default 200
k          = 1.0              # default 1.0
climb      = true             # default true
pad_frames = 0                # default 0
```

- **全 section / 全 key は省略可**。`description` のみ必須。`endpoint_relax` を完全省略すれば全 default で動く。
- **`formed` / `broken` は schema から削除**。`.rxn` の R/P bond set 差分から自動推論。
- **旧 schema (`[afir]`, `[scoring]`, `[sampling]`, top-level `formed`/`broken`) は `ConfigError`**。例外メッセージで Phase 11 schema への移行を案内。
- **`description` も任意化を検討したが、`.rxn` には description フィールドが無いため必須継続** (`meta.json` / log に出るため)。

### 4.1 8 反応の新 `.rxn.toml` (内容)

すべて `description` のみ必須、その他は default。**全 8 ファイルが 1 行 (description) または +`orientation` の 2 行**:

```toml
# examples/sn2.rxn.toml
description = "SN2 anion: CH3Cl + OH- -> CH3OH + Cl-"
```

```toml
# examples/diels_alder_endo.rxn.toml
description = "Diels-Alder endo: cyclopentadiene + maleic anhydride -> norbornene-2,3-dicarboxylic anhydride"
[placement]
orientation = "endo"
```

(他 6 反応も同様の最小形)

## 5. R / P placement 戦略 (`simple_placement`)

### 5.1 unimolecular (n_frags == 1)

- `embed_fragments_to_positions` の出力を passthrough。何もしない。
- SN1 dissoc (R 側 1 frag), SN1 recomb の P 側 (1 frag) などが該当。

### 5.2 bimolecular bridges=1 (formed=1 or broken=1)

- BondChanges から R 側 anchor atom を特定 (formed pair の atom-map → R idx)
- substrate fragment (anchor を含む方) の重心を原点に
- incoming fragment (もう一方) を `+z * initial_separation` の位置に配置
- 向き: incoming fragment の anchor atom が substrate の anchor に対して `-z` を向くよう rigid-body 回転
- Phase 9/10 のような 4π Fibonacci 探索や angular shadow blocking は **行わない** — 反応点同士を最短距離で向かい合わせる 1 通りのみ

### 5.3 bridges=2 (Diels-Alder)

- formed pair が 2 つあるので diene / dienophile の face 配置が必要
- `orientation="default"`: 既存 multi_anchor placement の "achiral 縮約形" 1 通り (DA simple 用)
- `orientation="endo"`: endo (cyclopentadiene + maleic anhydride 系)
- `orientation="exo"`: exo
- multi_anchor placement の現行コードから 1 trial 分の geometry 計算ロジックだけ流用、blocking / 候補列挙は廃止

### 5.4 P 側 placement

R 側と同じアルゴリズム (`simple_placement(side="product", heavy_mapping=...)`)。anchor 選択ルールが side で変わる:

- **`side="reactant"`**: `bond_changes.formed` の atom pairs を anchor (反応で *これから* 結合する 2 原子を向き合わせる)
- **`side="product"`**: `bond_changes.broken` の atom pairs を anchor。**ただし `heavy_mapping` で R→P atom indices に変換してから使う** (`.broken` が R-side indices で保持されているため)

意味: P 側では「反応で *切れた* bond の両端」が今は離れている 2 fragment の anchor になる (例: SN2 の P 側で C-Cl⁻ が離れる anchor)。これにより R と P で同じ atom-map ペアが anchor として機能し、CI-NEB の interpolation で原子が正しい方向に動く。

## 6. endpoint relax

```python
def relax_endpoint(
    atoms: Atoms,
    calc: Calculator,
    *,
    fmax: float = 0.01,
    max_steps: int = 500,
    optimizer: str = "FIRE",
) -> tuple[Atoms, dict]:
    """Return (relaxed_atoms, info_dict).
    info_dict: {'converged': bool, 'final_fmax': float, 'n_steps': int}
    """
```

- 内部で `atoms.calc = calc`、`FIRE(atoms, logfile=None)` または `BFGS(atoms, logfile=None)` を `fmax`, `max_steps` で実行
- `optimizer` choice 外なら `ValueError`
- 収束失敗 (`steps == max_steps` でも `fmax` に届かず) は **warn のみ**、relaxed_atoms はそのまま返す → CLI 側で `meta.json` に記録
- `atoms` は破壊的に変更しない (内部で copy)
- `info_dict["converged"]` は ASE optimizer の収束結果に追従

## 7. align

`reactx/align.py` の `align_product_to_reactant`:

- 引数 `bond_changes_product` を **削除**
- 引数 `heavy_mapping`, `reactant_h_groups`, `product_h_groups` は維持
- ロジックの 1+1 限定箇所なし → そのまま multi-bond に動く (確認済み)
- 既存 H permutation (重原子内グループ permutations 全探索) も維持
- minimize_rotation_and_translation も維持

## 8. CLI 改修

### 8.1 削除する flag

`--neb-refine`, `--neb-images`, `--seed`, `--traj-stride`, `--relax-fmax`

### 8.2 維持する flag

`--backend {uma,lj}`, `--model uma-m-1p1`, `--render`, `--blender-exe`

### 8.3 新規 flag

なし (TOML で全て制御)

### 8.4 `_cmd_run` フロー (擬似コード)

```python
def _cmd_run(args):
    cfg = load_config(args.rxn_path)        # 新 schema
    r_mol, p_mol, heavy_mapping = parse_rxn(args.rxn_path)
    r_h, p_h = Chem.AddHs(r_mol), Chem.AddHs(p_mol)
    bond_changes = BondChanges.from_reaction_diff(r_mol, p_mol)
    calc = make_calculator(args.backend, ...)

    # R side
    _, _, pos_r = embed_fragments_to_positions(r_mol)
    pos_r = simple_placement(
        r_h, pos_r, bond_changes,
        initial_separation=cfg.placement.initial_separation,
        side="reactant", orientation=cfg.placement.orientation,
    )
    atoms_r = build_atoms_from_positions(r_h, pos_r)
    atoms_r_relaxed, info_r = relax_endpoint(
        atoms_r, calc,
        fmax=cfg.endpoint_relax.fmax,
        max_steps=cfg.endpoint_relax.max_steps,
        optimizer=cfg.endpoint_relax.optimizer,
    )

    # P side  (anchor = bond_changes.broken、heavy_mapping で R→P index 変換)
    _, _, pos_p = embed_fragments_to_positions(p_mol)
    pos_p = simple_placement(
        p_h, pos_p, bond_changes,
        initial_separation=cfg.placement.initial_separation,
        side="product", orientation=cfg.placement.orientation,
        heavy_mapping=heavy_mapping,
    )
    atoms_p = build_atoms_from_positions(p_h, pos_p)
    atoms_p_relaxed, info_p = relax_endpoint(
        atoms_p, calc,
        fmax=cfg.endpoint_relax.fmax,
        max_steps=cfg.endpoint_relax.max_steps,
        optimizer=cfg.endpoint_relax.optimizer,
    )

    # Align
    rH = heavy_to_hydrogen_groups(r_h)
    pH = heavy_to_hydrogen_groups(p_h)
    atoms_p_aligned = align_product_to_reactant(
        atoms_r_relaxed, atoms_p_relaxed, heavy_mapping, rH, pH,
    )

    # NEB
    xyz = args.output / "trajectory.xyz"
    neb_info = run_neb(
        atoms_r_relaxed, atoms_p_aligned, calculator=calc,
        n_images=cfg.neb.n_images, output_xyz=xyz,
        fmax=cfg.neb.fmax, max_steps=cfg.neb.max_steps,
        k=cfg.neb.k, climb=cfg.neb.climb, pad_frames=cfg.neb.pad_frames,
    )

    write_meta(args.output, cfg, info_r, info_p, neb_info, t_start)
    if args.render: _invoke_blender(args, xyz)
    return 0
```

### 8.5 `meta.json` 新構造

```json
{
  "backend": "uma",
  "description": "...",
  "wall_clock_seconds": 423.1,
  "endpoint_relax_r": {
    "converged": true, "final_fmax": 0.0089, "n_steps": 132
  },
  "endpoint_relax_p": {
    "converged": true, "final_fmax": 0.0094, "n_steps": 156
  },
  "neb": {
    "n_images": 11,
    "converged": true,
    "final_fmax": 0.043,
    "image_energies": [...],
    "pad_frames": 0
  },
  "effective_params": {
    "placement": {"initial_separation": 4.0, "orientation": "default"},
    "endpoint_relax": {"fmax": 0.01, "max_steps": 500, "optimizer": "FIRE"},
    "neb": {"n_images": 11, "fmax": 0.05, "max_steps": 200, "k": 1.0, "climb": true, "pad_frames": 0}
  }
}
```

`trials[]`, `placement.n_candidates/n_blocked/n_valid`, `selected_trial`, `formed_thresholds`, `broken_thresholds`, `product_distance_residual`, `formed_latch_count`, `broken_latch_count`, `initial_latched_*`, `neb_refined` フィールドはすべて削除。

## 9. テスト

### 9.1 削除する test

- `tests/test_afir_constraint.py`
- `tests/test_artificial_force.py`
- `tests/test_path_relax.py`
- `tests/test_scoring.py`
- `tests/test_neb_refine_sn2.py`
- `tests/test_cli_neb_refine_guard.py`
- `tests/test_config_afir.py`
- `tests/test_wallclock_sn2.py` (AFIR 想定の wall-clock assertion なので無効化)

### 9.2 改修する test

- `tests/test_re1_sn2.py`: NEB trajectory のフレーム数 == n_images、最終フレームの C-O 距離 ≤ formed threshold, C-Cl 距離 ≥ broken threshold、**endpoint R 上で Cl-C-O 角度 ≥ 150° を確認** (Phase 10 と同じ Walden assertion を endpoint で評価)
- `tests/test_re1_proton_transfer.py`, `tests/test_re1_menshutkin.py`, `tests/test_re3_e2.py`, `tests/test_re3_sn1_dissoc.py`, `tests/test_re4_sn1_recomb.py`, `tests/test_diels_alder_simple.py`, `tests/test_diels_alder_endo.py`: 同様に endpoint geometry + NEB frame 数の assertion に書き直し
- `tests/test_examples_menshutkin.py`: 同上
- `tests/test_placement.py`: simple_placement の単体テストに置換
- `tests/test_align.py`: bond_changes_product 引数削除後の挙動確認
- `tests/test_config.py`: 新 schema validation テスト (旧 schema → ConfigError、新 schema → 正しく parse)
- `tests/test_cli.py`, `tests/test_cli_unimolecular.py`: NEB trajectory 前提 / 新 meta.json 構造

### 9.3 新規 test

- `tests/test_bond_changes.py`: 既存に `from_reaction_diff` のテスト追加 (8 反応すべて自動推論結果を assert)
- `tests/test_endpoint_relax.py`: FIRE/BFGS 切替、収束フラグ、未収束時の warn 挙動
- `tests/test_neb.py`: `run_neb` 戻り値に image_atoms 含まれることの確認

### 9.4 marker 維持

`pytest -m slow` で 8 反応 UMA 統合 test、`pytest -m blender` で Blender smoke test、それ以外は高速。

## 10. 段取り (実装計画)

順序は writing-plans で詰めるが、依存関係の骨格:

1. **schema 刷新** — `reactx/config.py` を新 dataclass + validator に置換。Phase 9/10 旧 schema は `ConfigError`
2. **bond_changes 自動推論** — `BondChanges.from_reaction_diff` 追加、`from_atom_map_pairs` 削除
3. **simple_placement** — `reactx/placement.py` を縮小、新 API
4. **endpoint_relax** — 新規モジュール
5. **align 一般化** — `reactx/align.py` の `bond_changes_product` 撤廃
6. **neb API 改修** — `run_neb` 戻り値拡張
7. **cli 全面書き直し** — 上記すべてを束ねる、新 `meta.json` 構造
8. **examples 8 ファイルを新 schema へ**
9. **AFIR モジュール削除** (artificial_force / path_relax / scoring)
10. **test 整理** (削除 / 改修 / 新規)
11. **README 全面更新** — Phase 11 changes 節、新 schema、アーキテクチャ図、wall-clock 再測

各 step を小さな commit にし、各 step の test が green になることを順次確認しながら進める。

## 11. リスク・未解決事項

- **endpoint relax の minimum 品質**: `fmax=0.01` で 500 step まで回しても収束しない反応があるかもしれない (特に E2 / DA の歪んだ初期配置)。未収束時は warn して NEB に渡す → NEB が saddle ではなく minimum 側に滑り落ちる可能性。代替: BFGS への自動切替や、収束失敗時の rc=1 exit を検討。Phase 11 step 4 で実測して判断。
- **simple_placement で fragment が衝突するケース**: `initial_separation=4.0` で vdW 半径合計より小さい配置になる反応 (大きい aryl 基など)。対応: relax で必ず分離するはずなので Phase 11 では追加の vdW チェックを **入れない**。問題が顕在化したら `initial_separation` を反応別に override する。
- **DA の orientation="default" 定義**: 現行 multi_anchor の "achiral 縮約形" のうち 1 つを採用。butadiene + ethylene のように対称な系では endo/exo が無いので "default" でよい。CP + MA のような非対称系で "default" を呼ぶと exo か endo か未定義になるので **TOML で明示必須** → `orientation="default"` + bridges>=2 で非対称系の場合は `ConfigError` を検討 (実装で明示)。
- **`reactx/scoring.py` の `cordero_radii_for_atoms` 依存**: 関数は `reactx/covalent_radii.py` にあるため scoring.py 削除で問題なし。`reactx/scoring.py` import 元 (cli.py) を全て切り離す。
- **wall-clock 予測**: AFIR 64 trial が消えるが、endpoint relax 2 回 + NEB 1 回が代わりに走る。実測 SN2 (Phase 9) 232s から推定: endpoint relax ~ 60s × 2 + NEB ~ 300s = **~7-8 min**。Phase 10 と同等。
- **least-bad fallback の消失**: Phase 9/10 は `reached_product=False` でも trial を選んで何か出していた。Phase 11 では NEB が動かなければ trajectory が空になる。NEB の `converged=False` 時も image 列は書き出すので、可視出力は得られる (品質は別問題)。
- **E2 の strict reached_product xfail (Phase 10) の扱い**: Phase 11 では `reached_product` 自体が存在しない。E2 endpoint geometry が正しく与えられていれば NEB は通る想定だが、要実測。
- **Phase 10 で SN2 Walden を達成した `pre_relax_steps` の機能**: Phase 11 では endpoint_relax が同じ役割を果たす (むしろもっと厳密)。SN2 R 側で OH⁻ を CH3 側に正しく動かす物理は同じく ion-dipole 相互作用なので、`fmax=0.01` まで relax すれば Walden 軸に収束するはず。要検証。

## 12. 削除する README 言及

- "Phase 8/9/10 changes" 節 (Phase 11 changes に統合)
- "対応反応" の 8 反応 / placement_kind 表 (NEB ベースで再構成)
- "per-pair AFIR with sticky latch" / "FIRE 緩和" 表現
- Wall-clock 表 (Phase 11 再測値で更新)
- "詳細仕様: docs/superpowers/specs/2026-05-08-afir-force-design.md" / "2026-05-15-phase-10-pre-relax-design.md" (Phase 11 spec を pointer に)
- アーキテクチャ図 (本 spec §3.1 を反映)
