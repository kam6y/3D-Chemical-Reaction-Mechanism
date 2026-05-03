# reactx Phase 6 — `.rxn.toml` Sidecar Config Design

- Status: Draft (awaiting user spec review)
- Date: 2026-05-03
- Owner: @kam6y
- Branch: `phase-6` (from `develop` after Phase 4 merge)
- 前提仕様: `docs/superpowers/specs/2026-05-03-phase-4-sn1-recomb-design.md`
- 後方互換性: **完全に放棄する**。CLI フラグ・`meta.json` スキーマ・内部 API すべて breaking change を許容。

## 1. 目的

CLI が抱えてしまった「物理パラメータの羅列フラグ」を `.rxn` と同じ stem の sidecar TOML (`<name>.rxn.toml`) に追い出し、反応クラス固有のチューニング値・結合変化情報を反応ファイルと一体で版管理できるようにする。これにより:

- `--reaction-type` および `--k-* / --r-* / --max-relax-steps / --n-angles / --cone-half-deg / --no-mmff-prescreen / --prescreen-keep / --prescreen-steps` を **全廃止**。CLI は環境依存・実行ごと変化するフラグだけに絞る。
- `compute_bond_changes` による atom-map diff の自動計算をやめ、**`formed`/`broken` を TOML で明示記述**。誤検出 (formal charge 差・結合次数誤判定など) と「atom-map ミスで黙って違う反応が走る」事故を原理的に排除。
- `presets.py` を削除して反応クラスを runtime ライブラリではなく、example のリポジトリ内 `*.rxn.toml` 6 本として表現。

## 2. Non-goals

明示的に **このフェーズではやらない** こと:

- **`.rxn` フォーマット本体の改造**: MDL Rxn は無改変。Sidecar TOML を別ファイルとして読む。
- **後方互換性**: 既存 CLI コマンドラインや `meta.json` 旧キーの維持はしない。テスト・README・examples は新仕様に揃える。
- **新しい反応クラスの追加**: Phase 6 は既存 6 反応 (sn2, proton_transfer, menshutkin, e2, sn1_dissoc, sn1_recomb) の TOML 化のみ。
- **NEB refine の対象拡大**: Phase 3/4 の「`len(formed)==1 and len(broken)==1` のみ許可」guard はそのまま。
- **TOML から MOL ブロックを生成する形式**: `.rxn` を捨てる選択肢 C は今回採らない。
- **CLI の sidecar パス上書き**: `<rxn_path>.toml` を機械的に join する。`--config` フラグで別パス指定はしない (YAGNI)。

## 3. 設計方針: required sidecar TOML

### 3.1 配置と命名

```
examples/sn2.rxn         (MDL Rxn, 構造のみ — 既存 6 本そのまま)
examples/sn2.rxn.toml    (新規, このフェーズで 6 本作成)
```

- `<rxn_path>` を CLI に渡したら、`<rxn_path>.with_suffix(".rxn.toml")` ではなく **`<rxn_path>.parent / (<rxn_path>.name + ".toml")`** を読む (シンプルに `.toml` 追加)。これで `.rxn` を `.rxn.toml` の単独 stem としてもよく、間違いが少ない。
- TOML が **存在しない / 読めない / 必須キー欠落** の場合は CLI が exit code 2 で終了し、エラーメッセージで期待パスとサンプルスキーマを案内する。

### 3.2 スキーマ

```toml
# Free-form description; written verbatim into meta.json for traceability.
description = "SN2: CH3Cl + OH- -> CH3OH + Cl-"

# Atom-map number pairs for bond changes (REQUIRED).
# Numbers refer to the AtomMapNum fields in the .rxn file (NOT 0-based heavy idx).
# At least one of formed / broken must be non-empty.
formed = [[1, 3]]
broken = [[1, 2]]

# Restraint parameters (REQUIRED block).
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
# r_form: OPTIONAL.
#   omitted             → element-pair table (Cordero) lookup per formed bond
#   scalar (e.g. 1.05)  → broadcast to every formed bond
#   list (e.g. [1.78])  → must match len(formed); per-bond override
# r_form = 1.05
# r_form = [1.78, 1.05]

# Sampling control (OPTIONAL block; defaults shown).
[sampling]
n_angles = 8
cone_half_deg = 30.0

# MMFF prescreen control (OPTIONAL block; defaults shown).
[prescreen]
enabled = true
keep = 3
steps = 30
```

#### 3.2.1 必須 / 省略可

| キー | 必須? | デフォルト |
|---|---|---|
| `description` | 必須 | — |
| `formed` | 必須 (空配列可) | — |
| `broken` | 必須 (空配列可) | — |
| `restraints.k_form` | 必須 | — |
| `restraints.k_broken` | 必須 | — |
| `restraints.r_broken` | 必須 | — |
| `restraints.max_relax_steps` | 必須 | — |
| `restraints.r_form` | 省略可 | 元素ペア表 lookup |
| `sampling.n_angles` | 省略可 | 8 |
| `sampling.cone_half_deg` | 省略可 | 30.0 |
| `prescreen.enabled` | 省略可 | true |
| `prescreen.keep` | 省略可 | 3 |
| `prescreen.steps` | 省略可 | 30 |

#### 3.2.2 検証ルール

検証は **2 段階** に分ける:

**Tier A: `load_config(rxn_path)` で行う schema 検証** (`.rxn` を読まずに済む):

1. `len(formed) == 0 and len(broken) == 0` → 「at least one bond change required」
2. `formed`/`broken` の各要素が長さ 2 の int リストでない → 「expected pair of int atom-map numbers」
3. `r_form` を list で渡したとき `len(r_form) != len(formed)` → 「r_form list length must match formed」
4. `restraints.k_form < 0` などの負値 → 「parameter must be non-negative」
5. `sampling.n_angles < 1`, `sampling.cone_half_deg <= 0` → range error
6. `prescreen.keep < 1`, `prescreen.steps < 1` → range error
7. **未知のキー** が TOML に含まれる → 「unknown config key」(typo 早期検知; `tomllib` の `load` 後に許可キー集合と照合)

**Tier B: `BondChanges.from_atom_map_pairs(formed, broken, atom_map_to_reactant_idx)` で行う構造検証** (`.rxn` を parse 済みの段階で `cli.py` 側から呼ぶ):

8. atom-map 番号が `.rxn` 内に存在しない → `KeyError("unknown atom-map number 99")`
9. 同一ペアが formed と broken の両方に出現する → `ValueError("formed and broken collide on (a, b)")`

すべて Tier A は `ValueError` 即終了、Tier B は `cli._cmd_run` の冒頭で発火し exit code 2 で終了する。

### 3.3 CLI の最終形

```
reactx run <rxn_path> -o <outdir>
  [--backend {uma,lj}] [--model <name>]
  [--seed <int>] [--relax-fmax <float>] [--traj-stride <int>]
  [--neb-refine [--neb-images <int>]]
  [--render [--blender-exe <path>]]
```

**廃止フラグ**: `--reaction-type`, `--k-form`, `--k-broken`, `--r-form`, `--r-broken`, `--max-relax-steps`, `--n-angles`, `--cone-half-deg`, `--no-mmff-prescreen`, `--prescreen-keep`, `--prescreen-steps`

**残すフラグの理由**:
- `--backend` / `--model` / `--blender-exe`: 環境依存。同じ反応を別ハードで走らせるたびに変える。
- `--seed`: 再現性デバッグで毎回変える。
- `--relax-fmax` / `--traj-stride`: 出力品質チューニング (TOML に書くほど反応固有でない)。
- `--neb-refine` / `--neb-images`: 1+1 反応のみ対応の制約があり、同じ `.rxn.toml` で NEB on/off を行き来したい。
- `--render` / `--blender-exe`: レンダリング有無は実行ごと選ぶ。

## 4. パイプライン全体像

```
.rxn  ──┐
        ├──> reactx.config.load_config(rxn_path)
.rxn.toml ┘                │
                           ▼
                  ReactionConfig (frozen dataclass)
                           │
        ┌──────────────────┼─────────────────────┐
        ▼                  ▼                     ▼
   BondChanges        restraint params      sampling/prescreen
        │                  │                     │
        ▼                  ▼                     ▼
   embed3d  ───>   build_restraints  ───>  sample_attack_rotations
                           │                     │
                           ▼                     ▼
                       prescreen (keep K of N) → path_relax → scoring
                                                    │
                                            (optional NEB refine)
                                                    ▼
                                         trajectory.xyz / meta.json
```

## 5. モジュール変更

### 5.1 新規 `reactx/config.py`

```python
@dataclass(frozen=True)
class RestraintConfig:
    k_form: float
    k_broken: float
    r_broken: float
    max_relax_steps: int
    r_form: float | list[float] | None  # None → element-table lookup

@dataclass(frozen=True)
class SamplingConfig:
    n_angles: int = 8
    cone_half_deg: float = 30.0

@dataclass(frozen=True)
class PrescreenConfig:
    enabled: bool = True
    keep: int = 3
    steps: int = 30

@dataclass(frozen=True)
class ReactionConfig:
    description: str
    formed: tuple[tuple[int, int], ...]   # atom-map pairs
    broken: tuple[tuple[int, int], ...]
    restraints: RestraintConfig
    sampling: SamplingConfig
    prescreen: PrescreenConfig

def load_config(rxn_path: Path) -> ReactionConfig:
    """Load <rxn_path>.toml and validate against ReactionConfig schema.

    Raises FileNotFoundError if sidecar is missing, ValueError on schema/range
    violations.
    """

def resolve_r_form_targets(
    cfg: ReactionConfig,
    syms: list[str],
    formed_idx_pairs: list[tuple[int, int]],
) -> list[float]:
    """Expand cfg.restraints.r_form into per-formed-bond targets.

    cfg.r_form == None       → [lookup_r_form(syms[a], syms[b]) for a, b in formed_idx_pairs]
    cfg.r_form is float      → [float(cfg.r_form)] * len(formed_idx_pairs)
    cfg.r_form is list[float]→ list(cfg.r_form) (length pre-checked in load_config)
    """
```

### 5.2 `reactx/bond_changes.py`

`compute_bond_changes` を **削除**。代わりに:

```python
def from_atom_map_pairs(
    formed_map: Sequence[tuple[int, int]],
    broken_map: Sequence[tuple[int, int]],
    atom_map_to_reactant_idx: dict[int, int],
) -> BondChanges:
    """Build a BondChanges from atom-map pairs and a reactant-side mapping."""
```

`BondChanges.__post_init__` の「formed と broken のどちらか1本以上」検証は残す。

### 5.3 `reactx/rxn_parser.py`

`parse_rxn` の戻り値 `(reactant, product, mapping)` の `mapping` (`reactant_idx → product_idx`) は据え置き (NEB refine の `bond_changes_product` 構築で必要)。新たに `atom_map_to_reactant_idx(reactant_mol)` を export して、`config.py` 側で TOML の atom-map 番号を 0-based heavy index に変換する用途で使う。`heavy_to_hydrogen_groups` は変更なし。

### 5.4 `reactx/cli.py`

- `build_parser` から廃止フラグを削除。`from reactx.presets import PRESETS as _PRESETS` 行削除。
- `_resolve_effective_params` を **削除**。代わりに `_cmd_run` 冒頭で `cfg = load_config(args.rxn_path)` し、以降は `cfg.restraints.k_form` 等を直接参照。
- `_resolve_effective_params` の `r_form_targets` 構築は `resolve_r_form_targets(cfg, syms_r, formed_pairs_idx)` で置換。
- `meta.json.effective_params` は形を保つ:
  ```json
  {
    "k_form": 0.5,
    "k_broken": 1.0,
    "r_broken": 4.0,
    "max_relax_steps": 100,
    "r_form_targets": [1.78]
  }
  ```
  に加えて `meta.json.description = cfg.description` を新設、`meta.json.reaction_type` キーを **削除**。
- prescreen の disable は `cfg.prescreen.enabled == False` で実現。CLI フラグは消える。
- unimolecular auto-clamp ロジックは残し、`cfg.sampling.n_angles` を初期値とする。`n_frags_reactant == 1 and cfg.sampling.n_angles > 1` の時に warning ログ + `effective_n_angles = 1` (現状と同じ振る舞い)。TOML 側に `n_angles = 1` を書いていれば warning は出ない。

### 5.5 `reactx/presets.py`

**ファイル削除。** `tests/test_presets.py` も削除。

### 5.6 `reactx/artificial_force.py`

`lookup_r_form` は `config.resolve_r_form_targets` から呼ばれるので残す。変更なし。

### 5.7 その他 (`prescreen.py`, `path_relax.py`, `scoring.py`, `embed3d.py`, `align.py`, `neb.py`, `trials.py`, `calculators.py`)

変更なし。これらの関数は CLI 層が組み立てた数値だけを受け取る純粋ユーティリティなので、上流が `args` から `cfg` に変わっても interface は不変。

## 6. example 移行表

各 `examples/<name>.rxn` の atom-map を再走査して TOML を作成する。

| `.rxn` | description | formed (map) | broken (map) | k_form | k_broken | r_broken | max_relax_steps | r_form | n_angles |
|---|---|---|---|---|---|---|---|---|---|
| sn2.rxn | "SN2 anion: CH3Cl + OH- -> CH3OH + Cl-" | `[[1,3]]` | `[[1,2]]` | 0.5 | 1.0 | 4.0 | 100 | 省略 | 8 |
| proton_transfer.rxn | "Proton transfer: HCl + NH3 -> Cl- + NH4+" | `[[1,3]]` | `[[1,2]]` | 0.5 | 1.0 | 4.0 | 100 | 1.05 | 8 |
| menshutkin.rxn | "Menshutkin: NH3 + CH3Cl -> CH3NH3+ + Cl-" | `[[1,5]]` | `[[5,9]]` | 2.0 | 2.0 | 5.0 | 200 | 省略 | 8 |
| e2.rxn | "E2 elimination: CH3CH2Cl + OH- -> CH2=CH2 + Cl- + H2O" | `[[4,5]]` | `[[2,5],[1,3]]` | 1.0 | 1.0 | 4.0 | 200 | 省略 | 8 |
| sn1_dissoc.rxn | "SN1 step 1 dissociation: (CH3)3CBr -> tBu+ + Br-" | `[]` | `[[1,5]]` | 0.0 | 2.0 | 6.0 | 200 | 省略 | **1** |
| sn1_recomb.rxn | "SN1 step 2 recombination: tBu+ + Cl- -> (CH3)3CCl" | `[[1,5]]` | `[]` | 1.0 | 0.0 | 4.0 | 200 | 省略 | 8 |

`sn1_dissoc.rxn` の `n_angles = 1` を TOML に明示することで、CLI 側の unimolecular auto-clamp ロジックと整合。auto-clamp は防御策として残すが、TOML が 8 と書いた場合は警告ログ + 1 へ強制 (Phase 3 と同じ挙動)。

## 7. テスト戦略

### 7.1 新規 `tests/test_config.py` (Tier A 検証)

- ファイル不在 → `FileNotFoundError`
- 必須キー欠落 → `ValueError("missing key 'restraints.k_form'")` 等
- 未知キー → `ValueError("unknown config key 'foo'")`
- `formed=[]` かつ `broken=[]` → `ValueError`
- `r_form` scalar / list / 省略の 3 パターンで `resolve_r_form_targets` が期待通り展開
- 範囲エラー (`n_angles=0`, `k_form=-1` など)
- formed/broken の各要素が長さ2の int でないとき `ValueError`

### 7.2 改修 `tests/test_bond_changes.py` (Tier B 検証)

- `compute_bond_changes` 関連テストを削除
- `BondChanges.from_atom_map_pairs` 正常系 (atom-map → 0-based heavy idx 変換が想定通り)
- 未知 atom-map 番号 → `KeyError`
- 同一ペアが formed と broken に出現 → `ValueError`
- 空 formed/broken → `BondChanges.__post_init__` の既存チェックに委譲

### 7.3 削除

- `tests/test_presets.py`

### 7.4 改修 `tests/test_re1_*.py`, `tests/test_re3_*.py`, `tests/test_re4_*.py`

- CLI 引数から `--reaction-type` および override flag を全削除
- `examples/*.rxn.toml` が常にロードされる前提で assertion を更新 (`meta.json.reaction_type` キーが消えた点)
- `meta.json.description` が TOML の値と一致することを assert

### 7.5 改修 `tests/test_cli_neb_refine_guard.py`

- `--reaction-type` 引数を削除
- `examples/*.rxn.toml` が必須なので、test fixture でも一時 TOML を併置

### 7.6 統合 smoke (slow)

既存 6 反応の slow テストが新仕様で全 pass することが Phase 6 の DoD。

## 8. README 更新

- 「Reaction-type presets」セクションを「Per-reaction `.rxn.toml` config」セクションに置換。
- 表は同じく example 6 行、ヘッダーは `description / formed / broken / k_form / k_broken / r_broken / max_relax_steps / r_form / n_angles` 等。
- 「使い方」コマンド例から `--reaction-type` を全削除:
  ```bash
  reactx run examples/sn2.rxn -o out/sn2/ --backend uma --render
  reactx run examples/menshutkin.rxn -o out/men/ --backend uma --render
  ```
- 「Migration: pre Phase 6」コラムは **書かない** (後方互換性放棄方針)。代わりに「Phase 6 で `--reaction-type` および `--k-* / --r-*` フラグは削除された」を1行記載するに留める。
- アーキテクチャ図を `.rxn + .rxn.toml → load_config → ReactionConfig → embed3d ...` に更新。

## 9. DoD 確認手順

1. `pytest -m "not slow and not blender"` 全 pass (新 `test_config.py` 含む)
2. `examples/*.rxn.toml` 6 本がコミットされている
3. 6 反応それぞれを新 CLI で実行:
   ```bash
   reactx run examples/sn2.rxn          -o out/sn2/         --backend uma
   reactx run examples/proton_transfer.rxn -o out/pt/       --backend uma
   reactx run examples/menshutkin.rxn   -o out/men/         --backend uma
   reactx run examples/e2.rxn           -o out/e2/          --backend uma
   reactx run examples/sn1_dissoc.rxn   -o out/sn1d/        --backend uma
   reactx run examples/sn1_recomb.rxn   -o out/sn1r/        --backend uma
   ```
   いずれも `meta.json.selected_trial >= 0`, `description` が TOML と一致
4. `pytest -m slow` で `test_re1_*` / `test_re3_*` / `test_re4_*` 全 pass
5. `--reaction-type` を渡すと `argparse` がエラー (廃止済みの確認)
6. `examples/sn2.rxn.toml` を一時的に消して `reactx run examples/sn2.rxn ...` を実行 → exit 2 + 期待パスを案内するエラーメッセージ
7. README の「Reaction-type presets」がなく「Per-reaction `.rxn.toml` config」に置換されている

## 10. 適用限界

| # | ケース | Phase 6 での扱い |
|---|---|---|
| 1 | TOML から MOL ブロックを生成 (反応構造を TOML 内に記述) | やらない (.rxn を引き続き使う) |
| 2 | `--config <path>` で sidecar を別パス指定 | やらない (`<rxn>.toml` 自動 join のみ) |
| 3 | 動的 preset (CLI で値を組み合わせる便利モード) | preset 概念ごと廃止 |
| 4 | 過去 CLI 互換 (`--reaction-type` 等) | 完全廃止、警告も出さない |
| 5 | 結合次数の変化 (σ/π 区別) | 引き続き Phase 5+ (今回の TOML スキーマには bond_order を入れない) |

## 11. リスク

- **TOML スキーマの硬直化**: `[restraints]` 等のセクション境界を切ると将来「`r_form` を per-bond 別 k_form に紐付けたい」のような拡張で破壊的変更が必要になる。Phase 6 では現行 preset で表現できる範囲の項目だけを入れ、より複雑な per-bond 表は後続フェーズで再設計可能なように、 dataclass + `@frozen` で固める方針。
- **未知キー検出の厳格さ**: TOML に typo (`prescreen.kept` 等) があると即エラーで停止する。これは仕様通りだが、ユーザーが手書きする以上、エラーメッセージで「あなたが入れた `prescreen.kept` は許可されていません。許可キー: keep, steps, enabled」を案内する必要がある。
- **atom-map 番号と heavy index の混乱**: TOML が atom-map (1-based, sparse), 内部処理が 0-based heavy index と二系統あるため、単体テストで誤解を防ぐ assertion を入れる。
- **`description` の自由文字列**: 検索性のために `meta.json` に書き込まれる。typo を許容するが、外部スクリプトで `description` を完全一致比較しないよう README に明記する。

## 12. 実装順序の推奨

1. `reactx/config.py` 新規 + `tests/test_config.py` 新規 (TDD)
2. `reactx/bond_changes.py` から `compute_bond_changes` 削除 + `from_atom_map_pairs` 追加 + 該当テスト改修
3. `examples/*.rxn.toml` 6 本作成 (table §6 通り)
4. `reactx/presets.py` 削除 + `tests/test_presets.py` 削除
5. `reactx/cli.py` の `build_parser` 整理 + `_resolve_effective_params` 削除 + `_cmd_run` で `cfg = load_config(...)` に置換
6. `tests/test_re1_*.py` / `tests/test_re3_*.py` / `tests/test_re4_*.py` / `tests/test_cli_neb_refine_guard.py` 改修
7. README 更新
8. `pytest -m "not slow"` → `pytest -m slow` の順に DoD 確認
