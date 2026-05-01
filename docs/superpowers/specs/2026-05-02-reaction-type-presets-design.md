# reactx — Reaction-Type Presets Design

- Status: Draft (awaiting user review)
- Date: 2026-05-02
- Owner: @kam6y
- Branch: `phase-Re1` (from 2012d11)
- 前提仕様: `docs/superpowers/specs/2026-04-27-reactx-phase-Re1-design.md`

## 1. 目的

Phase Re1 default 拘束 (`k_form=0.5, k_broken=1.0, r_broken=4.0, max_relax_steps=100`) は陰イオン求核剤の SN2 / proton transfer (= 外部熱的) 用にチューニングされている。中性求核剤 + イオン対生成 (Menshutkin: NH₃ + CH₃Cl → CH₃NH₃⁺ + Cl⁻) のような **内部熱的反応** では QM のバリア勾配が default の Hookean に勝って TS 手前で停滞し、別パラメータセット (`k_form=2.0, k_broken=2.0, r_broken=5.0, max_relax_steps=200`) が必要になる。

現状この知見はユーザーの手元 (auto-memory) に蓄積されているが、CLI から再現するには都度フラグ列を手で打つ必要がある。本変更でこれを **「反応タイプ別プリセット」** として CLI 引数 1 つで切り替え可能にし、各プリセットを自動テストで動作保証する。

## 2. スコープ

### 2.1 In scope

- CLI フラグ `--reaction-type {sn2_anion, proton_transfer, menshutkin}` を新設
- 上記 3 プリセットを `reactx/presets.py` に定義 (組み込み)
- 既存の個別フラグ (`--k-form` 等) によるプリセット値の上書きを許可
- `meta.json` に選択されたプリセット名と effective params を記録
- `examples/menshutkin.rxn` を追加
- 上記を検証する unit test + slow integration test
- README に使い方とプリセット表を追記
- 旧 memory (`memory/project_reactx_endothermic_tuning.md`) を削除

### 2.2 Out of scope

- ユーザー定義プリセット (YAML / TOML 読み込み) — 必要が出てから追加
- 反応タイプの自動検出 (.rxn 構造からの推論) — 名前による明示指定のみ
- `--n-angles` / `--cone-half-deg` のプリセット化 — これらは反応物理ではなく探索パラメータ
- generic bond-change engine (Phase 2 本丸) — 別 spec
- E2 / SN1 / E1 / β-H elimination 用プリセット — 上記が解禁されてから追加

## 3. プリセット定義

### 3.1 同梱する 3 プリセット

| name | k_form | k_broken | r_broken (Å) | max_relax_steps | r_form 上書き | 想定反応 |
|---|---|---|---|---|---|---|
| `sn2_anion` | 0.5 | 1.0 | 4.0 | 100 | なし (元素表) | 陰イオン求核剤の SN2 (例: F⁻ + CH₃Cl) |
| `proton_transfer` | 0.5 | 1.0 | 4.0 | 100 | 1.05 | 中性間 PT (例: HCl + NH₃) |
| `menshutkin` | 2.0 | 2.0 | 5.0 | 200 | なし (元素表) | 中性求核剤 → イオン対 (例: NH₃ + CH₃Cl) |

`r_form=なし (元素表)` は `artificial_force.lookup_r_form` で formed 結合の元素ペアから自動取得することを意味する (現挙動と同じ)。

### 3.2 プリセット選定の根拠

- `sn2_anion`: Phase Re1 default をそのまま命名。既存の `tests/test_re1_sn2.py` で動作確認済み (commit 1495cb1, 6eb4c8d)。
- `proton_transfer`: README に既出の `--r-form 1.05` と等価。`tests/test_re1_proton_transfer.py` で動作確認済み。
- `menshutkin`: ユーザーが auto-memory で蓄積した実測レシピ。session bc26ded9 で動作実証。

### 3.3 デフォルト挙動

`--reaction-type` 省略時は `sn2_anion` プリセットを暗黙適用する。これにより:

- 既存の CLI 呼び出し (フラグなし) は数値的に同じ挙動を保つ (= 後方互換)
- meta.json には常に `reaction_type` フィールドが入る (null になることはない)

## 4. アーキテクチャ

### 4.1 新規ファイル

#### `reactx/presets.py`

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ReactionPreset:
    name: str
    k_form: float
    k_broken: float
    r_broken: float
    max_relax_steps: int
    r_form: float | None = None  # None = 元素表 lookup

PRESETS: dict[str, ReactionPreset] = {
    "sn2_anion": ReactionPreset(
        name="sn2_anion",
        k_form=0.5, k_broken=1.0,
        r_broken=4.0, max_relax_steps=100,
    ),
    "proton_transfer": ReactionPreset(
        name="proton_transfer",
        k_form=0.5, k_broken=1.0,
        r_broken=4.0, max_relax_steps=100,
        r_form=1.05,
    ),
    "menshutkin": ReactionPreset(
        name="menshutkin",
        k_form=2.0, k_broken=2.0,
        r_broken=5.0, max_relax_steps=200,
    ),
}

def get_preset(name: str) -> ReactionPreset:
    """Return the preset by name. Raises ValueError if unknown."""
    if name not in PRESETS:
        raise ValueError(
            f"Unknown reaction-type preset {name!r}. "
            f"Available: {sorted(PRESETS)}"
        )
    return PRESETS[name]
```

設計判断:
- `frozen=True` の dataclass で immutability を担保
- 組み込みは Python dict で十分 (3 件 + 将来追加でも 10 件オーダー想定)。YAML 読み込みは out of scope
- `r_form: float | None` の None は「元素表 lookup」を意味する一級値

### 4.2 `reactx/cli.py` 変更

#### 4.2.1 argparse 変更

| flag | 旧 default | 新 default | 備考 |
|---|---|---|---|
| `--reaction-type` | (なし) | `"sn2_anion"` | 新規。choices は `sorted(PRESETS)` |
| `--r-form` | `None` | `None` | 変更なし (既に None) |
| `--r-broken` | `4.0` | `None` | sentinel に変更 |
| `--k-form` | `0.5` | `None` | sentinel に変更 |
| `--k-broken` | `1.0` | `None` | sentinel に変更 |
| `--max-relax-steps` | `100` | `None` | sentinel に変更 |

`--n-angles`, `--cone-half-deg`, `--seed`, `--relax-fmax`, `--traj-stride`, `--neb-*`, `--render`, `--blender-exe`, `--backend`, `--model` は無変更。

#### 4.2.2 上書きセマンティクス

`_cmd_run` の冒頭でプリセット読み込み後、None の各フィールドだけプリセット値で埋める:

```python
preset = get_preset(args.reaction_type)
k_form = preset.k_form if args.k_form is None else args.k_form
k_broken = preset.k_broken if args.k_broken is None else args.k_broken
r_broken = preset.r_broken if args.r_broken is None else args.r_broken
max_relax_steps = (
    preset.max_relax_steps if args.max_relax_steps is None else args.max_relax_steps
)
# r_form: 個別 flag > preset.r_form > 元素表 (lookup_r_form)
if args.r_form is not None:
    r_form_target = float(args.r_form)
elif preset.r_form is not None:
    r_form_target = preset.r_form
else:
    r_form_target = lookup_r_form(syms_r[formed_pair[0]], syms_r[formed_pair[1]])
```

この `r_form_target` 算出は現在 `_cmd_run` の中段にあるロジックを置き換える。

#### 4.2.3 meta.json 拡張

既存フィールドに加えて以下を追加:

```json
{
  ...,
  "reaction_type": "menshutkin",
  "effective_params": {
    "k_form": 2.0,
    "k_broken": 2.0,
    "r_broken": 5.0,
    "max_relax_steps": 200,
    "r_form": 1.47
  }
}
```

`effective_params.r_form` には実際に拘束に使った Å 値 (元素表 lookup の結果含む) を記録する。

### 4.3 変更しないモジュール

- `reactx/artificial_force.py` (build_restraints の API 不変)
- `reactx/path_relax.py` (relax_with_restraints の API 不変)
- `reactx/bond_changes.py`
- `reactx/embed3d.py`
- `reactx/scoring.py`
- `reactx/trials.py`
- `reactx/calculators.py`
- `reactx/neb.py`
- `reactx/rxn_parser.py`
- `reactx/align.py`
- `blender/render.py`

プリセットは CLI 層に閉じた設計とする。コア関数は引き続き scalar / dict を直接受ける。

## 5. テスト

### 5.1 Unit (fast, 新規)

`tests/test_presets.py`:

1. `get_preset("sn2_anion")` → 期待値 (k_form=0.5 等)
2. `get_preset("proton_transfer")` → 期待値 (r_form=1.05)
3. `get_preset("menshutkin")` → 期待値 (k_form=2.0, max_relax_steps=200 等)
4. `get_preset("unknown")` → `ValueError`、メッセージに利用可能名一覧
5. (CLI レベル) argparse の `choices` で未知名が弾かれる

### 5.2 CLI 上書きセマンティクス (fast, 新規)

`tests/test_cli.py` (既存ファイルに追加):

- `--reaction-type menshutkin` のみ → effective k_form=2.0, k_broken=2.0, r_broken=5.0, max_relax_steps=200
- `--reaction-type menshutkin --k-form 3.0` → effective k_form=3.0, 他は preset
- `--reaction-type sn2_anion --r-form 1.5` → effective r_form=1.5
- `--reaction-type proton_transfer` のみ → effective r_form=1.05
- フラグなし → effective = sn2_anion preset 値 (後方互換)

実装は CLI を実走させず、`build_parser().parse_args(...)` + 上書きロジックを切り出した関数を直接呼ぶ単体テスト。Calculator や relaxation は呼ばない。

### 5.3 Integration (slow, 既存 + 新規)

| ファイル | 内容 |
|---|---|
| `tests/test_re1_sn2.py` | **無変更** で pass (フラグなし default = sn2_anion preset) |
| `tests/test_re1_proton_transfer.py` | 既存ケース無変更 + `--reaction-type proton_transfer` ケース 1 件追加 |
| `tests/test_re1_menshutkin.py` (新規) | `examples/menshutkin.rxn` を `--reaction-type menshutkin` で走らせ、`reached_product=True` の trial が ≥1 件、trajectory で C-N が 2.5 Å → 1.5 Å に短縮、C-Cl が 1.8 Å → ≥3.5 Å に伸長することを確認 |
| `tests/test_wallclock_sn2.py` | 無変更 |
| `tests/test_neb_refine_sn2.py` | 無変更 |
| `tests/test_blender_smoke.py` | 無変更 (既存 SN2 + PT パラメトリゼーション) |

### 5.4 Examples

`examples/menshutkin.rxn` を新規追加:
- 反応: NH₃ + CH₃Cl → CH₃NH₃⁺ + Cl⁻
- 形式は既存 `examples/sn2.rxn` / `examples/proton_transfer.rxn` と同形式
- 形成: N-C / 切断: C-Cl

## 6. README 更新

`README.md` に「Reaction-type presets」節を追加 (既存「使い方」直後)。内容:

- プリセット 3 件のパラメータ表 (本 spec §3.1 と同一)
- CLI 例 3 件:
  ```bash
  reactx run examples/sn2.rxn -o out/sn2/ --backend uma --render
  reactx run examples/proton_transfer.rxn -o out/pt/ \
    --reaction-type proton_transfer --backend uma --render
  reactx run examples/menshutkin.rxn -o out/men/ \
    --reaction-type menshutkin --backend uma --render
  ```
- 「個別フラグ (`--k-form` 等) を指定するとプリセット値を上書きする」を 1 文
- **Menshutkin プリセットの根拠** を 2 文 — 「default は陰イオン求核剤の外部熱的反応用。中性求核剤がイオン対を生成する内部熱的反応では QM バリア勾配が default Hookean に勝って TS 手前で停滞するため、k_form / k_broken / r_broken / max_relax_steps を強化する」

これにより memory に蓄積していた知見はすべて README + コード (presets.py) で完結する。

## 7. メモリ削除

- `memory/project_reactx_endothermic_tuning.md` をファイルごと削除
- `memory/MEMORY.md` の該当エントリ行 (`- [reactx endothermic / ion-pair tuning](...)`) を削除

## 8. Definition of Done

1. `pytest tests/test_presets.py tests/test_cli.py` が pass (fast)
2. `pytest -m slow tests/test_re1_sn2.py tests/test_re1_proton_transfer.py tests/test_re1_menshutkin.py` が全 pass
3. `reactx run examples/menshutkin.rxn --reaction-type menshutkin -o out/men/ --backend uma` が `selected_trial >= 0` かつ `meta.json` の `trials[].reached_product` で ≥1 件 True
4. `meta.json` に `reaction_type` と `effective_params` が記録される (3 反応すべてで確認)
5. README に「Reaction-type presets」節があり、3 つの CLI 例と Menshutkin の根拠 2 文が含まれる
6. `memory/project_reactx_endothermic_tuning.md` と `MEMORY.md` の該当行が削除されている
7. 既存の `tests/test_re1_sn2.py` と `tests/test_re1_proton_transfer.py` の既存ケースが無変更で pass (後方互換性)

## 9. リスクと緩和

| リスク | 影響 | 緩和 |
|---|---|---|
| `examples/menshutkin.rxn` の生成構造が UMA で stable に最適化できない | DoD 3 失敗 | session bc26ded9 で動作実証済みの構造 / 設定をそのまま使用。失敗時は trajectory 末尾フレームの bond 距離を確認 |
| argparse default を None に変える破壊的変更で既存 CI / スクリプトが壊れる | 後方互換性破綻 | テスト 5.2 で「フラグなし → sn2_anion preset = 旧 default 値」を明示的に検証。挙動は数値的に同一 |
| Menshutkin slow test が CI/local で 200s × 8 angle = 数分かかる | 開発体験悪化 | `@pytest.mark.slow` でデフォルト除外。SN2 (~67s) / PT (~116s) より長くなることを README の wall-clock 表に追記 |
| プリセット値が将来の UMA model upgrade で陳腐化 | 動作劣化 | `meta.json` に `effective_params` を残すため再現/比較可能。プリセットは git 管理下 (presets.py) なので diff で変更履歴が追える |

## 10. Phase Re1 完了条件への影響

本変更は Phase Re1 spec §9 の DoD には影響しない (既存テストはすべて後方互換)。Phase Re1 の上に「使い勝手の改善」を一段重ねる位置付け。
