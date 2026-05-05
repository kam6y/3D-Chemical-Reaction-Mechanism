# reactx Phase 8 — Diels–Alder Cycloaddition (Multi-Anchor Placement) Design

- Status: Draft (awaiting user spec review)
- Date: 2026-05-05
- Owner: @kam6y
- Branch: `phase-re8` (from `develop` after Phase 7 merge)
- 前提仕様: `docs/superpowers/specs/2026-05-04-generic-placement-design.md`
- 後方互換性: **完全に放棄する**。TOML schema・`PlacementTrial` API・`meta.json` フィールドは breaking change を許容。

## 1. 目的

Phase 7 で確立した「全球面 Fibonacci サンプリング + steric blocking + UMA full relax」 のパイプラインは、`formed` が **substrate ↔ incoming fragment を 1 本だけ bridge する** ケース (現行 6 反応) に閉じている。`reactx/placement.py:_find_bridging_formed` は `len(bridges) > 1` で `NotImplementedError("multi-anchor placement (cycloaddition) is out of scope for Phase 7; ...")` を投げる。

本フェーズで:

- **Diels–Alder 系 (`bridges == 2`) を解放する**。`bridges == 2` で同一 incoming fragment の 2 つの anchor が substrate の 2 つの anchor と同時に結合するケースを、Phase 7 と同じ Fibonacci サンプリング枠組みの中で扱えるようにする。
- 2 anchor 同時到達のため、**rigid-body alignment (translation + Kabsch on 2 points)** を導入する。残る 1 自由度を **endo / exo の 2 値で離散化**し、対称な dienophile (ethylene 等) では RMSD 縮約で `achiral` 1 trial に統合する。
- `[restraints]` の `k_form` / `k_broken` / `r_broken` を `r_form` と完全対称な `scalar | list[float]` に拡張する。これにより per-bond な拘束力チューニングが Phase 8 系 (DA) でも、Phase ≤7 系 (E2 等) でも可能になる。
- 対応反応に **butadiene + ethylene** と **cyclopentadiene + maleic anhydride** の 2 つの DA を追加する。

## 2. Non-goals

明示的に **このフェーズではやらない** こと:

- `bridges >= 3` の cycloaddition (1,3-dipolar [3+2]、ene 反応の一部 等): `NotImplementedError` 維持
- Cheletropic 反応 (incoming 側 anchor が 1 つで substrate 側に 2 つ bridge): incoming 側 anchor の数 ≠ substrate 側 anchor の数 となる非対称トポロジーは将来 phase
- Metathesis (broken が fragment を跨ぐ): 既存 `NotImplementedError` 維持
- Termolecular (3 fragment 以上の同時会合): 既存 `NotImplementedError` 維持
- Open-shell / radical (spin > 1): 現状 `placement.py:233` の `info["spin"] = 1` ハードコードを維持
- NEB refine の `bridges == 2` 対応拡張: 現状の「`len(formed)==1 and len(broken)==1` のみ」guard は維持
- `formed = []` の `bridges == 0` ケース (cycloreversion など): スコープ外。SN1 dissoc の unimolecular passthrough のみ既存維持
- 動的 endo/exo 重み付け (kinetic / thermodynamic 区別) の scoring 介入: scoring は引き続き `peak_energy` 最小基準のみ
- `n_candidates` の cycloaddition 用デフォルト変更: 既存 `64` を維持し、内部で `2 × n_candidates` を展開

## 3. アーキテクチャ

### 3.1 全体フロー (Phase 7 比)

```
.rxn + .rxn.toml
  → rxn_parser + load_config → ReactionConfig
                                       │ (k_form, k_broken, r_broken も list 対応)
                                       ▼
                               BondChanges (formed / broken, 0-indexed)
                                       │
                                       ▼
                       embed_fragments_to_positions (変更なし)
                                       │
                                       ▼
                       placement.valid_placements
                       ├─ bridges == 0 (unimolecular)  → 既存 passthrough
                       ├─ bridges == 1                 → 既存 single-anchor (Phase 7)
                       ├─ bridges == 2                 → 新 multi-anchor (Phase 8)
                       │      │
                       │      ├─ Fibonacci on face direction (既存サンプラ流用、n=n_candidates)
                       │      ├─ rigid-body alignment per direction (translation + 2-point Kabsch)
                       │      ├─ endo/exo 展開 (×2)、対称検出で achiral 縮約
                       │      ├─ blocking: angular shadow + d_min ceiling + 新 unreachable / asymmetric
                       │      └─ survivor → PlacementTrial(orientation=…)
                       └─ bridges >= 3                 → NotImplementedError
                                       │
                                       ▼
                       FIRE + per-bond Hookean / PullApart 拘束
                                       │
                                       ▼
                       scoring → best trial (peak_energy 最小)
                                       │
                                       ▼
                       trajectory.xyz → blender/render.py → .blend
```

### 3.2 既存からの主要変更

| 項目 | Phase 7 | Phase 8 |
|---|---|---|
| Bridge 数 | 1 のみ | 1 または 2 |
| Placement DOF | 1 anchor + 1 direction | 2 anchor pair + 1 direction + endo/exo binary |
| `PlacementTrial.orientation` | 無 | `Literal["single", "endo", "exo", "achiral"]` |
| `RestraintConfig.k_form` | `float` | `float \| tuple[float, ...]` |
| `RestraintConfig.k_broken` | `float` | `float \| tuple[float, ...]` |
| `RestraintConfig.r_broken` | `float` | `float \| tuple[float, ...]` |
| `meta.json["placement_kind"]` | 無 | `"single_anchor" \| "multi_anchor"` |
| `meta.json["trials"][i]["orientation"]` | 無 | 上記 4 値のいずれか |
| `_find_bridging_formed` 振る舞い | bridges>1 で `NotImplementedError` | bridges==2 を accept、>=3 で `NotImplementedError` |

## 4. TOML schema 拡張

### 4.1 `RestraintConfig` 拡張 (4 キー対称化)

```python
@dataclass(frozen=True)
class RestraintConfig:
    k_form: float | tuple[float, ...]                    # ← scalar | list (元: scalar only)
    k_broken: float | tuple[float, ...]                  # ← scalar | list (元: scalar only)
    r_broken: float | tuple[float, ...]                  # ← scalar | list (元: scalar only)
    max_relax_steps: int
    r_form: float | tuple[float, ...] | None = None      # 既存
```

### 4.2 検証ルール (4 キーで完全対称)

`config._normalize_<key>` ヘルパに集約。共通実装 `_normalize_per_bond_value(raw, n_bonds, key, source, *, allow_zero, allow_none)`:

- `None`: `r_form` のみ許容 (element-table fallback)。他のキーで `None` は `ValueError`
- `int | float` (scalar): broadcast。値域 `allow_zero` で 0 を許容するか分岐 (`k_*` は ≥ 0、`r_*` は > 0)
- `list`:
  - `len == n_bonds` を要求
  - 要素はすべて非負 (k) または 正 (r)
  - `n_bonds == 0` の場合は `[]` のみ valid (filler scalar も許容で、ただし list 形式の場合は空)

### 4.3 `resolve_*_targets` 関数

`config.py` に 4 関数を追加 (`resolve_r_form_targets` と完全対称):

```python
def resolve_k_form_targets(cfg, formed_idx_pairs) -> list[float]: ...
def resolve_k_broken_targets(cfg, broken_idx_pairs) -> list[float]: ...
def resolve_r_broken_targets(cfg, broken_idx_pairs) -> list[float]: ...
# resolve_r_form_targets は既存
```

内部実装は共通ヘルパ `_resolve_per_bond(value, n_bonds, default_per_bond_fn=None)` に集約 (DRY)。

### 4.4 既存 6 反応 TOML への影響

scalar 形式のままで valid (`scalar | list` の片側) なので **migration 不要**。Phase 8 では既存 TOML は無変更。

ただし E2 のように len(broken)=2 でかつ chemistry 上 per-bond 値が異なる方が望ましいケースは、別 phase で「`r_broken = [3.0, 5.0]` (C-H と C-Cl)」のように更新する余地が残る (本 phase スコープ外)。

### 4.5 新規 example TOML

`examples/diels_alder_simple.rxn.toml`:
```toml
description = "Diels-Alder: butadiene + ethylene -> cyclohexene"
formed = [[1, 5], [4, 6]]
broken = []

[restraints]
k_form = [1.0, 1.0]
k_broken = 0.0           # scalar (broken=[] なので filler、無視される)
r_broken = 4.0           # scalar (filler)
max_relax_steps = 200
```

`examples/diels_alder_endo.rxn.toml`:
```toml
description = "Diels-Alder endo: cyclopentadiene + maleic anhydride -> norbornene-2,3-dicarboxylic anhydride"
formed = [[1, 5], [4, 6]]
broken = []

[restraints]
k_form = [1.5, 1.5]      # MA は π 共役で stiff、初期チューニング値
k_broken = 0.0
r_broken = 4.0
max_relax_steps = 250
```

`k_form` の絶対値は実装 + 動作確認時にチューニング。Phase 7 の Menshutkin で `k_form=4.0` までチューンしたのと同じ要領で、UMA の repulsive PES を抑える値を実測で決める。

対応する `.rxn` ファイル (RDKit `$RXN` 形式) は別途作成。atom-mapping は:
- `1, 4`: diene 末端炭素 (substrate anchors)
- `5, 6`: dienophile 炭素 (incoming anchors)
- formed[0] = 1—5 (= A1—I1)
- formed[1] = 4—6 (= A2—I2)

### 4.6 自動検出ロジック (TOML には現れない)

`placement.valid_placements` の冒頭で `_find_bridging_formed` 同等のロジックで bridges を抽出し、`len(bridges)` で分岐:

```python
if len(bridges) == 1:
    return _single_anchor_placement(...)   # Phase 7 既存
elif len(bridges) == 2:
    return _multi_anchor_placement(...)    # Phase 8 新
else:
    raise NotImplementedError(
        f"multi-anchor placement with {len(bridges)} bridges is out of scope "
        f"for Phase 8; only bridges==1 or bridges==2 are supported"
    )
```

`meta.json` に `placement_kind: "single_anchor" | "multi_anchor"` を必ず出力 (debug + user-visible)。

## 5. Multi-anchor placement アルゴリズム

### 5.1 Anchor 同定

```python
bridges = [(a, b) for (a, b) in formed
           if (a in substrate_set and b in fragment_set)
           or (b in substrate_set and a in fragment_set)]
# bridges を「substrate 側 atom が前」に正規化
bridges_norm = [
    (a, b) if a in substrate_set else (b, a)
    for (a, b) in bridges
]
A1, I1 = bridges_norm[0]
A2, I2 = bridges_norm[1]
```

順序の意味: `bridges_norm[i]` は `formed[i]` 由来。`r_form[i]` / `k_form[i]` がこれに対応する (per-bond 拘束との対応)。

### 5.2 Per-direction rigid-body placement

各 Fibonacci 方向 `d ∈ S²` (n=n_candidates) について:

```
Setup (1 回だけ計算、direction loop の外):
  M_sub = (pos[A1] + pos[A2]) / 2
  v_sub = pos[A2] - pos[A1]
  L_sub = |v_sub|
  u_sub = v_sub / L_sub
  M_inc = (pos[I1] + pos[I2]) / 2          # initial fragment 内座標
  v_inc = pos[I2] - pos[I1]
  L_inc = |v_inc|
  u_inc = v_inc / L_inc

Per direction d:
  Step A (placement distance d_min):
    既存 compute_d_min を以下で適用:
      anchor = M_sub
      substrate atoms = {S \ {A1, A2}}
      incoming atoms = F (fragment 全原子)
      incoming anchor = M_inc
    → d_min ∈ ℝ

  Step B (translation):
    M_inc を M_sub + d × d_min に移動
    fragment 全原子に同 translation を適用
    → fragment 全原子の中間 positions 更新

  Step C (2-point Kabsch rotation):
    rotation R = 「u_inc を u_sub に揃える単一回転」
      axis = (u_inc × u_sub) / |u_inc × u_sub|     (degenerate 時は ⊥ u_sub に任意 axis)
      angle = arccos(clip(u_inc · u_sub, -1, 1))
    fragment 全原子に R を適用 (M_inc 中心)
```

`u_inc × u_sub` が 0 (= u_inc が ±u_sub に平行) のとき: `u_inc · u_sub > 0` なら rotation 不要 (identity)、`< 0` なら 180° rotation 必要。180° rotation の axis は u_sub に直交する任意の単位ベクトル (e.g., `[0,0,1]` を u_sub から成分除去して正規化) を選ぶ。

### 5.3 endo / exo 展開

Step C 後、fragment を「u_sub 軸 (= 配置後の v_inc 軸) まわりに 0° と 180°」で 2 trial 展開:

```python
trial_endo_positions = positions.copy()
# rotate fragment around u_sub by 180°, center at M_inc (現在位置)
trial_exo_positions = _rotate_atoms(
    positions, indices=fragment, axis=u_sub, center=M_inc + d * d_min, angle=π
)
```

**対称性検出**: `np.allclose(trial_endo_positions[fragment], trial_exo_positions[fragment], atol=0.01)` (= RMSD < 0.01 Å) なら同一とみなし、片方を `orientation="achiral"` 単独 trial に縮約。Ethylene のような C2 対称な dienophile は自動的に achiral 化される (D2h の C2 axis が C5-C6 軸と一致)。MA 等の非対称 dienophile では `"endo"` / `"exo"` 両 trial が残る。

### 5.4 Reachability blocking (新 reason)

各 trial 配置後、以下を check:

```python
b1 = float(np.linalg.norm(positions[I1] - positions[A1]))
b2 = float(np.linalg.norm(positions[I2] - positions[A2]))

if max(b1, b2) > d_min_ceiling:
    blocked, reason = True, f"unreachable_dual_anchor:b1={b1:.2f},b2={b2:.2f}"
elif abs(b1 - b2) / max(b1, b2) > 0.40:
    blocked, reason = True, f"asymmetric_dual_anchor:b1={b1:.2f},b2={b2:.2f}"
```

**40% 閾値の根拠**: `d` がほぼ ⊥ `u_sub` のときは b1 ≈ b2 (対称配置)、`d` がほぼ ∥ `u_sub` のときは max(b1, b2) − min(b1, b2) ≈ |L_sub − L_inc| になる。butadiene + ethylene (L_sub ≈ 3.6 Å、L_inc ≈ 1.34 Å) で 40% 閾値だと `|b1−b2| ≤ 0.4 × max(b1, b2)` → 概ね d が u_sub 方向に対し 60° 以上開いた方向のみ生存。値はコード定数 `DUAL_ANCHOR_ASYMMETRY_THRESHOLD = 0.40` で表現し、TOML には露出しない。実装後の動作確認で再チューン。

### 5.5 Steric blocking (Phase 7 流用、適応)

既存 `evaluate_direction` の角度シャドウ + d_min ceiling は **anchor_pos = M_sub、substrate_positions = S \ {A1, A2} の座標、substrate_vdw 同、incoming_positions = F の座標、incoming_anchor_pos = M_inc** として再利用。

API 変更なし: 既存 `evaluate_direction(d, anchor_pos, substrate_positions, ..., incoming_anchor_pos, ...)` をそのまま multi-anchor 経路から呼ぶだけ。

### 5.6 PlacementTrial / PlacementResult への変更

```python
@dataclass(frozen=True)
class PlacementTrial:
    direction: np.ndarray
    d_min: float
    positions: np.ndarray
    orientation: Literal["single", "endo", "exo", "achiral"] = "single"   # 新

@dataclass(frozen=True)
class PlacementResult:
    trials: list[PlacementTrial]
    n_candidates: int
    n_blocked: int
    blocked_reasons: list[str | None]
    placement_kind: Literal["single_anchor", "multi_anchor"]              # 新
```

- single-anchor 経路: `placement_kind="single_anchor"`、各 trial は `orientation="single"`
- multi-anchor 経路: `placement_kind="multi_anchor"`、各 trial は `"endo" | "exo" | "achiral"` のいずれか
- unimolecular passthrough: `placement_kind="single_anchor"` (1 trial、`orientation="single"`)

### 5.7 期待される trial 数

| 反応 | n_candidates | endo/exo | achiral 縮約 | raw trials | 想定生存 |
|---|---|---|---|---|---|
| butadiene + ethylene | 64 | 自動展開 | 適用 (D2h ethylene) | 64 (achiral 後) | 推定 8〜12 |
| cyclopentadiene + MA | 64 | 自動展開 | 適用されず | 128 | 推定 14〜22 |

実測値は実装後の動作確認で README 「Wall-clock (実測)」表に追記。

## 6. Restraints (per-bond k/r 対応)

### 6.1 `build_restraints` シグネチャ更新

```python
def build_restraints(
    atoms: Atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    *,
    r_form: float | list[float] | None = None,
    r_broken: float | list[float] = 4.0,
    k_form: float | list[float] = 0.5,
    k_broken: float | list[float] = 1.0,
) -> list:
```

内部で 4 引数すべてを `_broadcast(value, n_bonds) -> list[float]` で per-bond 展開してから zip ループで constraint を生成:

```python
syms = atoms.get_chemical_symbols()
r_forms   = _broadcast_r_form(r_form, formed, syms)        # element-table fallback
k_forms   = _broadcast(k_form,   len(formed))
r_brokens = _broadcast(r_broken, len(broken))
k_brokens = _broadcast(k_broken, len(broken))

constraints: list = []
for (a, b), rt, k in zip(formed, r_forms, k_forms, strict=True):
    constraints.append(Hookean(a1=a, a2=b, rt=rt, k=k))
for (a, b), rt, k in zip(broken, r_brokens, k_brokens, strict=True):
    constraints.append(PullApart(a1=a, a2=b, k=k, rt=rt))
return constraints
```

### 6.2 古いコメント更新

`artificial_force.py:88-98` の docstring 「Per-bond targets ... Phase 4+ when reactions with formed≥2 are added (Diels-Alder etc.)」は Phase 8 で達成されるため、現在形に書き換える。

### 6.3 呼び出し側 (`cli.py`)

`config.resolve_*_targets` ヘルパ群で per-bond `list[float]` に展開し、`build_restraints` に list を渡す。`build_restraints` 内部の `_broadcast` は defensive の二重防御として残す。

## 7. Scoring / meta.json

### 7.1 `score_trials` / `select_best_trial`

`trial.orientation` フィールドが追加されただけで、scoring ロジック自体は変更不要。最良 trial 選択は引き続き `peak_energy` 最小基準。endo/exo 選好の bias は導入しない (kinetic product としての endo は UMA のエネルギーで自然に選ばれることを期待)。

### 7.2 `meta.json` への新フィールド

```json
{
  "selected_trial": 7,
  "placement_kind": "multi_anchor",
  "trials": [
    {
      "trial_id": 0,
      "direction": [0.234, -0.812, 0.534],
      "d_min": 3.21,
      "orientation": "endo",
      "peak_energy": -1234.56,
      "reached_product": true,
      "wall_clock_seconds": 28.4
    }
  ]
}
```

新フィールド (breaking change):
- 上位: `placement_kind: "single_anchor" | "multi_anchor"` を **すべての反応で必ず出力** (Phase 7 ログ既存出力にも追加)
- trial 内: `orientation: "single" | "endo" | "exo" | "achiral"` を **すべての反応で必ず出力**

Phase 7 既存テストは `meta.json` の追加フィールドを ignore する形で互換 (`pytest` 側で `>=` 検査ではなく完全一致をしている箇所があれば更新)。

## 8. テスト

### 8.1 Unit (高速、`pytest`)

`tests/test_placement.py` 追加:
- `test_multi_anchor_two_bridges_butadiene_ethylene_mock` — synthetic mock geometry で multi_anchor path が trial を返すこと
- `test_multi_anchor_three_bridges_raises_not_implemented`
- `test_endo_exo_expansion_symmetric_collapses_to_achiral` — 対称 fragment に対する RMSD 縮約
- `test_unreachable_dual_anchor_blocking_reason` — `b1 > d_min_ceiling` ケース
- `test_asymmetric_dual_anchor_blocking_reason` — `|b1-b2|/max > 0.40` ケース
- `test_two_point_kabsch_antiparallel_uincusub_180deg` — `u_inc · u_sub == -1` の数値安定性

`tests/test_config.py` 追加:
- `test_k_form_scalar_and_list_both_valid`
- `test_k_form_list_length_must_match_formed`
- `test_k_broken_list_length_must_match_broken`
- `test_r_broken_scalar_then_resolve_per_bond_broadcast`
- `test_resolve_*_targets_helper_returns_list_of_correct_length`

`tests/test_artificial_force.py` 追加:
- `test_build_restraints_with_per_bond_k_and_r_lists`
- `test_build_restraints_scalar_then_internal_broadcast_matches_list_input`

### 8.2 Slow integration (`pytest -m slow`)

`tests/test_diels_alder_simple.py`:
- `selected_trial >= 0`
- `meta.json["placement_kind"] == "multi_anchor"`
- 全 trial で `orientation in {"endo", "exo", "achiral"}`
- `reached_product` ≥ 1 件 True
- 最終 geometry で `|C1-C5| ≤ 1.7 Å` かつ `|C4-C6| ≤ 1.7 Å` (cyclohexene σ bond 完成)

`tests/test_diels_alder_endo.py`:
- 同上 + `orientation` 別の peak_energy 比較で endo ≤ exo を期待 (kinetic product)、tolerance 5 kcal/mol で UMA 揺らぎを許容
- `meta.json` に `"endo"` と `"exo"` 両 trial が記録されていること

### 8.3 既存テスト変更

- `tests/test_placement.py` の `test_*_raises_not_implemented_for_multi_bridge` を Phase 8 仕様に合わせて修正 (bridges=2 で `NotImplementedError` を期待していた既存 test がもしあれば bridges=3 に置換)
- `meta.json` 完全一致検査をしている既存 test を `placement_kind` / `orientation` の追加に合わせて更新

## 9. Error handling

- `bridges == 0` (= 同一 fragment 内に bridge なし): 既存 `ValueError("fragment has no formed bond bridging to substrate")` 維持
- `bridges == 1`: 既存 single-anchor path
- `bridges == 2`: 新 multi-anchor path
- `bridges >= 3`: `NotImplementedError("multi-anchor placement with N>=3 bridges is out of scope for Phase 8")`
- `_broken_bridges_fragments == True` (metathesis): 既存 `NotImplementedError` 維持
- `len(non_substrate) > 1` (termolecular): 既存 `NotImplementedError` 維持
- multi_anchor で全 trial blocked: `RuntimeError("anchor pair (A1, A2) has no valid placement direction (all 2*n_candidates blocked); check substrate geometry / r_form / d_min_ceiling")`
- u_inc が degenerate (`L_inc == 0`、つまり I1 == I2 となるべき配置 = atom-mapping ミス): `ValueError("incoming anchor pair I1,I2 are coincident; check formed bond atom-mapping")`

## 10. ファイル変更一覧

新規:
- `examples/diels_alder_simple.rxn`
- `examples/diels_alder_simple.rxn.toml`
- `examples/diels_alder_endo.rxn`
- `examples/diels_alder_endo.rxn.toml`
- `tests/test_diels_alder_simple.py` (slow)
- `tests/test_diels_alder_endo.py` (slow)
- `docs/superpowers/specs/2026-05-05-phase-8-cycloaddition-design.md` (本文書)
- `docs/superpowers/plans/2026-05-05-phase-8-cycloaddition.md` (writing-plans phase で生成)

変更:
- `reactx/config.py`:
  - `RestraintConfig.k_form` / `k_broken` / `r_broken` を `float | tuple[float, ...]` に
  - `_normalize_k_form` / `_normalize_k_broken` / `_normalize_r_broken` 追加
  - `resolve_k_form_targets` / `resolve_k_broken_targets` / `resolve_r_broken_targets` 追加
  - 共通ヘルパ `_resolve_per_bond` 追加
- `reactx/placement.py`:
  - `_find_bridging_formed` を `bridges == 2` 受け入れ、`>= 3` で NotImplemented 化
  - `_multi_anchor_placement(...)` 関数追加 (本仕様 §5.2-5.5)
  - `valid_placements` 冒頭で `bridges` 数に応じ単一 / multi に分岐
  - `PlacementTrial` に `orientation` フィールド追加
  - `PlacementResult` に `placement_kind` フィールド追加
  - `_rotate_atoms(positions, indices, axis, center, angle)` ヘルパ追加 (Rodrigues' formula)
  - 新定数 `DUAL_ANCHOR_ASYMMETRY_THRESHOLD = 0.40`
- `reactx/artificial_force.py`:
  - `build_restraints` シグネチャを `*: float | list[float]` に変更、内部で broadcast
  - 古いコメント (Phase 4+) を Phase 8 達成内容に書き換え
- `reactx/cli.py`:
  - `config.resolve_*_targets` を呼んで per-bond list を構築
  - `meta.json` に `placement_kind` / `orientation` を出力
- `tests/test_placement.py`、`tests/test_config.py`、`tests/test_artificial_force.py`: 上記新規ケース追加
- `README.md`:
  - 「対応反応」に Diels–Alder ([4+2] cycloaddition、`bridges == 2`、2 formed + 0 broken) を追加
  - 「Per-reaction `.rxn.toml` config」表に DA 2 行を追加
  - 「動作確認」に手順 8, 9 追加 (DA 2 種)
  - 「Wall-clock (実測)」表に DA 2 行を実装後実測で追記
  - 「方針と限界」の最後に「multi-anchor placement は `bridges == 2` までを対応、`bridges >= 3` は Phase 9+」を明記
  - 「アーキテクチャ」図に multi-anchor branch を反映

## 11. リスクと開放課題

- **Asymmetry threshold 40% のチューニング**: 厳しすぎると DA 系で trial が枯れ、緩すぎると non-cycloaddition 配置が混入し UMA で発散しやすくなる。butadiene + ethylene の動作確認時に「生存数 ≥ 5、UMA 全 trial 収束」を満たす値で実測フィット。コード定数のため後 phase で TOML 化可
- **endo/exo の 0°/180° 二値離散化**: 連続 rotation を sample していないため、90° 付近の orientation で UMA が見つける local minimum を捕まえない。教育目的 (kinetic product 視認) には十分だが、stereoselectivity の研究目的には不足。後 phase で `[cycloaddition] n_orientation_samples = 4` のような連続 sample 化を検討
- **対称性 RMSD 0.01 Å 閾値**: ethylene のような厳密対称分子では 0.01 Å で問題ないが、近似対称な分子 (例: monosubstituted ethylene) で誤縮約する可能性あり。実装後の動作確認で reactant 集合に対し RMSD 検査の log を残し、閾値を再チューン
- **`k_form = [1.5, 1.5]` (DA + MA) のチューニング**: gas-phase で DA は repulsive PES の領域を通る場合があり、k_form が小さすぎると NaN / non-convergence、大きすぎると UMA の force が頭打ちで relax 失敗。Phase 7 Menshutkin の `k_form=4.0` チューニングと同じ手順で実測フィット
- **MA の電子吸引基取り扱い**: maleic anhydride の C=O は π-acceptor で、UMA の charge / 軌道が partial になりうる。`Atoms.info["spin"] = 1` のままで closed-shell singlet として扱うが、UMA 実測で issue が出れば spin 関連は別 phase で再検討
- **NEB refine の DA 対応**: 本 phase ではしない。`--neb-refine` は引き続き `len(formed)==1 and len(broken)==1` のみ accept、DA (`len(formed)==2`) で NEB を要求すれば `exit 2` でリジェクト

## 12. 実装順 (writing-plans へのインプット)

1. `config.py`: `RestraintConfig` 拡張 + `_normalize_*` + `resolve_*_targets` + テスト
2. `artificial_force.py`: `build_restraints` per-bond 化 + テスト + cli 呼び出し更新
3. `placement.py`: `PlacementTrial.orientation` / `PlacementResult.placement_kind` 追加 + 既存 single-anchor path を `placement_kind="single_anchor"` で出力するよう更新 (no-op だが API 統一)
4. `placement.py`: `_rotate_atoms` ヘルパ + `_multi_anchor_placement` 実装 + dispatch 分岐 + unit test
5. `examples/diels_alder_simple.{rxn,rxn.toml}` 追加 + slow integration test 通過
6. `examples/diels_alder_endo.{rxn,rxn.toml}` 追加 + slow integration test 通過 + endo / exo orientation 分離検証
7. `cli.py`: `meta.json` に `placement_kind` / `orientation` 出力
8. README 更新 (対応反応、TOML config 表、動作確認、Wall-clock、方針と限界、アーキテクチャ図)

各ステップ TDD: テスト → 実装 → green → 次へ。詳細な plan は writing-plans skill で本仕様を入力に生成する。
