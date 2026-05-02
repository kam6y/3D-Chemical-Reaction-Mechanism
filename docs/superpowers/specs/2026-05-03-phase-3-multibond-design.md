# reactx Phase 3 — Multi-Bond Elementary Step Design

- Status: Draft (awaiting user spec review)
- Date: 2026-05-03
- Owner: @kam6y
- Branch: `phase-3` (from `phase-2`)
- 前提仕様: `docs/superpowers/specs/2026-04-27-reactx-phase-Re1-design.md`
- Phase Re1 で意図的に scope outside にされた「形成 1 + 切断 1 以外の elementary step」を扱えるようパイプラインを一般化する

## 1. 目的

Phase Re1 は SN2 / proton transfer / Menshutkin といった **形成 1 本 + 切断 1 本 で 1 原子を共有する** elementary step に scope を限定し、`compute_simple_bond_changes` がそれ以外で `NotImplementedError` を投げる契約だった。Phase 3 ではこの契約を以下の方向に拡張する:

- **対象反応 (MVP)**:
    - **E2 elimination** — 形成 1 + 切断 2 (例: CH₃CH₂Cl + OH⁻ → CH₂=CH₂ + Cl⁻ + H₂O)
    - **SN1 step 1 (heterolytic dissociation)** — 形成 0 + 切断 1 (例: (CH₃)₃CBr → (CH₃)₃C⁺ + Br⁻)
- **対象外** (= 引き続き `NotImplementedError`):
    - 形成 0 + bimolecular (SN1 step 2, cycloaddition) — 配置方向を決める基準が無い
    - broken が複数 substrate にまたがる metathesis — substrate fragment が一意に決まらない
    - π-only 反応 (Diels-Alder 等) — 結合次数変化を追跡しない方針 (§3 参照)

「正確な TS エネルギーではなく妥当なアニメーション」という Phase Re1 の方針は維持する。

## 2. Non-goals

明示的に **このフェーズではやらない** こと:

- **結合次数変化の追跡** (single↔double↔triple): `bond_changes` の比較は connectivity のみ。E2 の C=C π 形成は σ 距離拘束だけで動かし、π 結合の出現は UMA に任せる。
- **centroid-to-centroid placement (Tier 2)**: broken=0 や multi-substrate metathesis 等を扱うための「方向情報を捨てた汎用配置」は将来の Phase 4 行き。本フェーズでは dispatcher に hook を残すだけで実装はしない。
- **多結合反応の NEB refinement**: 既存の `bond_changes_product` swap ロジックは 1+1 にしか妥当でない。Phase 3 では `--neb-refine` 自体を「formed=1 かつ broken=1 の反応のみ」に CLI 側で制限する。multi-bond NEB endpoint construction は Phase 4+。
- **3 fragment 以上 (termolecular) の geometric 妥当性保証**: dispatcher は動くが、配置の物理的妥当性は限定的。warning ログのみ。
- **新規反応クラス preset の網羅** (Diels-Alder, radical, Michael 等): Phase 3 で追加する preset は `e2`, `sn1_dissoc` の 2 種のみ。

## 3. 設計方針: σ-only connectivity diff

`bond_changes._bond_set_in_self_idx` は現状 `(a, b)` の **隣接の有無** だけを集合化しており、bond order を見ていない。Phase 3 ではこれを変更しない。理由:

1. アニメーション目的では σ 距離拘束だけで物理が成立する (E2 の C–H 切断 + C–Cl 切断 + base 接近 → C–C 平面化は σ 力学だけで進む)
2. UMA (QM 力場) は π 結合形成を電子構造で正しく扱うため、artificial force で π を駆動する必要がない
3. 結合次数まで diff の対象にすると `shared_atom` 等の semantics がさらに複雑化し、Phase 3 のスコープを膨らませる

将来 Diels-Alder のように **σ 結合変化が 0 で π のみ変化する** 反応を扱う段では、`(a, b, order)` を要素とする diff に拡張し、artificial_force 側で π 変化を「無視」または「弱い距離拘束で近づける」かを別途設計する (Phase 5+)。

## 4. パイプライン全体像

```
.rxn ─> rxn_parser ─> compute_bond_changes (multi-bond 対応)
                              │
                    BondChanges(formed=[(a,b),...], broken=[(c,d),...])
                              ▼
                       embed3d.embed_mol_to_atoms
                              │
                              ▼ (multi-fragment 時)
              _place_fragments (dispatcher)
                ├ Tier 1: _directional_placement   ← Phase 3 で実装
                └ Tier 2: _centroid_placement      ← Phase 4+ stub (NotImplementedError)
                              │
                              ▼
              trials (n_angles)
                ├ unimolecular: 自動 1 にクランプ
                └ bimolecular: 既存通り
                              │
                              ▼
              artificial_force.build_restraints (既に lists 対応済)
                              │
                              ▼
              path_relax → scoring.reached_product (既に lists 対応済)
                              │
                              ▼
              prescreen (lists 対応済) / NEB refine (1+1 反応のみ許可)
```

## 5. モジュール変更

### 5.1 `reactx/bond_changes.py`

`SimpleBondChanges` を **削除**、`BondChanges` を新規追加 (rename + multi-bond 化)。`compute_simple_bond_changes` も同様に `compute_bond_changes` に rename。`SimpleBondChanges.shared_atom` プロパティは廃止 (利用者は唯一 `embed3d.py` で、§5.2 の `_directional_placement` 内で anchor / leaving を計算するロジックに置き換える)。

```python
@dataclass(frozen=True)
class BondChanges:
    """Multi-bond elementary step.

    Atom indices are in the reactant_mol_h (Chem.AddHs(reactant_mol)) coordinate system.
    formed/broken are tuples (not lists) for frozen dataclass hashability.
    """
    formed: tuple[tuple[int, int], ...]
    broken: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        for label, bonds in (("formed", self.formed), ("broken", self.broken)):
            seen: set[tuple[int, int]] = set()
            for a, b in bonds:
                if a == b:
                    raise ValueError(f"{label} bond {(a, b)} is a self-loop")
                key = (a, b) if a <= b else (b, a)
                if key in seen:
                    raise ValueError(f"{label} contains duplicate bond {key}")
                seen.add(key)
        if len(self.formed) + len(self.broken) == 0:
            raise ValueError("BondChanges must have at least one formed or broken bond")


def compute_bond_changes(
    reactant_mol_h: Chem.Mol,
    product_mol_h: Chem.Mol,
    heavy_mapping: dict[int, int],
) -> BondChanges:
    """Diff bonds between reactant and product mol_h (σ-only connectivity).

    Bond order changes (single↔double) are NOT detected; π formation is left
    to the QM calculator. See §3.
    """
```

`_build_full_atom_mapping` と `_bond_set_in_self_idx` の内部実装は変更しない (再利用)。

### 5.2 `reactx/embed3d.py`

`_place_nucleophile_backside` を **dispatcher + tier 関数群** に分解する。

```python
def _place_fragments(
    mol_h: Chem.Mol,
    frag_indices: tuple[tuple[int, ...], ...],
    positions: np.ndarray,
    bond_changes: BondChanges,
    *,
    rotation_perturbation: np.ndarray | None,
) -> np.ndarray:
    """Dispatch to the appropriate placement strategy.

    Tier 1 (directional, Phase 3): broken bond の方向情報がある反応 (E2 / SN2 / PT).
    Tier 2 (centroid, future):     broken=0 / multi-substrate metathesis.
    """
    substrate = _find_substrate_fragment(frag_indices, bond_changes.broken)
    if substrate is not None and bond_changes.broken:
        return _directional_placement(
            frag_indices, positions, bond_changes, substrate,
            rotation_perturbation=rotation_perturbation,
        )
    raise NotImplementedError(
        "Centroid-based placement (broken=0 / multi-substrate) is Phase 4+. "
        f"Got formed={bond_changes.formed}, broken={bond_changes.broken}, "
        f"frags={len(frag_indices)}."
    )


def _find_substrate_fragment(
    frag_indices: tuple[tuple[int, ...], ...],
    broken: tuple[tuple[int, int], ...],
) -> tuple[int, ...] | None:
    """Return the unique fragment containing both atoms of every broken bond.

    None if broken==() (Phase 4+ centroid placement) or if broken bonds span
    multiple fragments (multi-substrate metathesis, Phase 4+).
    """


def _directional_placement(
    frag_indices: tuple[tuple[int, ...], ...],
    positions: np.ndarray,
    bond_changes: BondChanges,
    substrate: tuple[int, ...],
    *,
    rotation_perturbation: np.ndarray | None,
) -> np.ndarray:
    """Tier 1 placement.

    For each non-substrate fragment F:
      bridging   = formed bond with one atom in substrate and one in F.
                   (ValueError if F has no fragment-bridging formed bond.)
      anchor     = the substrate end of bridging.
      relevant_b = broken bonds that contain anchor, sorted by smallest other-end index.
      leaving    = the other end of relevant_b[0] when non-empty;
                   else the substrate atom farthest from substrate centroid.
      backside   = -unit(anchor -> leaving)
      F の centroid を anchor + R @ backside * FRAGMENT_SEPARATION に移動.

    複数の non-substrate fragment が同じ anchor を共有する場合は
    NotImplementedError("multi-base attack on single anchor not supported")。
    3 fragments 以上は動作するが warning ログを出す。
    """
```

`embed_mol_to_atoms` の呼び出し側は `if len(frag_indices) > 1: positions = _place_fragments(...)` に変更するだけ。unimolecular (frag count=1) は dispatcher を通らない。

### 5.3 `reactx/presets.py`

```python
PRESETS["e2"] = ReactionPreset(
    name="e2",
    k_form=1.0, k_broken=1.0, r_broken=4.0,
    max_relax_steps=200,
    r_form=None,           # 元素ペア表 (典型: O-H=0.97 / N-H=1.01)
)
PRESETS["sn1_dissoc"] = ReactionPreset(
    name="sn1_dissoc",
    k_form=0.0, k_broken=2.0, r_broken=6.0,   # ion pair の十分な分離
    max_relax_steps=200,
    r_form=None,
)
```

`k_form=0.0` は formed=() の SN1 step 1 で実害なし (`build_restraints` の formed loop が空回りする)。明示的に 0 にすることで「使われない」意図を表す。

### 5.4 `reactx/cli.py`

主な変更点:

1. `formed_pair` / `broken_pair` (singular) を **削除**、`bond_changes.formed` / `bond_changes.broken` を直接使う。
2. `_resolve_effective_params` の戻り値を変更:
    - 旧: `r_form: float`
    - 新: `r_form_targets: list[float]` (formed bond 数分; `--r-form` scalar 指定時は全要素同値)
3. `reached_product(... r_form_targets=eff["r_form_targets"], ...)` 呼び出しはそのまま (API は既に list)。
4. log:
    - `r_form` 表示は formed=0 で `"-"`, formed=1 で値, formed≥2 で list 表記
5. `meta.json.effective_params.r_form` は scalar → list に **breaking change**。phase-3 ブランチで許容する。
6. `--n-angles` 自動クランプ:
    - 反応の reactant fragment 数が 1 (= unimolecular) のとき、`n_angles>1` でも 1 に強制し prescreen も skip。log に `unimolecular reaction; n_angles forced to 1, prescreen skipped` を出す。
7. `--neb-refine` ガード:
    - `len(formed) == 1 and len(broken) == 1` でない場合、`error: --neb-refine is only supported for 1 formed + 1 broken bond reactions in Phase 3 (multi-bond NEB endpoint construction is Phase 4+)` で argparse error (exit 2) にする。
8. `--reaction-type` choices に `e2`, `sn1_dissoc` を追加。
9. preset と反応形状の不整合検出:
    - `--reaction-type sn2_anion` で multi-bond .rxn を渡した場合は warning のみ (実行は続行)。

### 5.5 `reactx/scoring.py` / `reactx/artificial_force.py` / `reactx/prescreen.py`

API 変更なし。既に list ベース。`SimpleBondChanges` のインポートを `BondChanges` に rename するのみ (`prescreen.py` 等の docstring も合わせて修正)。

## 6. 例 .rxn / 統合テスト

### 6.1 `examples/e2.rxn`

CH₃CH₂Cl + OH⁻ → CH₂=CH₂ + Cl⁻ + H₂O。原子マッピング:
- 反応原子: 末端 C (substrate-C2), 中心 C (substrate-C1, C–Cl 持つ), Cl (leaving), 脱離 H (substrate-C2 上の H), O (base), H_O (base 上の H)
- formed: O–H (脱離 H と base O の新しい結合)
- broken: C2–H (脱離する H), C1–Cl (脱離する Cl)

### 6.2 `examples/sn1_dissoc.rxn`

(CH₃)₃CBr → (CH₃)₃C⁺ + Br⁻。原子マッピング:
- 反応原子: 中心 C, Br
- formed: なし
- broken: C–Br
- gas-phase での heterolytic dissociation は実体としては不安定 (UMA 上では energy が滑らかに上昇するだけだが、トラジェクトリ可視化目的としては成立)

### 6.3 統合テスト

| ファイル | 内容 |
|---|---|
| `tests/test_re3_e2.py` (新規, slow) | `examples/e2.rxn` + `--reaction-type e2` で E2E 実行。assert: `meta.json.selected_trial >= 0`, `trials[].reached_product` で少なくとも 1 件 True, `effective_params.r_form_targets` が 1 要素 list, `bond_changes` が 1 formed + 2 broken。 |
| `tests/test_re3_sn1_dissoc.py` (新規, slow) | `examples/sn1_dissoc.rxn` + `--reaction-type sn1_dissoc` で E2E 実行。assert: `meta.json.trials` が 1 件のみ (unimolecular auto-clamp), `selected_trial >= 0`, 最終フレームで C–Br 距離 ≥ 5.0 Å。 |
| `tests/test_blender_smoke.py` (改修) | parametrize に E2 ケースを追加。SN1 dissoc は ion pair の見栄え上 skip。 |

## 7. ユニットテスト

| ファイル | 内容 |
|---|---|
| `tests/test_bond_changes.py` (改修) | `SimpleBondChanges` → `BondChanges` rename。新規ケース: 1+2 (E2), 0+1 (SN1 step1), duplicate / self-loop で `ValueError`, formed=broken=() で `ValueError`, multi-substrate 形状でも `BondChanges` 自体は返る (scope 制限は embed3d 側)。 |
| `tests/test_embed3d_placement.py` (新規) | dispatcher の挙動: <br>(a) 1+1 (SN2 fixture) で directional に dispatch <br>(b) 1+2 (E2 形状) で directional, anchor=H, leaving=C <br>(c) broken=() で `NotImplementedError("centroid placement is Phase 4+")` <br>(d) multi-substrate metathesis で `NotImplementedError("multi-substrate")` <br>(e) bystander fragment で `ValueError` <br>(f) multi-base on single anchor で `NotImplementedError` <br>(g) unimolecular (1 frag) で dispatcher 呼ばれず |
| `tests/test_artificial_force.py` (既存) | 多 formed / 多 broken で restraints が線形に増えることを 1 ケース追加 (回帰用)。 |
| `tests/test_scoring.py` (既存) | `reached_product` に `r_form_targets=[1.05, 1.39]` (2 formed) を渡して per-bond 判定を検証。 |
| `tests/test_cli_unimolecular.py` (新規) | `--n-angles 8` を渡しても unimolecular .rxn (mock or sn1_dissoc) で `meta.json.trials` が 1 件だけになり、prescreen が skip されること。 |
| `tests/test_cli_neb_refine_guard.py` (新規) | E2 .rxn + `--neb-refine` で argparse error (exit 2)、メッセージに "Phase 3" / "1 formed + 1 broken" が含まれること。 |

回帰として、既存の `tests/test_re1_sn2.py`, `tests/test_re1_pt.py`, `tests/test_re1_menshutkin.py` は `BondChanges` rename + `r_form_targets` list 化の機械的追従のみ。**数値的な挙動 (selected_trial / wall_clock 範囲) は変化しない** ことを既存 assert で保証する。

## 8. DoD 確認手順

1. `pytest -m "not slow and not blender"` 全 pass
2. `reactx run examples/e2.rxn -o out/e2/ --reaction-type e2 --backend uma --render` を実行 → `meta.json.selected_trial >= 0`, `trials[].reached_product` ≥ 1, `out/e2/scene.blend` を Blender GUI で開いて C–H と C–Cl の同時切断 + base (OH⁻) 接近を視認
3. `reactx run examples/sn1_dissoc.rxn -o out/sn1d/ --reaction-type sn1_dissoc --backend uma` を実行 → `meta.json.trials` 1 件, `selected_trial >= 0`, `trajectory.xyz` で C–Br 距離が単調増加することを確認 (簡易 script で OK)
4. `pytest -m slow` で `test_re3_e2` + `test_re3_sn1_dissoc` + 既存 `test_re1_*` 全 pass

## 9. 適用限界 (今回も解消されない)

(A) 方向ベース placement の限界として、以下のケースは **Phase 4+** で対応:

| # | ケース | Phase 3 の扱い |
|---|---|---|
| 1 | Metathesis 系 (broken が複数 substrate にまたがる) | `NotImplementedError("multi-substrate")` |
| 2 | broken=0 の bimolecular (SN1 step2, addition) | `NotImplementedError("centroid placement is Phase 4+")` |
| 3 | 同じ anchor に複数 broken (retro-cycloaddition 等) | 決定論的に最小 index を選ぶ + docstring 明記 |
| 4 | 3 fragments 以上 (termolecular) | warning ログ + 実行続行 |
| 5 | 同じ anchor に複数 base (multi-base attack) | `NotImplementedError` |
| 6 | π-only 反応 (Diels-Alder 等) | bond_changes が変化を検出しない (σ-only diff) |

これらの拡張は Phase 4+ で:
- (1) → (B) centroid_placement 実装
- (2) → 同上 + 場合により charge-guided heuristic
- (6) → bond_changes に bond-order awareness 追加

## 10. リスク

- **E2 の reached_product 達成率**: 2 broken bond を同時に押し広げるため、`max_relax_steps=200` でも届かない可能性。preset チューニング (k_broken 増, max_relax_steps 増) が必要なら統合テスト pass を見て調整。
- **SN1 dissoc の物理的妥当性**: gas-phase での heterolytic dissociation は energy が滑らかに上昇するだけで TS 構造が出ない。アニメーション目的としては「C–Br が伸びる」ことだけ確認できれば十分とする。UMA の予測が ill-conditioned になる可能性は要監視。
- **meta.json schema breaking change**: `effective_params.r_form` (scalar) → `effective_params.r_form_targets` (list) は phase-3 ブランチでのみ許容する破壊的変更。merge 時に release note に明記。

## 11. 実装順序の推奨

1. **`bond_changes.py`** rename + multi-bond 化 + `__post_init__` ガード + ユニットテスト
2. 既存テスト (`test_re1_*`, `test_artificial_force.py`, `test_scoring.py`) を `BondChanges` rename に追従
3. **`embed3d.py`** dispatcher 切り出し (1+1 で挙動不変であることを SN2 ユニットテストで確認)
4. **`embed3d.py`** に `_directional_placement` の multi-bond 対応を追加 + `test_embed3d_placement.py`
5. **`presets.py`** に `e2`, `sn1_dissoc` 追加
6. **`cli.py`**: `formed_pair` → `formed_pairs`, `r_form` → `r_form_targets`, unimolecular auto-clamp, NEB guard
7. `examples/e2.rxn` + `tests/test_re3_e2.py`
8. `examples/sn1_dissoc.rxn` + `tests/test_re3_sn1_dissoc.py`
9. `tests/test_cli_unimolecular.py`, `tests/test_cli_neb_refine_guard.py`
10. README 更新 (DoD 手順, preset 表, 制限事項)
