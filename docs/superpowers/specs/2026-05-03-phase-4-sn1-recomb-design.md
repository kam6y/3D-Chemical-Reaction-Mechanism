# reactx Phase 4 — SN1 Step 2 (Tier 2 Centroid Placement) Design

- Status: Draft (awaiting user spec review)
- Date: 2026-05-03
- Owner: @kam6y
- Branch: `phase-4` (from `phase-3` / `develop` once merged)
- 前提仕様: `docs/superpowers/specs/2026-05-03-phase-3-multibond-design.md`
- Phase 3 で `embed3d._place_fragments` に `NotImplementedError` の hook として残された Tier 2 (broken=0 / multi-substrate) のうち、**SN1 step 2 (cation + nucleophile recombination)** だけを実装する

## 1. 目的

Phase 3 では「broken bond の方向情報がある反応 (Tier 1, directional placement)」のみ実装し、broken=0 の bimolecular は `NotImplementedError("centroid-based placement is Phase 4+")` を投げる契約だった。Phase 4 ではこの hook を `formed≥1 / broken=0 / 2 fragments` に限定して埋める:

- **対象反応 (MVP)**:
    - **SN1 step 2** — formed=1 + broken=0 (例: (CH₃)₃C⁺ + Cl⁻ → (CH₃)₃CCl)
- **対象外** (= 引き続き `NotImplementedError`):
    - 中性 addition (HCl + carbene 等) — 電荷ヒントが効かず配置根拠が薄い
    - cycloaddition / Diels–Alder — σ-only diff の前提を壊すため Phase 5+
    - multi-substrate metathesis (broken bonds が複数 fragment に跨る) — Phase 4 別 work item
    - formed≥2 + broken=0 の multi-bond addition — warning ログのみ、実行は続行するが geometric 妥当性は保証しない

「正確な TS エネルギーではなく妥当なアニメーション」という Phase Re1 / Phase 3 の方針は維持する。

## 2. Non-goals

明示的に **このフェーズではやらない** こと:

- **`--neb-refine` の対象拡大**: Phase 3 で導入した「`len(formed)==1 and len(broken)==1` のみ許可」guard はそのまま。SN1 step 2 (formed=1, broken=0) でも `--neb-refine` は exit 2 で reject される。理由: SN1 step 2 では product が unimolecular なので NEB endpoint 構築自体は別ロジックで成立可能だが、Tier 1 の swap 前提 (`bond_changes_product` 構築) と整合しないため、別途 Phase 4+ の work item とする。
- **結合次数追跡**: Phase 3 §3 の σ-only connectivity diff 方針を維持。SN1 step 2 では結合次数変化なしなので影響なし。
- **multi-substrate metathesis**: dispatcher の最終 `NotImplementedError` 分岐に「multi-substrate」を明示的に置く。実装は Phase 4 別 spec。
- **formed≥2 broken=0 の網羅**: 単一 anchor に複数 nucleophile が結合する反応 (multi-base attack) は Tier 1 と同じく `NotImplementedError`。1 anchor あたり 1 fragment という制約は Tier 2 にも引き継ぐ。
- **新規反応クラス preset の網羅**: Phase 4 で追加する preset は `sn1_recomb` の 1 種のみ。

## 3. 設計方針: plane-normal at anchor

Tier 1 の `_directional_placement` は「anchor → leaving への反対ベクトル」を nucleophile placement 方向にしていた。Tier 2 では leaving が無いため、別の方向決定基準が要る。3 案検討した:

| # | 案 | 長所 | 短所 |
|---|---|---|---|
| A | anti-centroid: `unit(anchor - substrate_centroid)` | 1 行で実装 | tBu⁺ 等の対称 cation で `centroid ≈ anchor`、ベクトルがほぼゼロで ill-defined。フォールバック必須。 |
| **B** | **plane-normal: anchor の重原子隣接から SVD 平面 fit、その法線** | **sp² の planar cation で「空の p 軌道方向」と物理的に一致。tBu⁺ / secondary / allyl / benzyl 全てで安定。** | **隣接 <3 や残差大時のフォールバックが必要、SVD 1 行 + case 分岐。** |
| C | pure cone over 180°: baseline 方向なし、球面 uniform sampling | sp² / sp³ 対称性に依存しない | 8 試行のうち substrate に正面衝突するケースが必ず混入、reached_product 達成率が低下する懸念。cone-half-deg と整合しない。 |

**B** を採用。Phase 3 の SN2 / E2 で確立した「multi-angle cone は baseline 方向の周りに 30° 程度散らす」モデルをそのまま流用できる。フォールバック階段は §5.2 で詳述。

## 4. パイプライン全体像

```
.rxn ─> rxn_parser ─> compute_bond_changes
                              │
                    BondChanges(formed=((1,5),), broken=())
                              ▼
                       embed3d.embed_mol_to_atoms
                              │
                              ▼ (multi-fragment 時)
              _place_fragments (dispatcher)
                ├ Tier 1: _directional_placement      ← Phase 3 (変更なし)
                │   条件: substrate is not None and broken
                ├ Tier 2: _planar_face_placement      ← Phase 4 で実装
                │   条件: not broken and formed
                └ NotImplementedError: multi-substrate / Phase 4+
                              │
                              ▼
              trials (n_angles=8, cone-half-deg=30°)
                              │
                              ▼
              build_restraints (formed=[(1,5)], broken=[])
                              │
                              ▼
              path_relax (UMA + FIRE) → scoring.reached_product
                              │
                              ▼
              prescreen (top-K=3 で動作可) / NEB refine (1+1 のみ; SN1 step 2 は reject)
```

## 5. モジュール変更

### 5.1 `reactx/embed3d.py`

`_place_fragments` の dispatcher に Tier 2 分岐を追加。Tier 1 (`_directional_placement`) と `_find_substrate_fragment` は変更しない。

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

    Tier 1 (directional, Phase 3): broken bond の方向情報がある反応 (E2 / SN2 / PT)。
    Tier 2 (planar face, Phase 4): broken=() かつ formed>=1 の bimolecular (SN1 step 2)。
    Phase 4+ (未実装):              multi-substrate metathesis (broken が複数 frag に跨る)。
    """
    substrate = _find_substrate_fragment(frag_indices, bond_changes.broken)
    if substrate is not None and bond_changes.broken:
        return _directional_placement(
            frag_indices, positions, bond_changes, substrate,
            rotation_perturbation=rotation_perturbation,
        )

    if not bond_changes.broken and bond_changes.formed:
        substrate = _find_substrate_by_size(frag_indices)
        return _planar_face_placement(
            mol_h, frag_indices, positions, bond_changes, substrate,
            rotation_perturbation=rotation_perturbation,
        )

    raise NotImplementedError(
        "multi-substrate metathesis (broken bonds spanning fragments) is "
        f"Phase 4+. Got formed={bond_changes.formed}, broken={bond_changes.broken}, "
        f"frags={len(frag_indices)}."
    )
```

新規ヘルパ:

```python
def _find_substrate_by_size(
    frag_indices: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    """Tier 2: 重原子数最大の fragment を substrate にする。

    tie の場合は最小 atom index を含む方を選ぶ (deterministic)。
    SN1 step 2 では cation = (CH₃)₃C⁺ (4 heavy) が Cl⁻ (1 heavy) より明確に大きい。
    """


def _planar_face_placement(
    mol_h: Chem.Mol,
    frag_indices: tuple[tuple[int, ...], ...],
    positions: np.ndarray,
    bond_changes: BondChanges,
    substrate: tuple[int, ...],
    *,
    rotation_perturbation: np.ndarray | None,
) -> np.ndarray:
    """Tier 2 placement: anchor の sp²-like 平面の法線方向に nucleophile を置く。

    For each non-substrate fragment F:
      bridging = formed bond で substrate↔F を跨ぐもの (空なら ValueError)。
      複数あれば canonical-ordered (= ((a, b) where a <= b) で sort して最小) の
      1 本を deterministic に採用。Phase 4 では SN1 step 2 で 1 本しか想定しないが、
      将来 multi-bond addition 等で複数になっても再現性を保つため。
      anchor   = bridging の substrate 側端。
      direction = _plane_normal_at_anchor(positions, anchor, mol_h, substrate)
      F の incoming 原子を anchor + R @ direction * FRAGMENT_SEPARATION に置く。

    Tier 1 と同じく、複数の non-substrate fragment が同じ anchor を共有するケースは
    `NotImplementedError("multi-base attack on single anchor not supported")`。
    """


def _plane_normal_at_anchor(
    positions: np.ndarray,
    anchor: int,
    mol_h: Chem.Mol,
    substrate: tuple[int, ...],
) -> np.ndarray:
    """Return a unit vector pointing away from the anchor's substrate plane.

    Strategy (priority order):
      1. anchor の substrate 内重原子隣接 (mol_h の bond から導出) を集める。
      2. **隣接 ≥3 かつ平面 fit 残差 < PLANE_FIT_TOLERANCE**:
            SVD で平面 fit (anchor 中心)、最小特異値の右特異ベクトル = 法線。
            符号 disambiguation: direction[2] < 0 なら反転 (常に +z 寄り)。
      3. **隣接 = 1 or 2、または平面 fit 残差が大きい**:
            direction = -unit(mean_neighbor - anchor)。
            norm < 1e-6 なら次へ。
      4. **degenerate**: direction = [0, 0, 1] + warning ログ。
    """
```

定数:
- `FRAGMENT_SEPARATION = 3.5` (既存、共有)
- `PLANE_FIT_TOLERANCE = 0.3` (Å) — SVD 残差 (最小特異値) 閾値。tBu⁺ の sp² 平面では残差 ≈ 0、sp³ への退化を検出するため。

### 5.2 `reactx/presets.py`

```python
PRESETS["sn1_recomb"] = ReactionPreset(
    name="sn1_recomb",
    k_form=1.0,
    k_broken=0.0,    # broken=() のため使われない、明示的に 0
    r_broken=4.0,    # 同上、placeholder (保存される effective_params の見栄え用)
    max_relax_steps=200,
    r_form=None,     # 元素表 (Cordero: C-Cl ≈ 1.78 Å)
)
```

`k_broken=0.0` は `broken=()` の SN1 step 2 で実害なし (`build_restraints` の broken loop が空回りする)。明示的に 0 にすることで「使われない」意図を表す。

### 5.3 `reactx/cli.py`

変更ほぼなし:

- `--reaction-type` の choices は `_PRESETS` 走査で自動的に `sn1_recomb` を含む (既存のロジック)。
- NEB refine guard (`len(bond_changes.formed) != 1 or len(bond_changes.broken) != 1`) はそのまま。SN1 step 2 (formed=1, broken=0) は exit 2 で reject される (= 既存の Phase 3 メッセージで「Phase 3」と「1 formed + 1 broken」を案内)。
- unimolecular auto-clamp はそのまま。SN1 step 2 は 2 fragments なので clamp 対象外、n_angles=8 が走る。
- `_resolve_effective_params` の `r_form_targets` 構築は formed_pairs 長で決まるので、formed=1 でそのまま動く。
- `bond_changes_product` 構築 (NEB refine path) は `len==1 and len==1` ガード後に通るので、SN1 step 2 では到達しない (= guard で先に return する)。

### 5.4 `reactx/bond_changes.py` / `artificial_force.py` / `prescreen.py` / `scoring.py` / `path_relax.py` / `align.py` / `neb.py` / `rxn_parser.py`

変更なし。`BondChanges(formed=((1,5),), broken=())` は既に `__post_init__` で許容される (formed+broken≥1 の条件)。`prescreen.py` の `r_broken_target` は `broken=[]` で空 loop になるため実害なし。

## 6. 例 .rxn / 統合テスト

### 6.1 `examples/sn1_recomb.rxn`

(CH₃)₃C⁺ + Cl⁻ → (CH₃)₃CCl。原子マッピング:
- 反応原子: 中心 C (atom map 1), 3 methyl C (atom map 2/3/4), Cl (atom map 5)
- formed: C(1) – Cl(5)
- broken: なし

reactant 側は `Phase 3 sn1_dissoc.rxn` の構造 (tBu⁺ + Br⁻) を Br→Cl に差し替えて反応方向を逆向きに使う形で構成。product 側は (CH₃)₃CCl 単一 fragment + neutral total charge。

### 6.2 統合テスト

| ファイル | 内容 |
|---|---|
| `tests/test_re4_sn1_recomb.py` (新規, slow) | `examples/sn1_recomb.rxn` + `--reaction-type sn1_recomb` で E2E 実行。assert: <br> - `meta.json.selected_trial >= 0` <br> - `trials[].reached_product` ≥1 件 True <br> - `effective_params.r_form_targets` が 1 要素 list で値 ≈ 1.78 ± 0.05 Å (Cordero) <br> - `bond_changes` は formed=((c, cl),), broken=() <br> - 最終フレームで C–Cl 距離 ≤ 1.95 Å (Cordero × 1.1) <br> - `meta.json.trials` の length が 8 (n_angles 自動 clamp が発動しない bimolecular 反応) |

## 7. ユニットテスト

| ファイル | 内容 |
|---|---|
| `tests/test_embed3d_placement.py` (改修) | 既存 Tier 1 ケースに加えて: <br> (h) **Tier 2 SN1 step 2 形状** (formed=1, broken=0, 2 frags, 新規 fixture: tBu⁺ + Cl⁻ を `Chem.MolFromSmiles("[C+](C)(C)C.[Cl-]")` で構築) で Cl の最終位置が `anchor + plane_normal * 3.5` から ±0.5 Å 以内 <br> (i) `_find_substrate_by_size`: 同じ heavy 数の場合に最小 atom index を含む側を選ぶ <br> (j) `_plane_normal_at_anchor`: 3 methyl の合成平面で +z 寄りの normal を返す <br> (k) `_plane_normal_at_anchor`: 隣接 1 個で `-unit(neighbor - anchor)` <br> (l) `_plane_normal_at_anchor`: 隣接 0 個で `[0,0,1]` + warning ログ <br> (m) `_planar_face_placement`: rotation_perturbation を identity 以外で渡すと target が動く <br> (n) **multi-substrate metathesis** (broken≠0 だが substrate=None) で `NotImplementedError("multi-substrate")` <br> (o) Tier 2 で複数 non-substrate fragment が同じ anchor を共有 (multi-base attack) で `NotImplementedError` |
| `tests/test_presets.py` (改修) | `sn1_recomb` preset の値を assert (k_form=1.0, k_broken=0.0, r_broken=4.0, max_relax_steps=200, r_form=None) |
| `tests/test_cli_neb_refine_guard.py` (改修) | parametrize に SN1 step 2 ケース (formed=1, broken=0) を追加、`--neb-refine` で exit 2 を確認 |

回帰テスト: 既存の `test_re1_sn2.py`, `test_re1_pt.py`, `test_re1_menshutkin.py`, `test_re3_e2.py`, `test_re3_sn1_dissoc.py` は **変更しない**。Tier 1 dispatch 経路の数値挙動が不変であることを既存 assert で保証する。

## 8. DoD 確認手順

1. `pytest -m "not slow and not blender"` 全 pass
2. `reactx run examples/sn1_recomb.rxn -o out/sn1r/ --reaction-type sn1_recomb --backend uma --render` を実行 → `meta.json.selected_trial >= 0`, `trials[].reached_product` ≥1, `out/sn1r/scene.blend` を Blender GUI で開いて Cl⁻ が tBu⁺ の平面に向かって接近 → C–Cl 結合形成を視認
3. `reactx run examples/sn1_recomb.rxn -o out/sn1r_neb/ --reaction-type sn1_recomb --neb-refine` を実行 → exit 2 + 「Phase 3」「1 formed + 1 broken」を含むエラーメッセージ
4. `pytest -m slow` で `test_re4_sn1_recomb` + 既存 `test_re1_*` / `test_re3_*` が全 pass
5. README に `sn1_recomb` preset 行を追加、Phase 4 セクションを `Phase Re1 + Phase 3 + Phase 4` に拡張

## 9. 適用限界 (このフェーズでも解消されない)

| # | ケース | Phase 4 の扱い |
|---|---|---|
| 1 | Metathesis 系 (broken が複数 substrate にまたがる) | `NotImplementedError("multi-substrate")` (Phase 4 別 spec) |
| 2 | 中性 addition (HCl + carbene 等) | `NotImplementedError` (Phase 5+; 電荷ヒントが効かない) |
| 3 | π-only 反応 (Diels-Alder 等) | `bond_changes` が変化を検出しない (σ-only diff、Phase 5+) |
| 4 | Tier 2 で複数 non-substrate fragment が同じ anchor 共有 | `NotImplementedError("multi-base attack on single anchor")` |
| 5 | Tier 2 で 3 fragments 以上 | warning ログ + 実行続行 (geometric 妥当性は限定的) |
| 6 | SN1 step 2 の `--neb-refine` | exit 2 (Phase 4+ で別途 endpoint 構築拡張) |
| 7 | 非対称 cation (allyl, benzyl 等) で両 face が非等価 | 片 face のみ sampling、cone-half-deg=30° で散らす。chirality は要再検討 (Phase 4 では racemate 等価扱い) |

## 10. リスク

- **plane-normal の sign 選択 (+z 寄り deterministic)**: tBu⁺ のような対称 cation では両 face 等価で問題ないが、allyl⁺ のような π 系では face によって product の立体が変わる可能性がある。Phase 4 では SN1 step 2 (sp² 単純 cation) のみを保証範囲とし、allyl⁺ 等は warning ログのみ + 実行続行とする。
- **平面 fit 残差閾値 (`PLANE_FIT_TOLERANCE = 0.3 Å`)**: ETKDG + MMFF が cation を完全に planar にしない場合、残差が閾値を超えてフォールバック (`-unit(mean_neighbor - anchor)`) に流れる可能性がある。tBu⁺ では SVD 残差は理論上 0、フォールバックが発火する場合は MMFF parameterize 失敗を意味する。DoD 段階で `_plane_normal_at_anchor` 単体テストで実測 fixture から残差を確認する。
- **Coulomb 引力 + Hookean の合算**: 反応 anion (Cl⁻) + cation (tBu⁺) は UMA で既に強い Coulomb 引力。`k_form=1.0` の Hookean を被せると過剰結合 (r < r_form) になる懸念。relax_with_restraints の動的解析で `r(C, Cl) -> r_form` への収束を DoD 段階で確認、必要なら preset の `k_form=0.5` への調整を検討する。
- **n_angles=8 の sampling 無駄**: 平面の片 face しか覆わないため、回転 8 通りでも substrate 体に正面衝突するケースは発生しない。逆に「同じ face を 8 通り cone でなぞる」のが過剰なら n_angles=4 で済む可能性もあるが、Phase Re1 / Phase 3 と同じ default を維持して preset 経由で個別に絞ることはしない。

## 11. 実装順序の推奨

1. `reactx/embed3d.py` に `_find_substrate_by_size`, `_plane_normal_at_anchor`, `_planar_face_placement` を追加 + Tier 2 dispatcher 分岐 (既存 Tier 1 の挙動が不変であることを既存 SN2/E2 ユニットテストで確認)
2. `tests/test_embed3d_placement.py` の (h)-(o) を実装
3. `reactx/presets.py` に `sn1_recomb` 追加 + `tests/test_presets.py` 改修
4. `examples/sn1_recomb.rxn` を作成 (Phase 3 `sn1_dissoc.rxn` から派生)
5. `tests/test_re4_sn1_recomb.py` (slow) 実装 + DoD 手順 2 で実機実行検証
6. `tests/test_cli_neb_refine_guard.py` parametrize に SN1 step 2 ケース追加
7. README 更新 (preset 表に `sn1_recomb` 行追加、Phase 4 セクションに DoD 手順を追記、wall-clock 表に SN1 recomb 行を追加)
