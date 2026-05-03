# reactx Phase 5 — Multi-Substrate Metathesis (4-Center Tier 3) Design

- Status: Draft (awaiting user spec review)
- Date: 2026-05-03
- Owner: @kam6y
- Branch: `phase-5` (from `develop` after Phase 4 merge)
- 前提仕様: `docs/superpowers/specs/2026-05-03-phase-4-sn1-recomb-design.md`
- Phase 4 で `embed3d._place_fragments` の最終分岐に残された `NotImplementedError("multi-substrate metathesis (broken bonds spanning fragments) is Phase 5+")` を、**2 fragments / formed=2 / broken=2 / broken bonds が両 fragment を跨ぐ 4-center metathesis** に限定して実装する。

## 1. 目的

Phase 3/4 で確立した Tier 1 (directional) / Tier 2 (planar face) の placement 戦略は、いずれも「broken bonds が単一 substrate fragment 内に収まる」または「broken=()」を前提としていた。Phase 5 ではこの前提を破る最初のケースとして「**broken bonds が複数 fragment に分散する multi-substrate metathesis**」(各 broken bond は依然として単一 fragment 内で完結するが、全 broken bond をまとめて含む単一 fragment が存在しない) を扱う Tier 3 placement を新設する:

- **対象反応 (MVP)**:
    - **二重置換 4-center metathesis** — formed=2 / broken=2 / 2 fragments (例: CH₃Cl + LiBr → CH₃Br + LiCl)
        - 4-center TS: C–Cl 切断 + Li–Br 切断 + C–Br 形成 + Li–Cl 形成
        - 全原子 UMA `omol` 訓練範囲 (Z=3 Li, Z=6 C, Z=17 Cl, Z=35 Br)
        - σ-only diff で完結 (π 結合変化なし、Phase 3 §3 の方針継続)
- **対象外** (= 引き続き `NotImplementedError`):
    - 4 fragments 以上の ionic salt metathesis (例: NaCl + AgNO₃) — Phase 6+ 別 spec
    - π-only 反応 (Diels–Alder 等) — bond_changes が変化を検出しない (σ-only diff、Phase 6+)
    - σ-bond metathesis with M=H 系 (将来同 Tier で例追加可、本フェーズでは preset 1 種のみ)
    - 非対称 metathesis across fragments (formed=2 + broken=1, formed=1 + broken=2 等の formed_count != broken_count) — placement 根拠が薄いため Phase 6+

「正確な TS エネルギーではなく妥当なアニメーション」という Phase Re1 〜 Phase 4 の方針は維持する。

## 2. Non-goals

明示的に **このフェーズではやらない** こと:

- **`--neb-refine` の対象拡大**: Phase 3 で導入した「`len(formed)==1 and len(broken)==1` のみ許可」guard はそのまま。Phase 5 metathesis (formed=2, broken=2) は exit 2 で reject される。理由: NEB endpoint の `bond_changes_product` swap が multi-bond で破綻するため (Phase 3 spec §2 の判断を継続)。
- **結合次数追跡**: Phase 3 §3 の σ-only connectivity diff 方針を維持。4-center metathesis では結合次数変化なしなので影響なし。
- **3+ fragments の Tier 3 対応**: dispatcher で 3 fragments 以上 multi-substrate なら `NotImplementedError("Phase 6+")`。Tier 3 そのものは「2 fragments 専用」として実装する。
- **新規反応クラス preset の網羅**: Phase 5 で追加するのは `metathesis_4center` 1 種のみ。他の 4-center 反応 (σ-bond metathesis, Finkelstein gas-phase 等) は preset を拡充するだけで動作する想定だが、Phase 5 の DoD には含めない。
- **chirality / face 選択**: Kabsch は最小二乗解 1 つを返す。両 face 等価でない非対称基質に対しては cone perturbation の散らしでカバー (Phase 4 §10 のリスク継承)。

## 3. 設計方針: Kabsch alignment for multi-anchor placement (perpendicular face)

Tier 1 の `_directional_placement` は「anchor → leaving への反対ベクトル」で nucleophile 方向を決め、Tier 2 の `_planar_face_placement` は「anchor の sp²-like 平面の法線」で方向を決めていた。いずれも **1 anchor あたり 1 方向ベクトル** の問題だった。Tier 3 では 2 fragments それぞれに **複数の reaction anchor** があり、formed bonds が両 fragment を跨ぐため、単一ベクトルでは決まらない。

### 3.1 トポロジー前提

4-center metathesis では `_find_substrate_fragment` が None を返す (broken が両 fragment に跨がる) わけではなく、broken bonds が **各 fragment 内で完結する** が **formed bonds が両 fragment を跨がる** トポロジーを取る:

| | reactant CH₃Cl + LiBr の例 |
|---|---|
| broken_within_reference | `{(C, Cl)}` (CH₃Cl 内に閉じる) |
| broken_within_moving | `{(Li, Br)}` (LiBr 内に閉じる) |
| formed_across_fragments | `{(C, Br), (Li, Cl)}` (どちらも reference↔moving) |

`_find_substrate_fragment` の実装 (line 137-154) は「全 broken bond を含む単一 fragment があるか」をチェックしており、broken bonds が複数 fragment に分散すると None を返す。これが Tier 3 の dispatch 条件。Tier 3 ロジック内では更に `broken_within_reference` / `broken_within_moving` がそれぞれ 1 本ずつ、formed が 2 本両 fragment を跨ぐことを assert する。

### 3.2 配置アルゴリズム: anchor 軸 + 垂直 face 配置

4-center TS は anchor pair と incoming pair が概矩形を成す:

```
       reference frame (CH₃Cl, fixed)
   anchor_a ──── anchor_b           ← C ──── Cl  (broken_within_reference)
       :              :              ← formed bonds (forming)
   incoming_a ── incoming_b         ← Br ──── Li  (broken_within_moving)
       moving frame (LiBr, transformed by Kabsch)
```

target 点を anchor 軸の **垂直方向** に置くことで、moving fragment の incoming pair が anchor pair と並行になる剛体変換を Kabsch が見つける:

1. **reference fragment** = `_find_substrate_by_size(frag_indices)` (例: CH₃Cl, 重原子 2 + 3H)。固定。
2. **moving fragment** = もう一方 (例: LiBr)。Kabsch で剛体変換される。
3. **Anchor pair の特定**: `broken_within_reference` の 2 endpoint = `(anchor_a, anchor_b)` (例: C, Cl)。
   - 各 formed bond の reference 側端は anchor_a または anchor_b に一致するはず。違う場合は `ValueError`。
4. **Incoming pair の特定**: 各 formed bond の moving 側端 = `(incoming_a, incoming_b)`。formed_a の reference 端を anchor_a と定義することで対応付ける。
5. **Anchor 軸の垂直 face 方向 `perp_dir`**:
   - `axis = unit(positions[anchor_a] - positions[anchor_b])` (例: C - Cl の方向、unit vector)
   - `anchor_midpoint = (positions[anchor_a] + positions[anchor_b]) / 2`
   - `offset = positions[moving_atoms].mean(axis=0) - anchor_midpoint` (moving 重心の anchor 軸中点からの偏差ベクトル)
   - `offset_perp = offset - (offset · axis) × axis` (axis 方向成分を除去した垂直成分)
   - `perp_dir = offset_perp / ||offset_perp||` (norm 正規化)
   - **Fallback (offset_perp の norm < 1e-6, = moving 重心が axis 上)**: 世界座標基底 `[+z, +y, +x]` を順に走査し、最初に `||e - (e · axis) × e|| > 1e-6` となる `e` を選び正規化。axis が unit vector なので最低 2 つの世界基底が非ゼロ垂直成分を持つ → 必ず一意に decideable。
6. **Target 配置**:
   - `target_a = positions[anchor_a] + FRAGMENT_SEPARATION/2 × perp_dir`
   - `target_b = positions[anchor_b] + FRAGMENT_SEPARATION/2 × perp_dir`
   - 結果: target_a と target_b は anchor 軸と並行 (距離 = anchor_a-anchor_b 距離 ≈ 1.78 Å)、anchor 軸から `FRAGMENT_SEPARATION/2 = 1.75 Å` 持ち上がる。
7. **Kabsch alignment**:
   - `src = positions[(incoming_a, incoming_b)]`, `dst = (target_a, target_b)`
   - `R, t = _kabsch_rigid_transform(src, dst)` (centroid 中心化 → SVD → reflection 防止)
   - N=2 corresponding points は剛体変換 6-DOF に対し constraint を与える (centroid 3 + axis-pair 方向 2 = 5)。残り 1 DOF (anchor 軸周りの回転) は最小二乗解として centroid 連結直線が anchor 軸に並行になる解を Kabsch が選ぶ。
   - incoming-incoming 距離 (Li-Br ≈ 2.2 Å) ≠ anchor-anchor 距離 (C-Cl ≈ 1.78 Å) なので 2 点完全一致は不可、Kabsch は両端の RMSD 最小解を返す (centroid を一致させ axis を並行化)。
8. **Cone perturbation**: `R_perturbed = rotation_perturbation @ R_kabsch`、`positions[moving]` 全体に `R_perturbed` と `t` を centroid 経由で適用。

### 3.3 配置結果の幾何 (CH₃Cl + LiBr 想定)

C を `(1.78, 0, 0)`、Cl を `(0, 0, 0)` に置いたとき、moving 重心が初期 `(0.89, 2, 0)` (例) の場合:

- axis = `(1, 0, 0)`、offset = `(0, 2, 0)`、`offset · axis = 0`、`perp_dir = (0, 1, 0)`
- target_C = `(1.78, 1.75, 0)`、target_Cl = `(0, 1.75, 0)`
- LiBr (Br at `(1.1, 0, 0)`, Li at `(-1.1, 0, 0)` in moving frame) を Kabsch で 2 target に fit:
  - centroid src = `(0, 0, 0)`, centroid dst = `(0.89, 1.75, 0)`、t = `(0.89, 1.75, 0)`
  - R = identity (axis already aligned along x)
  - Br placed at `(1.99, 1.75, 0)`、Li placed at `(-0.21, 1.75, 0)`
- 4-center 距離:
  - C–Br = √(0.21² + 1.75²) ≈ 1.76 Å (formed bond, 期待値 ≈ 1.94 Å Cordero)
  - Li–Cl = √(0.21² + 1.75²) ≈ 1.76 Å (formed bond, 期待値 ≈ 2.02 Å Cordero)
  - C–Cl = 1.78 Å (broken bond, 期待値 ≈ 4.0 Å r_broken まで relax で伸びる)
  - Li–Br = 2.2 Å (broken bond, 期待値 ≈ 4.0 Å r_broken まで relax で伸びる)

これで 4-center TS の妥当な初期配置が得られる。Hookean (k_form=1.0, k_broken=1.0) + UMA + FIRE で TS を経由して product (CH₃Br + LiCl) に向かう。

## 4. パイプライン全体像

```
.rxn ─> rxn_parser ─> compute_bond_changes
                              │
                    BondChanges(formed=((C,Br),(Li,Cl)), broken=((C,Cl),(Li,Br)))
                              ▼
                       embed3d.embed_mol_to_atoms
                              │
                              ▼ (multi-fragment 時)
              _place_fragments (dispatcher)
                ├ Tier 1: _directional_placement   (substrate is not None and broken)
                ├ Tier 2: _planar_face_placement   (not broken and formed)
                └ Tier 3: _kabsch_alignment        ← Phase 5 で実装
                              条件: substrate is None and broken and formed
                                    and len(frag_indices) == 2
                                    and len(formed) == 2 and len(broken) == 2
                              │
                              ▼
              trials (n_angles=8, cone-half-deg=30°)
                              │
                              ▼
              build_restraints (formed=[(C,Br),(Li,Cl)], broken=[(C,Cl),(Li,Br)])
                              │
                              ▼
              path_relax (UMA + FIRE) → scoring.reached_product
                              │
                              ▼
              prescreen (top-K=3) / NEB refine (1+1 のみ; metathesis は reject)
```

## 5. モジュール変更

### 5.1 `reactx/embed3d.py`

`_place_fragments` の dispatcher に Tier 3 分岐を最終 raise の手前に挿入。Tier 1/Tier 2 の挙動は完全に不変。

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
    Tier 2 (planar face, Phase 4): broken=() かつ formed=1 の bimolecular (SN1 step 2)。
    Tier 3 (Kabsch, Phase 5):     broken bonds が両 fragment を跨ぐ 2-fragment 4-center metathesis。
    Phase 6+ (未実装):            cycloaddition (formed>=2, broken=0) /
                                   3+ fragment ionic salt metathesis /
                                   非対称 metathesis (formed_count != broken_count across fragments)。
    """
    substrate = _find_substrate_fragment(frag_indices, bond_changes.broken)
    if substrate is not None and bond_changes.broken:
        return _directional_placement(
            frag_indices, positions, bond_changes, substrate,
            rotation_perturbation=rotation_perturbation,
        )

    if not bond_changes.broken and bond_changes.formed:
        if len(bond_changes.formed) > 1:
            raise NotImplementedError(
                "cycloaddition (broken=0, formed>=2) is Phase 6+. "
                f"Got formed={bond_changes.formed}, frags={len(frag_indices)}."
            )
        substrate = _find_substrate_by_size(frag_indices)
        return _planar_face_placement(
            mol_h, frag_indices, positions, bond_changes, substrate,
            rotation_perturbation=rotation_perturbation,
        )

    # Tier 3: Phase 5 metathesis (broken bonds 各 fragment 内に閉じ、formed が両 frag を跨ぐ)
    if (
        substrate is None
        and bond_changes.broken
        and bond_changes.formed
        and len(frag_indices) == 2
        and len(bond_changes.formed) == 2
        and len(bond_changes.broken) == 2
    ):
        # Tier 3 内で broken_within_reference / broken_within_moving の本数を再検証
        # (1+1 でない場合は _kabsch_alignment が NotImplementedError を投げる)
        return _kabsch_alignment(
            mol_h, frag_indices, positions, bond_changes,
            rotation_perturbation=rotation_perturbation,
        )

    raise NotImplementedError(
        "multi-substrate placement only supports 2-fragment 4-center metathesis "
        "(formed=2, broken=2) in Phase 5; other shapes are Phase 6+. "
        f"Got formed={bond_changes.formed}, broken={bond_changes.broken}, "
        f"frags={len(frag_indices)}."
    )
```

新規ヘルパ:

```python
def _kabsch_alignment(
    mol_h: Chem.Mol,
    frag_indices: tuple[tuple[int, ...], ...],
    positions: np.ndarray,
    bond_changes: BondChanges,
    *,
    rotation_perturbation: np.ndarray | None,
) -> np.ndarray:
    """Tier 3 placement: 2 fragments の formed bonds ペアを 4-center 矩形に剛体整列。

    Algorithm (詳細は Spec §3 参照):
      1. reference = _find_substrate_by_size(frag_indices), moving = もう一方
      2. broken_within_reference / broken_within_moving を分類。各 1 本ずつでなければ
         NotImplementedError("Phase 5 only supports 1+1 within-fragment broken bonds").
      3. anchor pair = broken_within_reference の 2 endpoint。
         formed bonds 各本から (anchor_in_reference, incoming_in_moving) を抽出。
         全ての formed bond の reference 端が anchor pair に含まれていなければ ValueError。
         同 anchor が複数 formed bond を持つ (multi-bond from single anchor) なら
         NotImplementedError("multi-bond from single anchor in metathesis")。
      4. perp_dir = anchor 軸 (anchor_a - anchor_b) と垂直な face direction。
         moving 重心の anchor 軸からの offset を垂直平面に射影し正規化。
         degenerate なら +z, +y の順で defensive fallback。
      5. target_a/b = positions[anchor_a/b] + FRAGMENT_SEPARATION/2 × perp_dir
      6. _kabsch_rigid_transform(incoming_positions, target_positions) で (R, t) を求める。
      7. R_perturbed = rotation_perturbation @ R (cone 散らし)。
      8. moving fragment 全体に R_perturbed, t を centroid 経由で適用。

    Edge cases:
      - len(broken_within_reference) != 1 or len(broken_within_moving) != 1 →
        NotImplementedError (Phase 5 範囲外、Phase 6+ で metathesis variants 対応)。
      - 2 incoming atoms が同一 (= multi-bond from single anchor) →
        NotImplementedError("multi-bond from single anchor in metathesis")。
      - anchor 軸が degenerate (anchor_a と anchor_b が ETKDG 後に重なる, < 1e-6 Å) →
        RuntimeError("anchor pair coincident after MMFF — broken-within-reference
        bond is broken")。
    """


def _kabsch_rigid_transform(
    src: np.ndarray,      # (N, 3) moving fragment の対応点 (incoming positions)
    dst: np.ndarray,      # (N, 3) reference 側の target 点
) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch / orthogonal Procrustes solve: src を dst に合わせる剛体変換 (R, t)。

    1. centroid 中心化: src_c = src - mean(src), dst_c = dst - mean(dst)
    2. cross-covariance H = src_c.T @ dst_c
    3. SVD: U, S, Vt = svd(H)
    4. reflection 防止: d = sign(det(Vt.T @ U.T)), D = diag(1, 1, d)
    5. R = Vt.T @ D @ U.T
    6. t = mean(dst) - R @ mean(src)

    返り値: (R: (3,3), t: (3,))。
    src.shape != dst.shape または N < 2 で ValueError。
    """
```

定数追加:

- `FRAGMENT_SEPARATION = 3.5` (既存、共有)
- 新規定数なし (Tier 3 は `FRAGMENT_SEPARATION/2 = 1.75` を inline で使用)

### 5.2 `reactx/presets.py`

```python
PRESETS["metathesis_4center"] = ReactionPreset(
    name="metathesis_4center",
    k_form=1.0,
    k_broken=1.0,
    r_broken=4.0,
    max_relax_steps=200,
    r_form=None,     # 元素表 (Cordero: C–Br ≈ 1.94, Li–Cl ≈ 2.02)
)
```

選定根拠:
- `k_form=1.0`, `k_broken=1.0` — Phase 3 e2 と同じ。formed/broken それぞれ 2 本ずつ走るので合計 4 本の Hookean が同時に効く。これより強くすると 4-center が潰れて moving fragment が reference に貼り付く懸念。
- `r_broken=4.0 Å` — default。broken=2 本で同じ閾値を使う。
- `max_relax_steps=200` — multi-bond は収束遅め、E2/sn1_recomb と同等。
- `r_form=None` — 元素表 fallback。Cordero 2008 共有結合半径から C-Br=1.94, Li-Cl=2.02 を期待。

### 5.3 `reactx/cli.py`

変更なし (= Phase 4 と同じ理由):
- `--reaction-type` の choices は `_PRESETS` 走査で `metathesis_4center` を自動含む
- NEB refine guard (`len(formed) != 1 or len(broken) != 1`) はそのまま — metathesis は exit 2 で reject される
- unimolecular auto-clamp は 2 fragments なので発動しない、n_angles=8 が走る
- `_resolve_effective_params` の `r_form_targets` は formed_pairs 長 (= 2) で list 構築
- `bond_changes_product` 構築 (NEB refine path) は guard で先に return するので metathesis では到達しない

### 5.4 不変モジュール

`reactx/bond_changes.py` / `reactx/artificial_force.py` / `reactx/prescreen.py` / `reactx/scoring.py` / `reactx/path_relax.py` / `reactx/align.py` / `reactx/neb.py` / `reactx/rxn_parser.py` — いずれも multi-bond 対応済み (Phase 3 で完了)、変更なし。

`BondChanges(formed=((C,Br),(Li,Cl)), broken=((C,Cl),(Li,Br)))` は `__post_init__` の `formed+broken >= 1` 条件を満たすので構築可能。

## 6. 例 .rxn / 統合テスト

### 6.1 `examples/metathesis_4center.rxn`

CH₃Cl + LiBr → CH₃Br + LiCl。原子マッピング:

- 反応原子: C (atom map 1), Cl (atom map 2), Li (atom map 3), Br (atom map 4)
- formed: C(1) – Br(4), Li(3) – Cl(2)
- broken: C(1) – Cl(2), Li(3) – Br(4)

reactant 1 (CH₃Cl): atom map 1=C, 2=Cl, + 3 implicit H。
reactant 2 (LiBr): atom map 3=Li, 4=Br。
product 1 (CH₃Br): atom map 1=C, 4=Br, + 3 implicit H。
product 2 (LiCl): atom map 3=Li, 2=Cl。

両側で総電荷 = 0、spin = 1 (closed-shell singlet)。LiBr と LiCl は contact ion pair として共有結合扱い (M CHG なし)。

### 6.2 統合テスト

| ファイル | 内容 |
|---|---|
| `tests/test_re5_metathesis.py` (新規, slow) | `examples/metathesis_4center.rxn` + `--reaction-type metathesis_4center` で E2E 実行。assert: <br> - `meta.json.selected_trial >= 0` <br> - `trials[].reached_product` ≥1 件 True <br> - `effective_params.r_form_targets` が 2 要素 list、値 ≈ [1.94, 2.02] ± 0.05 (Cordero) <br> - `bond_changes` が formed=2, broken=2 <br> - 最終フレームで C–Br ≤ 2.13 Å かつ Li–Cl ≤ 2.22 Å (Cordero × 1.1) <br> - `meta.json.trials` length == 8 (n_angles auto-clamp 不発動の bimolecular) <br> - `meta.json.reaction_type == "metathesis_4center"` |

## 7. ユニットテスト

| ファイル | 内容 |
|---|---|
| `tests/test_embed3d_placement.py` (改修) | 既存 Tier 1/2 ケースに加えて: <br> (p) **Tier 3 dispatch**: 2-fragment formed=2/broken=2/multi-substrate で `_kabsch_alignment` に流れることを確認 (`test_place_fragments_dispatches_tier3_for_metathesis`) <br> (q) `_kabsch_alignment` 主路: CH₃Cl + LiBr fixture で 4-center geometry を達成 — C-Br ≤ 2.5 Å かつ Li-Cl ≤ 2.5 Å、anchor 軸 (C-Cl) と incoming 軸 (Br-Li) が概並行 (cos angle > 0.7)、moving 重心が anchor 軸から FRAGMENT_SEPARATION/2 ± 0.5 Å 離れている <br> (r) `_kabsch_alignment`: rotation_perturbation を identity 以外で渡すと moving fragment 全体が回転される (Br と Li の相対距離は不変、絶対位置は回転後) <br> (s) `_kabsch_rigid_transform` 単体: 既知の R, t を src に適用した dst で逆算した (R', t') が原 R, t と一致 (forward → inverse identity) <br> (t) `_kabsch_rigid_transform` reflection 防止: rotoinversion を作る対応点に対しても det(R) >= 0 の rotation を返す <br> (u) `_kabsch_alignment`: broken_within_reference または broken_within_moving が 1 本でない (e.g. broken bonds が両 fragment を跨ぐ純粋 metathesis 以外) で `NotImplementedError("Phase 5 only supports 1+1 within-fragment broken bonds")` <br> (v) `_kabsch_alignment`: 2 formed bond が moving 側で同一原子を共有 (multi-bond from single anchor) で `NotImplementedError("multi-bond from single anchor in metathesis")` <br> (w) `_place_fragments`: 3 fragments + multi-substrate (formed と broken が両方ある) で `NotImplementedError("Phase 6+")` <br> (x) `_place_fragments`: 非対称 metathesis (formed=2, broken=1, 両 fragment 跨ぎ) で `NotImplementedError("Phase 6+")` <br> (y) `_kabsch_alignment` perp_dir フォールバック: moving 重心が anchor 軸上 (offset 射影 norm < 1e-6) で +z fallback が選ばれる |
| `tests/test_presets.py` (改修) | `metathesis_4center` preset の値を assert (k_form=1.0, k_broken=1.0, r_broken=4.0, max_relax_steps=200, r_form=None) + `PRESETS` キー集合に `metathesis_4center` を追加 |
| `tests/test_cli_neb_refine_guard.py` (改修) | parametrize に metathesis ケース (formed=2, broken=2) を追加、`--neb-refine` で exit 2 を確認 |
| `tests/test_rxn_parser.py` (改修) | `examples/metathesis_4center.rxn` の parse smoke test: formed=2 / broken=2 / 2 reactant fragments / 2 product fragments / formed が C-Br と Li-Cl を含む |

回帰テスト: 既存の `test_re1_*.py`, `test_re3_*.py`, `test_re4_*.py` は **変更しない**。Tier 1/2 dispatch 経路の数値挙動が不変であることを既存 assert で保証する。

## 8. DoD 確認手順

1. `pytest -m "not slow and not blender"` 全 pass
2. `reactx run examples/metathesis_4center.rxn -o out/m4c/ --reaction-type metathesis_4center --backend uma --render` を実行
   → `meta.json.selected_trial >= 0`, `trials[].reached_product` ≥1, `out/m4c/scene.blend` を Blender GUI で開いて 4-center TS → CH₃Br + LiCl への遷移を視認 (CH₃ が Cl から Br に乗り換え、Li が Br から Cl に乗り換える同時動作)
3. `reactx run examples/metathesis_4center.rxn -o out/m4c_neb/ --reaction-type metathesis_4center --neb-refine` を実行
   → exit 2 + 「Phase 3」「1 formed + 1 broken」を含むエラーメッセージ
4. `pytest -m slow` で `test_re5_metathesis` + 既存 `test_re1_*` / `test_re3_*` / `test_re4_*` が全 pass
5. README に `metathesis_4center` preset 行を追加、Phase 5 セクションを `Phase Re1 + Phase 3 + Phase 4 + Phase 5` に拡張、wall-clock 表に metathesis 行を追加、対応反応一覧を更新

## 9. 適用限界 (このフェーズでも解消されない)

| # | ケース | Phase 5 の扱い |
|---|---|---|
| 1 | 4 fragments 以上の ionic salt metathesis (NaCl + AgNO₃ 等) | `NotImplementedError("Phase 6+")` (centroid alignment for N>2 fragments が必要) |
| 2 | π-only 反応 (Diels–Alder 等) | `bond_changes` が変化を検出しない (σ-only diff、Phase 6+) |
| 3 | 同一 anchor から 2 formed bond (multi-bond from single atom in metathesis) | `NotImplementedError("multi-bond from single anchor in metathesis")` |
| 4 | metathesis の `--neb-refine` | exit 2 (Phase 6+ で別途 endpoint 構築拡張) |
| 5 | 非対称 metathesis (formed=2 + broken=1, formed=1 + broken=2 across-fragment) | dispatcher で `NotImplementedError("Phase 6+")` (formed_count != broken_count = 想定外) |
| 6 | キラル metathesis (face 選択依存) | Kabsch は最小二乗解 1 つを返す。chirality は cone perturbation でカバー (= racemate 等価扱い) |
| 7 | 5+ center metathesis (formed=3+ / broken=3+) | dispatcher で `NotImplementedError` (Tier 3 は formed=2/broken=2 限定) |

## 10. リスク

- **UMA の Li 取り扱い**: Li (Z=3) は `omol` 訓練範囲内だが、ionic 系で過剰電荷局在を起こす可能性がある。DoD 段階で実機実行で `reached_product` 達成率を確認、必要なら preset の `k_form` を 0.5 に下げる調整を検討する。
- **MMFF94 prescreen の Li 失敗**: MMFF94 は Li を parameterize できない可能性が高い → prescreen が `mmff_failed=true` にフォールバックして全 trial を UMA に流す。proton_transfer (HCl) と同様のパターンで wall-clock は ~1.5x 程度延長見込み。
- **Kabsch の 1-DOF redundancy (anchor 軸周り)**: 2-corresponding-point Kabsch では anchor 軸 (= incoming 軸) 周りの回転が一意に決まらない。実装で `det(R) >= 0` の sign correction で reflection は防げるが、anchor 軸周りの位相は最小二乗解として centroid-aligned かつ axis-parallel な回転が選ばれる。残り 1 DOF (= 「どの face 側から接近するか」) は perp_dir の選択で決定論化、cone perturbation で散らせるので実害は限定的。
- **r_broken=4.0 と FRAGMENT_SEPARATION/2=1.75 の整合**: target は anchor から 垂直に 1.75 Å。broken bond (anchor pair = C-Cl と Li-Br) は ETKDG 由来 1.78-2.2 Å のまま reference frame では不変、moving frame でも Kabsch は剛体変換なので不変。Hookean (k_broken=1.0, r_broken=4.0) で relax 開始から伸ばし始め、2-3 step 以内に broken 距離が伸び始める。
- **incoming 軸 (Li-Br) と anchor 軸 (C-Cl) の長さミスマッチ**: 2.2 Å vs 1.78 Å。Kabsch は両端の RMSD 最小解を返すので 4-center は完全な矩形でなく僅かな台形になる (両端で incoming-anchor 距離が ~0.21 Å 余分にずれる、最終 C-Br ≈ Li-Cl ≈ 1.76 Å @ §3.3)。relax 開始点としては十分機能するが、test (q) では完全矩形ではなく「C-Br ≤ 2.5 Å, Li-Cl ≤ 2.5 Å, axis cosine > 0.7」程度の緩い assertion を使う。
- **perp_dir の選択 (anchor 軸周り回転 1-DOF)**: ETKDG が moving 重心を anchor pair midpoint に対しどの方向に置くかは反復毎に確率的。`perp_dir = unit(offset - (offset·axis)×axis)` で初期 ETKDG 配置に依存させ、cone perturbation で 8 trial に散らす。完全に anchor 軸上 (= 確率測度ゼロ) のケースは defensive +z fallback でカバー。
- **対称性の破れ (reference/moving 選択)**: CH₃Cl と LiBr のように 1 fragment が complex (CH₃Cl, 5 atoms) で 1 fragment が simple (LiBr, 2 atoms) の場合、`_find_substrate_by_size` で必ず CH₃Cl が reference になるので決定論的。両 fragment が同サイズ (例: 2 つの 4 原子 fragment 同士の metathesis) のような将来ケースでは tie-break が動くが、Phase 5 MVP では発生しない。

## 11. 実装順序の推奨

1. `reactx/embed3d.py` に `_kabsch_rigid_transform` を追加 + ユニットテスト (s)(t)
2. `reactx/embed3d.py` に `_kabsch_alignment` 主路を追加 + ユニットテスト (q)(r)
3. `_kabsch_alignment` の error 分岐 (broken_within_reference/_moving が 1+1 でない / multi-bond from single anchor / perp_dir fallback) + テスト (u)(v)(y)
4. `_place_fragments` dispatcher に Tier 3 分岐を挿入 + テスト (p)(w)(x)
5. `examples/metathesis_4center.rxn` 作成 + `tests/conftest.py` に fixture 追加 + parse smoke test
6. `reactx/presets.py` に `metathesis_4center` 追加 + `tests/test_presets.py` 改修
7. `tests/test_re5_metathesis.py` (slow) 実装 + DoD 手順 2 で実機実行検証
8. `tests/test_cli_neb_refine_guard.py` parametrize に metathesis ケース追加
9. README 更新 (preset 表に `metathesis_4center` 行、Phase 5 セクション追加、wall-clock 表更新、対応反応一覧拡張、Phase tag header 追記)
