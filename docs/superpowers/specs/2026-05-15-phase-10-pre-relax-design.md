# reactx Phase 10 — Pre-relax (unbiased FIRE) before AFIR

- Status: Draft
- Date: 2026-05-15
- Owner: @kam6y
- Branch: `phase-10` (from `phase-9` HEAD, target merge to `develop`)
- 前提仕様: `docs/superpowers/specs/2026-05-08-afir-force-design.md` (Phase 9 AFIR + sticky latch)
- 後方互換性: develop ブランチ運用方針 (`[[feedback_no_backwards_compat]]`) により API breaking 許容。`pre_relax_steps` の **デフォルトは 30 (機能オン)** とし、既存 7 反応 (PT/Menshutkin/E2/SN1 dissoc/SN1 recomb/DA simple/DA endo) の挙動が変わることを許容する。問題が出た反応は個別 `.rxn.toml` で値を下げる/0 にして調整する。

## 1. 目的

Phase 9 の SN2 試行で、求核剤 OH⁻ が CH3Cl の **裏側 (Walden inversion 軸、Cl-C-O = 180°) ではなく 133.55° という不自然な角度** から接近する path が選ばれた。本フェーズではこれを scoring や sampling の小細工ではなく **「AFIR 力を加える前に短時間 unbiased FIRE 緩和を挟む」** という物理的補正で解決する。

CH3Cl は Cl δ-, C δ+ の双極子を持ち、OH⁻ は静電的に CH3 側 (C δ+ の反対) — すなわち Cl の裏側 — に引き寄せられる。この ion-dipole 引力は AFIR の幾何的 pull (中心間 attractor) より弱いが、AFIR がまだ active でない初期 30 step 程度なら placement 由来の任意な開始方向を **物理的に正しい pre-reaction complex 配置** に補正できる。

## 2. 問題の再分析 (Phase 9 SN2 の症状)

`out/sn2/meta.json` (Phase 9 実行) より:

- placement: 64 candidates → vdW shadow blocking で 20 survived → 全 trial に AFIR を適用
- 20 survivors のうち、Walden 軸 (Cl→C の反対方向) との角度 ≤ 30° は **trial 8 のみ** (6.4° off, peak −15669.188 eV)
- `score_trials` は `min(peak_energy)` で **trial 10** を選択 (46.4° off, peak −15669.207 eV)
- score 差 = 0.019 eV は UMA noise floor 以下 → 幾何学的な妥当性が scoring に反映されない構造的問題

**修正案の選択**: 解決アプローチは複数あった (A: sampling を Walden 軸付近に偏らせる, B: scoring に幾何 prior 追加, C: blocking を強化, D: pre-relax)。A-C は「正解の幾何」を事前知識として埋め込むためレシピ的 (SN2 専用 hack に近づく)。D は **物理的相互作用** で原子を動かすため、SN2 以外の反応にも転用可能で reactx の汎用性を保つ。本フェーズは D を採用する。

## 3. Non-goals

- **AFIR 強度の再 tune**: SN2 の `alpha_formed=4.0`, `alpha_broken=2.5` は維持。Pre-relax が成功すれば既存値で Walden 経由の path が選ばれるはず。
- **Scoring 改修**: peak_energy ベースの選択ロジックは不変。
- **既存 7 反応の挙動保証**: pre_relax がデフォルト ON になるため、E2/DA 等で path が変わる可能性がある。slow test が落ちた反応は `.rxn.toml` で `pre_relax_steps` を個別に下げる/0 にして再 tune する (本フェーズ内の必要なら)。
- **Pre-relax 専用 FIRE ハイパラ**: 既存 FIRE 設定 (`maxstep=0.1, dt=0.05, dtmax=0.2`) を Stage A でも流用。`pre_relax_steps` は step 数上限のみ。fmax は Stage B と共有 (`args.relax_fmax`, default 0.1)。

## 4. アーキテクチャ

### 4.1 パイプライン (変更点ハイライト)

```
placement.valid_placements
       │
       ▼
build_afir_constraint                       (constraint オブジェクトを構築するだけ; まだ未適用)
       │
       ▼
relax_with_restraints (2-stage 化)          ★Phase 10 で変更★
       │
       ├─ Stage A: pre_relax_steps > 0 の時、constraints=[] で FIRE
       │           → 終了時の geometry を final_state["frame_after_pre_relax"] に保存
       │
       └─ Stage B: 既存ロジック (constraint ON, max_relax_steps まで FIRE)
       │
       ▼
count_initial_latched (★pre-relax 後フレームに対して評価★)
       │
       ▼
TrialResult (frames は Stage A + B 連結)
```

### 4.2 `[afir]` schema 拡張

```toml
[afir]
alpha_formed   = ...
alpha_broken   = ...
max_relax_steps = 300
pre_relax_steps = 30          # 新規 (default 30)
```

- **default = 30**: 未指定なら 30 step の unbiased FIRE pre-relax を実行。
- **value = 0**: 明示的に 0 で skip (機能オフ; 個別反応で副作用が出た場合の opt-out)。
- **value > 0**: 非負整数。Stage A の step 数上限。
- **負値 / non-int**: `ConfigError`。
- `_AFIR_KEYS` に追加、`_AFIR_REQUIRED` には **追加しない**。

### 4.3 `relax_with_restraints` の 2-stage 化

```python
def relax_with_restraints(
    atoms: Atoms,
    restraints: list,
    calc: Calculator,
    *,
    pre_relax_steps: int = 0,        # 新規
    max_steps: int = 100,
    fmax: float = 0.1,
    traj_stride: int = 5,
) -> tuple[list[Atoms], list[float], dict]:
```

**動作 (`pre_relax_steps > 0` の場合)**:

1. atoms をコピーし `atoms.calc = calc`
2. `atoms.set_constraint([])` (Stage A は constraints OFF)
3. 初期 snapshot を frames[0] / energies[0] に記録
4. FIRE を Stage A 専用に作成、`pre_relax_steps` を `steps` 上限として `opt.run(fmax=fmax, steps=pre_relax_steps)`
5. Stage A 末尾の geometry を `final_state["frame_after_pre_relax"]` (Atoms snapshot, constraint stripped) に保存
6. `atoms.set_constraint(restraints)` で AFIRConstraint 適用
7. FIRE を **再生成** (state を持ち越さない) し、`opt.run(fmax=fmax, steps=max_steps)` で Stage B 実行
8. Stage A の frames/energies と Stage B の frames/energies を **連結** (末尾と先頭が同じ geometry でも重複許容)

**動作 (`pre_relax_steps == 0` の場合)**:
- 既存挙動 (Phase 9 と完全互換)。`final_state["frame_after_pre_relax"]` は **キーごと省略** (KeyError で「pre-relax していない」と判定できる)。

**戻り値 `final_state` の拡張**:
```python
{
    "formed_latched": [...],
    "broken_latched": [...],
    "frame_after_pre_relax": Atoms | <省略>,  # 新規 (pre-relax 実行時のみ)
}
```

### 4.4 `count_initial_latched` の評価フレーム変更 (CLI 側)

現状: `atoms_init` (placement 直後) に対して latch 数を計算。
Phase 10: `pre_relax_steps > 0` ならば **pre-relax 後フレーム** で計算。

CLI 実装 (`reactx/cli.py`):
```python
frames, energies, final_state = relax_with_restraints(
    atoms_init, afir_cs, calc,
    pre_relax_steps=cfg.afir.pre_relax_steps,
    max_steps=cfg.afir.max_relax_steps,
    fmax=args.relax_fmax,
    traj_stride=args.traj_stride,
)

ref_frame = final_state.get("frame_after_pre_relax", atoms_init)
initial_latched = count_initial_latched(
    ref_frame, formed_pairs, broken_pairs, ft, bt,
)
```

理由: pre-relax で原子が動くため、AFIR force の初期 latch 判定基準は pre-relax 後の geometry であるべき。meta.json の `initial_latched_formed` / `initial_latched_broken` は「AFIR force が active になる瞬間の latch 状態」を意味する。

### 4.5 `effective_params` への追加

`meta.json` の `effective_params` に `pre_relax_steps` を追加 (実際に使われた値; cfg.afir.pre_relax_steps をそのまま記録)。

### 4.6 `examples/*.rxn.toml`

default が 30 なので **どの example も `pre_relax_steps` を書く必要はない**。明示したい場合 (SN2 の検証など) のみ `[afir]` セクションに記載する。

```toml
[afir]
alpha_formed = 4.0
alpha_broken = 2.5
max_relax_steps = 300
# pre_relax_steps = 30 (default なので省略可)
```

他の 7 反応 (PT, Menshutkin, E2, SN1 dissoc, SN1 recomb, DA simple, DA endo) は **触らない**。pre-relax が default 30 で適用されるため、slow test の挙動が変わる可能性がある。落ちた場合は個別に `pre_relax_steps = 0` を追加するか、別値で tune する。

## 5. テスト計画 (TDD)

### 5.1 Unit — `tests/test_config_afir.py`

- `test_pre_relax_steps_default_is_30`: 未指定で `cfg.afir.pre_relax_steps == 30`
- `test_pre_relax_steps_accepts_positive_int`: `pre_relax_steps = 50` → AFIRSection.pre_relax_steps == 50
- `test_pre_relax_steps_zero_allowed`: 明示 0 で skip 意図 (opt-out)
- `test_pre_relax_steps_negative_rejected`: -1 → `ConfigError`
- `test_pre_relax_steps_non_int_rejected`: 1.5 / "30" → `ConfigError`

**注意**: 既存テスト (`test_load_minimal_sn2` 等) は `pre_relax_steps` を assert していなければそのまま pass。default 値の存在を仮定する assertion を追加する必要があれば既存テストにも一行追加する。

### 5.2 Unit — `tests/test_path_relax.py` (新規 or 既存拡張)

- `test_pre_relax_steps_zero_behaves_like_phase9`: `pre_relax_steps=0` で従来挙動。`final_state` に `frame_after_pre_relax` キーが **無い**
- `test_pre_relax_runs_without_constraints`: `pre_relax_steps=5`, AFIRConstraint 渡しても、Stage A の中で `atoms.constraints` は空。Stage A 末尾で `frame_after_pre_relax` snapshot が記録
- `test_two_stage_frames_concatenated`: frames リストは Stage A + B の連結。Stage A の最後 と Stage B の最初は同一 geometry (許容)

### 5.3 E2E (slow) — `tests/test_re2_sn2.py` を Phase 10 仕様に拡張

- 既存の `tests/test_re2_sn2.py` に Walden 角度 assertion を追加する (新ファイルを増やさない)
- `examples/sn2.rxn.toml` (pre_relax 未記載 = default 30 適用) で `reactx` を実行
- meta.json の `effective_params.pre_relax_steps == 30` を確認
- meta.json の `converged == True` および selected_trial の `reached_product == True`
- trajectory.xyz の全フレームを読み、各フレームで Cl(idx=1)-C(idx=0)-O(idx=2) の三点角度を計算
- Assertion: **最大角度 ≥ 150°** (Walden inversion 軸 = 180° に近い configuration が path 上に存在)
  - 単一フレーム判定ではなく path 全体での最大値を使う理由: pre-relax 後 / AFIR 初期 / TS 付近など Walden geometry を経由しているはず。trajectory.xyz から pre-relax 境界を特定する必要をなくす
  - 現状 (Phase 9) では trial 10 のすべての frame で Cl-C-O 角度が 133° 前後で推移している。150° を越えれば回帰防止になる

### 5.4 既存 7 反応の slow test

- pre_relax default 30 で実行し、各反応の `reached_product` が True を維持するか確認
- 落ちた反応: その反応の `.rxn.toml` に `pre_relax_steps = 0` を追加して opt-out (まず動作優先、再 tune は後続フェーズ)
- これは「実装後の検証」であり、TDD のテストとしては「既存テストが pass する」ことが目標

## 6. リスクと対策

| リスク | 対策 |
|---|---|
| Pre-relax 中に CH3Cl が解離 / OH⁻ が CH3Cl から逃げる | 30 step + fmax 0.1 では平衡到達前に止まるはず。E2E で確認。問題なら `pre_relax_steps` を 15-20 に下げる |
| Pre-relax 後に formed/broken pair が既に閾値を越えて即 latch | `initial_latched_*` で可視化。SN2 では C-O 距離が pre-relax で 1.633 Å を切ることはほぼないはず |
| 他反応 (E2/DA 等) で副作用 | default 30 が既存 path を破壊する可能性あり。落ちた反応は個別に `pre_relax_steps = 0` で opt-out。重要な reaction が path を変えた場合は再 tune を後続タスクに残す |
| Stage A の FIRE state を Stage B に持ち越すと最適化方向が不連続になる | Stage B 開始時に FIRE を **再生成** (`opt = FIRE(...)` を 2 回呼ぶ)。velocity も初期化される |

## 7. Step Plan (TDD)

1. **Config** (test → impl)
   - `tests/test_config_afir.py` に 5 ケース追加 → 失敗確認
   - `reactx/config.py`: `AFIRSection.pre_relax_steps`, `_AFIR_KEYS`, `_build_afir` validation 追加 → 全テスト pass
2. **Path-relax** (test → impl)
   - `tests/test_path_relax.py` 拡張 → 失敗確認
   - `reactx/path_relax.py`: 2-stage 実装 → pass
3. **CLI 配線**
   - `reactx/cli.py`: `pre_relax_steps` を `relax_with_restraints` に渡す。`count_initial_latched` の参照フレームを `final_state["frame_after_pre_relax"]` に変更。`effective_params` に追加
4. **example 設定**
   - `examples/sn2.rxn.toml` は default 30 で動くため変更不要。他反応も基本変更なし
5. **E2E 検証 (SN2)**
   - SN2 を実行し meta.json + trajectory.xyz で Cl-C-O 角度を確認
   - `tests/test_re2_sn2.py` に Walden 角度 assertion を追加
6. **既存 slow test 確認**
   - 7 反応の slow test を実行。落ちた反応は `pre_relax_steps = 0` を `.rxn.toml` に追加して opt-out
7. **README 更新**
   - Phase 10 セクションを追加。既存 Phase 9 セクションは保持

## 8. オープン質問 (実装中に決定)

- E2E で 30 step が過剰 / 不足だった場合の調整: 検証して 15/20/30 から選択
- Pre-relax 中の frames を trajectory.xyz に含めるかは **含める** で確定 (重複フレームは許容、可視化価値優先)
