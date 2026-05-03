# reactx — MMFF Prescreen Design

- Status: Draft (awaiting user review)
- Date: 2026-05-02
- Owner: @kam6y
- Branch: `phase-Re1` (from 3c96a8b)
- 前提仕様: `docs/superpowers/specs/2026-04-27-reactx-phase-Re1-design.md`, `docs/superpowers/specs/2026-05-02-reaction-type-presets-design.md`

## 1. 目的

Phase Re1 default では `--n-angles 8` の trial 全件を UMA で relax している (SN2 ~67s, PT ~116s, Menshutkin ~3 min)。README の実測では SN2 / PT で 8/8 trial が `reached_product=True` を達成しており、**少数の代表 trial を UMA に渡せば十分**。

本変更で UMA 呼び出し前に **MMFF94 拘束緩和による安価な prescreen** を挟み、wall-clock を半減させる。MMFF が parameterize できない系 (charged species, ion pair) では透過的に full UMA にフォールバックし、対応反応の幅は維持する。

## 2. スコープ

### 2.1 In scope

- 新モジュール `reactx/prescreen.py` — RDKit MMFF94 を ASE Calculator にラップ + top-K 選別
- 既存 `path_relax.relax_with_restraints` / `scoring` を MMFF calc で再利用
- CLI フラグ 3 つ追加: `--no-mmff-prescreen`, `--prescreen-keep`, `--prescreen-steps`
- `meta.json` に `prescreen` セクション追加
- MMFF parameterize 失敗時の透過フォールバック (warning + 全件 UMA へ)
- Unit test (toy system) + slow integration test (3 反応すべて) + wall-clock 閾値更新
- README の wall-clock 表更新 + フラグ追記

### 2.2 Out of scope

- UFF / GFN-FF など別 force field のサポート — MMFF94 のみ
- MMFF parameter override (atom type 強制指定など) — RDKit デフォルトのみ
- prescreen 結果のキャッシュ / 並列化 — 単純逐次実行
- prescreen で得た trajectory の出力 — UMA に渡す trial idx だけ伝搬、MMFF の中間 frame は破棄
- 反応タイプ別の prescreen パラメータ (preset.prescreen_steps 等) — 全 preset 共通の値で運用

## 3. アーキテクチャ

### 3.1 パイプラインへの差し込み

```
.rxn → parse → bond_changes
              ↓
       sample N rotations (default 8)
              ↓
       embed3d (×N)
              ↓
       ┌─── MMFF prescreen (NEW) ───┐
       │  for each trial:           │
       │    MMFF restraint relax    │
       │    (FIRE, 30 step)         │
       │  → select_top_k_indices    │
       │    (default K=3)           │
       └────────────┬────────────────┘
                    ↓
              UMA relax (×K, 既存と同一)
                    ↓
              score_trials で best 1
                    ↓
              (optional) NEB refine
                    ↓
              trajectory.xyz / blender
```

### 3.2 新規モジュール: `reactx/prescreen.py`

```python
from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import (
    MMFFGetMoleculeProperties, MMFFGetMoleculeForceField,
)

from reactx.path_relax import relax_with_restraints
from reactx.scoring import TrialResult, reached_product

log = logging.getLogger(__name__)

# kcal/mol → eV (CODATA 2018)
KCAL_PER_MOL_TO_EV = 0.0433641153
# kcal/mol/Å → eV/Å (force units use the same scale factor)


class MMFFParameterizationError(RuntimeError):
    """RDKit MMFF94 could not parameterize the molecule (e.g. ion, exotic atom type)."""


class RDKitMMFFCalculator(Calculator):
    """ASE Calculator backed by RDKit MMFF94.

    The calculator is bound to a fixed RDKit Mol topology supplied at __init__.
    Each `calculate` updates the conformer coordinates from the ASE Atoms,
    re-evaluates energy + gradients via MMFFGetMoleculeForceField, and
    converts kcal/mol → eV.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, rdkit_mol: Chem.Mol):
        super().__init__()
        # MMFF properties may return None for ions/unsupported types.
        props = MMFFGetMoleculeProperties(rdkit_mol)
        if props is None:
            raise MMFFParameterizationError(
                "MMFFGetMoleculeProperties returned None"
            )
        self._mol = Chem.Mol(rdkit_mol)  # owned conformer
        self._props = props

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        conf = self._mol.GetConformer()
        for i in range(self._mol.GetNumAtoms()):
            conf.SetAtomPosition(i, tuple(map(float, positions[i])))
        ff = MMFFGetMoleculeForceField(self._mol, self._props)
        if ff is None:
            raise MMFFParameterizationError(
                "MMFFGetMoleculeForceField returned None at evaluation"
            )
        e_kcal = ff.CalcEnergy()
        # CalcGrad returns gradient in kcal/mol/Å (∂E/∂x). Force = -gradient.
        grad = np.asarray(ff.CalcGrad(), dtype=float).reshape(-1, 3)
        self.results = {
            "energy": float(e_kcal) * KCAL_PER_MOL_TO_EV,
            "forces": -grad * KCAL_PER_MOL_TO_EV,
        }


@dataclass(frozen=True)
class PrescreenResult:
    enabled: bool
    kept: list[int]              # trial indices passed to UMA
    skipped: list[int]           # trial indices filtered out
    mmff_failed: bool            # True if fell back due to MMFF parameterize error
    wall_clock_seconds: float


def select_top_k_indices(trials: list[TrialResult], k: int) -> list[int]:
    """Top-K trial indices by score_trials' preference order:
    reached_product=True first, then lowest peak_energy. K spans both groups
    (e.g. 2 reached + 1 unreached when K=3 and only 2 trials reached).
    """
    reached = sorted(
        [t for t in trials if t.reached_product],
        key=lambda t: t.peak_energy,
    )
    not_reached = sorted(
        [t for t in trials if not t.reached_product],
        key=lambda t: t.peak_energy,
    )
    return [t.trial_idx for t in (reached + not_reached)[:k]]


def prescreen_trials(
    atoms_list: list[Atoms],
    rdkit_mol_list: list[Chem.Mol],
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    *,
    r_form_target: float,
    r_broken_target: float,
    k_form: float,
    k_broken: float,
    max_steps: int = 30,
    fmax: float = 0.5,            # looser than UMA (0.1) — coarse signal
    traj_stride: int = 5,
    k_keep: int = 3,
) -> PrescreenResult:
    """Run MMFF restraint relaxation per trial and return top-K trial indices.

    On any MMFF parameterize failure, falls back to keeping all trial indices
    and sets mmff_failed=True (warning logged once).
    """
    import time
    from reactx.artificial_force import build_restraints

    n = len(atoms_list)
    if k_keep >= n:
        return PrescreenResult(
            enabled=True, kept=list(range(n)), skipped=[],
            mmff_failed=False, wall_clock_seconds=0.0,
        )

    t0 = time.monotonic()
    trials: list[TrialResult] = []
    try:
        for i, (atoms_i, mol_i) in enumerate(zip(atoms_list, rdkit_mol_list, strict=True)):
            calc = RDKitMMFFCalculator(mol_i)
            restraints = build_restraints(
                atoms_i, formed=formed, broken=broken,
                r_form=r_form_target, r_broken=r_broken_target,
                k_form=k_form, k_broken=k_broken,
            )
            frames, energies = relax_with_restraints(
                atoms_i, restraints, calc,
                max_steps=max_steps, fmax=fmax, traj_stride=traj_stride,
            )
            ok = reached_product(
                frames[-1], formed=formed, broken=broken,
                r_form_targets=[r_form_target] * len(formed),
                r_broken_target=r_broken_target,
            )
            trials.append(TrialResult(
                trial_idx=i, rotation_deg=0.0,
                frames=frames, energies=energies,
                reached_product=ok,
                peak_energy=max(energies),
                n_steps=len(frames),
            ))
    except MMFFParameterizationError as err:
        log.warning(
            "MMFF prescreen failed (%s): falling back to full UMA on all %d trials.",
            err, n,
        )
        return PrescreenResult(
            enabled=True, kept=list(range(n)), skipped=[],
            mmff_failed=True,
            wall_clock_seconds=float(time.monotonic() - t0),
        )

    kept = select_top_k_indices(trials, k_keep)
    skipped = [i for i in range(n) if i not in kept]
    log.info(
        "prescreen: %d trials -> %d kept (idx=%s) in %.1fs",
        n, len(kept), kept, time.monotonic() - t0,
    )
    return PrescreenResult(
        enabled=True, kept=kept, skipped=skipped, mmff_failed=False,
        wall_clock_seconds=float(time.monotonic() - t0),
    )
```

設計判断:

- **MMFF calc を ASE Calculator として包む**: 既存 `relax_with_restraints` を変更せず再利用できる。
- **per-trial に新規 calc を立てる**: RDKit Mol は trial 間で異なる Conformer 座標を持ち、calc も Mol に紐づく。8 trial で 8 calc 作成 (~1ms/個) はオーバーヘッド無視可能。
- **`fmax=0.5`** (UMA は 0.1) — coarse な signal で十分。早く終わらせる。
- **`max_steps=30`** — UMA の 100 より短い。MMFF はバリアを精密に推定する必要がない (ranking さえ取れればよい)。
- **`MMFFParameterizationError` の透過フォールバック**: 1 trial でも fail したら全体を諦めて full UMA。部分成功 (一部 trial だけ MMFF で評価) は順位の信頼性が下がるため採用しない。

### 3.3 RDKit Mol の調達

`prescreen_trials` は trial ごとの ASE Atoms に対応する RDKit Mol が必要。Phase Re1 の `embed_mol_to_atoms` は ASE Atoms しか返さないため、CLI 層で **embed と同時に RDKit Mol も用意する** 改修が必要。

具体的には、`embed3d.embed_mol_to_atoms` が内部で生成している RDKit Mol (reactant H 付き) を「`atoms`」と一緒に返すように変更する。詳細:

- `embed_mol_to_atoms(...)` の返り値を `Atoms` から `tuple[Atoms, Chem.Mol]` に変更
- `_cmd_run` の embed ループも `(atoms, mol)` のペアで保持

旧 caller (もしテストで Atoms 単独を期待しているもの) は影響を受けるため、tests/test_embed3d.py の expectation を更新する必要がある。

### 3.4 既存モジュールへの影響

| モジュール | 変更内容 |
|---|---|
| `reactx/embed3d.py` | `embed_mol_to_atoms` の返り値を `(Atoms, Chem.Mol)` に拡張 |
| `reactx/cli.py` | `_cmd_run` の embed → relax 間に prescreen 呼び出し追加。CLI flag 3 つ追加。`meta.json` に `prescreen` セクション。 |
| `tests/test_embed3d.py` | 返り値 unpacking 修正 |

`reactx/path_relax.py` / `reactx/scoring.py` / `reactx/artificial_force.py` は **無変更**。

## 4. CLI 仕様

### 4.1 新規フラグ

| flag | type | default | 動作 |
|---|---|---|---|
| `--no-mmff-prescreen` | store_true | `False` (= prescreen ON) | prescreen を無効化、全 N trial を UMA に流す |
| `--prescreen-keep` | int | `3` | UMA に渡す trial 数 (top-K) |
| `--prescreen-steps` | int | `30` | MMFF 拘束緩和の FIRE step 数 |

`--prescreen-keep >= n_angles` の場合: prescreen は `kept = [0..n-1]` を返してショートサーキット (MMFF を呼ばない)。

### 4.2 meta.json 拡張

```json
{
  "backend": "uma",
  "reaction_type": "sn2_anion",
  "converged": true,
  "selected_trial": 3,
  "trials": [
    {"trial": 0, "reached_product": true,  "peak_energy": -16322.7, ...},
    {"trial": 3, "reached_product": true,  "peak_energy": -16320.1, ...},
    {"trial": 5, "reached_product": false, "peak_energy": -16310.4, ...}
  ],
  "prescreen": {
    "enabled": true,
    "kept": [0, 3, 5],
    "skipped": [1, 2, 4, 6, 7],
    "mmff_failed": false,
    "wall_clock_seconds": 1.8
  },
  "wall_clock_seconds": 28.4,
  "neb_refined": false,
  "effective_params": {...}
}
```

`trials[]` には UMA で実走した分のみ記録 (= `kept` と一致)。`skipped` の trial は UMA に渡されないので結果がない。

`prescreen.enabled=false` (CLI で `--no-mmff-prescreen`) のとき: `kept` / `skipped` / `mmff_failed` は `null`、`wall_clock_seconds` は `0.0`。

### 4.3 ログ出力例

成功時:
```
INFO  prescreen: 8 trials -> 3 kept (idx=[0,3,5]) in 1.8s
INFO  preset=sn2_anion effective: k_form=0.50 k_broken=1.00 r_broken=4.00 max_relax_steps=100 r_form=1.430
```

MMFF 失敗時:
```
WARN  MMFF prescreen failed (MMFFGetMoleculeProperties returned None): falling back to full UMA on all 8 trials.
```

ユーザー無効化:
```
INFO  prescreen: disabled (--no-mmff-prescreen)
```

## 5. テスト戦略

### 5.1 Unit (fast, UMA 不要)

#### `tests/test_prescreen.py` (新規)

| test | 内容 |
|---|---|
| `test_mmff_calc_water_energy` | H2O について `RDKitMMFFCalculator` の energy が finite で eV order (-1 ~ 0 eV 程度) |
| `test_mmff_calc_forces_finite` | 同 H2O で forces が `(3, 3)` 形状で finite |
| `test_mmff_calc_charged_raises` | `[O-]` の Mol で `MMFFParameterizationError` が raise |
| `test_select_top_k_reached_first` | 3 reached + 5 unreached、k=3 で全 reached が選ばれる |
| `test_select_top_k_mixed_when_few_reached` | 1 reached + 7 unreached、k=3 で reached + top 2 unreached |
| `test_select_top_k_all_unreached` | 0 reached、k=3 で peak_energy 最小の 3 件 |
| `test_prescreen_short_circuit_when_k_ge_n` | `k_keep=8`, `n=8` で MMFF を呼ばずに full pass |
| `test_prescreen_handles_mmff_failure` | charged Mol を渡すと `mmff_failed=True` + `kept=[0..n-1]` |

LJ calc を使った合成テストではなく、最小実分子 (H2O 等) で実 RDKit を呼ぶ。MMFF94 の H2O 評価は ms 単位で十分高速。

#### `tests/test_cli.py` 拡張

```python
def test_default_prescreen_flags():
    a = build_parser().parse_args(["run", "examples/sn2.rxn", "-o", "out/"])
    assert a.no_mmff_prescreen is False
    assert a.prescreen_keep == 3
    assert a.prescreen_steps == 30


def test_prescreen_disabled_flag():
    a = build_parser().parse_args([
        "run", "examples/sn2.rxn", "-o", "out/", "--no-mmff-prescreen",
    ])
    assert a.no_mmff_prescreen is True


def test_prescreen_keep_override():
    a = build_parser().parse_args([
        "run", "examples/sn2.rxn", "-o", "out/", "--prescreen-keep", "1",
    ])
    assert a.prescreen_keep == 1
```

### 5.2 Slow integration (UMA 必須)

| ファイル | 変更 |
|---|---|
| `tests/test_re1_sn2.py` | default で prescreen ON。`meta.json.prescreen.kept` 長 == 3 / `trials[]` 長 == 3 を assertion 追加 |
| `tests/test_re1_proton_transfer.py` | 同上 |
| `tests/test_re1_menshutkin.py` | 同上、ただし `mmff_failed=True` のケースも許容 (中性 → ion pair の MMFF 評価は境界条件)。`reached_product` を満たす trial が UMA で出ることだけ要求 |
| `tests/test_wallclock_sn2.py` | 閾値 60s 据え置き (UMA model load ~25-30s が固定コストで支配的のため、prescreen 短縮分は表に出にくい。実測 ~25-30s に対しハードウェア変動の余裕として 60s を維持) |
| `tests/test_prescreen_disabled_sn2.py` (新規) | `--no-mmff-prescreen` で `meta.json.prescreen.enabled=false` かつ `len(trials) == n_angles` を確認 |

### 5.3 Blender

`tests/test_blender_smoke.py` は無変更 (prescreen は trajectory 出力に影響しない)。

## 6. README 更新

「使い方」セクションに `--prescreen-keep` / `--no-mmff-prescreen` 追記:

```markdown
- `--prescreen-keep 3` (default): MMFF prescreen で UMA に渡す trial 数
- `--no-mmff-prescreen`: prescreen を無効化、全 trial を UMA に流す (Phase Re1 default 挙動)
```

「Wall-clock (実測)」セクションを更新:

| Reaction | wall-clock (旧, 8/8 UMA) | wall-clock (新, prescreen + 3/8 UMA) | 短縮率 |
|---|---|---|---|
| SN2 (`examples/sn2.rxn`) | ~67s | ~25-30s | ~55% |
| Proton transfer | ~116s | ~45-50s | ~60% |
| Menshutkin | ~3 min | ~1.5 min | ~50% |

「MMFF prescreen は中性求核剤の SN2 / PT で効率的に上位 trial を選別できるが、ion pair を含む系では parameterize に失敗してフォールバック (= 旧挙動) する場合がある。失敗は `meta.json.prescreen.mmff_failed=true` で確認できる。」を 1 段落で追記。

## 7. Definition of Done

1. fast suite 全 pass (新規 `test_prescreen.py` + 拡張 `test_cli.py`)
2. slow suite 全 pass (3 反応統合 + new `test_prescreen_disabled_sn2`)
3. SN2 default 実走で `wall_clock_seconds < 60` (= `test_wallclock_sn2` の現行閾値。UMA model load コストにより理論短縮値より緩めに設定)
4. `reactx run examples/sn2.rxn -o out/sn2/ --backend uma` で `meta.json.prescreen.enabled=true` / `kept` 長 == 3 / `trials` 長 == 3
5. `reactx run ... --no-mmff-prescreen` で `meta.json.prescreen.enabled=false` / `trials` 長 == 8
6. Menshutkin 実走で `mmff_failed=true` でも完走、`reached_product=True` の trial が ≥ 1 件 (fallback 経路の動作確認)
7. README の wall-clock 表が新値で更新、新フラグが「主要フラグ」に追記されている

## 8. リスクと緩和

| リスク | 影響 | 緩和 |
|---|---|---|
| MMFF が SN2 anion (O⁻) で fail し prescreen 効果が消失 | wall-clock 短縮 0 | RDKit MMFF94 は anion でも parameterize できるケースが多い。事前検証で確認、できなければ「default ON だが anion 系では fallback が頻発する」と README に明記 |
| MMFF の ranking が UMA とずれて「正解 trial」を skip する | best trial を取り逃す | `reached_product` を主基準、`peak_energy` は同列のみ。distance-based の geometric signal が支配的なため、MMFF energy 単体に依存しない |
| RDKit MMFF の gradient 単位が想定と異なる | 力が破綻、relaxation 暴走 | RDKit `CalcGrad` は kcal/mol/Å (公式 doc 記載)。ASE は eV/Å を期待。Unit test で H2O の force magnitude が物理的 range (~ 1 eV/Å order) であることを確認 |
| `embed_mol_to_atoms` の返り値変更が他 caller (test) を壊す | 既存テスト red | tests/test_embed3d.py を tuple unpacking に更新。grep で全 caller を洗い出し、CLI 以外の参照がないことを確認 |
| prescreen の wall-clock が想定より大きく (~5-10s) なって net gain が消える | DoD #3 失敗 | `max_steps=30, fmax=0.5` で控えめに設定。実測で >3s/8 trials なら steps を 20 に削減 |
| `--prescreen-keep 3` が SN2 では多すぎ / Menshutkin では少なすぎ | デフォルトがどちらかに最適化されない | preset ごとの prescreen_keep override は out of scope (§2.2)。CLI flag で個別に上書き可能 |

## 9. Phase 2 への影響

本変更は Phase Re1 / preset 設計の上に **使い勝手 + wall-clock の改善** を一段重ねる位置付け。Phase 2 ロードマップ (generic bond-change engine) には影響しない (`prescreen_trials` は formed/broken の数を仮定していないため、将来の SimpleBondChanges → BondChanges 拡張にも追従可能)。

将来 prescreen 自体を進化させる方向:
- preset 連動 (`preset.prescreen_keep`, `preset.prescreen_steps`)
- UFF fallback (MMFF 失敗時にも UFF で prescreen 継続)
- MMFF trajectory を `meta.json` の debug info として残す
- 並列 prescreen (8 trials × MMFF は並列化可能、UMA と違って GPU 競合しない)

これらは需要が出た時点で別 spec として追加する。
