# Reaction-Type Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `--reaction-type` CLI フラグを追加し、`sn2_anion` / `proton_transfer` / `menshutkin` の 3 プリセットを 1 引数で適用できるようにする。個別フラグはプリセットを上書きする。

**Architecture:** プリセットは `reactx/presets.py` の Python dict として組み込み。CLI 層 (`reactx/cli.py`) で `--reaction-type` を受け取り、None の個別フラグだけプリセット値で埋めるシンプルな merge ロジック。コア関数 (`artificial_force.py` / `path_relax.py` / etc.) の API は不変。

**Tech Stack:** Python 3.11, argparse, pytest, ASE (constraints), RDKit (.rxn parsing), UMA (slow tests のみ)。

**前提仕様:** `docs/superpowers/specs/2026-05-02-reaction-type-presets-design.md`

---

## File Structure

**新規作成:**
- `reactx/presets.py` — `ReactionPreset` dataclass + `PRESETS` dict + `get_preset()` の 1 ファイル / 1 責務 (50 行程度)
- `tests/test_presets.py` — `presets.py` の unit test
- `examples/menshutkin.rxn` — NH₃ + CH₃Cl → CH₃NH₃⁺ + Cl⁻ の RXN fixture
- `tests/test_re1_menshutkin.py` — Menshutkin の slow 統合テスト

**変更:**
- `reactx/cli.py` — `--reaction-type` 引数追加 / 個別フラグ default を None 化 / preset merge ロジック / `meta.json` 出力に 2 フィールド追加
- `tests/test_cli.py` — 既存 default 期待値更新 + 上書きセマンティクスの新テスト
- `tests/test_re1_sn2.py` — **preflight**: 既存の F 参照を O に修正 (HEAD 状態で broken のため)
- `tests/test_re1_proton_transfer.py` — `--reaction-type proton_transfer` を使うケース 1 件追加
- `README.md` — 「Reaction-type presets」節追加

**削除:**
- `memory/project_reactx_endothermic_tuning.md`
- `memory/MEMORY.md` の該当エントリ行

---

## Task 0: Preflight — `tests/test_re1_sn2.py` の F→O 参照修正

**Why:** commit 2012d11 で `examples/sn2.rxn` を CH₃Cl + F⁻ → CH₃F + Cl⁻ から CH₃Cl + O⁻ → CH₃O + Cl⁻ に置き換えたが、`test_re1_sn2.py` の `syms.index("F")` がそのまま残っているため、HEAD で `pytest -m slow tests/test_re1_sn2.py` を走らせると `ValueError: 'F' is not in list` で fail する。spec §8 DoD #7 (既存 SN2 test が無変更で pass) を満たすために本タスクで修正する。

**Files:**
- Modify: `tests/test_re1_sn2.py` (4 箇所の F → O 置換)

- [ ] **Step 0.1: 現状確認**

```bash
grep -n "\"F\"\|f_idx\|d_cf\|v_cf" tests/test_re1_sn2.py
```

期待: 4 行ヒット (line 36, 41, 52, 53 付近)。

- [ ] **Step 0.2: F → O 置換**

`tests/test_re1_sn2.py` を以下のように書き換える (差分 4 箇所):

```python
# line 36 付近
    f_idx = syms.index("F")
# ↓
    o_idx = syms.index("O")

# line 41 付近 (for f in frames: ループ内)
        v_cf = f.positions[f_idx] - f.positions[c_idx]
# ↓
        v_co = f.positions[o_idx] - f.positions[c_idx]

# line 43 付近
        cos_t = float(np.dot(v_cf, v_ccl) / (
            np.linalg.norm(v_cf) * np.linalg.norm(v_ccl)
        ))
# ↓
        cos_t = float(np.dot(v_co, v_ccl) / (
            np.linalg.norm(v_co) * np.linalg.norm(v_ccl)
        ))

# line 47-48 (assertion message)
    assert max(angles) >= 120.0, (
        f"F-C-Cl angle never reached 120° on the trajectory: max={max(angles):.1f}°"
    )
# ↓
    assert max(angles) >= 120.0, (
        f"O-C-Cl angle never reached 120° on the trajectory: max={max(angles):.1f}°"
    )

# line 52-53
    d_cf_first = frames[0].get_distance(c_idx, f_idx)
    d_cf_last = frames[-1].get_distance(c_idx, f_idx)
# ↓
    d_co_first = frames[0].get_distance(c_idx, o_idx)
    d_co_last = frames[-1].get_distance(c_idx, o_idx)

# line 56-58 (assertion + message)
    assert d_cf_last < d_cf_first - 0.5, (
        f"C-F should shrink: {d_cf_first:.2f} -> {d_cf_last:.2f}"
    )
# ↓
    assert d_co_last < d_co_first - 0.5, (
        f"C-O should shrink: {d_co_first:.2f} -> {d_co_last:.2f}"
    )
```

- [ ] **Step 0.3: 静的検証 (UMA を起動せず)**

```bash
python -c "import ast; ast.parse(open('tests/test_re1_sn2.py').read()); print('OK')"
```

期待: `OK`。実行は UMA 必須なので skip (DoD で別途確認)。

- [ ] **Step 0.4: コミット**

```bash
git add tests/test_re1_sn2.py
git commit -m "test(re1_sn2): align with O-nucleophile sn2.rxn (post-2012d11)"
```

---

## Task 1: `reactx/presets.py` を TDD で追加

**Files:**
- Create: `reactx/presets.py`
- Create: `tests/test_presets.py`

- [ ] **Step 1.1: 失敗するテストを書く**

`tests/test_presets.py` を新規作成:

```python
"""Unit tests for reaction-type presets."""
import pytest

from reactx.presets import PRESETS, ReactionPreset, get_preset


def test_sn2_anion_values():
    p = get_preset("sn2_anion")
    assert p.name == "sn2_anion"
    assert p.k_form == 0.5
    assert p.k_broken == 1.0
    assert p.r_broken == 4.0
    assert p.max_relax_steps == 100
    assert p.r_form is None  # 元素表 fallback


def test_proton_transfer_values():
    p = get_preset("proton_transfer")
    assert p.name == "proton_transfer"
    assert p.k_form == 0.5
    assert p.k_broken == 1.0
    assert p.r_broken == 4.0
    assert p.max_relax_steps == 100
    assert p.r_form == 1.05


def test_menshutkin_values():
    p = get_preset("menshutkin")
    assert p.name == "menshutkin"
    assert p.k_form == 2.0
    assert p.k_broken == 2.0
    assert p.r_broken == 5.0
    assert p.max_relax_steps == 200
    assert p.r_form is None


def test_get_preset_unknown_raises():
    with pytest.raises(ValueError) as exc:
        get_preset("not_a_preset")
    msg = str(exc.value)
    assert "not_a_preset" in msg
    assert "sn2_anion" in msg
    assert "menshutkin" in msg
    assert "proton_transfer" in msg


def test_presets_dict_keys():
    assert set(PRESETS) == {"sn2_anion", "proton_transfer", "menshutkin"}


def test_preset_is_frozen():
    p = get_preset("sn2_anion")
    with pytest.raises((AttributeError, Exception)):
        p.k_form = 99.0  # frozen dataclass should reject mutation


def test_reaction_preset_dataclass_signature():
    # Default r_form should be None when not provided
    p = ReactionPreset(
        name="ad_hoc", k_form=1.0, k_broken=2.0,
        r_broken=4.5, max_relax_steps=120,
    )
    assert p.r_form is None
```

- [ ] **Step 1.2: テストが失敗することを確認**

```bash
pytest tests/test_presets.py -v
```

期待: `ModuleNotFoundError: No module named 'reactx.presets'` 等で全 fail。

- [ ] **Step 1.3: 最小実装**

`reactx/presets.py` を新規作成:

```python
"""Built-in reaction-type presets for the reactx CLI.

Each preset bundles a tested set of artificial-force restraint parameters
(k_form, k_broken, r_broken, max_relax_steps, optional r_form override) for
a class of reactions. Preset values applied via `--reaction-type` can still
be overridden by individual CLI flags.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReactionPreset:
    name: str
    k_form: float
    k_broken: float
    r_broken: float
    max_relax_steps: int
    r_form: float | None = None  # None = element-pair table lookup


PRESETS: dict[str, ReactionPreset] = {
    "sn2_anion": ReactionPreset(
        name="sn2_anion",
        k_form=0.5,
        k_broken=1.0,
        r_broken=4.0,
        max_relax_steps=100,
    ),
    "proton_transfer": ReactionPreset(
        name="proton_transfer",
        k_form=0.5,
        k_broken=1.0,
        r_broken=4.0,
        max_relax_steps=100,
        r_form=1.05,
    ),
    "menshutkin": ReactionPreset(
        name="menshutkin",
        k_form=2.0,
        k_broken=2.0,
        r_broken=5.0,
        max_relax_steps=200,
    ),
}


def get_preset(name: str) -> ReactionPreset:
    """Return the built-in preset by name. Raises ValueError if unknown."""
    if name not in PRESETS:
        raise ValueError(
            f"Unknown reaction-type preset {name!r}. "
            f"Available: {sorted(PRESETS)}"
        )
    return PRESETS[name]
```

- [ ] **Step 1.4: テスト pass を確認**

```bash
pytest tests/test_presets.py -v
```

期待: 7 tests passed。

- [ ] **Step 1.5: コミット**

```bash
git add reactx/presets.py tests/test_presets.py
git commit -m "feat(presets): add ReactionPreset + 3 built-in presets (sn2_anion, proton_transfer, menshutkin)"
```

---

## Task 2: `reactx/cli.py` に `--reaction-type` を配線 (argparse 部のみ)

**Files:**
- Modify: `reactx/cli.py:29-65` (`build_parser`)
- Modify: `tests/test_cli.py:5-21` (`test_default_flags_parse`)

- [ ] **Step 2.1: テストを更新 (期待 default 変更 + `--reaction-type` 追加)**

`tests/test_cli.py` の既存 `test_default_flags_parse` を以下に置き換え、新テスト 2 件を追加:

```python
"""CLI argument parsing smoke tests (no UMA invocation)."""
import pytest

from reactx.cli import build_parser


def test_default_flags_parse():
    p = build_parser()
    a = p.parse_args(["run", "examples/sn2.rxn", "-o", "out/"])
    assert a.cmd == "run"
    assert a.n_angles == 8
    assert a.cone_half_deg == 30.0
    assert a.seed == 0
    # Preset-overrideable flags now default to None (sentinel).
    assert a.r_form is None
    assert a.r_broken is None
    assert a.k_form is None
    assert a.k_broken is None
    assert a.max_relax_steps is None
    # Default reaction type:
    assert a.reaction_type == "sn2_anion"
    # Unchanged:
    assert a.relax_fmax == 0.1
    assert a.traj_stride == 5
    assert a.neb_refine is False
    assert a.neb_images == 7


def test_neb_refine_flag():
    p = build_parser()
    a = p.parse_args(["run", "examples/sn2.rxn", "-o", "out/", "--neb-refine"])
    assert a.neb_refine is True


def test_n_angles_override():
    p = build_parser()
    a = p.parse_args(["run", "examples/sn2.rxn", "-o", "out/", "--n-angles", "1"])
    assert a.n_angles == 1


def test_r_form_override():
    p = build_parser()
    a = p.parse_args(["run", "examples/sn2.rxn", "-o", "out/", "--r-form", "1.05"])
    assert a.r_form == 1.05


def test_reaction_type_explicit():
    p = build_parser()
    a = p.parse_args([
        "run", "examples/sn2.rxn", "-o", "out/",
        "--reaction-type", "menshutkin",
    ])
    assert a.reaction_type == "menshutkin"


def test_reaction_type_unknown_rejected():
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args([
            "run", "examples/sn2.rxn", "-o", "out/",
            "--reaction-type", "not_a_preset",
        ])
```

- [ ] **Step 2.2: テストが失敗することを確認**

```bash
pytest tests/test_cli.py -v
```

期待: 既存 `test_default_flags_parse` が `assert a.r_broken is None` 等で fail、`test_reaction_type_*` 2 件が `AttributeError: 'Namespace' object has no attribute 'reaction_type'` で fail。

- [ ] **Step 2.3: `build_parser` を更新**

`reactx/cli.py` の `build_parser` (line 29-65) の以下 5 行を修正:

```python
    run.add_argument("--r-form", type=float, default=None,
                     help="Override formed-bond target distance (Å). "
                          "Default: from preset, then element pair table.")
    run.add_argument("--r-broken", type=float, default=None,
                     help="Override repulsion target distance (Å). "
                          "Default: from preset.")
    run.add_argument("--k-form", type=float, default=None,
                     help="Override Hookean k for formed bonds. Default: from preset.")
    run.add_argument("--k-broken", type=float, default=None,
                     help="Override repulsion k for broken bonds. Default: from preset.")

    run.add_argument("--max-relax-steps", type=int, default=None,
                     help="Override FIRE max steps. Default: from preset.")
```

そして `--seed` の直後 (現 line 45 の直後あたり) に新引数を追加:

```python
    from reactx.presets import PRESETS as _PRESETS
    run.add_argument("--reaction-type", choices=sorted(_PRESETS), default="sn2_anion",
                     help="Built-in restraint preset for the reaction class. "
                          "Individual --k-* / --r-* / --max-relax-steps flags override.")
```

(Top-of-file import に `from reactx.presets import PRESETS` を追加してもよいが、循環インポート回避のため関数内 import を採用。)

- [ ] **Step 2.4: テスト pass を確認**

```bash
pytest tests/test_cli.py -v
```

期待: 全 6 tests passed。

- [ ] **Step 2.5: コミット**

```bash
git add reactx/cli.py tests/test_cli.py
git commit -m "feat(cli): add --reaction-type flag, defer k_*/r_*/max_relax_steps defaults to preset"
```

---

## Task 3: CLI に `_resolve_effective_params` ヘルパーを追加 (TDD)

**Why:** プリセット値と個別フラグの merge ロジックを `_cmd_run` の中に直書きするとテストが UMA 起動を伴って重くなる。純粋関数として切り出して fast unit test で挙動を固定する。

**Files:**
- Modify: `reactx/cli.py` (新ヘルパー関数追加 + `_cmd_run` への配線)
- Modify: `tests/test_cli.py` (`_resolve_effective_params` のユニットテスト追加)

- [ ] **Step 3.1: 失敗するテストを書く**

`tests/test_cli.py` の **import 群 (ファイル冒頭)** に追加:

```python
import pytest

from reactx.cli import _resolve_effective_params
```

(既存の `import pytest` / `from reactx.cli import build_parser` がある場合は重複させない。)

そして同ファイルの **末尾** に以下のテストを追加:

```python
def _make_args(**overrides):
    """Build argparse Namespace for tests by parsing then overriding."""
    p = build_parser()
    a = p.parse_args(["run", "examples/sn2.rxn", "-o", "out/"])
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def test_resolve_defaults_to_sn2_anion_preset():
    args = _make_args()
    syms = ["C", "Cl", "H", "H", "H", "F"]  # formed = (0, 5) → C-F
    eff = _resolve_effective_params(args, syms, formed_pair=(0, 5))
    assert eff["reaction_type"] == "sn2_anion"
    assert eff["k_form"] == 0.5
    assert eff["k_broken"] == 1.0
    assert eff["r_broken"] == 4.0
    assert eff["max_relax_steps"] == 100
    # sn2_anion has no r_form override → element-table lookup C-F = 1.39
    assert eff["r_form"] == pytest.approx(1.39)


def test_resolve_menshutkin_preset():
    args = _make_args(reaction_type="menshutkin")
    syms = ["N", "C", "Cl", "H", "H", "H", "H", "H", "H"]
    eff = _resolve_effective_params(args, syms, formed_pair=(0, 1))  # N-C
    assert eff["reaction_type"] == "menshutkin"
    assert eff["k_form"] == 2.0
    assert eff["k_broken"] == 2.0
    assert eff["r_broken"] == 5.0
    assert eff["max_relax_steps"] == 200
    # menshutkin r_form = None → element table N-C = 1.47
    assert eff["r_form"] == pytest.approx(1.47)


def test_resolve_proton_transfer_preset_uses_r_form_1_05():
    args = _make_args(reaction_type="proton_transfer")
    syms = ["H", "Cl", "N", "H", "H"]
    eff = _resolve_effective_params(args, syms, formed_pair=(2, 0))  # N-H
    assert eff["r_form"] == pytest.approx(1.05)


def test_individual_flag_overrides_preset():
    args = _make_args(reaction_type="menshutkin", k_form=3.5, r_broken=6.0)
    syms = ["N", "C", "Cl"]
    eff = _resolve_effective_params(args, syms, formed_pair=(0, 1))
    assert eff["k_form"] == 3.5  # overridden
    assert eff["r_broken"] == 6.0  # overridden
    assert eff["k_broken"] == 2.0  # from preset
    assert eff["max_relax_steps"] == 200  # from preset


def test_r_form_individual_flag_overrides_preset_r_form():
    args = _make_args(reaction_type="proton_transfer", r_form=1.10)
    syms = ["H", "Cl", "N"]
    eff = _resolve_effective_params(args, syms, formed_pair=(2, 0))
    assert eff["r_form"] == pytest.approx(1.10)  # individual flag wins over preset 1.05
```

- [ ] **Step 3.2: テストが失敗することを確認**

```bash
pytest tests/test_cli.py -v
```

期待: 5 件の新テストが `ImportError: cannot import name '_resolve_effective_params'` で fail。

- [ ] **Step 3.3: `_resolve_effective_params` を実装**

`reactx/cli.py` の冒頭 import 群に追加 (line 22-24 付近):

```python
from reactx.artificial_force import build_restraints, lookup_r_form
from reactx.bond_changes import SimpleBondChanges, compute_simple_bond_changes
from reactx.calculators import make_calculator
from reactx.embed3d import embed_mol_to_atoms
from reactx.neb import run_neb
from reactx.path_relax import relax_with_restraints
from reactx.presets import get_preset
from reactx.rxn_parser import heavy_to_hydrogen_groups, parse_rxn
from reactx.scoring import TrialResult, reached_product, score_trials
from reactx.trials import sample_attack_rotations
```

`_cmd_run` の前 (例えば line 67-71 の `_configure_reactx_logging` の後) に新関数を追加:

```python
def _resolve_effective_params(
    args, syms: list[str], formed_pair: tuple[int, int],
) -> dict:
    """Merge preset + individual-flag overrides into a flat dict.

    Resolution order for each scalar:
        individual flag (not None) > preset value > (r_form only) element table

    Returns keys: reaction_type, k_form, k_broken, r_broken, max_relax_steps, r_form.
    """
    preset = get_preset(args.reaction_type)
    k_form = preset.k_form if args.k_form is None else float(args.k_form)
    k_broken = preset.k_broken if args.k_broken is None else float(args.k_broken)
    r_broken = preset.r_broken if args.r_broken is None else float(args.r_broken)
    max_relax_steps = (
        preset.max_relax_steps if args.max_relax_steps is None
        else int(args.max_relax_steps)
    )
    if args.r_form is not None:
        r_form = float(args.r_form)
    elif preset.r_form is not None:
        r_form = float(preset.r_form)
    else:
        r_form = lookup_r_form(syms[formed_pair[0]], syms[formed_pair[1]])
    return {
        "reaction_type": preset.name,
        "k_form": k_form,
        "k_broken": k_broken,
        "r_broken": r_broken,
        "max_relax_steps": max_relax_steps,
        "r_form": r_form,
    }
```

- [ ] **Step 3.4: テスト pass を確認**

```bash
pytest tests/test_cli.py -v
```

期待: 全 11 tests passed (既存 6 + 新 5)。

- [ ] **Step 3.5: コミット**

```bash
git add reactx/cli.py tests/test_cli.py
git commit -m "feat(cli): add _resolve_effective_params helper for preset/flag merge"
```

---

## Task 4: `_cmd_run` を `_resolve_effective_params` 経由に切り替え + meta.json 拡張

**Files:**
- Modify: `reactx/cli.py:122-329` (`_cmd_run` と `_write_outputs_and_exit`)

- [ ] **Step 4.1: `_cmd_run` 内の現行 r_form / 拘束パラメータ計算を置換**

`reactx/cli.py:163-225` 付近の以下のブロックを書き換える。

**置換前 (line 163-225 の関連部分):**

```python
    formed_pair = bond_changes.formed
    broken_pair = bond_changes.broken
    syms_r = [a.GetSymbol() for a in r_h.GetAtoms()]
    if args.r_form is None:
        r_form_target = lookup_r_form(syms_r[formed_pair[0]], syms_r[formed_pair[1]])
    else:
        r_form_target = float(args.r_form)
```

**置換後:**

```python
    formed_pair = bond_changes.formed
    broken_pair = bond_changes.broken
    syms_r = [a.GetSymbol() for a in r_h.GetAtoms()]
    eff = _resolve_effective_params(args, syms_r, formed_pair)
    r_form_target = eff["r_form"]
    log.info(
        "preset=%s effective: k_form=%.2f k_broken=%.2f r_broken=%.2f "
        "max_relax_steps=%d r_form=%.3f",
        eff["reaction_type"], eff["k_form"], eff["k_broken"],
        eff["r_broken"], eff["max_relax_steps"], eff["r_form"],
    )
```

そして同関数内の `build_restraints(...)` 呼び出し (line 188-196 付近) を:

```python
        restraints = build_restraints(
            atoms_init,
            formed=[formed_pair],
            broken=[broken_pair],
            r_form=r_form_target,
            r_broken=args.r_broken,
            k_form=args.k_form,
            k_broken=args.k_broken,
        )
```

から:

```python
        restraints = build_restraints(
            atoms_init,
            formed=[formed_pair],
            broken=[broken_pair],
            r_form=r_form_target,
            r_broken=eff["r_broken"],
            k_form=eff["k_form"],
            k_broken=eff["k_broken"],
        )
```

`relax_with_restraints(...)` 呼び出し (line 198-203 付近) を:

```python
            frames, energies = relax_with_restraints(
                atoms_init, restraints, calc,
                max_steps=args.max_relax_steps,
                fmax=args.relax_fmax,
                traj_stride=args.traj_stride,
            )
```

から:

```python
            frames, energies = relax_with_restraints(
                atoms_init, restraints, calc,
                max_steps=eff["max_relax_steps"],
                fmax=args.relax_fmax,
                traj_stride=args.traj_stride,
            )
```

`reached_product(...)` 呼び出し (line 212-218 付近) を:

```python
        ok = reached_product(
            frames[-1],
            formed=[formed_pair],
            broken=[broken_pair],
            r_form_targets=[r_form_target],
            r_broken_target=args.r_broken,
        )
```

から:

```python
        ok = reached_product(
            frames[-1],
            formed=[formed_pair],
            broken=[broken_pair],
            r_form_targets=[r_form_target],
            r_broken_target=eff["r_broken"],
        )
```

- [ ] **Step 4.2: `_write_outputs_and_exit` に `effective_params` を渡せるように拡張**

`_write_outputs_and_exit` のシグネチャと meta dict 構築を変更:

**置換前 (line 287-323 付近):**

```python
def _write_outputs_and_exit(
    args: argparse.Namespace,
    trials: list[TrialResult],
    t_start: float,
    *,
    neb_refined: bool,
    rc: int,
) -> int:
    selected = -1
    converged = False
    if rc == 0 and trials:
        try:
            best = score_trials(trials)
            selected = best.trial_idx
            converged = best.reached_product
        except ValueError:
            pass
    meta = {
        "backend": args.backend,
        "converged": converged,
        "selected_trial": selected,
        "trials": [
            ...
        ],
        "wall_clock_seconds": float(time.monotonic() - t_start),
        "neb_refined": neb_refined,
    }
```

**置換後:**

```python
def _write_outputs_and_exit(
    args: argparse.Namespace,
    trials: list[TrialResult],
    t_start: float,
    *,
    neb_refined: bool,
    rc: int,
    effective: dict | None = None,
) -> int:
    selected = -1
    converged = False
    if rc == 0 and trials:
        try:
            best = score_trials(trials)
            selected = best.trial_idx
            converged = best.reached_product
        except ValueError:
            pass
    meta = {
        "backend": args.backend,
        "reaction_type": (effective or {}).get("reaction_type", args.reaction_type),
        "converged": converged,
        "selected_trial": selected,
        "trials": [
            {
                "trial": t.trial_idx,
                "reached_product": t.reached_product,
                "peak_energy": float(t.peak_energy)
                    if math.isfinite(t.peak_energy) else None,
                "n_steps": t.n_steps,
                "rotation_deg": float(t.rotation_deg),
            }
            for t in trials
        ],
        "wall_clock_seconds": float(time.monotonic() - t_start),
        "neb_refined": neb_refined,
        "effective_params": (
            {
                k: effective[k]
                for k in ("k_form", "k_broken", "r_broken", "max_relax_steps", "r_form")
            }
            if effective is not None else None
        ),
    }
    meta_clean = _sanitize_for_json(meta)
    (args.output / "meta.json").write_text(json.dumps(meta_clean, indent=2))

    if rc == 0 and trials:
        best = score_trials(trials)
        (args.output / "energies.json").write_text(json.dumps(best.energies))
    return rc
```

`_cmd_run` 内の `_write_outputs_and_exit` 呼び出し 2 箇所 (early-fail と success path) を `effective=eff` を渡すように更新。**ただし** early-fail の経路 (line 227-229 付近) は `eff` が定義される前に実行される可能性があるため、`eff` の定義を `parse_rxn` 直後に巻き上げる必要がある。

**安全な順序:**

1. `parse_rxn` → `r_h, p_h` を作る
2. `compute_simple_bond_changes` → `bond_changes`
3. `bond_changes_product` 構築
4. `formed_pair`, `broken_pair`, `syms_r` を取り出す ← この時点で `eff = _resolve_effective_params(...)` を計算
5. `make_calculator(...)`
6. trial ループ
7. (どちらの経路でも) `_write_outputs_and_exit(..., effective=eff)`

つまり Step 4.1 で挿入した `eff = _resolve_effective_params(...)` ブロックを、`make_calculator` の前 (line 156 の直前) に移す。`_write_outputs_and_exit` の 2 箇所の呼び出しに `effective=eff` を追加。

- [ ] **Step 4.3: meta.json 拡張のユニットテストを追加**

`tests/test_cli.py` の **import 群 (ファイル冒頭)** に追加:

```python
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from reactx.cli import _write_outputs_and_exit
from reactx.scoring import TrialResult
```

(既存の import と重複する場合はスキップ。)

そして同ファイルの **末尾** に以下のテストを追加:

```python
def test_meta_json_includes_reaction_type_and_effective_params(tmp_path: Path):
    args = argparse.Namespace(
        backend="lj",
        output=tmp_path,
        reaction_type="menshutkin",
    )
    eff = {
        "reaction_type": "menshutkin",
        "k_form": 2.0, "k_broken": 2.0,
        "r_broken": 5.0, "max_relax_steps": 200,
        "r_form": 1.47,
    }
    trials = [TrialResult(
        trial_idx=0, rotation_deg=0.0, frames=[], energies=[1.0, 2.0],
        reached_product=True, peak_energy=2.0, n_steps=2,
    )]
    # Patch score_trials to avoid relying on its internals here.
    with patch("reactx.cli.score_trials", return_value=trials[0]):
        rc = _write_outputs_and_exit(
            args, trials, t_start=0.0,
            neb_refined=False, rc=0, effective=eff,
        )
    assert rc == 0
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["reaction_type"] == "menshutkin"
    assert meta["effective_params"] == {
        "k_form": 2.0, "k_broken": 2.0,
        "r_broken": 5.0, "max_relax_steps": 200,
        "r_form": 1.47,
    }
```

- [ ] **Step 4.4: テスト pass を確認**

```bash
pytest tests/test_cli.py -v
```

期待: 全 12 tests passed (既存 11 + 新 1)。

- [ ] **Step 4.5: 全 fast テストが回帰していないか確認**

```bash
pytest -v
```

期待: 全 fast tests pass (slow / blender マーカーは除外)。

- [ ] **Step 4.6: コミット**

```bash
git add reactx/cli.py tests/test_cli.py
git commit -m "feat(cli): apply preset via effective_params and record in meta.json"
```

---

## Task 5: `examples/menshutkin.rxn` を追加

**Files:**
- Create: `examples/menshutkin.rxn`
- Modify: `tests/conftest.py` (fixture 追加)
- Create: `tests/test_examples_menshutkin.py` (parser smoke test, fast)

- [ ] **Step 5.1: `examples/menshutkin.rxn` を作成**

`$RXN` フォーマット (既存 `examples/proton_transfer.rxn` と同じ V2000) で、NH₃ + CH₃Cl → CH₃NH₃⁺ + Cl⁻ を表現:

- 反応物 1: NH₃ (N + 3H, atom map 1=N, 2-4=H)
- 反応物 2: CH₃Cl (C + 3H + Cl, atom map 5=C, 6-8=H, 9=Cl)
- 生成物 1: CH₃NH₃⁺ (N + C + 6H, atom map 1=N, 2-4=H originally on N + 5=C, 6-8=H originally on C, 全体に +1 charge)
- 生成物 2: Cl⁻ (atom map 9=Cl, charge -1)

`examples/menshutkin.rxn` 内容:

```
$RXN

      RDKit

  2  2
$MOL

     RDKit          2D

  4  3  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  1  0  0
   -0.9000    0.5000    0.0000 H   0  0  0  0  0  0  0  0  0  2  0  0
    0.9000    0.5000    0.0000 H   0  0  0  0  0  0  0  0  0  3  0  0
    0.0000   -1.0000    0.0000 H   0  0  0  0  0  0  0  0  0  4  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
M  END
$MOL

     RDKit          2D

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  5  0  0
   -0.6000    0.8000    0.0000 H   0  0  0  0  0  0  0  0  0  6  0  0
    0.6000    0.8000    0.0000 H   0  0  0  0  0  0  0  0  0  7  0  0
    0.0000   -0.9000    0.0000 H   0  0  0  0  0  0  0  0  0  8  0  0
    1.5000    0.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  9  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
M  END
$MOL

     RDKit          2D

  8  7  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  1  0  0
   -0.9000    0.5000    0.0000 H   0  0  0  0  0  0  0  0  0  2  0  0
    0.9000    0.5000    0.0000 H   0  0  0  0  0  0  0  0  0  3  0  0
    0.0000   -1.0000    0.0000 H   0  0  0  0  0  0  0  0  0  4  0  0
    1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  5  0  0
    2.1000    0.8000    0.0000 H   0  0  0  0  0  0  0  0  0  6  0  0
    2.1000   -0.8000    0.0000 H   0  0  0  0  0  0  0  0  0  7  0  0
    1.5000    1.0000    0.0000 H   0  0  0  0  0  0  0  0  0  8  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
  5  6  1  0
  5  7  1  0
  5  8  1  0
M  CHG  1   1   1
M  END
$MOL

     RDKit          2D

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  9  0  0
M  CHG  1   1  -1
M  END
```

**重要**: 生成物 1 で N (atom map 1) と C (atom map 5) を結合した bond `1  5  1  0` が「形成される結合」を表す。反応物側ではこの結合が無い。同様に反応物 2 の bond `1  5  1  0` (atom map 5=C, 9=Cl) は生成物 2 では Cl が単独原子になり消失=「切断される結合」。

- [ ] **Step 5.2: fixture 追加**

`tests/conftest.py` に追加:

```python
@pytest.fixture()
def menshutkin_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "menshutkin.rxn"
```

- [ ] **Step 5.3: parser smoke test を追加 (fast, UMA 不要)**

`tests/test_examples_menshutkin.py` を新規作成:

```python
"""Smoke test that examples/menshutkin.rxn parses + bond changes are correct."""
from pathlib import Path

from rdkit import Chem

from reactx.bond_changes import compute_simple_bond_changes
from reactx.rxn_parser import parse_rxn


def test_menshutkin_rxn_parses(menshutkin_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(menshutkin_rxn_path)
    # 9 heavy atom mapping: N(1), 3H(2-4), C(5), 3H(6-8), Cl(9)
    assert set(mapping) == {1, 2, 3, 4, 5, 6, 7, 8, 9}


def test_menshutkin_bond_changes_n_c_formed_c_cl_broken(menshutkin_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(menshutkin_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_simple_bond_changes(r_h, p_h, mapping)
    # Symbols on reactant-side at the bond endpoints
    syms = [a.GetSymbol() for a in r_h.GetAtoms()]
    formed_syms = sorted([syms[bc.formed[0]], syms[bc.formed[1]]])
    broken_syms = sorted([syms[bc.broken[0]], syms[bc.broken[1]]])
    assert formed_syms == ["C", "N"]
    assert broken_syms == ["C", "Cl"]
```

- [ ] **Step 5.4: テスト pass を確認**

```bash
pytest tests/test_examples_menshutkin.py -v
```

期待: 2 tests passed。fail する場合は `.rxn` の atom map / bond テーブル / charge を見直す。

- [ ] **Step 5.5: コミット**

```bash
git add examples/menshutkin.rxn tests/conftest.py tests/test_examples_menshutkin.py
git commit -m "feat(examples): add menshutkin.rxn (NH3 + CH3Cl -> CH3NH3+ + Cl-)"
```

---

## Task 6: Menshutkin slow 統合テストを追加

**Files:**
- Create: `tests/test_re1_menshutkin.py`

- [ ] **Step 6.1: テストを書く**

`tests/test_re1_menshutkin.py` 新規作成:

```python
"""End-to-end Menshutkin test (NH3 + CH3Cl -> CH3NH3+ + Cl-).

Slow: requires UMA + GPU (~3 min). Validates the menshutkin preset:
- reached_product=True for at least one trial
- C-N forms (final ≤ 1.7 Å)
- C-Cl breaks (final ≥ 3.5 Å; full r_broken=5.0 may be unreachable due to ion-pair Coulomb attraction)
"""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main


@pytest.mark.slow
def test_re1_menshutkin_end_to_end(tmp_path: Path, menshutkin_rxn_path: Path):
    out = tmp_path / "menshutkin"
    rc = main([
        "run", str(menshutkin_rxn_path), "-o", str(out),
        "--backend", "uma",
        "--reaction-type", "menshutkin",
        "--n-angles", "4",  # smaller for test speed
    ])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["reaction_type"] == "menshutkin"
    assert meta["effective_params"]["k_form"] == 2.0
    assert meta["effective_params"]["k_broken"] == 2.0
    assert meta["effective_params"]["r_broken"] == 5.0
    assert meta["effective_params"]["max_relax_steps"] == 200
    assert meta["selected_trial"] >= 0
    assert any(t["reached_product"] for t in meta["trials"]), (
        f"No trial reached product. trials={meta['trials']}"
    )

    frames = read(str(out / "trajectory.xyz"), index=":")
    syms = frames[0].get_chemical_symbols()
    n_idx = syms.index("N")
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")

    d_nc_first = frames[0].get_distance(n_idx, c_idx)
    d_nc_last = frames[-1].get_distance(n_idx, c_idx)
    d_ccl_first = frames[0].get_distance(c_idx, cl_idx)
    d_ccl_last = frames[-1].get_distance(c_idx, cl_idx)

    assert d_nc_last < 1.7, (
        f"N-C should form: {d_nc_first:.2f} -> {d_nc_last:.2f} (target ≤ 1.7)"
    )
    assert d_ccl_last >= 3.5, (
        f"C-Cl should break: {d_ccl_first:.2f} -> {d_ccl_last:.2f} (target ≥ 3.5)"
    )
```

- [ ] **Step 6.2: 静的検証**

```bash
python -c "import ast; ast.parse(open('tests/test_re1_menshutkin.py').read()); print('OK')"
```

期待: `OK`。

- [ ] **Step 6.3: ローカル GPU + UMA で実走**

```bash
pytest -m slow tests/test_re1_menshutkin.py -v -s
```

期待: pass (~3 分)。fail する場合:
- `meta.json` を見て trials の `reached_product` フラグを確認
- `trajectory.xyz` 末尾フレームで C-N / C-Cl 距離を visualize
- 全滅なら `--n-angles 8` に増やすか、`r_broken` を 5.0 → 4.5 に下げて再試行 (preset の値見直しが必要かもしれない)

- [ ] **Step 6.4: コミット**

```bash
git add tests/test_re1_menshutkin.py
git commit -m "test(re1_menshutkin): end-to-end NH3 + CH3Cl -> ion pair with menshutkin preset"
```

---

## Task 7: `proton_transfer` preset を使うケースを既存 PT slow test に追加

**Files:**
- Modify: `tests/test_re1_proton_transfer.py`

- [ ] **Step 7.1: 新ケースを追加**

`tests/test_re1_proton_transfer.py` の末尾に追加:

```python
@pytest.mark.slow
def test_re1_proton_transfer_via_preset(
    tmp_path: Path, proton_transfer_rxn_path: Path,
):
    """Verify --reaction-type proton_transfer is equivalent to --r-form 1.05."""
    out = tmp_path / "proton_transfer_preset"
    rc = main([
        "run", str(proton_transfer_rxn_path), "-o", str(out),
        "--backend", "uma",
        "--reaction-type", "proton_transfer",
        "--n-angles", "4",
        "--max-relax-steps", "50",  # individual flag overrides preset's 100
    ])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["reaction_type"] == "proton_transfer"
    assert meta["effective_params"]["r_form"] == pytest.approx(1.05)
    assert meta["effective_params"]["max_relax_steps"] == 50  # override won
    assert any(t["reached_product"] for t in meta["trials"])
```

- [ ] **Step 7.2: 静的検証**

```bash
python -c "import ast; ast.parse(open('tests/test_re1_proton_transfer.py').read()); print('OK')"
```

期待: `OK`。

- [ ] **Step 7.3: ローカル UMA で実走 (任意, 既存 PT が動くことが前提)**

```bash
pytest -m slow tests/test_re1_proton_transfer.py -v
```

期待: 2 tests pass (既存 + 新規)。

- [ ] **Step 7.4: コミット**

```bash
git add tests/test_re1_proton_transfer.py
git commit -m "test(re1_pt): cover --reaction-type proton_transfer + flag override"
```

---

## Task 8: README に「Reaction-type presets」節を追加

**Files:**
- Modify: `README.md`

- [ ] **Step 8.1: 新節を追加**

`README.md` の「使い方」節 (line 16-38 付近) の直後、「Phase Re1 動作確認」節 (line 40 付近) の直前に以下を挿入:

```markdown
## Reaction-type presets

`--reaction-type` で反応クラスごとにチューニング済みの拘束パラメータをまとめて適用できる。個別フラグ (`--k-form` / `--k-broken` / `--r-broken` / `--max-relax-steps` / `--r-form`) を併指定するとプリセット値を **常に上書き** する。`--reaction-type` 省略時は `sn2_anion` (現 default 相当)。

| name | k_form | k_broken | r_broken (Å) | max_relax_steps | r_form (Å) | 想定反応 |
|---|---|---|---|---|---|---|
| `sn2_anion` (default) | 0.5 | 1.0 | 4.0 | 100 | 元素表 | 陰イオン求核剤の SN2 (例: O⁻ + CH₃Cl) |
| `proton_transfer` | 0.5 | 1.0 | 4.0 | 100 | 1.05 | 中性間 PT (例: HCl + NH₃) |
| `menshutkin` | 2.0 | 2.0 | 5.0 | 200 | 元素表 | 中性求核剤 → イオン対 (例: NH₃ + CH₃Cl) |

```bash
# SN2 (sn2_anion is the default; the flag is optional)
reactx run examples/sn2.rxn -o out/sn2/ --backend uma --render

# Proton transfer
reactx run examples/proton_transfer.rxn -o out/pt/ \
  --reaction-type proton_transfer --backend uma --render

# Menshutkin (NH3 + CH3Cl -> CH3NH3+ + Cl-)
reactx run examples/menshutkin.rxn -o out/men/ \
  --reaction-type menshutkin --backend uma --render
```

**Menshutkin プリセットの根拠**: 既定値は陰イオン求核剤の外部熱的反応 (SN2 / PT) 用にチューニングされている。中性求核剤 + イオン対生成のような **内部熱的反応** では QM のバリア勾配が default の Hookean に勝って TS 手前で停滞するため、`k_form` / `k_broken` を倍化して引力・斥力を強化し、`r_broken` を 5.0 Å まで引き伸ばし、`max_relax_steps` を 200 に拡大している。

各実行で実際に適用された effective parameters は `out/<rxn>/meta.json` の `effective_params` に記録され、再現性を担保する。
```

- [ ] **Step 8.2: README が壊れていないか軽く確認**

```bash
python -c "from pathlib import Path; t = Path('README.md').read_text(encoding='utf-8'); assert 'Reaction-type presets' in t; assert 'menshutkin' in t; assert '`sn2_anion` (default)' in t; print('OK')"
```

期待: `OK`。

- [ ] **Step 8.3: コミット**

```bash
git add README.md
git commit -m "docs(readme): add Reaction-type presets section + Menshutkin example"
```

---

## Task 9: Memory ファイル削除

**Files:**
- Delete: `C:\Users\daiya\.claude\projects\C--Users-daiya-Workspace-3D-Chemical-Reaction-Mechanism\memory\project_reactx_endothermic_tuning.md`
- Modify: `C:\Users\daiya\.claude\projects\C--Users-daiya-Workspace-3D-Chemical-Reaction-Mechanism\memory\MEMORY.md`

memory ディレクトリは git 管理外のため commit は不要。

- [ ] **Step 9.1: memory ファイルを削除**

```powershell
Remove-Item "C:\Users\daiya\.claude\projects\C--Users-daiya-Workspace-3D-Chemical-Reaction-Mechanism\memory\project_reactx_endothermic_tuning.md"
```

- [ ] **Step 9.2: `MEMORY.md` から該当行を削除**

`memory/MEMORY.md` の以下の 1 行を削除:

```
- [reactx endothermic / ion-pair tuning](project_reactx_endothermic_tuning.md) — Menshutkin 等で必要な k_form/k_broken/r_broken 強化の実測レシピ
```

削除後、`MEMORY.md` は空になるか、他のエントリだけが残る (現状は 1 エントリのみ)。

- [ ] **Step 9.3: 削除確認**

```powershell
Test-Path "C:\Users\daiya\.claude\projects\C--Users-daiya-Workspace-3D-Chemical-Reaction-Mechanism\memory\project_reactx_endothermic_tuning.md"
```

期待: `False`。

```powershell
Get-Content "C:\Users\daiya\.claude\projects\C--Users-daiya-Workspace-3D-Chemical-Reaction-Mechanism\memory\MEMORY.md"
```

期待: 該当行が含まれない。

---

## DoD 最終確認

すべての Task 完了後に以下を実行:

- [ ] **fast suite**

```bash
pytest -v
```

期待: 全 fast tests pass (presets + cli + examples_menshutkin + 既存 fast の合計)。

- [ ] **slow suite (UMA + GPU)**

```bash
pytest -m slow -v
```

期待: `test_re1_sn2`, `test_re1_proton_transfer` (既存 + 新), `test_re1_menshutkin`, `test_neb_refine_sn2`, `test_wallclock_sn2` がすべて pass。

- [ ] **3 反応の実走 + meta.json 確認**

```bash
reactx run examples/sn2.rxn -o /tmp/out/sn2 --backend uma
reactx run examples/proton_transfer.rxn -o /tmp/out/pt --reaction-type proton_transfer --backend uma
reactx run examples/menshutkin.rxn -o /tmp/out/men --reaction-type menshutkin --backend uma
```

各 `meta.json` で `reaction_type` と `effective_params` が出ていること、`selected_trial >= 0` を確認。

- [ ] **README 確認**

`README.md` の「Reaction-type presets」節がレンダリングされ、3 つの CLI 例と Menshutkin の根拠が含まれること。

- [ ] **memory 確認**

`memory/project_reactx_endothermic_tuning.md` が存在せず、`MEMORY.md` の該当行も無いこと。

すべて pass で spec §8 DoD 完了。
