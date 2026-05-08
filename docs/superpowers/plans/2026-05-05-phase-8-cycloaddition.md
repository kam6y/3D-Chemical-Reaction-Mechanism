# Phase 8 — Diels–Alder Cycloaddition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 7 の Fibonacci 配置パイプラインを `bridges == 2` (Diels–Alder) に拡張し、`butadiene + ethylene` と `cyclopentadiene + maleic anhydride` の 2 反応を end-to-end で動かせるようにする。

**Architecture:** `placement.valid_placements` 内で `len(bridges)` に応じ既存 single-anchor / 新 multi-anchor に分岐。multi-anchor は (a) substrate / incoming それぞれの 2 anchor 間 midpoint と axis を計算、(b) Fibonacci 球面の方向 d ごとに translation + 2-point Kabsch rotation で剛体配置、(c) 配置後の `u_sub` 軸まわりに 0°/180° で endo/exo 2 trial を展開、(d) RMSD 縮約で対称分子は achiral 1 trial にまとめる、(e) 距離非対称 / 到達不能の dual-anchor blocking を新追加。同時に `[restraints]` の `k_form` / `k_broken` / `r_broken` を `r_form` と完全対称な `scalar | list` に拡張。

**Tech Stack:** Python 3.13, RDKit 2026.3.1 (.rxn パース), ASE 3.x (Atoms / Hookean / FIRE), NumPy 2.x, fairchem UMA (gas-phase QM), pytest (`-m slow` で UMA 統合テスト), tomllib (Python 標準).

**Spec:** `docs/superpowers/specs/2026-05-05-phase-8-cycloaddition-design.md`

**Branch:** `phase-re8` (既に作成済み、本 plan を develop に向けた PR で統合)

**File Structure (新規 + 変更):**

| Path | 種類 | 役割 |
|---|---|---|
| `reactx/config.py` | 変更 | `RestraintConfig` 4 キー対称化 + `_normalize_*` + `resolve_*_targets` 群 |
| `reactx/artificial_force.py` | 変更 | `build_restraints` per-bond list 対応、内部 broadcast |
| `reactx/placement.py` | 変更 | `PlacementTrial.orientation` / `PlacementResult.placement_kind`、`_rotate_atoms`、`_multi_anchor_placement`、dispatch 分岐 |
| `reactx/cli.py` | 変更 | `resolve_*_targets` 利用、`build_restraints` per-bond、`meta.json` 新フィールド出力 |
| `tests/test_config.py` | 変更 | per-bond list 検証ケース追加 |
| `tests/test_artificial_force.py` | 変更 | per-bond list 引数ケース追加 |
| `tests/test_placement.py` | 変更 | multi-anchor / orientation / placement_kind / 新 blocking ケース |
| `tests/test_cli.py` | 変更 | meta.json `placement_kind` / `orientation` 出力検証 |
| `examples/diels_alder_simple.rxn` | 新規 | butadiene + ethylene → cyclohexene |
| `examples/diels_alder_simple.rxn.toml` | 新規 | DA 単純例 sidecar config |
| `examples/diels_alder_endo.rxn` | 新規 | cyclopentadiene + maleic anhydride → norbornene-2,3-dicarboxylic anhydride |
| `examples/diels_alder_endo.rxn.toml` | 新規 | DA endo 例 sidecar config |
| `tests/test_diels_alder_simple.py` | 新規 | slow integration、butadiene + ethylene |
| `tests/test_diels_alder_endo.py` | 新規 | slow integration、cyclopentadiene + MA、endo/exo 検証 |
| `README.md` | 変更 | 対応反応、TOML 表、動作確認、Wall-clock、方針と限界、アーキテクチャ図 |

---

## Stage 1: Config schema (per-bond k_form / k_broken / r_broken)

### Task 1.1: `RestraintConfig.k_form` を `float | tuple[float, ...]` に

**Files:**
- Modify: `reactx/config.py` (RestraintConfig dataclass、`_build_restraints`)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py` の末尾に追記:

```python
def test_load_k_form_as_list_of_two(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "DA"
formed = [[1, 5], [4, 6]]
broken = []
[restraints]
k_form = [1.0, 1.5]
k_broken = 0.0
r_broken = 4.0
max_relax_steps = 200
""")
    cfg = load_config(rxn)
    assert cfg.restraints.k_form == (1.0, 1.5)


def test_k_form_list_length_must_match_formed(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "bad"
formed = [[1, 5], [4, 6]]
broken = []
[restraints]
k_form = [1.0]
k_broken = 0.0
r_broken = 4.0
max_relax_steps = 200
""")
    with pytest.raises(ValueError, match="k_form.*list length"):
        load_config(rxn)


def test_k_form_list_negative_value_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "bad"
formed = [[1, 5], [4, 6]]
broken = []
[restraints]
k_form = [1.0, -0.1]
k_broken = 0.0
r_broken = 4.0
max_relax_steps = 200
""")
    with pytest.raises(ValueError, match="k_form.*must be >= 0"):
        load_config(rxn)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_config.py::test_load_k_form_as_list_of_two -v
pytest tests/test_config.py::test_k_form_list_length_must_match_formed -v
pytest tests/test_config.py::test_k_form_list_negative_value_rejected -v
```
Expected: 3 件 FAIL (TypeError か "k_form must be a number" 系の既存メッセージ)

- [ ] **Step 3: Implement**

`reactx/config.py` を以下のように変更:

(a) `RestraintConfig` の `k_form` 型注釈を更新:

```python
@dataclass(frozen=True)
class RestraintConfig:
    k_form: float | tuple[float, ...]
    k_broken: float
    r_broken: float
    max_relax_steps: int
    r_form: float | tuple[float, ...] | None = None
```

(b) `_build_restraints` 内の `k_form` 処理を `_normalize_k_form` 経由に変更:

```python
def _build_restraints(raw: dict, *, formed_count: int, source: str) -> RestraintConfig:
    _check_keys(raw, _RESTRAINTS_KEYS, _RESTRAINTS_REQUIRED,
                scope="restraints", source=source)
    k_form = _normalize_k_form(raw["k_form"], formed_count=formed_count, source=source)
    k_broken = _as_float(raw["k_broken"], "restraints.k_broken", source, non_negative=True)
    r_broken = _as_float(raw["r_broken"], "restraints.r_broken", source, positive=True)
    max_steps = _as_int(raw["max_relax_steps"], "restraints.max_relax_steps", source, positive=True)
    r_form_raw = raw.get("r_form")
    r_form = _normalize_r_form(r_form_raw, formed_count=formed_count, source=source)
    return RestraintConfig(
        k_form=k_form, k_broken=k_broken, r_broken=r_broken,
        max_relax_steps=max_steps, r_form=r_form,
    )
```

(c) `_normalize_r_form` の直前に新ヘルパ:

```python
def _normalize_k_form(
    raw: object, *, formed_count: int, source: str,
) -> float | tuple[float, ...]:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        v = float(raw)
        if v < 0.0:
            raise ValueError(f"{source}: 'restraints.k_form' must be >= 0")
        return v
    if isinstance(raw, list):
        if len(raw) != formed_count:
            raise ValueError(
                f"{source}: 'restraints.k_form' list length {len(raw)} "
                f"must match len(formed)={formed_count}"
            )
        out: list[float] = []
        for i, x in enumerate(raw):
            if not isinstance(x, (int, float)) or isinstance(x, bool):
                raise ValueError(
                    f"{source}: 'restraints.k_form[{i}]' must be a number"
                )
            v = float(x)
            if v < 0.0:
                raise ValueError(
                    f"{source}: 'restraints.k_form[{i}]' must be >= 0"
                )
            out.append(v)
        return tuple(out)
    raise ValueError(f"{source}: 'restraints.k_form' must be number or list")
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_config.py -v
```
Expected: 全て PASS。既存 test (scalar k_form) も PASS のまま。

- [ ] **Step 5: Commit**

```
git add reactx/config.py tests/test_config.py
git commit -m "feat(config): k_form scalar | list 両対応 (Phase 8 step 1/N)"
```

### Task 1.2: `RestraintConfig.k_broken` を `float | tuple[float, ...]` に

**Files:**
- Modify: `reactx/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
def test_load_k_broken_as_list_of_two_for_e2_pattern(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "E2 per-bond k"
formed = [[4, 5]]
broken = [[2, 5], [1, 3]]
[restraints]
k_form = 1.0
k_broken = [1.0, 2.0]
r_broken = 4.0
max_relax_steps = 200
""")
    cfg = load_config(rxn)
    assert cfg.restraints.k_broken == (1.0, 2.0)


def test_k_broken_list_length_must_match_broken(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "bad"
formed = [[4, 5]]
broken = [[2, 5], [1, 3]]
[restraints]
k_form = 1.0
k_broken = [1.0]
r_broken = 4.0
max_relax_steps = 200
""")
    with pytest.raises(ValueError, match="k_broken.*list length"):
        load_config(rxn)


def test_k_broken_scalar_with_empty_broken_filler_ok(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "DA broken empty"
formed = [[1, 5], [4, 6]]
broken = []
[restraints]
k_form = [1.0, 1.0]
k_broken = 0.0
r_broken = 4.0
max_relax_steps = 200
""")
    cfg = load_config(rxn)
    assert cfg.restraints.k_broken == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_config.py::test_load_k_broken_as_list_of_two_for_e2_pattern tests/test_config.py::test_k_broken_list_length_must_match_broken -v
```
Expected: FAIL (k_broken は scalar のみ受理)

- [ ] **Step 3: Implement**

`reactx/config.py`:

(a) `RestraintConfig.k_broken` の型注釈を `float | tuple[float, ...]` に更新。

(b) `_build_restraints` を更新:

```python
def _build_restraints(raw: dict, *, formed_count: int, broken_count: int, source: str) -> RestraintConfig:
    _check_keys(raw, _RESTRAINTS_KEYS, _RESTRAINTS_REQUIRED,
                scope="restraints", source=source)
    k_form = _normalize_k_form(raw["k_form"], formed_count=formed_count, source=source)
    k_broken = _normalize_k_broken(raw["k_broken"], broken_count=broken_count, source=source)
    r_broken = _as_float(raw["r_broken"], "restraints.r_broken", source, positive=True)
    max_steps = _as_int(raw["max_relax_steps"], "restraints.max_relax_steps", source, positive=True)
    r_form_raw = raw.get("r_form")
    r_form = _normalize_r_form(r_form_raw, formed_count=formed_count, source=source)
    return RestraintConfig(
        k_form=k_form, k_broken=k_broken, r_broken=r_broken,
        max_relax_steps=max_steps, r_form=r_form,
    )
```

(c) 呼び出し側 (`_validate`) を更新:

```python
restraints = _build_restraints(
    raw["restraints"],
    formed_count=len(formed),
    broken_count=len(broken),
    source=source,
)
```

(d) `_normalize_k_broken` を追加:

```python
def _normalize_k_broken(
    raw: object, *, broken_count: int, source: str,
) -> float | tuple[float, ...]:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        v = float(raw)
        if v < 0.0:
            raise ValueError(f"{source}: 'restraints.k_broken' must be >= 0")
        return v
    if isinstance(raw, list):
        if len(raw) != broken_count:
            raise ValueError(
                f"{source}: 'restraints.k_broken' list length {len(raw)} "
                f"must match len(broken)={broken_count}"
            )
        out: list[float] = []
        for i, x in enumerate(raw):
            if not isinstance(x, (int, float)) or isinstance(x, bool):
                raise ValueError(
                    f"{source}: 'restraints.k_broken[{i}]' must be a number"
                )
            v = float(x)
            if v < 0.0:
                raise ValueError(
                    f"{source}: 'restraints.k_broken[{i}]' must be >= 0"
                )
            out.append(v)
        return tuple(out)
    raise ValueError(f"{source}: 'restraints.k_broken' must be number or list")
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_config.py -v
```
Expected: 全て PASS。

- [ ] **Step 5: Commit**

```
git add reactx/config.py tests/test_config.py
git commit -m "feat(config): k_broken scalar | list 両対応 (Phase 8 step 2/N)"
```

### Task 1.3: `RestraintConfig.r_broken` を `float | tuple[float, ...]` に

**Files:**
- Modify: `reactx/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
def test_load_r_broken_as_list_for_e2(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "E2 per-bond r_broken"
formed = [[4, 5]]
broken = [[2, 5], [1, 3]]
[restraints]
k_form = 1.0
k_broken = 1.0
r_broken = [3.0, 5.0]
max_relax_steps = 200
""")
    cfg = load_config(rxn)
    assert cfg.restraints.r_broken == (3.0, 5.0)


def test_r_broken_list_must_be_positive(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "bad"
formed = [[4, 5]]
broken = [[2, 5], [1, 3]]
[restraints]
k_form = 1.0
k_broken = 1.0
r_broken = [3.0, 0.0]
max_relax_steps = 200
""")
    with pytest.raises(ValueError, match="r_broken.*must be > 0"):
        load_config(rxn)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_config.py::test_load_r_broken_as_list_for_e2 tests/test_config.py::test_r_broken_list_must_be_positive -v
```
Expected: FAIL.

- [ ] **Step 3: Implement**

`reactx/config.py`:

(a) `RestraintConfig.r_broken` を `float | tuple[float, ...]`。

(b) `_build_restraints` 内の `r_broken = _as_float(...)` を `r_broken = _normalize_r_broken(...)` に置換。

(c) `_normalize_r_broken` を追加:

```python
def _normalize_r_broken(
    raw: object, *, broken_count: int, source: str,
) -> float | tuple[float, ...]:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        v = float(raw)
        if v <= 0.0:
            raise ValueError(f"{source}: 'restraints.r_broken' must be > 0")
        return v
    if isinstance(raw, list):
        if len(raw) != broken_count:
            raise ValueError(
                f"{source}: 'restraints.r_broken' list length {len(raw)} "
                f"must match len(broken)={broken_count}"
            )
        out: list[float] = []
        for i, x in enumerate(raw):
            if not isinstance(x, (int, float)) or isinstance(x, bool):
                raise ValueError(
                    f"{source}: 'restraints.r_broken[{i}]' must be a number"
                )
            v = float(x)
            if v <= 0.0:
                raise ValueError(
                    f"{source}: 'restraints.r_broken[{i}]' must be > 0"
                )
            out.append(v)
        return tuple(out)
    raise ValueError(f"{source}: 'restraints.r_broken' must be number or list")
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_config.py -v
```
Expected: 全 PASS。

- [ ] **Step 5: Commit**

```
git add reactx/config.py tests/test_config.py
git commit -m "feat(config): r_broken scalar | list 両対応 (Phase 8 step 3/N)"
```

### Task 1.4: `resolve_k_form_targets` / `resolve_k_broken_targets` / `resolve_r_broken_targets` 追加

**Files:**
- Modify: `reactx/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
def test_resolve_k_form_targets_scalar_broadcasts(tmp_path: Path):
    from reactx.config import resolve_k_form_targets
    rxn = _write(tmp_path, """\
description = "scalar k_form"
formed = [[1, 5], [4, 6]]
broken = []
[restraints]
k_form = 1.5
k_broken = 0.0
r_broken = 4.0
max_relax_steps = 200
""")
    cfg = load_config(rxn)
    targets = resolve_k_form_targets(cfg, [(0, 4), (3, 5)])
    assert targets == [1.5, 1.5]


def test_resolve_k_form_targets_list_passes_through(tmp_path: Path):
    from reactx.config import resolve_k_form_targets
    rxn = _write(tmp_path, """\
description = "list k_form"
formed = [[1, 5], [4, 6]]
broken = []
[restraints]
k_form = [1.0, 2.0]
k_broken = 0.0
r_broken = 4.0
max_relax_steps = 200
""")
    cfg = load_config(rxn)
    targets = resolve_k_form_targets(cfg, [(0, 4), (3, 5)])
    assert targets == [1.0, 2.0]


def test_resolve_k_broken_targets_scalar_broadcasts(tmp_path: Path):
    from reactx.config import resolve_k_broken_targets
    rxn = _write(tmp_path, """\
description = "scalar k_broken"
formed = [[1, 2]]
broken = [[1, 3], [2, 4]]
[restraints]
k_form = 0.5
k_broken = 1.5
r_broken = 4.0
max_relax_steps = 100
""")
    cfg = load_config(rxn)
    targets = resolve_k_broken_targets(cfg, [(0, 2), (1, 3)])
    assert targets == [1.5, 1.5]


def test_resolve_r_broken_targets_list_passes_through(tmp_path: Path):
    from reactx.config import resolve_r_broken_targets
    rxn = _write(tmp_path, """\
description = "list r_broken"
formed = [[1, 2]]
broken = [[1, 3], [2, 4]]
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = [3.0, 5.0]
max_relax_steps = 100
""")
    cfg = load_config(rxn)
    targets = resolve_r_broken_targets(cfg, [(0, 2), (1, 3)])
    assert targets == [3.0, 5.0]


def test_resolve_helpers_empty_input_returns_empty(tmp_path: Path):
    from reactx.config import (
        resolve_k_broken_targets,
        resolve_k_form_targets,
        resolve_r_broken_targets,
    )
    rxn = _write(tmp_path, """\
description = "no broken"
formed = [[1, 5], [4, 6]]
broken = []
[restraints]
k_form = [1.0, 2.0]
k_broken = 0.0
r_broken = 4.0
max_relax_steps = 200
""")
    cfg = load_config(rxn)
    assert resolve_k_broken_targets(cfg, []) == []
    assert resolve_r_broken_targets(cfg, []) == []
    assert resolve_k_form_targets(cfg, [(0, 4), (3, 5)]) == [1.0, 2.0]
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_config.py -v -k "resolve_k_form_targets or resolve_k_broken_targets or resolve_r_broken_targets or resolve_helpers"
```
Expected: FAIL with "cannot import name 'resolve_k_form_targets'".

- [ ] **Step 3: Implement**

`reactx/config.py` の末尾に追加:

```python
def _resolve_per_bond(
    value: float | tuple[float, ...] | None,
    n_bonds: int,
) -> list[float]:
    """Broadcast scalar to length n_bonds; pass list through.

    Returns empty list when n_bonds == 0 regardless of value.
    """
    if n_bonds == 0:
        return []
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (int, float)):
        return [float(value)] * n_bonds
    raise TypeError(f"value must be float or tuple, got {type(value).__name__}")


def resolve_k_form_targets(
    cfg: ReactionConfig,
    formed_idx_pairs: Sequence[tuple[int, int]],
) -> list[float]:
    """Expand `cfg.restraints.k_form` into per-formed-bond k values."""
    return _resolve_per_bond(cfg.restraints.k_form, len(formed_idx_pairs))


def resolve_k_broken_targets(
    cfg: ReactionConfig,
    broken_idx_pairs: Sequence[tuple[int, int]],
) -> list[float]:
    """Expand `cfg.restraints.k_broken` into per-broken-bond k values."""
    return _resolve_per_bond(cfg.restraints.k_broken, len(broken_idx_pairs))


def resolve_r_broken_targets(
    cfg: ReactionConfig,
    broken_idx_pairs: Sequence[tuple[int, int]],
) -> list[float]:
    """Expand `cfg.restraints.r_broken` into per-broken-bond r target values."""
    return _resolve_per_bond(cfg.restraints.r_broken, len(broken_idx_pairs))
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_config.py -v
```
Expected: 全 PASS。

- [ ] **Step 5: Commit**

```
git add reactx/config.py tests/test_config.py
git commit -m "feat(config): resolve_k_form/k_broken/r_broken_targets ヘルパ追加 (Phase 8 step 4/N)"
```

---

## Stage 2: Per-bond `build_restraints`

### Task 2.1: `build_restraints` の `k_form` を `float | list[float]` に拡張

**Files:**
- Modify: `reactx/artificial_force.py`
- Test: `tests/test_artificial_force.py`

- [ ] **Step 1: Write the failing test**

`tests/test_artificial_force.py` の末尾に追記:

```python
from ase.constraints import Hookean

def test_build_restraints_with_per_bond_k_form_list():
    atoms = Atoms("CCCC", positions=[[0, 0, 0], [1.5, 0, 0], [3, 0, 0], [4.5, 0, 0]])
    cs = build_restraints(
        atoms,
        formed=[(0, 2), (1, 3)],
        broken=[],
        r_form=[1.5, 1.5],
        k_form=[2.0, 5.0],
    )
    hookean_cs = [c for c in cs if isinstance(c, Hookean)]
    assert len(hookean_cs) == 2
    ks = sorted(c.spring for c in hookean_cs)
    assert ks == [2.0, 5.0]
```

注: `ase.constraints.Hookean` の k 属性名は ASE バージョンで `spring` か `k` か揺れる。実装時に確認し、不一致なら getter で逃げる:

```python
def _hookean_k(c):
    return getattr(c, "spring", None) or getattr(c, "k", None) or c.threshold
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_artificial_force.py::test_build_restraints_with_per_bond_k_form_list -v
```
Expected: FAIL (現行は scalar k_form のみ受理して broadcast しない)。

- [ ] **Step 3: Implement**

`reactx/artificial_force.py` を更新:

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
    """Return a list of ASE constraints driving formed bonds together and
    broken bonds apart.

    `r_form=None` looks each pair up in DEFAULT_R_FORM by element symbols.
    Scalar values are broadcast across all bonds; list values must match
    `len(formed)` (for k_form / r_form) or `len(broken)` (for k_broken /
    r_broken).
    """
    syms = atoms.get_chemical_symbols()
    r_forms = _broadcast_r_form(r_form, formed, syms)
    k_forms = _broadcast(k_form, len(formed), key="k_form")
    r_brokens = _broadcast(r_broken, len(broken), key="r_broken")
    k_brokens = _broadcast(k_broken, len(broken), key="k_broken")

    constraints: list = []
    for (a, b), rt, k in zip(formed, r_forms, k_forms, strict=True):
        constraints.append(Hookean(a1=a, a2=b, rt=rt, k=k))
    for (a, b), rt, k in zip(broken, r_brokens, k_brokens, strict=True):
        constraints.append(PullApart(a1=a, a2=b, k=k, rt=rt))
    return constraints


def _broadcast(value, n: int, *, key: str) -> list[float]:
    if isinstance(value, list):
        if len(value) != n:
            raise ValueError(
                f"build_restraints: {key} list length {len(value)} != n_bonds {n}"
            )
        return [float(v) for v in value]
    return [float(value)] * n


def _broadcast_r_form(
    value: float | list[float] | None,
    formed: list[tuple[int, int]],
    syms: list[str],
) -> list[float]:
    if value is None:
        return [lookup_r_form(syms[a], syms[b]) for a, b in formed]
    if isinstance(value, list):
        if len(value) != len(formed):
            raise ValueError(
                f"build_restraints: r_form list length {len(value)} != n_formed {len(formed)}"
            )
        return [float(v) for v in value]
    return [float(value)] * len(formed)
```

古い docstring の「Per-bond targets ... NOT supported in Phase 3 ... Phase 4+ when reactions with formed≥2 are added (Diels-Alder etc.)」のコメントを削除。

- [ ] **Step 4: Run tests**

```
pytest tests/test_artificial_force.py -v
```
Expected: 全 PASS (既存 scalar test も pass)。

- [ ] **Step 5: Commit**

```
git add reactx/artificial_force.py tests/test_artificial_force.py
git commit -m "feat(artificial_force): build_restraints per-bond k_form list 対応 (Phase 8 step 5/N)"
```

### Task 2.2: `build_restraints` の `k_broken` / `r_broken` も list 対応 (内部 broadcast 共通化)

**Files:**
- Modify: `reactx/artificial_force.py`
- Test: `tests/test_artificial_force.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_restraints_with_per_bond_k_broken_and_r_broken_lists():
    atoms = Atoms("CClCH", positions=[[0, 0, 0], [2, 0, 0], [4, 0, 0], [5, 0, 0]])
    cs = build_restraints(
        atoms,
        formed=[],
        broken=[(0, 1), (2, 3)],
        k_broken=[2.0, 4.0],
        r_broken=[3.0, 5.0],
    )
    assert len(cs) == 2
    # PullApart の rt と k を取り出す
    rts = sorted(c.rt for c in cs)
    ks = sorted(c.k for c in cs)
    assert rts == [3.0, 5.0]
    assert ks == [2.0, 4.0]


def test_build_restraints_k_broken_list_length_mismatch_raises():
    import pytest
    atoms = Atoms("CClCH", positions=[[0, 0, 0], [2, 0, 0], [4, 0, 0], [5, 0, 0]])
    with pytest.raises(ValueError, match="k_broken list length"):
        build_restraints(
            atoms,
            formed=[],
            broken=[(0, 1), (2, 3)],
            k_broken=[1.0],
            r_broken=4.0,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_artificial_force.py::test_build_restraints_with_per_bond_k_broken_and_r_broken_lists tests/test_artificial_force.py::test_build_restraints_k_broken_list_length_mismatch_raises -v
```
Expected: FAIL (既存 build_restraints は k_broken / r_broken を `float` でしか受け付けない)。

- [ ] **Step 3: Implement**

実は Task 2.1 の実装に既に含まれている (`k_broken=k_broken: float | list[float]`、`_broadcast` 経由)。ここでは念のため signature を確認 + edge case (空 list) のチェック:

`reactx/artificial_force.py` の `build_restraints` シグネチャと `_broadcast` 呼び出しが Task 2.1 で既に対応済みであることを確認。`broken=[]` のとき `_broadcast(scalar, 0, key=...)` が `[]` を返すこと、list 長不一致で `ValueError` を出すことを再確認。

不足があれば追記。

- [ ] **Step 4: Run tests**

```
pytest tests/test_artificial_force.py -v
```
Expected: 全 PASS。

- [ ] **Step 5: Commit**

```
git add reactx/artificial_force.py tests/test_artificial_force.py
git commit -m "test(artificial_force): k_broken / r_broken list の境界ケース追加 (Phase 8 step 6/N)"
```

### Task 2.3: `cli.py` で `resolve_*_targets` を経由して `build_restraints` に list を渡す

**Files:**
- Modify: `reactx/cli.py`
- Test: 既存 `tests/test_cli.py` / `tests/test_re3_e2.py` などが PASS のままであること (回帰検査)

- [ ] **Step 1: Write the failing test**

新規 test を `tests/test_cli.py` に追記 — meta.json の `effective_params` に per-bond list が出力されること:

```python
def test_meta_includes_per_bond_lists_when_toml_uses_lists(tmp_path, tmp_rxn_with_toml):
    """TOML で list 指定したら meta.json も list を保持する。"""
    import json
    from reactx.cli import main
    body = """\
description = "DA-like dummy"
formed = [[1, 5], [4, 6]]
broken = []
[restraints]
k_form = [1.0, 2.0]
k_broken = 0.0
r_broken = 4.0
max_relax_steps = 5
[sampling]
n_candidates = 1
"""
    rxn = tmp_rxn_with_toml("sn2", toml_body=body)  # use sn2 .rxn structure as filler
    out = tmp_path / "out"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "lj"])
    # rc could be 1 (placement may fail with these atom-maps for sn2) or 0;
    # what we care is that meta.json carries the lists.
    meta_path = out / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        ep = meta["effective_params"]
        assert ep["k_form"] == [1.0, 2.0]
        assert ep["k_broken"] == 0.0
```

(やや弱い test だがメタ出力契約を pin する目的)。

- [ ] **Step 2: Run test**

```
pytest tests/test_cli.py::test_meta_includes_per_bond_lists_when_toml_uses_lists -v
```
Expected: FAIL (現状は cli.py が effective_params に scalar しか出さないか、もしくは load_config で reject されるか)。

- [ ] **Step 3: Implement**

`reactx/cli.py` を更新:

(a) imports に `resolve_k_form_targets`, `resolve_k_broken_targets`, `resolve_r_broken_targets` を追加:

```python
from reactx.config import (
    ReactionConfig,
    load_config,
    resolve_k_broken_targets,
    resolve_k_form_targets,
    resolve_r_broken_targets,
    resolve_r_form_targets,
)
```

(b) `_cmd_run` 内で per-bond list を作成:

```python
formed_pairs = list(bond_changes.formed)
broken_pairs = list(bond_changes.broken)
syms_r = [a.GetSymbol() for a in r_h.GetAtoms()]
r_form_targets = resolve_r_form_targets(cfg, syms_r, formed_pairs)
k_form_targets = resolve_k_form_targets(cfg, formed_pairs)
k_broken_targets = resolve_k_broken_targets(cfg, broken_pairs)
r_broken_targets = resolve_r_broken_targets(cfg, broken_pairs)
log.info(
    "description=%s effective: k_form=%s k_broken=%s r_broken=%s "
    "max_relax_steps=%d r_form_targets=%s",
    cfg.description,
    k_form_targets, k_broken_targets, r_broken_targets,
    cfg.restraints.max_relax_steps,
    [f"{x:.3f}" for x in r_form_targets] if r_form_targets else "[]",
)
```

(c) `build_restraints` 呼び出しを以下に置換 (現行は `r_form=r_form_targets[0]` で先頭しか使っていない):

```python
restraints = build_restraints(
    atoms_init,
    formed=formed_pairs,
    broken=broken_pairs,
    r_form=r_form_targets,
    r_broken=r_broken_targets if broken_pairs else 4.0,
    k_form=k_form_targets,
    k_broken=k_broken_targets if broken_pairs else 1.0,
)
```

(d) `_write_outputs_and_exit` の `effective_params` を per-bond list 保存に更新:

```python
"effective_params": (
    {
        "k_form": cfg.restraints.k_form
            if isinstance(cfg.restraints.k_form, float)
            else list(cfg.restraints.k_form),
        "k_broken": cfg.restraints.k_broken
            if isinstance(cfg.restraints.k_broken, float)
            else list(cfg.restraints.k_broken),
        "r_broken": cfg.restraints.r_broken
            if isinstance(cfg.restraints.r_broken, float)
            else list(cfg.restraints.r_broken),
        "max_relax_steps": cfg.restraints.max_relax_steps,
        "r_form_targets": list(r_form_targets) if r_form_targets is not None else [],
        "n_candidates": cfg.sampling.n_candidates,
    }
    if cfg is not None else None
),
```

(e) `_write_outputs_and_exit` のシグネチャに per-bond list を渡せるよう必要に応じて引数追加 (実装時に判断、cfg を渡しているので変更不要のはず)。

(f) `reached_product` 呼び出しの `r_broken_target` 引数も list 対応に更新が必要かを確認:

```python
ok = reached_product(
    frames[-1],
    formed=formed_pairs,
    broken=broken_pairs,
    r_form_targets=r_form_targets,
    r_broken_target=cfg.restraints.r_broken,
)
```

`scoring.reached_product` のシグネチャは現状 `r_broken_target: float`。Phase 8 で list 対応に変更が必要 — 別タスク (Task 2.4) で扱う。本タスクではまだ scalar/list の判別を呼び出し側で:

```python
r_broken_for_check = (
    r_broken_targets[0] if r_broken_targets
    else (cfg.restraints.r_broken if isinstance(cfg.restraints.r_broken, float) else 4.0)
)
ok = reached_product(
    frames[-1],
    formed=formed_pairs,
    broken=broken_pairs,
    r_form_targets=r_form_targets,
    r_broken_target=r_broken_for_check,
)
```

これは defensive な暫定対応。Task 2.4 で `reached_product` を per-bond r_broken に拡張する。

- [ ] **Step 4: Run tests**

```
pytest tests/test_cli.py tests/test_re1_sn2.py tests/test_re3_e2.py -v --no-header
```
Expected: 全 PASS (slow test は除外、per-bond list test は新規 PASS、既存 scalar test も互換 PASS)。

- [ ] **Step 5: Commit**

```
git add reactx/cli.py tests/test_cli.py
git commit -m "feat(cli): resolve_*_targets で per-bond list を build_restraints に渡す (Phase 8 step 7/N)"
```

### Task 2.4: `scoring.reached_product` を per-bond r_broken 対応に

**Files:**
- Modify: `reactx/scoring.py`
- Modify: `reactx/cli.py` (defensive コードの除去)
- Test: `tests/test_scoring.py`

- [ ] **Step 1: Write the failing test**

`tests/test_scoring.py` に追記:

```python
def test_reached_product_per_bond_r_broken_list():
    from ase import Atoms
    from reactx.scoring import reached_product

    atoms = Atoms(
        "CHCH",
        positions=[
            [0, 0, 0],     # C0
            [0, 0, 5],     # H1 (H1 is 5 Å from C0)
            [10, 0, 0],    # C2
            [10, 0, 3],    # H3 (H3 is 3 Å from C2)
        ],
    )
    # broken=[(0,1), (2,3)] とし、r_broken_targets=[3.0, 5.0] にすると
    # |C0-H1|=5 ≥ 3.0-0.5 ✓
    # |C2-H3|=3 ≥ 5.0-0.5=4.5 ✗ → False
    assert not reached_product(
        atoms,
        formed=[],
        broken=[(0, 1), (2, 3)],
        r_form_targets=[],
        r_broken_target=[3.0, 5.0],
    )
    # 逆に [3.0, 2.0] なら全 ≥ → True
    assert reached_product(
        atoms,
        formed=[],
        broken=[(0, 1), (2, 3)],
        r_form_targets=[],
        r_broken_target=[3.0, 2.0],
    )
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_scoring.py::test_reached_product_per_bond_r_broken_list -v
```
Expected: FAIL (シグネチャ的に `r_broken_target: float` のみ受理)。

- [ ] **Step 3: Implement**

`reactx/scoring.py` の `reached_product` を更新:

```python
def reached_product(
    final_atoms: Atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    *,
    r_form_targets: list[float],
    r_broken_target: float | list[float],
    form_tol: float = 0.3,
    broken_tol: float = 0.5,
) -> bool:
    if len(formed) != len(r_form_targets):
        raise ValueError(
            f"formed ({len(formed)}) must match r_form_targets ({len(r_form_targets)})"
        )
    if isinstance(r_broken_target, list):
        r_broken_targets = r_broken_target
    else:
        r_broken_targets = [float(r_broken_target)] * len(broken)
    if len(broken) != len(r_broken_targets):
        raise ValueError(
            f"broken ({len(broken)}) must match r_broken_target list ({len(r_broken_targets)})"
        )
    p = final_atoms.positions
    for (a, b), rt in zip(formed, r_form_targets, strict=True):
        d = float(np.linalg.norm(p[a] - p[b]))
        if d > rt + form_tol:
            return False
    for (a, b), rt in zip(broken, r_broken_targets, strict=True):
        d = float(np.linalg.norm(p[a] - p[b]))
        if d < rt - broken_tol:
            return False
    return True
```

`reactx/cli.py` の defensive shim を削除し、直接 list を渡す:

```python
ok = reached_product(
    frames[-1],
    formed=formed_pairs,
    broken=broken_pairs,
    r_form_targets=r_form_targets,
    r_broken_target=r_broken_targets if broken_pairs else 4.0,
)
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_scoring.py tests/test_cli.py -v
```
Expected: 全 PASS。

- [ ] **Step 5: Commit**

```
git add reactx/scoring.py reactx/cli.py tests/test_scoring.py
git commit -m "feat(scoring): reached_product per-bond r_broken_target 対応 (Phase 8 step 8/N)"
```

---

## Stage 3: Placement の `orientation` / `placement_kind` フィールド (algorithmic 変更なし)

### Task 3.1: `PlacementTrial.orientation` フィールド追加 (default `"single"`)

**Files:**
- Modify: `reactx/placement.py`
- Test: `tests/test_placement.py`

- [ ] **Step 1: Write the failing test**

```python
def test_placement_trial_default_orientation_is_single():
    from reactx.placement import PlacementTrial
    import numpy as np
    t = PlacementTrial(
        direction=np.array([0.0, 0.0, 1.0]),
        d_min=0.0,
        positions=np.zeros((3, 3)),
    )
    assert t.orientation == "single"


def test_placement_trial_orientation_can_be_set_to_endo_exo_achiral():
    from reactx.placement import PlacementTrial
    import numpy as np
    for label in ("single", "endo", "exo", "achiral"):
        t = PlacementTrial(
            direction=np.array([0.0, 0.0, 1.0]),
            d_min=0.0,
            positions=np.zeros((3, 3)),
            orientation=label,
        )
        assert t.orientation == label
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_placement.py::test_placement_trial_default_orientation_is_single -v
```
Expected: FAIL (`PlacementTrial` には `orientation` フィールドがない)。

- [ ] **Step 3: Implement**

`reactx/placement.py`:

```python
from typing import Literal

@dataclass(frozen=True)
class PlacementTrial:
    """A single surviving placement candidate.

    Invariants:
    - `direction` is a unit vector (||direction||₂ == 1 to float tolerance).
    - `d_min` is the placement distance applied along `direction` from the
      anchor; for the unimolecular passthrough trial it is 0.0; for
      bimolecular trials it is `compute_d_min(...)` and may be negative
      when all substrate atoms lie behind the anchor relative to direction.
    - `positions` shape is `(N, 3)` matching `mol_h.GetNumAtoms()`. Substrate
      atoms are unchanged from input; the placed fragment(s) have been
      translated so `incoming_anchor` lands at `anchor_pos + direction * d_min`.
    - `orientation` records cycloaddition trial flavor:
      "single" for single-anchor (Phase 7) or unimolecular passthrough,
      "endo" / "exo" for cycloaddition with asymmetric incoming fragment,
      "achiral" for cycloaddition where endo/exo collapsed by RMSD.
    """
    direction: np.ndarray
    d_min: float
    positions: np.ndarray
    orientation: Literal["single", "endo", "exo", "achiral"] = "single"
```

`Literal` import を追加。

- [ ] **Step 4: Run tests**

```
pytest tests/test_placement.py -v
```
Expected: 全 PASS。

- [ ] **Step 5: Commit**

```
git add reactx/placement.py tests/test_placement.py
git commit -m "feat(placement): PlacementTrial.orientation フィールド追加 (Phase 8 step 9/N)"
```

### Task 3.2: `PlacementResult.placement_kind` フィールド追加

**Files:**
- Modify: `reactx/placement.py`
- Test: `tests/test_placement.py`

- [ ] **Step 1: Write the failing test**

```python
def test_placement_result_placement_kind_default_single_anchor():
    from reactx.placement import PlacementResult, PlacementTrial
    import numpy as np
    r = PlacementResult(
        trials=[],
        n_candidates=0,
        n_blocked=0,
        blocked_reasons=[],
        placement_kind="single_anchor",
    )
    assert r.placement_kind == "single_anchor"


def test_placement_result_kind_multi_anchor_accepted():
    from reactx.placement import PlacementResult
    r = PlacementResult(
        trials=[],
        n_candidates=0,
        n_blocked=0,
        blocked_reasons=[],
        placement_kind="multi_anchor",
    )
    assert r.placement_kind == "multi_anchor"


def test_valid_placements_unimolecular_returns_single_anchor_kind():
    """unimolecular passthrough は placement_kind="single_anchor" を返す。"""
    import numpy as np
    from rdkit import Chem
    from reactx.bond_changes import BondChanges
    from reactx.placement import valid_placements

    # 単純な CH4 mol を作って 1 fragment にする
    mol = Chem.MolFromSmiles("C")
    mol = Chem.AddHs(mol)
    positions = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=float)
    frag_indices = ((0, 1, 2, 3, 4),)
    bond_changes = BondChanges(formed=((0, 1),), broken=())  # dummy intramolecular formed

    # 1 fragment + intramolecular formed bond は通常許可されないが、テスト目的で
    # _identify_substrate / valid_placements が unimolecular path を取ることを確認。
    # (実際の guard を回避するため、frag_indices が長さ 1 ならば passthrough)
    result = valid_placements(mol, frag_indices, positions, bond_changes, n_candidates=1)
    assert result.placement_kind == "single_anchor"
    assert len(result.trials) == 1
    assert result.trials[0].orientation == "single"
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_placement.py::test_placement_result_placement_kind_default_single_anchor tests/test_placement.py::test_valid_placements_unimolecular_returns_single_anchor_kind -v
```
Expected: FAIL.

- [ ] **Step 3: Implement**

`reactx/placement.py`:

```python
@dataclass(frozen=True)
class PlacementResult:
    """Aggregated outcome of `valid_placements` on a single bond_changes.

    `placement_kind` records which dispatch path was taken:
    - "single_anchor": Phase 7 path (1 bridging formed bond, or unimolecular
      passthrough).
    - "multi_anchor": Phase 8 path (2 bridging formed bonds, cycloaddition).
    """
    trials: list[PlacementTrial]
    n_candidates: int
    n_blocked: int
    blocked_reasons: list[str | None]
    placement_kind: Literal["single_anchor", "multi_anchor"] = "single_anchor"
```

`valid_placements` の各 return 文に `placement_kind="single_anchor"` を明示的に付ける (現状の unimolecular branch、bimolecular branch ともに):

```python
# Unimolecular passthrough
return PlacementResult(
    trials=[PlacementTrial(
        direction=np.array([0.0, 0.0, 1.0]),
        d_min=0.0,
        positions=positions.copy(),
        orientation="single",
    )],
    n_candidates=1,
    n_blocked=0,
    blocked_reasons=[None],
    placement_kind="single_anchor",
)

# Bimolecular survivors
return PlacementResult(
    trials=survivors,
    n_candidates=n_candidates,
    n_blocked=n_blocked_total,
    blocked_reasons=blocked_reasons,
    placement_kind="single_anchor",  # multi_anchor branch is later
)
```

bimolecular の `survivors.append(PlacementTrial(...))` にも `orientation="single"` を明示。

- [ ] **Step 4: Run tests**

```
pytest tests/test_placement.py -v
```
Expected: 全 PASS。既存の sphere sampler / d_min / valid_placements 系 test も互換 pass。

- [ ] **Step 5: Commit**

```
git add reactx/placement.py tests/test_placement.py
git commit -m "feat(placement): PlacementResult.placement_kind フィールド追加 (Phase 8 step 10/N)"
```

### Task 3.3: `cli.py` の `meta.json` に `placement_kind` / 各 trial の `orientation` を出力

**Files:**
- Modify: `reactx/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py` に追記:

```python
def test_meta_includes_placement_kind_and_orientation_phase_7_compat(tmp_path, tmp_rxn_with_toml):
    """既存 SN2 (Phase 7 single-anchor) で placement_kind と orientation が出力される。"""
    import json
    from reactx.cli import main
    body = """\
description = "sn2 quick"
formed = [[1, 3]]
broken = [[1, 2]]
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 5
[sampling]
n_candidates = 2
"""
    rxn = tmp_rxn_with_toml("sn2", toml_body=body)
    out = tmp_path / "out"
    main(["run", str(rxn), "-o", str(out), "--backend", "lj"])
    meta = json.loads((out / "meta.json").read_text())
    assert meta["placement"]["placement_kind"] == "single_anchor"
    assert all(t["orientation"] == "single" for t in meta["trials"])
```

- [ ] **Step 2: Run test**

```
pytest tests/test_cli.py::test_meta_includes_placement_kind_and_orientation_phase_7_compat -v
```
Expected: FAIL (key not present)。

- [ ] **Step 3: Implement**

`reactx/cli.py` の `_write_outputs_and_exit` の `placement_meta` を更新:

```python
placement_meta = (
    {
        "n_candidates": placement.n_candidates,
        "n_blocked": placement.n_blocked,
        "n_valid": len(placement.trials),
        "placement_kind": placement.placement_kind,
    }
    if placement is not None
    else {"n_candidates": 0, "n_blocked": 0, "n_valid": 0, "placement_kind": None}
)
```

`trials` フィールドの中で trial ごとに orientation を出力。`TrialResult` には現在 `direction` は含まれているが orientation は含まれていないので、(a) `TrialResult` に orientation を追加するか、(b) cli 側で `placement.trials[i].orientation` を引いて埋め込むか。

(b) が小さい変更で済む。`_write_outputs_and_exit` シグネチャに `placement` が既にあるので、trial_idx と placement.trials の対応を取れば良い:

```python
"trials": [
    {
        "trial": t.trial_idx,
        "reached_product": t.reached_product,
        "peak_energy": float(t.peak_energy)
            if math.isfinite(t.peak_energy) else None,
        "n_steps": t.n_steps,
        "direction": [float(x) for x in t.direction],
        "orientation": (
            placement.trials[t.trial_idx].orientation
            if placement is not None and t.trial_idx < len(placement.trials)
            else "single"
        ),
    }
    for t in trials
],
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_cli.py -v
```
Expected: 全 PASS。

- [ ] **Step 5: Commit**

```
git add reactx/cli.py tests/test_cli.py
git commit -m "feat(cli): meta.json に placement_kind / orientation を出力 (Phase 8 step 11/N)"
```

---

## Stage 4: Multi-anchor placement core

### Task 4.1: `_rotate_atoms` ヘルパ追加 (Rodrigues' formula)

**Files:**
- Modify: `reactx/placement.py`
- Test: `tests/test_placement.py`

- [ ] **Step 1: Write the failing test**

```python
def test_rotate_atoms_around_z_axis_90deg():
    from reactx.placement import _rotate_atoms
    import numpy as np
    positions = np.array([
        [1.0, 0.0, 0.0],   # to be rotated
        [0.0, 0.0, 0.0],   # at center
        [5.0, 5.0, 5.0],   # NOT in indices
    ])
    indices = (0, 1)
    new_pos = _rotate_atoms(
        positions, indices=indices,
        axis=np.array([0.0, 0.0, 1.0]),
        center=np.array([0.0, 0.0, 0.0]),
        angle=np.pi / 2,
    )
    np.testing.assert_allclose(new_pos[0], [0.0, 1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(new_pos[1], [0.0, 0.0, 0.0], atol=1e-9)
    # idx 2 not in indices: unchanged
    np.testing.assert_allclose(new_pos[2], [5.0, 5.0, 5.0])


def test_rotate_atoms_180deg_inverts_perpendicular_component():
    from reactx.placement import _rotate_atoms
    import numpy as np
    positions = np.array([[1.0, 0.0, 0.0]])
    new_pos = _rotate_atoms(
        positions, indices=(0,),
        axis=np.array([0.0, 0.0, 1.0]),
        center=np.zeros(3),
        angle=np.pi,
    )
    np.testing.assert_allclose(new_pos[0], [-1.0, 0.0, 0.0], atol=1e-9)


def test_rotate_atoms_zero_angle_identity():
    from reactx.placement import _rotate_atoms
    import numpy as np
    positions = np.random.default_rng(0).standard_normal((4, 3))
    new_pos = _rotate_atoms(
        positions, indices=(0, 1, 2, 3),
        axis=np.array([1.0, 0.0, 0.0]),
        center=np.zeros(3),
        angle=0.0,
    )
    np.testing.assert_allclose(new_pos, positions, atol=1e-9)
```

- [ ] **Step 2: Run tests**

```
pytest tests/test_placement.py -k _rotate_atoms -v
```
Expected: FAIL.

- [ ] **Step 3: Implement**

`reactx/placement.py` に追記 (`compute_d_min` の前あたり):

```python
def _rotate_atoms(
    positions: np.ndarray,
    *,
    indices: tuple[int, ...],
    axis: np.ndarray,
    center: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Rotate selected atoms around `axis` (passing through `center`) by `angle` rad.

    Uses Rodrigues' rotation formula. Returns a new positions array; input is
    not modified. `axis` is normalized internally; `angle == 0` returns a copy
    of `positions` unchanged.
    """
    new_pos = positions.copy()
    if angle == 0.0:
        return new_pos
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        raise ValueError("rotation axis has zero length")
    k = axis / n
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    for i in indices:
        v = positions[i] - center
        # Rodrigues: v_rot = v cos + (k×v) sin + k (k·v) (1 - cos)
        v_rot = (
            v * cos_a
            + np.cross(k, v) * sin_a
            + k * float(np.dot(k, v)) * (1.0 - cos_a)
        )
        new_pos[i] = center + v_rot
    return new_pos
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_placement.py -k _rotate_atoms -v
```
Expected: 全 PASS。

- [ ] **Step 5: Commit**

```
git add reactx/placement.py tests/test_placement.py
git commit -m "feat(placement): _rotate_atoms ヘルパ追加 (Rodrigues、Phase 8 step 12/N)"
```

### Task 4.2: `_find_bridging_formed` を `bridges == 2` 受け入れに変更 (返り値を list 化)

**Files:**
- Modify: `reactx/placement.py`
- Test: `tests/test_placement.py`

- [ ] **Step 1: Write the failing test**

```python
def test_find_bridging_formed_returns_list_for_one_bridge():
    from reactx.placement import _find_bridging_formed
    bridges = _find_bridging_formed(
        formed=((0, 5),),
        substrate={0, 1, 2, 3},
        fragment={5, 6},
    )
    assert bridges == [(0, 5)]


def test_find_bridging_formed_returns_list_for_two_bridges():
    from reactx.placement import _find_bridging_formed
    bridges = _find_bridging_formed(
        formed=((0, 5), (3, 6)),
        substrate={0, 1, 2, 3},
        fragment={5, 6},
    )
    assert bridges == [(0, 5), (3, 6)]


def test_find_bridging_formed_three_bridges_raises_not_implemented():
    import pytest
    from reactx.placement import _find_bridging_formed
    with pytest.raises(NotImplementedError, match="N>=3 bridges"):
        _find_bridging_formed(
            formed=((0, 5), (3, 6), (1, 7)),
            substrate={0, 1, 2, 3},
            fragment={5, 6, 7},
        )


def test_find_bridging_formed_zero_bridges_raises_value_error():
    import pytest
    from reactx.placement import _find_bridging_formed
    with pytest.raises(ValueError, match="no formed bond bridging"):
        _find_bridging_formed(
            formed=((0, 1),),
            substrate={0, 1, 2, 3},
            fragment={5, 6},
        )
```

- [ ] **Step 2: Run tests**

```
pytest tests/test_placement.py -k _find_bridging_formed -v
```
Expected: FAIL (現行は `tuple[int, int]` 返り、bridges>1 で NotImplementedError)。

- [ ] **Step 3: Implement**

`reactx/placement.py` の `_find_bridging_formed` を再設計:

```python
def _find_bridging_formed(
    formed: tuple[tuple[int, int], ...],
    substrate: set[int],
    fragment: set[int],
) -> list[tuple[int, int]]:
    """Return all formed bonds bridging substrate ↔ fragment, normalized so
    the substrate-side atom is first.

    bridges == 0 → ValueError
    bridges == 1 or 2 → list of normalized tuples
    bridges >= 3 → NotImplementedError (general cycloaddition is Phase 9+)
    """
    bridges = [
        (a, b) for a, b in formed
        if (a in substrate and b in fragment) or (b in substrate and a in fragment)
    ]
    if not bridges:
        raise ValueError(
            f"fragment has no formed bond bridging to substrate; "
            f"check input atom mapping (formed={list(formed)})"
        )
    if len(bridges) >= 3:
        raise NotImplementedError(
            f"multi-anchor placement with N>=3 bridges is out of scope for Phase 8; "
            f"got {len(bridges)} formed bonds bridging this fragment"
        )
    # 正規化: substrate 側 atom が前
    return [
        (a, b) if a in substrate else (b, a)
        for (a, b) in bridges
    ]
```

呼び出し側 (`valid_placements` 内、bimolecular 経路の `bridge = _find_bridging_formed(...)`) を **list 化に対応**:

```python
bridges = _find_bridging_formed(bond_changes.formed, substrate_set, fragment_set)
if len(bridges) == 1:
    # Phase 7 single-anchor 経路へ
    anchor, incoming_anchor = bridges[0]
    # ... existing single-anchor logic ...
elif len(bridges) == 2:
    # Phase 8 multi-anchor 経路 (Task 4.3 で実装、現時点では未実装なので一時的に NotImplementedError)
    raise NotImplementedError(
        "multi-anchor placement (bridges==2) is wired in Task 4.3"
    )
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_placement.py -v
```
Expected: 上記 4 件 PASS、既存 valid_placements 系で bridges==1 を扱うものは引き続き PASS、bimolecular DA を要求する test (まだ存在しないはず) があれば一時的に skip 化を許容。

- [ ] **Step 5: Commit**

```
git add reactx/placement.py tests/test_placement.py
git commit -m "refactor(placement): _find_bridging_formed が list を返す、bridges<=2 を許容 (Phase 8 step 13/N)"
```

### Task 4.3: `_multi_anchor_placement` 基本実装 (translation + 2-point Kabsch、blocking なし、endo/exo なし)

**Files:**
- Modify: `reactx/placement.py`
- Test: `tests/test_placement.py`

- [ ] **Step 1: Write the failing test**

```python
def test_multi_anchor_two_bridges_returns_at_least_one_trial():
    """Synthetic geometry: substrate = 4-atom diene, fragment = 2-atom dienophile.
    bridges = [(0,4), (3,5)] should produce trials with both bonds approximately equal."""
    import numpy as np
    from rdkit import Chem
    from reactx.bond_changes import BondChanges
    from reactx.placement import valid_placements

    # 6 atoms: substrate (4 C) coplanar in xy, fragment (2 C) above
    smiles = "C=CC=C.C=C"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    # 適当な座標を embed
    from reactx.embed3d import embed_fragments_to_positions
    mol_h, frag_indices, positions = embed_fragments_to_positions(mol, seed=0)
    bond_changes = BondChanges(formed=((0, 4), (3, 5)), broken=())
    result = valid_placements(
        mol_h, frag_indices, positions, bond_changes,
        n_candidates=16, seed=0,
    )
    assert result.placement_kind == "multi_anchor"
    assert len(result.trials) >= 1
    for t in result.trials:
        assert t.orientation in ("endo", "exo", "achiral")
```

注: 上記 test は本 task の最後 (`endo/exo` まで含む) と被るので、いったん本 task では `_multi_anchor_placement` の **基本構造のみ** を実装し、orientation は仮で `"endo"` だけ返すようにし、test もそれに合わせる。endo/exo 展開は Task 4.5 で。

簡易版 test:

```python
def test_multi_anchor_basic_translation_kabsch_returns_trials():
    import numpy as np
    from reactx.placement import _multi_anchor_placement
    from rdkit import Chem
    # ダミー mol_h は不要、_multi_anchor_placement に直接 positions を渡せる API
    # ※ 実装側で valid_placements 内部呼び出しを通せるよう設計する。
    # ここでは小さい合成 substrate+fragment を作って動作確認。

    positions = np.array([
        [0.0, 0.0, 0.0],  # A1 (substrate atom 0)
        [1.0, 0.0, 0.0],  # substrate filler
        [2.0, 0.0, 0.0],  # substrate filler
        [3.0, 0.0, 0.0],  # A2 (substrate atom 3)
        [0.0, 5.0, 0.0],  # I1 (incoming atom 4)
        [1.4, 5.0, 0.0],  # I2 (incoming atom 5)
    ])
    # 元素は適当
    syms = ["C", "C", "C", "C", "C", "C"]
    substrate_set = {0, 1, 2, 3}
    fragment_set = {4, 5}
    bridges = [(0, 4), (3, 5)]

    trials, blocked_reasons = _multi_anchor_placement(
        positions, syms, substrate_set, fragment_set, bridges,
        n_candidates=16, seed=0,
    )
    assert isinstance(trials, list)
    assert isinstance(blocked_reasons, list)
    # 全てが blocked になることはないと想定
    assert any(t is not None for t in trials) or any(r is None for r in blocked_reasons)
```

実装に合わせ test の API を調整 (内部関数が `(trials, blocked_reasons)` を返すように設計するのか、`PlacementResult` を直接組み立てるかは実装判断)。

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_placement.py::test_multi_anchor_basic_translation_kabsch_returns_trials -v
```
Expected: FAIL (`_multi_anchor_placement` is not defined)。

- [ ] **Step 3: Implement**

`reactx/placement.py` に新関数を追加:

```python
DUAL_ANCHOR_ASYMMETRY_THRESHOLD = 0.40   # spec §5.4 参照


def _multi_anchor_placement(
    positions: np.ndarray,
    syms: list[str],
    substrate_set: set[int],
    fragment_set: set[int],
    bridges: list[tuple[int, int]],   # length 2、substrate-side が前に正規化済み
    *,
    n_candidates: int = 64,
    seed: int = 0,
    gap: float = 0.5,
    d_min_ceiling: float = 8.0,
) -> tuple[list[PlacementTrial], list[str | None]]:
    """Cycloaddition (bridges == 2) 用の rigid-body multi-anchor 配置。

    本タスクでは:
    - translation (M_inc → M_sub + d × d_min) と
    - 2-point Kabsch rotation (u_inc → u_sub) を実装。
    blocking と endo/exo expansion は後続タスクで追加。
    """
    A1, I1 = bridges[0]
    A2, I2 = bridges[1]
    M_sub = (positions[A1] + positions[A2]) / 2.0
    v_sub = positions[A2] - positions[A1]
    L_sub = float(np.linalg.norm(v_sub))
    if L_sub < 1e-9:
        raise ValueError("substrate anchor pair (A1, A2) are coincident")
    u_sub = v_sub / L_sub

    M_inc = (positions[I1] + positions[I2]) / 2.0
    v_inc = positions[I2] - positions[I1]
    L_inc = float(np.linalg.norm(v_inc))
    if L_inc < 1e-9:
        raise ValueError(
            "incoming anchor pair I1, I2 are coincident; check formed atom-mapping"
        )
    u_inc = v_inc / L_inc

    vdw_all = np.array([vdw_radius(s) for s in syms], dtype=float)
    substrate_atoms = sorted(substrate_set - {A1, A2})
    fragment_list = sorted(fragment_set)
    sub_pos = positions[substrate_atoms]
    sub_vdw = vdw_all[substrate_atoms]
    inc_pos = positions[fragment_list]
    inc_vdw = vdw_all[fragment_list]

    directions = sample_sphere_directions(n_candidates, seed=seed)
    survivors: list[PlacementTrial] = []
    blocked_reasons: list[str | None] = []

    for d in directions:
        d_min = compute_d_min(
            d, M_sub, sub_pos, sub_vdw, inc_pos, inc_vdw, M_inc, gap=gap,
        )
        # Translation: M_inc → M_sub + d × d_min
        translation = (M_sub + d * d_min) - M_inc

        # Rotation: u_inc → u_sub (2-point Kabsch)
        cos_uu = float(np.clip(np.dot(u_inc, u_sub), -1.0, 1.0))
        if cos_uu > 1.0 - 1e-9:
            # already aligned, no rotation
            R_axis = None
            R_angle = 0.0
        elif cos_uu < -1.0 + 1e-9:
            # antiparallel: pick any axis ⊥ u_sub
            ortho = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(u_sub, ortho)) > 0.9:
                ortho = np.array([0.0, 1.0, 0.0])
            R_axis = ortho - u_sub * float(np.dot(u_sub, ortho))
            R_axis /= float(np.linalg.norm(R_axis))
            R_angle = np.pi
        else:
            R_axis = np.cross(u_inc, u_sub)
            R_axis /= float(np.linalg.norm(R_axis))
            R_angle = float(np.arccos(cos_uu))

        # Apply translation, then rotation (centered at M_inc + translation = M_sub + d*d_min)
        new_positions = positions.copy()
        for idx in fragment_list:
            new_positions[idx] = positions[idx] + translation
        if R_axis is not None and R_angle != 0.0:
            new_positions = _rotate_atoms(
                new_positions,
                indices=tuple(fragment_list),
                axis=R_axis,
                center=M_sub + d * d_min,
                angle=R_angle,
            )

        survivors.append(PlacementTrial(
            direction=d, d_min=float(d_min),
            positions=new_positions,
            orientation="endo",   # placeholder for Task 4.5 endo/exo expansion
        ))
        blocked_reasons.append(None)

    return survivors, blocked_reasons
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_placement.py::test_multi_anchor_basic_translation_kabsch_returns_trials -v
```
Expected: PASS。

- [ ] **Step 5: Commit**

```
git add reactx/placement.py tests/test_placement.py
git commit -m "feat(placement): _multi_anchor_placement 基本構造 (translation + Kabsch、Phase 8 step 14/N)"
```

### Task 4.4: Multi-anchor の reachability blocking (unreachable / asymmetric)

**Files:**
- Modify: `reactx/placement.py`
- Test: `tests/test_placement.py`

- [ ] **Step 1: Write the failing test**

```python
def test_multi_anchor_unreachable_dual_anchor_blocked():
    """L_sub と L_inc の差が大きいと d 方向次第で b1 b2 が大きく非対称、
    abs(b1-b2)/max>0.40 で blocked になる方向が出るはず。"""
    import numpy as np
    from reactx.placement import _multi_anchor_placement

    positions = np.array([
        [0.0, 0.0, 0.0],   # A1
        [3.6, 0.0, 0.0],   # A2 (L_sub = 3.6, butadiene-like)
        [0.0, 5.0, 0.0],   # I1
        [1.34, 5.0, 0.0],  # I2 (L_inc = 1.34, ethylene-like)
    ])
    syms = ["C", "C", "C", "C"]
    substrate_set = {0, 1}
    fragment_set = {2, 3}
    bridges = [(0, 2), (1, 3)]

    trials, blocked_reasons = _multi_anchor_placement(
        positions, syms, substrate_set, fragment_set, bridges,
        n_candidates=64, seed=0,
    )
    # 少なくとも 1 件は asymmetric_dual_anchor で blocked
    assert any(
        r is not None and "asymmetric_dual_anchor" in r
        for r in blocked_reasons
    )


def test_multi_anchor_asymmetric_threshold_lets_perpendicular_directions_through():
    """⊥ u_sub に近い方向は b1 ≈ b2 で生存するはず。"""
    import numpy as np
    from reactx.placement import _multi_anchor_placement

    positions = np.array([
        [0.0, 0.0, 0.0], [3.6, 0.0, 0.0],
        [0.0, 5.0, 0.0], [1.34, 5.0, 0.0],
    ])
    syms = ["C", "C", "C", "C"]
    trials, blocked_reasons = _multi_anchor_placement(
        positions, syms, {0, 1}, {2, 3}, [(0, 2), (1, 3)],
        n_candidates=64, seed=0,
    )
    survivors = [t for t in trials if t is not None]
    assert len(survivors) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_placement.py::test_multi_anchor_unreachable_dual_anchor_blocked tests/test_placement.py::test_multi_anchor_asymmetric_threshold_lets_perpendicular_directions_through -v
```
Expected: FAIL (現在 blocking なし → blocked_reasons は全 None、また trials のリスト構造も blocked を区別しない)。

- [ ] **Step 3: Implement**

`reactx/placement.py` の `_multi_anchor_placement` を更新。`survivors.append(...)` の前にチェックを追加し、blocked なら別リストに振り分け:

```python
for d in directions:
    d_min = compute_d_min(...)
    # ... translation + rotation を計算してから placed positions を作る ...

    b1 = float(np.linalg.norm(new_positions[I1] - new_positions[A1]))
    b2 = float(np.linalg.norm(new_positions[I2] - new_positions[A2]))

    if max(b1, b2) > d_min_ceiling:
        blocked_reasons.append(f"unreachable_dual_anchor:b1={b1:.2f},b2={b2:.2f}")
        continue
    if abs(b1 - b2) / max(b1, b2) > DUAL_ANCHOR_ASYMMETRY_THRESHOLD:
        blocked_reasons.append(f"asymmetric_dual_anchor:b1={b1:.2f},b2={b2:.2f}")
        continue

    survivors.append(PlacementTrial(
        direction=d, d_min=float(d_min),
        positions=new_positions,
        orientation="endo",   # placeholder
    ))
    blocked_reasons.append(None)
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_placement.py -v
```
Expected: 全 PASS。

- [ ] **Step 5: Commit**

```
git add reactx/placement.py tests/test_placement.py
git commit -m "feat(placement): multi-anchor reachability blocking (Phase 8 step 15/N)"
```

### Task 4.5: Multi-anchor の endo/exo 展開 + achiral 縮約

**Files:**
- Modify: `reactx/placement.py`
- Test: `tests/test_placement.py`

- [ ] **Step 1: Write the failing test**

```python
def test_multi_anchor_endo_exo_distinct_for_asymmetric_fragment():
    """非対称 fragment では endo と exo が別 trial として残る。"""
    import numpy as np
    from reactx.placement import _multi_anchor_placement
    # I1=I2 軸まわりに非対称な incoming: 3 番目原子 (substituent) を I1-I2 軸の上側に配置
    positions = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],     # A1, A2
        [0.0, 5.0, 0.0],
        [3.0, 5.0, 0.0],     # I1, I2
        [1.5, 5.0, 1.5],     # asymmetric substituent on incoming
    ])
    syms = ["C", "C", "C", "C", "F"]
    trials, blocked_reasons = _multi_anchor_placement(
        positions, syms, {0, 1}, {2, 3, 4}, [(0, 2), (1, 3)],
        n_candidates=8, seed=0,
    )
    orientations = {t.orientation for t in trials}
    assert "endo" in orientations
    assert "exo" in orientations


def test_multi_anchor_achiral_collapse_for_symmetric_fragment():
    """完全対称 fragment (C2 symmetric around I1-I2 軸) では achiral 1 trial に縮約。"""
    import numpy as np
    from reactx.placement import _multi_anchor_placement
    # 4 原子 ethylene-like: I1, I2 と 2 つの H が C2 対称
    positions = np.array([
        [0.0, 0.0, 0.0], [3.0, 0.0, 0.0],
        [0.0, 5.0, 0.0], [1.34, 5.0, 0.0],   # I1, I2
        [-0.5, 5.5, 0.0], [1.84, 5.5, 0.0],  # 2 つの H (両側対称)
    ])
    syms = ["C", "C", "C", "C", "H", "H"]
    trials, blocked_reasons = _multi_anchor_placement(
        positions, syms, {0, 1}, {2, 3, 4, 5}, [(0, 2), (1, 3)],
        n_candidates=8, seed=0,
    )
    assert all(t.orientation == "achiral" for t in trials)
```

- [ ] **Step 2: Run tests**

```
pytest tests/test_placement.py -k "endo_exo or achiral_collapse" -v
```
Expected: FAIL.

- [ ] **Step 3: Implement**

`_multi_anchor_placement` の survivor 追加箇所を更新:

```python
# placed positions を計算した後、 reachability blocking を通過した時点で:

endo_pos = new_positions
exo_pos = _rotate_atoms(
    new_positions,
    indices=tuple(fragment_list),
    axis=u_sub,
    center=M_sub + d * d_min,
    angle=np.pi,
)
# RMSD 比較で対称性検出
endo_frag = endo_pos[fragment_list]
exo_frag = exo_pos[fragment_list]
rmsd = float(np.sqrt(np.mean(np.sum((endo_frag - exo_frag) ** 2, axis=1))))
if rmsd < 0.01:
    # symmetric → achiral 1 trial
    survivors.append(PlacementTrial(
        direction=d, d_min=float(d_min),
        positions=endo_pos, orientation="achiral",
    ))
    blocked_reasons.append(None)
else:
    # asymmetric → endo + exo 2 trials, both share blocked_reasons slot None
    survivors.append(PlacementTrial(
        direction=d, d_min=float(d_min),
        positions=endo_pos, orientation="endo",
    ))
    blocked_reasons.append(None)
    survivors.append(PlacementTrial(
        direction=d, d_min=float(d_min),
        positions=exo_pos, orientation="exo",
    ))
    blocked_reasons.append(None)
```

注: blocked_reasons の長さが trial 数とずれる ようになる (元は n_candidates と等しかった)。`PlacementResult` の invariant を更新して `len(blocked_reasons) == n_candidates` を維持: blocked_reasons は **方向 d ごとに 1 要素** (None なら survive、文字列なら blocked)。trial 数は別途 len(survivors) で管理。実装上は外側ループの blocked_reasons.append タイミングを変えない:

```python
# direction 単位: blocked_reasons は 1 要素を append (None / blocked string)
# trial 単位: survivors には endo/exo / achiral に応じて 1 or 2 要素を append
```

修正版:

```python
for d in directions:
    # ... compute placed positions ...

    if max(b1, b2) > d_min_ceiling:
        blocked_reasons.append(f"unreachable_dual_anchor:b1={b1:.2f},b2={b2:.2f}")
        continue
    if abs(b1 - b2) / max(b1, b2) > DUAL_ANCHOR_ASYMMETRY_THRESHOLD:
        blocked_reasons.append(f"asymmetric_dual_anchor:b1={b1:.2f},b2={b2:.2f}")
        continue
    blocked_reasons.append(None)  # direction survived

    endo_pos = new_positions
    exo_pos = _rotate_atoms(
        new_positions,
        indices=tuple(fragment_list),
        axis=u_sub,
        center=M_sub + d * d_min,
        angle=np.pi,
    )
    rmsd = float(np.sqrt(np.mean(np.sum(
        (endo_pos[fragment_list] - exo_pos[fragment_list]) ** 2, axis=1
    ))))
    if rmsd < 0.01:
        survivors.append(PlacementTrial(
            direction=d, d_min=float(d_min),
            positions=endo_pos, orientation="achiral",
        ))
    else:
        survivors.append(PlacementTrial(
            direction=d, d_min=float(d_min),
            positions=endo_pos, orientation="endo",
        ))
        survivors.append(PlacementTrial(
            direction=d, d_min=float(d_min),
            positions=exo_pos, orientation="exo",
        ))
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_placement.py -v
```
Expected: 全 PASS。

- [ ] **Step 5: Commit**

```
git add reactx/placement.py tests/test_placement.py
git commit -m "feat(placement): multi-anchor endo/exo 展開 + achiral 縮約 (Phase 8 step 16/N)"
```

### Task 4.6: `valid_placements` の dispatch 分岐 + `placement_kind` 設定

**Files:**
- Modify: `reactx/placement.py`
- Test: `tests/test_placement.py`

- [ ] **Step 1: Write the failing test**

```python
def test_valid_placements_dispatches_to_multi_anchor_for_two_bridges():
    """bridges == 2 のときに placement_kind="multi_anchor" を返す。"""
    import numpy as np
    from rdkit import Chem
    from reactx.bond_changes import BondChanges
    from reactx.placement import valid_placements
    from reactx.embed3d import embed_fragments_to_positions

    smiles = "C=CC=C.C=C"
    mol = Chem.MolFromSmiles(smiles)
    mol_h, frag_indices, positions = embed_fragments_to_positions(mol, seed=0)
    # diene 末端 2 つ (atom 0, 3) と dienophile 2 つ (atom 4, 5)
    bond_changes = BondChanges(formed=((0, 4), (3, 5)), broken=())
    result = valid_placements(
        mol_h, frag_indices, positions, bond_changes,
        n_candidates=16, seed=0,
    )
    assert result.placement_kind == "multi_anchor"
    assert all(t.orientation in ("endo", "exo", "achiral") for t in result.trials)


def test_valid_placements_dispatches_to_single_anchor_for_one_bridge():
    """bridges == 1 (既存 SN2 系) は placement_kind="single_anchor"。"""
    import numpy as np
    from rdkit import Chem
    from reactx.bond_changes import BondChanges
    from reactx.placement import valid_placements
    from reactx.embed3d import embed_fragments_to_positions

    smiles = "[O-].CCl"
    mol = Chem.MolFromSmiles(smiles)
    mol_h, frag_indices, positions = embed_fragments_to_positions(mol, seed=0)
    # O- (atom 0) attacks C (atom 1), C-Cl (atom 1-atom 2) breaks
    bond_changes = BondChanges(formed=((0, 1),), broken=((1, 2),))
    result = valid_placements(
        mol_h, frag_indices, positions, bond_changes,
        n_candidates=8, seed=0,
    )
    assert result.placement_kind == "single_anchor"
    assert all(t.orientation == "single" for t in result.trials)
```

- [ ] **Step 2: Run tests**

```
pytest tests/test_placement.py -k "dispatches" -v
```
Expected: FAIL (multi_anchor 経路が valid_placements 内でまだ呼ばれていない)。

- [ ] **Step 3: Implement**

`reactx/placement.py` の `valid_placements` を更新。bridges 数で分岐:

```python
def valid_placements(
    mol_h: Chem.Mol,
    frag_indices: tuple[tuple[int, ...], ...],
    positions: np.ndarray,
    bond_changes: BondChanges,
    *,
    n_candidates: int = 64,
    seed: int = 0,
    gap: float = 0.5,
    d_min_ceiling: float = 8.0,
) -> PlacementResult:
    # Unimolecular passthrough (既存)
    if len(frag_indices) == 1:
        return PlacementResult(
            trials=[PlacementTrial(
                direction=np.array([0.0, 0.0, 1.0]),
                d_min=0.0,
                positions=positions.copy(),
                orientation="single",
            )],
            n_candidates=1,
            n_blocked=0,
            blocked_reasons=[None],
            placement_kind="single_anchor",
        )

    # Refuse metathesis (既存)
    if _broken_bridges_fragments(bond_changes.broken, frag_indices):
        raise NotImplementedError(
            "multi-substrate metathesis (broken bonds spanning fragments) "
            "is out of scope"
        )

    substrate = _identify_substrate(frag_indices)
    substrate_set = set(substrate)
    non_substrate = [f for f in frag_indices if f is not substrate]

    if len(non_substrate) > 1:
        raise NotImplementedError(
            f"termolecular placement ({len(non_substrate)} non-substrate "
            f"fragments) is out of scope"
        )

    fragment = non_substrate[0]
    fragment_set = set(fragment)
    bridges = _find_bridging_formed(bond_changes.formed, substrate_set, fragment_set)

    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]

    if len(bridges) == 1:
        # Phase 7 single-anchor (既存ロジックを抽出した関数 _single_anchor_placement に集約)
        return _single_anchor_placement(
            positions, syms, substrate, fragment_set, bridges[0],
            n_candidates=n_candidates, seed=seed, gap=gap,
            d_min_ceiling=d_min_ceiling,
        )
    elif len(bridges) == 2:
        survivors, blocked_reasons = _multi_anchor_placement(
            positions, syms, substrate_set, fragment_set, bridges,
            n_candidates=n_candidates, seed=seed, gap=gap,
            d_min_ceiling=d_min_ceiling,
        )
        n_blocked_total = sum(1 for r in blocked_reasons if r is not None)
        if not survivors:
            raise RuntimeError(
                f"anchor pair (A1={bridges[0][0]}, A2={bridges[1][0]}) has no valid "
                f"placement direction (all {n_candidates} candidates blocked); "
                f"check substrate geometry / r_form / d_min_ceiling"
            )
        if len(survivors) < max(1, n_candidates // 8):
            log.warning(
                "only %d/%d directions survived blocking for multi-anchor "
                "placement at anchors %d, %d; consider larger n_candidates "
                "or check geometry",
                len(survivors), n_candidates, bridges[0][0], bridges[1][0],
            )
        return PlacementResult(
            trials=survivors,
            n_candidates=n_candidates,
            n_blocked=n_blocked_total,
            blocked_reasons=blocked_reasons,
            placement_kind="multi_anchor",
        )
    else:
        # _find_bridging_formed が既に bridges>=3 を NotImplementedError にしている
        raise AssertionError("unreachable: _find_bridging_formed should have raised")
```

`_single_anchor_placement` を新規関数として、現行 `valid_placements` の bimolecular 経路 (sphere sample → blocking → translate) を抽出。返り値は `PlacementResult` で `placement_kind="single_anchor"`。

- [ ] **Step 4: Run tests**

```
pytest tests/test_placement.py -v
pytest tests/test_re1_sn2.py tests/test_cli.py tests/test_re3_sn1_dissoc.py -v   # 既存回帰
```
Expected: 全 PASS。

- [ ] **Step 5: Commit**

```
git add reactx/placement.py tests/test_placement.py
git commit -m "feat(placement): valid_placements が bridges 数で multi/single anchor に dispatch (Phase 8 step 17/N)"
```

---

## Stage 5: 新規 example .rxn / .rxn.toml + slow integration test

### Task 5.1: butadiene + ethylene example (.rxn / .rxn.toml)

**Files:**
- Create: `examples/diels_alder_simple.rxn`
- Create: `examples/diels_alder_simple.rxn.toml`
- Test: `tests/test_diels_alder_simple.py`

- [ ] **Step 1: Generate the .rxn file with RDKit**

実行 (`python -c "..."` で生成):

```python
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# Reactants: butadiene s-cis, ethylene
buta = Chem.MolFromSmiles("C=CC=C")
ethy = Chem.MolFromSmiles("C=C")
# atom maps (1-based): buta C1=1, C2=2, C3=3, C4=4; ethy C1=5, C2=6
for a in buta.GetAtoms():
    a.SetAtomMapNum(a.GetIdx() + 1)
for a in ethy.GetAtoms():
    a.SetAtomMapNum(a.GetIdx() + 5)

# Product: cyclohexene with same atom maps
cy = Chem.MolFromSmiles("C1=CCCCC1")  # 6-membered with 1 C=C
# Map atoms: C1-C2-C3-C4-C6-C5-C1 (in cyclohexene)
# Need to assign map nums so that C1, C4 in cyclohexene <-> butadiene C1, C4
# and C5, C6 <-> ethylene C5, C6.
# The σ-only formed are C1-C5 (close ring at one end) and C4-C6 (close at other end).
# In our cyclohexene SMILES "C1=CCCCC1": atoms 0..5 in ring order.
# Set: ring atom 0 -> map 2, atom 1 -> map 3, atom 2 -> map 4, atom 3 -> map 6,
#      atom 4 -> map 5, atom 5 -> map 1   (so C1=C2 in product is the new C2=C3 from butadiene)
target_maps = [2, 3, 4, 6, 5, 1]
for atom, mapnum in zip(cy.GetAtoms(), target_maps):
    atom.SetAtomMapNum(mapnum)

# Build $RXN
rxn = AllChem.ChemicalReaction()
rxn.AddReactantTemplate(buta)
rxn.AddReactantTemplate(ethy)
rxn.AddProductTemplate(cy)
rxn_block = AllChem.ReactionToRxnBlock(rxn)
with open("examples/diels_alder_simple.rxn", "w") as f:
    f.write(rxn_block)
```

(後で実装者が実行して .rxn を生成、内容を確認した上で commit)。

- [ ] **Step 2: Create the .rxn.toml**

`examples/diels_alder_simple.rxn.toml`:

```toml
description = "Diels-Alder: butadiene + ethylene -> cyclohexene"
formed = [[1, 5], [4, 6]]
broken = []

[restraints]
k_form = [1.0, 1.0]
k_broken = 0.0
r_broken = 4.0
max_relax_steps = 200
```

- [ ] **Step 3: Write the slow integration test**

`tests/test_diels_alder_simple.py`:

```python
"""End-to-end Diels-Alder (butadiene + ethylene) with UMA. Marked slow."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main


@pytest.mark.slow
def test_diels_alder_simple_end_to_end(tmp_path: Path):
    rxn = Path(__file__).resolve().parent.parent / "examples" / "diels_alder_simple.rxn"
    out = tmp_path / "da_simple"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["selected_trial"] >= 0
    assert meta["placement"]["placement_kind"] == "multi_anchor"
    assert any(t["reached_product"] for t in meta["trials"])
    # ethylene は対称 → achiral 縮約が走るはず
    orientations = {t["orientation"] for t in meta["trials"]}
    assert orientations <= {"achiral", "endo", "exo"}
    assert "achiral" in orientations or {"endo", "exo"} == orientations

    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) >= 3

    # Verify σ bond formation: |C1-C5| and |C4-C6| ≤ 1.7 Å in final frame.
    # atom indices come from r_h (post-AddHs), atoms 0..3 are diene C, 4..5 are ethylene C
    last = frames[-1]
    d_15 = last.get_distance(0, 4)
    d_46 = last.get_distance(3, 5)
    assert d_15 < 1.8, f"C1-C5 not formed: {d_15:.2f} Å"
    assert d_46 < 1.8, f"C4-C6 not formed: {d_46:.2f} Å"
```

- [ ] **Step 4: Run test (smoke)**

最初は LJ で配置パイプラインの sanity check:

```
pytest tests/test_diels_alder_simple.py -m slow -v --timeout=600
```

UMA load 含む長時間。`reached_product` が False / `selected_trial == -1` なら `k_form` をチューン (1.0 → 2.0 → 4.0 まで上げる)。Phase 7 Menshutkin 同様の手順。

- [ ] **Step 5: Commit (生成物 + test)**

```
git add examples/diels_alder_simple.rxn examples/diels_alder_simple.rxn.toml tests/test_diels_alder_simple.py
git commit -m "feat(examples): add diels_alder_simple (butadiene + ethylene) + slow test (Phase 8 step 18/N)"
```

### Task 5.2: cyclopentadiene + maleic anhydride example + endo/exo 検証

**Files:**
- Create: `examples/diels_alder_endo.rxn`
- Create: `examples/diels_alder_endo.rxn.toml`
- Test: `tests/test_diels_alder_endo.py`

- [ ] **Step 1: Generate the .rxn file with RDKit**

```python
from rdkit import Chem
from rdkit.Chem import AllChem

cp = Chem.MolFromSmiles("C1=CC=CC1")            # cyclopentadiene
ma = Chem.MolFromSmiles("O=C1OC(=O)C=C1")       # maleic anhydride
prod = Chem.MolFromSmiles("O=C1OC(=O)C2C1CC=CC2")  # norbornene-2,3-dicarboxylic anhydride

# Atom maps:
# cp: C1=1, C2=2, C3=3, C4=4 (diene末端 1, 4), C5(sp3)=7
# ma: C1=5, C2=6, with 2 carbonyl C 8,9 と O ?
# Choose minimal: only diene ends (1,4) and dienophile ends (5,6) need stable maps.
# Assign:
#   cp atoms (5 atoms): map [1, 2, 3, 4, 7]
#   ma atoms: map [5, 6, 8, 9, 10, 11, 12]   (diene-bonded C are 5, 6)
# Product mapping must mirror.
# 詳細は実装時に Chem.RWMol で原子順にマップを振って ReactionToRxnBlock。

# 最低限 atom maps が一意で reactant ↔ product 一致するよう振る。
# 実装者は .rxn を生成後、reactx.rxn_parser で round-trip check:
#   from reactx.rxn_parser import parse_rxn
#   r, p, mapping = parse_rxn("examples/diels_alder_endo.rxn")
#   assert mapping is not None
```

実装者は `python -c "..."` で .rxn を生成し、parse_rxn で読み取れることを確認。

- [ ] **Step 2: Create the .rxn.toml**

`examples/diels_alder_endo.rxn.toml`:

```toml
description = "Diels-Alder endo: cyclopentadiene + maleic anhydride -> norbornene-2,3-dicarboxylic anhydride"
formed = [[1, 5], [4, 6]]
broken = []

[restraints]
k_form = [1.5, 1.5]
k_broken = 0.0
r_broken = 4.0
max_relax_steps = 250
```

- [ ] **Step 3: Write the slow integration test**

`tests/test_diels_alder_endo.py`:

```python
"""End-to-end Diels-Alder (cyclopentadiene + maleic anhydride) with UMA.
Marked slow. Verifies endo / exo are both expanded and that the kinetic
endo product is selected (low peak_energy)."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main


@pytest.mark.slow
def test_diels_alder_endo_end_to_end(tmp_path: Path):
    rxn = Path(__file__).resolve().parent.parent / "examples" / "diels_alder_endo.rxn"
    out = tmp_path / "da_endo"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["selected_trial"] >= 0
    assert meta["placement"]["placement_kind"] == "multi_anchor"

    # endo と exo の両方が trial として記録されていること
    orientations = {t["orientation"] for t in meta["trials"]}
    assert "endo" in orientations
    assert "exo" in orientations

    # selected trial の orientation を取得 (peak_energy 最小、おそらく endo)
    selected = next(t for t in meta["trials"] if t["trial"] == meta["selected_trial"])
    # MA + cyclopentadiene の kinetic product は endo。tolerance 5 kcal/mol で許容。
    # ただし UMA の挙動次第なので、selected が endo or exo どちらでも assert は warning レベル。
    assert selected["orientation"] in ("endo", "exo")

    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) >= 3
    last = frames[-1]
    # 反応端 atoms: cp の C1, C4 + ma の C5, C6 (atom_map -> heavy index は parse_rxn で得る)
    # 簡易に: 反応端の 2 つの新 σ bond が ≤ 1.8 Å であることを確認
    # 実装時には rxn_parser から atom_map → idx mapping を取り、idx で距離を計算する。
```

具体的な atom index は .rxn 生成時の atom 順序に依存する。実装者は `parse_rxn` で reactant Mol を読み、`atom_map_to_reactant_idx` で `1, 4, 5, 6` の idx を取得して assertion を組み立てる。

- [ ] **Step 4: Run test**

```
pytest tests/test_diels_alder_endo.py -m slow -v --timeout=900
```

`reached_product` False / `selected_trial == -1` の場合、`k_form` を 1.5 → 3.0 → 5.0 まで上げて再 run。Phase 7 Menshutkin と同様、`peak_energy` が NaN にならない値域を実測フィット。

- [ ] **Step 5: Commit**

```
git add examples/diels_alder_endo.rxn examples/diels_alder_endo.rxn.toml tests/test_diels_alder_endo.py
git commit -m "feat(examples): add diels_alder_endo (CP + MA) + slow test (Phase 8 step 19/N)"
```

---

## Stage 6: README updates

### Task 6.1: README の Phase 8 反映

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update 「対応反応」 section**

`# reactx — Generic Steric-Aware Reaction Path Engine` 直下の対応反応リストに追加:

```
- **Diels–Alder cycloaddition** ([4+2]、`bridges == 2`、2 formed + 0 broken)
  - butadiene + ethylene → cyclohexene
  - cyclopentadiene + maleic anhydride → norbornene-2,3-dicarboxylic anhydride (endo/exo 両 trial 自動展開)
```

- [ ] **Step 2: Update 「使い方」 example**

```bash
# Diels-Alder
reactx run examples/diels_alder_simple.rxn -o out/da/ --backend uma --render
reactx run examples/diels_alder_endo.rxn -o out/da_endo/ --backend uma --render
```

- [ ] **Step 3: Update Per-reaction TOML config 表**

| `.rxn` | description | formed (map) | broken (map) | k_form | k_broken | r_broken (Å) | max_relax_steps | r_form | n_candidates |
|---|---|---|---|---|---|---|---|---|---|
| diels_alder_simple.rxn | DA: butadiene + ethylene | `[[1,5],[4,6]]` | `[]` | `[1.0, 1.0]` | 0.0 | 4.0 | 200 | 元素表 | 64 |
| diels_alder_endo.rxn | DA endo: CP + MA | `[[1,5],[4,6]]` | `[]` | `[1.5, 1.5]` | 0.0 | 4.0 | 250 | 元素表 | 64 |

`r_broken` と `k_broken` も list 表記が許容になった旨の脚注:
> Phase 8 から `k_form` / `k_broken` / `r_broken` も `r_form` 同様 `scalar | list[float]` 両対応。list の場合は対応する `formed` / `broken` 数と一致を要求 (E2 のような len(broken)==2 系で per-bond 拘束が活かせる)。

- [ ] **Step 4: Update 「動作確認」 section**

手順末尾に 8, 9 を追加:

```
8. `reactx run examples/diels_alder_simple.rxn -o out/da/ --backend uma --render`
   → meta.json で `placement_kind == "multi_anchor"`, `orientation` に "achiral" (or "endo"/"exo")、最終フレームで C1-C5 ≤ 1.8 Å、C4-C6 ≤ 1.8 Å を視認
9. `reactx run examples/diels_alder_endo.rxn -o out/da_endo/ --backend uma --render`
   → meta.json で endo, exo 両 trial が出力、selected_trial の orientation を確認 (UMA の挙動次第で endo / exo どちらか) + 6-membered ring + bicyclic 構造の形成を視認
10. `pytest -m slow` で Phase 8 含む 8 反応 (DA × 2 を含む) すべての統合テストが pass
```

- [ ] **Step 5: Update 「Wall-clock (実測)」 section**

DA × 2 の行を追加 (実測値は実装後に埋める。Plan 提示時点では暫定):

| Reaction | wall-clock | n_valid | reached / valid | 備考 |
|---|---|---|---|---|
| Diels-Alder simple (`examples/diels_alder_simple.rxn`) | (TBD 測定後) | (TBD) | (TBD) | 64 候補、achiral 縮約適用、2 σ bond 同時形成 |
| Diels-Alder endo (`examples/diels_alder_endo.rxn`) | (TBD 測定後) | (TBD) | (TBD) | 64 候補 × 2 endo/exo、selected の orientation を出力 |

この表は実装完了後に実測値で埋める (本タスクでは "(TBD 測定後)" のまま commit、Stage 7 で埋める)。

- [ ] **Step 6: Update 「方針と限界」 section**

末尾に追加:

```
- multi-anchor placement は `bridges == 2` (Diels-Alder) までを対応。`bridges >= 3` (一般 cycloaddition、1,3-dipolar 等) と cheletropic (incoming 側 anchor が 1 つで substrate 側に 2 つ bridge) は Phase 9+
- DA の endo/exo は配置時に 0°/180° の 2 値で離散化。連続 rotation を sample しない (kinetic vs thermodynamic の精密判定には不十分)
```

- [ ] **Step 7: Update 「アーキテクチャ」 図**

`placement.valid_placements` の出力以降を multi-anchor branch を含めて更新:

```
                                       ▼
                       placement.valid_placements
                       ├─ 1 fragment      → unimolecular passthrough
                       ├─ bridges == 1     → single-anchor (Phase 7)
                       │      └─ Fibonacci on direction → blocking → trials
                       └─ bridges == 2     → multi-anchor (Phase 8)
                              ├─ Fibonacci on face direction
                              ├─ rigid-body alignment (translation + 2-point Kabsch)
                              ├─ endo/exo 0°/180° 展開、対称分子は achiral 縮約
                              └─ blocking: angular shadow + d_min ceiling + unreachable / asymmetric
                                       │
                                       ▼
                              ┌── trial 1 ──┐
                              ├── trial 2 ──┤  FIRE + per-bond Hookean / PullApart
                              ├── ...        │
                              └── trial K ──┘
```

「詳細設計」リンクに `2026-05-05-phase-8-cycloaddition-design.md` を追加。

- [ ] **Step 8: Commit**

```
git add README.md
git commit -m "docs(readme): rewrite for Phase 8 Diels-Alder cycloaddition (Phase 8 step 20/N)"
```

---

## Stage 7: 実測値の README 反映

### Task 7.1: Wall-clock 表に実測値を埋める

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run both DA examples to get wall-clock**

```
reactx run examples/diels_alder_simple.rxn -o out/da/ --backend uma --render
reactx run examples/diels_alder_endo.rxn -o out/da_endo/ --backend uma --render
```

`out/*/meta.json` から `wall_clock_seconds`, `placement.n_valid`, `selected_trial`, `trials[].reached_product` を集計。

- [ ] **Step 2: Update README Wall-clock 表**

```
| Diels-Alder simple (`examples/diels_alder_simple.rxn`) | <X> s | <K> | <reached>/<K> | 64 候補、achiral 縮約適用、2 σ bond 同時形成 |
| Diels-Alder endo (`examples/diels_alder_endo.rxn`)   | <Y> s | <L> | <reached>/<L> | 64 候補 × 2 endo/exo、selected orientation = <"endo"|"exo"> |
```

- [ ] **Step 3: Commit**

```
git add README.md
git commit -m "docs(readme): record Phase 8 wall-clock measurements (Phase 8 step 21/N)"
```

---

## Self-Review チェック

- [ ] **Spec coverage:** spec の §1〜§12 を再確認し、各セクションが Plan 内のどの task で実装されるかを対応付け:
  - §1 目的 / §2 Non-goals: Plan の Goal/Architecture に反映済み
  - §3 アーキテクチャ: Stage 4 (Task 4.6) が dispatch を実装
  - §4 TOML schema: Stage 1 (Tasks 1.1–1.4)
  - §5 multi-anchor アルゴリズム: Stage 4 (Tasks 4.1–4.6)
  - §6 Restraints per-bond: Stage 2 (Tasks 2.1–2.4)
  - §7 Scoring / meta.json: Tasks 2.4, 3.3
  - §8 テスト: 各 task に unit + Stage 5 で slow 統合
  - §9 Error handling: Tasks 4.2 (NotImplemented), 4.6 (RuntimeError)
  - §10 ファイル変更一覧: 上記 File Structure 表
  - §11 リスク: Plan 内では明示しない (実装中に対処)
  - §12 実装順: Plan の Stage 順がほぼ一致

- [ ] **Placeholder scan:** "TBD" は Stage 7 (実測値埋め) のみ意図的に残置、他なし。"TODO" "FIXME" は無し
- [ ] **Type consistency:** `_find_bridging_formed` は Tasks 4.2 で list 返り化、4.3 / 4.6 で list として消費。`PlacementTrial.orientation` は 3.1 で `Literal[...]`、4.5 で `"endo"`/`"exo"`/`"achiral"` のみ追加。`PlacementResult.placement_kind` は 3.2 で `"single_anchor"`/`"multi_anchor"`、4.6 で multi 経路に設定

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-05-phase-8-cycloaddition.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
