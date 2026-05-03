# reactx Phase 6 — `.rxn.toml` Sidecar Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all reaction-class-specific tuning values (`k_form`, `k_broken`, `r_broken`, `max_relax_steps`, `r_form`, `n_angles`, prescreen settings) and the explicit list of `formed`/`broken` atom-map pairs out of CLI flags / `presets.py` and into a per-reaction sidecar TOML at `<rxn_path>.toml`, so the CLI shrinks to environment-and-output-only flags.

**Architecture:** New `reactx/config.py` parses `<rxn>.toml` into a frozen `ReactionConfig` dataclass with Tier A schema validation. `BondChanges.from_atom_map_pairs` does Tier B structural validation against the parsed `.rxn`. `reactx/presets.py` is deleted; `reactx/bond_changes.py` loses `compute_bond_changes` (the auto atom-map diff). `reactx/cli.py` reads the TOML at `_cmd_run` start. Six `examples/*.rxn.toml` files replace the in-code preset table. README and all tests are rewritten to the new CLI shape — **no backwards compatibility**.

**Tech Stack:** Python 3.11+ (`tomllib` stdlib), RDKit, ASE, pytest. Existing pipeline modules (`embed3d`, `path_relax`, `prescreen`, `scoring`, `artificial_force`) are unchanged — only CLI plumbing and config plumbing change.

**Spec:** `docs/superpowers/specs/2026-05-03-rxn-config-sidecar-design.md`

---

## File Structure

**New:**
- `reactx/config.py` — `RestraintConfig`, `SamplingConfig`, `PrescreenConfig`, `ReactionConfig`, `load_config`, `resolve_r_form_targets`
- `tests/test_config.py` — Tier A schema validation
- `examples/sn2.rxn.toml`
- `examples/proton_transfer.rxn.toml`
- `examples/menshutkin.rxn.toml`
- `examples/e2.rxn.toml`
- `examples/sn1_dissoc.rxn.toml`
- `examples/sn1_recomb.rxn.toml`

**Modified:**
- `reactx/bond_changes.py` — `compute_bond_changes` deleted; `from_atom_map_pairs` classmethod added
- `reactx/rxn_parser.py` — add `atom_map_to_reactant_idx(reactant_mol)` helper
- `reactx/cli.py` — drop preset-related and per-parameter flags; call `load_config` at run start
- `tests/test_bond_changes.py` — drop `compute_bond_changes` tests; add `from_atom_map_pairs` tests
- `tests/test_cli.py` — drop `_resolve_effective_params` tests; add new argparse-smoke tests
- `tests/test_cli_neb_refine_guard.py` — drop `--reaction-type`, write `.rxn.toml` for tmp `.rxn`
- `tests/test_cli_unimolecular.py` — same
- `tests/test_examples_menshutkin.py` — replace `compute_bond_changes` use with `BondChanges.from_atom_map_pairs`
- `tests/test_re1_sn2.py`, `test_re1_proton_transfer.py`, `test_re1_menshutkin.py` — drop `--reaction-type` / `--n-angles` / `--max-relax-steps` flags; use `tmp_rxn_with_toml` fixture
- `tests/test_re3_e2.py`, `test_re3_sn1_dissoc.py` — same; replace `meta["reaction_type"]` assertions with `meta["description"]`
- `tests/test_re4_sn1_recomb.py` — same
- `tests/test_neb_refine_sn2.py` — drop CLI flags
- `tests/test_prescreen_disabled_sn2.py` — drop `--no-mmff-prescreen`; set `prescreen.enabled = false` in fixture TOML
- `tests/test_wallclock_sn2.py` — drop CLI flags
- `tests/conftest.py` — add `tmp_rxn_with_toml` fixture
- `README.md` — replace "Reaction-type presets" section with "Per-reaction `.rxn.toml` config"

**Deleted:**
- `reactx/presets.py`
- `tests/test_presets.py`

---

## Task 1: Add `reactx/config.py` with Tier A schema validation

**Files:**
- Create: `reactx/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests for happy-path load**

Create `tests/test_config.py`:

```python
"""Tier A (.rxn.toml schema) validation tests for reactx.config."""
from pathlib import Path

import pytest

from reactx.config import (
    PrescreenConfig,
    ReactionConfig,
    RestraintConfig,
    SamplingConfig,
    load_config,
    resolve_r_form_targets,
)


def _write(tmp_path: Path, body: str) -> Path:
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n")  # body irrelevant for Tier A
    (tmp_path / "x.rxn.toml").write_text(body)
    return rxn


def test_load_minimal_required_keys(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "test"
formed = [[1, 2]]
broken = []
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
""")
    cfg = load_config(rxn)
    assert isinstance(cfg, ReactionConfig)
    assert cfg.description == "test"
    assert cfg.formed == ((1, 2),)
    assert cfg.broken == ()
    assert cfg.restraints == RestraintConfig(
        k_form=0.5, k_broken=1.0, r_broken=4.0,
        max_relax_steps=100, r_form=None,
    )
    assert cfg.sampling == SamplingConfig(n_angles=8, cone_half_deg=30.0)
    assert cfg.prescreen == PrescreenConfig(enabled=True, keep=3, steps=30)


def test_load_full_keys(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "full"
formed = [[1, 5], [2, 6]]
broken = [[3, 4]]
[restraints]
k_form = 1.0
k_broken = 2.0
r_broken = 5.0
max_relax_steps = 200
r_form = [1.78, 1.05]
[sampling]
n_angles = 1
cone_half_deg = 45.0
[prescreen]
enabled = false
keep = 5
steps = 50
""")
    cfg = load_config(rxn)
    assert cfg.formed == ((1, 5), (2, 6))
    assert cfg.broken == ((3, 4),)
    assert cfg.restraints.r_form == (1.78, 1.05)
    assert cfg.sampling.n_angles == 1
    assert cfg.sampling.cone_half_deg == 45.0
    assert cfg.prescreen.enabled is False
    assert cfg.prescreen.keep == 5
    assert cfg.prescreen.steps == 50


def test_load_scalar_r_form(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "pt"
formed = [[1, 3]]
broken = [[1, 2]]
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
r_form = 1.05
""")
    cfg = load_config(rxn)
    assert cfg.restraints.r_form == 1.05
```

- [ ] **Step 2: Run the tests and confirm they fail with `ImportError` (config module missing)**

Run: `pytest tests/test_config.py -x`
Expected: collection error / `ImportError: cannot import name 'load_config' from 'reactx.config'`

- [ ] **Step 3: Implement `reactx/config.py`**

Create `reactx/config.py`:

```python
"""Per-reaction sidecar TOML config (`<rxn_path>.toml`).

Loads and validates the schema described in
`docs/superpowers/specs/2026-05-03-rxn-config-sidecar-design.md` §3.2.

Tier A (this module): schema + range validation, runs without parsing the
.rxn file. Tier B (atom-map presence in the .rxn) lives in
reactx.bond_changes.from_atom_map_pairs.
"""
from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from reactx.artificial_force import lookup_r_form


@dataclass(frozen=True)
class RestraintConfig:
    k_form: float
    k_broken: float
    r_broken: float
    max_relax_steps: int
    r_form: float | tuple[float, ...] | None = None


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
    formed: tuple[tuple[int, int], ...]
    broken: tuple[tuple[int, int], ...]
    restraints: RestraintConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    prescreen: PrescreenConfig = field(default_factory=PrescreenConfig)


_TOP_LEVEL_KEYS = {"description", "formed", "broken", "restraints", "sampling", "prescreen"}
_RESTRAINTS_KEYS = {"k_form", "k_broken", "r_broken", "max_relax_steps", "r_form"}
_RESTRAINTS_REQUIRED = {"k_form", "k_broken", "r_broken", "max_relax_steps"}
_SAMPLING_KEYS = {"n_angles", "cone_half_deg"}
_PRESCREEN_KEYS = {"enabled", "keep", "steps"}
_TOP_LEVEL_REQUIRED = {"description", "formed", "broken", "restraints"}


def sidecar_path(rxn_path: Path) -> Path:
    """`examples/sn2.rxn` -> `examples/sn2.rxn.toml`."""
    return rxn_path.parent / (rxn_path.name + ".toml")


def load_config(rxn_path: Path) -> ReactionConfig:
    """Load `<rxn_path>.toml` and return a validated ReactionConfig.

    Raises:
        FileNotFoundError: when the sidecar TOML is missing.
        ValueError: on schema violations (Tier A).
    """
    toml_path = sidecar_path(rxn_path)
    if not toml_path.is_file():
        raise FileNotFoundError(
            f"sidecar TOML not found: expected {toml_path} alongside {rxn_path}"
        )
    raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    return _validate(raw, source=str(toml_path))


def _validate(raw: dict, *, source: str) -> ReactionConfig:
    _check_keys(raw, _TOP_LEVEL_KEYS, _TOP_LEVEL_REQUIRED, scope="<top>", source=source)

    description = raw["description"]
    if not isinstance(description, str) or not description:
        raise ValueError(f"{source}: 'description' must be a non-empty string")

    formed = _to_pair_tuple(raw["formed"], key="formed", source=source)
    broken = _to_pair_tuple(raw["broken"], key="broken", source=source)
    if not formed and not broken:
        raise ValueError(
            f"{source}: at least one of 'formed' or 'broken' must be non-empty"
        )

    restraints = _build_restraints(raw["restraints"], formed_count=len(formed), source=source)
    sampling = _build_sampling(raw.get("sampling", {}), source=source)
    prescreen = _build_prescreen(raw.get("prescreen", {}), source=source)

    return ReactionConfig(
        description=description,
        formed=formed,
        broken=broken,
        restraints=restraints,
        sampling=sampling,
        prescreen=prescreen,
    )


def _check_keys(
    raw: dict, allowed: set[str], required: set[str], *, scope: str, source: str,
) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            f"{source}: unknown config key(s) under {scope}: "
            f"{sorted(unknown)}; allowed: {sorted(allowed)}"
        )
    missing = required - set(raw)
    if missing:
        raise ValueError(
            f"{source}: missing required key(s) under {scope}: {sorted(missing)}"
        )


def _to_pair_tuple(
    raw: object, *, key: str, source: str,
) -> tuple[tuple[int, int], ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{source}: '{key}' must be a list of [int,int] pairs")
    pairs: list[tuple[int, int]] = []
    for i, p in enumerate(raw):
        if (
            not isinstance(p, list)
            or len(p) != 2
            or not all(isinstance(x, int) and not isinstance(x, bool) for x in p)
        ):
            raise ValueError(
                f"{source}: '{key}[{i}]' must be a [int,int] pair, got {p!r}"
            )
        pairs.append((int(p[0]), int(p[1])))
    return tuple(pairs)


def _build_restraints(raw: dict, *, formed_count: int, source: str) -> RestraintConfig:
    _check_keys(raw, _RESTRAINTS_KEYS, _RESTRAINTS_REQUIRED,
                scope="restraints", source=source)
    k_form = _as_float(raw["k_form"], "restraints.k_form", source, non_negative=True)
    k_broken = _as_float(raw["k_broken"], "restraints.k_broken", source, non_negative=True)
    r_broken = _as_float(raw["r_broken"], "restraints.r_broken", source, positive=True)
    max_steps = _as_int(raw["max_relax_steps"], "restraints.max_relax_steps", source, positive=True)
    r_form_raw = raw.get("r_form", None)
    r_form = _normalize_r_form(r_form_raw, formed_count=formed_count, source=source)
    return RestraintConfig(
        k_form=k_form, k_broken=k_broken, r_broken=r_broken,
        max_relax_steps=max_steps, r_form=r_form,
    )


def _normalize_r_form(
    raw: object, *, formed_count: int, source: str,
) -> float | tuple[float, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        v = float(raw)
        if v <= 0.0:
            raise ValueError(f"{source}: 'restraints.r_form' must be positive")
        return v
    if isinstance(raw, list):
        if len(raw) != formed_count:
            raise ValueError(
                f"{source}: 'restraints.r_form' list length {len(raw)} "
                f"must match len(formed)={formed_count}"
            )
        out: list[float] = []
        for i, x in enumerate(raw):
            if not isinstance(x, (int, float)) or isinstance(x, bool):
                raise ValueError(
                    f"{source}: 'restraints.r_form[{i}]' must be a number"
                )
            v = float(x)
            if v <= 0.0:
                raise ValueError(
                    f"{source}: 'restraints.r_form[{i}]' must be positive"
                )
            out.append(v)
        return tuple(out)
    raise ValueError(f"{source}: 'restraints.r_form' must be number, list, or absent")


def _build_sampling(raw: dict, *, source: str) -> SamplingConfig:
    _check_keys(raw, _SAMPLING_KEYS, set(), scope="sampling", source=source)
    n_angles = _as_int(raw.get("n_angles", 8), "sampling.n_angles", source, positive=True)
    cone = _as_float(raw.get("cone_half_deg", 30.0), "sampling.cone_half_deg",
                     source, positive=True)
    return SamplingConfig(n_angles=n_angles, cone_half_deg=cone)


def _build_prescreen(raw: dict, *, source: str) -> PrescreenConfig:
    _check_keys(raw, _PRESCREEN_KEYS, set(), scope="prescreen", source=source)
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{source}: 'prescreen.enabled' must be bool")
    keep = _as_int(raw.get("keep", 3), "prescreen.keep", source, positive=True)
    steps = _as_int(raw.get("steps", 30), "prescreen.steps", source, positive=True)
    return PrescreenConfig(enabled=enabled, keep=keep, steps=steps)


def _as_float(
    raw: object, key: str, source: str, *,
    non_negative: bool = False, positive: bool = False,
) -> float:
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ValueError(f"{source}: '{key}' must be a number")
    v = float(raw)
    if positive and v <= 0.0:
        raise ValueError(f"{source}: '{key}' must be > 0")
    if non_negative and v < 0.0:
        raise ValueError(f"{source}: '{key}' must be >= 0")
    return v


def _as_int(
    raw: object, key: str, source: str, *, positive: bool = False,
) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError(f"{source}: '{key}' must be an int")
    if positive and raw <= 0:
        raise ValueError(f"{source}: '{key}' must be > 0")
    return int(raw)


def resolve_r_form_targets(
    cfg: ReactionConfig,
    syms: Sequence[str],
    formed_idx_pairs: Sequence[tuple[int, int]],
) -> list[float]:
    """Expand `cfg.restraints.r_form` into per-formed-bond target distances.

    The 0-based heavy-atom indices in `formed_idx_pairs` come from
    `BondChanges.from_atom_map_pairs(...)`, NOT from cfg.formed (which is
    atom-map numbers).
    """
    r = cfg.restraints.r_form
    if r is None:
        return [lookup_r_form(syms[a], syms[b]) for a, b in formed_idx_pairs]
    if isinstance(r, float):
        return [r] * len(formed_idx_pairs)
    return list(r)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/test_config.py -x`
Expected: all 3 tests pass.

- [ ] **Step 5: Add validation tests**

Append to `tests/test_config.py`:

```python
def test_missing_sidecar_raises_filenotfound(tmp_path: Path):
    rxn = tmp_path / "missing.rxn"
    rxn.write_text("$RXN\n")
    with pytest.raises(FileNotFoundError, match="sidecar TOML not found"):
        load_config(rxn)


def test_unknown_top_level_key_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "x"
formed = [[1, 2]]
broken = []
foo = 1
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
""")
    with pytest.raises(ValueError, match="unknown config key.*foo"):
        load_config(rxn)


def test_unknown_restraints_key_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "x"
formed = [[1, 2]]
broken = []
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
kept = 9
""")
    with pytest.raises(ValueError, match="unknown config key.*kept"):
        load_config(rxn)


def test_missing_required_restraint_key_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "x"
formed = [[1, 2]]
broken = []
[restraints]
k_form = 0.5
r_broken = 4.0
max_relax_steps = 100
""")
    with pytest.raises(ValueError, match="missing required.*k_broken"):
        load_config(rxn)


def test_empty_formed_and_broken_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "x"
formed = []
broken = []
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
""")
    with pytest.raises(ValueError, match="at least one of 'formed' or 'broken'"):
        load_config(rxn)


def test_negative_k_form_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "x"
formed = [[1, 2]]
broken = []
[restraints]
k_form = -0.1
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
""")
    with pytest.raises(ValueError, match="k_form.*>= 0"):
        load_config(rxn)


def test_zero_n_angles_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "x"
formed = [[1, 2]]
broken = []
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
[sampling]
n_angles = 0
""")
    with pytest.raises(ValueError, match="n_angles.*> 0"):
        load_config(rxn)


def test_r_form_list_length_mismatch_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "x"
formed = [[1, 2]]
broken = []
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
r_form = [1.0, 2.0]
""")
    with pytest.raises(ValueError, match="r_form.*list length 2.*formed.*=1"):
        load_config(rxn)


def test_formed_pair_must_be_two_ints(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "x"
formed = [[1, 2, 3]]
broken = []
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
""")
    with pytest.raises(ValueError, match=r"formed\[0\].*\[int,int\]"):
        load_config(rxn)


def test_resolve_r_form_targets_lookup_when_none():
    cfg = ReactionConfig(
        description="x",
        formed=((1, 2),),
        broken=(),
        restraints=RestraintConfig(
            k_form=0.5, k_broken=1.0, r_broken=4.0,
            max_relax_steps=100, r_form=None,
        ),
    )
    syms = ["C", "O"]
    out = resolve_r_form_targets(cfg, syms, [(0, 1)])
    assert out == [pytest.approx(1.43)]  # element-table C-O


def test_resolve_r_form_targets_scalar_broadcast():
    cfg = ReactionConfig(
        description="x",
        formed=((1, 2), (3, 4)),
        broken=(),
        restraints=RestraintConfig(
            k_form=0.5, k_broken=1.0, r_broken=4.0,
            max_relax_steps=100, r_form=1.10,
        ),
    )
    out = resolve_r_form_targets(cfg, ["C", "C", "C", "C"], [(0, 1), (2, 3)])
    assert out == [1.10, 1.10]


def test_resolve_r_form_targets_list_passthrough():
    cfg = ReactionConfig(
        description="x",
        formed=((1, 2), (3, 4)),
        broken=(),
        restraints=RestraintConfig(
            k_form=0.5, k_broken=1.0, r_broken=4.0,
            max_relax_steps=100, r_form=(1.78, 1.05),
        ),
    )
    out = resolve_r_form_targets(cfg, ["C", "C", "C", "C"], [(0, 1), (2, 3)])
    assert out == [1.78, 1.05]


def test_resolve_r_form_targets_empty_formed():
    cfg = ReactionConfig(
        description="x",
        formed=(),
        broken=((1, 2),),
        restraints=RestraintConfig(
            k_form=0.5, k_broken=1.0, r_broken=4.0,
            max_relax_steps=100, r_form=None,
        ),
    )
    assert resolve_r_form_targets(cfg, ["C", "Br"], []) == []
```

- [ ] **Step 6: Run all of `test_config.py` and confirm they pass**

Run: `pytest tests/test_config.py -x`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add reactx/config.py tests/test_config.py
git commit -m "feat(config): add reactx.config.load_config + ReactionConfig (Tier A schema)"
```

---

## Task 2: Replace `compute_bond_changes` with `BondChanges.from_atom_map_pairs` (Tier B)

**Files:**
- Modify: `reactx/bond_changes.py`
- Modify: `reactx/rxn_parser.py`
- Modify: `tests/test_bond_changes.py`
- Modify: `tests/test_examples_menshutkin.py`

- [ ] **Step 1: Add `atom_map_to_reactant_idx` helper to `reactx/rxn_parser.py`**

Append to `reactx/rxn_parser.py` (after `heavy_to_hydrogen_groups`):

```python
def atom_map_to_reactant_idx(reactant_mol: Chem.Mol) -> dict[int, int]:
    """Return {atom_map_number: 0-based atom index} for the (pre-AddHs) reactant.

    The mapping is identical for the post-AddHs Mol because Chem.AddHs appends
    new H atoms at indices >= original count, preserving every existing atom's
    index. Use this dict to translate TOML-side atom-map pairs into 0-based
    indices that BondChanges, embed3d, and build_restraints expect.
    """
    return {
        a.GetAtomMapNum(): a.GetIdx()
        for a in reactant_mol.GetAtoms()
        if a.GetAtomMapNum()
    }
```

- [ ] **Step 2: Replace `compute_bond_changes` with `from_atom_map_pairs` in `reactx/bond_changes.py`**

Edit `reactx/bond_changes.py` — delete `compute_bond_changes`, `_bond_set_in_self_idx`, `_ordered`, `_build_full_atom_mapping`. Replace the file body (keep the `BondChanges` dataclass) with:

```python
"""BondChanges container + atom-map -> 0-based index converter."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BondChanges:
    """Multi-bond elementary step: tuples of formed and broken bonds.

    Atom indices are 0-based heavy-atom indices in the reactant Mol's ordering
    (which matches the post-AddHs Mol because Chem.AddHs preserves indices).
    Tuples (not lists) for frozen-dataclass hashability.
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
                    raise ValueError(
                        f"{label} contains duplicate bond {(a, b)} "
                        f"(canonical form {key} already seen)"
                    )
                seen.add(key)
        if len(self.formed) + len(self.broken) == 0:
            raise ValueError("BondChanges must have at least one formed or broken bond")

        formed_canonical = {
            (a, b) if a <= b else (b, a) for a, b in self.formed
        }
        broken_canonical = {
            (a, b) if a <= b else (b, a) for a, b in self.broken
        }
        overlap = formed_canonical & broken_canonical
        if overlap:
            raise ValueError(
                f"formed and broken collide on bond(s) {sorted(overlap)}"
            )

    @classmethod
    def from_atom_map_pairs(
        cls,
        formed_map: Sequence[tuple[int, int]],
        broken_map: Sequence[tuple[int, int]],
        atom_map_to_idx: dict[int, int],
    ) -> "BondChanges":
        """Build a BondChanges from TOML atom-map pairs and a reactant lookup.

        Raises:
            KeyError: when an atom-map number is not present in atom_map_to_idx.
        """
        return cls(
            formed=tuple(_translate(p, atom_map_to_idx, "formed") for p in formed_map),
            broken=tuple(_translate(p, atom_map_to_idx, "broken") for p in broken_map),
        )


def _translate(
    pair: tuple[int, int],
    atom_map_to_idx: dict[int, int],
    label: str,
) -> tuple[int, int]:
    a, b = pair
    if a not in atom_map_to_idx:
        raise KeyError(f"{label}: unknown atom-map number {a}")
    if b not in atom_map_to_idx:
        raise KeyError(f"{label}: unknown atom-map number {b}")
    return (atom_map_to_idx[a], atom_map_to_idx[b])
```

- [ ] **Step 3: Rewrite `tests/test_bond_changes.py`**

Replace the file:

```python
"""Unit tests for BondChanges + from_atom_map_pairs (Tier B)."""
import pytest

from reactx.bond_changes import BondChanges


def test_self_loop_rejected():
    with pytest.raises(ValueError, match="self-loop"):
        BondChanges(formed=((0, 0),), broken=((1, 2),))


def test_duplicate_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        BondChanges(formed=((0, 1), (1, 0)), broken=())


def test_empty_total_rejected():
    with pytest.raises(ValueError, match="at least one"):
        BondChanges(formed=(), broken=())


def test_formed_broken_overlap_rejected():
    with pytest.raises(ValueError, match="formed and broken collide"):
        BondChanges(formed=((0, 1),), broken=((1, 0),))


def test_multi_bond_topologies_accepted():
    bc = BondChanges(formed=((4, 5),), broken=((0, 2), (1, 4)))
    assert len(bc.formed) == 1
    assert len(bc.broken) == 2
    bc = BondChanges(formed=(), broken=((0, 1),))
    assert len(bc.formed) == 0
    assert len(bc.broken) == 1


def test_from_atom_map_pairs_translates_indices():
    # atom-map 1->idx 0, 5->idx 4
    m2i = {1: 0, 5: 4}
    bc = BondChanges.from_atom_map_pairs(
        formed_map=[(1, 5)],
        broken_map=[],
        atom_map_to_idx=m2i,
    )
    assert bc.formed == ((0, 4),)
    assert bc.broken == ()


def test_from_atom_map_pairs_unknown_map_number_raises_key_error():
    m2i = {1: 0, 2: 1}
    with pytest.raises(KeyError, match="unknown atom-map number 99"):
        BondChanges.from_atom_map_pairs(
            formed_map=[(1, 99)],
            broken_map=[],
            atom_map_to_idx=m2i,
        )


def test_from_atom_map_pairs_overlap_via_translation_raises():
    """formed=(1,5) broken=(5,1) collapse to the same canonical pair after translation."""
    m2i = {1: 0, 5: 4}
    with pytest.raises(ValueError, match="formed and broken collide"):
        BondChanges.from_atom_map_pairs(
            formed_map=[(1, 5)],
            broken_map=[(5, 1)],
            atom_map_to_idx=m2i,
        )
```

- [ ] **Step 4: Run the bond-changes tests; expect all pass**

Run: `pytest tests/test_bond_changes.py -x`
Expected: all 8 tests pass.

- [ ] **Step 5: Rewrite `tests/test_examples_menshutkin.py`** (drops `compute_bond_changes`)

Replace the file:

```python
"""Smoke test that examples/menshutkin.rxn parses + atom map covers all atoms."""
from pathlib import Path

from reactx.bond_changes import BondChanges
from reactx.rxn_parser import atom_map_to_reactant_idx, parse_rxn


def test_menshutkin_rxn_parses(menshutkin_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(menshutkin_rxn_path)
    assert len(mapping) == 9
    assert r_mol.GetNumAtoms() == 9
    assert p_mol.GetNumAtoms() == 9
    r_map_nums = {a.GetAtomMapNum() for a in r_mol.GetAtoms()}
    p_map_nums = {a.GetAtomMapNum() for a in p_mol.GetAtoms()}
    assert r_map_nums == {1, 2, 3, 4, 5, 6, 7, 8, 9}
    assert p_map_nums == {1, 2, 3, 4, 5, 6, 7, 8, 9}


def test_menshutkin_n_c_formed_c_cl_broken_via_toml_pairs(menshutkin_rxn_path: Path):
    """TOML claims formed=[[1,5]] broken=[[5,9]] — translating gives the right symbols."""
    r_mol, _, _ = parse_rxn(menshutkin_rxn_path)
    m2i = atom_map_to_reactant_idx(r_mol)
    bc = BondChanges.from_atom_map_pairs(
        formed_map=[(1, 5)],
        broken_map=[(5, 9)],
        atom_map_to_idx=m2i,
    )
    syms = [a.GetSymbol() for a in r_mol.GetAtoms()]
    assert sorted([syms[bc.formed[0][0]], syms[bc.formed[0][1]]]) == ["C", "N"]
    assert sorted([syms[bc.broken[0][0]], syms[bc.broken[0][1]]]) == ["C", "Cl"]
```

- [ ] **Step 6: Run the menshutkin smoke test**

Run: `pytest tests/test_examples_menshutkin.py -x`
Expected: 2 tests pass.

- [ ] **Step 7: Commit**

```bash
git add reactx/bond_changes.py reactx/rxn_parser.py tests/test_bond_changes.py tests/test_examples_menshutkin.py
git commit -m "refactor(bond_changes): replace auto-diff with TOML atom-map translation (Tier B)"
```

---

## Task 3: Add `examples/*.rxn.toml` for all 6 reactions

**Files:**
- Create: `examples/sn2.rxn.toml`
- Create: `examples/proton_transfer.rxn.toml`
- Create: `examples/menshutkin.rxn.toml`
- Create: `examples/e2.rxn.toml`
- Create: `examples/sn1_dissoc.rxn.toml`
- Create: `examples/sn1_recomb.rxn.toml`

- [ ] **Step 1: Create `examples/sn2.rxn.toml`**

```toml
description = "SN2 anion: CH3Cl + OH- -> CH3OH + Cl-"
formed = [[1, 3]]
broken = [[1, 2]]

[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
```

- [ ] **Step 2: Create `examples/proton_transfer.rxn.toml`**

```toml
description = "Proton transfer: HCl + NH3 -> Cl- + NH4+"
formed = [[1, 3]]
broken = [[1, 2]]

[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
r_form = 1.05
```

- [ ] **Step 3: Create `examples/menshutkin.rxn.toml`**

```toml
description = "Menshutkin: NH3 + CH3Cl -> CH3NH3+ + Cl-"
formed = [[1, 5]]
broken = [[5, 9]]

[restraints]
k_form = 2.0
k_broken = 2.0
r_broken = 5.0
max_relax_steps = 200
```

- [ ] **Step 4: Create `examples/e2.rxn.toml`**

```toml
description = "E2 elimination: CH3CH2Cl + OH- -> CH2=CH2 + Cl- + H2O"
formed = [[4, 5]]
broken = [[2, 5], [1, 3]]

[restraints]
k_form = 1.0
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 200
```

- [ ] **Step 5: Create `examples/sn1_dissoc.rxn.toml`**

```toml
description = "SN1 step 1 dissociation: (CH3)3CBr -> tBu+ + Br-"
formed = []
broken = [[1, 5]]

[restraints]
k_form = 0.0
k_broken = 2.0
r_broken = 6.0
max_relax_steps = 200

[sampling]
n_angles = 1
```

- [ ] **Step 6: Create `examples/sn1_recomb.rxn.toml`**

```toml
description = "SN1 step 2 recombination: tBu+ + Cl- -> (CH3)3CCl"
formed = [[1, 5]]
broken = []

[restraints]
k_form = 1.0
k_broken = 0.0
r_broken = 4.0
max_relax_steps = 200
```

- [ ] **Step 7: Verify all six TOMLs load without errors**

Run:
```bash
python -c "from pathlib import Path; from reactx.config import load_config; [print(load_config(Path(f'examples/{n}.rxn'))) for n in ['sn2','proton_transfer','menshutkin','e2','sn1_dissoc','sn1_recomb']]"
```
Expected: 6 `ReactionConfig(...)` printouts, no exceptions.

- [ ] **Step 8: Commit**

```bash
git add examples/sn2.rxn.toml examples/proton_transfer.rxn.toml examples/menshutkin.rxn.toml examples/e2.rxn.toml examples/sn1_dissoc.rxn.toml examples/sn1_recomb.rxn.toml
git commit -m "feat(examples): add per-reaction sidecar TOML config for all 6 reactions"
```

---

## Task 4: Delete `reactx/presets.py` and `tests/test_presets.py`

**Files:**
- Delete: `reactx/presets.py`
- Delete: `tests/test_presets.py`

- [ ] **Step 1: Confirm no other module imports `reactx.presets`**

Run: `git grep -nE "(from reactx\.presets|import reactx\.presets)"`
Expected: only `reactx/cli.py` (lines around 23, 49) and `tests/test_presets.py`. The `cli.py` references will be removed in Task 5.

- [ ] **Step 2: Delete the files**

```bash
rm reactx/presets.py tests/test_presets.py
```

- [ ] **Step 3: Confirm pytest collection still works (cli.py still imports presets, but tests of presets are gone)**

Run: `pytest --collect-only -q tests/test_config.py tests/test_bond_changes.py 2>&1 | tail -5`
Expected: collects without error. (The wider test suite still references presets via cli.py — fixed in Task 5.)

- [ ] **Step 4: Commit**

```bash
git add -u reactx/presets.py tests/test_presets.py
git commit -m "refactor(presets): delete reactx.presets (replaced by per-reaction .rxn.toml)"
```

---

## Task 5: Rewrite `reactx/cli.py` to consume `ReactionConfig`

**Files:**
- Modify: `reactx/cli.py`

- [ ] **Step 1: Replace `build_parser` to drop preset/parameter flags**

In `reactx/cli.py`, replace the entire `build_parser` function with:

```python
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reactx")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run full pipeline on a .rxn file (with sidecar .rxn.toml)")
    run.add_argument("rxn_path", type=Path)
    run.add_argument("-o", "--output", type=Path, required=True)

    run.add_argument("--backend", choices=["uma", "lj"], default="uma")
    run.add_argument("--model", type=str, default="uma-m-1p1",
                     help="UMA model name (uma-m-1p1, uma-s-1p2, ...)")

    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--relax-fmax", type=float, default=0.1)
    run.add_argument("--traj-stride", type=int, default=5)

    run.add_argument("--neb-refine", action="store_true",
                     help="Refine the best trial trajectory with a short NEB "
                          "(only supported for 1 formed + 1 broken reactions)")
    run.add_argument("--neb-images", type=int, default=7)

    run.add_argument("--render", action="store_true",
                     help="Also invoke blender/render.py after pipeline")
    run.add_argument("--blender-exe", type=str, default="blender")
    return p
```

- [ ] **Step 2: Remove `_resolve_effective_params` and the `from reactx.presets ...` import line**

Delete the entire `_resolve_effective_params` function from `reactx/cli.py`. Also remove the top-level `from reactx.presets import get_preset` import (around line 23).

- [ ] **Step 3: Replace imports + rewrite `_cmd_run`**

In `reactx/cli.py`:

(a) At the top, replace the existing imports of `from reactx.bond_changes import BondChanges, compute_bond_changes` with:

```python
from reactx.bond_changes import BondChanges
from reactx.config import ReactionConfig, load_config, resolve_r_form_targets
from reactx.rxn_parser import atom_map_to_reactant_idx, heavy_to_hydrogen_groups, parse_rxn
```

(b) Replace the body of `_cmd_run` from the start through the line `eff = _resolve_effective_params(args, syms_r, formed_pairs)` block with:

```python
def _cmd_run(args: argparse.Namespace) -> int:
    _configure_reactx_logging()

    if not args.rxn_path.exists():
        log.error("Error: .rxn not found: %s", args.rxn_path)
        return 1

    try:
        cfg = load_config(args.rxn_path)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        log.error(
            "A sidecar TOML config is REQUIRED. Create %s.toml. "
            "See examples/sn2.rxn.toml for the schema.",
            args.rxn_path,
        )
        return 2
    except ValueError as exc:
        log.error("Invalid sidecar TOML: %s", exc)
        return 2

    if args.backend == "uma":
        rc = _check_hf_auth()
        if rc != 0:
            return rc

    args.output.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()

    r_mol, p_mol, heavy_mapping = parse_rxn(args.rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    try:
        bond_changes = BondChanges.from_atom_map_pairs(
            formed_map=cfg.formed,
            broken_map=cfg.broken,
            atom_map_to_idx=atom_map_to_reactant_idx(r_mol),
        )
    except (KeyError, ValueError) as exc:
        log.error("Invalid bond_changes from %s.toml: %s", args.rxn_path, exc)
        return 2

    if args.neb_refine and (
        len(bond_changes.formed) != 1 or len(bond_changes.broken) != 1
    ):
        log.error(
            "--neb-refine is only supported for 1 formed + 1 broken bond "
            "reactions in Phase 3 (got formed=%d, broken=%d). Multi-bond NEB "
            "endpoint construction is Phase 4+. Re-run without --neb-refine.",
            len(bond_changes.formed), len(bond_changes.broken),
        )
        return 2

    if len(bond_changes.formed) == 1 and len(bond_changes.broken) == 1:
        a_form_r, b_form_r = bond_changes.formed[0]
        a_brk_r, b_brk_r = bond_changes.broken[0]
        bond_changes_product = BondChanges(
            formed=((heavy_mapping[a_brk_r], heavy_mapping[b_brk_r]),),
            broken=((heavy_mapping[a_form_r], heavy_mapping[b_form_r]),),
        )
    else:
        bond_changes_product = None

    formed_pairs = list(bond_changes.formed)
    broken_pairs = list(bond_changes.broken)
    syms_r = [a.GetSymbol() for a in r_h.GetAtoms()]
    r_form_targets = resolve_r_form_targets(cfg, syms_r, formed_pairs)
    log.info(
        "description=%s effective: k_form=%.2f k_broken=%.2f r_broken=%.2f "
        "max_relax_steps=%d r_form_targets=%s",
        cfg.description,
        cfg.restraints.k_form, cfg.restraints.k_broken,
        cfg.restraints.r_broken, cfg.restraints.max_relax_steps,
        [f"{x:.3f}" for x in r_form_targets] if r_form_targets else "[]",
    )

    model_kwargs = {"model_name": args.model} if args.backend == "uma" else {}
    calc = make_calculator(args.backend, **model_kwargs)

    n_frags_reactant = len(Chem.GetMolFrags(r_h))
    effective_n_angles = cfg.sampling.n_angles
    if n_frags_reactant == 1 and effective_n_angles > 1:
        log.info(
            "unimolecular reaction (1 reactant fragment); "
            "n_angles forced from %d to 1, prescreen skipped",
            effective_n_angles,
        )
        effective_n_angles = 1

    rotations = sample_attack_rotations(
        n=effective_n_angles, cone_half_deg=cfg.sampling.cone_half_deg, seed=args.seed,
    )
```

(c) Replace the prescreen-section block (the `if n_frags_reactant == 1: ... elif args.no_mmff_prescreen: ... elif embedded_by_idx: ...` chain) with:

```python
    prescreen_meta: dict | None = None
    keep_trial_indices: list[int]
    if n_frags_reactant == 1:
        log.info("prescreen: skipped (single trial / unimolecular)")
        keep_trial_indices = sorted(embedded_by_idx.keys())
        prescreen_meta = {
            "enabled": False, "kept": None, "skipped": None,
            "mmff_failed": None, "wall_clock_seconds": 0.0,
        }
    elif not cfg.prescreen.enabled:
        log.info("prescreen: disabled (config: prescreen.enabled = false)")
        keep_trial_indices = sorted(embedded_by_idx.keys())
        prescreen_meta = {
            "enabled": False, "kept": None, "skipped": None,
            "mmff_failed": None, "wall_clock_seconds": 0.0,
        }
    elif embedded_by_idx:
        ordered = sorted(embedded_by_idx.keys())
        atoms_for_prescreen = [embedded_by_idx[i][0] for i in ordered]
        mol_h_template = Chem.AddHs(r_mol)
        pre = prescreen_trials(
            atoms_list=atoms_for_prescreen,
            mol_h_template=mol_h_template,
            formed=formed_pairs, broken=broken_pairs,
            r_form_target=r_form_targets[0] if r_form_targets else 1.6,
            r_broken_target=cfg.restraints.r_broken,
            k_form=cfg.restraints.k_form, k_broken=cfg.restraints.k_broken,
            max_steps=cfg.prescreen.steps,
            k_keep=cfg.prescreen.keep,
        )
        keep_trial_indices = [ordered[k] for k in pre.kept]
        skipped_trial_indices = [ordered[s] for s in pre.skipped]
        prescreen_meta = {
            "enabled": True,
            "kept": list(keep_trial_indices),
            "skipped": list(skipped_trial_indices),
            "mmff_failed": pre.mmff_failed,
            "wall_clock_seconds": pre.wall_clock_seconds,
        }
    else:
        keep_trial_indices = []
```

(d) Replace the `for i in keep_trial_indices: ...` block's `build_restraints(...)` call args and the relax `max_steps=...` call:

```python
        restraints = build_restraints(
            atoms_init,
            formed=formed_pairs,
            broken=broken_pairs,
            r_form=r_form_targets[0] if r_form_targets else None,
            r_broken=cfg.restraints.r_broken,
            k_form=cfg.restraints.k_form,
            k_broken=cfg.restraints.k_broken,
        )
        try:
            frames, energies = relax_with_restraints(
                atoms_init, restraints, calc,
                max_steps=cfg.restraints.max_relax_steps,
                fmax=args.relax_fmax,
                traj_stride=args.traj_stride,
            )
```

and replace the `reached_product(...)` call's `r_broken_target=eff["r_broken"]` with `r_broken_target=cfg.restraints.r_broken`.

(e) Replace the `_write_outputs_and_exit(...)` calls so they receive `cfg` and `r_form_targets` instead of `eff`:

```python
        return _write_outputs_and_exit(
            args, trials, t_start, neb_refined=False, rc=1,
            cfg=cfg, r_form_targets=r_form_targets, prescreen_meta=prescreen_meta,
        )
```

```python
    rc = _write_outputs_and_exit(
        args, trials, t_start, neb_refined=neb_refined, rc=0,
        cfg=cfg, r_form_targets=r_form_targets, prescreen_meta=prescreen_meta,
    )
```

- [ ] **Step 4: Rewrite `_write_outputs_and_exit`**

Replace the function with:

```python
def _write_outputs_and_exit(
    args: argparse.Namespace,
    trials: list[TrialResult],
    t_start: float,
    *,
    neb_refined: bool,
    rc: int,
    cfg: ReactionConfig | None = None,
    r_form_targets: list[float] | None = None,
    prescreen_meta: dict | None = None,
) -> int:
    best: TrialResult | None = None
    if rc == 0 and trials:
        try:
            best = score_trials(trials)
        except ValueError:
            best = None
    selected = best.trial_idx if best is not None else -1
    converged = best.reached_product if best is not None else False
    meta: dict = {
        "backend": args.backend,
        "description": cfg.description if cfg is not None else None,
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
        "prescreen": prescreen_meta,
        "wall_clock_seconds": float(time.monotonic() - t_start),
        "neb_refined": neb_refined,
        "effective_params": (
            {
                "k_form": cfg.restraints.k_form,
                "k_broken": cfg.restraints.k_broken,
                "r_broken": cfg.restraints.r_broken,
                "max_relax_steps": cfg.restraints.max_relax_steps,
                "r_form_targets": list(r_form_targets) if r_form_targets is not None else [],
            }
            if cfg is not None else None
        ),
    }
    meta_clean = _sanitize_for_json(meta)
    (args.output / "meta.json").write_text(json.dumps(meta_clean, indent=2))

    if best is not None:
        (args.output / "energies.json").write_text(json.dumps(best.energies))
    return rc
```

- [ ] **Step 5: Confirm `reactx/cli.py` compiles and imports**

Run: `python -c "import reactx.cli; print('ok')"`
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add reactx/cli.py
git commit -m "refactor(cli): consume ReactionConfig; drop preset/per-param flags"
```

---

## Task 6: Rewrite `tests/test_cli.py`

**Files:**
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Replace the file entirely**

```python
"""CLI argument parsing + meta.json writer tests (no UMA invocation)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from reactx.cli import _write_outputs_and_exit, build_parser
from reactx.config import (
    PrescreenConfig,
    ReactionConfig,
    RestraintConfig,
    SamplingConfig,
)
from reactx.scoring import TrialResult


def _sample_cfg() -> ReactionConfig:
    return ReactionConfig(
        description="sample",
        formed=((1, 2),),
        broken=((1, 3),),
        restraints=RestraintConfig(
            k_form=2.0, k_broken=2.0, r_broken=5.0,
            max_relax_steps=200, r_form=None,
        ),
        sampling=SamplingConfig(),
        prescreen=PrescreenConfig(),
    )


def test_default_flags_parse():
    p = build_parser()
    a = p.parse_args(["run", "examples/sn2.rxn", "-o", "out/"])
    assert a.cmd == "run"
    assert a.backend == "uma"
    assert a.model == "uma-m-1p1"
    assert a.seed == 0
    assert a.relax_fmax == 0.1
    assert a.traj_stride == 5
    assert a.neb_refine is False
    assert a.neb_images == 7
    assert a.render is False
    assert a.blender_exe == "blender"


def test_dropped_flags_now_rejected():
    p = build_parser()
    for flag in [
        "--reaction-type", "--k-form", "--k-broken", "--r-form", "--r-broken",
        "--max-relax-steps", "--n-angles", "--cone-half-deg",
        "--no-mmff-prescreen", "--prescreen-keep", "--prescreen-steps",
    ]:
        with pytest.raises(SystemExit):
            args = ["run", "examples/sn2.rxn", "-o", "out/", flag]
            if flag != "--no-mmff-prescreen":
                args.append("0")
            p.parse_args(args)


def test_neb_refine_flag():
    p = build_parser()
    a = p.parse_args(["run", "examples/sn2.rxn", "-o", "out/", "--neb-refine"])
    assert a.neb_refine is True


def test_meta_json_includes_description_and_effective_params(tmp_path: Path):
    args = argparse.Namespace(backend="lj", output=tmp_path)
    cfg = _sample_cfg()
    r_form_targets = [1.47]
    trials = [TrialResult(
        trial_idx=0, rotation_deg=0.0, frames=[], energies=[1.0, 2.0],
        reached_product=True, peak_energy=2.0, n_steps=2,
    )]
    with patch("reactx.cli.score_trials", return_value=trials[0]):
        rc = _write_outputs_and_exit(
            args, trials, t_start=0.0,
            neb_refined=False, rc=0,
            cfg=cfg, r_form_targets=r_form_targets, prescreen_meta=None,
        )
    assert rc == 0
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["description"] == "sample"
    assert "reaction_type" not in meta
    assert meta["effective_params"] == {
        "k_form": 2.0, "k_broken": 2.0,
        "r_broken": 5.0, "max_relax_steps": 200,
        "r_form_targets": [1.47],
    }


def test_meta_json_prescreen_block(tmp_path: Path):
    args = argparse.Namespace(backend="lj", output=tmp_path)
    cfg = _sample_cfg()
    pre = {
        "enabled": True, "kept": [0, 3, 5], "skipped": [1, 2, 4, 6, 7],
        "mmff_failed": False, "wall_clock_seconds": 1.8,
    }
    trials = [TrialResult(
        trial_idx=0, rotation_deg=0.0, frames=[], energies=[1.0, 2.0],
        reached_product=True, peak_energy=2.0, n_steps=2,
    )]
    with patch("reactx.cli.score_trials", return_value=trials[0]):
        rc = _write_outputs_and_exit(
            args, trials, t_start=0.0,
            neb_refined=False, rc=0,
            cfg=cfg, r_form_targets=[1.47], prescreen_meta=pre,
        )
    assert rc == 0
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["prescreen"] == pre


def test_meta_json_when_cfg_missing_writes_null_effective(tmp_path: Path):
    """rc != 0 path may pass cfg=None (e.g. when load_config failed early)."""
    args = argparse.Namespace(backend="lj", output=tmp_path)
    rc = _write_outputs_and_exit(
        args, trials=[], t_start=0.0,
        neb_refined=False, rc=1,
        cfg=None, r_form_targets=None, prescreen_meta=None,
    )
    assert rc == 1
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["description"] is None
    assert meta["effective_params"] is None
```

- [ ] **Step 2: Run the test file**

Run: `pytest tests/test_cli.py -x`
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cli.py
git commit -m "test(cli): rewrite tests for ReactionConfig-based CLI"
```

---

## Task 7: Add `tmp_rxn_with_toml` fixture for downstream tests

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add the fixture at the bottom of `tests/conftest.py`**

Append:

```python
@pytest.fixture()
def tmp_rxn_with_toml(tmp_path: Path):
    """Copy `examples/<stem>.rxn` to tmp_path, write a fresh sidecar TOML.

    Usage:
        rxn_path = tmp_rxn_with_toml("sn2", toml_body='''\\
            description = "sn2 fast"
            formed = [[1, 3]]
            broken = [[1, 2]]
            [restraints]
            k_form = 0.5
            k_broken = 1.0
            r_broken = 4.0
            max_relax_steps = 30
            [sampling]
            n_angles = 1
        ''')

    Returns the temp `.rxn` Path. The .rxn body is unchanged from
    examples/<stem>.rxn; only the sidecar TOML is configurable.
    """
    examples = Path(__file__).resolve().parent.parent / "examples"

    def _make(stem: str, *, toml_body: str) -> Path:
        src_rxn = examples / f"{stem}.rxn"
        dst_rxn = tmp_path / f"{stem}.rxn"
        dst_rxn.write_bytes(src_rxn.read_bytes())
        (tmp_path / f"{stem}.rxn.toml").write_text(toml_body, encoding="utf-8")
        return dst_rxn

    return _make
```

- [ ] **Step 2: Confirm fixture is importable from a probe test**

Run: `pytest --collect-only -q tests/conftest.py 2>&1 | tail -3`
Expected: no collection errors.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test(conftest): add tmp_rxn_with_toml fixture"
```

---

## Task 8: Migrate `tests/test_cli_neb_refine_guard.py` to TOML fixture

**Files:**
- Modify: `tests/test_cli_neb_refine_guard.py`

- [ ] **Step 1: Replace the file**

```python
"""Test that --neb-refine is rejected for multi-bond reactions in Phase 3."""
from pathlib import Path

from reactx import cli


_E2_TOML = """\
description = "e2 fast"
formed = [[4, 5]]
broken = [[2, 5], [1, 3]]
[restraints]
k_form = 1.0
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 5
[sampling]
n_angles = 1
"""

_SN1_RECOMB_TOML = """\
description = "sn1_recomb fast"
formed = [[1, 5]]
broken = []
[restraints]
k_form = 1.0
k_broken = 0.0
r_broken = 4.0
max_relax_steps = 5
[sampling]
n_angles = 1
"""

_SN2_TOML = """\
description = "sn2 fast"
formed = [[1, 3]]
broken = [[1, 2]]
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 5
[sampling]
n_angles = 1
"""


def test_neb_refine_rejected_for_e2_reaction(tmp_path, monkeypatch, caplog, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("e2", toml_body=_E2_TOML)
    out = tmp_path / "out"
    monkeypatch.setattr(cli, "_check_hf_auth", lambda: 0)
    monkeypatch.setattr(cli, "_configure_reactx_logging", lambda: None)
    cli.log.propagate = True
    caplog.set_level("INFO", logger="reactx")

    rc = cli.main([
        "run", str(rxn), "-o", str(out),
        "--backend", "lj", "--neb-refine",
    ])
    assert rc == 2
    msgs = " ".join(rec.getMessage() for rec in caplog.records)
    assert "Phase 3" in msgs
    assert "1 formed" in msgs and "1 broken" in msgs


def test_neb_refine_rejected_for_sn1_recomb_reaction(tmp_path, monkeypatch, caplog, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn1_recomb", toml_body=_SN1_RECOMB_TOML)
    out = tmp_path / "out"
    monkeypatch.setattr(cli, "_check_hf_auth", lambda: 0)
    monkeypatch.setattr(cli, "_configure_reactx_logging", lambda: None)
    cli.log.propagate = True
    caplog.set_level("INFO", logger="reactx")

    rc = cli.main([
        "run", str(rxn), "-o", str(out),
        "--backend", "lj", "--neb-refine",
    ])
    assert rc == 2
    msgs = " ".join(rec.getMessage() for rec in caplog.records)
    assert "Phase 3" in msgs
    assert "1 formed" in msgs and "1 broken" in msgs


def test_neb_refine_accepted_for_sn2(tmp_path, monkeypatch, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn2", toml_body=_SN2_TOML)
    out = tmp_path / "out"
    monkeypatch.setattr(cli, "_check_hf_auth", lambda: 0)
    cli.main([
        "run", str(rxn), "-o", str(out),
        "--backend", "lj", "--neb-refine",
    ])
    assert (out / "meta.json").exists()
```

- [ ] **Step 2: Run the file**

Run: `pytest tests/test_cli_neb_refine_guard.py -x`
Expected: 3 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cli_neb_refine_guard.py
git commit -m "test(cli_neb_refine_guard): migrate to TOML fixture"
```

---

## Task 9: Migrate `tests/test_cli_unimolecular.py` to TOML fixture

**Files:**
- Modify: `tests/test_cli_unimolecular.py`

- [ ] **Step 1: Replace the file**

```python
"""Tests for unimolecular reaction handling: n_angles auto-clamp + prescreen skip."""
import json
from pathlib import Path

import pytest


_SN1_DISSOC_FAST = """\
description = "sn1_dissoc fast"
formed = []
broken = [[1, 5]]
[restraints]
k_form = 0.0
k_broken = 2.0
r_broken = 6.0
max_relax_steps = 5
[sampling]
n_angles = 8
"""


@pytest.fixture()
def fake_unimolecular_pipeline(monkeypatch):
    import numpy as np
    from ase import Atoms
    from reactx import calculators, embed3d, path_relax

    def fake_calc(*args, **kwargs):
        from ase.calculators.lj import LennardJones
        return LennardJones()

    def fake_embed(mol, **kw):
        atoms = Atoms("CCCCC", positions=np.array([
            [0, 0, 0], [1.5, 0, 0], [3.0, 0, 0], [-1.5, 0, 0], [0, 1.5, 0],
        ]))
        return atoms

    def fake_relax(atoms_init, restraints, calc, **kw):
        return [atoms_init], [0.0]

    monkeypatch.setattr(calculators, "make_calculator", fake_calc)
    monkeypatch.setattr(embed3d, "embed_mol_to_atoms", fake_embed)
    monkeypatch.setattr(path_relax, "relax_with_restraints", fake_relax)


def test_unimolecular_n_angles_clamped_to_one(
    fake_unimolecular_pipeline, tmp_path: Path, tmp_rxn_with_toml,
):
    """When reactant has 1 fragment and TOML claims n_angles=8, clamp to 1."""
    rxn = tmp_rxn_with_toml("sn1_dissoc", toml_body=_SN1_DISSOC_FAST)
    out = tmp_path / "out"
    from reactx.cli import main
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "lj"])
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text())
    assert len(meta["trials"]) == 1
    assert meta["prescreen"]["enabled"] is False
    assert meta["prescreen"]["kept"] is None
```

- [ ] **Step 2: Run the file**

Run: `pytest tests/test_cli_unimolecular.py -x`
Expected: 1 test passes.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cli_unimolecular.py
git commit -m "test(cli_unimolecular): migrate to TOML fixture"
```

---

## Task 10: Migrate slow integration tests (`test_re1_*`, `test_re3_*`, `test_re4_*`, `test_neb_refine_sn2`, `test_prescreen_disabled_sn2`, `test_wallclock_sn2`)

**Files:**
- Modify: `tests/test_re1_sn2.py`
- Modify: `tests/test_re1_proton_transfer.py`
- Modify: `tests/test_re1_menshutkin.py`
- Modify: `tests/test_re3_e2.py`
- Modify: `tests/test_re3_sn1_dissoc.py`
- Modify: `tests/test_re4_sn1_recomb.py`
- Modify: `tests/test_neb_refine_sn2.py`
- Modify: `tests/test_prescreen_disabled_sn2.py`
- Modify: `tests/test_wallclock_sn2.py`

For each file the migration pattern is identical:

1. Build a `_<reaction>_FAST` TOML literal with `max_relax_steps` reduced to the value previously passed via `--max-relax-steps` and `n_angles` reduced to the value previously passed via `--n-angles`.
2. Replace the `cli.main([...])` argument list to **drop** `--reaction-type`, `--n-angles`, `--max-relax-steps`, `--no-mmff-prescreen`, `--prescreen-keep`, `--prescreen-steps`.
3. Replace `examples/<x>.rxn` path with the `tmp_rxn_with_toml(...)` result.
4. Replace any `meta["reaction_type"]` assertion with `meta["description"]`.

- [ ] **Step 1: Migrate `tests/test_re1_sn2.py`**

Replace the file:

```python
"""End-to-end SN2 test using UMA. Marked slow, requires HF auth + GPU."""
import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from reactx.cli import main


_SN2_FAST = """\
description = "SN2 fast"
formed = [[1, 3]]
broken = [[1, 2]]
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 50
[sampling]
n_angles = 4
"""


@pytest.mark.slow
def test_re1_sn2_end_to_end(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn2", toml_body=_SN2_FAST)
    out = tmp_path / "sn2"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["selected_trial"] >= 0
    pre = meta["prescreen"]
    assert pre["enabled"] is True
    if not pre["mmff_failed"]:
        assert len(pre["kept"]) == 3
        assert len(meta["trials"]) == 3
    assert any(t["reached_product"] for t in meta["trials"])

    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) >= 3

    syms = frames[0].get_chemical_symbols()
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    o_idx = syms.index("O")

    angles = []
    for f in frames:
        v_co = f.positions[o_idx] - f.positions[c_idx]
        v_ccl = f.positions[cl_idx] - f.positions[c_idx]
        cos_t = float(np.dot(v_co, v_ccl) / (
            np.linalg.norm(v_co) * np.linalg.norm(v_ccl)
        ))
        angles.append(float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))))
    assert max(angles) >= 120.0

    d_co_first = frames[0].get_distance(c_idx, o_idx)
    d_co_last = frames[-1].get_distance(c_idx, o_idx)
    d_ccl_first = frames[0].get_distance(c_idx, cl_idx)
    d_ccl_last = frames[-1].get_distance(c_idx, cl_idx)
    assert d_co_last < d_co_first - 0.5
    assert d_ccl_last > d_ccl_first + 0.5
```

- [ ] **Step 2: Migrate `tests/test_re1_proton_transfer.py`**

Insert this TOML literal at module top:

```python
_PT_FAST = """\
description = "Proton transfer fast"
formed = [[1, 3]]
broken = [[1, 2]]
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 50
r_form = 1.05
[sampling]
n_angles = 4
"""
```

In the test body, replace the `cli.main([...])` argv with `["run", str(rxn), "-o", str(out), "--backend", "uma"]` and replace `examples/proton_transfer.rxn` references with `tmp_rxn_with_toml("proton_transfer", toml_body=_PT_FAST)`. Drop `--reaction-type proton_transfer`, `--n-angles`, `--max-relax-steps` from argv. If `meta["reaction_type"]` is asserted, replace with `meta["description"] == "Proton transfer fast"`.

- [ ] **Step 3: Migrate `tests/test_re1_menshutkin.py`** with this TOML literal:

```python
_MEN_FAST = """\
description = "Menshutkin fast"
formed = [[1, 5]]
broken = [[5, 9]]
[restraints]
k_form = 2.0
k_broken = 2.0
r_broken = 5.0
max_relax_steps = 100
[sampling]
n_angles = 4
"""
```

Same migration steps as step 2 (drop `--reaction-type menshutkin --n-angles --max-relax-steps` from argv; replace rxn path with `tmp_rxn_with_toml("menshutkin", toml_body=_MEN_FAST)`).

- [ ] **Step 4: Migrate `tests/test_re3_e2.py`**:

```python
_E2_FAST = """\
description = "E2 fast"
formed = [[4, 5]]
broken = [[2, 5], [1, 3]]
[restraints]
k_form = 1.0
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
[sampling]
n_angles = 4
"""
```

Same migration; the file currently asserts `meta["reaction_type"] == "e2"` (line 29) — **replace with** `assert meta["description"] == "E2 fast"`.

- [ ] **Step 5: Migrate `tests/test_re3_sn1_dissoc.py`**:

```python
_SN1D_FAST = """\
description = "SN1 dissoc fast"
formed = []
broken = [[1, 5]]
[restraints]
k_form = 0.0
k_broken = 2.0
r_broken = 6.0
max_relax_steps = 100
[sampling]
n_angles = 1
"""
```

Same migration. (`n_angles = 1` is required by unimolecular auto-clamp; preserves test semantics.)

- [ ] **Step 6: Migrate `tests/test_re4_sn1_recomb.py`**:

```python
_SN1R_FAST = """\
description = "SN1 recomb fast"
formed = [[1, 5]]
broken = []
[restraints]
k_form = 1.0
k_broken = 0.0
r_broken = 4.0
max_relax_steps = 100
[sampling]
n_angles = 4
"""
```

Same migration; if the file asserts `meta["reaction_type"] == "sn1_recomb"`, replace with `meta["description"] == "SN1 recomb fast"`.

- [ ] **Step 7: Migrate `tests/test_neb_refine_sn2.py`**:

```python
_SN2_NEB = """\
description = "SN2 neb-refine fast"
formed = [[1, 3]]
broken = [[1, 2]]
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 50
[sampling]
n_angles = 4
"""
```

Same migration; keep `--neb-refine` and `--neb-images` flags in argv (these stayed CLI-side), drop the others.

- [ ] **Step 8: Migrate `tests/test_prescreen_disabled_sn2.py`**:

```python
_SN2_NO_PRESCREEN = """\
description = "SN2 no-prescreen"
formed = [[1, 3]]
broken = [[1, 2]]
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 30
[sampling]
n_angles = 3
[prescreen]
enabled = false
"""
```

argv becomes `["run", str(rxn), "-o", str(out), "--backend", "uma"]` (drop `--no-mmff-prescreen`, `--n-angles`, `--max-relax-steps`).

- [ ] **Step 9: Migrate `tests/test_wallclock_sn2.py`**:

```python
_SN2_WALLCLOCK = """\
description = "SN2 wallclock"
formed = [[1, 3]]
broken = [[1, 2]]
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 50
[sampling]
n_angles = 4
"""
```

Same migration; preserve the wall-clock assertion logic, only swap the CLI shape.

- [ ] **Step 10: Run the fast subset to confirm no regression in non-slow paths**

Run: `pytest -m "not slow and not blender" -x`
Expected: all non-slow tests pass.

- [ ] **Step 11: Commit**

```bash
git add tests/test_re1_sn2.py tests/test_re1_proton_transfer.py tests/test_re1_menshutkin.py \
        tests/test_re3_e2.py tests/test_re3_sn1_dissoc.py tests/test_re4_sn1_recomb.py \
        tests/test_neb_refine_sn2.py tests/test_prescreen_disabled_sn2.py tests/test_wallclock_sn2.py
git commit -m "test(integration): migrate slow tests to TOML fixture"
```

---

## Task 11: Update `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the "Reaction-type presets" section**

Locate the section starting `## Reaction-type presets` (around line 54) through the end of its preset table and code blocks (ending around line 82, before the "## Phase Re1 + Phase 3 + Phase 4 動作確認" section).

Replace with:

````markdown
## Per-reaction `.rxn.toml` config

各 `examples/<name>.rxn` には同階層に同名 stem の sidecar TOML (`<name>.rxn.toml`) を **必須で** 配置する。CLI は `<rxn_path>.toml` を機械的にロードし、結合変化情報 (`formed` / `broken`, atom-map 番号) と物理パラメータ (`k_form` / `k_broken` / `r_broken` / `max_relax_steps` / 任意 `r_form`) と sampling/prescreen 設定をすべてここから取る。

最小例 (`examples/sn2.rxn.toml`):

```toml
description = "SN2 anion: CH3Cl + OH- -> CH3OH + Cl-"
formed = [[1, 3]]
broken = [[1, 2]]

[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
```

| `.rxn` | description | formed (map) | broken (map) | k_form | k_broken | r_broken (Å) | max_relax_steps | r_form | n_angles |
|---|---|---|---|---|---|---|---|---|---|
| sn2.rxn | SN2 anion (`O⁻ + CH₃Cl`) | `[[1,3]]` | `[[1,2]]` | 0.5 | 1.0 | 4.0 | 100 | 元素表 | 8 |
| proton_transfer.rxn | Proton transfer (`HCl + NH₃`) | `[[1,3]]` | `[[1,2]]` | 0.5 | 1.0 | 4.0 | 100 | 1.05 | 8 |
| menshutkin.rxn | Menshutkin (`NH₃ + CH₃Cl`) | `[[1,5]]` | `[[5,9]]` | 2.0 | 2.0 | 5.0 | 200 | 元素表 | 8 |
| e2.rxn | E2 elimination | `[[4,5]]` | `[[2,5],[1,3]]` | 1.0 | 1.0 | 4.0 | 200 | 元素表 | 8 |
| sn1_dissoc.rxn | SN1 step 1 解離 | `[]` | `[[1,5]]` | 0.0 | 2.0 | 6.0 | 200 | — | **1** |
| sn1_recomb.rxn | SN1 step 2 recombination | `[[1,5]]` | `[]` | 1.0 | 0.0 | 4.0 | 200 | 元素表 (C-Cl 1.78) | 8 |

`r_form` は省略時に Cordero (2008) 共有結合半径表で per-bond ルックアップ、scalar で全 formed 同値、list で per-bond 指定。`[sampling]` / `[prescreen]` は省略可能で、それぞれ `n_angles=8 cone_half_deg=30.0`、`enabled=true keep=3 steps=30` がデフォルト。

```bash
reactx run examples/sn2.rxn -o out/sn2/ --backend uma --render
reactx run examples/menshutkin.rxn -o out/men/ --backend uma --render
```

> **Phase 6 で削除されたフラグ:** `--reaction-type`, `--k-form`, `--k-broken`, `--r-form`, `--r-broken`, `--max-relax-steps`, `--n-angles`, `--cone-half-deg`, `--no-mmff-prescreen`, `--prescreen-keep`, `--prescreen-steps`。これらはすべて `.rxn.toml` 側で指定する。
````

- [ ] **Step 2: Update the "使い方" code block (lines 27-33)**

Replace the existing `reactx run ...` examples in the 「使い方」section with:

```bash
# SN2 (description / formed / broken / params are all in examples/sn2.rxn.toml)
reactx run examples/sn2.rxn -o out/sn2/ --backend uma --render
# Proton transfer
reactx run examples/proton_transfer.rxn -o out/pt/ --backend uma --render
```

Also remove the prose right after these examples that mentions `--reaction-type` and the link to the preset section. Replace with:

```
反応クラスごとの全パラメータは `examples/<name>.rxn.toml` に集約されている (下節 [Per-reaction .rxn.toml config](#per-reaction-rxntoml-config) 参照)。
```

- [ ] **Step 3: Update the "主要フラグ" bullets (lines 37-46)**

Replace the bullet list with:

```
主要フラグ (環境/出力依存のみ):

- `--backend {uma,lj}` (default: `uma`)
- `--model uma-m-1p1` (UMA model name)
- `--seed 0` (sampling 再現性デバッグ)
- `--relax-fmax 0.1`, `--traj-stride 5` (出力品質)
- `--neb-refine` + `--neb-images 7` (1 formed + 1 broken のみ対応)
- `--render` + `--blender-exe blender`
```

- [ ] **Step 4: Update the architecture diagram (lines 152-164)**

Replace the diagram with:

```
.rxn + .rxn.toml ─> rxn_parser + load_config ─> ReactionConfig
                                                       │
                                                       ▼
                                               BondChanges (formed/broken)
                                                       │
                                                       ▼
                                              embed3d (rotation perturb)
                                                ├ trial 1
                                                ├ trial 2  ─┐
                                                ├ ...        │ FIRE + Hookean/PullApart restraints
                                                └ trial N  ─┘
                                                       │
                                                       ▼
                                                scoring → best trial
                                                       │
                                          (optional) neb refinement
                                                       │
                                                       ▼
                                               trajectory.xyz → blender/render.py → .blend
```

- [ ] **Step 5: Remove the obsolete "Breaking changes (Phase 3)" callout (lines 11-14)** since after Phase 6 the public surface has changed entirely.

Replace it with:

```
> **Breaking changes (Phase 6, develop ← phase-6):**
> - CLI フラグ `--reaction-type` および `--k-* / --r-* / --max-relax-steps / --n-angles / --cone-half-deg / --no-mmff-prescreen / --prescreen-keep / --prescreen-steps` は **全削除**。各反応の設定は `<rxn_path>.toml` (sidecar TOML) に書く。
> - `meta.json.reaction_type` キー → 削除、代わりに `meta.json.description` (TOML の `description` 値そのまま) を書く。
> - `reactx.presets` モジュール削除、`reactx.bond_changes.compute_bond_changes` 削除。`BondChanges.from_atom_map_pairs(formed_map, broken_map, atom_map_to_idx)` を使用。
```

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs(readme): replace preset section with .rxn.toml config (Phase 6)"
```

---

## Task 12: DoD verification

**Files:** none (read-only)

- [ ] **Step 1: Full fast-suite pytest**

Run: `pytest -m "not slow and not blender"`
Expected: all tests pass, no collection errors, no `ImportError`.

- [ ] **Step 2: Confirm `--reaction-type` rejected by argparse**

Run: `reactx run examples/sn2.rxn -o /tmp/x --reaction-type sn2_anion`
Expected: argparse error mentioning `unrecognized arguments: --reaction-type`.

- [ ] **Step 3: Confirm missing sidecar produces clean error**

Run:
```bash
mkdir -p /tmp/no_toml && cp examples/sn2.rxn /tmp/no_toml/
python -c "from reactx.cli import main; raise SystemExit(main(['run','/tmp/no_toml/sn2.rxn','-o','/tmp/no_toml/out','--backend','lj']))"
```
Expected: exits with code 2 and stderr/log mentions `sidecar TOML not found` and references `examples/sn2.rxn.toml` schema.

- [ ] **Step 4: (Optional, requires UMA) Run one slow test**

Run: `pytest tests/test_re1_sn2.py -x -m slow -v`
Expected: passes (or, if UMA is unavailable, skipped/xfail).

- [ ] **Step 5: Final ruff lint**

Run: `python -m ruff check reactx tests`
Expected: zero warnings.

- [ ] **Step 6: Final commit gate** — verify `git status` is clean.

```bash
git status
```
Expected: `nothing to commit, working tree clean`.

---

## Notes for the implementer

- **Task ordering matters**: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8/9/10 (parallelizable after 7) → 11 → 12.
- **Intermediate test breakage is expected (and bounded)**: After Task 2 commit, `reactx.cli` still imports `compute_bond_changes` and `presets`, so `import reactx.cli` may fail. This is why Task 2 step 4 only runs `pytest tests/test_bond_changes.py tests/test_examples_menshutkin.py` (not the full fast suite). After Task 4 commit (presets deletion), `reactx.cli` is fully broken until Task 5 fixes it. From Task 5 onward each task should leave the fast suite green; verify with `pytest -m "not slow and not blender"` at each task's final commit.
- **Tasks 8/9/10 parallelization**: Once Task 7 (the fixture) is committed, the per-file migrations in Task 10 are independent and can be dispatched to separate subagents. A single subagent can also handle them serially without coordination.
- **Coverage**: All tests except `test_blender_smoke.py` are touched by this plan. If a test currently using `compute_bond_changes` or `--reaction-type` was overlooked, apply the migration pattern from Task 10 step 1 (the fully worked SN2 example).
- **Plan vs spec divergence**: If implementation reveals the spec is wrong, **stop and update the spec first** before continuing — do not silently diverge.
