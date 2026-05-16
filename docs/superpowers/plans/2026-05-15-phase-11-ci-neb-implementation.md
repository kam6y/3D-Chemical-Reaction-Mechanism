# Phase 11 — Pure CI-NEB pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** AFIR / sticky latch / placement / scoring を全廃し、`.rxn` から R/P endpoint を直接構築 → UMA で minimum まで緩和 → CI-NEB で経路生成する Pure CI-NEB パイプラインに置き換える。

**Architecture:** `.rxn` の R/P 両構造を `embed_fragments_to_positions` で 3D 化し、反応点 anchor を `initial_separation` Å で向き合わせる単純 placement → 両端を UMA で `fmax=0.01` まで relax → `align_product_to_reactant` で原子順序揃え → 既存 `neb.py` (IDPP + 2-phase CI-NEB) で経路生成。trial sweep 廃止により 1 反応 = 1 trajectory。

**Tech Stack:** Python 3.11+, ASE (FIRE / BFGS / NEB / IDPP), RDKit (ETKDGv3 + MMFF94), UMA via FAIR-chem, pytest, tomllib (stdlib). 既存テストフレームワーク (`pytest -m slow` / `-m blender`) を維持。

**Spec:** `docs/superpowers/specs/2026-05-15-phase-11-ci-neb-design.md`

---

## 実行順序メモ

破壊的変更を伴うため、以下の順で進めると **各タスク単体で commit 可能** な粒度を保てる:

| Phase | Tasks | コードベース状態 |
|---|---|---|
| A. 新規モジュール追加 | 1, 2 | 既存挙動維持、新 API が並存 |
| B. config + placement 置換 | 3, 4 | 旧 `.rxn.toml` が `ConfigError` で reject、旧 placement API 削除 |
| C. align + neb 微改修 | 5, 6 | align の 1+1 制限解除、neb 戻り値拡張 |
| D. cli 全面書き直し | 7 | 新パイプラインが動く |
| E. AFIR 系削除 | 8 | obsolete モジュール削除 |
| F. examples 移行 | 9 | 8 反応すべて新 schema |
| G. 統合テスト書き直し | 10 | NEB ベースの assertion |
| H. ドキュメント | 11 | README Phase 11 化 |

Phase B–E の間は古いテストが赤になるが許容。各タスク末尾の commit で「壊れるが意図通り」を明示する。

---

### Task 1: `BondChanges.from_reaction_diff` 追加

`.rxn` の R/P bond list 差分から `formed` / `broken` を自動推論する API を追加する。これにより TOML から `formed` / `broken` を撤去できる。

**Files:**
- Modify: `reactx/bond_changes.py`
- Test: `tests/test_bond_changes.py`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_bond_changes.py` の末尾に追加:

```python
import pytest
from pathlib import Path

from reactx.bond_changes import BondChanges
from reactx.rxn_parser import parse_rxn

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"


@pytest.mark.parametrize(
    "rxn_name,expected_formed_count,expected_broken_count",
    [
        ("sn2", 1, 1),
        ("proton_transfer", 1, 1),
        ("menshutkin", 1, 1),
        ("e2", 1, 2),
        ("sn1_dissoc", 0, 1),
        ("sn1_recomb", 1, 0),
        ("diels_alder_simple", 2, 0),
        ("diels_alder_endo", 2, 0),
    ],
)
def test_from_reaction_diff_counts(rxn_name, expected_formed_count, expected_broken_count):
    r_mol, p_mol, _ = parse_rxn(EXAMPLES / f"{rxn_name}.rxn")
    bc = BondChanges.from_reaction_diff(r_mol, p_mol)
    assert len(bc.formed) == expected_formed_count, (
        f"{rxn_name}: expected {expected_formed_count} formed, got {len(bc.formed)}"
    )
    assert len(bc.broken) == expected_broken_count, (
        f"{rxn_name}: expected {expected_broken_count} broken, got {len(bc.broken)}"
    )


def test_from_reaction_diff_sn2_matches_atom_indices():
    r_mol, p_mol, _ = parse_rxn(EXAMPLES / "sn2.rxn")
    bc = BondChanges.from_reaction_diff(r_mol, p_mol)
    # SN2: C(1) + Cl(2) + OH-(3,4) -> C(1)-OH(3,4) + Cl-(2)
    # atom map 1 -> C (reactant idx 0), 2 -> Cl (idx 1), 3 -> O (idx 2)
    # formed: C-O (0,2) or (2,0)
    # broken: C-Cl (0,1) or (1,0)
    formed_canon = {tuple(sorted(p)) for p in bc.formed}
    broken_canon = {tuple(sorted(p)) for p in bc.broken}
    assert formed_canon == {(0, 2)}
    assert broken_canon == {(0, 1)}
```

- [ ] **Step 2: テストが失敗することを確認**

```
pytest tests/test_bond_changes.py::test_from_reaction_diff_counts -v
```
Expected: FAIL with `AttributeError: type object 'BondChanges' has no attribute 'from_reaction_diff'`

- [ ] **Step 3: `reactx/bond_changes.py` に `from_reaction_diff` 実装**

`reactx/bond_changes.py` の `from_atom_map_pairs` の **直後** に追加:

```python
    @classmethod
    def from_reaction_diff(
        cls,
        r_mol,
        p_mol,
    ) -> BondChanges:
        """Build BondChanges by diffing R/P bond sets in atom-map space.

        Returns BondChanges with 0-based atom indices in the reactant Mol's ordering.
        A bond is 'formed' if it exists in product but not in reactant (atom-map equal).
        A bond is 'broken' if it exists in reactant but not in product.
        Bond order changes (single<->double) are ignored — only connectivity diffs.

        Raises:
            ValueError: when any heavy atom on either side lacks an atom map number,
                or when R/P atom-map sets differ.
        """
        r_map_to_idx: dict[int, int] = {}
        for atom in r_mol.GetAtoms():
            mapnum = atom.GetAtomMapNum()
            if mapnum == 0:
                raise ValueError(
                    f"reactant atom {atom.GetIdx()} has no atom map number"
                )
            r_map_to_idx[mapnum] = atom.GetIdx()
        p_map_to_idx: dict[int, int] = {}
        for atom in p_mol.GetAtoms():
            mapnum = atom.GetAtomMapNum()
            if mapnum == 0:
                raise ValueError(
                    f"product atom {atom.GetIdx()} has no atom map number"
                )
            p_map_to_idx[mapnum] = atom.GetIdx()
        if set(r_map_to_idx.keys()) != set(p_map_to_idx.keys()):
            raise ValueError(
                f"atom map mismatch: reactant {sorted(r_map_to_idx)} "
                f"vs product {sorted(p_map_to_idx)}"
            )

        def _bond_set_in_mapnums(mol, map_to_idx_reverse: dict[int, int]) -> set[tuple[int, int]]:
            idx_to_map = {idx: mn for mn, idx in map_to_idx_reverse.items()}
            out: set[tuple[int, int]] = set()
            for bond in mol.GetBonds():
                a = idx_to_map[bond.GetBeginAtomIdx()]
                b = idx_to_map[bond.GetEndAtomIdx()]
                out.add((a, b) if a <= b else (b, a))
            return out

        r_bonds = _bond_set_in_mapnums(r_mol, r_map_to_idx)
        p_bonds = _bond_set_in_mapnums(p_mol, p_map_to_idx)

        formed_mapnum_pairs = sorted(p_bonds - r_bonds)
        broken_mapnum_pairs = sorted(r_bonds - p_bonds)

        formed = tuple((r_map_to_idx[a], r_map_to_idx[b]) for a, b in formed_mapnum_pairs)
        broken = tuple((r_map_to_idx[a], r_map_to_idx[b]) for a, b in broken_mapnum_pairs)

        if not formed and not broken:
            raise ValueError(
                "from_reaction_diff: no bond changes detected between R and P "
                "(both have identical connectivity in atom-map space)"
            )
        return cls(formed=formed, broken=broken)
```

- [ ] **Step 4: テストが通ることを確認**

```
pytest tests/test_bond_changes.py -v
```
Expected: 9 passed (8 parametrized + 1 SN2 indices test)

- [ ] **Step 5: Commit**

```bash
git add reactx/bond_changes.py tests/test_bond_changes.py
git commit -m "feat(bond_changes): add from_reaction_diff for auto-detected bond changes (Phase 11 step 1)"
```

---

### Task 2: `endpoint_relax` モジュール新設

UMA で R/P endpoint を minimum まで緩和する関数を提供する。FIRE と BFGS を切替可能にする。

**Files:**
- Create: `reactx/endpoint_relax.py`
- Create: `tests/test_endpoint_relax.py`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_endpoint_relax.py` を新規作成:

```python
"""Tests for reactx.endpoint_relax (Phase 11)."""
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones

from reactx.endpoint_relax import relax_endpoint


def _diatomic_too_close() -> Atoms:
    # LJ minimum is at r = 2^(1/6) * sigma. Default sigma=1 -> r_min ~ 1.122.
    # Start at r=1.0 (compressed), relax should push atoms toward 1.122.
    return Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def test_relax_endpoint_fire_converges_lj_diatomic():
    atoms = _diatomic_too_close()
    relaxed, info = relax_endpoint(
        atoms, LennardJones(), fmax=0.001, max_steps=200, optimizer="FIRE",
    )
    assert info["converged"] is True
    r = float(((relaxed.positions[1] - relaxed.positions[0]) ** 2).sum() ** 0.5)
    assert 1.10 < r < 1.13, f"r={r} not near LJ minimum 1.122"
    assert info["final_fmax"] <= 0.001
    assert info["n_steps"] > 0


def test_relax_endpoint_bfgs_converges_lj_diatomic():
    atoms = _diatomic_too_close()
    relaxed, info = relax_endpoint(
        atoms, LennardJones(), fmax=0.001, max_steps=200, optimizer="BFGS",
    )
    assert info["converged"] is True


def test_relax_endpoint_returns_copy_not_mutated_input():
    atoms = _diatomic_too_close()
    original_positions = atoms.positions.copy()
    relax_endpoint(atoms, LennardJones(), fmax=0.001, max_steps=50)
    # 入力 atoms の positions は変わらない
    assert (atoms.positions == original_positions).all()


def test_relax_endpoint_unknown_optimizer_raises():
    atoms = _diatomic_too_close()
    with pytest.raises(ValueError, match="optimizer"):
        relax_endpoint(atoms, LennardJones(), optimizer="NEWTON")


def test_relax_endpoint_non_converged_returns_info_flag():
    # 1 step では LJ diatomic も収束しない
    atoms = _diatomic_too_close()
    relaxed, info = relax_endpoint(
        atoms, LennardJones(), fmax=1e-9, max_steps=1, optimizer="FIRE",
    )
    assert info["converged"] is False
    assert info["n_steps"] >= 1
```

- [ ] **Step 2: テストが失敗することを確認**

```
pytest tests/test_endpoint_relax.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'reactx.endpoint_relax'`

- [ ] **Step 3: `reactx/endpoint_relax.py` 実装**

```python
"""Single-endpoint geometry relaxation for CI-NEB inputs (Phase 11).

Replaces the AFIR-driven endpoint search of Phase 9/10. The reactant and
product are each minimised independently with a real calculator (UMA in
production, LJ in tests) before being handed to ``run_neb``.
"""
from __future__ import annotations

import logging

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS, FIRE

log = logging.getLogger(__name__)

_OPTIMIZERS = {"FIRE": FIRE, "BFGS": BFGS}


def relax_endpoint(
    atoms: Atoms,
    calc: Calculator,
    *,
    fmax: float = 0.01,
    max_steps: int = 500,
    optimizer: str = "FIRE",
) -> tuple[Atoms, dict]:
    """Relax `atoms` to a local minimum using `calc`. Non-destructive.

    Returns:
        (relaxed_atoms, info) where info = {"converged", "final_fmax", "n_steps"}.
        `converged` is True iff fmax was reached within max_steps.
        Non-convergence is **warned** but not raised; the partially relaxed
        geometry is returned so the caller can decide whether to proceed.
    """
    if optimizer not in _OPTIMIZERS:
        raise ValueError(
            f"optimizer must be one of {sorted(_OPTIMIZERS)}, got {optimizer!r}"
        )
    work = atoms.copy()
    work.calc = calc
    opt_cls = _OPTIMIZERS[optimizer]
    opt = opt_cls(work, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=max_steps))
    forces = work.get_forces()
    final_fmax = float(((forces ** 2).sum(axis=1).max()) ** 0.5)
    info = {
        "converged": converged,
        "final_fmax": final_fmax,
        "n_steps": int(opt.nsteps),
    }
    if not converged:
        log.warning(
            "relax_endpoint(%s) did not converge in %d steps: final_fmax=%.4f > target=%.4f",
            optimizer, max_steps, final_fmax, fmax,
        )
    work.calc = None
    return work, info
```

- [ ] **Step 4: テストが通ることを確認**

```
pytest tests/test_endpoint_relax.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add reactx/endpoint_relax.py tests/test_endpoint_relax.py
git commit -m "feat(endpoint_relax): add minimum-finding endpoint relax module (Phase 11 step 2)"
```

---

### Task 3: TOML schema を新 `[placement]` / `[endpoint_relax]` / `[neb]` に置換

`reactx/config.py` を Phase 11 schema に書き換える。`[afir]` / `[scoring]` / `[sampling]` を検出したら `ConfigError`。`formed` / `broken` top-level も同様に reject。

**Files:**
- Modify: `reactx/config.py` (全面書き直し)
- Create: `tests/test_config_phase11.py`
- Delete (in step 7): `tests/test_config.py`, `tests/test_config_afir.py` ← Task 3 では削除しない (Task 10 でまとめて削除)

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_config_phase11.py` を新規作成:

```python
"""Tests for Phase 11 config schema."""
from pathlib import Path

import pytest

from reactx.config import (
    ConfigError,
    PlacementSection,
    EndpointRelaxSection,
    NEBSection,
    ReactionConfig,
    load_config,
)


def _write_rxn_and_toml(tmp_path: Path, body: str) -> Path:
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n\n  RDKit\n\n  0  0\n")
    (tmp_path / "x.rxn.toml").write_text(body)
    return rxn


def test_minimal_config_only_description(tmp_path):
    rxn = _write_rxn_and_toml(tmp_path, 'description = "minimal"\n')
    cfg = load_config(rxn)
    assert cfg.description == "minimal"
    assert cfg.placement == PlacementSection()  # defaults
    assert cfg.endpoint_relax == EndpointRelaxSection()
    assert cfg.neb == NEBSection()


def test_default_values(tmp_path):
    rxn = _write_rxn_and_toml(tmp_path, 'description = "x"\n')
    cfg = load_config(rxn)
    assert cfg.placement.initial_separation == 4.0
    assert cfg.placement.orientation == "default"
    assert cfg.endpoint_relax.fmax == 0.01
    assert cfg.endpoint_relax.max_steps == 500
    assert cfg.endpoint_relax.optimizer == "FIRE"
    assert cfg.neb.n_images == 11
    assert cfg.neb.fmax == 0.05
    assert cfg.neb.max_steps == 200
    assert cfg.neb.k == 1.0
    assert cfg.neb.climb is True
    assert cfg.neb.pad_frames == 0


def test_all_sections_explicit(tmp_path):
    body = '''description = "all"

[placement]
initial_separation = 5.5
orientation = "endo"

[endpoint_relax]
fmax = 0.005
max_steps = 800
optimizer = "BFGS"

[neb]
n_images = 13
fmax = 0.03
max_steps = 250
k = 0.5
climb = false
pad_frames = 2
'''
    rxn = _write_rxn_and_toml(tmp_path, body)
    cfg = load_config(rxn)
    assert cfg.placement.initial_separation == 5.5
    assert cfg.placement.orientation == "endo"
    assert cfg.endpoint_relax.fmax == 0.005
    assert cfg.endpoint_relax.max_steps == 800
    assert cfg.endpoint_relax.optimizer == "BFGS"
    assert cfg.neb.n_images == 13
    assert cfg.neb.fmax == 0.03
    assert cfg.neb.max_steps == 250
    assert cfg.neb.k == 0.5
    assert cfg.neb.climb is False
    assert cfg.neb.pad_frames == 2


@pytest.mark.parametrize("obsolete", [
    "[afir]\nalpha_formed = 1.0\nalpha_broken = 1.0\nmax_relax_steps = 100\n",
    "[scoring]\nr_broken_threshold = 3.0\n",
    "[sampling]\nn_candidates = 32\n",
    "formed = [[1, 2]]\n",
    "broken = [[1, 2]]\n",
    "[restraints]\nk_form = 1.0\n",
    "[prescreen]\nn_angles = 8\n",
])
def test_obsolete_keys_rejected(tmp_path, obsolete):
    body = 'description = "x"\n' + obsolete
    rxn = _write_rxn_and_toml(tmp_path, body)
    with pytest.raises(ConfigError, match="Phase 11"):
        load_config(rxn)


@pytest.mark.parametrize("bad_val,scope,key", [
    (0.0, "placement", "initial_separation"),
    (-1.0, "placement", "initial_separation"),
    ("invalid_orient", "placement", "orientation"),
    (0.0, "endpoint_relax", "fmax"),
    (-0.1, "endpoint_relax", "fmax"),
    (0, "endpoint_relax", "max_steps"),
    ("Newton", "endpoint_relax", "optimizer"),
    (2, "neb", "n_images"),
    (0.0, "neb", "fmax"),
    (0, "neb", "max_steps"),
    (0.0, "neb", "k"),
    (-1, "neb", "pad_frames"),
])
def test_invalid_values_rejected(tmp_path, bad_val, scope, key):
    body = f'description = "x"\n[{scope}]\n{key} = {bad_val!r}\n'
    rxn = _write_rxn_and_toml(tmp_path, body)
    with pytest.raises(ConfigError):
        load_config(rxn)


def test_missing_description_rejected(tmp_path):
    rxn = _write_rxn_and_toml(tmp_path, "[placement]\ninitial_separation = 3.0\n")
    with pytest.raises(ConfigError, match="description"):
        load_config(rxn)


def test_missing_toml_raises_file_not_found(tmp_path):
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n")
    with pytest.raises(FileNotFoundError):
        load_config(rxn)


def test_unknown_top_level_key_rejected(tmp_path):
    body = 'description = "x"\nweird_key = 42\n'
    rxn = _write_rxn_and_toml(tmp_path, body)
    with pytest.raises(ConfigError, match="unknown"):
        load_config(rxn)
```

- [ ] **Step 2: テストが失敗することを確認**

```
pytest tests/test_config_phase11.py -v
```
Expected: FAIL (current `config.py` exposes `AFIRSection` / `ScoringSection`, not the Phase 11 sections)

- [ ] **Step 3: `reactx/config.py` を Phase 11 schema に全面書き直し**

`reactx/config.py` 全体を以下に置換:

```python
"""Per-reaction sidecar TOML config (Phase 11).

Schema sections: [placement], [endpoint_relax], [neb].
Only `description` (top-level) is required; all sections are optional.
Spec: docs/superpowers/specs/2026-05-15-phase-11-ci-neb-design.md §4
"""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    """Raised when `<rxn_path>.toml` violates the Phase 11 schema."""


@dataclass(frozen=True)
class PlacementSection:
    initial_separation: float = 4.0
    orientation: str = "default"  # "default" | "endo" | "exo"


@dataclass(frozen=True)
class EndpointRelaxSection:
    fmax: float = 0.01
    max_steps: int = 500
    optimizer: str = "FIRE"  # "FIRE" | "BFGS"


@dataclass(frozen=True)
class NEBSection:
    n_images: int = 11
    fmax: float = 0.05
    max_steps: int = 200
    k: float = 1.0
    climb: bool = True
    pad_frames: int = 0


@dataclass(frozen=True)
class ReactionConfig:
    description: str
    placement: PlacementSection
    endpoint_relax: EndpointRelaxSection
    neb: NEBSection


_TOP_LEVEL_KEYS = {"description", "placement", "endpoint_relax", "neb"}
_TOP_LEVEL_REQUIRED = {"description"}
_OBSOLETE_TOP_LEVEL = {
    "afir", "scoring", "sampling", "restraints", "prescreen",
    "formed", "broken", "k_form", "k_broken", "r_broken", "r_form",
}
_PLACEMENT_KEYS = {"initial_separation", "orientation"}
_ENDPOINT_KEYS = {"fmax", "max_steps", "optimizer"}
_NEB_KEYS = {"n_images", "fmax", "max_steps", "k", "climb", "pad_frames"}
_VALID_ORIENTATIONS = {"default", "endo", "exo"}
_VALID_OPTIMIZERS = {"FIRE", "BFGS"}


def sidecar_path(rxn_path: Path) -> Path:
    return rxn_path.parent / (rxn_path.name + ".toml")


def load_config(rxn_path: Path) -> ReactionConfig:
    toml_path = sidecar_path(rxn_path)
    if not toml_path.is_file():
        raise FileNotFoundError(
            f"sidecar TOML not found: expected {toml_path} alongside {rxn_path}"
        )
    raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    return _validate(raw, source=str(toml_path))


def _validate(raw: dict, *, source: str) -> ReactionConfig:
    for k in _OBSOLETE_TOP_LEVEL:
        if k in raw:
            raise ConfigError(
                f"{source}: '{k}' is removed in Phase 11. "
                f"Migrate to [placement] / [endpoint_relax] / [neb] schema. "
                f"See spec docs/superpowers/specs/2026-05-15-phase-11-ci-neb-design.md §4"
            )
    _check_keys(raw, _TOP_LEVEL_KEYS, _TOP_LEVEL_REQUIRED,
                scope="<top>", source=source)
    description = raw["description"]
    if not isinstance(description, str) or not description.strip():
        raise ConfigError(f"{source}: 'description' must be a non-empty string")

    placement = _build_placement(raw.get("placement", {}), source=source)
    endpoint = _build_endpoint(raw.get("endpoint_relax", {}), source=source)
    neb = _build_neb(raw.get("neb", {}), source=source)

    return ReactionConfig(
        description=description,
        placement=placement,
        endpoint_relax=endpoint,
        neb=neb,
    )


def _build_placement(raw: dict, *, source: str) -> PlacementSection:
    _check_keys(raw, _PLACEMENT_KEYS, set(), scope="[placement]", source=source)
    sep = raw.get("initial_separation", 4.0)
    if not isinstance(sep, (int, float)) or isinstance(sep, bool) or not math.isfinite(sep) or sep <= 0:
        raise ConfigError(
            f"{source}: [placement].initial_separation must be a positive finite number (got {sep!r})"
        )
    orient = raw.get("orientation", "default")
    if orient not in _VALID_ORIENTATIONS:
        raise ConfigError(
            f"{source}: [placement].orientation must be one of {sorted(_VALID_ORIENTATIONS)} (got {orient!r})"
        )
    return PlacementSection(initial_separation=float(sep), orientation=orient)


def _build_endpoint(raw: dict, *, source: str) -> EndpointRelaxSection:
    _check_keys(raw, _ENDPOINT_KEYS, set(), scope="[endpoint_relax]", source=source)
    fmax = raw.get("fmax", 0.01)
    if not isinstance(fmax, (int, float)) or isinstance(fmax, bool) or not math.isfinite(fmax) or fmax <= 0:
        raise ConfigError(
            f"{source}: [endpoint_relax].fmax must be a positive finite number (got {fmax!r})"
        )
    steps = raw.get("max_steps", 500)
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ConfigError(
            f"{source}: [endpoint_relax].max_steps must be a positive integer (got {steps!r})"
        )
    opt = raw.get("optimizer", "FIRE")
    if opt not in _VALID_OPTIMIZERS:
        raise ConfigError(
            f"{source}: [endpoint_relax].optimizer must be one of {sorted(_VALID_OPTIMIZERS)} (got {opt!r})"
        )
    return EndpointRelaxSection(fmax=float(fmax), max_steps=int(steps), optimizer=opt)


def _build_neb(raw: dict, *, source: str) -> NEBSection:
    _check_keys(raw, _NEB_KEYS, set(), scope="[neb]", source=source)
    n_images = raw.get("n_images", 11)
    if not isinstance(n_images, int) or isinstance(n_images, bool) or n_images < 3:
        raise ConfigError(
            f"{source}: [neb].n_images must be an integer >= 3 (got {n_images!r})"
        )
    fmax = raw.get("fmax", 0.05)
    if not isinstance(fmax, (int, float)) or isinstance(fmax, bool) or not math.isfinite(fmax) or fmax <= 0:
        raise ConfigError(
            f"{source}: [neb].fmax must be a positive finite number (got {fmax!r})"
        )
    steps = raw.get("max_steps", 200)
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ConfigError(
            f"{source}: [neb].max_steps must be a positive integer (got {steps!r})"
        )
    k = raw.get("k", 1.0)
    if not isinstance(k, (int, float)) or isinstance(k, bool) or not math.isfinite(k) or k <= 0:
        raise ConfigError(
            f"{source}: [neb].k must be a positive finite number (got {k!r})"
        )
    climb = raw.get("climb", True)
    if not isinstance(climb, bool):
        raise ConfigError(
            f"{source}: [neb].climb must be a bool (got {climb!r})"
        )
    pad = raw.get("pad_frames", 0)
    if not isinstance(pad, int) or isinstance(pad, bool) or pad < 0:
        raise ConfigError(
            f"{source}: [neb].pad_frames must be a non-negative integer (got {pad!r})"
        )
    return NEBSection(
        n_images=int(n_images), fmax=float(fmax), max_steps=int(steps),
        k=float(k), climb=climb, pad_frames=int(pad),
    )


def _check_keys(raw: dict, allowed: set[str], required: set[str], *,
                scope: str, source: str) -> None:
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{source}: '{scope}' must be a table, got {type(raw).__name__}"
        )
    missing = required - raw.keys()
    if missing:
        raise ConfigError(
            f"{source}: {scope} missing required keys: {sorted(missing)}"
        )
    extra = set(raw.keys()) - allowed
    if extra:
        raise ConfigError(
            f"{source}: {scope} contains unknown keys: {sorted(extra)}"
        )
```

- [ ] **Step 4: テストが通ることを確認**

```
pytest tests/test_config_phase11.py -v
```
Expected: all tests pass (~ 25 cases)

- [ ] **Step 5: Commit (旧 test は意図的に壊れた状態を許容)**

```bash
git add reactx/config.py tests/test_config_phase11.py
git commit -m "feat(config): replace [afir]/[scoring]/[sampling] with Phase 11 schema (Phase 11 step 3)"
```

注: `tests/test_config.py`, `tests/test_config_afir.py`, `tests/test_re*.py` の AFIR 系テストは Task 10 で削除する。本タスク完了時点で `pytest` 全体は赤になるが、新 schema テストは緑であることを確認すれば次タスクへ進んでよい。

---

### Task 4: `simple_placement` で `placement.py` を縮小

Fibonacci sphere / blocking / multi-anchor / endo-exo 展開を全廃し、反応点 anchor を `+z` 方向に `initial_separation` Å 離す 1 関数のみに置き換える。

**Files:**
- Modify: `reactx/placement.py` (全面書き直し)
- Modify: `tests/test_placement.py` (新 API のテストに書き換え)

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_placement.py` を以下で **完全置換** (旧 valid_placements テストは破棄):

```python
"""Tests for Phase 11 simple_placement."""
import numpy as np
import pytest
from rdkit import Chem

from reactx.bond_changes import BondChanges
from reactx.embed3d import embed_fragments_to_positions
from reactx.placement import build_atoms_from_positions, simple_placement
from reactx.rxn_parser import parse_rxn
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _r_inputs(rxn_name: str):
    r_mol, p_mol, heavy_mapping = parse_rxn(EXAMPLES / f"{rxn_name}.rxn")
    mol_h, _, base_positions = embed_fragments_to_positions(r_mol, seed=0)
    bc = BondChanges.from_reaction_diff(r_mol, p_mol)
    return mol_h, base_positions, bc, heavy_mapping


def test_simple_placement_unimolecular_passthrough():
    """sn1_dissoc: R 側は 1 fragment、何もせず passthrough."""
    mol_h, base_positions, bc, _ = _r_inputs("sn1_dissoc")
    out = simple_placement(
        mol_h, base_positions, bc,
        initial_separation=4.0, side="reactant", orientation="default",
    )
    assert np.allclose(out, base_positions)


def test_simple_placement_bimolecular_separation_at_least_target():
    """SN2: substrate (CH3Cl) と incoming (OH-) を 4 Å 離す."""
    mol_h, base_positions, bc, _ = _r_inputs("sn2")
    out = simple_placement(
        mol_h, base_positions, bc,
        initial_separation=4.0, side="reactant", orientation="default",
    )
    # formed = C-O so anchor pair = (C_idx, O_idx). Distance >= 4.0 Å.
    a, b = bc.formed[0]
    r = float(np.linalg.norm(out[b] - out[a]))
    assert r >= 4.0 - 1e-6, f"anchor distance {r} < target 4.0"
    # 結合相手の incoming fragment 全体は anchor から 4Å 程度離れている (重心 distance)
    frag_indices = Chem.GetMolFrags(mol_h)
    assert len(frag_indices) == 2
    com0 = out[list(frag_indices[0])].mean(axis=0)
    com1 = out[list(frag_indices[1])].mean(axis=0)
    assert float(np.linalg.norm(com1 - com0)) > 2.0


def test_simple_placement_product_uses_broken_anchor(tmp_path):
    """P 側 placement では bond_changes.broken を anchor として使う."""
    mol_h_r, base_positions_r, bc, heavy_mapping = _r_inputs("sn2")
    # 同じ R 側情報を P 側仕様で呼び出す: side="product" + heavy_mapping
    r_mol, p_mol, _ = parse_rxn(EXAMPLES / "sn2.rxn")
    mol_h_p, _, base_positions_p = embed_fragments_to_positions(p_mol, seed=0)
    out = simple_placement(
        mol_h_p, base_positions_p, bc,
        initial_separation=4.0, side="product", orientation="default",
        heavy_mapping=heavy_mapping,
    )
    # SN2 product: Cl- が CH3OH から 4Å 離れる
    # bc.broken = [(R_C_idx, R_Cl_idx)]; P 側では heavy_mapping で変換
    a_r, b_r = bc.broken[0]
    a_p = heavy_mapping[a_r]
    b_p = heavy_mapping[b_r]
    r = float(np.linalg.norm(out[b_p] - out[a_p]))
    assert r >= 4.0 - 1e-6


def test_simple_placement_invalid_side_raises():
    mol_h, base_positions, bc, _ = _r_inputs("sn2")
    with pytest.raises(ValueError, match="side"):
        simple_placement(
            mol_h, base_positions, bc,
            initial_separation=4.0, side="other", orientation="default",
        )


def test_simple_placement_product_without_heavy_mapping_raises():
    mol_h, base_positions, bc, _ = _r_inputs("sn2")
    with pytest.raises(ValueError, match="heavy_mapping"):
        simple_placement(
            mol_h, base_positions, bc,
            initial_separation=4.0, side="product", orientation="default",
        )


def test_build_atoms_from_positions_preserves_symbols():
    mol_h, base_positions, _, _ = _r_inputs("sn2")
    atoms = build_atoms_from_positions(mol_h, base_positions)
    assert len(atoms) == mol_h.GetNumAtoms()
    assert atoms.get_chemical_symbols() == [a.GetSymbol() for a in mol_h.GetAtoms()]
```

- [ ] **Step 2: テストが失敗することを確認**

```
pytest tests/test_placement.py -v
```
Expected: FAIL (旧 `simple_placement` は未実装、旧 `valid_placements` は API 違い)

- [ ] **Step 3: `reactx/placement.py` を全面書き直し**

`reactx/placement.py` を以下で完全置換:

```python
"""Simplified fragment placement for Phase 11 Pure CI-NEB pipeline.

Replaces the Phase 7 Fibonacci sphere / vdW shadow blocking / multi-anchor
machinery with a single deterministic placement: anchor atoms are placed
``initial_separation`` Å apart along +z. Subsequent endpoint relaxation
finds the physically correct geometry (e.g. SN2 Walden inversion angle).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from ase import Atoms
from rdkit import Chem

from reactx.bond_changes import BondChanges


def build_atoms_from_positions(mol_h: Chem.Mol, positions: np.ndarray) -> Atoms:
    symbols = [atom.GetSymbol() for atom in mol_h.GetAtoms()]
    return Atoms(symbols=symbols, positions=positions)


def simple_placement(
    mol_h: Chem.Mol,
    base_positions: np.ndarray,
    bond_changes: BondChanges,
    *,
    initial_separation: float,
    side: Literal["reactant", "product"],
    orientation: str = "default",
    heavy_mapping: dict[int, int] | None = None,
) -> np.ndarray:
    """Return per-atom 3D positions with fragments separated along +z.

    Args:
        mol_h: RDKit Mol AFTER ``Chem.AddHs``. Atom order matches base_positions.
        base_positions: per-atom 3D coords from ``embed_fragments_to_positions``
            (multiple fragments may overlap at the origin).
        bond_changes: from ``BondChanges.from_reaction_diff(r_mol, p_mol)``,
            holding 0-based atom indices in REACTANT atom order.
        initial_separation: target anchor-pair distance [Å].
        side: ``"reactant"`` uses ``bond_changes.formed`` as anchors;
            ``"product"`` uses ``bond_changes.broken`` after translating
            indices via ``heavy_mapping``.
        orientation: ``"default"`` / ``"endo"`` / ``"exo"``. Currently only
            affects bridges>=2 placements; unimolecular and bridges=1 ignore.
        heavy_mapping: required when ``side="product"``. Maps R-side heavy
            atom index -> P-side heavy atom index.

    Returns:
        New positions array (input not mutated).
    """
    if side not in ("reactant", "product"):
        raise ValueError(f"side must be 'reactant' or 'product', got {side!r}")
    if side == "product" and heavy_mapping is None:
        raise ValueError("heavy_mapping is required when side='product'")

    frag_indices = Chem.GetMolFrags(mol_h)
    positions = base_positions.copy()

    if len(frag_indices) == 1:
        return positions

    anchors_r = bond_changes.formed if side == "reactant" else bond_changes.broken
    if not anchors_r:
        raise ValueError(
            f"simple_placement(side={side!r}): no anchor pairs available "
            f"({'formed' if side == 'reactant' else 'broken'} is empty)"
        )
    if side == "product":
        anchors = [(heavy_mapping[a], heavy_mapping[b]) for a, b in anchors_r]
    else:
        anchors = list(anchors_r)

    bridges = _count_bridges(anchors, frag_indices)
    if bridges == 1:
        return _place_bridges_one(positions, anchors, frag_indices, initial_separation)
    if bridges == 2:
        return _place_bridges_two(positions, anchors, frag_indices, initial_separation, orientation)
    raise NotImplementedError(
        f"simple_placement supports bridges in {{1, 2}}, got {bridges}"
    )


def _count_bridges(anchors: list[tuple[int, int]],
                   frag_indices: tuple[tuple[int, ...], ...]) -> int:
    """Count how many anchor pairs cross fragment boundaries."""
    atom_to_frag = {}
    for fi, atoms in enumerate(frag_indices):
        for a in atoms:
            atom_to_frag[a] = fi
    bridges = 0
    for i, j in anchors:
        if atom_to_frag[i] != atom_to_frag[j]:
            bridges += 1
    return bridges


def _place_bridges_one(
    positions: np.ndarray,
    anchors: list[tuple[int, int]],
    frag_indices: tuple[tuple[int, ...], ...],
    initial_separation: float,
) -> np.ndarray:
    """Place 2 fragments so the single bridge anchor sits at +z * separation."""
    atom_to_frag = {a: fi for fi, atoms in enumerate(frag_indices) for a in atoms}
    i, j = anchors[0]
    fi_i = atom_to_frag[i]
    # Identify substrate (any fragment containing both atoms of *any* intra-fragment anchor)
    # When there is only one bridging anchor, define substrate as the larger fragment.
    sizes = [len(f) for f in frag_indices]
    substrate_frag = int(np.argmax(sizes))
    incoming_frag = 1 - substrate_frag if len(frag_indices) == 2 else None
    if incoming_frag is None:
        raise NotImplementedError("bridges=1 only supports exactly 2 fragments")
    if fi_i == substrate_frag:
        sub_anchor, inc_anchor = i, j
    else:
        sub_anchor, inc_anchor = j, i

    # Translate substrate: center sub_anchor at origin
    sub_atoms = list(frag_indices[substrate_frag])
    positions[sub_atoms] -= positions[sub_anchor]
    # Translate incoming: center inc_anchor at +z * initial_separation
    inc_atoms = list(frag_indices[incoming_frag])
    target = np.array([0.0, 0.0, initial_separation])
    positions[inc_atoms] += target - positions[inc_anchor]
    return positions


def _place_bridges_two(
    positions: np.ndarray,
    anchors: list[tuple[int, int]],
    frag_indices: tuple[tuple[int, ...], ...],
    initial_separation: float,
    orientation: str,
) -> np.ndarray:
    """Place 2 fragments face-to-face for cycloaddition (Diels-Alder).

    For Phase 11 we use a deterministic face alignment: each fragment's
    two anchor atoms define a vector; we orient both fragments so those
    vectors are parallel along x, then offset along z by ``initial_separation``.
    `orientation` ("default" | "endo" | "exo") flips the relative rotation
    of the second fragment by 180° about z for "exo" vs the others.
    """
    if len(frag_indices) != 2:
        raise NotImplementedError(
            f"bridges=2 placement supports exactly 2 fragments, got {len(frag_indices)}"
        )
    atom_to_frag = {a: fi for fi, atoms in enumerate(frag_indices) for a in atoms}
    (a1, b1), (a2, b2) = anchors
    # Group each anchor end by fragment.
    frag_a1 = atom_to_frag[a1]
    if frag_a1 == atom_to_frag[a2]:
        frag0_anchors = [a1, a2]
        frag1_anchors = [b1, b2]
    else:
        frag0_anchors = [a1, b2]
        frag1_anchors = [b1, a2]

    def _orient_to_x_axis(frag_atoms: list[int], anchor_atoms: list[int],
                          flip: bool = False) -> None:
        centroid = positions[anchor_atoms].mean(axis=0)
        positions[frag_atoms] -= centroid
        v = positions[anchor_atoms[1]] - positions[anchor_atoms[0]]
        v_norm = float(np.linalg.norm(v))
        if v_norm < 1e-6:
            return
        v_unit = v / v_norm
        target = np.array([1.0, 0.0, 0.0])
        axis = np.cross(v_unit, target)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm > 1e-6:
            angle = float(np.arccos(np.clip(v_unit @ target, -1.0, 1.0)))
            axis_unit = axis / axis_norm
            R = _rotation_matrix(axis_unit, angle)
            positions[frag_atoms] = positions[frag_atoms] @ R.T
        if flip:
            R_flip = _rotation_matrix(np.array([0.0, 0.0, 1.0]), np.pi)
            positions[frag_atoms] = positions[frag_atoms] @ R_flip.T

    frag0_atoms = list(frag_indices[0])
    frag1_atoms = list(frag_indices[1])
    flip_frag1 = (orientation == "exo")
    _orient_to_x_axis(frag0_atoms, frag0_anchors, flip=False)
    _orient_to_x_axis(frag1_atoms, frag1_anchors, flip=flip_frag1)
    # Offset fragment 1 along +z
    positions[frag1_atoms] += np.array([0.0, 0.0, initial_separation])
    return positions


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    return np.array([
        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
    ])
```

- [ ] **Step 4: テストが通ることを確認**

```
pytest tests/test_placement.py -v
```
Expected: all tests pass (~6)

- [ ] **Step 5: Commit**

```bash
git add reactx/placement.py tests/test_placement.py
git commit -m "refactor(placement): replace 4π sampling with simple_placement (Phase 11 step 4)"
```

---

### Task 5: `align.py` の 1+1 限定を撤廃

`bond_changes_product` 引数を取らずに、`heavy_mapping` + `h_groups` のみで動くようにする。実体は既に汎用なので、引数シグネチャの確認と既存 test の更新のみ。

**Files:**
- Modify: `reactx/align.py` (シグネチャ確認、変更は実質なし)
- Modify: `tests/test_align.py` (引数を heavy_mapping 中心に整理)

- [ ] **Step 1: 現状確認**

`reactx/align.py` の `align_product_to_reactant` シグネチャを読む。既に `bond_changes_product` 引数は **ない** (`heavy_mapping` と `h_groups` のみ)。**実装はそのまま、Task 5 は test の整理のみで終わる。**

- [ ] **Step 2: 既存 test 確認**

```
pytest tests/test_align.py -v
```

すべて通ることを確認。

- [ ] **Step 3: multi-bond ケース (E2) の追加 test を書く**

`tests/test_align.py` の末尾に追加:

```python
def test_align_works_on_multi_bond_reaction():
    """E2: 1 formed + 2 broken でも align は heavy_mapping だけで動く."""
    from pathlib import Path
    from rdkit import Chem
    from reactx.embed3d import embed_fragments_to_positions
    from reactx.placement import build_atoms_from_positions
    from reactx.rxn_parser import parse_rxn, heavy_to_hydrogen_groups
    from reactx.align import align_product_to_reactant

    EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
    r_mol, p_mol, heavy_mapping = parse_rxn(EXAMPLES / "e2.rxn")
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    _, _, pos_r = embed_fragments_to_positions(r_mol, seed=0)
    _, _, pos_p = embed_fragments_to_positions(p_mol, seed=0)
    atoms_r = build_atoms_from_positions(r_h, pos_r)
    atoms_p = build_atoms_from_positions(p_h, pos_p)
    rH = heavy_to_hydrogen_groups(r_h)
    pH = heavy_to_hydrogen_groups(p_h)
    aligned = align_product_to_reactant(atoms_r, atoms_p, heavy_mapping, rH, pH)
    assert len(aligned) == len(atoms_r)
    assert aligned.get_chemical_symbols() == atoms_r.get_chemical_symbols()
```

- [ ] **Step 4: テストが通ることを確認**

```
pytest tests/test_align.py -v
```
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add tests/test_align.py
git commit -m "test(align): verify align works on multi-bond E2 (Phase 11 step 5)"
```

---

### Task 6: `neb.py` の戻り値に `image_atoms` を追加

CLI 側で `ase.io.read` で書き戻し / 読み直しを省略するため、`run_neb` の戻り値に image Atoms list を追加する。

**Files:**
- Modify: `reactx/neb.py`
- Create: `tests/test_neb.py`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_neb.py` を新規作成:

```python
"""Tests for reactx.neb.run_neb return value extension (Phase 11)."""
from pathlib import Path

from ase import Atoms
from ase.calculators.lj import LennardJones

from reactx.neb import run_neb


def _lj_endpoint(r: float) -> Atoms:
    return Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [r, 0.0, 0.0]])


def test_run_neb_returns_image_atoms(tmp_path: Path):
    reactant = _lj_endpoint(1.2)
    product = _lj_endpoint(1.4)
    xyz = tmp_path / "traj.xyz"
    info = run_neb(
        reactant, product, calculator=LennardJones(),
        n_images=5, output_xyz=xyz,
        fmax=0.5, max_steps=10,
    )
    assert "image_atoms" in info
    assert isinstance(info["image_atoms"], list)
    assert len(info["image_atoms"]) == 5
    for img in info["image_atoms"]:
        assert isinstance(img, Atoms)
        assert len(img) == 2
```

- [ ] **Step 2: テストが失敗することを確認**

```
pytest tests/test_neb.py -v
```
Expected: FAIL — `info["image_atoms"]` not in current return value

- [ ] **Step 3: `reactx/neb.py` の戻り値拡張**

`reactx/neb.py` の `return { ... }` を以下に置換:

```python
    return {
        "n_images": n_images,
        "converged": converged,
        "final_fmax": final_fmax,
        "image_energies": image_energies,
        "pad_frames": pad_frames,
        "image_atoms": [img.copy() for img in images],
    }
```

(image_atoms は calc を持ったままだとシリアライズで困るので `.copy()` してから返す。`.copy()` した Atoms は calc が外れる。)

- [ ] **Step 4: テストが通ることを確認**

```
pytest tests/test_neb.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add reactx/neb.py tests/test_neb.py
git commit -m "feat(neb): include image_atoms in run_neb return value (Phase 11 step 6)"
```

---

### Task 7: `cli.py` を Pure CI-NEB パイプラインに全面書き直し

trial sweep 撤廃、AFIR import 削除、`--neb-refine` / `--neb-images` / `--seed` / `--traj-stride` / `--relax-fmax` フラグ撤廃。新 `meta.json` 構造に置換。

**Files:**
- Modify: `reactx/cli.py` (全面書き直し)

- [ ] **Step 1: `reactx/cli.py` を以下で完全置換**

```python
"""CLI entry point: reactx run <rxn> -o <outdir> [options] (Phase 11)."""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

from ase.io import write
from rdkit import Chem

from reactx.align import align_product_to_reactant
from reactx.bond_changes import BondChanges
from reactx.calculators import make_calculator
from reactx.config import ConfigError, ReactionConfig, load_config
from reactx.embed3d import embed_fragments_to_positions
from reactx.endpoint_relax import relax_endpoint
from reactx.neb import run_neb
from reactx.placement import build_atoms_from_positions, simple_placement
from reactx.rxn_parser import heavy_to_hydrogen_groups, parse_rxn

log = logging.getLogger("reactx")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reactx")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Run the Pure CI-NEB pipeline on a .rxn file")
    run.add_argument("rxn_path", type=Path)
    run.add_argument("-o", "--output", type=Path, required=True)
    run.add_argument("--backend", choices=["uma", "lj"], default="uma")
    run.add_argument("--model", type=str, default="uma-m-1p1",
                     help="UMA model name (uma-m-1p1, uma-s-1p2, ...)")
    run.add_argument("--render", action="store_true",
                     help="Invoke blender/render.py after the pipeline")
    run.add_argument("--blender-exe", type=str, default="blender")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _cmd_run(args)


def _configure_reactx_logging() -> None:
    if log.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[reactx] %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def _sanitize_for_json(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _check_hf_auth() -> int:
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import LocalTokenNotFoundError
    except ImportError as exc:
        log.error("UMA backend requires huggingface_hub: %s", exc)
        return 1
    try:
        HfApi().whoami()
    except LocalTokenNotFoundError:
        log.error("Hugging Face token not found. Run `hf auth login` first.")
        return 1
    except Exception as exc:  # noqa: BLE001
        log.error("Hugging Face authentication check failed (%s: %s).",
                  type(exc).__name__, exc)
        return 1
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    _configure_reactx_logging()
    if not args.rxn_path.exists():
        log.error("Error: .rxn not found: %s", args.rxn_path)
        return 1
    try:
        cfg = load_config(args.rxn_path)
    except ConfigError as exc:
        log.error("config error: %s", exc)
        return 2
    except FileNotFoundError as exc:
        log.error("%s", exc)
        log.error("A sidecar TOML config is REQUIRED. See examples/sn2.rxn.toml.")
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
    try:
        bond_changes = BondChanges.from_reaction_diff(r_mol, p_mol)
    except ValueError as exc:
        log.error("bond_changes inference failed: %s", exc)
        return 2

    log.info("description=%s placement.initial_separation=%.2f endpoint_relax.fmax=%.4f neb.n_images=%d",
             cfg.description, cfg.placement.initial_separation,
             cfg.endpoint_relax.fmax, cfg.neb.n_images)

    model_kwargs = {"model_name": args.model} if args.backend == "uma" else {}
    calc = make_calculator(args.backend, **model_kwargs)

    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)

    # === R side ===
    try:
        _, _, pos_r = embed_fragments_to_positions(r_mol, seed=0)
    except RuntimeError as exc:
        log.error("R-side per-fragment embed failed: %s", exc)
        return 1
    try:
        pos_r = simple_placement(
            r_h, pos_r, bond_changes,
            initial_separation=cfg.placement.initial_separation,
            side="reactant", orientation=cfg.placement.orientation,
        )
    except (ValueError, NotImplementedError) as exc:
        log.error("R-side placement failed: %s", exc)
        return 2
    atoms_r = build_atoms_from_positions(r_h, pos_r)
    log.info("R-side endpoint relax: optimizer=%s fmax=%.4f max_steps=%d",
             cfg.endpoint_relax.optimizer, cfg.endpoint_relax.fmax,
             cfg.endpoint_relax.max_steps)
    atoms_r_relaxed, info_r = relax_endpoint(
        atoms_r, calc,
        fmax=cfg.endpoint_relax.fmax,
        max_steps=cfg.endpoint_relax.max_steps,
        optimizer=cfg.endpoint_relax.optimizer,
    )
    log.info("R-side relax: converged=%s n_steps=%d final_fmax=%.4f",
             info_r["converged"], info_r["n_steps"], info_r["final_fmax"])

    # === P side ===
    try:
        _, _, pos_p = embed_fragments_to_positions(p_mol, seed=0)
    except RuntimeError as exc:
        log.error("P-side per-fragment embed failed: %s", exc)
        return 1
    try:
        pos_p = simple_placement(
            p_h, pos_p, bond_changes,
            initial_separation=cfg.placement.initial_separation,
            side="product", orientation=cfg.placement.orientation,
            heavy_mapping=heavy_mapping,
        )
    except (ValueError, NotImplementedError) as exc:
        log.error("P-side placement failed: %s", exc)
        return 2
    atoms_p = build_atoms_from_positions(p_h, pos_p)
    log.info("P-side endpoint relax")
    atoms_p_relaxed, info_p = relax_endpoint(
        atoms_p, calc,
        fmax=cfg.endpoint_relax.fmax,
        max_steps=cfg.endpoint_relax.max_steps,
        optimizer=cfg.endpoint_relax.optimizer,
    )
    log.info("P-side relax: converged=%s n_steps=%d final_fmax=%.4f",
             info_p["converged"], info_p["n_steps"], info_p["final_fmax"])

    # === Align ===
    try:
        atoms_p_aligned = align_product_to_reactant(
            atoms_r_relaxed, atoms_p_relaxed, heavy_mapping,
            heavy_to_hydrogen_groups(r_h), heavy_to_hydrogen_groups(p_h),
        )
    except ValueError as exc:
        log.error("align failed: %s", exc)
        return 1

    # === NEB ===
    xyz = args.output / "trajectory.xyz"
    log.info("running CI-NEB: n_images=%d max_steps=%d", cfg.neb.n_images, cfg.neb.max_steps)
    neb_info = run_neb(
        atoms_r_relaxed, atoms_p_aligned, calculator=calc,
        n_images=cfg.neb.n_images, output_xyz=xyz,
        fmax=cfg.neb.fmax, max_steps=cfg.neb.max_steps,
        climb=cfg.neb.climb, pad_frames=cfg.neb.pad_frames,
    )
    log.info("NEB: converged=%s final_fmax=%.4f", neb_info["converged"], neb_info["final_fmax"])

    # === energies.json ===
    (args.output / "energies.json").write_text(
        json.dumps([float(e) for e in neb_info["image_energies"]])
    )

    # === meta.json ===
    meta = {
        "backend": args.backend,
        "description": cfg.description,
        "wall_clock_seconds": float(time.monotonic() - t_start),
        "endpoint_relax_r": {
            "converged": info_r["converged"],
            "final_fmax": info_r["final_fmax"],
            "n_steps": info_r["n_steps"],
        },
        "endpoint_relax_p": {
            "converged": info_p["converged"],
            "final_fmax": info_p["final_fmax"],
            "n_steps": info_p["n_steps"],
        },
        "neb": {
            "n_images": neb_info["n_images"],
            "converged": neb_info["converged"],
            "final_fmax": neb_info["final_fmax"],
            "image_energies": [float(e) for e in neb_info["image_energies"]],
            "pad_frames": neb_info["pad_frames"],
        },
        "effective_params": {
            "placement": {
                "initial_separation": cfg.placement.initial_separation,
                "orientation": cfg.placement.orientation,
            },
            "endpoint_relax": {
                "fmax": cfg.endpoint_relax.fmax,
                "max_steps": cfg.endpoint_relax.max_steps,
                "optimizer": cfg.endpoint_relax.optimizer,
            },
            "neb": {
                "n_images": cfg.neb.n_images,
                "fmax": cfg.neb.fmax,
                "max_steps": cfg.neb.max_steps,
                "k": cfg.neb.k,
                "climb": cfg.neb.climb,
                "pad_frames": cfg.neb.pad_frames,
            },
        },
    }
    (args.output / "meta.json").write_text(json.dumps(_sanitize_for_json(meta), indent=2))

    if args.render:
        rc = _invoke_blender(args, xyz)
        if rc != 0:
            return rc

    log.info("OK: wrote %s", xyz)
    return 0


def _invoke_blender(args: argparse.Namespace, xyz: Path) -> int:
    import shutil
    import subprocess
    if shutil.which(args.blender_exe) is None:
        log.error("Blender executable not found on PATH: %s", args.blender_exe)
        return 1
    script = Path(__file__).resolve().parent.parent / "blender" / "render.py"
    blend = args.output / "scene.blend"
    cmd = [args.blender_exe, "--background", "--python", str(script),
           "--", str(xyz), str(blend)]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error("Error: blender exited with code %d", result.returncode)
        return 1
    return 0
```

- [ ] **Step 2: import 整合性確認**

```
python -c "from reactx.cli import main, build_parser; print('OK')"
```
Expected: `OK` (現時点で artificial_force / path_relax / scoring は **まだ削除していない** のでこの import チェックは通る; 削除は Task 8)

- [ ] **Step 3: Commit**

```bash
git add reactx/cli.py
git commit -m "refactor(cli): replace AFIR pipeline with Pure CI-NEB (Phase 11 step 7)"
```

---

### Task 8: 旧 AFIR / scoring / path_relax モジュールと対応 test を削除

`cli.py` から参照が消えたので、obsolete モジュールを物理削除する。同時に obsolete test も削除して `pytest` が AttributeError を出さない状態にする。

**Files:**
- Delete: `reactx/artificial_force.py`
- Delete: `reactx/path_relax.py`
- Delete: `reactx/scoring.py`
- Delete: `tests/test_afir_constraint.py`
- Delete: `tests/test_artificial_force.py`
- Delete: `tests/test_path_relax.py`
- Delete: `tests/test_scoring.py`
- Delete: `tests/test_config.py` (旧 schema 専用)
- Delete: `tests/test_config_afir.py`
- Delete: `tests/test_neb_refine_sn2.py` (`--neb-refine` フラグ廃止)
- Delete: `tests/test_cli_neb_refine_guard.py` (同上)
- Delete: `tests/test_wallclock_sn2.py` (AFIR ベースの wall-clock 想定)

- [ ] **Step 1: import チェック (削除前)**

```
python -m pytest tests/test_bond_changes.py tests/test_endpoint_relax.py tests/test_neb.py tests/test_placement.py tests/test_align.py tests/test_config_phase11.py -v
```
Expected: 上記新規 test がすべて green であること。

- [ ] **Step 2: ファイル削除**

```bash
git rm reactx/artificial_force.py reactx/path_relax.py reactx/scoring.py
git rm tests/test_afir_constraint.py tests/test_artificial_force.py tests/test_path_relax.py tests/test_scoring.py tests/test_config.py tests/test_config_afir.py tests/test_neb_refine_sn2.py tests/test_cli_neb_refine_guard.py tests/test_wallclock_sn2.py
```

- [ ] **Step 3: 残存参照チェック**

```
grep -rn "artificial_force\|path_relax\|AFIRConstraint\|build_afir_constraint\|scoring" reactx/ tests/ examples/ --include="*.py" --include="*.toml" 2>&1 | grep -v "test_scoring\|reactx/scoring\|test_path_relax\|test_artificial_force\|test_afir_constraint" || echo "no remaining refs"
```
Expected: `no remaining refs` (or only deleted-file references — should be empty if `git rm` succeeded)

- [ ] **Step 4: 高速 test スイートが通ることを確認** (slow / blender 除外)

```
pytest -m "not slow and not blender" -v
```
Expected: 残存 test がすべて green。test_re*.py / test_examples_menshutkin.py / test_diels_alder_*.py / test_cli.py / test_cli_unimolecular.py のうち AFIR を import するものはまだ残るが、これらは `@pytest.mark.slow` か import エラーで collection 段階で抜ける可能性がある → エラーが出たら **Task 10 で削除予定なので一旦 deselect**:

```
pytest -m "not slow and not blender" --ignore=tests/test_re1_sn2.py --ignore=tests/test_re1_proton_transfer.py --ignore=tests/test_re1_menshutkin.py --ignore=tests/test_re3_e2.py --ignore=tests/test_re3_sn1_dissoc.py --ignore=tests/test_re4_sn1_recomb.py --ignore=tests/test_diels_alder_simple.py --ignore=tests/test_diels_alder_endo.py --ignore=tests/test_examples_menshutkin.py --ignore=tests/test_cli.py --ignore=tests/test_cli_unimolecular.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git commit -m "chore: delete AFIR/scoring/path_relax modules and obsolete tests (Phase 11 step 8)"
```

---

### Task 9: `examples/*.rxn.toml` を Phase 11 schema に書き直し

8 反応すべての sidecar TOML を新 schema (description + optional sections) に置き換える。

**Files:**
- Modify: `examples/sn2.rxn.toml`
- Modify: `examples/proton_transfer.rxn.toml`
- Modify: `examples/menshutkin.rxn.toml`
- Modify: `examples/e2.rxn.toml`
- Modify: `examples/sn1_dissoc.rxn.toml`
- Modify: `examples/sn1_recomb.rxn.toml`
- Modify: `examples/diels_alder_simple.rxn.toml`
- Modify: `examples/diels_alder_endo.rxn.toml`

- [ ] **Step 1: `examples/sn2.rxn.toml` を置換**

```toml
description = "SN2 anion: CH3Cl + OH- -> CH3OH + Cl-"
```

- [ ] **Step 2: `examples/proton_transfer.rxn.toml` を置換**

```toml
description = "Proton transfer: HCl + NH3 -> Cl- + NH4+"
```

- [ ] **Step 3: `examples/menshutkin.rxn.toml` を置換**

```toml
description = "Menshutkin: NH3 + CH3Cl -> CH3NH3+ + Cl-"
```

- [ ] **Step 4: `examples/e2.rxn.toml` を置換**

```toml
description = "E2 elimination"
```

- [ ] **Step 5: `examples/sn1_dissoc.rxn.toml` を置換**

```toml
description = "SN1 step 1 dissociation: tBuBr -> tBu+ + Br-"

[placement]
initial_separation = 6.0
```

(SN1 dissoc では P 側 fragment 分離を 6 Å に伸ばす — Phase 9 の `r_broken_threshold=6.0` 相当の意図を残す)

- [ ] **Step 6: `examples/sn1_recomb.rxn.toml` を置換**

```toml
description = "SN1 step 2 recombination: tBu+ + Cl- -> tBuCl"
```

- [ ] **Step 7: `examples/diels_alder_simple.rxn.toml` を置換**

```toml
description = "Diels-Alder: butadiene + ethylene -> cyclohexene"

[placement]
orientation = "default"
```

- [ ] **Step 8: `examples/diels_alder_endo.rxn.toml` を置換**

```toml
description = "Diels-Alder endo: cyclopentadiene + maleic anhydride -> norbornene-2,3-dicarboxylic anhydride"

[placement]
orientation = "endo"
```

- [ ] **Step 9: 全 TOML が load_config を通ることを確認**

```
python -c "from reactx.config import load_config; from pathlib import Path; [print(p.stem, '->', load_config(p.with_suffix('')).description) for p in Path('examples').glob('*.rxn.toml')]"
```
Expected: 8 行の出力、エラーなし

- [ ] **Step 10: Commit**

```bash
git add examples/*.rxn.toml
git commit -m "refactor(examples): migrate 8 reactions to Phase 11 schema (Phase 11 step 9)"
```

---

### Task 10: 統合テスト書き直し (per-reaction + cli/conftest)

各反応 1 個ずつ、NEB trajectory ベースの assertion で書き直す。`@pytest.mark.slow` を維持し UMA を使う。

**Files:**
- Modify: `tests/conftest.py` (新 schema の `tmp_rxn_with_toml` fixture)
- Modify: `tests/test_cli.py`
- Modify: `tests/test_cli_unimolecular.py`
- Modify: `tests/test_re1_sn2.py`
- Modify: `tests/test_re1_proton_transfer.py`
- Modify: `tests/test_re1_menshutkin.py`
- Modify: `tests/test_re3_e2.py`
- Modify: `tests/test_re3_sn1_dissoc.py`
- Modify: `tests/test_re4_sn1_recomb.py`
- Modify: `tests/test_diels_alder_simple.py`
- Modify: `tests/test_diels_alder_endo.py`
- Modify: `tests/test_examples_menshutkin.py`

- [ ] **Step 1: `tests/conftest.py` を確認・必要なら更新**

```
cat tests/conftest.py
```

既存 `tmp_rxn_with_toml(name, toml_body)` fixture が `.rxn` + `.rxn.toml` を tmp_path に生成する形であれば変更不要。`toml_body` が Phase 11 schema なら新 schema が通る。

- [ ] **Step 2: `tests/test_re1_sn2.py` を Phase 11 trajectory assertion に書き換え**

`tests/test_re1_sn2.py` を以下で完全置換:

```python
"""SN2 integration test (Phase 11 Pure CI-NEB)."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from reactx.cli import main

_SN2_TOML = 'description = "SN2 anion test"\n'


@pytest.mark.slow
def test_sn2_neb_trajectory_and_walden(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn2", toml_body=_SN2_TOML)
    out = tmp_path / "sn2"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text())
    assert meta["neb"]["n_images"] == 11
    # endpoint relax must succeed
    assert meta["endpoint_relax_r"]["converged"] is True
    assert meta["endpoint_relax_p"]["converged"] is True
    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) == 11
    # Walden inversion: in the reactant endpoint, OH- approaches from the
    # backside of CH3Cl. Cl-C-O angle must exceed 150°.
    r_frame = frames[0]
    # atom map 1 -> C (idx 0), 2 -> Cl (idx 1), 3 -> O (idx 2) per the .rxn
    c, cl, o = r_frame.positions[0], r_frame.positions[1], r_frame.positions[2]
    v_ccl = cl - c
    v_co = o - c
    cos_theta = (v_ccl @ v_co) / (np.linalg.norm(v_ccl) * np.linalg.norm(v_co))
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))
    assert angle_deg >= 150.0, f"Walden angle {angle_deg:.1f}° < 150° threshold"
    # In the product endpoint, C-O is a real bond (< 1.7 Å) and C-Cl is dissociated (> 3.0 Å)
    p_frame = frames[-1]
    c_p, cl_p, o_p = p_frame.positions[0], p_frame.positions[1], p_frame.positions[2]
    assert float(np.linalg.norm(o_p - c_p)) < 1.7
    assert float(np.linalg.norm(cl_p - c_p)) > 3.0
```

- [ ] **Step 3: `tests/test_re1_proton_transfer.py` を書き換え**

```python
"""Proton transfer integration test (Phase 11)."""
import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from reactx.cli import main

_PT_TOML = 'description = "Proton transfer test"\n'


@pytest.mark.slow
def test_proton_transfer_neb(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("proton_transfer", toml_body=_PT_TOML)
    out = tmp_path / "pt"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text())
    assert meta["endpoint_relax_r"]["converged"] is True
    assert meta["endpoint_relax_p"]["converged"] is True
    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) == 11
```

- [ ] **Step 4: 残りの 6 統合テストも同パターンで書き換え**

`test_re1_menshutkin.py`, `test_re3_e2.py`, `test_re3_sn1_dissoc.py`, `test_re4_sn1_recomb.py`, `test_diels_alder_simple.py`, `test_diels_alder_endo.py`, `test_examples_menshutkin.py` を上記 PT パターンと同じ形に書き換える (description のみ変更、出力フレーム数と endpoint convergence のみ assert)。

例 — `tests/test_re3_e2.py`:

```python
"""E2 elimination integration test (Phase 11)."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main

_E2_TOML = 'description = "E2 elimination test"\n'


@pytest.mark.slow
def test_e2_neb(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("e2", toml_body=_E2_TOML)
    out = tmp_path / "e2"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text())
    assert meta["endpoint_relax_r"]["converged"] is True
    assert meta["endpoint_relax_p"]["converged"] is True
    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) == 11
```

例 — `tests/test_diels_alder_endo.py`:

```python
"""Diels-Alder endo integration test (Phase 11)."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main

_DA_ENDO_TOML = '''description = "DA endo test"

[placement]
orientation = "endo"
'''


@pytest.mark.slow
def test_diels_alder_endo_neb(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("diels_alder_endo", toml_body=_DA_ENDO_TOML)
    out = tmp_path / "da_endo"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text())
    assert meta["endpoint_relax_r"]["converged"] is True
    assert meta["endpoint_relax_p"]["converged"] is True
    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) == 11
```

(他 5 反応も同パターン。SN1 dissoc は `[placement] initial_separation = 6.0` を toml_body に含める。)

- [ ] **Step 5: `tests/test_cli.py` / `tests/test_cli_unimolecular.py` を新 meta.json 構造で更新**

`tests/test_cli.py` の中で `meta["trials"]` / `meta["selected_trial"]` / `meta["placement"]` を参照している箇所をすべて新 meta 構造 (`meta["neb"]` / `meta["endpoint_relax_r"]` 等) に置き換える。LJ backend で速く回せる test だけは残す。`tests/test_cli_unimolecular.py` は SN1 dissoc 系の sanity check として残す。

具体的には、`grep -n 'trials\|selected_trial\|placement_kind\|reached_product\|formed_latch_count' tests/test_cli.py tests/test_cli_unimolecular.py` で出る参照を以下置換:

- `meta["trials"][i]["reached_product"]` → 削除 (concept なし)
- `meta["selected_trial"]` → 削除
- `meta["placement"]` → 削除
- `meta["converged"]` → `meta["neb"]["converged"]`
- `meta["neb_refined"]` → 削除
- 残すべき assertion は `meta["neb"]["image_energies"]`, `meta["endpoint_relax_r"]["converged"]` など

- [ ] **Step 6: 高速 test スイートが通ることを確認**

```
pytest -m "not slow and not blender" -v
```
Expected: all green

- [ ] **Step 7: slow test 1 件 (SN2) を実行して動作確認**

```
pytest tests/test_re1_sn2.py -v -m slow
```
Expected: PASS (UMA model download 後 ~ 6-8 min)

- [ ] **Step 8: Commit**

```bash
git add tests/
git commit -m "test: rewrite integration tests for Pure CI-NEB pipeline (Phase 11 step 10)"
```

---

### Task 11: README を Phase 11 化

タイトル変更 ("Generic Pure CI-NEB Reaction Path Engine")、Phase 11 changes 節を冒頭に追加、Phase 8/9/10 changes 節を削除、TOML schema 表を新 schema に更新、`--neb-refine` 言及を削除、アーキテクチャ図を spec §3.1 ベースに刷新、wall-clock 表を後で再測する旨に書き換え。

**Files:**
- Modify: `README.md`

- [ ] **Step 1: README タイトル + 冒頭 4 行を置換**

`README.md` 1-5 行目:

```markdown
# reactx — Generic Pure CI-NEB Reaction Path Engine

2D 反応機構 (`.rxn`) と sidecar TOML config から、R/P 両構造を直接 3D embed → UMA で両端 endpoint を minimum まで緩和 → CI-NEB (IDPP interpolation + 2-phase climb) で経路を生成し、Blender で ball-and-stick アニメーションを生成するパイプライン。

目的は妥当なアニメーション (正確な TS エネルギーは目標としない)。対応反応:
```

- [ ] **Step 2: "Phase X changes" 節を削除し "Phase 11 changes" を追加**

`README.md` の `## Phase 9 changes` / `## Phase 10 changes` 節 (17-39 行目あたり) を以下で置換:

```markdown
## Phase 11 changes

Phase 10 までの **per-pair AFIR + sticky latch + 4π Fibonacci sampling + vdW blocking** を全廃し、`.rxn` の R/P 両構造から直接 endpoint を構築する **Pure CI-NEB** パイプラインに置換:

- AFIR インフラ (`reactx/artificial_force.py` / `reactx/path_relax.py` / `reactx/scoring.py`) を削除
- `BondChanges.from_reaction_diff(r_mol, p_mol)` で formed/broken を **atom-map 差分から自動推論** (TOML から `formed` / `broken` 撤廃)
- placement を `simple_placement` 1 関数に縮小 (Fibonacci / blocking / multi-anchor / endo-exo 展開を全廃)。bimolecular では反応点 anchor を `initial_separation=4.0 Å` で +z 方向に向き合わせる 1 通りのみ
- `reactx/endpoint_relax.py` 新設: UMA で R/P 両端を `fmax=0.01 eV/Å` まで minimum 緩和
- TOML schema を `[placement]` / `[endpoint_relax]` / `[neb]` の 3 section に刷新 (全 section / 全 key 省略可、`description` 1 行で動く)
- CLI から `--neb-refine` / `--neb-images` / `--seed` / `--traj-stride` / `--relax-fmax` を撤廃。NEB は常時実行
- `meta.json` 構造刷新 (trial 配列廃止、`endpoint_relax_r` / `endpoint_relax_p` / `neb` フラット構造)

詳細仕様: `docs/superpowers/specs/2026-05-15-phase-11-ci-neb-design.md`
```

- [ ] **Step 3: TOML 表 + コード例を新 schema に書き換え**

`Per-reaction .rxn.toml config` 節の SN2 例と表を以下で置換:

```markdown
## Per-reaction `.rxn.toml` config

各 `examples/<name>.rxn` には同階層に同名 stem の sidecar TOML (`<name>.rxn.toml`) を **必須で** 配置する。`description` のみ必須、その他は省略可。

最小例 (`examples/sn2.rxn.toml`):

```toml
description = "SN2 anion: CH3Cl + OH- -> CH3OH + Cl-"
```

全 section を明示する例:

```toml
description = "..."

[placement]
initial_separation = 4.0   # default 4.0 Å、bimolecular 反応点間距離
orientation        = "default"  # "default" | "endo" | "exo" (DA のみ参照)

[endpoint_relax]
fmax       = 0.01          # default 0.01 eV/Å
max_steps  = 500           # default 500
optimizer  = "FIRE"        # "FIRE" | "BFGS"

[neb]
n_images   = 11            # default 11
fmax       = 0.05          # default 0.05
max_steps  = 200           # default 200
k          = 1.0           # default 1.0
climb      = true          # default true
pad_frames = 0             # default 0
```

| `.rxn` | description | 特記事項 |
|---|---|---|
| sn2.rxn | SN2 anion (`O⁻ + CH₃Cl`) | all default |
| proton_transfer.rxn | Proton transfer (`HCl + NH₃`) | all default |
| menshutkin.rxn | Menshutkin (`NH₃ + CH₃Cl`) | all default |
| e2.rxn | E2 elimination | all default |
| sn1_dissoc.rxn | SN1 step 1 dissociation | `[placement] initial_separation = 6.0` (大きい解離距離) |
| sn1_recomb.rxn | SN1 step 2 recombination | all default |
| diels_alder_simple.rxn | DA: butadiene + ethylene | `[placement] orientation = "default"` |
| diels_alder_endo.rxn | DA endo: CP + MA | `[placement] orientation = "endo"` |
```

- [ ] **Step 4: アーキテクチャ図を Phase 11 ベースに更新**

`## アーキテクチャ` 節の ASCII 図と「主要ファイル」表を spec §3.1 / §3.3 を反映して置換:

````markdown
## アーキテクチャ

```
.rxn + .rxn.toml ─> parse_rxn + load_config ─> ReactionConfig + heavy_mapping
                                                    │
                                                    ▼
                              BondChanges.from_reaction_diff(r_mol, p_mol)
                                                    │
                       ┌────────────────────────────┴───────────────────────────┐
                       ▼                                                        ▼
        embed_fragments_to_positions(r_mol)                  embed_fragments_to_positions(p_mol)
                       │                                                        │
                       ▼                                                        ▼
       simple_placement(side="reactant")                  simple_placement(side="product",
        anchor = bond_changes.formed                       heavy_mapping=...)
                       │                                   anchor = bond_changes.broken
                       ▼                                                        │
        relax_endpoint(atoms_r, calc=UMA,                                       ▼
                       fmax=0.01, max_steps=500)        relax_endpoint(atoms_p, calc=UMA, ...)
                       │                                                        │
                       └───────────────────────────────┬────────────────────────┘
                                                       ▼
                                  align_product_to_reactant
                                  (heavy_mapping + h_groups)
                                                       │
                                                       ▼
                                  run_neb (IDPP + 2-phase CI-NEB, FIRE)
                                                       │
                                                       ▼
                                  trajectory.xyz → blender/render.py → .blend
```

主要ファイル:

- `reactx/bond_changes.py` — `BondChanges.from_reaction_diff` (R/P bond set 差分)
- `reactx/placement.py` — `simple_placement` (反応点 anchor を +z 方向に separation Å)
- `reactx/endpoint_relax.py` — `relax_endpoint` (FIRE/BFGS で minimum 緩和)
- `reactx/align.py` — `align_product_to_reactant` (heavy_mapping ベース、汎用)
- `reactx/neb.py` — `run_neb` (IDPP + warmup + climb)
- `reactx/config.py` — `[placement]` + `[endpoint_relax]` + `[neb]` schema
- `reactx/cli.py` — Pure CI-NEB パイプライン

詳細設計: `docs/superpowers/specs/2026-05-15-phase-11-ci-neb-design.md`
````

- [ ] **Step 5: "方針と限界" 節の AFIR 言及を削除**

`## 方針と限界` 節から AFIR / sticky latch / `--neb-refine` 関連の項目を削除し、Pure CI-NEB の限界 (IDPP failure cases / endpoint relax 未収束時の挙動 / DA の orientation 離散) を記述。

- [ ] **Step 6: 主要フラグ表を更新**

`## 使い方` 節の主要フラグリストを以下で置換:

```markdown
主要フラグ:

- `--backend {uma,lj}` (default: `uma`)
- `--model uma-m-1p1` (UMA model name)
- `--render` + `--blender-exe blender`
```

- [ ] **Step 7: "Wall-clock" 表を再測 placeholder に置換**

`## Wall-clock (実測)` 節の表を以下に置換:

```markdown
## Wall-clock (実測予定)

Phase 11 で AFIR の n_candidates=64 ループが消え、代わりに R/P endpoint relax 2 回 + NEB 1 回が走る構成に。8 反応すべて UMA-m-1p1 + RTX 5070 Ti で再測予定 (PR merge 前)。
```

- [ ] **Step 8: README 全体読み返し**

```
cat README.md | grep -n "AFIR\|sticky\|latch\|reached_product\|alpha_formed\|alpha_broken\|r_broken_threshold\|r_formed_threshold\|pre_relax_steps\|neb-refine\|formed_latch_count" || echo "no residue"
```
Expected: `no residue`

- [ ] **Step 9: Commit**

```bash
git add README.md
git commit -m "docs(readme): rewrite for Phase 11 Pure CI-NEB (Phase 11 step 11)"
```

---

### Task 12: 全体 smoke test + 最終確認

`pytest -m "not slow and not blender"` 全件、slow test 1-2 件、SN2 の Walden 角度回帰、wall-clock の手測。

- [ ] **Step 1: 高速 test 全件**

```
pytest -m "not slow and not blender" -v
```
Expected: all green

- [ ] **Step 2: SN2 slow test**

```
pytest tests/test_re1_sn2.py -v -m slow
```
Expected: PASS。Walden 角度 ≥ 150° の assertion が通ること。

- [ ] **Step 3: 他 7 反応の slow test (時間がかかる、~1 hour 全部で)**

```
pytest tests/test_re1_proton_transfer.py tests/test_re1_menshutkin.py tests/test_re3_e2.py tests/test_re3_sn1_dissoc.py tests/test_re4_sn1_recomb.py tests/test_diels_alder_simple.py tests/test_diels_alder_endo.py -v -m slow
```
Expected: all PASS。endpoint_relax_r / endpoint_relax_p が両方 converged=True。

- [ ] **Step 4: 1 反応で実出力を Blender まで確認 (任意)**

```
reactx run examples/sn2.rxn -o out/sn2_phase11/ --backend uma --render
```
Expected: `out/sn2_phase11/scene.blend` が生成され、Walden inversion アニメーションが視認できる。

- [ ] **Step 5: 残存 dead reference 全件チェック**

```
grep -rn "AFIRConstraint\|build_afir_constraint\|relax_with_restraints\|reached_product\|sticky" reactx/ tests/ blender/ examples/ docs/ --include="*.py" --include="*.toml" --include="*.md" || echo "no residue"
```
Expected: docs/ 内の **過去 spec / 過去 plan** にのみ言及が残る (削除しない)。`reactx/` / `tests/` / `blender/` / `examples/` には残らない。

- [ ] **Step 6: 最終 commit (なくてもよい、cleanup commit)**

もし上記 step で何か微修正が必要だった場合のみ:
```bash
git add -A
git commit -m "chore: post-Phase 11 cleanup"
```

---

## 完了条件

- [ ] 全 12 タスク green
- [ ] `pytest -m "not slow and not blender"` PASS
- [ ] `pytest -m slow` 全 8 反応 PASS (要 UMA + GPU)
- [ ] `reactx/artificial_force.py`, `reactx/path_relax.py`, `reactx/scoring.py` がリポジトリから消えている
- [ ] `examples/*.rxn.toml` 8 件すべてが新 schema (description + optional sections)
- [ ] `README.md` に Phase 11 changes / 新 schema / 新アーキテクチャ図が記載
- [ ] `meta.json` の構造が spec §8.5 と一致
- [ ] PR の base branch = `develop` (`[[feedback_pr_base_develop]]` 準拠)

## リスク再掲

- endpoint relax が `fmax=0.01` で 500 step 以内に収束しない反応が出る可能性 (E2 / DA endo 等の歪んだ初期配置)。Task 12 step 3 で実測し、未収束反応は個別 TOML で `endpoint_relax.fmax` を緩める / `max_steps` を伸ばす / `optimizer="BFGS"` に切替で対応。
- DA simple の `orientation="default"` で endo/exo が一意に決まらない場合 → 実測して問題が出れば `examples/diels_alder_simple.rxn.toml` で明示する。
- SN2 で Walden 角度が ≥ 150° に届かない場合 → endpoint_relax の `fmax` を更に絞る (例: 0.001) か、`initial_separation` を伸ばす。
