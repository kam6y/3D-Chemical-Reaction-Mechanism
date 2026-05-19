# Explicit Endpoint Inputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace runtime Fibonacci fragment placement with explicit reactant/product XYZ endpoint inputs for CI-NEB.

**Architecture:** `.rxn` remains the source for atom mapping and connectivity, while `.rxn.toml` requires `reactant_structure` and `product_structure` paths. A new endpoint loader reads and validates the XYZ endpoints, CLI relaxes those endpoints directly, and placement-only code is deleted.

**Tech Stack:** Python 3.11, RDKit, ASE, pytest, TOML via `tomllib`, existing `reactx` CLI and test fixtures.

---

## File Structure

- Modify `reactx/config.py`: replace Phase 11 placement schema with explicit endpoint path schema.
- Create `reactx/endpoints.py`: load endpoint XYZ files, validate atom count and element order, annotate charges/spin, return provenance metadata.
- Modify `reactx/cli.py`: remove runtime embedding/placement/candidate selection, load explicit endpoints, write endpoint provenance and bond-change metadata.
- Delete `reactx/placement.py`: remove the placement engine from the codebase.
- Delete `tests/test_placement.py`: placement behavior is no longer part of the product.
- Delete `tests/test_embed3d.py` and `reactx/embed3d.py` if `rg "embed_fragments_to_positions|reactx.embed3d"` shows no non-test caller after CLI migration.
- Modify `tests/test_config_phase11.py`: update schema expectations.
- Create `tests/test_endpoints.py`: endpoint loader unit tests.
- Modify `tests/test_cli.py`, `tests/test_cli_unimolecular.py`, slow example tests: explicit endpoint TOML and metadata assertions.
- Modify `tests/conftest.py`: copy example endpoint files into temporary test directories.
- Modify `tests/test_align.py`: stop importing `reactx.placement`; build simple `ase.Atoms` directly.
- Modify `examples/*.rxn.toml`: add explicit endpoint paths and remove `[placement]`.
- Create `examples/*.reactant.xyz` and `examples/*.product.xyz`: endpoint files for every existing example reaction.
- Modify `README.md`: document explicit endpoint input pipeline.

---

## Task 1: Config Schema

**Files:**
- Modify: `reactx/config.py`
- Modify: `tests/test_config_phase11.py`

- [ ] **Step 1: Rewrite config tests for required endpoint paths**

Replace `tests/test_config_phase11.py` with:

```python
"""Tests for explicit endpoint config schema."""
from pathlib import Path

import pytest

from reactx.config import ConfigError, EndpointRelaxSection, NEBSection, load_config


def _write_rxn_and_toml(tmp_path: Path, body: str) -> Path:
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n\n  RDKit\n\n  0  0\n", encoding="utf-8")
    (tmp_path / "x.rxn.toml").write_text(body, encoding="utf-8")
    return rxn


def test_minimal_config_requires_endpoint_paths(tmp_path):
    rxn = _write_rxn_and_toml(
        tmp_path,
        """description = "minimal"
reactant_structure = "x.reactant.xyz"
product_structure = "x.product.xyz"
""",
    )
    cfg = load_config(rxn)
    assert cfg.description == "minimal"
    assert cfg.reactant_structure == "x.reactant.xyz"
    assert cfg.product_structure == "x.product.xyz"
    assert cfg.endpoint_relax == EndpointRelaxSection()
    assert cfg.neb == NEBSection()


def test_default_values(tmp_path):
    rxn = _write_rxn_and_toml(
        tmp_path,
        """description = "x"
reactant_structure = "r.xyz"
product_structure = "p.xyz"
""",
    )
    cfg = load_config(rxn)
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
    body = """description = "all"
reactant_structure = "r.xyz"
product_structure = "p.xyz"

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
"""
    rxn = _write_rxn_and_toml(tmp_path, body)
    cfg = load_config(rxn)
    assert cfg.reactant_structure == "r.xyz"
    assert cfg.product_structure == "p.xyz"
    assert cfg.endpoint_relax.fmax == 0.005
    assert cfg.endpoint_relax.max_steps == 800
    assert cfg.endpoint_relax.optimizer == "BFGS"
    assert cfg.neb.n_images == 13
    assert cfg.neb.fmax == 0.03
    assert cfg.neb.max_steps == 250
    assert cfg.neb.k == 0.5
    assert cfg.neb.climb is False
    assert cfg.neb.pad_frames == 2


@pytest.mark.parametrize(
    "body,match",
    [
        ('reactant_structure = "r.xyz"\nproduct_structure = "p.xyz"\n', "description"),
        ('description = "x"\nproduct_structure = "p.xyz"\n', "reactant_structure"),
        ('description = "x"\nreactant_structure = "r.xyz"\n', "product_structure"),
        ('description = "x"\nreactant_structure = ""\nproduct_structure = "p.xyz"\n', "reactant_structure"),
        ('description = "x"\nreactant_structure = "r.xyz"\nproduct_structure = ""\n', "product_structure"),
    ],
)
def test_required_values_rejected(tmp_path, body, match):
    rxn = _write_rxn_and_toml(tmp_path, body)
    with pytest.raises(ConfigError, match=match):
        load_config(rxn)


@pytest.mark.parametrize(
    "obsolete",
    [
        "[placement]\norientation = \"endo\"\n",
        "[placement]\nn_candidates = 32\n",
        "[afir]\nalpha_formed = 1.0\n",
        "[scoring]\nr_broken_threshold = 3.0\n",
        "[sampling]\nn_candidates = 32\n",
        "formed = [[1, 2]]\n",
        "broken = [[1, 2]]\n",
        "[restraints]\nk_form = 1.0\n",
        "[prescreen]\nn_angles = 8\n",
    ],
)
def test_obsolete_keys_rejected(tmp_path, obsolete):
    body = """description = "x"
reactant_structure = "r.xyz"
product_structure = "p.xyz"
""" + obsolete
    rxn = _write_rxn_and_toml(tmp_path, body)
    with pytest.raises(ConfigError, match="explicit endpoint"):
        load_config(rxn)


@pytest.mark.parametrize(
    "bad_val,scope,key",
    [
        (0.0, "endpoint_relax", "fmax"),
        (-0.1, "endpoint_relax", "fmax"),
        (0, "endpoint_relax", "max_steps"),
        ("Newton", "endpoint_relax", "optimizer"),
        (2, "neb", "n_images"),
        (0.0, "neb", "fmax"),
        (0, "neb", "max_steps"),
        (0.0, "neb", "k"),
        (-1, "neb", "pad_frames"),
    ],
)
def test_invalid_values_rejected(tmp_path, bad_val, scope, key):
    body = f"""description = "x"
reactant_structure = "r.xyz"
product_structure = "p.xyz"
[{scope}]
{key} = {bad_val!r}
"""
    rxn = _write_rxn_and_toml(tmp_path, body)
    with pytest.raises(ConfigError):
        load_config(rxn)


def test_missing_toml_raises_file_not_found(tmp_path):
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_config(rxn)


def test_unknown_top_level_key_rejected(tmp_path):
    body = """description = "x"
reactant_structure = "r.xyz"
product_structure = "p.xyz"
weird_key = 42
"""
    rxn = _write_rxn_and_toml(tmp_path, body)
    with pytest.raises(ConfigError, match="unknown"):
        load_config(rxn)
```

- [ ] **Step 2: Run config tests and verify failure**

Run: `pytest tests/test_config_phase11.py -v`

Expected: FAIL because `PlacementSection` still exists, `ReactionConfig` has no endpoint path fields, and `[placement]` is still accepted.

- [ ] **Step 3: Implement explicit endpoint schema**

Update `reactx/config.py` to this shape:

```python
"""Per-reaction sidecar TOML config for explicit endpoint CI-NEB."""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    """Raised when `<rxn_path>.toml` violates the reactx config schema."""


@dataclass(frozen=True)
class EndpointRelaxSection:
    fmax: float = 0.01
    max_steps: int = 500
    optimizer: str = "FIRE"


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
    reactant_structure: str
    product_structure: str
    endpoint_relax: EndpointRelaxSection
    neb: NEBSection


_TOP_LEVEL_KEYS = {
    "description",
    "reactant_structure",
    "product_structure",
    "endpoint_relax",
    "neb",
}
_TOP_LEVEL_REQUIRED = {"description", "reactant_structure", "product_structure"}
_OBSOLETE_TOP_LEVEL = {
    "afir",
    "broken",
    "formed",
    "k_broken",
    "k_form",
    "placement",
    "prescreen",
    "r_broken",
    "r_form",
    "restraints",
    "sampling",
    "scoring",
}
_ENDPOINT_KEYS = {"fmax", "max_steps", "optimizer"}
_NEB_KEYS = {"n_images", "fmax", "max_steps", "k", "climb", "pad_frames"}
_VALID_OPTIMIZERS = {"FIRE", "BFGS"}


def sidecar_path(rxn_path: Path) -> Path:
    """`examples/sn2.rxn` -> `examples/sn2.rxn.toml`."""
    return rxn_path.parent / (rxn_path.name + ".toml")


def load_config(rxn_path: Path) -> ReactionConfig:
    """Load `<rxn_path>.toml` and return a validated ReactionConfig."""
    toml_path = sidecar_path(rxn_path)
    if not toml_path.is_file():
        raise FileNotFoundError(
            f"sidecar TOML not found: expected {toml_path} alongside {rxn_path}"
        )
    raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    return _validate(raw, source=str(toml_path))


def _validate(raw: dict, *, source: str) -> ReactionConfig:
    for key in _OBSOLETE_TOP_LEVEL:
        if key in raw:
            raise ConfigError(
                f"{source}: '{key}' is removed. "
                "Use explicit endpoint schema with reactant_structure and product_structure."
            )

    _check_keys(raw, _TOP_LEVEL_KEYS, _TOP_LEVEL_REQUIRED, scope="<top>", source=source)

    description = raw["description"]
    if not isinstance(description, str) or not description.strip():
        raise ConfigError(f"{source}: 'description' must be a non-empty string")

    reactant_structure = _path_string(
        raw["reactant_structure"], key="reactant_structure", source=source
    )
    product_structure = _path_string(
        raw["product_structure"], key="product_structure", source=source
    )

    endpoint_relax = _build_endpoint_relax(
        raw.get("endpoint_relax", {}),
        source=source,
    )
    neb = _build_neb(raw.get("neb", {}), source=source)

    return ReactionConfig(
        description=description,
        reactant_structure=reactant_structure,
        product_structure=product_structure,
        endpoint_relax=endpoint_relax,
        neb=neb,
    )


def _path_string(value, *, key: str, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{source}: '{key}' must be a non-empty string path")
    return value


def _build_endpoint_relax(raw: dict, *, source: str) -> EndpointRelaxSection:
    _check_keys(raw, _ENDPOINT_KEYS, set(), scope="[endpoint_relax]", source=source)
    fmax = _positive_float(
        raw.get("fmax", 0.01),
        key="[endpoint_relax].fmax",
        source=source,
    )
    max_steps = _positive_int(
        raw.get("max_steps", 500),
        key="[endpoint_relax].max_steps",
        source=source,
    )
    optimizer = raw.get("optimizer", "FIRE")
    if optimizer not in _VALID_OPTIMIZERS:
        raise ConfigError(
            f"{source}: '[endpoint_relax].optimizer' must be one of "
            f"{sorted(_VALID_OPTIMIZERS)}, got {optimizer!r}"
        )
    return EndpointRelaxSection(
        fmax=fmax,
        max_steps=max_steps,
        optimizer=optimizer,
    )


def _build_neb(raw: dict, *, source: str) -> NEBSection:
    _check_keys(raw, _NEB_KEYS, set(), scope="[neb]", source=source)
    n_images = _positive_int(raw.get("n_images", 11), key="[neb].n_images", source=source)
    if n_images < 3:
        raise ConfigError(f"{source}: '[neb].n_images' must be >= 3")
    fmax = _positive_float(raw.get("fmax", 0.05), key="[neb].fmax", source=source)
    max_steps = _positive_int(
        raw.get("max_steps", 200),
        key="[neb].max_steps",
        source=source,
    )
    k = _positive_float(raw.get("k", 1.0), key="[neb].k", source=source)
    climb = raw.get("climb", True)
    if not isinstance(climb, bool):
        raise ConfigError(f"{source}: '[neb].climb' must be boolean")
    pad_frames = _non_negative_int(
        raw.get("pad_frames", 0),
        key="[neb].pad_frames",
        source=source,
    )
    return NEBSection(
        n_images=n_images,
        fmax=fmax,
        max_steps=max_steps,
        k=k,
        climb=climb,
        pad_frames=pad_frames,
    )


def _check_keys(
    raw: dict,
    allowed: set[str],
    required: set[str],
    *,
    scope: str,
    source: str,
) -> None:
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: {scope} must be a table")
    missing = required - raw.keys()
    if missing:
        raise ConfigError(f"{source}: {scope} missing required keys: {sorted(missing)}")
    extra = set(raw.keys()) - allowed
    if extra:
        raise ConfigError(f"{source}: {scope} contains unknown keys: {sorted(extra)}")


def _positive_float(value, *, key: str, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{source}: '{key}' must be a positive number")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ConfigError(f"{source}: '{key}' must be a finite positive number")
    return float(value)


def _positive_int(value, *, key: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{source}: '{key}' must be a positive integer")
    return int(value)


def _non_negative_int(value, *, key: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{source}: '{key}' must be a non-negative integer")
    return int(value)
```

- [ ] **Step 4: Run config tests and commit**

Run: `pytest tests/test_config_phase11.py -v`

Expected: PASS.

Commit:

```bash
git add reactx/config.py tests/test_config_phase11.py
git commit -m "feat: require explicit endpoint config"
```

---

## Task 2: Endpoint Loader

**Files:**
- Create: `reactx/endpoints.py`
- Create: `tests/test_endpoints.py`

- [ ] **Step 1: Add endpoint loader tests**

Create `tests/test_endpoints.py`:

```python
"""Tests for explicit endpoint XYZ loading."""
from pathlib import Path

import pytest
from rdkit import Chem

from reactx.config import ReactionConfig, EndpointRelaxSection, NEBSection
from reactx.endpoints import EndpointError, load_endpoint_pair


def _cfg() -> ReactionConfig:
    return ReactionConfig(
        description="x",
        reactant_structure="r.xyz",
        product_structure="p.xyz",
        endpoint_relax=EndpointRelaxSection(),
        neb=NEBSection(),
    )


def _write_xyz(path: Path, symbols: list[str]) -> None:
    lines = [str(len(symbols)), path.stem]
    for i, sym in enumerate(symbols):
        lines.append(f"{sym} {float(i):.6f} 0.000000 0.000000")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mol_h(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(mol)
    return Chem.AddHs(mol)


def test_load_endpoint_pair_resolves_relative_paths_and_sets_metadata(tmp_path):
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n", encoding="utf-8")
    reactant_mol_h = _mol_h("[NH4+]")
    product_mol_h = _mol_h("[NH4+]")
    symbols = [a.GetSymbol() for a in reactant_mol_h.GetAtoms()]
    _write_xyz(tmp_path / "r.xyz", symbols)
    _write_xyz(tmp_path / "p.xyz", symbols)

    atoms_r, atoms_p, meta = load_endpoint_pair(
        _cfg(),
        rxn_path=rxn,
        reactant_mol_h=reactant_mol_h,
        product_mol_h=product_mol_h,
    )

    assert atoms_r.get_chemical_symbols() == symbols
    assert atoms_p.get_chemical_symbols() == symbols
    assert list(atoms_r.get_initial_charges()) == [
        float(a.GetFormalCharge()) for a in reactant_mol_h.GetAtoms()
    ]
    assert atoms_r.info["charge"] == 1
    assert atoms_r.info["spin"] == 1
    assert meta == {
        "mode": "explicit_xyz",
        "reactant_structure": str((tmp_path / "r.xyz").resolve()),
        "product_structure": str((tmp_path / "p.xyz").resolve()),
    }


def test_load_endpoint_pair_rejects_missing_file(tmp_path):
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n", encoding="utf-8")
    mol_h = _mol_h("C")
    _write_xyz(tmp_path / "r.xyz", [a.GetSymbol() for a in mol_h.GetAtoms()])
    with pytest.raises(EndpointError, match="product_structure"):
        load_endpoint_pair(
            _cfg(),
            rxn_path=rxn,
            reactant_mol_h=mol_h,
            product_mol_h=mol_h,
        )


def test_load_endpoint_pair_rejects_atom_count_mismatch(tmp_path):
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n", encoding="utf-8")
    mol_h = _mol_h("C")
    _write_xyz(tmp_path / "r.xyz", ["C"])
    _write_xyz(tmp_path / "p.xyz", [a.GetSymbol() for a in mol_h.GetAtoms()])
    with pytest.raises(EndpointError, match="reactant_structure.*atom count"):
        load_endpoint_pair(
            _cfg(),
            rxn_path=rxn,
            reactant_mol_h=mol_h,
            product_mol_h=mol_h,
        )


def test_load_endpoint_pair_rejects_element_order_mismatch(tmp_path):
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n", encoding="utf-8")
    mol_h = _mol_h("CO")
    symbols = [a.GetSymbol() for a in mol_h.GetAtoms()]
    wrong = symbols.copy()
    wrong[0], wrong[1] = wrong[1], wrong[0]
    _write_xyz(tmp_path / "r.xyz", wrong)
    _write_xyz(tmp_path / "p.xyz", symbols)
    with pytest.raises(EndpointError, match="reactant_structure.*element order"):
        load_endpoint_pair(
            _cfg(),
            rxn_path=rxn,
            reactant_mol_h=mol_h,
            product_mol_h=mol_h,
        )


def test_load_endpoint_pair_rejects_multiple_frames(tmp_path):
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n", encoding="utf-8")
    mol_h = _mol_h("C")
    symbols = [a.GetSymbol() for a in mol_h.GetAtoms()]
    _write_xyz(tmp_path / "p.xyz", symbols)
    one = "\n".join([str(len(symbols)), "frame"] + [
        f"{sym} {float(i):.6f} 0.000000 0.000000"
        for i, sym in enumerate(symbols)
    ]) + "\n"
    (tmp_path / "r.xyz").write_text(one + one, encoding="utf-8")
    with pytest.raises(EndpointError, match="multiple frames"):
        load_endpoint_pair(
            _cfg(),
            rxn_path=rxn,
            reactant_mol_h=mol_h,
            product_mol_h=mol_h,
        )
```

- [ ] **Step 2: Run endpoint tests and verify failure**

Run: `pytest tests/test_endpoints.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'reactx.endpoints'`.

- [ ] **Step 3: Implement endpoint loader**

Create `reactx/endpoints.py`:

```python
"""Explicit reactant/product endpoint structure loading."""
from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read
from rdkit import Chem

from reactx.config import ReactionConfig, sidecar_path


class EndpointError(ValueError):
    """Raised when explicit endpoint structure files are invalid."""


def load_endpoint_pair(
    cfg: ReactionConfig,
    *,
    rxn_path: Path,
    reactant_mol_h: Chem.Mol,
    product_mol_h: Chem.Mol,
) -> tuple[Atoms, Atoms, dict]:
    """Load, validate, and annotate explicit endpoint structures."""
    toml_dir = sidecar_path(rxn_path).parent
    r_path = _resolve_endpoint_path(cfg.reactant_structure, base_dir=toml_dir)
    p_path = _resolve_endpoint_path(cfg.product_structure, base_dir=toml_dir)
    atoms_r = _load_one_endpoint(
        r_path,
        side="reactant_structure",
        expected_mol_h=reactant_mol_h,
    )
    atoms_p = _load_one_endpoint(
        p_path,
        side="product_structure",
        expected_mol_h=product_mol_h,
    )
    return atoms_r, atoms_p, {
        "mode": "explicit_xyz",
        "reactant_structure": str(r_path.resolve()),
        "product_structure": str(p_path.resolve()),
    }


def _resolve_endpoint_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _load_one_endpoint(
    path: Path,
    *,
    side: str,
    expected_mol_h: Chem.Mol,
) -> Atoms:
    if not path.is_file():
        raise EndpointError(f"{side}: endpoint file not found: {path}")
    try:
        frames = read(str(path), index=":")
    except Exception as exc:  # noqa: BLE001
        raise EndpointError(
            f"{side}: failed to read endpoint file {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) != 1:
        raise EndpointError(
            f"{side}: expected exactly one endpoint frame, got multiple frames ({len(frames)})"
        )
    atoms = frames[0].copy()
    _validate_symbols(atoms, expected_mol_h=expected_mol_h, side=side, path=path)
    _annotate_charges_and_spin(atoms, expected_mol_h)
    return atoms


def _validate_symbols(
    atoms: Atoms,
    *,
    expected_mol_h: Chem.Mol,
    side: str,
    path: Path,
) -> None:
    expected = [atom.GetSymbol() for atom in expected_mol_h.GetAtoms()]
    actual = atoms.get_chemical_symbols()
    if len(actual) != len(expected):
        raise EndpointError(
            f"{side}: atom count mismatch in {path}: expected {len(expected)}, got {len(actual)}"
        )
    if actual != expected:
        raise EndpointError(
            f"{side}: element order mismatch in {path}: expected {expected}, got {actual}"
        )


def _annotate_charges_and_spin(atoms: Atoms, mol_h: Chem.Mol) -> None:
    charges = [atom.GetFormalCharge() for atom in mol_h.GetAtoms()]
    atoms.set_initial_charges(charges)
    atoms.info["charge"] = int(sum(charges))
    atoms.info["spin"] = 1
```

- [ ] **Step 4: Run endpoint tests and commit**

Run: `pytest tests/test_endpoints.py -v`

Expected: PASS.

Commit:

```bash
git add reactx/endpoints.py tests/test_endpoints.py
git commit -m "feat: load explicit endpoint structures"
```

---

## Task 3: Generate And Migrate Example Endpoints

**Files:**
- Create: `examples/*.reactant.xyz`
- Create: `examples/*.product.xyz`
- Modify: `examples/*.rxn.toml`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add a temporary endpoint generation script**

Create `scripts/generate_explicit_endpoints.py`:

```python
"""Generate explicit endpoint XYZ files from the current Phase 11 placement path.

This is a one-time migration script. Delete it after endpoint files are written.
"""
from __future__ import annotations

from pathlib import Path

from ase.io import write
from rdkit import Chem

from reactx.bond_changes import BondChanges
from reactx.config import load_config
from reactx.embed3d import embed_fragments_to_positions
from reactx.placement import build_atoms_from_positions, valid_placements
from reactx.rxn_parser import parse_rxn

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
STEMS = [
    "sn2",
    "proton_transfer",
    "menshutkin",
    "e2",
    "sn1_dissoc",
    "sn1_recomb",
    "diels_alder_simple",
    "diels_alder_endo",
]


def main() -> int:
    for stem in STEMS:
        rxn_path = EXAMPLES / f"{stem}.rxn"
        r_mol, p_mol, heavy_mapping = parse_rxn(rxn_path)
        cfg = load_config(rxn_path)
        orientation = getattr(getattr(cfg, "placement", None), "orientation", "default")
        bond_changes = BondChanges.from_reaction_diff(r_mol, p_mol)

        r_h, _, pos_r = embed_fragments_to_positions(r_mol, seed=0)
        p_h, _, pos_p = embed_fragments_to_positions(p_mol, seed=0)
        atoms_r = build_atoms_from_positions(
            r_h,
            _select_positions(
                r_h,
                pos_r,
                bond_changes,
                seed=0,
                orientation=orientation,
                prefer_walden=(stem == "sn2"),
            ),
        )
        atoms_p = build_atoms_from_positions(
            p_h,
            _select_positions(
                p_h,
                pos_p,
                _product_side_changes(bond_changes, heavy_mapping),
                seed=2,
                orientation=orientation,
                prefer_walden=False,
            ),
        )
        write(EXAMPLES / f"{stem}.reactant.xyz", atoms_r, format="xyz")
        write(EXAMPLES / f"{stem}.product.xyz", atoms_p, format="xyz")
        print(f"wrote {stem}.reactant.xyz and {stem}.product.xyz")
    return 0


def _product_side_changes(
    bond_changes: BondChanges,
    heavy_mapping: dict[int, int],
) -> BondChanges:
    return BondChanges(
        formed=tuple((heavy_mapping[a], heavy_mapping[b]) for a, b in bond_changes.broken),
        broken=(),
    )


def _select_positions(
    mol_h,
    positions,
    bond_changes: BondChanges,
    *,
    seed: int,
    orientation: str,
    prefer_walden: bool,
):
    placement = valid_placements(
        mol_h,
        Chem.GetMolFrags(mol_h),
        positions,
        bond_changes,
        n_candidates=64,
        seed=seed,
    )
    trials = [
        trial
        for trial in placement.trials
        if orientation not in ("endo", "exo")
        or trial.orientation in ("single", "achiral", orientation)
    ]
    if not trials:
        trials = list(placement.trials)
    if prefer_walden and len(bond_changes.formed) == 1 and len(bond_changes.broken) == 1:
        formed = bond_changes.formed[0]
        broken = bond_changes.broken[0]
        shared = set(formed) & set(broken)
        if len(shared) == 1:
            center = next(iter(shared))
            incoming = formed[0] if formed[1] == center else formed[1]
            leaving = broken[0] if broken[1] == center else broken[1]
            return max(
                trials,
                key=lambda trial: _angle_degrees(
                    trial.positions[leaving] - trial.positions[center],
                    trial.positions[incoming] - trial.positions[center],
                ),
            ).positions
    return min(trials, key=lambda trial: float(trial.d_min)).positions


def _angle_degrees(a, b) -> float:
    import math
    import numpy as np

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    cos_theta = float(np.clip((a @ b) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the migration script**

Run: `python scripts/generate_explicit_endpoints.py`

Expected output includes all eight stems:

```text
wrote sn2.reactant.xyz and sn2.product.xyz
wrote proton_transfer.reactant.xyz and proton_transfer.product.xyz
wrote menshutkin.reactant.xyz and menshutkin.product.xyz
wrote e2.reactant.xyz and e2.product.xyz
wrote sn1_dissoc.reactant.xyz and sn1_dissoc.product.xyz
wrote sn1_recomb.reactant.xyz and sn1_recomb.product.xyz
wrote diels_alder_simple.reactant.xyz and diels_alder_simple.product.xyz
wrote diels_alder_endo.reactant.xyz and diels_alder_endo.product.xyz
```

- [ ] **Step 3: Validate generated endpoint atom order**

Run: `pytest tests/test_endpoints.py -v`

Expected: PASS.

Then run this sanity command:

```powershell
python -c "from pathlib import Path; from rdkit import Chem; from reactx.rxn_parser import parse_rxn; from ase.io import read; stems=['sn2','proton_transfer','menshutkin','e2','sn1_dissoc','sn1_recomb','diels_alder_simple','diels_alder_endo']; root=Path('examples'); [print(stem, len(read(str(root/f'{stem}.reactant.xyz'))), len(Chem.AddHs(parse_rxn(root/f'{stem}.rxn')[0]).GetAtoms()), len(read(str(root/f'{stem}.product.xyz'))), len(Chem.AddHs(parse_rxn(root/f'{stem}.rxn')[1]).GetAtoms())) for stem in stems]"
```

Expected: each line shows matching reactant and product atom counts, for example `sn2 6 6 6 6`.

- [ ] **Step 4: Rewrite example TOML files**

For each `examples/<stem>.rxn.toml`, replace the body with:

```toml
description = "<keep the existing description>"
reactant_structure = "<stem>.reactant.xyz"
product_structure = "<stem>.product.xyz"
```

For `examples/diels_alder_endo.rxn.toml`, remove `[placement]`; the endo stereochemistry now lives in `diels_alder_endo.reactant.xyz` and `diels_alder_endo.product.xyz`.

- [ ] **Step 5: Update temporary RXN fixture to copy endpoint files**

In `tests/conftest.py`, replace the body of `_make()` inside `tmp_rxn_with_toml` with:

```python
    def _make(stem: str, *, toml_body: str) -> Path:
        src_rxn = examples / f"{stem}.rxn"
        dst_rxn = tmp_path / f"{stem}.rxn"
        dst_rxn.write_bytes(src_rxn.read_bytes())
        for suffix in ("reactant.xyz", "product.xyz"):
            src = examples / f"{stem}.{suffix}"
            if src.exists():
                (tmp_path / f"{stem}.{suffix}").write_bytes(src.read_bytes())
        (tmp_path / f"{stem}.rxn.toml").write_text(toml_body, encoding="utf-8")
        return dst_rxn
```

- [ ] **Step 6: Delete the temporary migration script and commit**

Delete `scripts/generate_explicit_endpoints.py`.

Run: `pytest tests/test_config_phase11.py tests/test_endpoints.py -v`

Expected: PASS.

Commit:

```bash
git add examples tests/conftest.py
git add -u scripts/generate_explicit_endpoints.py
git commit -m "test: add explicit endpoint example structures"
```

---

## Task 4: CLI Explicit Endpoint Pipeline

**Files:**
- Modify: `reactx/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_cli_unimolecular.py`
- Modify: `tests/test_re1_sn2.py`
- Modify: `tests/test_diels_alder_simple.py`
- Modify: `tests/test_diels_alder_endo.py`

- [ ] **Step 1: Update fast CLI TOML fixtures**

In `tests/test_cli.py`, replace `_FAST_TOML` with:

```python
_FAST_TOML = """\
description = "sn2 quick"
reactant_structure = "sn2.reactant.xyz"
product_structure = "sn2.product.xyz"

[endpoint_relax]
fmax = 100.0
max_steps = 1

[neb]
n_images = 3
fmax = 100.0
max_steps = 1
"""
```

In `tests/test_cli_unimolecular.py`, replace `_SN1_DISSOC_FAST` with:

```python
_SN1_DISSOC_FAST = """\
description = "sn1_dissoc fast"
reactant_structure = "sn1_dissoc.reactant.xyz"
product_structure = "sn1_dissoc.product.xyz"

[endpoint_relax]
fmax = 100.0
max_steps = 1

[neb]
n_images = 3
fmax = 100.0
max_steps = 1
"""
```

- [ ] **Step 2: Add CLI metadata and no-placement assertions**

In `tests/test_cli.py`, add `import sys` near the imports.

Add this test:

```python
def test_cli_explicit_endpoints_do_not_import_placement(
    tmp_path: Path,
    tmp_rxn_with_toml,
):
    sys.modules.pop("reactx.placement", None)
    rxn = tmp_rxn_with_toml("sn2", toml_body=_FAST_TOML)
    out = tmp_path / "out"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "lj"])
    assert rc == 0
    assert "reactx.placement" not in sys.modules
```

In `test_cli_lj_writes_phase11_outputs`, after loading `meta`, add:

```python
    assert meta["endpoint_source"]["mode"] == "explicit_xyz"
    assert meta["endpoint_source"]["reactant_structure"].endswith("sn2.reactant.xyz")
    assert meta["endpoint_source"]["product_structure"].endswith("sn2.product.xyz")
    assert meta["bond_changes"]["formed"]
    assert meta["bond_changes"]["broken"]
```

In `tests/test_cli_unimolecular.py`, after `assert "placement" not in meta`, add:

```python
    assert meta["endpoint_source"]["mode"] == "explicit_xyz"
```

- [ ] **Step 3: Run CLI tests and verify failure**

Run: `pytest tests/test_cli.py tests/test_cli_unimolecular.py -v`

Expected: FAIL because CLI still imports `reactx.placement`, still expects `cfg.placement`, and does not write `endpoint_source`.

- [ ] **Step 4: Refactor CLI imports and endpoint loading**

In `reactx/cli.py`:

Remove these imports:

```python
from reactx.embed3d import embed_fragments_to_positions
from reactx.placement import (
    PlacementResult,
    PlacementTrial,
    build_atoms_from_positions,
    valid_placements,
)
```

Add:

```python
from reactx.endpoints import EndpointError, load_endpoint_pair
```

Replace the log line that mentions placement with:

```python
    log.info(
        "description=%s endpoint_relax.fmax=%.4f neb.n_images=%d",
        cfg.description,
        cfg.endpoint_relax.fmax,
        cfg.neb.n_images,
    )
```

Replace the entire R/P embed-placement-selection block with:

```python
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)

    try:
        atoms_r, atoms_p, endpoint_source = load_endpoint_pair(
            cfg,
            rxn_path=args.rxn_path,
            reactant_mol_h=r_h,
            product_mol_h=p_h,
        )
    except EndpointError as exc:
        log.error("endpoint error: %s", exc)
        return 2

    try:
        atoms_r_relaxed, info_r = relax_endpoint(
            atoms_r,
            calc,
            fmax=cfg.endpoint_relax.fmax,
            max_steps=cfg.endpoint_relax.max_steps,
            optimizer=cfg.endpoint_relax.optimizer,
        )
    except ValueError as exc:
        log.error("R-side endpoint relax failed: %s", exc)
        return 1
    log.info(
        "R-side endpoint relax: converged=%s n_steps=%d final_fmax=%.4f energy=%.6f",
        info_r["converged"],
        info_r["n_steps"],
        info_r["final_fmax"],
        info_r["energy"],
    )

    try:
        atoms_p_relaxed, info_p = relax_endpoint(
            atoms_p,
            calc,
            fmax=cfg.endpoint_relax.fmax,
            max_steps=cfg.endpoint_relax.max_steps,
            optimizer=cfg.endpoint_relax.optimizer,
        )
    except ValueError as exc:
        log.error("P-side endpoint relax failed: %s", exc)
        return 1
    log.info(
        "P-side endpoint relax: converged=%s n_steps=%d final_fmax=%.4f energy=%.6f",
        info_p["converged"],
        info_p["n_steps"],
        info_p["final_fmax"],
        info_p["energy"],
    )
```

In `meta`, add top-level endpoint and bond-change metadata:

```python
        "endpoint_source": endpoint_source,
        "bond_changes": {
            "formed": [[int(a), int(b)] for a, b in bond_changes.formed],
            "broken": [[int(a), int(b)] for a, b in bond_changes.broken],
        },
```

Remove this entire object from `effective_params`:

```python
            "placement": {
                "orientation": cfg.placement.orientation,
                "n_candidates": cfg.placement.n_candidates,
                "relaxed_candidates": cfg.placement.relaxed_candidates,
                "r_side": _placement_meta(placement_r, info_r),
                "p_side": _placement_meta(placement_p, info_p),
            },
```

Delete helper functions from `_placement_changes_for_side` through `_placement_meta`.

- [ ] **Step 5: Update slow test TOML constants**

In `tests/test_re1_sn2.py`, replace `_SN2_TOML` with:

```python
_SN2_TOML = """\
description = "SN2 anion test"
reactant_structure = "sn2.reactant.xyz"
product_structure = "sn2.product.xyz"
"""
```

In `tests/test_diels_alder_simple.py`, replace `_DA_SIMPLE_TOML` with:

```python
_DA_SIMPLE_TOML = """\
description = "DA simple test"
reactant_structure = "diels_alder_simple.reactant.xyz"
product_structure = "diels_alder_simple.product.xyz"
"""
```

In `tests/test_diels_alder_endo.py`, replace its TOML fixture with:

```python
_DA_ENDO_TOML = """\
description = "DA endo test"
reactant_structure = "diels_alder_endo.reactant.xyz"
product_structure = "diels_alder_endo.product.xyz"
"""
```

- [ ] **Step 6: Run CLI tests and commit**

Run: `pytest tests/test_cli.py tests/test_cli_unimolecular.py -v`

Expected: PASS.

Commit:

```bash
git add reactx/cli.py tests/test_cli.py tests/test_cli_unimolecular.py tests/test_re1_sn2.py tests/test_diels_alder_simple.py tests/test_diels_alder_endo.py
git commit -m "feat: run cli from explicit endpoints"
```

---

## Task 5: Delete Placement Engine And Update Remaining Tests

**Files:**
- Delete: `reactx/placement.py`
- Delete: `tests/test_placement.py`
- Delete: `reactx/embed3d.py`
- Delete: `tests/test_embed3d.py`
- Modify: `tests/test_align.py`
- Modify: `reactx/rxn_parser.py`
- Modify: `reactx/vdw_radii.py`

- [ ] **Step 1: Confirm remaining placement/embed references**

Run: `rg -n "reactx\\.placement|valid_placements|build_atoms_from_positions|embed_fragments_to_positions|reactx\\.embed3d" reactx tests`

Expected: references remain in `reactx/placement.py`, `tests/test_placement.py`, `reactx/embed3d.py`, `tests/test_embed3d.py`, and `tests/test_align.py`.

- [ ] **Step 2: Rewrite align test helper to use ASE directly**

In `tests/test_align.py`, remove:

```python
    from reactx.embed3d import embed_fragments_to_positions
    from reactx.placement import build_atoms_from_positions
```

Add near the top imports:

```python
from ase import Atoms
```

In the affected test, replace the embed/build block with:

```python
    symbols_r = [a.GetSymbol() for a in r_h.GetAtoms()]
    symbols_p = [a.GetSymbol() for a in p_h.GetAtoms()]
    atoms_r = Atoms(
        symbols=symbols_r,
        positions=[[float(i), 0.0, 0.0] for i in range(len(symbols_r))],
    )
    atoms_p = Atoms(
        symbols=symbols_p,
        positions=[[float(i), 0.1, 0.0] for i in range(len(symbols_p))],
    )
```

- [ ] **Step 3: Delete placement and embed modules/tests**

Delete these files:

```text
reactx/placement.py
tests/test_placement.py
reactx/embed3d.py
tests/test_embed3d.py
```

Update `reactx/rxn_parser.py` docstring for `atom_map_to_reactant_idx` to remove the placement reference:

```python
    indices that BondChanges and endpoint construction expect.
```

Update `reactx/vdw_radii.py` module docstring from:

```python
"""Shared van der Waals radius table for placement and rendering.
```

to:

```python
"""Shared van der Waals radius table for rendering and geometry checks.
```

- [ ] **Step 4: Verify no deleted APIs remain referenced**

Run: `rg -n "reactx\\.placement|valid_placements|build_atoms_from_positions|embed_fragments_to_positions|reactx\\.embed3d|PlacementSection|cfg\\.placement" reactx tests README.md examples`

Expected: no output except README references that will be removed in Task 6.

- [ ] **Step 5: Run focused tests and commit**

Run: `pytest tests/test_align.py tests/test_config_phase11.py tests/test_endpoints.py tests/test_cli.py tests/test_cli_unimolecular.py -v`

Expected: PASS.

Commit:

```bash
git add -u reactx tests
git commit -m "refactor: remove placement engine"
```

---

## Task 6: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite README introduction and schema**

Update README so the opening says:

```markdown
# reactx - Explicit-endpoint CI-NEB Reaction Path Engine

2D reaction mechanisms (`.rxn`) plus sidecar TOML config are used for atom
mapping and connectivity changes. Complete reactant/product 3D endpoint
structures are supplied as XYZ files, relaxed independently, then connected with
CI-NEB (IDPP interpolation + two-phase climb). The resulting trajectory can be
rendered in Blender as a ball-and-stick animation.
```

Replace the Phase 11 section with:

```markdown
## Explicit endpoint inputs

- `.rxn` provides atom mapping and reactant/product connectivity.
- `.rxn.toml` requires `reactant_structure` and `product_structure`.
- XYZ endpoint files provide the complete NEB endpoint coordinates.
- CLI does not run Fibonacci fragment placement or candidate selection.
- `meta.json` contains `endpoint_source`, `bond_changes`, endpoint relaxation
  summaries, NEB summaries, and effective parameters.
```

Replace the TOML schema block with:

```toml
description = "SN2 anion: CH3Cl + OH- -> CH3OH + Cl-"
reactant_structure = "sn2.reactant.xyz"
product_structure = "sn2.product.xyz"

[endpoint_relax]
fmax = 0.01
max_steps = 500
optimizer = "FIRE"       # "FIRE" | "BFGS"

[neb]
n_images = 11
fmax = 0.05
max_steps = 200
k = 1.0
climb = true
pad_frames = 0
```

- [ ] **Step 2: Update architecture diagram**

Replace the README architecture block with:

```text
.rxn + .rxn.toml -> parse_rxn + load_config -> ReactionConfig + atom mapping
                                      |
                                      v
                    BondChanges.from_reaction_diff(r_mol, p_mol)
                                      |
                                      v
                    load explicit reactant/product XYZ endpoints
                                      |
                                      v
                    validate atom count and element order
                                      |
              +-----------------------+-----------------------+
              v                                               v
       relax_endpoint(R, UMA)                         relax_endpoint(P, UMA)
              +-----------------------+-----------------------+
                                      v
                      align_product_to_reactant
                                      |
                                      v
                      run_neb (IDPP + two-phase CI-NEB)
                                      |
                                      v
                      trajectory.xyz -> blender/render.py -> .blend
```

Update key files list to include `reactx/endpoints.py` and remove `reactx/placement.py` / `reactx/embed3d.py`.

- [ ] **Step 3: Update examples table**

Use this table shape:

```markdown
| `.rxn` | endpoint files | notes |
|---|---|---|
| `sn2.rxn` | `sn2.reactant.xyz`, `sn2.product.xyz` | SN2 Walden endpoint orientation |
| `proton_transfer.rxn` | `proton_transfer.reactant.xyz`, `proton_transfer.product.xyz` | all defaults |
| `menshutkin.rxn` | `menshutkin.reactant.xyz`, `menshutkin.product.xyz` | all defaults |
| `e2.rxn` | `e2.reactant.xyz`, `e2.product.xyz` | all defaults |
| `sn1_dissoc.rxn` | `sn1_dissoc.reactant.xyz`, `sn1_dissoc.product.xyz` | all defaults |
| `sn1_recomb.rxn` | `sn1_recomb.reactant.xyz`, `sn1_recomb.product.xyz` | all defaults |
| `diels_alder_simple.rxn` | `diels_alder_simple.reactant.xyz`, `diels_alder_simple.product.xyz` | explicit simple orientation |
| `diels_alder_endo.rxn` | `diels_alder_endo.reactant.xyz`, `diels_alder_endo.product.xyz` | explicit endo orientation |
```

- [ ] **Step 4: Verify README has no placement-era usage references**

Run: `rg -n "Fibonacci|placement|n_candidates|relaxed_candidates|orientation|embed_fragments" README.md`

Expected: no output, except one historical sentence is acceptable only if it explicitly says the placement path was removed.

- [ ] **Step 5: Commit documentation**

Run: `pytest tests/test_config_phase11.py tests/test_endpoints.py -v`

Expected: PASS.

Commit:

```bash
git add README.md
git commit -m "docs: document explicit endpoint inputs"
```

---

## Task 7: Full Verification

**Files:**
- No new files.
- May modify files only to fix failures found by verification.

- [ ] **Step 1: Run static search for removed schema and APIs**

Run:

```powershell
rg -n "PlacementSection|cfg\.placement|reactx\.placement|valid_placements|build_atoms_from_positions|embed_fragments_to_positions|reactx\.embed3d|\[placement\]|n_candidates|relaxed_candidates" reactx tests examples README.md
```

Expected: no output. If output appears in a design/plan document, do not change historical docs. If output appears in production, tests, examples, or README, remove or rewrite it.

- [ ] **Step 2: Run default test suite**

Run: `pytest -v`

Expected: PASS with slow and blender tests deselected by `pyproject.toml`.

- [ ] **Step 3: Run focused SN2 fast CLI smoke**

Run:

```powershell
python -m reactx.cli run examples/sn2.rxn -o out/sn2-explicit-smoke --backend lj
```

Expected: exit code 0 and `out/sn2-explicit-smoke/meta.json` contains:

```json
"endpoint_source": {
  "mode": "explicit_xyz"
}
```

- [ ] **Step 4: Run lint if available**

Run: `ruff check .`

Expected: PASS. If `ruff` is unavailable, install is not required; record that verification gap in the final handoff.

- [ ] **Step 5: Commit verification fixes**

If any verification fixes were made:

```bash
git add reactx tests examples README.md
git commit -m "fix: complete explicit endpoint migration"
```

If no fixes were made, do not create an empty commit.

---

## Self-Review

- Spec coverage: The plan covers explicit TOML endpoint paths, endpoint loading and validation, CLI no-placement path, metadata, examples migration, placement deletion, README updates, and default verification.
- Red-flag scan: The executable tasks contain concrete code, commands, and expected outcomes.
- Type consistency: `ReactionConfig.reactant_structure`, `ReactionConfig.product_structure`, `EndpointError`, and `load_endpoint_pair()` are introduced before use in CLI tasks. `endpoint_source` metadata shape is consistent across loader, CLI tests, and README.
