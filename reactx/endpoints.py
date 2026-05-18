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
