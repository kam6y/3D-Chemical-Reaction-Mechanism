"""Tests for explicit endpoint structure loading."""
from pathlib import Path

import pytest
from rdkit import Chem

from reactx.config import EndpointRelaxSection, NEBSection, ReactionConfig
from reactx.endpoints import EndpointError, load_endpoint_pair


def _cfg(
    *,
    reactant_structure: str = "r.xyz",
    product_structure: str = "p.xyz",
) -> ReactionConfig:
    return ReactionConfig(
        description="x",
        reactant_structure=reactant_structure,
        product_structure=product_structure,
        endpoint_relax=EndpointRelaxSection(),
        neb=NEBSection(),
    )


def _write_xyz(path: Path, symbols: list[str], *, frames: int = 1) -> None:
    blocks = []
    for frame in range(frames):
        lines = [str(len(symbols)), f"frame {frame}"]
        lines.extend(f"{symbol} {i:.1f} 0.0 0.0" for i, symbol in enumerate(symbols))
        blocks.append("\n".join(lines))
    path.write_text("\n".join(blocks) + "\n", encoding="utf-8")


def _mol_h(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    mol_h = Chem.AddHs(mol)
    Chem.SanitizeMol(mol_h)
    return mol_h


def _symbols(mol_h: Chem.Mol) -> list[str]:
    return [atom.GetSymbol() for atom in mol_h.GetAtoms()]


def test_load_endpoint_pair_resolves_relative_paths_and_returns_metadata(tmp_path):
    sidecar_dir = tmp_path / "case"
    sidecar_dir.mkdir()
    rxn_path = sidecar_dir / "reaction.rxn"
    rxn_path.write_text("$RXN\n", encoding="utf-8")

    reactant_h = _mol_h("[NH4+]")
    product_h = _mol_h("[Cl-]")
    _write_xyz(sidecar_dir / "r.xyz", _symbols(reactant_h))
    _write_xyz(sidecar_dir / "p.xyz", _symbols(product_h))

    atoms_r, atoms_p, meta = load_endpoint_pair(
        _cfg(),
        rxn_path=rxn_path,
        reactant_mol_h=reactant_h,
        product_mol_h=product_h,
    )

    assert atoms_r.get_chemical_symbols() == _symbols(reactant_h)
    assert atoms_p.get_chemical_symbols() == _symbols(product_h)
    assert meta == {
        "mode": "explicit_xyz",
        "reactant_structure": str((sidecar_dir / "r.xyz").resolve()),
        "product_structure": str((sidecar_dir / "p.xyz").resolve()),
    }


def test_load_endpoint_pair_rejects_missing_endpoint_file(tmp_path):
    rxn_path = tmp_path / "reaction.rxn"
    rxn_path.write_text("$RXN\n", encoding="utf-8")
    mol_h = _mol_h("C")
    _write_xyz(tmp_path / "p.xyz", _symbols(mol_h))

    with pytest.raises(EndpointError, match="reactant_structure.*not found"):
        load_endpoint_pair(
            _cfg(),
            rxn_path=rxn_path,
            reactant_mol_h=mol_h,
            product_mol_h=mol_h,
        )


def test_load_endpoint_pair_rejects_atom_count_mismatch(tmp_path):
    rxn_path = tmp_path / "reaction.rxn"
    rxn_path.write_text("$RXN\n", encoding="utf-8")
    mol_h = _mol_h("C")
    _write_xyz(tmp_path / "r.xyz", ["C"])
    _write_xyz(tmp_path / "p.xyz", _symbols(mol_h))

    with pytest.raises(EndpointError, match="reactant_structure.*atom count mismatch"):
        load_endpoint_pair(
            _cfg(),
            rxn_path=rxn_path,
            reactant_mol_h=mol_h,
            product_mol_h=mol_h,
        )


def test_load_endpoint_pair_rejects_element_order_mismatch(tmp_path):
    rxn_path = tmp_path / "reaction.rxn"
    rxn_path.write_text("$RXN\n", encoding="utf-8")
    mol_h = _mol_h("C")
    wrong_order = ["H", "C", "H", "H", "H"]
    _write_xyz(tmp_path / "r.xyz", wrong_order)
    _write_xyz(tmp_path / "p.xyz", _symbols(mol_h))

    with pytest.raises(EndpointError, match="reactant_structure.*element order mismatch"):
        load_endpoint_pair(
            _cfg(),
            rxn_path=rxn_path,
            reactant_mol_h=mol_h,
            product_mol_h=mol_h,
        )


def test_load_endpoint_pair_rejects_multiple_frames(tmp_path):
    rxn_path = tmp_path / "reaction.rxn"
    rxn_path.write_text("$RXN\n", encoding="utf-8")
    mol_h = _mol_h("C")
    _write_xyz(tmp_path / "r.xyz", _symbols(mol_h), frames=2)
    _write_xyz(tmp_path / "p.xyz", _symbols(mol_h))

    with pytest.raises(EndpointError, match="reactant_structure.*multiple frames"):
        load_endpoint_pair(
            _cfg(),
            rxn_path=rxn_path,
            reactant_mol_h=mol_h,
            product_mol_h=mol_h,
        )


def test_load_endpoint_pair_sets_initial_charges_and_spin(tmp_path):
    rxn_path = tmp_path / "reaction.rxn"
    rxn_path.write_text("$RXN\n", encoding="utf-8")
    reactant_h = _mol_h("[NH4+]")
    product_h = _mol_h("[Cl-]")
    _write_xyz(tmp_path / "r.xyz", _symbols(reactant_h))
    _write_xyz(tmp_path / "p.xyz", _symbols(product_h))

    atoms_r, atoms_p, _ = load_endpoint_pair(
        _cfg(),
        rxn_path=rxn_path,
        reactant_mol_h=reactant_h,
        product_mol_h=product_h,
    )

    assert atoms_r.get_initial_charges().tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert atoms_r.info["charge"] == 1
    assert atoms_r.info["spin"] == 1
    assert atoms_p.get_initial_charges().tolist() == [-1.0]
    assert atoms_p.info["charge"] == -1
    assert atoms_p.info["spin"] == 1
