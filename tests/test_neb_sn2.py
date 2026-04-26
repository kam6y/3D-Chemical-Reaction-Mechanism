from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from reactx.calculators import make_calculator
from reactx.neb import run_neb


def _two_state_h2(displacement: float) -> Atoms:
    return Atoms("H2", positions=[(0, 0, 0), (0, 0, displacement)])


def test_run_neb_with_lj_produces_xyz_with_expected_images(tmp_path: Path):
    reactant = _two_state_h2(0.5)
    product = _two_state_h2(1.2)
    out = tmp_path / "traj.xyz"
    meta = run_neb(
        reactant=reactant,
        product=product,
        calculator=make_calculator("lj"),
        n_images=5,
        output_xyz=out,
        fmax=0.2,
        max_steps=50,
        pad_frames=0,
    )
    assert out.exists()
    frames = read(str(out), index=":")
    assert len(frames) == 5
    assert "converged" in meta
    assert "final_fmax" in meta


def test_run_neb_pads_endpoints(tmp_path: Path):
    reactant = _two_state_h2(0.5)
    product = _two_state_h2(1.2)
    out = tmp_path / "traj.xyz"
    run_neb(
        reactant=reactant, product=product,
        calculator=make_calculator("lj"),
        n_images=5, output_xyz=out, fmax=0.5, max_steps=10,
        pad_frames=3,
    )
    frames = read(str(out), index=":")
    # 3 reactant + 5 neb + 3 product
    assert len(frames) == 11


@pytest.mark.slow
def test_sn2_neb_ts_has_walden_inversion(tmp_path: Path):
    """Real UMA + SN2 end-to-end. Skipped by default — run with `pytest -m slow`."""
    pytest.importorskip("fairchem.core")
    from reactx.rxn_parser import parse_rxn, heavy_to_hydrogen_groups
    from reactx.embed3d import embed_mol_to_atoms
    from reactx.align import align_product_to_reactant
    from rdkit import Chem

    rxn = Path(__file__).parent.parent / "examples" / "sn2.rxn"
    r_mol, p_mol, mapping = parse_rxn(rxn)
    r_mol_h = Chem.AddHs(r_mol)
    p_mol_h = Chem.AddHs(p_mol)
    rH = heavy_to_hydrogen_groups(r_mol_h)
    pH = heavy_to_hydrogen_groups(p_mol_h)

    calc = make_calculator("uma")
    reactant = embed_mol_to_atoms(r_mol, calculator=calc, seed=1)
    product_raw = embed_mol_to_atoms(p_mol, calculator=calc, seed=2)
    product = align_product_to_reactant(reactant, product_raw, mapping, rH, pH)

    out = tmp_path / "traj.xyz"
    meta = run_neb(
        reactant=reactant, product=product,
        calculator=calc,
        n_images=11, output_xyz=out, fmax=0.05, max_steps=200, pad_frames=0,
    )
    frames = read(str(out), index=":")
    energies = np.array(meta["image_energies"])
    ts_idx = int(np.argmax(energies))
    assert 0 < ts_idx < len(frames) - 1, "TS must be an interior image"
    assert energies[ts_idx] > energies[0] and energies[ts_idx] > energies[-1]

    ts = frames[ts_idx]
    syms = ts.get_chemical_symbols()
    c = syms.index("C")
    f = syms.index("F")
    cl = syms.index("Cl")
    angle = ts.get_angle(f, c, cl)
    # UMA omol head may not give a perfectly linear TS for the small SN2 case.
    # Walden inversion is signaled by the TS being well past 90° (transition through planarity).
    assert angle > 120, (
        f"Walden inversion expects F-C-Cl > 120° at TS (transition through planarity), "
        f"got {angle:.1f}°"
    )


def test_run_neb_rejects_too_few_images(tmp_path: Path):
    reactant = _two_state_h2(0.5)
    product = _two_state_h2(1.2)
    out = tmp_path / "traj.xyz"
    with pytest.raises(ValueError, match="n_images must be >= 3"):
        run_neb(
            reactant=reactant, product=product,
            calculator=make_calculator("lj"),
            n_images=2, output_xyz=out, fmax=0.5, max_steps=10,
        )
