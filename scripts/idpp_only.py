"""Generate a smooth animation path via IDPP interpolation only.

NEB optimization tends to collapse images into R-complex and P-complex wells,
producing visually discontinuous animations. IDPP interpolation preserves
smooth pairwise-distance evolution without running NEB — perfect for viewing
the full Walden inversion arc.

Output: out_uma_idpp/{trajectory.xyz, meta.json, energies.json}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase.io import write
from ase.mep import NEB
from ase.mep.neb import idpp_interpolate
from rdkit import Chem

from reactx.align import align_product_to_reactant
from reactx.calculators import make_calculator
from reactx.embed3d import embed_mol_to_atoms
from reactx.rxn_parser import heavy_to_hydrogen_groups, parse_rxn


def main(n_images: int = 21, model: str = "uma-s-1p1",
         outdir: Path = Path("out_uma_idpp")) -> None:
    rxn = Path("examples/sn2.rxn")
    r_mol, p_mol, mapping = parse_rxn(rxn)

    def calc_fac():
        return make_calculator("uma", model_name=model)

    print(f"[idpp] embedding reactant/product with {model}", flush=True)
    reactant = embed_mol_to_atoms(r_mol, calculator=calc_fac(), seed=1)
    product_raw = embed_mol_to_atoms(p_mol, calculator=calc_fac(), seed=2)
    r_mol_h, p_mol_h = Chem.AddHs(r_mol), Chem.AddHs(p_mol)
    rH = heavy_to_hydrogen_groups(r_mol_h)
    pH = heavy_to_hydrogen_groups(p_mol_h)
    product = align_product_to_reactant(reactant, product_raw, mapping, rH, pH)

    images = [reactant.copy()]
    for _ in range(n_images - 2):
        images.append(reactant.copy())
    images.append(product.copy())

    shared = calc_fac()
    for img in images:
        img.calc = shared

    neb = NEB(images, allow_shared_calculator=True)
    print(f"[idpp] linear interpolate + IDPP refine", flush=True)
    neb.interpolate(method="linear")
    idpp_interpolate(neb, traj=None, log=None, fmax=0.01, steps=2000)

    energies = [float(img.get_potential_energy()) for img in images]

    outdir.mkdir(parents=True, exist_ok=True)
    write(str(outdir / "trajectory.xyz"), images, format="extxyz")
    meta = {
        "method": "idpp_only",
        "n_images": n_images,
        "model": model,
        "image_energies": energies,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    (outdir / "energies.json").write_text(json.dumps(energies))
    print(f"[idpp] wrote {outdir}", flush=True)
    for i, e in enumerate(energies):
        print(f"  image {i:2d}  E = {e:.4f}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    main(n_images=n)
