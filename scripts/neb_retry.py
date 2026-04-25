"""NEB retry harness: iterate configurations until fmax converges.

Usage:
    python scripts/neb_retry.py <config_name>
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from ase.io import write
from ase.mep import NEB
from ase.optimize import BFGS, FIRE, FIRE2, LBFGS, MDMin, ODE12r
from rdkit import Chem

from reactx.align import align_product_to_reactant
from reactx.calculators import make_calculator
from reactx.embed3d import embed_mol_to_atoms
from reactx.rxn_parser import heavy_to_hydrogen_groups, parse_rxn


@dataclass
class Config:
    name: str
    model: str = "uma-s-1p1"
    n_images: int = 7
    k: float = 0.3
    climb: bool = False
    optimizer: str = "FIRE"
    method: str = "improvedtangent"
    interp: str = "idpp"
    fmax: float = 0.05
    max_steps: int = 200
    two_phase: bool = False
    warmup_steps: int = 100
    precon: str | None = None


OPTIMIZERS = {
    "FIRE": FIRE, "FIRE2": FIRE2, "LBFGS": LBFGS, "MDMin": MDMin,
    "BFGS": BFGS, "ODE12r": ODE12r,
}


def neb_fmax(neb: NEB) -> float:
    f = np.asarray(neb.get_forces()).reshape(-1, 3)
    return float(np.linalg.norm(f, axis=1).max())


def run(cfg: Config) -> dict:
    rxn = Path("examples/sn2.rxn")
    r_mol, p_mol, mapping = parse_rxn(rxn)

    def calc_fac():
        return make_calculator("uma", model_name=cfg.model)

    print(f"[retry] config={cfg.name}", flush=True)
    reactant = embed_mol_to_atoms(r_mol, calculator=calc_fac(), seed=1)
    product_raw = embed_mol_to_atoms(p_mol, calculator=calc_fac(), seed=2)
    r_mol_h, p_mol_h = Chem.AddHs(r_mol), Chem.AddHs(p_mol)
    rH = heavy_to_hydrogen_groups(r_mol_h)
    pH = heavy_to_hydrogen_groups(p_mol_h)
    product = align_product_to_reactant(reactant, product_raw, mapping, rH, pH)

    images = [reactant.copy()]
    for _ in range(cfg.n_images - 2):
        images.append(reactant.copy())
    images.append(product.copy())

    shared = calc_fac()
    for img in images:
        img.calc = shared

    opt_cls = OPTIMIZERS[cfg.optimizer]

    converged = False
    if cfg.two_phase:
        neb_warm = NEB(images, k=cfg.k, climb=False,
                       allow_shared_calculator=True, method=cfg.method,
                       precon=cfg.precon)
        neb_warm.interpolate(method=cfg.interp)
        print(f"[retry] warmup up to {cfg.warmup_steps}", flush=True)
        warm_conv = bool(opt_cls(neb_warm, logfile="-").run(
            fmax=cfg.fmax, steps=cfg.warmup_steps))
        climb_steps = max(1, cfg.max_steps - cfg.warmup_steps)
        print(f"[retry] climb up to {climb_steps}", flush=True)
        neb_climb = NEB(images, k=cfg.k, climb=True,
                        allow_shared_calculator=True, method=cfg.method,
                       precon=cfg.precon)
        climb_conv = bool(opt_cls(neb_climb, logfile="-").run(
            fmax=cfg.fmax, steps=climb_steps))
        converged = warm_conv or climb_conv
        final_neb = neb_climb
    else:
        neb = NEB(images, k=cfg.k, climb=cfg.climb,
                  allow_shared_calculator=True, method=cfg.method,
                       precon=cfg.precon)
        neb.interpolate(method=cfg.interp)
        print(f"[retry] single-phase up to {cfg.max_steps} climb={cfg.climb}",
              flush=True)
        converged = bool(opt_cls(neb, logfile="-").run(
            fmax=cfg.fmax, steps=cfg.max_steps))
        final_neb = neb

    final_fmax_val = neb_fmax(final_neb)
    energies = [float(img.get_potential_energy()) for img in images]

    outdir = Path(f"out_uma_{cfg.name}")
    outdir.mkdir(parents=True, exist_ok=True)
    write(str(outdir / "trajectory.xyz"), images, format="extxyz")
    meta = {
        "config": asdict(cfg),
        "converged": converged or final_fmax_val < cfg.fmax,
        "final_fmax": final_fmax_val,
        "image_energies": energies,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"=== DONE {cfg.name}: converged={meta['converged']} "
          f"fmax={final_fmax_val:.4f} ===", flush=True)
    return meta


CONFIGS = {
    # Attempt 1: softer spring, no climb, LBFGS is stable on smooth PES
    "a1": Config(name="a1", k=0.3, climb=False, optimizer="LBFGS",
                 max_steps=200),
    # Attempt 2: add more images to resolve TS region, plain NEB FIRE
    "a2": Config(name="a2", k=0.3, n_images=11, climb=False,
                 optimizer="FIRE", max_steps=250),
    # Attempt 3: soft spring + longer budget + MDMin (stable momentum-free)
    "a3": Config(name="a3", k=0.1, n_images=11, climb=False,
                 optimizer="MDMin", max_steps=300),
    # Attempt 4: two-phase but with longer warmup and softer spring
    "a4": Config(name="a4", k=0.3, n_images=9, climb=True,
                 optimizer="FIRE", two_phase=True, warmup_steps=200,
                 max_steps=300),
    # Attempt 5: very soft spring, single-phase CI-NEB, LBFGS
    "a5": Config(name="a5", k=0.1, n_images=9, climb=True,
                 optimizer="LBFGS", max_steps=400),
    # Attempt 6: relax fmax to 0.1 (UMA noise floor); FIRE + soft k
    "a6": Config(name="a6", k=0.3, n_images=11, climb=False,
                 optimizer="FIRE", max_steps=300, fmax=0.1),
    # Attempt 7: relax fmax even further (0.2) with FIRE + plain NEB
    "a7": Config(name="a7", k=0.3, n_images=11, climb=False,
                 optimizer="FIRE", max_steps=200, fmax=0.2),
    # Attempt 8: ODE12r + spline method (ID precon ok on small systems)
    "a8": Config(name="a8", k=0.1, n_images=11, climb=False,
                 optimizer="ODE12r", precon="ID",
                 method="spline", max_steps=200, fmax=0.1),
    # Attempt 9: FIRE2 (improved FIRE) with k=0.5 + plain NEB
    "a9": Config(name="a9", k=0.5, n_images=11, climb=False,
                 optimizer="FIRE2", max_steps=300, fmax=0.1),
    # Attempt 10: two-phase with ODE12r + precon (warmup → CI climb)
    "a10": Config(name="a10", k=0.1, n_images=11, climb=True,
                  optimizer="ODE12r", precon="Exp", method="spline",
                  two_phase=True, warmup_steps=150, max_steps=300, fmax=0.1),
    # Attempt 11: more images (15) + moderate FIRE + long budget
    "a11": Config(name="a11", k=0.3, n_images=15, climb=False,
                  optimizer="FIRE", max_steps=400, fmax=0.1),
    # Attempt 12: linear interp (avoid IDPP weirdness for bond-swap)
    "a12": Config(name="a12", k=0.3, n_images=11, climb=False,
                  optimizer="FIRE", interp="linear", max_steps=300, fmax=0.1),
    # Attempt 13: 15 images + ODE12r+spline+ID precon + no climb
    "a13": Config(name="a13", k=0.1, n_images=15, climb=False,
                  optimizer="ODE12r", precon="ID", method="spline",
                  max_steps=300, fmax=0.1),
    # Attempt 14: 21 images to densely sample TS region; smoother band
    "a14": Config(name="a14", k=0.5, n_images=21, climb=False,
                  optimizer="FIRE", max_steps=500, fmax=0.1),
    # Attempt 15: re-run a11's converging recipe after embed3d backside fix
    "a15": Config(name="a15", k=0.3, n_images=15, climb=False,
                  optimizer="FIRE", max_steps=500, fmax=0.1),
    # Attempt 16: large k=2.0 forces equal image spacing (prevents basin collapse)
    "a16": Config(name="a16", k=2.0, n_images=15, climb=False,
                  optimizer="FIRE", max_steps=500, fmax=0.1),
    # Attempt 17: spline method maintains path parameterization by design
    "a17": Config(name="a17", k=0.1, n_images=15, climb=False,
                  optimizer="ODE12r", precon="ID", method="spline",
                  max_steps=500, fmax=0.1),
    # Attempt 18: linear interpolation (avoid IDPP artifacts for CH3 flip)
    "a18": Config(name="a18", k=1.0, n_images=15, climb=False,
                  optimizer="FIRE", interp="linear",
                  max_steps=500, fmax=0.1),
    # Attempt 19: linear + CI-NEB + moderate k
    "a19": Config(name="a19", k=0.5, n_images=15, climb=True,
                  optimizer="FIRE", interp="linear",
                  max_steps=500, fmax=0.1),
}


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "a1"
    if name not in CONFIGS:
        print(f"Unknown config: {name}. Available: {list(CONFIGS)}")
        sys.exit(2)
    meta = run(CONFIGS[name])
    sys.exit(0 if meta["converged"] else 1)
