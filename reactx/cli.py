"""CLI entry point: reactx run <rxn> -o <outdir> [options]."""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

from rdkit import Chem

from reactx.align import align_product_to_reactant
from reactx.calculators import make_calculator
from reactx.embed3d import embed_mol_to_atoms
from reactx.neb import run_neb
from reactx.rxn_parser import heavy_to_hydrogen_groups, parse_rxn

log = logging.getLogger("reactx")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reactx")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run full pipeline on a .rxn file")
    run.add_argument("rxn_path", type=Path)
    run.add_argument("-o", "--output", type=Path, required=True)
    run.add_argument("--images", type=int, default=15)
    run.add_argument("--fmax", type=float, default=0.05)
    run.add_argument("--max-steps", type=int, default=500)
    run.add_argument("--backend", choices=["uma", "lj"], default="uma")
    run.add_argument("--model", type=str, default="uma-m-1p1",
                     help="UMA model name (uma-m-1p1, uma-s-1p2, ...)")
    run.add_argument("--pad-frames", type=int, default=3)
    run.add_argument("--render", action="store_true",
                     help="Also invoke blender/render.py after NEB")
    run.add_argument("--blender-exe", type=str, default="blender")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _cmd_run(args)


def _configure_reactx_logging() -> None:
    log = logging.getLogger("reactx")
    if log.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[reactx] %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def _fmt_fmax(v: float) -> str:
    return "nan" if math.isnan(v) else f"{v:.4f}"


def _sanitize_for_json(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _cmd_run(args: argparse.Namespace) -> int:
    _configure_reactx_logging()

    if not args.rxn_path.exists():
        log.error("Error: .rxn not found: %s", args.rxn_path)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    r_mol, p_mol, mapping = parse_rxn(args.rxn_path)

    model_kwargs = {"model_name": args.model} if args.backend == "uma" else {}
    # Reuse one calculator across embed+NEB; UMA models (~11 GB) OOM if rebuilt.
    calc = make_calculator(args.backend, **model_kwargs)
    reactant = embed_mol_to_atoms(r_mol, calculator=calc, seed=1)
    product_raw = embed_mol_to_atoms(p_mol, calculator=calc, seed=2)

    rH = heavy_to_hydrogen_groups(Chem.AddHs(r_mol))
    pH = heavy_to_hydrogen_groups(Chem.AddHs(p_mol))
    product = align_product_to_reactant(reactant, product_raw, mapping, rH, pH)

    xyz = args.output / "trajectory.xyz"
    meta = run_neb(
        reactant=reactant,
        product=product,
        calculator=calc,
        n_images=args.images,
        output_xyz=xyz,
        fmax=args.fmax,
        max_steps=args.max_steps,
        pad_frames=args.pad_frames,
    )

    meta_clean = _sanitize_for_json(meta)
    (args.output / "meta.json").write_text(json.dumps(meta_clean, indent=2))
    (args.output / "energies.json").write_text(json.dumps(meta_clean["image_energies"]))

    if args.render:
        rc = _invoke_blender(args, xyz)
        if rc != 0:
            return rc

    log.info(
        "OK: wrote %s (converged=%s, fmax=%s)",
        xyz, meta["converged"], _fmt_fmax(meta["final_fmax"]),
    )
    return 0


def _invoke_blender(args: argparse.Namespace, xyz: Path) -> int:
    import subprocess
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
