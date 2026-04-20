"""CLI entry point: reactx run <rxn> -o <outdir> [options]."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from rdkit import Chem

from reactx.align import align_product_to_reactant
from reactx.calculators import make_calculator
from reactx.embed3d import embed_mol_to_atoms
from reactx.neb import run_neb
from reactx.rxn_parser import heavy_to_hydrogen_groups, parse_rxn


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reactx")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run full pipeline on a .rxn file")
    run.add_argument("rxn_path", type=Path)
    run.add_argument("-o", "--output", type=Path, required=True)
    run.add_argument("--images", type=int, default=11)
    run.add_argument("--fmax", type=float, default=0.05)
    run.add_argument("--max-steps", type=int, default=200)
    run.add_argument("--backend", choices=["uma", "lj"], default="uma")
    run.add_argument("--pad-frames", type=int, default=3)
    run.add_argument("--render", action="store_true",
                     help="Also invoke blender/render.py after NEB")
    run.add_argument("--blender-exe", type=str, default="blender")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        return _cmd_run(args)
    return 2


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
    if not args.rxn_path.exists():
        print(f"Error: .rxn not found: {args.rxn_path}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    r_mol, p_mol, mapping = parse_rxn(args.rxn_path)

    reactant = embed_mol_to_atoms(
        r_mol, calculator=make_calculator(args.backend), seed=1,
    )
    product_raw = embed_mol_to_atoms(
        p_mol, calculator=make_calculator(args.backend), seed=2,
    )

    r_mol_h = Chem.AddHs(r_mol)
    p_mol_h = Chem.AddHs(p_mol)
    rH = heavy_to_hydrogen_groups(r_mol_h)
    pH = heavy_to_hydrogen_groups(p_mol_h)
    product = align_product_to_reactant(reactant, product_raw, mapping, rH, pH)

    xyz = args.output / "trajectory.xyz"
    meta = run_neb(
        reactant=reactant,
        product=product,
        calculator_factory=lambda: make_calculator(args.backend),
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

    print(f"OK: wrote {xyz} (converged={meta['converged']}, fmax={_fmt_fmax(meta['final_fmax'])})")
    return 0


def _invoke_blender(args: argparse.Namespace, xyz: Path) -> int:
    import subprocess
    script = Path(__file__).resolve().parent.parent / "blender" / "render.py"
    blend = args.output / "scene.blend"
    cmd = [args.blender_exe, "--background", "--python", str(script),
           "--", str(xyz), str(blend)]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            f"Error: blender exited with code {result.returncode}",
            file=sys.stderr,
        )
        return 1
    return 0
