"""CLI entry point: reactx run <rxn> -o <outdir> [options] (Phase 11)."""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

from rdkit import Chem

from reactx.align import align_product_to_reactant
from reactx.bond_changes import BondChanges
from reactx.calculators import make_calculator
from reactx.config import ConfigError, load_config
from reactx.endpoint_relax import relax_endpoint
from reactx.endpoints import EndpointError, load_endpoint_pair
from reactx.neb import run_neb
from reactx.rxn_parser import heavy_to_hydrogen_groups, parse_rxn

log = logging.getLogger("reactx")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reactx")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run the Pure CI-NEB pipeline on a .rxn file")
    run.add_argument("rxn_path", type=Path)
    run.add_argument("-o", "--output", type=Path, required=True)
    run.add_argument("--backend", choices=["uma", "lj"], default="uma")
    run.add_argument(
        "--model",
        type=str,
        default="uma-m-1p1",
        help="UMA model name (uma-m-1p1, uma-s-1p2, ...)",
    )
    run.add_argument(
        "--render",
        action="store_true",
        help="Invoke blender/render.py after the pipeline",
    )
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
        log.error(
            "Hugging Face authentication check failed (%s: %s).",
            type(exc).__name__,
            exc,
        )
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

    log.info(
        "description=%s endpoint_relax.fmax=%.4f neb.n_images=%d",
        cfg.description,
        cfg.endpoint_relax.fmax,
        cfg.neb.n_images,
    )

    model_kwargs = {"model_name": args.model} if args.backend == "uma" else {}
    calc = make_calculator(args.backend, **model_kwargs)

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
    except Exception as exc:  # noqa: BLE001
        log.error("R-side endpoint relax failed: %s", exc)
        return 1
    log.info(
        "R-side endpoint relax: converged=%s n_steps=%d "
        "final_fmax=%.4f energy=%.6f",
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
    except Exception as exc:  # noqa: BLE001
        log.error("P-side endpoint relax failed: %s", exc)
        return 1
    log.info(
        "P-side endpoint relax: converged=%s n_steps=%d "
        "final_fmax=%.4f energy=%.6f",
        info_p["converged"],
        info_p["n_steps"],
        info_p["final_fmax"],
        info_p["energy"],
    )

    try:
        atoms_p_aligned = align_product_to_reactant(
            atoms_r_relaxed,
            atoms_p_relaxed,
            heavy_mapping,
            heavy_to_hydrogen_groups(r_h),
            heavy_to_hydrogen_groups(p_h),
        )
    except ValueError as exc:
        log.error("align failed: %s", exc)
        return 1

    xyz = args.output / "trajectory.xyz"
    log.info(
        "running CI-NEB: n_images=%d max_steps=%d",
        cfg.neb.n_images,
        cfg.neb.max_steps,
    )
    neb_info = run_neb(
        atoms_r_relaxed,
        atoms_p_aligned,
        calculator=calc,
        n_images=cfg.neb.n_images,
        output_xyz=xyz,
        fmax=cfg.neb.fmax,
        max_steps=cfg.neb.max_steps,
        k=cfg.neb.k,
        climb=cfg.neb.climb,
        pad_frames=cfg.neb.pad_frames,
    )
    log.info(
        "NEB: converged=%s final_fmax=%.4f",
        neb_info["converged"],
        float(neb_info["final_fmax"]),
    )

    (args.output / "energies.json").write_text(
        json.dumps([float(e) for e in neb_info["image_energies"]])
    )

    meta = {
        "backend": args.backend,
        "description": cfg.description,
        "wall_clock_seconds": float(time.monotonic() - t_start),
        "endpoint_source": endpoint_source,
        "bond_changes": {
            "formed": [list(pair) for pair in bond_changes.formed],
            "broken": [list(pair) for pair in bond_changes.broken],
        },
        "endpoint_relax_r": {
            "converged": bool(info_r["converged"]),
            "final_fmax": float(info_r["final_fmax"]),
            "n_steps": int(info_r["n_steps"]),
        },
        "endpoint_relax_p": {
            "converged": bool(info_p["converged"]),
            "final_fmax": float(info_p["final_fmax"]),
            "n_steps": int(info_p["n_steps"]),
        },
        "neb": {
            "n_images": int(neb_info["n_images"]),
            "converged": bool(neb_info["converged"]),
            "final_fmax": float(neb_info["final_fmax"]),
            "image_energies": [float(e) for e in neb_info["image_energies"]],
            "k": float(neb_info["k"]),
            "pad_frames": int(neb_info["pad_frames"]),
        },
        "effective_params": {
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
    (args.output / "meta.json").write_text(
        json.dumps(_sanitize_for_json(meta), indent=2)
    )

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
    cmd = [
        args.blender_exe,
        "--background",
        "--python",
        str(script),
        "--",
        str(xyz),
        str(blend),
    ]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error("Error: blender exited with code %d", result.returncode)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
