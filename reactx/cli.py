"""CLI entry point: reactx run <rxn> -o <outdir> [options]."""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
from ase.io import read, write
from rdkit import Chem

from reactx.align import align_product_to_reactant
from reactx.artificial_force import build_restraints, lookup_r_form
from reactx.bond_changes import SimpleBondChanges, compute_simple_bond_changes
from reactx.calculators import make_calculator
from reactx.embed3d import embed_mol_to_atoms
from reactx.neb import run_neb
from reactx.path_relax import relax_with_restraints
from reactx.rxn_parser import heavy_to_hydrogen_groups, parse_rxn
from reactx.scoring import TrialResult, reached_product, score_trials
from reactx.trials import sample_attack_rotations

log = logging.getLogger("reactx")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reactx")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run full pipeline on a .rxn file")
    run.add_argument("rxn_path", type=Path)
    run.add_argument("-o", "--output", type=Path, required=True)

    run.add_argument("--backend", choices=["uma", "lj"], default="uma")
    run.add_argument("--model", type=str, default="uma-m-1p1",
                     help="UMA model name (uma-m-1p1, uma-s-1p2, ...)")

    run.add_argument("--n-angles", type=int, default=8,
                     help="Number of attack-angle trials (>=1; index 0 is identity)")
    run.add_argument("--cone-half-deg", type=float, default=30.0,
                     help="Half-angle of cone within which trials are sampled")
    run.add_argument("--seed", type=int, default=0)

    run.add_argument("--r-form", type=float, default=None,
                     help="Override formed-bond target distance (Å). "
                          "Default: auto from element pair table.")
    run.add_argument("--r-broken", type=float, default=4.0)
    run.add_argument("--k-form", type=float, default=1.5)
    run.add_argument("--k-broken", type=float, default=1.0)

    run.add_argument("--max-relax-steps", type=int, default=100)
    run.add_argument("--relax-fmax", type=float, default=0.1)
    run.add_argument("--traj-stride", type=int, default=5)

    run.add_argument("--neb-refine", action="store_true",
                     help="Refine the best trial trajectory with a short NEB")
    run.add_argument("--neb-images", type=int, default=7)

    run.add_argument("--render", action="store_true",
                     help="Also invoke blender/render.py after pipeline")
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
        log.error(
            "Hugging Face token not found. Run `hf auth login` first "
            "(UMA models are gated and require an authorized account)."
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        log.error(
            "Hugging Face authentication check failed (%s: %s). "
            "Run `hf auth login` and ensure UMA model access is approved.",
            type(exc).__name__, exc,
        )
        return 1
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    _configure_reactx_logging()

    if not args.rxn_path.exists():
        log.error("Error: .rxn not found: %s", args.rxn_path)
        return 1

    if args.backend == "uma":
        rc = _check_hf_auth()
        if rc != 0:
            return rc

    args.output.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()

    r_mol, p_mol, mapping = parse_rxn(args.rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bond_changes = compute_simple_bond_changes(r_h, p_h, mapping)

    # bond_changes is reactant-space; for product embedding (neb-refine path)
    # we feed embed3d a product-space SimpleBondChanges constructed by
    # swapping roles + remapping indices. embed3d identifies the substrate as
    # the fragment containing both atoms of `broken`, so for product we set
    # `broken` = the reactant-formed bond (e.g. C-F in CH3F), which keeps the
    # substrate fragment correctly grouped. `formed` then defines where the
    # nucleophile-side fragment (Cl- in product) is placed via backside.
    a_form_r, b_form_r = bond_changes.formed
    a_brk_r, b_brk_r = bond_changes.broken
    bond_changes_product = SimpleBondChanges(
        formed=(mapping[a_brk_r], mapping[b_brk_r]),    # nucleophile-fragment placement direction
        broken=(mapping[a_form_r], mapping[b_form_r]),  # substrate-cohesion bond (intact in product)
    )

    model_kwargs = {"model_name": args.model} if args.backend == "uma" else {}
    calc = make_calculator(args.backend, **model_kwargs)

    rotations = sample_attack_rotations(
        n=args.n_angles, cone_half_deg=args.cone_half_deg, seed=args.seed,
    )

    formed_pair = bond_changes.formed
    broken_pair = bond_changes.broken
    syms_r = [a.GetSymbol() for a in r_h.GetAtoms()]
    if args.r_form is None:
        r_form_target = lookup_r_form(syms_r[formed_pair[0]], syms_r[formed_pair[1]])
    else:
        r_form_target = float(args.r_form)

    trials: list[TrialResult] = []
    for i, R in enumerate(rotations):
        rot_deg = _angle_from_identity_deg(R)
        log.info("trial %d/%d (rotation_deg=%.1f)", i + 1, len(rotations), rot_deg)
        try:
            atoms_init = embed_mol_to_atoms(
                r_mol, calculator=None, seed=1 + i,
                bond_changes=bond_changes, rotation_perturbation=R,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("trial %d embed failed: %s", i, exc)
            trials.append(TrialResult(
                trial_idx=i, rotation_deg=rot_deg, frames=[], energies=[],
                reached_product=False, peak_energy=float("inf"), n_steps=0,
            ))
            continue

        restraints = build_restraints(
            atoms_init,
            formed=[formed_pair],
            broken=[broken_pair],
            r_form=r_form_target,
            r_broken=args.r_broken,
            k_form=args.k_form,
            k_broken=args.k_broken,
        )
        try:
            frames, energies = relax_with_restraints(
                atoms_init, restraints, calc,
                max_steps=args.max_relax_steps,
                fmax=args.relax_fmax,
                traj_stride=args.traj_stride,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("trial %d relax failed: %s", i, exc)
            trials.append(TrialResult(
                trial_idx=i, rotation_deg=rot_deg, frames=[], energies=[],
                reached_product=False, peak_energy=float("inf"), n_steps=0,
            ))
            continue

        ok = reached_product(
            frames[-1],
            formed=[formed_pair],
            broken=[broken_pair],
            r_form_targets=[r_form_target],
            r_broken_target=args.r_broken,
        )
        peak = max(energies) if energies else float("inf")
        trials.append(TrialResult(
            trial_idx=i, rotation_deg=rot_deg,
            frames=frames, energies=energies,
            reached_product=ok, peak_energy=float(peak),
            n_steps=len(frames),
        ))

    if not any(t.frames for t in trials):
        log.error("All trials failed. See meta.json for details.")
        return _write_outputs_and_exit(args, trials, t_start, neb_refined=False, rc=1)

    best = score_trials(trials)
    log.info(
        "selected trial %d (rotation_deg=%.1f, reached=%s, peak=%.4f)",
        best.trial_idx, best.rotation_deg, best.reached_product, best.peak_energy,
    )

    final_frames = best.frames
    neb_refined = False
    if args.neb_refine and len(final_frames) >= 2:
        log.info("running NEB refinement (%d images)", args.neb_images)
        product_raw = embed_mol_to_atoms(
            p_mol, calculator=calc, seed=2,
            bond_changes=bond_changes_product, rotation_perturbation=None,
        )
        rH = heavy_to_hydrogen_groups(r_h)
        pH = heavy_to_hydrogen_groups(p_h)
        product = align_product_to_reactant(
            final_frames[0], product_raw, mapping, rH, pH,
        )
        xyz_tmp = args.output / "trajectory_neb.xyz"
        run_neb(
            reactant=final_frames[0],
            product=product,
            calculator=calc,
            n_images=args.neb_images,
            output_xyz=xyz_tmp,
            fmax=0.05,
            max_steps=200,
            pad_frames=0,
        )
        # Re-read NEB frames as the new trajectory
        final_frames = read(str(xyz_tmp), index=":")
        neb_refined = True

    xyz = args.output / "trajectory.xyz"
    write(str(xyz), final_frames, format="extxyz")

    rc = _write_outputs_and_exit(args, trials, t_start, neb_refined=neb_refined, rc=0)
    if rc != 0:
        return rc

    if args.render:
        rc = _invoke_blender(args, xyz)
        if rc != 0:
            return rc

    log.info("OK: wrote %s (selected trial=%d)", xyz, best.trial_idx)
    return 0


def _angle_from_identity_deg(R: np.ndarray) -> float:
    """Return rotation angle of R in degrees (0 for identity)."""
    cos_t = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_t)))


def _write_outputs_and_exit(
    args: argparse.Namespace,
    trials: list[TrialResult],
    t_start: float,
    *,
    neb_refined: bool,
    rc: int,
) -> int:
    selected = -1
    converged = False
    if rc == 0 and trials:
        try:
            best = score_trials(trials)
            selected = best.trial_idx
            converged = best.reached_product
        except ValueError:
            pass
    meta = {
        "backend": args.backend,
        "converged": converged,
        "selected_trial": selected,
        "trials": [
            {
                "trial": t.trial_idx,
                "reached_product": t.reached_product,
                "peak_energy": float(t.peak_energy)
                    if math.isfinite(t.peak_energy) else None,
                "n_steps": t.n_steps,
                "rotation_deg": float(t.rotation_deg),
            }
            for t in trials
        ],
        "wall_clock_seconds": float(time.monotonic() - t_start),
        "neb_refined": neb_refined,
    }
    meta_clean = _sanitize_for_json(meta)
    (args.output / "meta.json").write_text(json.dumps(meta_clean, indent=2))

    if rc == 0 and trials:
        best = score_trials(trials)
        (args.output / "energies.json").write_text(json.dumps(best.energies))
    return rc


def _invoke_blender(args: argparse.Namespace, xyz: Path) -> int:
    import shutil
    import subprocess
    if shutil.which(args.blender_exe) is None:
        log.error(
            "Blender executable not found on PATH: %s. Install Blender 4.x "
            "(https://www.blender.org/download/) or pass --blender-exe "
            "/path/to/blender.",
            args.blender_exe,
        )
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
