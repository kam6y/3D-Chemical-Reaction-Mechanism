"""CLI entry point: reactx run <rxn> -o <outdir> [options]."""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

from ase.io import read, write
from rdkit import Chem

from reactx.bond_changes import BondChanges
from reactx.config import ReactionConfig, load_config, resolve_r_form_targets
from reactx.embed3d import embed_fragments_to_positions
from reactx.neb import run_neb_top_k
from reactx.placement import (
    PlacementResult,
    valid_placements,
)
from reactx.rxn_parser import atom_map_to_reactant_idx, parse_rxn
from reactx.scoring import ScreeningTrialResult, top_k_trials
from reactx.screening import screen_all_trials

log = logging.getLogger("reactx")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reactx")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Run full pipeline on a .rxn file (with sidecar .rxn.toml)")
    run.add_argument("rxn_path", type=Path)
    run.add_argument("-o", "--output", type=Path, required=True)
    run.add_argument("--backend", choices=["uma", "lj"], default="uma")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--relax-fmax", type=float, default=0.1)
    run.add_argument("--traj-stride", type=int, default=5)
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
    try:
        cfg = load_config(args.rxn_path)
    except FileNotFoundError as exc:
        log.error("%s", exc)
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
    breakdown: dict[str, float] = {}

    r_mol, _, _ = parse_rxn(args.rxn_path)
    r_h = Chem.AddHs(r_mol)
    try:
        bond_changes = BondChanges.from_atom_map_pairs(
            formed_map=cfg.formed,
            broken_map=cfg.broken,
            atom_map_to_idx=atom_map_to_reactant_idx(r_mol),
        )
    except (KeyError, ValueError) as exc:
        log.error("Invalid bond_changes: %s", exc)
        return 2

    formed_pairs = list(bond_changes.formed)
    syms_r = [a.GetSymbol() for a in r_h.GetAtoms()]
    r_form_targets = resolve_r_form_targets(cfg, syms_r, formed_pairs)
    log.info(
        "description=%s effective: k_form=%.2f k_broken=%.2f r_broken=%.2f "
        "max_relax_steps=%d r_form_targets=%s",
        cfg.description,
        cfg.restraints.k_form, cfg.restraints.k_broken,
        cfg.restraints.r_broken, cfg.restraints.max_relax_steps,
        [f"{x:.3f}" for x in r_form_targets] if r_form_targets else "[]",
    )

    # --- placement
    t_placement = time.monotonic()
    n_frags_reactant = len(Chem.GetMolFrags(r_h))
    effective_n_candidates = cfg.sampling.n_candidates
    if n_frags_reactant == 1 and effective_n_candidates > 1:
        log.info("unimolecular reaction; n_candidates clamped %d -> 1",
                 effective_n_candidates)
        effective_n_candidates = 1
    try:
        mol_h_r, frag_indices_r, base_positions = embed_fragments_to_positions(
            r_mol, seed=args.seed,
        )
    except RuntimeError as exc:
        log.error("per-fragment embed failed: %s", exc)
        return 1
    try:
        placement = valid_placements(
            mol_h_r, frag_indices_r, base_positions, bond_changes,
            n_candidates=effective_n_candidates, seed=args.seed,
        )
    except NotImplementedError as exc:
        log.error("placement not supported: %s", exc)
        return 2
    except (RuntimeError, ValueError) as exc:
        log.error("placement failed: %s", exc)
        return 1
    log.info("placement: %d/%d survived (%d blocked)",
             len(placement.trials), placement.n_candidates, placement.n_blocked)
    breakdown["placement"] = time.monotonic() - t_placement

    # --- Stage 1: screening
    t_screen = time.monotonic()
    screen_results = screen_all_trials(
        placement, mol_h_r, bond_changes, cfg,
        backend=args.backend,
        screening_model=cfg.model.screening_model,
        workers=cfg.parallel.screening_workers,
        relax_fmax=args.relax_fmax,
        traj_stride=args.traj_stride,
        seed=args.seed,
    )
    breakdown["screening"] = time.monotonic() - t_screen

    if not any(r.frames for r in screen_results):
        log.error("All screening trials failed; see meta.json")
        return _write_outputs(
            args, cfg, screen_results, top_k_indices=[], neb_results=[],
            placement=placement, t_start=t_start, breakdown=breakdown,
            r_form_targets=r_form_targets, rc=1,
        )

    # --- top-K
    top_k_list = top_k_trials(screen_results, cfg.neb.top_k)
    top_k_indices = [r.trial_idx for r in top_k_list]
    log.info("top-%d trials by screening score: %s",
             cfg.neb.top_k, top_k_indices)

    # --- Stage 2: NEB
    t_neb = time.monotonic()
    try:
        neb_results = run_neb_top_k(
            top_k_list,
            backend=args.backend,
            neb_model=cfg.model.neb_model,
            workers=cfg.parallel.neb_workers,
            n_images=cfg.neb.n_images,
            fmax=cfg.neb.fmax,
            max_steps=cfg.neb.max_steps,
            pad_frames=cfg.neb.pad_frames,
            interp_factor=cfg.neb.interp_factor,
            output_dir=args.output,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("Stage 2 NEB failed: %s: %s", type(exc).__name__, exc)
        return _write_outputs(
            args, cfg, screen_results, top_k_indices=top_k_indices, neb_results=[],
            placement=placement, t_start=t_start, breakdown=breakdown,
            r_form_targets=r_form_targets, rc=1,
        )
    breakdown["neb"] = time.monotonic() - t_neb

    if not neb_results:
        log.error("All Stage 2 NEB jobs failed; see meta.json")
        return _write_outputs(
            args, cfg, screen_results, top_k_indices=top_k_indices, neb_results=[],
            placement=placement, t_start=t_start, breakdown=breakdown,
            r_form_targets=r_form_targets, rc=1,
        )

    final = min(neb_results, key=lambda r: r["peak_energy"])
    final_xyz_src = args.output / final["xyz_path"]
    final_frames = read(str(final_xyz_src), index=":")
    final_xyz = args.output / "trajectory.xyz"
    write(str(final_xyz), final_frames, format="extxyz")
    log.info("selected trial=%d (NEB peak=%.4f)",
             final["trial_idx"], final["peak_energy"])

    rc = _write_outputs(
        args, cfg, screen_results, top_k_indices=top_k_indices,
        neb_results=neb_results,
        placement=placement, t_start=t_start, breakdown=breakdown,
        r_form_targets=r_form_targets, rc=0, selected_trial=final["trial_idx"],
    )
    if rc != 0:
        return rc

    if args.render:
        rc_render = _invoke_blender(args, final_xyz)
        if rc_render != 0:
            return rc_render
    log.info("OK: wrote %s", final_xyz)
    return 0


def _write_outputs(
    args, cfg: ReactionConfig,
    screen_results: list[ScreeningTrialResult],
    *,
    top_k_indices: list[int],
    neb_results: list[dict],
    placement: PlacementResult | None,
    t_start: float,
    breakdown: dict[str, float],
    r_form_targets: list[float],
    rc: int,
    selected_trial: int = -1,
) -> int:
    placement_meta = (
        {"n_candidates": placement.n_candidates,
         "n_blocked": placement.n_blocked,
         "n_valid": len(placement.trials)}
        if placement is not None
        else {"n_candidates": 0, "n_blocked": 0, "n_valid": 0}
    )
    meta = {
        "backend": args.backend,
        "description": cfg.description,
        "selected_trial": selected_trial,
        "placement": placement_meta,
        "screening_trials": [
            {
                "trial_idx": r.trial_idx,
                "reached_product": r.reached_product,
                "peak_energy": float(r.peak_energy)
                    if math.isfinite(r.peak_energy) else None,
                "n_steps": r.n_steps,
                "direction": [float(x) for x in r.direction],
                "error": r.error,
            }
            for r in screen_results
        ],
        "top_k_indices": list(top_k_indices),
        "neb_results": list(neb_results),
        "wall_clock_seconds": float(time.monotonic() - t_start),
        "wall_clock_breakdown": {k: float(v) for k, v in breakdown.items()},
        "effective_params": {
            "k_form": cfg.restraints.k_form,
            "k_broken": cfg.restraints.k_broken,
            "r_broken": cfg.restraints.r_broken,
            "max_relax_steps": cfg.restraints.max_relax_steps,
            "r_form_targets": list(r_form_targets) if r_form_targets else [],
            "n_candidates": cfg.sampling.n_candidates,
            "screening_model": cfg.model.screening_model,
            "neb_model": cfg.model.neb_model,
            "top_k": cfg.neb.top_k,
            "n_images": cfg.neb.n_images,
            "screening_workers": cfg.parallel.screening_workers,
            "neb_workers": cfg.parallel.neb_workers,
        },
    }
    (args.output / "meta.json").write_text(
        json.dumps(_sanitize_for_json(meta), indent=2)
    )
    if neb_results:
        best = min(neb_results, key=lambda r: r["peak_energy"])
        (args.output / "energies.json").write_text(json.dumps(best["image_energies"]))
    return rc


def _invoke_blender(args, xyz: Path) -> int:
    import shutil
    import subprocess
    if shutil.which(args.blender_exe) is None:
        log.error("Blender executable not found on PATH: %s", args.blender_exe)
        return 1
    script = Path(__file__).resolve().parent.parent / "blender" / "render.py"
    blend = args.output / "scene.blend"
    cmd = [args.blender_exe, "--background", "--python", str(script),
           "--", str(xyz), str(blend)]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error("blender exited with code %d", result.returncode)
        return 1
    return 0
