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
from reactx.artificial_force import build_afir_constraint
from reactx.bond_changes import BondChanges
from reactx.calculators import make_calculator
from reactx.config import (
    ConfigError,
)
from reactx.config import (
    ReactionConfigV9 as ReactionConfig,
)
from reactx.config import (
    load_config_v9 as load_config,
)
from reactx.config import (
    resolve_alpha_broken_v9 as resolve_alpha_broken,
)
from reactx.config import (
    resolve_alpha_formed_v9 as resolve_alpha_formed,
)
from reactx.embed3d import embed_fragments_to_positions
from reactx.neb import run_neb
from reactx.path_relax import relax_with_restraints
from reactx.placement import (
    PlacementResult,
    build_atoms_from_positions,
    valid_placements,
)
from reactx.rxn_parser import atom_map_to_reactant_idx, heavy_to_hydrogen_groups, parse_rxn
from reactx.scoring import (
    TrialResultV9 as TrialResult,
)
from reactx.scoring import (
    count_initial_latched,
    product_distance_residual,
    resolve_broken_thresholds,
    resolve_formed_thresholds,
)
from reactx.scoring import (
    reached_product_v9 as reached_product,
)
from reactx.scoring import (
    score_trials_v9 as score_trials,
)

log = logging.getLogger("reactx")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reactx")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run full pipeline on a .rxn file (with sidecar .rxn.toml)")
    run.add_argument("rxn_path", type=Path)
    run.add_argument("-o", "--output", type=Path, required=True)

    run.add_argument("--backend", choices=["uma", "lj"], default="uma")
    run.add_argument("--model", type=str, default="uma-m-1p1",
                     help="UMA model name (uma-m-1p1, uma-s-1p2, ...)")

    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--relax-fmax", type=float, default=0.1)
    run.add_argument("--traj-stride", type=int, default=5)

    run.add_argument("--neb-refine", action="store_true",
                     help="Refine the best trial trajectory with a short NEB "
                          "(only supported for 1 formed + 1 broken reactions)")
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


def _sanitize_for_json(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _scalar_or_list(value):
    if isinstance(value, tuple):
        return list(value)
    return value


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
    except ConfigError as exc:
        log.error("config error: %s", exc)
        return 2
    except FileNotFoundError as exc:
        log.error("%s", exc)
        log.error(
            "A sidecar TOML config is REQUIRED. Create %s.toml. "
            "See examples/sn2.rxn.toml for the schema.",
            args.rxn_path,
        )
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
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    try:
        bond_changes = BondChanges.from_atom_map_pairs(
            formed_map=cfg.formed,
            broken_map=cfg.broken,
            atom_map_to_idx=atom_map_to_reactant_idx(r_mol),
        )
    except (KeyError, ValueError) as exc:
        log.error("Invalid bond_changes from %s.toml: %s", args.rxn_path, exc)
        return 2

    if args.neb_refine and (
        len(bond_changes.formed) != 1 or len(bond_changes.broken) != 1
    ):
        log.error(
            "--neb-refine is only supported for 1 formed + 1 broken bond "
            "reactions in Phase 3 (got formed=%d, broken=%d). Multi-bond NEB "
            "endpoint construction is Phase 4+. Re-run without --neb-refine.",
            len(bond_changes.formed), len(bond_changes.broken),
        )
        return 2

    # bond_changes is reactant-space; for product embedding (neb-refine path)
    # we feed embed3d a product-space BondChanges constructed by
    # swapping roles + remapping indices. embed3d identifies the substrate as
    # the fragment containing both atoms of `broken`, so for product we set
    # `broken` = the reactant-formed bond (e.g. C-F in CH3F), which keeps the
    # substrate fragment correctly grouped. `formed` then defines where the
    # nucleophile-side fragment (Cl- in product) is placed via backside.
    # NEB refine は 1+1 反応のみ対応; それ以外の場合は bond_changes_product=None
    # として、neb-refine が要求された時点で embed3d 側でエラーになる。
    if len(bond_changes.formed) == 1 and len(bond_changes.broken) == 1:
        a_form_r, b_form_r = bond_changes.formed[0]
        a_brk_r, b_brk_r = bond_changes.broken[0]
        bond_changes_product = BondChanges(
            formed=((heavy_mapping[a_brk_r], heavy_mapping[b_brk_r]),),
            broken=((heavy_mapping[a_form_r], heavy_mapping[b_form_r]),),
        )
    else:
        bond_changes_product = None

    formed_pairs = list(bond_changes.formed)
    broken_pairs = list(bond_changes.broken)

    log.info(
        "description=%s effective: alpha_formed=%s alpha_broken=%s "
        "max_relax_steps=%d",
        cfg.description,
        cfg.afir.alpha_formed, cfg.afir.alpha_broken,
        cfg.afir.max_relax_steps,
    )

    model_kwargs = {"model_name": args.model} if args.backend == "uma" else {}
    calc = make_calculator(args.backend, **model_kwargs)

    n_frags_reactant = len(Chem.GetMolFrags(r_h))
    effective_n_candidates = cfg.sampling.n_candidates
    if n_frags_reactant == 1 and effective_n_candidates > 1:
        log.info(
            "unimolecular reaction (1 reactant fragment); "
            "n_candidates forced from %d to 1",
            effective_n_candidates,
        )
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

    log.info(
        "placement: %d/%d candidates survived blocking (%d blocked)",
        len(placement.trials), placement.n_candidates, placement.n_blocked,
    )

    af = resolve_alpha_formed(cfg)
    ab = resolve_alpha_broken(cfg)

    trials: list[TrialResult] = []
    for i, t in enumerate(placement.trials):
        atoms_init = build_atoms_from_positions(mol_h_r, t.positions)
        log.info("trial %d (direction=%s)",
                 i, np.array2string(t.direction, precision=3))

        ft = resolve_formed_thresholds(
            atoms_init, formed_pairs, cfg.scoring.r_formed_threshold,
        )
        bt = resolve_broken_thresholds(
            atoms_init, broken_pairs, cfg.scoring.r_broken_threshold,
        )

        initial_latched = count_initial_latched(
            atoms_init, formed_pairs, broken_pairs, ft, bt,
        )

        afir_cs = build_afir_constraint(
            atoms_init, formed_pairs, broken_pairs,
            alpha_formed=af, alpha_broken=ab,
            formed_thresholds=ft, broken_thresholds=bt,
        )

        try:
            frames, energies, final_state = relax_with_restraints(
                atoms_init, afir_cs, calc,
                max_steps=cfg.afir.max_relax_steps,
                fmax=args.relax_fmax,
                traj_stride=args.traj_stride,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("trial %d relax failed: %s", i, exc)
            trials.append(TrialResult(
                trial_idx=i, direction=t.direction, frames=[], energies=[],
                reached_product=False, peak_energy=float("inf"), n_steps=0,
                formed_thresholds=ft, broken_thresholds=bt,
                product_distance_residual=float("inf"),
                formed_latch_count=0, broken_latch_count=0,
                initial_latched_formed=initial_latched["formed"],
                initial_latched_broken=initial_latched["broken"],
            ))
            continue

        ok = reached_product(frames[-1], formed_pairs, broken_pairs, ft, bt)
        residual = product_distance_residual(
            frames[-1], formed_pairs, broken_pairs, ft, bt,
        )
        peak = max(energies) if energies else float("inf")
        formed_latch_count = sum(final_state.get("formed_latched", []))
        broken_latch_count = sum(final_state.get("broken_latched", []))

        trials.append(TrialResult(
            trial_idx=i, direction=t.direction,
            frames=frames, energies=energies,
            reached_product=ok, peak_energy=float(peak),
            n_steps=len(frames),
            formed_thresholds=ft, broken_thresholds=bt,
            product_distance_residual=residual,
            formed_latch_count=formed_latch_count,
            broken_latch_count=broken_latch_count,
            initial_latched_formed=initial_latched["formed"],
            initial_latched_broken=initial_latched["broken"],
        ))

    if not any(t.frames for t in trials):
        log.error("All trials failed. See meta.json for details.")
        return _write_outputs_and_exit(
            args, trials, t_start, neb_refined=False, rc=1,
            cfg=cfg, placement=placement,
        )

    best = score_trials(trials)
    log.info(
        "selected trial %d (direction=%s, reached=%s, peak=%.4f)",
        best.trial_idx, np.array2string(best.direction, precision=3),
        best.reached_product, best.peak_energy,
    )

    final_frames = best.frames
    neb_refined = False
    if args.neb_refine and len(final_frames) >= 2:
        log.info("running NEB refinement (%d images)", args.neb_images)
        from ase.optimize import BFGS
        try:
            mol_h_p, frag_indices_p, p_positions = embed_fragments_to_positions(
                p_mol, seed=2,
            )
        except RuntimeError as exc:
            log.error("NEB product per-fragment embed failed: %s", exc)
            return 1
        try:
            placement_p = valid_placements(
                mol_h_p, frag_indices_p, p_positions, bond_changes_product,
                n_candidates=8, seed=2,
            )
        except NotImplementedError as exc:
            log.error("NEB product placement not supported: %s", exc)
            return 2
        except (RuntimeError, ValueError) as exc:
            log.error("NEB product placement failed: %s", exc)
            return 1
        product_raw = build_atoms_from_positions(
            mol_h_p, placement_p.trials[0].positions,
        )
        product_raw.calc = calc
        BFGS(product_raw, logfile=None).run(fmax=0.01, steps=300)
        rH = heavy_to_hydrogen_groups(r_h)
        pH = heavy_to_hydrogen_groups(p_h)
        product = align_product_to_reactant(
            final_frames[0], product_raw, heavy_mapping, rH, pH,
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

    rc = _write_outputs_and_exit(
        args, trials, t_start, neb_refined=neb_refined, rc=0,
        cfg=cfg, placement=placement,
    )
    if rc != 0:
        return rc

    if args.render:
        rc = _invoke_blender(args, xyz)
        if rc != 0:
            return rc

    log.info("OK: wrote %s (selected trial=%d)", xyz, best.trial_idx)
    return 0


def _write_outputs_and_exit(
    args: argparse.Namespace,
    trials: list[TrialResult],
    t_start: float,
    *,
    neb_refined: bool,
    rc: int,
    cfg: ReactionConfig | None = None,
    placement: PlacementResult | None = None,
) -> int:
    best: TrialResult | None = None
    if rc == 0 and trials:
        try:
            best = score_trials(trials)
        except ValueError:
            best = None
    selected = best.trial_idx if best is not None else -1
    converged = best.reached_product if best is not None else False
    placement_meta = (
        {
            "n_candidates": placement.n_candidates,
            "n_blocked": placement.n_blocked,
            "n_valid": len(placement.trials),
            "placement_kind": placement.placement_kind,
        }
        if placement is not None
        else {"n_candidates": 0, "n_blocked": 0, "n_valid": 0, "placement_kind": None}
    )
    meta: dict = {
        "backend": args.backend,
        "description": cfg.description if cfg is not None else None,
        "converged": converged,
        "selected_trial": selected,
        "placement": placement_meta,
        "trials": [
            {
                "trial": t.trial_idx,
                "reached_product": t.reached_product,
                "peak_energy": float(t.peak_energy)
                    if math.isfinite(t.peak_energy) else None,
                "n_steps": t.n_steps,
                "direction": [float(x) for x in t.direction],
                "orientation": (
                    placement.trials[t.trial_idx].orientation
                    if placement is not None and t.trial_idx < len(placement.trials)
                    else "single"
                ),
                "formed_thresholds": [float(x) for x in t.formed_thresholds],
                "broken_thresholds": [float(x) for x in t.broken_thresholds],
                "product_distance_residual": (
                    float(t.product_distance_residual)
                    if math.isfinite(t.product_distance_residual) else None
                ),
                "formed_latch_count": t.formed_latch_count,
                "broken_latch_count": t.broken_latch_count,
                "initial_latched_formed": t.initial_latched_formed,
                "initial_latched_broken": t.initial_latched_broken,
            }
            for t in trials
        ],
        "wall_clock_seconds": float(time.monotonic() - t_start),
        "neb_refined": neb_refined,
        "effective_params": (
            {
                "alpha_formed": _scalar_or_list(cfg.afir.alpha_formed),
                "alpha_broken": _scalar_or_list(cfg.afir.alpha_broken),
                "max_relax_steps": cfg.afir.max_relax_steps,
                "r_broken_threshold": (
                    _scalar_or_list(cfg.scoring.r_broken_threshold)
                    if cfg.scoring.r_broken_threshold is not None else None
                ),
                "r_formed_threshold": (
                    _scalar_or_list(cfg.scoring.r_formed_threshold)
                    if cfg.scoring.r_formed_threshold is not None else None
                ),
                "n_candidates": cfg.sampling.n_candidates,
            }
            if cfg is not None else None
        ),
    }
    meta_clean = _sanitize_for_json(meta)
    (args.output / "meta.json").write_text(json.dumps(meta_clean, indent=2))

    if best is not None:
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
