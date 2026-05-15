"""Constrained relaxation that yields a trajectory of frames.

Phase 9: returns 3-tuple (frames, energies, final_constraint_state).
Phase 10: optional Stage A (unbiased FIRE pre-relax) before Stage B
(AFIR-constrained relax). When `pre_relax_steps > 0`, FIRE first runs
without any constraints; the geometry at Stage A end is recorded in
`final_state["frame_after_pre_relax"]` so the caller can recompute
initial-latch state against it. Stage A frames are concatenated with
Stage B frames; the boundary geometry may appear twice (acceptable —
visualisation value over deduplication).
"""
from __future__ import annotations

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import FIRE


def relax_with_restraints(
    atoms: Atoms,
    restraints: list,
    calc: Calculator,
    *,
    pre_relax_steps: int = 0,
    max_steps: int = 100,
    fmax: float = 0.1,
    traj_stride: int = 5,
) -> tuple[list[Atoms], list[float], dict]:
    atoms = atoms.copy()
    atoms.calc = calc

    frames: list[Atoms] = []
    energies: list[float] = []
    final_state: dict = {}

    if pre_relax_steps > 0:
        atoms.set_constraint([])
        frames.append(_snapshot(atoms))
        energies.append(float(atoms.get_potential_energy()))

        opt_a = FIRE(atoms, logfile=None, dt=0.05, a=0.1, maxstep=0.1, dtmax=0.2)

        def _record_a():
            frames.append(_snapshot(atoms))
            energies.append(float(atoms.get_potential_energy()))

        opt_a.attach(_record_a, interval=traj_stride)
        opt_a.run(fmax=fmax, steps=pre_relax_steps)
        if opt_a.nsteps % traj_stride != 0 or opt_a.nsteps == 0:
            _record_a()

        final_state["frame_after_pre_relax"] = _snapshot(atoms)

    if restraints:
        atoms.set_constraint(restraints)
    else:
        atoms.set_constraint([])

    frames.append(_snapshot(atoms))
    energies.append(float(atoms.get_potential_energy()))

    opt_b = FIRE(atoms, logfile=None, dt=0.05, a=0.1, maxstep=0.1, dtmax=0.2)

    def _record_b():
        frames.append(_snapshot(atoms))
        energies.append(float(atoms.get_potential_energy()))

    opt_b.attach(_record_b, interval=traj_stride)
    opt_b.run(fmax=fmax, steps=max_steps)
    if opt_b.nsteps % traj_stride != 0 or opt_b.nsteps == 0:
        _record_b()

    if atoms.constraints:
        c = atoms.constraints[0]
        if hasattr(c, "formed_latched"):
            final_state["formed_latched"] = list(c.formed_latched)
            final_state["broken_latched"] = list(c.broken_latched)

    return frames, energies, final_state


def _snapshot(atoms: Atoms) -> Atoms:
    a = atoms.copy()
    a.calc = None
    a.set_constraint([])
    return a
