"""Constrained relaxation that yields a trajectory of frames.

Phase 9: returns 3-tuple (frames, energies, final_constraint_state).
The third element captures AFIRConstraint latch state at relax end —
read via `atoms.constraints[0]` so it works regardless of whether ASE
copies constraint instances internally. `_snapshot()` also strips
constraints so latch state never leaks into trajectory.xyz.
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
    max_steps: int = 100,
    fmax: float = 0.1,
    traj_stride: int = 5,
) -> tuple[list[Atoms], list[float], dict]:
    atoms = atoms.copy()
    atoms.calc = calc
    if restraints:
        atoms.set_constraint(restraints)

    frames: list[Atoms] = [_snapshot(atoms)]
    energies: list[float] = [float(atoms.get_potential_energy())]

    opt = FIRE(atoms, logfile=None, dt=0.05, a=0.1, maxstep=0.1, dtmax=0.2)

    def _record():
        frames.append(_snapshot(atoms))
        energies.append(float(atoms.get_potential_energy()))

    opt.attach(_record, interval=traj_stride)
    opt.run(fmax=fmax, steps=max_steps)
    step_count = opt.nsteps

    if step_count % traj_stride != 0 or step_count == 0:
        _record()

    final_state: dict = {}
    if atoms.constraints:
        c = atoms.constraints[0]
        if hasattr(c, "formed_latched"):
            final_state = {
                "formed_latched": list(c.formed_latched),
                "broken_latched": list(c.broken_latched),
            }

    return frames, energies, final_state


def _snapshot(atoms: Atoms) -> Atoms:
    a = atoms.copy()
    a.calc = None
    a.set_constraint([])
    return a
