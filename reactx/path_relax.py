"""Constrained relaxation that yields a trajectory of frames.

Used by Phase Re1 to convert (initial geometry + bond-change restraints) into
a reaction path: FIRE relaxation under Hookean (attractive) + PullApart
(repulsive) constraints drives the system from reactant toward product.
Frames are snapshotted every `traj_stride` optimizer steps; the final frame
is always included.

Use this with a shared calculator (e.g. UMA) to avoid model reloads between
trials.
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
) -> tuple[list[Atoms], list[float]]:
    """Run FIRE under the given restraints; return (frames, energies).

    The first frame is the initial state; thereafter snapshots are taken every
    `traj_stride` steps, plus the final state. `frames[i].calc` is None to
    avoid keeping references to the shared calculator in trajectory output.
    """
    atoms = atoms.copy()
    atoms.calc = calc
    if restraints:
        atoms.set_constraint(restraints)

    frames: list[Atoms] = [_snapshot(atoms)]
    energies: list[float] = [float(atoms.get_potential_energy())]

    # `maxstep=0.1` Å caps per-step displacement to suppress overshoot when
    # Hookean restraints pull strongly across long distances; `dtmax=0.2` fs
    # prevents FIRE from accelerating the integration timestep, which was the
    # source of spring-like bouncing observed in early SN2 trajectories.
    opt = FIRE(atoms, logfile=None, dt=0.05, a=0.1, maxstep=0.1, dtmax=0.2)

    def _record():
        frames.append(_snapshot(atoms))
        energies.append(float(atoms.get_potential_energy()))

    opt.attach(_record, interval=traj_stride)
    opt.run(fmax=fmax, steps=max_steps)
    step_count = opt.nsteps

    if step_count % traj_stride != 0 or step_count == 0:
        _record()

    return frames, energies


def _snapshot(atoms: Atoms) -> Atoms:
    """Detach calculator so frame is independent and serializable."""
    a = atoms.copy()
    a.calc = None
    return a
