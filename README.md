# reactx — Generic Pure CI-NEB Reaction Path Engine

2D reaction mechanisms (`.rxn`) plus sidecar TOML config are converted into
direct R/P 3D endpoints, independently relaxed to local minima with UMA, then
connected with CI-NEB (IDPP interpolation + two-phase climb). The resulting
trajectory can be rendered in Blender as a ball-and-stick animation.

The goal is plausible reaction animation, not quantitatively accurate transition
state energetics. Supported example reactions:

- SN2 (`O- + CH3Cl`)
- Proton transfer (`HCl + NH3`)
- Menshutkin (`NH3 + CH3Cl`)
- E2 elimination
- SN1 dissociation
- SN1 recombination
- Diels-Alder cycloaddition

## Phase 11 changes

Phase 11 replaces the previous multi-trial force-search path engine with a Pure
CI-NEB pipeline built directly from the `.rxn` reactant and product endpoints:

- `BondChanges.from_reaction_diff(r_mol, p_mol)` infers formed/broken bonds
  from atom-map connectivity differences. TOML no longer contains `formed` or
  `broken`.
- `reactx/placement.py` is reduced to deterministic `simple_placement`.
  Bimolecular endpoints place reaction anchors along `+z` at
  `initial_separation=4.0`.
- `reactx/endpoint_relax.py` relaxes R and P endpoints independently with
  FIRE or BFGS before NEB.
- TOML schema is now `[placement]`, `[endpoint_relax]`, and `[neb]`.
  `description` is the only required key.
- CLI always runs NEB. Legacy sampling, relaxation, and optional refinement
  flags have been removed.
- `meta.json` now contains `endpoint_relax_r`, `endpoint_relax_p`, `neb`, and
  `effective_params`.

Detailed spec: `docs/superpowers/specs/2026-05-15-phase-11-ci-neb-design.md`

## Setup

```bash
python -m pip install -e .[dev]
hf auth login
```

Install Blender 4.x separately if you want `--render`.

## Usage

```bash
reactx run examples/sn2.rxn -o out/sn2/ --backend uma --render
reactx run examples/proton_transfer.rxn -o out/pt/ --backend uma --render
reactx run examples/diels_alder_simple.rxn -o out/da/ --backend uma --render
reactx run examples/diels_alder_endo.rxn -o out/da_endo/ --backend uma --render
```

Main flags:

- `--backend {uma,lj}` (default: `uma`)
- `--model uma-m-1p1`
- `--render` plus `--blender-exe blender`

Outputs:

- `trajectory.xyz` — NEB image sequence
- `energies.json` — NEB image energies
- `meta.json` — endpoint relax info, NEB info, effective parameters
- `scene.blend` — Blender scene when `--render` is used

## Per-reaction `.rxn.toml` config

Each `examples/<name>.rxn` requires `<name>.rxn.toml` beside it. Only
`description` is required.

Minimal example:

```toml
description = "SN2 anion: CH3Cl + OH- -> CH3OH + Cl-"
```

Full schema:

```toml
description = "..."

[placement]
initial_separation = 4.0
orientation = "default"  # "default" | "endo" | "exo"

[endpoint_relax]
fmax = 0.01
max_steps = 500
optimizer = "FIRE"       # "FIRE" | "BFGS"

[neb]
n_images = 11
fmax = 0.05
max_steps = 200
k = 1.0
climb = true
pad_frames = 0
```

| `.rxn` | description | notes |
|---|---|---|
| `sn2.rxn` | SN2 anion | all defaults |
| `proton_transfer.rxn` | Proton transfer | all defaults |
| `menshutkin.rxn` | Menshutkin | all defaults |
| `e2.rxn` | E2 elimination | all defaults |
| `sn1_dissoc.rxn` | SN1 step 1 dissociation | `initial_separation = 6.0` |
| `sn1_recomb.rxn` | SN1 step 2 recombination | all defaults |
| `diels_alder_simple.rxn` | Diels-Alder simple | `orientation = "default"` |
| `diels_alder_endo.rxn` | Diels-Alder endo | `orientation = "endo"` |

## Architecture

```text
.rxn + .rxn.toml -> parse_rxn + load_config -> ReactionConfig + atom mapping
                                      |
                                      v
                    BondChanges.from_reaction_diff(r_mol, p_mol)
                                      |
              +-----------------------+-----------------------+
              v                                               v
 embed_fragments_to_positions(r_mol)            embed_fragments_to_positions(p_mol)
              |                                               |
              v                                               v
 simple_placement(side="reactant")              simple_placement(side="product")
              |                                               |
              v                                               v
 relax_endpoint(atoms_r, UMA)                   relax_endpoint(atoms_p, UMA)
              +-----------------------+-----------------------+
                                      v
                      align_product_to_reactant
                                      |
                                      v
                      run_neb (IDPP + two-phase CI-NEB)
                                      |
                                      v
                      trajectory.xyz -> blender/render.py -> .blend
```

Key files:

- `reactx/bond_changes.py` — R/P bond-set diff
- `reactx/placement.py` — deterministic endpoint placement
- `reactx/endpoint_relax.py` — FIRE/BFGS endpoint relaxation
- `reactx/align.py` — atom-map and H permutation alignment
- `reactx/neb.py` — IDPP + CI-NEB
- `reactx/config.py` — Phase 11 TOML schema
- `reactx/cli.py` — Pure CI-NEB pipeline

## Rendering

`blender/render.py` imports XYZ frames and adds ball-and-stick rendering. Atom
ball sizes use Alvarez van der Waals radii scaled by `REACTX_VDW_SCALE`
(default `0.25`). Bonds are detected per frame from Cordero covalent radii and
animated with visibility keyframes, so forming and breaking bonds appear and
disappear during the trajectory.

## Limits

- The trajectory is a CI-NEB approximation for animation; quantitative TS
  energies are not guaranteed.
- Endpoint relaxation may return `converged=false`; the partially relaxed
  endpoint is still passed to NEB and recorded in `meta.json`.
- IDPP can create poor initial paths for difficult atom-rearrangement cases.
- Diels-Alder orientation is discrete (`default`, `endo`, `exo`); continuous
  face rotation is not sampled.
- Radical, open-shell, solvent, and multi-step mechanisms are out of scope.

## Wall-clock (planned)

Phase 11 removes the multi-candidate loop and instead runs two endpoint
relaxations plus one NEB. The 8 example reactions need to be remeasured with
UMA-m-1p1 on the target GPU before publishing final timings.

## Tests

```bash
pytest
pytest -m slow
pytest -m blender
```

`pytest` excludes `slow` and `blender` by default via `pyproject.toml`.
