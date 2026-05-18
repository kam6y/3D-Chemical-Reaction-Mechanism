# reactx - Generic Pure CI-NEB Reaction Path Engine

`reactx` converts mapped 2D reaction mechanisms into CI-NEB reaction
animations. A `.rxn` file supplies atom mapping and reactant/product
connectivity, the sidecar `.rxn.toml` supplies run configuration, and explicit
XYZ endpoint files supply the complete reactant and product coordinates. The
endpoints are relaxed independently and then connected with CI-NEB using IDPP
interpolation and a two-phase climb.

The goal is plausible reaction animation, not quantitatively accurate transition
state energetics. Supported example reactions:

- SN2 (`O- + CH3Cl`)
- Proton transfer (`HCl + NH3`)
- Menshutkin (`NH3 + CH3Cl`)
- E2 elimination
- SN1 dissociation
- SN1 recombination
- Diels-Alder cycloaddition
- Diels-Alder endo cycloaddition

## Explicit endpoint inputs

The normal CLI path is driven by explicit endpoint structures:

- The `.rxn` file provides atom mapping plus reactant and product connectivity.
- The sidecar `.rxn.toml` requires `reactant_structure` and
  `product_structure`.
- The referenced XYZ files provide complete NEB endpoint coordinates, with atom
  order matching the mapped `.rxn`.
- The CLI reads those endpoint files directly, relaxes them, and runs one NEB.
  It does not generate or select alternative 3D endpoint coordinates.
- `meta.json` records `endpoint_source`, `bond_changes`, reactant and product
  endpoint relaxation summaries, NEB summaries, and `effective_params`.

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

- `trajectory.xyz` - NEB image sequence
- `energies.json` - NEB optimization image energies. When
  `guide_bond_changes = true`, these include the bond-distance guide bias.
- `unbiased_energies.json` - per-image energies re-evaluated with the original
  calculator only; use these for energy display and comparison.
- `meta.json` - endpoint source, bond changes, relax info, NEB info, effective
  parameters
- `scene.blend` - Blender scene when `--render` is used

## Per-reaction `.rxn.toml` config

Each `examples/<name>.rxn` requires `<name>.rxn.toml` beside it. The config
must name the reactant and product XYZ endpoint files.

Minimal example:

```toml
description = "SN2 anion: CH3Cl + OH- -> CH3OH + Cl-"
reactant_structure = "sn2.reactant.xyz"
product_structure = "sn2.product.xyz"
```

User-facing schema:

```toml
description = "..."
reactant_structure = "name.reactant.xyz"
product_structure = "name.product.xyz"

[endpoint_relax]
fmax = 0.01
max_steps = 500
optimizer = "FIRE"       # "FIRE" | "BFGS"

[neb]
n_images = 11
fmax = 0.05
max_steps = 200
k = 1.0
method = "eb"               # "eb" | "improvedtangent" | "aseneb" | "spline" | "string"
remove_rotation_and_translation = true
climb = true
pad_frames = 0
guide_bond_changes = true
guide_k = 0.25              # conservative harmonic bond-distance guide strength
```

`guide_bond_changes` applies only to internal NEB images. It adds a harmonic
guide for each inferred formed and broken bond, with each target distance
linearly interpolated from the relaxed reactant endpoint distance to the aligned
product endpoint distance. This intentionally biases optimization to improve
reaction-coordinate frame distribution. `meta.json` records
`biased_optimization`, `guide_bond_changes`, `guide_k`, `guided_bonds`,
`image_energies`, and `unbiased_image_energies`; prefer
`unbiased_image_energies` or `unbiased_energies.json` for plots and comparison.

| `.rxn` | reactant endpoint | product endpoint | description |
|---|---|---|---|
| `sn2.rxn` | `sn2.reactant.xyz` | `sn2.product.xyz` | SN2 anion |
| `proton_transfer.rxn` | `proton_transfer.reactant.xyz` | `proton_transfer.product.xyz` | Proton transfer |
| `menshutkin.rxn` | `menshutkin.reactant.xyz` | `menshutkin.product.xyz` | Menshutkin |
| `e2.rxn` | `e2.reactant.xyz` | `e2.product.xyz` | E2 elimination |
| `sn1_dissoc.rxn` | `sn1_dissoc.reactant.xyz` | `sn1_dissoc.product.xyz` | SN1 step 1 dissociation |
| `sn1_recomb.rxn` | `sn1_recomb.reactant.xyz` | `sn1_recomb.product.xyz` | SN1 step 2 recombination |
| `diels_alder_simple.rxn` | `diels_alder_simple.reactant.xyz` | `diels_alder_simple.product.xyz` | Diels-Alder simple |
| `diels_alder_endo.rxn` | `diels_alder_endo.reactant.xyz` | `diels_alder_endo.product.xyz` | Diels-Alder endo |

## Architecture

```text
.rxn + .rxn.toml -> parse_rxn + load_config -> ReactionConfig + atom mapping
                                      |
                                      v
                    BondChanges.from_reaction_diff(r_mol, p_mol)
                                      |
                                      v
                 load_endpoint_pair from explicit XYZ endpoint files
                                      |
              +-----------------------+-----------------------+
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

- `reactx/endpoints.py` - explicit XYZ endpoint loading and validation
- `reactx/bond_changes.py` - R/P bond-set diff
- `reactx/endpoint_relax.py` - FIRE/BFGS endpoint relaxation
- `reactx/align.py` - atom-map and H permutation alignment
- `reactx/neb.py` - IDPP + EB/CI-NEB with rotation/translation removal
- `reactx/neb_guide.py` - generic formed/broken bond-distance guide
- `reactx/config.py` - TOML schema
- `reactx/cli.py` - explicit-endpoint CI-NEB pipeline

## Rendering

`blender/render.py` imports XYZ frames and adds ball-and-stick rendering. Atom
ball sizes use Alvarez van der Waals radii scaled by `REACTX_VDW_SCALE`
(default `0.25`). Bonds are detected per frame from Cordero covalent radii and
animated with visibility keyframes, so forming and breaking bonds appear and
disappear during the trajectory.

## Limits

- The trajectory is a CI-NEB approximation for animation; quantitative TS
  energies are not guaranteed.
- Guided NEB optimization biases `image_energies`; use
  `unbiased_image_energies` for calculator-only energy display and comparison.
- Endpoint relaxation may return `converged=false`; the partially relaxed
  endpoint is still passed to NEB and recorded in `meta.json`.
- NEB can still converge to poor lateral paths for difficult rearrangements;
  the default elastic-band method and rotation/translation removal reduce this
  but do not replace chemical validation of the final trajectory.
- Radical, open-shell, solvent, and multi-step mechanisms are out of scope.

## Wall-clock

The examples run one endpoint relaxation per side followed by one NEB. The 8
example reactions need to be measured with UMA-m-1p1 on the target GPU before
publishing final timings.

## Tests

```bash
pytest
pytest -m slow
pytest -m blender
```

`pytest` excludes `slow` and `blender` by default via `pyproject.toml`.
