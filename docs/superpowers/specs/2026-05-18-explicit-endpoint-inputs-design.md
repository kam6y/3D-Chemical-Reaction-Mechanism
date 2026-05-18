# reactx Issue #15 - Explicit 3D endpoint inputs

- Status: Approved
- Date: 2026-05-18
- Owner: @kam6y
- Issue: https://github.com/kam6y/3D-Chemical-Reaction-Mechanism/issues/15
- Backward compatibility: Not required. Existing placement-driven config is removed from the default path and rejected by schema validation.

## 1. Goal

NEB endpoint structures are core inputs, not values that should be inferred by generic fragment placement heuristics. Issue #15 replaces the current Fibonacci-sphere placement path with explicit reactant/product 3D endpoint files.

The new default pipeline must:

1. Let users provide complete reactant and product endpoint coordinates.
2. Keep `.rxn` as the source of atom mapping and reactant/product connectivity.
3. Use explicit endpoint coordinates directly for endpoint relaxation and CI-NEB.
4. Remove the placement engine from the production path and delete its unused code.
5. Record endpoint provenance in `meta.json`.

## 2. Non-goals

- No legacy fallback to automatic Fibonacci placement.
- No SDF/RXN-like custom container format in this phase.
- No ASE-readable structure pair plus explicit TOML mapping table in this phase.
- No reaction-type-specific stereochemical scoring.
- No quantitative transition-state energy guarantee.

## 3. Input format

Each `.rxn` continues to have a required sidecar TOML file. The sidecar now requires explicit endpoint paths:

```toml
description = "SN2 anion: CH3Cl + OH- -> CH3OH + Cl-"
reactant_structure = "sn2.reactant.xyz"
product_structure = "sn2.product.xyz"

[endpoint_relax]
fmax = 0.01
max_steps = 500
optimizer = "FIRE"

[neb]
n_images = 11
fmax = 0.05
max_steps = 200
k = 1.0
climb = true
pad_frames = 0
```

Only `description`, `reactant_structure`, and `product_structure` are required. `[endpoint_relax]` and `[neb]` are optional and keep their Phase 11 defaults.

Paths are resolved relative to the sidecar TOML directory unless absolute.

The following are invalid in the new schema:

- `[placement]`
- `orientation`
- `n_candidates`
- `relaxed_candidates`
- old AFIR/scoring/sampling/restraint keys
- top-level `formed` / `broken`

These keys raise `ConfigError` with a message that points to the explicit endpoint schema.

## 4. Architecture

The CLI pipeline becomes:

```text
.rxn + .rxn.toml
        |
        v
parse_rxn -> r_mol, p_mol, heavy_mapping
        |
        v
BondChanges.from_reaction_diff(r_mol, p_mol)
        |
        v
load explicit R/P endpoint structures
        |
        v
validate atom count and element order against Chem.AddHs(r_mol/p_mol)
        |
        v
relax_endpoint(R), relax_endpoint(P)
        |
        v
align_product_to_reactant
        |
        v
run_neb
        |
        v
trajectory.xyz, energies.json, meta.json
```

`embed_fragments_to_positions()`, `valid_placements()`, and endpoint candidate selection are removed from the CLI path.

## 5. Components

### 5.1 `reactx/config.py`

Update the schema dataclasses:

- Add `reactant_structure: Path | str`.
- Add `product_structure: Path | str`.
- Remove `PlacementSection`.
- Remove `placement` from `ReactionConfig`.
- Permit only top-level `description`, `reactant_structure`, `product_structure`, `endpoint_relax`, and `neb`.

Validation requirements:

- `description` must be a non-empty string.
- endpoint path values must be non-empty strings.
- `[endpoint_relax]` and `[neb]` retain current validation rules.
- obsolete keys produce `ConfigError`.

### 5.2 New `reactx/endpoints.py`

Create a focused endpoint loader module.

Endpoint loader API:

```python
def load_endpoint_pair(
    cfg: ReactionConfig,
    *,
    rxn_path: Path,
    reactant_mol_h: Chem.Mol,
    product_mol_h: Chem.Mol,
) -> tuple[Atoms, Atoms, dict]:
    """Load, validate, and annotate explicit endpoint structures."""
```

Responsibilities:

- Resolve relative endpoint paths from the sidecar TOML directory.
- Read structures with ASE.
- Reject multiple-frame endpoint files for this phase.
- Validate atom count against `Chem.AddHs(r_mol)` / `Chem.AddHs(p_mol)`.
- Validate element order exactly.
- Set initial charges from the RDKit hydrogen-expanded molecule.
- Set `atoms.info["charge"]` to the sum of formal charges.
- Set `atoms.info["spin"] = 1`.
- Return metadata containing absolute or normalized endpoint paths and source mode.

Only XYZ/extxyz support is required for issue #15. The loader may rely on ASE format inference, but tests must cover `.xyz`.

### 5.3 `reactx/cli.py`

Remove the placement path:

- Remove imports from `reactx.embed3d` and `reactx.placement`.
- Remove `_select_relaxed_endpoint()`.
- Remove `_orientation_filtered_trials()`.
- Remove `_placement_changes_for_side()`.
- Remove `_placement_meta()`.
- Load explicit endpoints after parsing `.rxn`.
- Run endpoint relaxation on loaded endpoints.
- Align product endpoint to reactant.
- Run NEB as before.

`BondChanges.from_reaction_diff()` remains useful for validation and metadata, even though it no longer drives placement.

### 5.4 Placement code cleanup

Because backward compatibility is not required, remove the placement engine once the CLI uses explicit endpoints.

Required cleanup:

- Delete `reactx/placement.py`.
- Delete or rewrite placement-specific tests.
- Move any still-useful utility, such as coordinate-to-`Atoms` construction, into `reactx/endpoints.py`.

Do not keep placement code only as a dormant fallback.

### 5.5 Examples

Update each example TOML to include endpoint files:

- `examples/sn2.reactant.xyz`
- `examples/sn2.product.xyz`
- equivalent files for the other example reactions

The endpoint files must use the exact atom order expected by `Chem.AddHs()` for the corresponding `.rxn` side.

All existing example reactions must be migrated to explicit endpoint files in the same change so the documented examples remain runnable under the new schema. SN2 remains the acceptance-critical stereochemical example because issue #15 is motivated by Walden inversion.

## 6. Error handling

CLI exit behavior:

- Config/schema errors: exit code `2`.
- Missing endpoint file: exit code `2`.
- Unreadable endpoint file: exit code `2`.
- Multiple-frame endpoint file: exit code `2`.
- Atom count mismatch: exit code `2`.
- Element order mismatch: exit code `2`.
- Bond-change inference failure: exit code `2`.
- Endpoint relaxation non-convergence: do not abort; record in metadata.
- NEB partial convergence: keep current behavior and write outputs when possible.

Error messages should name the side (`reactant_structure` or `product_structure`), the expected value, and the actual value where applicable.

## 7. Metadata

`meta.json` must include endpoint provenance:

```json
{
  "endpoint_source": {
    "mode": "explicit_xyz",
    "reactant_structure": "examples/sn2.reactant.xyz",
    "product_structure": "examples/sn2.product.xyz"
  },
  "bond_changes": {
    "formed": [[0, 1]],
    "broken": [[1, 2]]
  }
}
```

`effective_params.placement` is removed. `effective_params.endpoint_relax` and `effective_params.neb` remain.

The endpoint source paths may be stored as TOML-relative normalized paths or resolved paths, but the format must be deterministic and covered by tests.

## 8. Tests

Add or update tests for:

- Minimal config with required endpoint paths.
- Missing `reactant_structure` / `product_structure` rejected.
- `[placement]` rejected.
- Obsolete AFIR/scoring/sampling keys rejected.
- Endpoint loader resolves paths relative to sidecar TOML.
- Endpoint loader rejects missing files.
- Endpoint loader rejects atom count mismatch.
- Endpoint loader rejects element order mismatch.
- Endpoint loader applies formal charges and spin metadata.
- CLI explicit endpoint path does not call `valid_placements()`.
- CLI writes `meta.json` with `endpoint_source.mode == "explicit_xyz"`.
- SN2 explicit endpoint example preserves Walden inversion without placement.

Delete tests that only verify the removed placement engine unless the code remains for a non-production reason. If any placement helper remains, tests must describe its retained purpose clearly.

## 9. Documentation

Update README:

- Rename the current Phase 11 placement-oriented description to the explicit endpoint pipeline.
- Show the new TOML schema.
- List endpoint files in the examples table.
- Remove discussion of Fibonacci candidate placement from normal usage.
- State that `.rxn` still provides mapping and connectivity, while XYZ files provide NEB endpoint coordinates.

## 10. Acceptance criteria

- Users can specify complete reactant and product 3D endpoint files.
- The default CLI path never calls Fibonacci placement.
- SN2 example keeps Walden inversion using explicit endpoints.
- `meta.json` records endpoint source.
- `[placement]` and old placement keys are rejected.
- `reactx/placement.py` and placement-only tests are deleted.
- The default test suite passes.
