# Prior-Informed FMTO — staging repository

This repository is a curated staging copy for the paper **“Prior-Informed
Feature-Mapped Topology Optimization with Latent Feature Relaxation and
Adaptive Feature-Space Reopening.”** It contains the smallest practical subset
identified from the local JAX-FEM and NHSDF development trees.

> **Staging status:** the source selection, paths, model, five benchmark meshes,
> and configurations are present. This is not yet a publication-ready release:
> the two bridge variants must be validated, a full pipeline reproduction must
> be completed, and a project-level license must be chosen after upstream
> permissions are confirmed.

The original projects were not modified.

## Pipeline

1. `simp_frontend/`: differentiable SIMP exploration with an adaptive handover
   monitor.
2. `bridge/`: density-to-feature inverse fitting and morphological reopening
   priors.
3. `NN_TopOpt/`: latent Neural-SDF FMTO, feature-space reopening, observation,
   and rollback.

## Repository layout

```text
simp_frontend/       JAX-FEM SIMP front ends
bridge/              active density-to-feature copies from jax-fem
NN_TopOpt/           FMTO backend and MMA implementation
models/              neural implicit models
dataset_generation/  geometric descriptor utilities required at runtime
configs/simp/        five SIMP benchmark inputs
configs/fmto/        five FMTO inputs and mapped-feature templates
configs/model/       VAE architecture
test_problems/       five meshes and problem registry
model_weights/       pretrained quadrilateral VAE/DeepSDF checkpoint
z_limits/            latent normalization statistics
examples/smoke/      small SIMP-to-bridge interchange fixture
review_candidates/   non-active duplicate requiring comparison
provenance/          upstream README and dependency snapshot
```

## Environment draft

Python 3.9 or 3.10 is recommended until the complete environment is locked.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

The JAX/JAXLIB installation may need to be adjusted for the host CUDA version.
The JAX-FEM dependency is pinned to the inspected upstream commit in
`requirements.txt`.

## Staging commands

Run from the repository root. These commands smoke-test the staged layout; full
benchmark reproduction still requires running and checking all five cases in
the intended environment.

```bash
# Select a SIMP configuration; use simpproblem_mask for a masked domain.
FMTO_SIMP_CONFIG="$PWD/configs/simp/MBB_HALF.yaml" \
  python -m simp_frontend.simpproblem

# Run the bridge by itself on the included small fixture.
python -m bridge.density2feature \
  --input_pt examples/smoke/jax_mbb_beam.pt \
  --template_yaml configs/fmto/MBB_beam_half.yaml \
  --output_yaml artifacts/generated_configs/MBB_beam_half.yaml

# Run the FMTO backend with a generated or curated configuration.
FMTO_CONFIG="$PWD/configs/fmto/MBB_beam_half.yaml" \
  python NN_TopOpt/nntopo_example.py
```

Generated data belongs under `artifacts/`, which is ignored by Git.

## Known release blockers

- `nntopo_example.py` still has a default benchmark and should gain a complete
  CLI for config, device, seed, output directory, and ablations.
- `bridge/density2feature_mask.py` is the active JAX-side copy. A newer,
  divergent NHSDF copy is retained in `review_candidates/`; merge only after a
  benchmark comparison.
- A full pipeline test has not been run in this staging directory.
- The project-level license is unresolved; see `LICENSE.TODO.md` and
  `THIRD_PARTY_NOTICES.md`.

## Validation

The lightweight repository check does not import JAX or PyTorch:

```bash
python -m unittest tests.test_repository_layout
```

## Citation

The draft citation metadata is in `CITATION.cff`. Add the final GitHub URL and
DOI before publication. Preserve the JAX-FEM and NHSDF citations in any public
release.
