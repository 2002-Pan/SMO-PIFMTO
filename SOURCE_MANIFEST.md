# Source manifest

## JAX-FEM development tree

Source root: `/home/ubuntu/projects/jax-fem`

- `simp_frontend/simpproblem.py` <- `simpproblem.py`
- `simp_frontend/simpproblem_mask.py` <- `simpproblem_mask.py`
- `bridge/density2feature.py` <- `density2feature.py`
- `bridge/density2feature_mask.py` <- `density2feature_mask.py`
- `configs/simp/*.yaml` <- selected `configs/problems/*.yaml`
- `examples/smoke/jax_mbb_beam.pt` <- `test_samples/jax_mbb_beam.pt`

Upstream Git revision: `0d767d1b686006781d21b98372936e58d6c49417`.
The copied root-level research scripts were untracked in that Git worktree.

## NHSDF development tree

Source root: `/home/ubuntu/projects/NHSDF-TOp-main`

- `NN_TopOpt/`, `models/`, and `dataset_generation/`: selected runtime source.
- `configs/fmto/`: five selected quadrilateral FMTO configurations.
- `configs/model/VAE_DeepSDF_quad.yaml`: neural architecture.
- `test_problems/`: five meshes plus the registry.
- `model_weights/frst_VAE_DeepSDF_quad_Bprec_full.pt`:
  SHA-256 `1ba603ec9cae273ea9c98e2e60c1cafb7ecda8c22b8f08c5bcc03d15ba9f0d51`.
- `z_limits/VAE_DeepSDF_quad_full_stats.npz`:
  SHA-256 `687ab25873e0fe93ad518194bdb95ef452a23de7d44e425139e5b835a3689ee1`.

The source root was not itself a Git repository, so no source commit is
available for these files.

## Duplicate bridge candidate

The active `bridge/density2feature_mask.py` came from JAX-FEM and had source
SHA-256 `88a0d97b9ab47270f5e64f47d0ed12368bb3a5c0b277e27603c923f58969e72f`.
The later NHSDF variant is retained at
`review_candidates/density2feature_mask_nhsdf.py` and had source SHA-256
`449a4d15d36bbeec72e1c644bc19e4f8f2298cf0d5349c13574e731249310375`.
Do not silently replace one with the other; compare outputs on all five tasks.

## Staging-only modifications

- Replaced hard-coded `/home/ubuntu/projects/...` paths in active SIMP and
  bridge files with repository-relative paths.
- Routed generated configurations and optimization output to ignored
  `artifacts/` directories.
- Added `FMTO_SIMP_CONFIG` and `FMTO_CONFIG` environment overrides.
- Adjusted copied FMTO configuration paths to `configs/model/`.
- Added package markers, documentation, validation, and Git metadata.
