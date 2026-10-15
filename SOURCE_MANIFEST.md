# Source manifest

This manifest separates the paper-specific research artifact from external
research dependencies.

## Paper-specific implementation

The following tracked components contain the implementation and inputs used by
the manuscript:

- `simp_frontend/`: differentiable SIMP exploration and handover monitoring;
- `bridge/`: density-to-feature fitting and prior construction;
- `NN_TopOpt/NN_TopOpt.py`, `NN_TopOpt/TopOpt.py`,
  `NN_TopOpt/shape_bridge.py`, and `NN_TopOpt/nntopo_example.py`: the FMTO
  extensions, adaptive feature-space reopening, and experiment entry point;
- `NN_TopOpt/mmapy2/`: the MMA implementation used by the modified optimizer;
- `configs/simp/` and `configs/fmto/`: five manuscript benchmark inputs; and
- `test_problems/problems.json`: paper-aligned support/load registry.

The SIMP and bridge files were developed in a local JAX-FEM worktree based on
JAX-FEM revision `0d767d1b686006781d21b98372936e58d6c49417`. The FMTO files
were developed from NHSDF-TOp and contain the paper-specific changes on top of
that implementation.

## External NHSDF components

The following unmodified NHSDF components are intentionally not tracked:

- neural model definitions used at runtime;
- geometry helper functions;
- the quadrilateral neural-model configuration;
- the pretrained quadrilateral model;
- latent normalization statistics; and
- the five benchmark mesh files.

`scripts/fetch_nhsdf_assets.py` retrieves those files from
<https://github.com/Alexander19970212/NHSDF-TOp> at commit
`b5631e7bb20d98b6a8a55dd55856274b98825e9d` and verifies the checksums of the
pretrained model and latent statistics before installing them into ignored
local paths.

## Staging adjustments

- hard-coded development-machine paths were replaced with repository-relative
  paths;
- generated configurations and outputs were routed to ignored locations;
- `FMTO_SIMP_CONFIG` and `FMTO_CONFIG` environment overrides were added;
- the five SIMP YAML files and the support/load registry were aligned with the
  manuscript benchmark definitions; and
- documentation, dependency retrieval, validation, and Git metadata were added.
