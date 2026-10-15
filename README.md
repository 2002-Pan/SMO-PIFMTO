# SMO-PIFMTO

This repository is the research-code companion to the manuscript
**“Prior-Informed Feature-Mapped Topology Optimization with Latent Feature
Relaxation and Adaptive Feature-Space Reopening.”** Its purpose is to support
reproduction of the principal numerical examples reported in the paper; it is
not intended to be a general-purpose software package.

## Reproducibility scope

The repository contains the paper-specific implementation and inputs:

- differentiable SIMP front ends and the adaptive handover monitor;
- density-to-feature inverse fitting used for prior-informed initialization;
- the FMTO extensions for latent feature relaxation and adaptive feature-space
  reopening;
- paper-aligned SIMP inputs and FMTO inputs for all five benchmark problems;
- a small SIMP-to-feature smoke-test fixture; and
- lightweight checks for repository structure and configuration integrity.

The neural implicit representation builds on the public
[NHSDF-TOp](https://github.com/Alexander19970212/NHSDF-TOp) implementation of
Kolomeitsev and Phan. Unmodified NHSDF runtime modules, pretrained parameters,
latent statistics, and meshes are not vendored here. The setup helper retrieves
them from the upstream repository at the pinned commit used for this artifact.

## Repository layout

```text
simp_frontend/       JAX-FEM SIMP front ends
bridge/              SIMP-density to FMTO-feature conversion
NN_TopOpt/           paper-specific FMTO and reopening implementation
configs/simp/        paper-aligned SIMP inputs for five benchmarks
configs/fmto/        paper-specific FMTO inputs for five benchmarks
test_problems/       paper-aligned boundary/load registry
scripts/             helper for retrieving pinned NHSDF dependencies
examples/smoke/      small SIMP-to-feature interchange fixture
tests/               lightweight reproducibility checks
```

## External research dependencies

| Dependency | Pinned revision | Use |
| --- | --- | --- |
| [JAX-FEM](https://github.com/deepmodeling/jax-fem) | `0d767d1b686006781d21b98372936e58d6c49417` | differentiable finite-element/SIMP backend |
| [NHSDF-TOp](https://github.com/Alexander19970212/NHSDF-TOp) | `b5631e7bb20d98b6a8a55dd55856274b98825e9d` | neural implicit models and baseline FMTO assets |

The accompanying files identify their provenance in `SOURCE_MANIFEST.md` and
`THIRD_PARTY_NOTICES.md`.

## Setup

Python 3.9 or 3.10 is recommended. Git is required for fetching the pinned
NHSDF assets.

```bash
git clone https://github.com/2002-Pan/SMO-PIFMTO.git
cd SMO-PIFMTO

# Download only the upstream NHSDF components required by this artifact.
python scripts/fetch_nhsdf_assets.py

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1` and set `$env:PYTHONPATH = $PWD`.

Install the CUDA-compatible JAX/JAXLIB and PyTorch builds appropriate for the
host before running the GPU experiments if the default wheels are unsuitable.

## Five benchmark inputs

| Paper task | SIMP input | FMTO input |
| --- | --- | --- |
| Half MBB beam | `configs/simp/MBB_HALF.yaml` | `configs/fmto/MBB_beam_half.yaml` |
| Distributed-load cantilever | `configs/simp/Canti_beam_distrl.yaml` | `configs/fmto/Cant_beam_distrL.yaml` |
| Square beam | `configs/simp/Square_beam.yaml` | `configs/fmto/Square_beam.yaml` |
| Square cantilever | `configs/simp/Square_cantilever_beam.yaml` | `configs/fmto/Square_cantilever_beam.yaml` |
| Bracket | `configs/simp/Brecket.yaml` | `configs/fmto/Brecket.yaml` |

## Running the workflow

Run commands from the repository root.

```bash
# 1. Run a SIMP case. Use simpproblem_mask for the bracket/masked domain.
FMTO_SIMP_CONFIG="$PWD/configs/simp/MBB_HALF.yaml" \
  python -m simp_frontend.simpproblem

# 2. Convert the SIMP density field to an FMTO feature initialization.
python -m bridge.density2feature \
  --input_pt examples/smoke/jax_mbb_beam.pt \
  --template_yaml configs/fmto/MBB_beam_half.yaml \
  --output_yaml artifacts/generated_configs/MBB_beam_half.yaml

# 3. Run the prior-informed FMTO stage.
FMTO_CONFIG="$PWD/configs/fmto/MBB_beam_half.yaml" \
  python NN_TopOpt/nntopo_example.py
```

Generated configurations and optimization output belong under `artifacts/` or
the ignored run-log directories. Numerical results may vary slightly with GPU,
CUDA, JAX, and PyTorch versions.

## Validation

The repository check does not import JAX or PyTorch:

```bash
python -m unittest tests.test_repository_layout
```

This check validates the tracked research artifact. A full numerical
reproduction additionally requires the external assets and the intended GPU
environment.

## Suggested manuscript statement

> **Replication of results.** The paper-specific implementation, benchmark
> configurations, and scripts supporting reproduction of the reported results
> are available at https://github.com/2002-Pan/SMO-PIFMTO. The neural implicit
> feature representation builds upon the publicly available NHSDF
> implementation of Kolomeitsev and Phan; the original NHSDF source code and
> pretrained models are available at
> https://github.com/Alexander19970212/NHSDF-TOp.

## Citation

Citation metadata for this research artifact are provided in `CITATION.cff`.
Please also cite JAX-FEM and the original NHSDF work when using their respective
components.
