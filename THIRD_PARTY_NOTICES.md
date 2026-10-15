# Third-party notices

## JAX-FEM

- Repository: <https://github.com/deepmodeling/jax-fem>
- Pinned revision: `0d767d1b686006781d21b98372936e58d6c49417`
- License: GNU General Public License v3, reproduced in
  `LICENSES/JAX-FEM-GPL-3.0.txt`.

The SIMP front ends and bridge were developed in a local worktree based on this
revision. Preserve the JAX-FEM attribution and license when redistributing
JAX-FEM-derived material.

## NHSDF-TOp

- Repository: <https://github.com/Alexander19970212/NHSDF-TOp>
- Pinned revision: `b5631e7bb20d98b6a8a55dd55856274b98825e9d`
- Associated work: A. Kolomeitsev and A.-H. Phan, “Feature-Mapping Topology
  Optimization with Neural Heaviside Signed Distance Functions,” ICML 2025.

The FMTO implementation in this artifact builds on NHSDF-TOp. Unmodified
runtime modules, pretrained parameters, latent statistics, and meshes are
retrieved directly from the upstream repository by
`scripts/fetch_nhsdf_assets.py` and are not included in this Git repository.
Consult the upstream project for its current terms and citation guidance.

This notice records provenance and does not grant rights to third-party work.
