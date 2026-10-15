"""Fetch the pinned, unmodified NHSDF components required by this artifact."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess


REPOSITORY = "https://github.com/Alexander19970212/NHSDF-TOp.git"
COMMIT = "b5631e7bb20d98b6a8a55dd55856274b98825e9d"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "external" / "NHSDF-TOp"

FILES = {
    "NN_TopOpt/mesh_utils.py": "NN_TopOpt/mesh_utils.py",
    "models/sdf_models.py": "models/sdf_models.py",
    "models/rs_loss_models.py": "models/rs_loss_models.py",
    "dataset_generation/utils_generation.py": "dataset_generation/utils_generation.py",
    "configs/NN_sdf_experiments/final_experiments/VAEs/VAE_DeepSDF_quad.yaml":
        "configs/model/VAE_DeepSDF_quad.yaml",
    "model_weights/frst_VAE_DeepSDF_quad_Bprec_full.pt":
        "model_weights/frst_VAE_DeepSDF_quad_Bprec_full.pt",
    "z_limits/VAE_DeepSDF_quad_full_stats.npz":
        "z_limits/VAE_DeepSDF_quad_full_stats.npz",
    "test_problems/MBB_beam_half.msh": "test_problems/MBB_beam_half.msh",
    "test_problems/Cantilever_beam_distributed_load.msh":
        "test_problems/Cantilever_beam_distributed_load.msh",
    "test_problems/Square_beam.msh": "test_problems/Square_beam.msh",
    "test_problems/Square_cantilever_beam.msh":
        "test_problems/Square_cantilever_beam.msh",
    "test_problems/Brecket.msh": "test_problems/Brecket.msh",
}

EXPECTED_SHA256 = {
    "model_weights/frst_VAE_DeepSDF_quad_Bprec_full.pt":
        "1ba603ec9cae273ea9c98e2e60c1cafb7ecda8c22b8f08c5bcc03d15ba9f0d51",
    "z_limits/VAE_DeepSDF_quad_full_stats.npz":
        "687ab25873e0fe93ad518194bdb95ef452a23de7d44e425139e5b835a3689ee1",
}


def run(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.strip()


def obtain_source(destination: Path) -> Path:
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY, str(destination))
    if not (destination / ".git").is_dir():
        raise RuntimeError(f"Existing source is not a Git checkout: {destination}")
    if run("git", "status", "--porcelain", cwd=destination):
        raise RuntimeError(f"Refusing to modify a dirty NHSDF checkout: {destination}")
    run("git", "fetch", "origin", COMMIT, cwd=destination)
    run("git", "checkout", "--detach", COMMIT, cwd=destination)
    return destination


def verify_source(source: Path) -> None:
    missing = [relative for relative in FILES if not (source / relative).is_file()]
    if missing:
        raise FileNotFoundError("Missing NHSDF files:\n" + "\n".join(missing))

    if (source / ".git").is_dir():
        actual = run("git", "rev-parse", "HEAD", cwd=source)
        if actual != COMMIT:
            raise RuntimeError(f"Expected NHSDF commit {COMMIT}, found {actual}")

    for relative, expected in EXPECTED_SHA256.items():
        actual = hashlib.sha256((source / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {relative}: {actual}")


def install_files(source: Path) -> None:
    for source_relative, target_relative in FILES.items():
        target = ROOT / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / source_relative, target)
        print(f"installed {target_relative}")

    # Package markers are local glue, not copies of upstream implementation.
    for marker in (ROOT / "models/__init__.py", ROOT / "dataset_generation/__init__.py"):
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch(exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        help="Use an existing NHSDF checkout at the pinned commit instead of cloning.",
    )
    args = parser.parse_args()

    source = args.source.resolve() if args.source else obtain_source(DEFAULT_SOURCE)
    verify_source(source)
    install_files(source)
    print(f"NHSDF assets installed from {COMMIT}")


if __name__ == "__main__":
    main()
