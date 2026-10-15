import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTest(unittest.TestCase):
    def test_required_files_exist(self):
        required = [
            "simp_frontend/simpproblem.py",
            "simp_frontend/simpproblem_mask.py",
            "bridge/density2feature.py",
            "bridge/density2feature_mask.py",
            "NN_TopOpt/NN_TopOpt.py",
            "NN_TopOpt/TopOpt.py",
            "NN_TopOpt/shape_bridge.py",
            "models/sdf_models.py",
            "dataset_generation/utils_generation.py",
            "model_weights/frst_VAE_DeepSDF_quad_Bprec_full.pt",
            "z_limits/VAE_DeepSDF_quad_full_stats.npz",
        ]
        missing = [item for item in required if not (ROOT / item).is_file()]
        self.assertEqual([], missing)

    def test_five_benchmark_configs_and_meshes_exist(self):
        simp_configs = {
            "MBB_HALF.yaml",
            "Canti_beam_distrl.yaml",
            "Square_beam.yaml",
            "Square_cantilever_beam.yaml",
            "Brecket.yaml",
        }
        fmto_configs = {
            "MBB_beam_half.yaml",
            "Cant_beam_distrL.yaml",
            "Square_beam.yaml",
            "Square_cantilever_beam.yaml",
            "Brecket.yaml",
        }
        meshes = {
            "MBB_beam_half.msh",
            "Cantilever_beam_distributed_load.msh",
            "Square_beam.msh",
            "Square_cantilever_beam.msh",
            "Brecket.msh",
        }
        self.assertTrue(simp_configs <= {p.name for p in (ROOT / "configs/simp").glob("*.yaml")})
        self.assertTrue(fmto_configs <= {p.name for p in (ROOT / "configs/fmto").glob("*.yaml")})
        self.assertTrue(meshes <= {p.name for p in (ROOT / "test_problems").glob("*.msh")})

    def test_python_sources_parse(self):
        failures = []
        for path in ROOT.rglob("*.py"):
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (SyntaxError, UnicodeError) as exc:
                failures.append(f"{path.relative_to(ROOT)}: {exc}")
        self.assertEqual([], failures)

    def test_no_github_oversized_files(self):
        oversized = [
            str(path.relative_to(ROOT))
            for path in ROOT.rglob("*")
            if path.is_file() and path.stat().st_size > 100 * 1024 * 1024
        ]
        self.assertEqual([], oversized)


if __name__ == "__main__":
    unittest.main()
