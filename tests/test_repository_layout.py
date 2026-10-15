import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTest(unittest.TestCase):
    def test_required_tracked_artifact_files_exist(self):
        required = [
            "simp_frontend/simpproblem.py",
            "simp_frontend/simpproblem_mask.py",
            "bridge/density2feature.py",
            "bridge/density2feature_mask.py",
            "NN_TopOpt/NN_TopOpt.py",
            "NN_TopOpt/TopOpt.py",
            "NN_TopOpt/shape_bridge.py",
            "NN_TopOpt/nntopo_example.py",
            "scripts/fetch_nhsdf_assets.py",
            "test_problems/problems.json",
            "SOURCE_MANIFEST.md",
            "THIRD_PARTY_NOTICES.md",
        ]
        missing = [item for item in required if not (ROOT / item).is_file()]
        self.assertEqual([], missing)

    def test_five_benchmark_configs_exist(self):
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
        self.assertEqual(simp_configs, {p.name for p in (ROOT / "configs/simp").glob("*.yaml")})
        self.assertEqual(fmto_configs, {p.name for p in (ROOT / "configs/fmto").glob("*.yaml")})

    def test_problem_registry_has_paper_tasks(self):
        registry = json.loads((ROOT / "test_problems/problems.json").read_text(encoding="utf-8"))
        self.assertEqual(
            {
                "MBB_beam_half",
                "Cantilever_beam_distributed_load",
                "Square_beam",
                "Square_cantilever_beam",
                "Brecket",
            },
            set(registry),
        )
        self.assertEqual([[0.0, 1.0], [0.0, -0.0025]], registry["Square_beam"]["loads"][0])
        self.assertEqual([0.0, -1.0], registry["Square_cantilever_beam"]["loads"][0][1])

    def test_python_sources_parse(self):
        failures = []
        ignored_roots = {"external", "models", "dataset_generation"}
        for path in ROOT.rglob("*.py"):
            if path.relative_to(ROOT).parts[0] in ignored_roots:
                continue
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (SyntaxError, UnicodeError) as exc:
                failures.append(f"{path.relative_to(ROOT)}: {exc}")
        self.assertEqual([], failures)

    def test_no_github_oversized_tracked_artifacts(self):
        oversized = [
            str(path.relative_to(ROOT))
            for path in ROOT.rglob("*")
            if path.is_file()
            and path.relative_to(ROOT).parts[0] not in {".git", "external"}
            and path.stat().st_size > 100 * 1024 * 1024
        ]
        self.assertEqual([], oversized)


if __name__ == "__main__":
    unittest.main()
