import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.main import main, parse_parameter_overrides, resolve_problem_names


class MainIntegrationTest(unittest.TestCase):
    def test_parses_parameter_overrides(self):
        self.assertEqual(
            parse_parameter_overrides(
                ["step_size=0.02", "max_iterations=25"],
                "--solver-param",
            ),
            {"step_size": 0.02, "max_iterations": 25},
        )

    def test_resolves_comma_and_space_separated_problems(self):
        self.assertEqual(
            resolve_problem_names(
                ["guven,wang", "reverse_guven"],
                "projected_gradient_descent",
            ),
            ["guven", "wang", "reverse_guven"],
        )

    def test_all_selects_only_solver_compatible_problems(self):
        self.assertEqual(
            resolve_problem_names(["all"], "linprog"),
            ["obj_fun_2", "guven", "wang", "reverse_guven"],
        )
        self.assertEqual(
            resolve_problem_names(["all"], "slsqp"),
            ["obj_fun_1", "guven", "reverse_guven"],
        )

    def test_explicit_incompatible_problem_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incompatible combination"):
            resolve_problem_names(["obj_fun_1"], "linprog")

    def test_writes_result_tree_for_each_selected_problem(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "benchmark.yaml"
            output_root = root / "results"
            config = {
                "benchmark": {"name": "tiny", "random_seed": 0},
                "geometry": {
                    "type": "testcube",
                    "domain_dim": [3, 3, 3],
                    "offset": [1, 1, 1],
                },
                "process": {
                    "layer_height": 0.05,
                    "exposure_time": 1.0,
                    "atten_coef": 1.0,
                    "sigma": 1.0,
                    "I_max": 1.0,
                    "E_crit": 0.1,
                    "min_infl_factor": 0.1,
                },
                "initialization": {"type": "target_scaled", "scale": 1.0},
                "target_energy": {"type": "attenuated_layers", "scale": 0.5},
                "metrics": [
                    "undercured_voxels",
                    "overcured_voxels",
                    "outside_energy",
                    "min_inside_energy",
                    "runtime_seconds",
                    "optimization_success",
                    "iterations",
                ],
                "output": {
                    "directory": str(output_root),
                    "comparison_csv": "comparison.csv",
                    "save_intensity_arrays": False,
                    "save_plots": False,
                },
            }
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with patch.object(
                sys,
                "argv",
                [
                    "main",
                    "--config",
                    str(config_path),
                    "--problem",
                    "guven,wang",
                    "--solver",
                    "linprog",
                ],
            ):
                main()

            run_directory = output_root / "tiny" / "guven" / "linprog_highs"
            result = json.loads(
                (run_directory / "result.json").read_text(encoding="utf-8")
            )
            self.assertTrue(result["optimization_success"])
            self.assertEqual(result["formulation"], "linear_program")
            self.assertEqual(result["constraint_handling"], "direct")
            self.assertEqual(result["penalty_parameters"], {})
            self.assertAlmostEqual(
                result["base_objective_value"],
                result["formulation_objective_value"],
            )
            snapshot_path = run_directory / "resolved_config.yaml"
            self.assertTrue(snapshot_path.is_file())
            snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(snapshot["selection"]["problem"], "guven")
            self.assertEqual(snapshot["selection"]["solver"], "linprog_highs")
            wang_result = (
                output_root
                / "tiny"
                / "wang"
                / "linprog_highs"
                / "result.json"
            )
            self.assertTrue(wang_result.is_file())
            comparison_path = output_root / "tiny" / "comparison.csv"
            with comparison_path.open(newline="", encoding="utf-8") as comparison_file:
                comparison_rows = list(csv.DictReader(comparison_file))
            self.assertEqual(len(comparison_rows), 2)


if __name__ == "__main__":
    unittest.main()
