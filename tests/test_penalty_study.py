import csv
import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.penalty_study import load_study_config, run_study


class PenaltyStudyTest(unittest.TestCase):
    def test_rejects_empty_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "study.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "benchmark_config": "benchmark.yaml",
                        "problem": "guven",
                        "penalty_weights": [1],
                        "output_directory": "",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "non-empty path"):
                load_study_config(config_path)

    def test_tiny_guven_study_writes_history_summary_and_plots(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark_path = root / "benchmark.yaml"
            output_directory = root / "output"
            study_path = root / "study.yaml"
            benchmark_path.write_text(
                yaml.safe_dump(
                    {
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
                        "initialization": {
                            "type": "target_scaled",
                            "scale": 1.0,
                        },
                        "target_energy": {
                            "type": "attenuated_layers",
                            "reference": "initial_intensity",
                            "scale": 0.5,
                        },
                        "metrics": [
                            "undercured_voxels",
                            "overcured_voxels",
                            "outside_energy",
                            "min_inside_energy",
                            "runtime_seconds",
                            "optimization_success",
                        ],
                        "output": {
                            "directory": str(root / "unused"),
                            "comparison_csv": "comparison.csv",
                            "save_intensity_arrays": False,
                            "save_plots": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            study_path.write_text(
                yaml.safe_dump(
                    {
                        "benchmark_config": str(benchmark_path),
                        "problem": "guven",
                        "solver": "projected_gradient_descent",
                        "problem_parameters": {},
                        "solver_parameters": {
                            "step_size": 0.03,
                            "max_iterations": 2,
                            "objective_tolerance": None,
                        },
                        "penalty_weights": [1, 10],
                        "convergence_plot_weights": [1, 10],
                        "output_directory": str(output_directory),
                    }
                ),
                encoding="utf-8",
            )

            summaries, histories = run_study(study_path)

            self.assertEqual(len(summaries), 2)
            self.assertEqual(len(histories[1.0]), 3)
            self.assertEqual(
                [record.iteration for record in histories[1.0]],
                [0, 1, 2],
            )
            self.assertAlmostEqual(
                histories[10.0][-1].penalized_objective,
                histories[10.0][-1].original_objective
                + histories[10.0][-1].weighted_penalty,
            )
            self.assertGreaterEqual(summaries[0].logged_runtime_seconds, 0.0)
            self.assertGreaterEqual(summaries[0].runtime_seconds, 0.0)
            with (output_directory / "penalty_study.csv").open(
                newline="", encoding="utf-8"
            ) as summary_file:
                summary_rows = list(csv.DictReader(summary_file))
                self.assertEqual(len(summary_rows), 2)
                self.assertIn("final_constraint_violation", summary_rows[0])
                self.assertIn("logged_runtime_seconds", summary_rows[0])
            with (output_directory / "convergence_penalty_1.csv").open(
                newline="", encoding="utf-8"
            ) as history_file:
                history_rows = list(csv.DictReader(history_file))
                self.assertEqual(len(history_rows), 3)
                self.assertIn("raw_penalty", history_rows[0])
            for artifact in (
                "convergence_penalty_1.csv",
                "convergence_penalty_10.csv",
                "objective_vs_penalty.png",
                "constraint_vs_penalty.png",
                "cure_violations_vs_penalty.png",
                "convergence_comparison.png",
                "resolved_study_config.yaml",
            ):
                self.assertTrue((output_directory / artifact).is_file(), artifact)


if __name__ == "__main__":
    unittest.main()
