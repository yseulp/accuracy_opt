import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.optimization_history import (
    _default_output_directory,
    run_history_analysis,
)


class OptimizationHistoryTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.benchmark_path = self.root / "benchmark.yaml"
        self.optimization_path = self.root / "optimization.yaml"
        self.benchmark_path.write_text(
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
                    "initialization": {"type": "target_scaled", "scale": 1.0},
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
                        "directory": str(self.root / "unused"),
                        "comparison_csv": "comparison.csv",
                        "save_intensity_arrays": False,
                        "save_plots": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        self.optimization_path.write_text(
            yaml.safe_dump(
                {
                    "problems": {},
                    "penalties": {
                        "guven": {"penalty_weight": 10.0},
                        "wang": {"penalty_weight": 10.0},
                        "reverse_guven": {"penalty_weight": 10.0},
                    },
                    "solvers": {
                        "projected_gradient_descent": {
                            "step_size": 0.03,
                            "max_iterations": 2,
                            "objective_tolerance": None,
                        },
                        "slsqp": {"max_iterations": 10, "ftol": 1e-8},
                        "trust_constr": {
                            "max_iterations": 20,
                            "gtol": 1e-6,
                            "xtol": 1e-6,
                            "barrier_tol": 1e-6,
                        },
                        "linprog_highs": {},
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_default_output_root_is_separate_from_production_runs(self):
        actual = _default_output_directory(
            {
                "benchmark": {"name": "tiny"},
                "output": {"directory": str(self.root / "results")},
            },
            "guven",
            "projected_gradient_descent",
        )

        self.assertEqual(
            actual,
            self.root
            / "results"
            / "optimization_history"
            / "tiny"
            / "guven"
            / "projected_gradient_descent",
        )

    def test_pgd_history_records_updates_and_separate_penalty_terms(self):
        output = self.root / "pgd_history"
        history, metadata = run_history_analysis(
            benchmark_config_path=self.benchmark_path,
            optimization_config_path=self.optimization_path,
            problem_name="guven",
            solver_name="projected_gradient_descent",
            output_directory=output,
            save_pdf=True,
        )

        self.assertEqual([record.iteration for record in history], [0, 1, 2])
        self.assertTrue(metadata["history_available"])
        self.assertEqual(metadata["constraint_handling"], "penalty")
        for record in history:
            self.assertAlmostEqual(
                record.formulation_objective,
                record.original_objective + record.weighted_penalty,
            )
            self.assertIsNotNone(record.raw_penalty)
        self.assertTrue((output / "optimization_history.csv").is_file())
        self.assertTrue((output / "optimization_history.png").is_file())
        self.assertTrue((output / "optimization_history.pdf").is_file())

    def test_highs_records_only_final_state_and_limitation(self):
        output = self.root / "highs_history"
        history, metadata = run_history_analysis(
            benchmark_config_path=self.benchmark_path,
            optimization_config_path=self.optimization_path,
            problem_name="guven",
            solver_name="linprog",
            output_directory=output,
        )

        self.assertFalse(metadata["history_available"])
        self.assertEqual(len(history), 1)
        self.assertIsNone(history[0].iteration)
        self.assertEqual(history[0].record_type, "final_only")
        self.assertTrue(metadata["solution_available"])
        with (output / "optimization_history.csv").open(
            newline="", encoding="utf-8"
        ) as history_file:
            self.assertEqual(len(list(csv.DictReader(history_file))), 1)
        saved_metadata = json.loads(
            (output / "optimization_history_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("does not expose iterate callbacks", saved_metadata["history_limitation"])
        self.assertTrue((output / "optimization_history.png").is_file())

    def test_pgd_history_respects_objective_tolerance(self):
        history, metadata = run_history_analysis(
            benchmark_config_path=self.benchmark_path,
            optimization_config_path=self.optimization_path,
            problem_name="guven",
            solver_name="projected_gradient_descent",
            solver_parameters={"objective_tolerance": 1e20},
            output_directory=self.root / "pgd_tolerance_history",
        )

        self.assertEqual(len(history), 2)
        self.assertTrue(metadata["success"])
        self.assertTrue(metadata["converged"])

    def test_failed_highs_run_is_marked_as_fallback_not_history(self):
        infeasible_path = self.root / "infeasible_benchmark.yaml"
        benchmark_config = yaml.safe_load(
            self.benchmark_path.read_text(encoding="utf-8")
        )
        benchmark_config["process"]["E_crit"] = 100.0
        infeasible_path.write_text(
            yaml.safe_dump(benchmark_config),
            encoding="utf-8",
        )

        history, metadata = run_history_analysis(
            benchmark_config_path=infeasible_path,
            optimization_config_path=self.optimization_path,
            problem_name="guven",
            solver_name="linprog",
            output_directory=self.root / "failed_highs_history",
        )

        self.assertFalse(metadata["solution_available"])
        self.assertFalse(metadata["success"])
        self.assertEqual(history[0].record_type, "failed_fallback")

    def test_slsqp_history_uses_major_iteration_callbacks(self):
        history, metadata = run_history_analysis(
            benchmark_config_path=self.benchmark_path,
            optimization_config_path=self.optimization_path,
            problem_name="guven",
            solver_name="slsqp",
            output_directory=self.root / "slsqp_history",
        )

        self.assertTrue(metadata["history_available"])
        self.assertEqual(history[0].iteration, 0)
        self.assertEqual(
            metadata["iteration_semantics"],
            "one SLSQP major iteration callback",
        )
        self.assertLessEqual(history[-1].iteration, metadata["reported_iterations"])
        iterations = [record.iteration for record in history]
        self.assertEqual(len(iterations), len(set(iterations)))

    def test_trust_constr_history_uses_state_iteration(self):
        history, metadata = run_history_analysis(
            benchmark_config_path=self.benchmark_path,
            optimization_config_path=self.optimization_path,
            problem_name="guven",
            solver_name="trust_constr",
            output_directory=self.root / "trust_history",
        )

        self.assertTrue(metadata["history_available"])
        self.assertEqual(history[0].iteration, 0)
        self.assertEqual(
            metadata["iteration_semantics"],
            "trust-constr callback state.nit",
        )
        self.assertEqual(history[-1].iteration, metadata["reported_iterations"])
        iterations = [record.iteration for record in history]
        self.assertEqual(len(iterations), len(set(iterations)))


if __name__ == "__main__":
    unittest.main()
