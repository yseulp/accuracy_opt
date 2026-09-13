import tempfile
import unittest
from pathlib import Path

from scripts.optimization.config import (
    defaults_for_selection,
    load_optimization_defaults,
)
from scripts.optimization.solvers import resolve_selection


class OptimizationConfigTest(unittest.TestCase):
    def test_loads_user_configurable_defaults(self):
        config = load_optimization_defaults(
            Path("configs/optimization_defaults.yaml")
        )

        self.assertEqual(config["penalties"]["wang"]["penalty_weight"], 50.0)
        self.assertEqual(
            config["solvers"]["projected_gradient_descent"]["step_size"],
            0.03,
        )

    def test_selection_uses_only_relevant_problem_defaults(self):
        config = load_optimization_defaults(
            Path("configs/optimization_defaults.yaml")
        )
        problem_defaults, penalty_defaults, solver_defaults = defaults_for_selection(
            config, "guven", "projected_gradient_descent"
        )
        pgd = resolve_selection(
            "guven",
            "projected_gradient_descent",
            problem_defaults=problem_defaults,
            penalty_defaults=penalty_defaults,
            solver_defaults=solver_defaults,
        )
        problem_defaults, penalty_defaults, solver_defaults = defaults_for_selection(
            config, "guven", "linprog"
        )
        linear = resolve_selection(
            "guven",
            "linprog",
            problem_defaults=problem_defaults,
            penalty_defaults=penalty_defaults,
            solver_defaults=solver_defaults,
        )

        self.assertEqual(pgd["problem_parameters"], {})
        self.assertEqual(pgd["penalty_parameters"], {"penalty_weight": 100.0})
        self.assertEqual(linear["problem_parameters"], {})
        self.assertEqual(linear["penalty_parameters"], {})
        self.assertEqual(linear["solver_parameters"], {})

    def test_cli_override_has_highest_priority(self):
        selection = resolve_selection(
            "wang",
            "projected_gradient_descent",
            penalty_defaults={"penalty_weight": 50.0},
            solver_defaults={"step_size": 0.03},
            penalty_parameters={"penalty_weight": 250.0},
            solver_parameters={"step_size": 0.02},
        )

        self.assertEqual(selection["penalty_parameters"]["penalty_weight"], 250.0)
        self.assertEqual(selection["solver_parameters"]["step_size"], 0.02)

    def test_rejects_unknown_problem_parameter(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "optimization.yaml"
            config_path.write_text(
                "penalties:\n  guven:\n    penalty_weigth: 100\nsolvers: {}\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "penalty_weigth"):
                load_optimization_defaults(config_path)


if __name__ == "__main__":
    unittest.main()
