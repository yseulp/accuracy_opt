import unittest
from pathlib import Path

from scripts.benchmark.config import load_benchmark_config


class BenchmarkConfigTest(unittest.TestCase):
    def test_example_config_contains_only_benchmark_settings(self):
        config = load_benchmark_config(
            Path("configs/benchmark_testcube_20.yaml")
        )

        self.assertEqual(config["geometry"]["domain_dim"], [20, 20, 20])
        self.assertEqual(config["visualization"]["slice_y"], 10)
        self.assertNotIn("runs", config)
        self.assertNotIn("problem", config)
        self.assertNotIn("solver", config)


if __name__ == "__main__":
    unittest.main()
