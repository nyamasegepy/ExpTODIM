import unittest

import numpy as np

from todim_family_engine import (
    METHOD_ORDER,
    built_in_example,
    compare_methods,
    load_builtin_case,
    normalize_matrix,
    run_todim_family,
    sensitivity_analysis,
    summarize_sensitivity,
)


class TODIMFamilyEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = built_in_example()
        cls.results = sensitivity_analysis(cls.data, methods=METHOD_ORDER)
        cls.summary = summarize_sensitivity(cls.results).set_index("method")

    def test_five_distinct_methods(self):
        self.assertEqual(len(METHOD_ORDER), 5)
        self.assertNotIn("power_todim", METHOD_ORDER)
        self.assertEqual(METHOD_ORDER[-1], "exptodim")

    def test_max_normalization_uses_inverse_ratio_for_costs(self):
        matrix = np.array([[10.0, 10.0], [20.0, 20.0]])
        normalized = normalize_matrix(matrix, np.array([1, 0]), "max")
        np.testing.assert_allclose(normalized[:, 0], [0.5, 1.0])
        np.testing.assert_allclose(normalized[:, 1], [1.0, 0.5])

    def test_reference_ranking_matches_manuscript(self):
        result = run_todim_family(
            self.data, "exptodim", "sum", lambda_=2.25, rho=3.0
        )
        ranking = [self.data.alternatives[index] for index in result["ranking"]]
        self.assertEqual(ranking, ["A1", "A2", "A3", "A4", "A5"])
        np.testing.assert_allclose(
            result["normalized_scores"],
            [1.0, 0.5049, 0.2532, 0.2408, 0.0],
            atol=5e-4,
        )

    def test_default_grid_has_580_method_specific_scenarios(self):
        self.assertEqual(len(self.results), 580)
        counts = self.results.groupby("method").size().to_dict()
        self.assertEqual(
            counts,
            {
                "classical_todim": 20,
                "generalized_inverse": 180,
                "generalized_monotone": 180,
                "log_todim": 100,
                "exptodim": 100,
            },
        )

    def test_robustness_summary_matches_manuscript(self):
        expected = {
            "classical_todim": (1.000, 0.000, 0.700, 0.600, 20),
            "generalized_inverse": (1.000, 0.000, 0.700, 0.600, 180),
            "generalized_monotone": (1.000, 0.739, 0.974, 0.948, 47),
            "log_todim": (1.000, 0.950, 0.995, 0.990, 5),
            "exptodim": (1.000, 0.880, 0.988, 0.976, 12),
        }
        for method, values in expected.items():
            row = self.summary.loc[method]
            actual = (
                round(row.top_one_stability, 3),
                round(row.full_ranking_stability, 3),
                round(row.mean_spearman, 3),
                round(row.mean_kendall, 3),
                int(row.changed_rankings),
            )
            self.assertEqual(actual, values)

    def test_method_comparison_has_no_duplicate_power_entry(self):
        comparison = compare_methods(self.data)
        self.assertEqual(comparison["method"].tolist(), list(METHOD_ORDER))
        self.assertEqual(len(comparison), 5)
        logarithmic = comparison.set_index("method").loc["log_todim"]
        self.assertAlmostEqual(logarithmic.spearman_vs_reference, 1.0)
        self.assertAlmostEqual(logarithmic.kendall_tau_vs_reference, 1.0)

    def test_synthetic_cases_are_loadable(self):
        for case_name in (
            "Balanced trade-offs",
            "Loss-aversion sensitive",
            "Close-score rho test",
            "Weight-sensitive case",
            "Benefit/cost sense test",
        ):
            case = load_builtin_case(case_name)
            self.assertGreaterEqual(case.matrix.shape[0], 2)
            self.assertEqual(case.matrix.shape[1], len(case.criteria))
            self.assertAlmostEqual(case.weights.sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
