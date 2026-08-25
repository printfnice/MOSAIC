#!/usr/bin/env python
"""Protocol tests for the V8.2 size-matched random ADT null evaluator."""

from __future__ import annotations

import unittest

import numpy as np

from run_size_matched_random_adt_null import (
    _summary,
    build_scenarios,
    validate_scenario_masks,
)


class SizeMatchedRandomAdtNullTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = [
            "CD45RA",
            "CD45RO",
            "CD27",
            "CD95",
            "CD28",
            "CD127",
            "CD3-1",
            "CD4-1",
            "CD8",
            "TCR-1",
            "CD19",
            "CD14",
        ]

    def test_size_matched_masks_equal_targeted_counts(self) -> None:
        scenarios = build_scenarios(self.features, donor="P1", seed=41)
        memory_count = int(scenarios["marker_memory"]["mask"].sum())
        tcell_count = int(scenarios["marker_tcell"]["mask"].sum())
        self.assertEqual(memory_count, 6)
        self.assertEqual(tcell_count, 7)
        self.assertEqual(
            int(scenarios["random_size_matched_6"]["mask"].sum()),
            memory_count,
        )
        self.assertEqual(
            int(scenarios["random_size_matched_15"]["mask"].sum()),
            tcell_count,
        )

    def test_random_mask_is_reproducible_and_unit_specific(self) -> None:
        first = build_scenarios(self.features, donor="P1", seed=41)
        second = build_scenarios(self.features, donor="P1", seed=41)
        other = build_scenarios(self.features, donor="P2", seed=41)
        np.testing.assert_array_equal(
            first["random_size_matched_6"]["mask"],
            second["random_size_matched_6"]["mask"],
        )
        self.assertFalse(
            np.array_equal(
                first["random_size_matched_6"]["mask"],
                other["random_size_matched_6"]["mask"],
            )
        )

    def test_mask_validation_rejects_label_dependent_manifest(self) -> None:
        scenarios = build_scenarios(self.features, donor="P1", seed=41)
        validate_scenario_masks(scenarios, self.features)
        scenarios["random_size_matched_6"]["mask_seed"] = "selected_using_test_label"
        with self.assertRaises(ValueError):
            validate_scenario_masks(scenarios, self.features)

    def test_summary_contains_balanced_accuracy(self) -> None:
        y_true = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray(
            [
                [0.9, 0.1],
                [0.6, 0.4],
                [0.2, 0.8],
                [0.7, 0.3],
            ],
            dtype=float,
        )
        summary = _summary(y_true, probabilities)
        self.assertIn("balanced_accuracy", summary)
        self.assertAlmostEqual(summary["balanced_accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
