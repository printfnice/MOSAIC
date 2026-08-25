#!/usr/bin/env python
"""Tests for the secondary-cohort audit-transfer protocol."""

from __future__ import annotations

import unittest

import pandas as pd

from run_secondary_cohort_audit_transfer import (
    classify_cohort_identity,
    summarize_transfer,
    validate_external_predictions,
)


class SecondaryCohortAuditTransferTests(unittest.TestCase):
    def test_external_labels_cannot_select_the_policy(self) -> None:
        frame = pd.DataFrame(
            {
                "cell_id": ["a", "b", "c"],
                "mask_condition": ["full", "full", "random_50"],
                "label": ["A", "B", "A"],
                "true_l1": ["T", "T", "T"],
                "prediction": ["A", "A", "A"],
                "pred_l1": ["T", "T", "T"],
                "decision": ["l3_accept", "l3_accept", "l3_accept"],
                "confidence": [0.4, 0.3, 0.2],
                "parent_safe": [True, False, True],
                "external_label_used_for_policy_selection": ["no", "no", "no"],
            }
        )
        full = validate_external_predictions(frame)
        self.assertEqual(len(full), 2)
        self.assertTrue((full["external_label_used_for_policy_selection"] == "no").all())

    def test_transfer_summary_reports_acceptance_and_parent_safety(self) -> None:
        frame = pd.DataFrame(
            {
                "label": ["A", "B", "C", "D"],
                "true_l1": ["T", "T", "B", "B"],
                "prediction": ["A", "C", "B", "D"],
                "pred_l1": ["T", "B", "B", "B"],
                "decision": ["l3_accept", "fallback_l2", "l3_accept", "fallback_l2"],
                "parent_safe": [True, False, True, True],
            }
        )
        summary = summarize_transfer(frame)
        self.assertEqual(summary["n_eval"], 4)
        self.assertEqual(summary["n_accepted"], 2)
        self.assertAlmostEqual(summary["accept_rate"], 0.5)
        self.assertAlmostEqual(summary["accepted_l3_accuracy"], 0.5)
        self.assertAlmostEqual(summary["parent_safe_rate"], 0.75)

    def test_same_donor_set_is_not_independent_external_evidence(self) -> None:
        status = classify_cohort_identity(
            dataset_name="GSE164378 5P",
            observed_donors={"P1", "P2", "P3"},
            main_donors={"P1", "P2", "P3"},
        )
        self.assertEqual(status["independent_external_cohort"], False)
        self.assertEqual(status["transfer_type"], "same-study assay-cohort replication")


if __name__ == "__main__":
    unittest.main()
