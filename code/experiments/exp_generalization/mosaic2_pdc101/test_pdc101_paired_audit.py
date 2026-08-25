#!/usr/bin/env python
"""Tests for the PDC101 paired audit protocol."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from run_pdc101_paired_audit import (
    boundary_status,
    mcnemar_exact_pvalue,
    run_audit,
    summarize_gate_distribution,
    validate_prediction_frames,
)


class PDC101PairedAuditTests(unittest.TestCase):
    def test_exact_mcnemar_uses_only_discordant_pairs(self) -> None:
        self.assertAlmostEqual(mcnemar_exact_pvalue(1, 3), 0.625)
        self.assertAlmostEqual(mcnemar_exact_pvalue(0, 0), 1.0)

    def test_prediction_join_requires_unique_matching_labels(self) -> None:
        mmochi = pd.DataFrame(
            {
                "MMoCHi_obs_names": ["a", "b"],
                "sort_label": ["cd4_cm", "cd8_cm"],
                "mmochi_prediction": ["cd4_cm", "cd8_emra"],
                "mmochi_certainty": [0.9, 0.8],
                "external_holdout": [True, True],
            }
        )
        mosaic = pd.DataFrame(
            {
                "cell_id": ["a", "b"],
                "sort_label": ["cd4_cm", "cd8_cm"],
                "prediction": ["cd4_cm", "cd8_cm"],
                "adt_gate": [0.4, 0.6],
            }
        )
        merged = validate_prediction_frames(mmochi, mosaic)
        self.assertEqual(len(merged), 2)
        self.assertEqual(int(merged["mosaic_correct"].sum()), 2)
        self.assertEqual(int(merged["mmochi_correct"].sum()), 1)

    def test_boundary_status_is_predeclared_by_class_family(self) -> None:
        self.assertEqual(boundary_status("cd4_cm"), "central_memory")
        self.assertEqual(boundary_status("cd8_cm"), "central_memory")
        self.assertEqual(boundary_status("cd8_emra"), "other")

    def test_gate_summary_reports_count_and_location_statistics(self) -> None:
        frame = pd.DataFrame(
            {
                "sort_label": ["cd4_cm", "cd8_cm", "cd8_emra"],
                "adt_gate": [0.2, 0.6, 0.9],
                "mosaic_correct": [True, False, True],
                "boundary_status": [
                    "central_memory",
                    "central_memory",
                    "other",
                ],
            }
        )
        summary = summarize_gate_distribution(frame)
        self.assertEqual(set(summary["group"]), {"central_memory", "other"})
        self.assertEqual(int(summary.loc[summary.group == "other", "n"].iloc[0]), 1)
        self.assertAlmostEqual(
            float(summary.loc[summary.group == "central_memory", "mean"].iloc[0]),
            0.4,
        )

    def test_full_audit_summary_exposes_run_log_keys(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            mmochi_path = root / "mmochi.csv"
            mosaic_path = root / "mosaic.csv"
            out_dir = root / "out"
            pd.DataFrame(
                {
                    "MMoCHi_obs_names": ["a", "b"],
                    "sort_label": ["cd4_cm", "cd8_cm"],
                    "mmochi_prediction": ["cd4_cm", "cd8_emra"],
                    "mmochi_certainty": [0.9, 0.8],
                    "external_holdout": [True, True],
                }
            ).to_csv(mmochi_path, index=False)
            pd.DataFrame(
                {
                    "cell_id": ["a", "b"],
                    "sort_label": ["cd4_cm", "cd8_cm"],
                    "prediction": ["cd4_cm", "cd8_cm"],
                    "adt_gate": [0.4, 0.6],
                }
            ).to_csv(mosaic_path, index=False)
            summary = run_audit(mmochi_path, mosaic_path, out_dir)
            self.assertEqual(summary["status"], "ready")
            self.assertEqual(summary["n_joined_pairs"], 2)


if __name__ == "__main__":
    unittest.main()
