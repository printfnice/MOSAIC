# V8.2 PDC101 paired prediction audit

This artifact compares the valid no-HTO MOSAIC-HPM-lite prediction file with
the official MMoCHi external-holdout prediction file on exactly the same
PDC101 cells. It performs a one-to-one cell-ID join, checks truth-label
agreement, and reports paired correctness counts and an exact two-sided
McNemar test.

The saved `adt_gate` is summarized within MOSAIC by boundary class and is not
treated as directly comparable to MMoCHi certainty. A cross-method modality
weight test is therefore marked `not_estimable`; no weight is reconstructed
from final predictions. Both source runs exclude HTO features as model input.
This is an audit of existing held-out predictions and does not retrain either
method or select a threshold using the holdout labels.
