# V8.2 PBMC size-matched random ADT null

This artifact evaluates frozen MOSAIC-N v33 checkpoints under legacy v33
panel masks plus random masks with exactly the same feature counts as the
six-feature memory-marker and fifteen-feature T-cell targeted masks.

Masks are generated from the eligible protein feature names and declared
donor/checkpoint seed only. They are fixed across cells within a unit and
never use test labels, predictions, thresholds, or test errors. Metrics are
computed on labels observed in the corresponding training donors, matching
the v33 known-label policy. This is a protocol/interpretability null, not a
predefined performance-improvement claim.

The raw run is reproducible with the command recorded in `config.yaml` and
the unit-level provenance is recorded in `split_seed_metadata.json`.
