# Packaged Test-Data Demo

This directory contains a small, checkpoint-free test-data path for verifying
installation, input parsing, output generation, and the released audit schema.
It is not a scientific benchmark and does not contain raw participant data.

## Inputs

- `inputs.npz`: `gene` with shape `(10, 3000)`, `protein` with shape `(10, 224)`,
  and `cell_id` with shape `(10,)`.
- `input_manifest.csv`: per-array shape, dtype, preprocessing status, and
  release checksum metadata.
- `audit_record_generated.csv`: the packaged audit record that is validated by
  the demo. It is included as a release artifact and is not generated from a
  public checkpoint.

## Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python demo/run_demo.py --package . --output-dir demo/output
```

The command exits with status 0 when the input and audit schemas are coherent.
It writes `demo/output/audit_record_validated.csv` and
`demo/output/demo_summary.json`. The CSV contains the cell ID, final and branch
predictions, confidence/uncertainty, branch margins, modality weights, branch
conflict, HSR gate, and HSR delta norm. The JSON records the input shapes,
output path, CPU device, and explicit flags that no checkpoint or scientific
inference was used.

The demo validates a precomputed audit record against the packaged test inputs;
it does not train a model, run trained-model inference, or reproduce manuscript
performance. Full scientific reruns require the public datasets and trained
artifacts described in the repository documentation.
