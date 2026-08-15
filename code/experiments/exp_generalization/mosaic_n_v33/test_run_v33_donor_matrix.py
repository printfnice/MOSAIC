from pathlib import Path

from run_v33_donor_matrix import (
    FORMAL_METHODS,
    build_mlp_command,
    build_mosaic_command,
    expected_run_keys,
    run_is_complete,
)


def test_expected_run_keys_cover_all_donors_seeds_and_methods() -> None:
    keys = expected_run_keys(
        donors=["P1", "P2"],
        seeds=[41, 42, 43],
        methods=FORMAL_METHODS,
    )
    assert len(keys) == 24
    assert ("P1", 42, "mosaic_full") in keys
    assert ("P2", 43, "mosaic_no_kd") in keys


def test_mlp_teacher_command_exports_train_probabilities(tmp_path: Path) -> None:
    command = build_mlp_command(
        python=Path("/env/bin/python"),
        cache_path=tmp_path / "cache.npz",
        out_dir=tmp_path / "mlp_seed42",
        seed=42,
        epochs=35,
        save_teacher=True,
    )
    text = " ".join(command)
    assert "--save-train-probabilities" in command
    assert "--label-smoothing 0.03" in text
    assert "--cache-path" in command


def test_mosaic_commands_lock_full_and_ablation_settings(tmp_path: Path) -> None:
    common = {
        "python": Path("/env/bin/python"),
        "cache_path": tmp_path / "cache.npz",
        "teacher_path": tmp_path / "teacher.csv",
        "seed": 42,
        "epochs": 35,
        "selection_reference_accuracy": 0.7,
        "selection_reference_weighted_f1": 0.6,
    }
    full = " ".join(
        build_mosaic_command(
            out_dir=tmp_path / "full",
            method="mosaic_full",
            **common,
        )
    )
    no_hsr = " ".join(
        build_mosaic_command(
            out_dir=tmp_path / "no_hsr",
            method="mosaic_no_hsr",
            **common,
        )
    )
    no_kd = " ".join(
        build_mosaic_command(
            out_dir=tmp_path / "no_kd",
            method="mosaic_no_kd",
            **common,
        )
    )

    assert "--hsr-mode hierarchy" in full
    assert "--hsr-gate-floor 0.02" in full
    assert "--hsr-gate-max 0.2" in full
    assert "--distill-alpha 0.05" in full
    assert "--hsr-mode off" in no_hsr
    assert "--hsr-loss-weight 0.0" in no_hsr
    assert "--distill-alpha 0.0" in no_kd


def test_run_is_complete_requires_reproducibility_unit(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    for name in [
        "config.json",
        "model.pt",
        "results_summary.csv",
        "predictions_full.csv",
        "probabilities_test.csv",
        "per_class_metrics.csv",
        "run.log",
    ]:
        (out_dir / name).touch()
    assert run_is_complete(out_dir, method="mosaic_full")

    (out_dir / "per_class_metrics.csv").unlink()
    assert not run_is_complete(out_dir, method="mosaic_full")
