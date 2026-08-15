from pathlib import Path

from audit_v33_preflight import evaluate_preflight, write_outputs


def test_evaluate_preflight_distinguishes_ready_and_blocked_paths(tmp_path: Path) -> None:
    required = {
        "pbmc_gene": tmp_path / "pbmc_gene.h5ad",
        "pbmc_protein": tmp_path / "pbmc_protein.h5ad",
        "pdc101": tmp_path / "pdc101.h5ad",
        "mmochi_result": tmp_path / "mmochi.csv",
    }
    for path in required.values():
        path.touch()

    frame = evaluate_preflight(
        required_paths=required,
        gpu_name="NVIDIA GeForce RTX 4090",
        gpu_memory_free_mb=24000,
        disk_free_gb=37.0,
        conda_envs={"TOSICA", "MMoCHi"},
    )

    assert set(frame["status"]) == {"pass"}
    assert set(required).issubset(set(frame["check"]))
    assert frame.set_index("check").loc["gpu_memory", "status"] == "pass"
    assert frame.set_index("check").loc["data_disk", "status"] == "pass"


def test_evaluate_preflight_flags_missing_environment_and_low_disk(tmp_path: Path) -> None:
    frame = evaluate_preflight(
        required_paths={"missing": tmp_path / "missing.file"},
        gpu_name="",
        gpu_memory_free_mb=0,
        disk_free_gb=3.0,
        conda_envs={"TOSICA"},
    )

    indexed = frame.set_index("check")
    assert indexed.loc["missing", "status"] == "blocker"
    assert indexed.loc["gpu_memory", "status"] == "blocker"
    assert indexed.loc["data_disk", "status"] == "blocker"
    assert indexed.loc["env_MMoCHi", "status"] == "blocker"


def test_write_outputs_creates_csv_and_json(tmp_path: Path) -> None:
    frame = evaluate_preflight(
        required_paths={},
        gpu_name="RTX",
        gpu_memory_free_mb=16000,
        disk_free_gb=20.0,
        conda_envs={"TOSICA", "MMoCHi"},
    )
    csv_path, json_path = write_outputs(frame, tmp_path)

    assert csv_path.exists()
    assert json_path.exists()
    assert '"blocker_count": 0' in json_path.read_text(encoding="utf-8")

