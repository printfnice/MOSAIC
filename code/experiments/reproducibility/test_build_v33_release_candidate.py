import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from experiments.reproducibility import build_v33_release_candidate as release_module
from experiments.reproducibility.build_v33_release_candidate import (
    DEFAULT_COMMANDS,
    ReleaseBlockedError,
    build_release_candidate,
    default_source_groups,
)


def _write(path: Path, text: str = "fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    _write(root / "experiments/run_v33.py", "print('v33')\n")
    _write(root / "configs/models/v33.yaml", "name: mosaic_n_v33\n")
    _write(root / "configs/splits/nested_lodo/P1.csv", "cell_id,split\n1,test\n")
    _write(root / "results/tables/evidence.csv", "metric,value\naccuracy,0.9\n")
    _write(root / "results/models/model.pt", "must not be copied\n")
    _write(root / "data/raw.h5ad", "must not be copied\n")
    _write(root / "results/tables/oversized.csv", "x" * 2048)
    return root


def test_missing_required_artifact_blocks_release(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)

    with pytest.raises(ReleaseBlockedError, match="missing required evidence"):
        build_release_candidate(
            root=root,
            output_dir=tmp_path / "release",
            required_artifacts=["results/tables/absent.csv"],
            source_groups={"code": ["experiments/run_v33.py"]},
            command_lines=["python experiments/run_v33.py"],
            environment_metadata={"python_version": "fixture"},
        )

    assert not (tmp_path / "release").exists()


def test_release_collects_lightweight_material_and_records_exclusions(
    tmp_path: Path,
) -> None:
    root = _fixture_root(tmp_path)
    output = tmp_path / "release"

    build_release_candidate(
        root=root,
        output_dir=output,
        required_artifacts=["results/tables/evidence.csv"],
        source_groups={
            "code": ["experiments/run_v33.py"],
            "config": ["configs/models/v33.yaml"],
            "manifest": ["configs/splits/nested_lodo/P1.csv"],
            "evidence": [
                "results/tables/evidence.csv",
                "results/tables/oversized.csv",
            ],
            "forbidden": ["results/models/model.pt", "data/raw.h5ad"],
        },
        command_lines=["python experiments/run_v33.py --all"],
        environment_metadata={"python_version": "fixture", "cuda": "fixture"},
        max_file_bytes=1024,
    )

    assert (output / "code/experiments/run_v33.py").is_file()
    assert (output / "config/configs/models/v33.yaml").is_file()
    assert (output / "manifest/configs/splits/nested_lodo/P1.csv").is_file()
    assert (output / "evidence/results/tables/evidence.csv").is_file()
    assert not list(output.rglob("*.pt"))
    assert not list(output.rglob("*.h5ad"))
    assert not (output / "evidence/results/tables/oversized.csv").exists()

    exclusions = pd.read_csv(output / "excluded_files.csv")
    assert set(exclusions["reason"]) == {
        "forbidden_path_or_suffix",
        "exceeds_size_limit",
    }
    assert {"commands.txt", "environment_metadata.json", "release_manifest.csv"} <= {
        path.name for path in output.iterdir()
    }
    environment = json.loads(
        (output / "environment_metadata.json").read_text(encoding="utf-8")
    )
    assert environment["python_version"] == "fixture"


def test_checksum_file_covers_and_verifies_packaged_payload(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    output = tmp_path / "release"
    build_release_candidate(
        root=root,
        output_dir=output,
        required_artifacts=["results/tables/evidence.csv"],
        source_groups={
            "code": ["experiments/run_v33.py"],
            "config": ["configs/models/v33.yaml"],
            "manifest": ["configs/splits/nested_lodo/P1.csv"],
            "evidence": ["results/tables/evidence.csv"],
        },
        command_lines=["python experiments/run_v33.py"],
        environment_metadata={"python_version": "fixture"},
    )

    manifest = pd.read_csv(output / "release_manifest.csv")
    assert set(manifest["category"]) == {
        "code",
        "config",
        "manifest",
        "evidence",
    }
    assert manifest["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()

    checksum_lines = (
        output / "CHECKSUMS.sha256"
    ).read_text(encoding="utf-8").splitlines()
    checksums = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in checksum_lines
    }
    assert "CHECKSUMS.sha256" not in checksums
    assert {
        "commands.txt",
        "environment_metadata.json",
        "release_manifest.csv",
    } <= set(checksums)
    for relative_path, expected in checksums.items():
        payload = (output / relative_path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected


def test_missing_required_code_source_blocks_release(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    with pytest.raises(ReleaseBlockedError, match="missing required source files"):
        build_release_candidate(
            root=root,
            output_dir=tmp_path / "release",
            required_artifacts=["results/tables/evidence.csv"],
            source_groups={
                "code": ["experiments/absent.py"],
                "config": ["configs/models/v33.yaml"],
                "manifest": ["configs/splits/nested_lodo/P1.csv"],
                "evidence": ["results/tables/evidence.csv"],
            },
            command_lines=["python experiments/absent.py"],
            environment_metadata={"python_version": "fixture"},
        )


def test_license_is_packaged_at_release_root(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write(root / "LICENSE", "fixture license\n")
    output = tmp_path / "release"
    build_release_candidate(
        root=root,
        output_dir=output,
        required_artifacts=["results/tables/evidence.csv"],
        source_groups={
            "code": ["experiments/run_v33.py"],
            "config": ["configs/models/v33.yaml"],
            "manifest": ["configs/splits/nested_lodo/P1.csv"],
            "evidence": ["results/tables/evidence.csv"],
            "metadata": ["LICENSE"],
        },
        command_lines=["python experiments/run_v33.py"],
        environment_metadata={"python_version": "fixture"},
    )
    assert (output / "LICENSE").read_text(encoding="utf-8") == "fixture license\n"


def test_default_pdc_reproduction_command_uses_supported_cli() -> None:
    command = next(value for value in DEFAULT_COMMANDS if "pdc101_mosaic_n.py" in value)
    assert "--all" not in command


def test_default_release_includes_synthetic_smoke_testdata(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(
        root / "experiments/reproducibility/testdata/v33_synthetic_smoke.csv",
        "cell_id,split,label,rna_1,adt_1\nc1,train,A,1.0,0.0\n",
    )

    groups = default_source_groups(root)

    assert groups["testdata"] == [
        "experiments/reproducibility/testdata/v33_synthetic_smoke.csv"
    ]


def test_default_commands_include_release_smoke_test() -> None:
    command = next(
        command
        for command in DEFAULT_COMMANDS
        if "v33_release_smoke_test.py" in command
    )
    assert command.startswith("OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ")
    assert "--release-dir output/release/mosaic_n_v33" in command


def test_environment_metadata_tolerates_unexecutable_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(command, **kwargs):
        if command[0] == "git":
            return SimpleNamespace(stdout="fixture\n")
        raise OSError(8, "Exec format error", command[0])

    monkeypatch.setattr(release_module.subprocess, "run", fake_run)

    metadata = release_module.collect_environment_metadata(tmp_path)

    assert metadata["gpu"] == []
