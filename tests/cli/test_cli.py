"""Subprocess coverage for the installed Phase 1 CLI contract (ADR-008)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from vbdmat.fixtures import write_phase1_fixtures
from vbdmat.io import read_material_label_manifest, read_volume, write_volume
from vbdmat.pipeline import zarr_store_sha256


@pytest.fixture
def inputs(tmp_path: Path) -> Path:
    path = tmp_path / "inputs with spaces"
    write_phase1_fixtures(path)
    return path


def _run(
    *arguments: object, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["vbdmat", *(str(item) for item in arguments)],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "VBDMAT_DEBUG": "0"},
    )


def _json(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return json.loads(result.stdout)


def test_import_json_paths_with_spaces_and_api_equivalence(
    inputs: Path, tmp_path: Path
) -> None:
    output = tmp_path / "output with spaces" / "material.zarr"
    result = _run(
        "import-voxels",
        inputs / "window_coupon.voxels.json",
        output,
        "--json",
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert _json(result)["asset_type"] == "material-label"
    expected = tmp_path / "api.zarr"
    write_volume(
        expected, read_material_label_manifest(inputs / "window_coupon.voxels.json")
    )
    assert zarr_store_sha256(output) == zarr_store_sha256(expected)


def test_import_convert_inspect_and_validate(inputs: Path, tmp_path: Path) -> None:
    material = tmp_path / "wedge.zarr"
    optical = tmp_path / "optical.zarr"
    imported = _run(
        "import-voxels",
        inputs / "stepped_wedge.voxels.json",
        material,
        "--json",
    )
    assert imported.returncode == 0
    assert _json(imported)["asset_type"] == "material-label"

    converted = _run("convert", material, optical, "--json")
    assert converted.returncode == 0
    assert read_volume(optical).asset_type.value == "optical-property"

    inspected = _run("inspect", optical, "--json")
    document = _json(inspected)
    assert inspected.returncode == 0
    assert document["geometry"]["storage_order"] == "zyx"
    assert document["geometry"]["length_unit"] == "m"
    assert document["optical_basis"]["identifier"] == "linear-srgb-effective-v1"
    assert document["calibration"] == "provisional-uncalibrated"

    validated = _run("validate", material, "--json")
    assert validated.returncode == 0
    assert _json(validated)["validation"] == {
        "mode": "full-read",
        "status": "ok",
    }


def test_run_and_bundle_inspection(inputs: Path, tmp_path: Path) -> None:
    config = json.loads(
        (Path("examples/phase1/configs/window_coupon.run.json")).read_text()
    )
    config["input"]["path"] = str(inputs / "window_coupon.voxels.json")
    config["output"]["path"] = str(tmp_path / "run output")
    config_path = tmp_path / "config with spaces.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    run = _run("run", config_path, "--json", cwd=tmp_path)
    assert run.returncode == 0
    bundle = Path(_json(run)["path"])
    inspected = _run("inspect", bundle, "--json", cwd=tmp_path)
    assert inspected.returncode == 0
    assert _json(inspected)["asset_kind"] == "run-bundle"
    assert len(_json(inspected)["assets"]) == 2


def test_overwrite_requires_authorization(inputs: Path, tmp_path: Path) -> None:
    output = tmp_path / "material.zarr"
    first = _run("import-voxels", inputs / "window_coupon.voxels.json", output)
    refused = _run("import-voxels", inputs / "window_coupon.voxels.json", output)
    replaced = _run(
        "import-voxels",
        inputs / "window_coupon.voxels.json",
        output,
        "--overwrite",
    )

    assert first.returncode == 0
    assert refused.returncode == 2
    assert refused.stdout == ""
    assert "refusing to overwrite" in refused.stderr
    assert replaced.returncode == 0


def test_documented_exit_categories(inputs: Path, tmp_path: Path) -> None:
    usage = _run("import-voxels", inputs / "window_coupon.voxels.json")
    bad_manifest = tmp_path / "bad.voxels.json"
    bad_manifest.write_text(
        json.dumps({"format": "not-vbdmat"}), encoding="utf-8"
    )
    validation = _run("import-voxels", bad_manifest, tmp_path / "bad-out.zarr")
    io_error = _run(
        "import-voxels",
        Path(
            "examples/phase1/inputs/invalid/window_coupon.bad_checksum.voxels.json"
        ).resolve(),
        tmp_path / "bad.zarr",
    )
    material = tmp_path / "material.zarr"
    optical = tmp_path / "optical.zarr"
    assert (
        _run("import-voxels", inputs / "window_coupon.voxels.json", material).returncode
        == 0
    )
    assert _run("convert", material, optical).returncode == 0
    conversion = _run("convert", optical, tmp_path / "not-material.zarr")
    optional = _run("export", "openvdb", optical, tmp_path / "export")

    assert usage.returncode == 2
    assert validation.returncode == 3
    assert io_error.returncode == 4
    assert conversion.returncode == 5
    assert optional.returncode == 6
    for result in (usage, validation, io_error, conversion, optional):
        assert "Traceback" not in result.stderr


def test_error_json_is_parseable_and_diagnostics_stay_on_stderr(
    tmp_path: Path,
) -> None:
    result = _run(
        "import-voxels", tmp_path / "missing.json", tmp_path / "out.zarr", "--json"
    )
    document = _json(result)

    assert result.returncode == 4
    assert document["status"] == "error"
    assert document["exit_code"] == 4
    assert "file not found" in result.stderr

    usage = _run("import-voxels", "manifest.json", "--json")
    assert usage.returncode == 2
    assert _json(usage)["exit_code"] == 2


def test_help_documents_contract_and_console_entry_point() -> None:
    top = _run("--help")
    convert = _run("convert", "--help")

    assert top.returncode == 0
    assert "provisional and uncalibrated" in top.stdout
    assert "physical print predictions" in top.stdout
    assert "voxelize" not in top.stdout  # removed from the core CLI (ADR-009)
    assert "vbdmat.voxels manifest" in top.stdout
    assert "provisional" in convert.stdout
    assert "uncalibrated" in convert.stdout
