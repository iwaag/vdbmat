"""Step 9 analytic and end-to-end verification for the Phase 1 workflow."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vbdmat.core import MaterialMixtureVolume, OpticalPropertyVolume
from vbdmat.fixtures import (
    homogeneous_transparent,
    transparent_opaque_interface,
    two_material_mixture_ramp,
    write_phase1_fixtures,
)
from vbdmat.io import (
    read_material_label_manifest,
    read_optical_region,
    read_volume,
    write_volume,
)
from vbdmat.optics import map_material_volume_to_optical, phase0_provisional_mapping
from vbdmat.pipeline import PipelineConfig, run_pipeline, zarr_store_sha256


def _run_cli(*arguments: object, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["vbdmat", *(str(item) for item in arguments)],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "VBDMAT_DEBUG": "0"},
    )


def _assert_optical_arrays_equal(
    left: OpticalPropertyVolume, right: OpticalPropertyVolume
) -> None:
    assert left.geometry == right.geometry
    assert left.optical_basis == right.optical_basis
    for field in ("sigma_a", "sigma_s", "g", "ior"):
        np.testing.assert_array_equal(getattr(left, field), getattr(right, field))


def test_mixture_fraction_totals_survive_persistence_and_mapping(
    tmp_path: Path,
) -> None:
    fixture = two_material_mixture_ramp()
    assert isinstance(fixture.volume, MaterialMixtureVolume)
    material_path = tmp_path / "mixture.zarr"
    optical_path = tmp_path / "optical.zarr"

    write_volume(material_path, fixture.volume)
    restored = read_volume(material_path)
    assert isinstance(restored, MaterialMixtureVolume)
    np.testing.assert_array_equal(restored.fractions, fixture.volume.fractions)
    np.testing.assert_array_equal(restored.material_ids, fixture.volume.material_ids)
    np.testing.assert_allclose(restored.fractions.sum(axis=-1), 1.0, atol=1e-6)

    expected_totals = dict(fixture.manifest.material_fraction_totals)
    actual_totals = restored.fractions.sum(axis=(0, 1, 2), dtype=np.float64)
    for index, material_id in enumerate(restored.material_ids):
        assert actual_totals[index] == pytest.approx(expected_totals[int(material_id)])

    mapped = map_material_volume_to_optical(restored, phase0_provisional_mapping())
    write_volume(optical_path, mapped)
    persisted = read_volume(optical_path)
    assert isinstance(persisted, OpticalPropertyVolume)
    _assert_optical_arrays_equal(persisted, mapped)


def test_homogeneous_mapping_remains_exact_through_both_zarr_stages(
    tmp_path: Path,
) -> None:
    fixture = homogeneous_transparent()
    material_path = tmp_path / "material.zarr"
    optical_path = tmp_path / "optical.zarr"
    write_volume(material_path, fixture.volume)
    material = read_volume(material_path)
    optical = map_material_volume_to_optical(material, phase0_provisional_mapping())
    write_volume(optical_path, optical)
    restored = read_volume(optical_path)
    assert isinstance(restored, OpticalPropertyVolume)

    assert np.all(restored.sigma_a == np.asarray((2.0, 1.0, 0.5)))
    assert np.all(restored.sigma_s == 0.0)
    assert np.all(restored.g == 0.0)
    assert np.all(restored.ior == np.float32(1.48))


def test_relocating_manifest_and_payload_preserves_scientific_input(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "generated"
    write_phase1_fixtures(generated)
    volumes = []
    for name in ("location-a", "nested/location-b"):
        destination = tmp_path / name
        destination.mkdir(parents=True)
        for filename in (
            "window_coupon.voxels.json",
            "window_coupon.material_id.npy",
        ):
            shutil.copy2(generated / filename, destination / filename)
        volumes.append(
            read_material_label_manifest(destination / "window_coupon.voxels.json")
        )

    left, right = volumes
    assert left.geometry == right.geometry
    assert left.palette == right.palette
    assert left.provenance == right.provenance
    np.testing.assert_array_equal(left.material_id, right.material_id)


def test_optical_reads_are_independent_of_requested_chunk_partition(
    tmp_path: Path,
) -> None:
    optical = map_material_volume_to_optical(
        transparent_opaque_interface().volume, phase0_provisional_mapping()
    )
    path = tmp_path / "optical.zarr"
    write_volume(path, optical)
    whole = read_optical_region(
        path, (slice(None), slice(None), slice(None))
    )

    for field in ("sigma_a", "sigma_s", "g", "ior"):
        expected = getattr(whole, field)
        stitched = np.empty_like(expected)
        for start, stop in ((0, 1), (1, 5), (5, 6)):
            region = read_optical_region(
                path, (slice(None), slice(None), slice(start, stop))
            )
            stitched[:, :, start:stop] = getattr(region, field)
        np.testing.assert_array_equal(stitched, expected)


def test_cli_run_and_pipeline_api_produce_identical_canonical_artifacts(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    write_phase1_fixtures(inputs)
    config = PipelineConfig(
        input_kind="direct-voxel",
        input_path="inputs/window_coupon.voxels.json",
        output_path="cli-run",
    )
    config_path = tmp_path / "run.json"
    config_path.write_text(config.canonical_json(), encoding="utf-8")

    api_result = run_pipeline(config, base_dir=str(tmp_path))
    api_bundle = tmp_path / "api-run"
    api_result.output_path.rename(api_bundle)

    cli = _run_cli("run", config_path, "--json", cwd=tmp_path)
    assert cli.returncode == 0, cli.stderr
    cli_document: dict[str, Any] = json.loads(cli.stdout)
    cli_bundle = Path(cli_document["path"])
    assert cli_document["config_digest"] == api_result.config_digest
    assert cli_document["run_id"] == api_result.run_id
    assert zarr_store_sha256(cli_bundle / "material.zarr") == zarr_store_sha256(
        api_bundle / "material.zarr"
    )
    assert zarr_store_sha256(cli_bundle / "optical.zarr") == zarr_store_sha256(
        api_bundle / "optical.zarr"
    )


def test_bundle_validation_attributes_corruption_and_missing_stage(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    write_phase1_fixtures(inputs)
    first = run_pipeline(
        PipelineConfig(
            input_kind="direct-voxel",
            input_path="window_coupon.voxels.json",
            output_path="corrupt-run",
        ),
        base_dir=str(inputs),
    ).output_path
    (first / "config.json").write_text(
        (first / "config.json").read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    corrupted = _run_cli("validate", first, "--json", cwd=tmp_path)
    assert corrupted.returncode == 3
    assert "checksum mismatch: config.json" in json.loads(corrupted.stdout)["message"]

    second = run_pipeline(
        PipelineConfig(
            input_kind="direct-voxel",
            input_path="window_coupon.voxels.json",
            output_path="missing-stage-run",
        ),
        base_dir=str(inputs),
    ).output_path
    shutil.rmtree(second / "optical.zarr")
    missing = _run_cli("validate", second, "--json", cwd=tmp_path)
    assert missing.returncode == 4
    assert "cannot open Zarr v3 group" in json.loads(missing.stdout)["message"]
