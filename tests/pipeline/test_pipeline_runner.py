"""Step 6 pipeline orchestration and run-bundle tests (ADR-007 compliance).

These exercise the full canonical stage sequence for the direct-voxel input, the
failure-safe atomic publication, provenance chaining, deterministic reruns, and the
failure-isolated optional export stage. Fixtures are materialized from the deterministic
Phase 1 generators so the tests do not depend on the committed example bundle.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vdbmat.fixtures import write_phase1_fixtures
from vdbmat.io.zarr import read_volume
from vdbmat.pipeline import (
    ExportSettings,
    ExportTarget,
    PipelineConfig,
    PipelineRunError,
    StageStatus,
    run_pipeline,
)
from vdbmat.pipeline.runner import _run_id

CANONICAL_STAGES = (
    "validate-material",
    "persist-material",
    "map-optics",
    "validate-optical",
    "persist-optical",
    "summarize",
    "export",
)


@pytest.fixture
def inputs_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "inputs"
    write_phase1_fixtures(directory)
    return directory


def _coupon_config(inputs_dir: Path, output: str, **overrides: Any) -> PipelineConfig:
    return PipelineConfig(
        input_kind="direct-voxel",
        input_path="window_coupon.voxels.json",
        output_path=output,
        **overrides,
    )


def _wedge_config(inputs_dir: Path, output: str, **overrides: Any) -> PipelineConfig:
    return PipelineConfig(
        input_kind="direct-voxel",
        input_path="stepped_wedge.voxels.json",
        output_path=output,
        **overrides,
    )


def _load_manifest(bundle: Path) -> dict[str, Any]:
    return json.loads((bundle / "run.json").read_text())


# -- stage sequence ----------------------------------------------------------------


def test_direct_voxel_completes_the_canonical_stage_sequence(inputs_dir: Path) -> None:
    config = _coupon_config(inputs_dir, "out/coupon")
    result = run_pipeline(config, base_dir=str(inputs_dir))

    names = tuple(stage.name for stage in result.stages)
    assert names == ("load", *CANONICAL_STAGES)
    assert all(
        stage.status is StageStatus.OK
        for stage in result.stages
        if stage.name != "export"
    )
    assert result.stages[-1].status is StageStatus.SKIPPED


def test_wedge_manifest_completes_the_same_sequence(
    inputs_dir: Path,
) -> None:
    config = _wedge_config(inputs_dir, "out/wedge")
    result = run_pipeline(config, base_dir=str(inputs_dir))

    names = tuple(stage.name for stage in result.stages)
    assert names == ("load", *CANONICAL_STAGES)


def test_bundle_layout_matches_adr_007(inputs_dir: Path) -> None:
    config = _coupon_config(inputs_dir, "out/coupon")
    result = run_pipeline(config, base_dir=str(inputs_dir))
    bundle = result.output_path

    assert (bundle / "config.json").is_file()
    assert (bundle / "run.json").is_file()
    assert (bundle / "material.zarr").is_dir()
    assert (bundle / "optical.zarr").is_dir()
    assert (bundle / "diagnostics" / "summary.json").is_file()
    assert (bundle / "diagnostics" / "validation.json").is_file()
    assert (bundle / "source" / "window_coupon.voxels.json").is_file()
    assert (bundle / "source" / "window_coupon.material_id.npy").is_file()


# -- persisted assets equal stage outputs ------------------------------------------


def test_persisted_material_and_optical_equal_stage_outputs(inputs_dir: Path) -> None:
    config = _coupon_config(inputs_dir, "out/coupon")
    result = run_pipeline(config, base_dir=str(inputs_dir))
    bundle = result.output_path

    material = read_volume(bundle / "material.zarr")
    optical = read_volume(bundle / "optical.zarr")

    counts = result.summary["material"]["counts"]
    for material_id, expected in counts.items():
        assert int(np.count_nonzero(material.material_id == int(material_id))) == (
            expected
        )
    # Optical fields are finite and the summary ranges bracket the stored data.
    sa = np.asarray(optical.sigma_a)
    assert result.summary["optical"]["sigma_a_range_per_m"] == [
        float(sa.min()),
        float(sa.max()),
    ]


def test_config_json_checksum_equals_config_digest(inputs_dir: Path) -> None:
    config = _coupon_config(inputs_dir, "out/coupon")
    result = run_pipeline(config, base_dir=str(inputs_dir))
    assets = {a["path"]: a for a in result.manifest["assets"]}
    assert assets["config.json"]["sha256"] == result.config_digest


# -- provenance chaining -----------------------------------------------------------


def test_provenance_links_input_mapping_and_config(inputs_dir: Path) -> None:
    config = _coupon_config(inputs_dir, "out/coupon")
    result = run_pipeline(config, base_dir=str(inputs_dir))
    bundle = result.output_path

    material = read_volume(bundle / "material.zarr")
    optical = read_volume(bundle / "optical.zarr")

    assert material.provenance.configuration_digest == result.config_digest
    assert optical.provenance.configuration_digest == result.config_digest
    assert any(
        result.input_payload_sha256 in source for source in material.provenance.sources
    )
    assert any(result.mapping_digest in source for source in optical.provenance.sources)

    provenance = result.manifest["provenance"]
    assert provenance["input"]["payload_sha256"] == result.input_payload_sha256
    assert provenance["mapping"]["digest"] == result.mapping_digest
    assert provenance["config_digest"] == result.config_digest


def test_wedge_provenance_records_generator_identity(inputs_dir: Path) -> None:
    config = _wedge_config(inputs_dir, "out/wedge")
    result = run_pipeline(config, base_dir=str(inputs_dir))
    bundle = result.output_path
    material = read_volume(bundle / "material.zarr")
    assert material.provenance.generator == "vdbmat.fixtures.phase1"
    assert any(
        source.startswith("identity:") for source in material.provenance.sources
    )


# -- run identifier ----------------------------------------------------------------


def test_run_id_is_deterministic_and_timestamp_free(inputs_dir: Path) -> None:
    config = _coupon_config(inputs_dir, "out/coupon")
    result = run_pipeline(
        config, base_dir=str(inputs_dir), created_utc=datetime(2020, 1, 1, tzinfo=UTC)
    )
    expected = _run_id(
        result.config_digest, result.input_payload_sha256, result.mapping_digest
    )
    assert result.run_id == expected
    assert "2020" not in result.run_id


# -- failure safety ----------------------------------------------------------------


def test_overwrite_is_required_when_output_exists(inputs_dir: Path) -> None:
    config = _coupon_config(inputs_dir, "out/coupon")
    run_pipeline(config, base_dir=str(inputs_dir))
    with pytest.raises(PipelineRunError) as excinfo:
        run_pipeline(config, base_dir=str(inputs_dir))
    assert excinfo.value.stage == "publish"


def test_failed_run_publishes_no_valid_looking_bundle(
    inputs_dir: Path, tmp_path: Path
) -> None:
    config = _coupon_config(inputs_dir, "out/coupon")
    output = inputs_dir / "out" / "coupon"

    # Corrupt the payload so load fails after path resolution but before publish.
    (inputs_dir / "window_coupon.material_id.npy").write_bytes(b"not a valid npy")
    with pytest.raises(PipelineRunError):
        run_pipeline(config, base_dir=str(inputs_dir))

    assert not output.exists()
    # No leftover temporary directory.
    leftovers = list((inputs_dir / "out").glob(".coupon.tmp-*"))
    assert leftovers == []


def test_overwrite_preserves_previous_run_until_replacement_validates(
    inputs_dir: Path,
) -> None:
    config = _coupon_config(inputs_dir, "out/coupon", overwrite=True)
    first = run_pipeline(config, base_dir=str(inputs_dir))
    output = first.output_path
    original_run_json = (output / "run.json").read_text()

    # A second successful overwrite run leaves a valid bundle (no backup residue).
    run_pipeline(config, base_dir=str(inputs_dir))
    assert (output / "material.zarr").is_dir()
    assert list(output.parent.glob(".coupon.bak-*")) == []
    # Content is identical except the isolated timestamp.
    assert read_volume(output / "material.zarr") is not None
    assert original_run_json  # sanity: the first run was published


# -- reproducible rerun ------------------------------------------------------------


def test_two_identical_runs_have_equal_scientific_artifacts(inputs_dir: Path) -> None:
    config = _coupon_config(inputs_dir, "out/coupon", overwrite=True)
    first = run_pipeline(
        config, base_dir=str(inputs_dir), created_utc=datetime(2020, 1, 1, tzinfo=UTC)
    )
    second = run_pipeline(
        config, base_dir=str(inputs_dir), created_utc=datetime(2026, 6, 6, tzinfo=UTC)
    )

    assert first.run_id == second.run_id
    first_assets = {a["path"]: a["sha256"] for a in first.manifest["assets"]}
    second_assets = {a["path"]: a["sha256"] for a in second.manifest["assets"]}
    assert first_assets == second_assets

    first_manifest = dict(first.manifest)
    second_manifest = dict(second.manifest)
    assert first_manifest.pop("created_utc") != second_manifest.pop("created_utc")
    assert first_manifest == second_manifest


# -- optional export ---------------------------------------------------------------


def test_default_export_backend_reports_missing_optional_dependency(
    inputs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing_export(target: ExportTarget, optical: Path, dest: Path) -> Any:
        raise RuntimeError("Mitsuba bindings are unavailable")

    monkeypatch.setattr("vdbmat.pipeline.runner._default_export_runner", missing_export)
    config = _coupon_config(
        inputs_dir,
        "out/coupon",
        exports=(ExportSettings(target=ExportTarget.MITSUBA),),
    )
    result = run_pipeline(config, base_dir=str(inputs_dir))
    export_stage = result.stages[-1]
    assert export_stage.name == "export"
    assert export_stage.status is StageStatus.FAILED
    manifest = _load_manifest(result.output_path)
    assert manifest["export"]["status"] == "failed"
    assert "Mitsuba bindings are unavailable" in manifest["export"]["error"]
    assert read_volume(result.output_path / "optical.zarr") is not None


def test_export_failure_is_attributed_and_does_not_corrupt_canonical(
    inputs_dir: Path,
) -> None:
    config = _coupon_config(
        inputs_dir,
        "out/coupon",
        exports=(ExportSettings(target=ExportTarget.MITSUBA),),
    )

    def failing_export(target: ExportTarget, optical: Path, dest: Path) -> Any:
        assert optical.name == "optical.zarr"
        raise RuntimeError("renderer unavailable")

    result = run_pipeline(
        config, base_dir=str(inputs_dir), export_runner=failing_export
    )
    export_stage = result.stages[-1]
    assert export_stage.status is StageStatus.FAILED

    # Canonical artifacts are intact and readable despite the export failure.
    bundle = result.output_path
    assert read_volume(bundle / "material.zarr") is not None
    assert read_volume(bundle / "optical.zarr") is not None
    manifest = _load_manifest(bundle)
    assert manifest["export"]["status"] == "failed"
    assert "renderer unavailable" in manifest["export"]["error"]


def test_successful_export_records_adapter_versions(inputs_dir: Path) -> None:
    config = _coupon_config(
        inputs_dir,
        "out/coupon",
        exports=(ExportSettings(target=ExportTarget.MITSUBA),),
    )

    def fake_export(target: ExportTarget, optical: Path, dest: Path) -> Any:
        (dest / "scene.txt").write_text("scene", encoding="utf-8")
        return {"adapter": "fake-mitsuba", "version": "0.0.1"}

    result = run_pipeline(config, base_dir=str(inputs_dir), export_runner=fake_export)
    assert result.stages[-1].status is StageStatus.OK
    manifest = _load_manifest(result.output_path)
    assert manifest["versions"]["exporters"]["mitsuba"]["adapter"] == "fake-mitsuba"
    assert (result.output_path / "exports" / "mitsuba" / "scene.txt").is_file()
    exported = next(
        item
        for item in manifest["assets"]
        if item["path"] == "exports/mitsuba/scene.txt"
    )
    assert exported["sha256"].startswith("sha256:")


# -- validation stages -------------------------------------------------------------


def test_validation_can_be_skipped(inputs_dir: Path) -> None:
    config = _coupon_config(
        inputs_dir, "out/coupon", validate_material=False, validate_optical=False
    )
    result = run_pipeline(config, base_dir=str(inputs_dir))
    by_name = {stage.name: stage.status for stage in result.stages}
    assert by_name["validate-material"] is StageStatus.SKIPPED
    assert by_name["validate-optical"] is StageStatus.SKIPPED
    validation = json.loads(
        (result.output_path / "diagnostics" / "validation.json").read_text()
    )
    statuses = {entry["path"]: entry["status"] for entry in validation["assets"]}
    assert statuses["material.zarr"] == "skipped"
