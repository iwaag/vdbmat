from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).parents[1] / "examples" / "pipeline_run" / "demo"
sys.path.insert(0, str(DEMO_DIR))

from mitsuba_stage_mappings import MappingCandidate  # noqa: E402
from mitsuba_stage_regen import (  # noqa: E402
    DerivedBundle,
    RegenError,
    derived_bundle_key,
    locate_bundle_source,
    regenerate_optical,
)

from vdbmat.fixtures import write_phase1_fixtures  # noqa: E402
from vdbmat.pipeline import (  # noqa: E402
    PipelineConfig,
    run_pipeline,
    zarr_store_sha256,
)

_BUILTIN_MAPPING = (
    Path(__file__).parents[1]
    / "examples"
    / "pipeline_run"
    / "mappings"
    / "phase0-provisional-materials-v1.optical-mapping.json"
)


def _builtin_document() -> dict[str, object]:
    return json.loads(_BUILTIN_MAPPING.read_text(encoding="utf-8"))


def _write_mapping(path: Path, document: dict[str, object]) -> MappingCandidate:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return MappingCandidate(root_relative=path.name, path=path)


@pytest.fixture
def coupon_bundle(tmp_path: Path) -> Path:
    inputs_dir = tmp_path / "inputs"
    write_phase1_fixtures(inputs_dir)
    config = PipelineConfig(
        input_kind="direct-voxel",
        input_path="window_coupon.voxels.json",
        output_path="bundle",
        mapping_name="phase0-provisional-materials-v1",
    )
    result = run_pipeline(config, base_dir=str(inputs_dir))
    return result.output_path


# -- locate_bundle_source ------------------------------------------------------------


def test_locate_bundle_source_finds_manifest_and_digest(coupon_bundle: Path) -> None:
    manifest_path, payload_sha256 = locate_bundle_source(coupon_bundle)

    assert manifest_path == coupon_bundle / "source" / "window_coupon.voxels.json"
    run_manifest = json.loads((coupon_bundle / "run.json").read_text())
    assert payload_sha256 == run_manifest["input_payload_sha256"]


def test_locate_bundle_source_rejects_missing_run_json(tmp_path: Path) -> None:
    with pytest.raises(RegenError) as excinfo:
        locate_bundle_source(tmp_path)
    assert excinfo.value.stage == "validate"


def test_locate_bundle_source_rejects_tampered_run_manifest(
    coupon_bundle: Path,
) -> None:
    run_json_path = coupon_bundle / "run.json"
    run_manifest = json.loads(run_json_path.read_text())
    run_manifest["input_payload_sha256"] = (
        "sha256:" + "0" * 64
    )
    run_json_path.write_text(json.dumps(run_manifest))

    with pytest.raises(RegenError, match="digest mismatch") as excinfo:
        locate_bundle_source(coupon_bundle)
    assert excinfo.value.stage == "validate"


def test_locate_bundle_source_rejects_ambiguous_assets(coupon_bundle: Path) -> None:
    run_json_path = coupon_bundle / "run.json"
    run_manifest = json.loads(run_json_path.read_text())
    run_manifest["assets"] = [
        entry
        for entry in run_manifest["assets"]
        if not entry["path"].endswith(".voxels.json")
    ]
    run_json_path.write_text(json.dumps(run_manifest))

    with pytest.raises(RegenError, match="found 0") as excinfo:
        locate_bundle_source(coupon_bundle)
    assert excinfo.value.stage == "validate"


# -- regenerate_optical: basic behaviour ----------------------------------------------


def test_regenerate_optical_publishes_derived_bundle(
    coupon_bundle: Path, tmp_path: Path
) -> None:
    mapping = _write_mapping(
        tmp_path / "mapping.optical-mapping.json", _builtin_document()
    )
    work_root = tmp_path / "derived"

    derived = regenerate_optical(coupon_bundle, mapping, work_root)

    assert isinstance(derived, DerivedBundle)
    assert derived.reused is False
    assert derived.optical_zarr == derived.bundle_path / "optical.zarr"
    assert derived.optical_zarr.is_dir()
    run_manifest = json.loads((derived.bundle_path / "run.json").read_text())
    assert run_manifest["mapping_digest"] == derived.mapping_digest
    assert run_manifest["input_payload_sha256"] == derived.source_payload_sha256


def test_regenerate_optical_records_mapping_digest_in_provenance(
    coupon_bundle: Path, tmp_path: Path
) -> None:
    mapping = _write_mapping(
        tmp_path / "mapping.optical-mapping.json", _builtin_document()
    )
    derived = regenerate_optical(coupon_bundle, mapping, tmp_path / "derived")

    summary = json.loads(
        (derived.bundle_path / "diagnostics" / "summary.json").read_text()
    )
    assert summary["digests"]["mapping"] == derived.mapping_digest


def test_regenerate_optical_leaves_source_bundle_untouched(
    coupon_bundle: Path, tmp_path: Path
) -> None:
    mapping = _write_mapping(
        tmp_path / "mapping.optical-mapping.json", _builtin_document()
    )
    before = zarr_store_sha256(coupon_bundle / "optical.zarr")
    before_material = zarr_store_sha256(coupon_bundle / "material.zarr")

    regenerate_optical(coupon_bundle, mapping, tmp_path / "derived")

    assert zarr_store_sha256(coupon_bundle / "optical.zarr") == before
    assert zarr_store_sha256(coupon_bundle / "material.zarr") == before_material


# -- regenerate_optical: cache reuse --------------------------------------------------


def test_regenerate_optical_reuses_cache_on_repeat_call(
    coupon_bundle: Path, tmp_path: Path
) -> None:
    mapping = _write_mapping(
        tmp_path / "mapping.optical-mapping.json", _builtin_document()
    )
    work_root = tmp_path / "derived"

    first = regenerate_optical(coupon_bundle, mapping, work_root)
    run_json_path = first.bundle_path / "run.json"
    first_mtime_ns = run_json_path.stat().st_mtime_ns

    second = regenerate_optical(coupon_bundle, mapping, work_root)

    assert second.reused is True
    assert second.bundle_path == first.bundle_path
    assert run_json_path.stat().st_mtime_ns == first_mtime_ns


def test_regenerate_optical_recomputes_after_mapping_edit(
    coupon_bundle: Path, tmp_path: Path
) -> None:
    mapping_path = tmp_path / "mapping.optical-mapping.json"
    mapping = _write_mapping(mapping_path, _builtin_document())
    work_root = tmp_path / "derived"

    first = regenerate_optical(coupon_bundle, mapping, work_root)

    edited = _builtin_document()
    edited["materials"][1]["sigma_a_rgb_per_m"] = [9.0, 9.0, 9.0]
    _write_mapping(mapping_path, edited)

    second = regenerate_optical(coupon_bundle, mapping, work_root)

    assert second.reused is False
    assert second.bundle_path != first.bundle_path
    assert second.mapping_digest != first.mapping_digest


def test_derived_bundle_key_changes_with_either_input() -> None:
    base = derived_bundle_key("sha256:" + "1" * 64, "sha256:" + "2" * 64)
    other_source = derived_bundle_key("sha256:" + "3" * 64, "sha256:" + "2" * 64)
    other_mapping = derived_bundle_key("sha256:" + "1" * 64, "sha256:" + "4" * 64)

    assert len({base, other_source, other_mapping}) == 3


def test_regenerate_optical_two_mappings_produce_different_optical_digest(
    coupon_bundle: Path, tmp_path: Path
) -> None:
    mapping_a = _write_mapping(
        tmp_path / "a.optical-mapping.json", _builtin_document()
    )
    tinted = _builtin_document()
    tinted["materials"][1]["sigma_a_rgb_per_m"] = [9.0, 9.0, 9.0]
    mapping_b = _write_mapping(tmp_path / "b.optical-mapping.json", tinted)
    work_root = tmp_path / "derived"

    derived_a = regenerate_optical(coupon_bundle, mapping_a, work_root)
    derived_b = regenerate_optical(coupon_bundle, mapping_b, work_root)

    assert derived_a.bundle_path != derived_b.bundle_path
    assert zarr_store_sha256(derived_a.optical_zarr) != zarr_store_sha256(
        derived_b.optical_zarr
    )


# -- regenerate_optical: failures ------------------------------------------------------


def test_regenerate_optical_rejects_invalid_mapping_json(
    coupon_bundle: Path, tmp_path: Path
) -> None:
    mapping_path = tmp_path / "broken.optical-mapping.json"
    mapping_path.write_text("{", encoding="utf-8")
    mapping = MappingCandidate(root_relative=mapping_path.name, path=mapping_path)

    with pytest.raises(RegenError) as excinfo:
        regenerate_optical(coupon_bundle, mapping, tmp_path / "derived")
    assert excinfo.value.stage == "validate"


def test_regenerate_optical_rejects_missing_materials(
    coupon_bundle: Path, tmp_path: Path
) -> None:
    document = _builtin_document()
    document["materials"] = [
        item for item in document["materials"] if item["material_id"] != 3
    ]
    mapping = _write_mapping(tmp_path / "incomplete.optical-mapping.json", document)

    with pytest.raises(RegenError, match="material_id") as excinfo:
        regenerate_optical(coupon_bundle, mapping, tmp_path / "derived")
    assert excinfo.value.stage == "map"


def test_regenerate_optical_rejects_name_mismatch(
    coupon_bundle: Path, tmp_path: Path
) -> None:
    document = _builtin_document()
    for item in document["materials"]:
        if item["material_id"] == 1:
            item["name"] = "wrong-name"
    mapping = _write_mapping(tmp_path / "mismatched.optical-mapping.json", document)

    with pytest.raises(RegenError) as excinfo:
        regenerate_optical(coupon_bundle, mapping, tmp_path / "derived")
    assert excinfo.value.stage == "map"


def test_regenerate_optical_rejects_work_root_inside_source_bundle(
    coupon_bundle: Path, tmp_path: Path
) -> None:
    mapping = _write_mapping(
        tmp_path / "mapping.optical-mapping.json", _builtin_document()
    )

    with pytest.raises(RegenError, match="collides") as excinfo:
        regenerate_optical(coupon_bundle, mapping, coupon_bundle / "derived")
    assert excinfo.value.stage == "validate"
