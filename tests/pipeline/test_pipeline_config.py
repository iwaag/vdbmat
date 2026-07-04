"""Tests for the versioned Phase 1 pipeline configuration (plan Step 5, ADR-009)."""

import json
import re
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from vbdmat.optics import phase0_provisional_mapping
from vbdmat.pipeline import (
    DEFAULT_MAPPING_NAME,
    PIPELINE_CONFIG_SCHEMA,
    ExportSettings,
    ExportTarget,
    InputKind,
    PipelineConfig,
    PipelineConfigError,
    RendererConfig,
)


def direct_config(**overrides: object) -> PipelineConfig:
    base = {
        "input_kind": InputKind.DIRECT_VOXEL,
        "input_path": "inputs/window_coupon.voxels.json",
        "output_path": "runs/window_coupon",
    }
    base.update(overrides)
    return PipelineConfig(**base)  # type: ignore[arg-type]


# -- Construction and immutability ------------------------------------------------


def test_direct_config_defaults_are_explicit() -> None:
    config = direct_config()

    assert config.input_kind is InputKind.DIRECT_VOXEL
    assert config.mapping_name == DEFAULT_MAPPING_NAME
    assert config.overwrite is False
    assert config.random_seed == 0
    assert config.exports == ()
    assert config.renderer is None


def test_config_is_frozen() -> None:
    config = direct_config()
    with pytest.raises(FrozenInstanceError):
        config.overwrite = True  # type: ignore[misc]


def test_mapping_resolves_to_phase0_and_reports_digest() -> None:
    config = direct_config()
    assert config.resolve_mapping() == phase0_provisional_mapping()
    assert config.mapping_digest == phase0_provisional_mapping().digest


# -- Digest stability and sensitivity ---------------------------------------------


def test_equivalent_configurations_hash_identically() -> None:
    a = direct_config()
    b = direct_config()
    assert a.digest == b.digest
    assert a.canonical_json() == b.canonical_json()


def test_json_roundtrip_hashes_identically() -> None:
    config = direct_config(exports=(ExportSettings(ExportTarget.MITSUBA),))
    restored = PipelineConfig.from_json(config.canonical_json())
    assert restored.digest == config.digest


@pytest.mark.parametrize(
    "changed",
    [
        {"input_path": "inputs/other.voxels.json"},
        {"output_path": "runs/other"},
        {"overwrite": True},
        {"random_seed": 7},
        {"validate_optical": False},
        {"exports": (ExportSettings(ExportTarget.MITSUBA),)},
        {"renderer": RendererConfig(references=("scenes/coupon.xml",))},
    ],
)
def test_meaningful_changes_alter_the_digest(changed: dict[str, object]) -> None:
    assert direct_config().digest != direct_config(**changed).digest


# -- Renderer/export independence of canonical results ----------------------------


def test_renderer_and_exports_do_not_change_scientific_digest() -> None:
    plain = direct_config()
    with_renderer = direct_config(
        exports=(ExportSettings(ExportTarget.MITSUBA),),
        renderer=RendererConfig(references=("scenes/wedge.xml",)),
    )
    # The whole-config identity differs, but the canonical-result identity does not.
    assert plain.digest != with_renderer.digest
    assert plain.scientific_digest == with_renderer.scientific_digest


def test_scientific_digest_ignores_input_path_and_output() -> None:
    a = direct_config()
    b = direct_config(input_path="elsewhere/coupon.voxels.json", output_path="runs/x")
    assert a.scientific_digest == b.scientific_digest


def test_scientific_digest_tracks_validation_and_seed() -> None:
    a = direct_config()
    b = direct_config(validate_material=False)
    assert a.scientific_digest != b.scientific_digest
    assert a.scientific_digest != direct_config(random_seed=11).scientific_digest


# -- Round-trip fidelity ----------------------------------------------------------


def test_roundtrip_preserves_every_declared_setting() -> None:
    config = direct_config(
        mapping_name=DEFAULT_MAPPING_NAME,
        validate_material=False,
        validate_optical=True,
        exports=(
            ExportSettings(ExportTarget.MITSUBA),
            ExportSettings(ExportTarget.OPENVDB),
        ),
        overwrite=True,
        random_seed=42,
        renderer=RendererConfig(references=("a.xml", "b.xml")),
    )
    restored = PipelineConfig.from_json_dict(config.to_json_dict())
    assert restored == config


def test_recorded_schema_and_mapping_digest() -> None:
    document = direct_config().to_json_dict()
    assert document["schema"] == {
        "name": PIPELINE_CONFIG_SCHEMA.name,
        "version": str(PIPELINE_CONFIG_SCHEMA.version),
    }
    assert document["schema"]["version"] == "2.0.0"
    assert document["mapping"]["digest"] == phase0_provisional_mapping().digest


def test_canonical_json_is_sorted_and_tight() -> None:
    text = direct_config().canonical_json()
    assert text == json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))


# -- File-based mappings (ADR-009 D3) ----------------------------------------------


def _mapping_file_config(tmp_path: Path) -> PipelineConfig:
    from vbdmat.optics import write_optical_mapping

    write_optical_mapping(tmp_path / "mapping.json", phase0_provisional_mapping())
    return direct_config(
        mapping_name=None,
        mapping_path="mapping.json",
        mapping_digest=phase0_provisional_mapping().digest,
    )


def test_mapping_path_resolves_and_verifies_digest(tmp_path: Path) -> None:
    config = _mapping_file_config(tmp_path)
    resolved = config.resolve_mapping(str(tmp_path))
    assert resolved == phase0_provisional_mapping()


def test_mapping_path_with_wrong_digest_fails_at_resolution(tmp_path: Path) -> None:
    from vbdmat.optics import write_optical_mapping

    write_optical_mapping(tmp_path / "mapping.json", phase0_provisional_mapping())
    config = direct_config(
        mapping_name=None,
        mapping_path="mapping.json",
        mapping_digest="sha256:" + "0" * 64,
    )
    with pytest.raises(PipelineConfigError, match=re.escape("mapping.digest")):
        config.resolve_mapping(str(tmp_path))


def test_mapping_path_requires_declared_digest() -> None:
    with pytest.raises(PipelineConfigError, match=re.escape("mapping.digest")):
        direct_config(mapping_name=None, mapping_path="mapping.json")


def test_mapping_name_and_path_are_mutually_exclusive() -> None:
    with pytest.raises(PipelineConfigError, match="exactly one"):
        direct_config(
            mapping_name=DEFAULT_MAPPING_NAME,
            mapping_path="mapping.json",
            mapping_digest="sha256:" + "0" * 64,
        )


def test_mapping_path_requires_base_dir_to_resolve(tmp_path: Path) -> None:
    config = _mapping_file_config(tmp_path)
    with pytest.raises(PipelineConfigError, match="base_dir"):
        config.resolve_mapping()


def test_file_and_builtin_mapping_share_the_scientific_digest(tmp_path: Path) -> None:
    # The scientific identity depends on the mapping *content* digest only, not on
    # whether it was supplied as a builtin name or an external document.
    assert (
        _mapping_file_config(tmp_path).scientific_digest
        == direct_config().scientific_digest
    )


def test_path_based_config_roundtrips(tmp_path: Path) -> None:
    config = _mapping_file_config(tmp_path)
    restored = PipelineConfig.from_json_dict(config.to_json_dict())
    assert restored == config
    document = config.to_json_dict()
    assert document["mapping"]["path"] == "mapping.json"
    assert "name" not in document["mapping"]


def test_json_document_requires_exactly_one_mapping_reference() -> None:
    document = direct_config().to_json_dict()
    document["mapping"] = {"digest": "sha256:" + "0" * 64}
    with pytest.raises(PipelineConfigError, match="exactly one"):
        PipelineConfig.from_json_dict(document)


# -- Invalid combinations fail before any output ----------------------------------


def test_unknown_mapping_name_fails() -> None:
    with pytest.raises(PipelineConfigError, match=re.escape("mapping.name")):
        direct_config(mapping_name="does-not-exist")


def test_unsupported_input_kind_fails() -> None:
    with pytest.raises(PipelineConfigError, match=re.escape("input.kind")):
        PipelineConfig(
            input_kind="mesh",  # type: ignore[arg-type]
            input_path="m.stl",
            output_path="runs/m",
        )


def test_empty_paths_fail() -> None:
    with pytest.raises(PipelineConfigError, match=re.escape("input.path")):
        direct_config(input_path="   ")
    with pytest.raises(PipelineConfigError, match=re.escape("output.path")):
        direct_config(output_path="")


def test_duplicate_export_targets_fail() -> None:
    with pytest.raises(PipelineConfigError, match="duplicate export target"):
        direct_config(
            exports=(
                ExportSettings(ExportTarget.MITSUBA),
                ExportSettings(ExportTarget.MITSUBA),
            )
        )


# -- Deserialization guards -------------------------------------------------------


def test_incompatible_major_schema_rejected() -> None:
    document = direct_config().to_json_dict()
    document["schema"]["version"] = "1.0.0"
    with pytest.raises(PipelineConfigError, match="incompatible major version"):
        PipelineConfig.from_json_dict(document)


def test_compatible_minor_schema_accepted() -> None:
    document = direct_config().to_json_dict()
    document["schema"]["version"] = "2.5.0"
    restored = PipelineConfig.from_json_dict(document)
    assert restored.digest == direct_config().digest


def test_unknown_top_level_key_rejected() -> None:
    document = direct_config().to_json_dict()
    document["surprise"] = True
    with pytest.raises(PipelineConfigError, match="unknown keys"):
        PipelineConfig.from_json_dict(document)


def test_v1_mesh_input_document_is_rejected() -> None:
    # Removed mesh path (ADR-009 D1): a v1-style voxelization block is an unknown key.
    document = direct_config().to_json_dict()
    document["input"]["voxelization"] = {"source_unit": "mm"}
    with pytest.raises(PipelineConfigError, match="unknown keys"):
        PipelineConfig.from_json_dict(document)


def test_recorded_mapping_digest_mismatch_rejected() -> None:
    document = direct_config().to_json_dict()
    document["mapping"]["digest"] = "sha256:" + "0" * 64
    with pytest.raises(PipelineConfigError, match=re.escape("mapping.digest")):
        PipelineConfig.from_json_dict(document)


def test_invalid_json_text_reports_field() -> None:
    with pytest.raises(PipelineConfigError, match="invalid JSON"):
        PipelineConfig.from_json("{not json")


# -- Path resolution needs an explicit base directory -----------------------------


def test_paths_are_recorded_verbatim() -> None:
    config = direct_config(input_path="inputs/coupon.voxels.json")
    assert config.input_path == "inputs/coupon.voxels.json"
    assert config.to_json_dict()["input"]["path"] == "inputs/coupon.voxels.json"


def test_resolution_requires_explicit_base_dir() -> None:
    config = direct_config()
    resolved = config.resolve_input_path("/work/project")
    assert resolved.startswith("/work/project")
    assert resolved.endswith("window_coupon.voxels.json")


def test_replace_keeps_config_valid() -> None:
    config = replace(direct_config(), overwrite=True)
    assert config.overwrite is True
    assert config.digest != direct_config().digest


# -- Committed example configurations stay valid ----------------------------------

EXAMPLE_CONFIGS = (
    Path(__file__).resolve().parents[2] / "examples" / "phase1" / "configs"
)


@pytest.mark.parametrize(
    "name",
    ["window_coupon.run.json", "stepped_wedge.run.json"],
)
def test_committed_example_configs_parse(name: str) -> None:
    text = (EXAMPLE_CONFIGS / name).read_text(encoding="utf-8")
    config = PipelineConfig.from_json(text)
    assert config.input_kind is InputKind.DIRECT_VOXEL
    # Re-serializing and re-parsing is stable (the committed file is a valid config).
    assert PipelineConfig.from_json(config.canonical_json()).digest == config.digest
