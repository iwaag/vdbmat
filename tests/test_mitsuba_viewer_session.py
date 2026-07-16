from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).parents[1] / "examples" / "pipeline_run" / "demo"
sys.path.insert(0, str(DEMO_DIR))

import mitsuba_viewer_session as session_module  # noqa: E402
from mitsuba_stage import (  # noqa: E402
    BackdropSettings,
    RenderSettings,
    StageConfig,
    stage_config_to_dict,
)
from mitsuba_stage_inputs import resolve_candidate  # noqa: E402
from mitsuba_stage_presets import stage_config_digest  # noqa: E402
from mitsuba_viewer_session import (  # noqa: E402
    SessionInputRef,
    SessionMappingRef,
    SessionPresetRef,
    ViewerSession,
    ViewerSessionError,
    create_viewer_session,
    resolve_viewer_session,
    verify_derived_optical,
    viewer_session_from_dict,
    viewer_session_from_json,
    viewer_session_to_dict,
    write_viewer_session,
)

from vdbmat.fixtures import transparent_opaque_interface  # noqa: E402
from vdbmat.io import write_volume  # noqa: E402
from vdbmat.optics import (  # noqa: E402
    map_material_volume_to_optical,
    optical_mapping_to_json_dict,
    phase0_provisional_mapping,
)
from vdbmat.pipeline import zarr_store_sha256  # noqa: E402

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64


def _optical_volume():
    return map_material_volume_to_optical(
        transparent_opaque_interface().volume, phase0_provisional_mapping()
    )


def _write_optical(path: Path) -> None:
    write_volume(path, _optical_volume())


def _write_bundle(path: Path) -> None:
    path.mkdir(parents=True)
    _write_optical(path / "optical.zarr")
    (path / "run.json").write_text(
        json.dumps(
            {
                "schema": {"name": "vdbmat.run", "version": "1.0.0"},
                "run_id": "run-session-test",
            }
        ),
        encoding="utf-8",
    )


def _write_preset(path: Path, config: StageConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stage_config_to_dict(config)), encoding="utf-8")


def _standalone_session(
    *,
    config: StageConfig | None = None,
    preset: SessionPresetRef | None = None,
) -> ViewerSession:
    stage = StageConfig() if config is None else config
    return ViewerSession(
        input=SessionInputRef(
            kind=session_module.InputKind.OPTICAL_ZARR,
            path="catalog/model.zarr",
            optical_sha256=_DIGEST_A,
        ),
        stage_config=stage,
        effective_digest=stage_config_digest(stage),
        variant="llvm_ad_rgb",
        seed=20260628,
        preset=preset,
    )


def _bundle_session(
    *,
    config: StageConfig | None = None,
    preset: SessionPresetRef | None = None,
    mapping: SessionMappingRef | None = None,
) -> ViewerSession:
    stage = StageConfig() if config is None else config
    return ViewerSession(
        input=SessionInputRef(
            kind=session_module.InputKind.RUN_BUNDLE,
            path="catalog/bundle",
            optical_sha256=_DIGEST_A,
            run_manifest_sha256=_DIGEST_B,
        ),
        stage_config=stage,
        effective_digest=stage_config_digest(stage),
        variant="llvm_ad_rgb",
        seed=20260628,
        preset=preset,
        mapping=mapping,
    )


def _bundle_session_with_mapping() -> ViewerSession:
    return _bundle_session(
        mapping=SessionMappingRef(
            path="tinted.optical-mapping.json",
            digest=_DIGEST_A,
            derived_optical_sha256=_DIGEST_B,
        )
    )


def _write_mapping_document(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(optical_mapping_to_json_dict(phase0_provisional_mapping())),
        encoding="utf-8",
    )


def test_writer_reader_round_trip_preserves_all_fields() -> None:
    config = StageConfig(
        render=RenderSettings(width=320, height=240, spp=16, max_depth=12),
        backdrop=BackdropSettings(checker_scale=11),
    )
    original = _standalone_session(
        config=config,
        preset=SessionPresetRef(
            path="highkey.stage.json", digest=stage_config_digest(config)
        ),
    )

    document = viewer_session_to_dict(original)
    restored = viewer_session_from_dict(document)

    assert restored == original
    assert document["format"] == "vdbmat.viewer-session"
    assert document["format_version"] == "1.1.0"
    assert "render" not in document["stage"]["effective"]
    assert document["render"] == {
        "width": 320,
        "height": 240,
        "spp": 16,
        "max_depth": 12,
    }


@pytest.mark.parametrize(
    ("section", "key"),
    [
        (None, "unknown"),
        ("input", "unknown"),
        ("stage", "unknown"),
        ("render", "unknown"),
        ("mitsuba", "unknown"),
    ],
)
def test_reader_rejects_unknown_keys(section: str | None, key: str) -> None:
    document = viewer_session_to_dict(_standalone_session())
    target = document if section is None else document[section]
    target[key] = True

    with pytest.raises(ViewerSessionError, match="unknown keys"):
        viewer_session_from_dict(document)


@pytest.mark.parametrize("version", ["0.9.0", "1.2.0", "2.0.0"])
def test_reader_rejects_unsupported_version(version: str) -> None:
    document = viewer_session_to_dict(_standalone_session())
    document["format_version"] = version

    with pytest.raises(ViewerSessionError, match="format_version"):
        viewer_session_from_dict(document)


def test_reader_accepts_legacy_1_0_0_document() -> None:
    document = viewer_session_to_dict(_standalone_session())
    document["format_version"] = "1.0.0"

    restored = viewer_session_from_dict(document)

    assert restored.mapping is None


def test_reader_rejects_mapping_section_on_1_0_0_document() -> None:
    document = viewer_session_to_dict(_bundle_session_with_mapping())
    document["format_version"] = "1.0.0"

    with pytest.raises(
        ViewerSessionError, match=r"1\.0\.0 must not declare a mapping section"
    ):
        viewer_session_from_dict(document)


def test_reader_rejects_render_inside_effective_stage() -> None:
    document = viewer_session_to_dict(_standalone_session())
    document["stage"]["effective"]["render"] = dict(document["render"])

    with pytest.raises(ViewerSessionError, match=r"stage\.effective.*unknown"):
        viewer_session_from_dict(document)


def test_reader_requires_all_effective_stage_fields() -> None:
    document = viewer_session_to_dict(_standalone_session())
    del document["stage"]["effective"]["backdrop"]["color1"]

    with pytest.raises(
        ViewerSessionError, match=r"stage\.effective\.backdrop.*missing"
    ):
        viewer_session_from_dict(document)


def test_reader_rejects_bool_seed() -> None:
    document = viewer_session_to_dict(_standalone_session())
    document["mitsuba"]["seed"] = True

    with pytest.raises(ViewerSessionError, match=r"seed must be an integer"):
        viewer_session_from_dict(document)


@pytest.mark.parametrize("digest", ["abc", "sha256:" + "A" * 64, "a" * 64])
def test_reader_rejects_invalid_digest(digest: str) -> None:
    document = viewer_session_to_dict(_standalone_session())
    document["input"]["optical_sha256"] = digest

    with pytest.raises(ViewerSessionError, match=r"sha256:<64 lowercase hex>"):
        viewer_session_from_dict(document)


def test_reader_rejects_effective_digest_mismatch() -> None:
    document = viewer_session_to_dict(_standalone_session())
    document["stage"]["effective_digest"] = _DIGEST_B

    with pytest.raises(ViewerSessionError, match="effective_digest mismatch"):
        viewer_session_from_dict(document)


def test_reader_rejects_preset_digest_that_does_not_match_effective_config() -> None:
    document = viewer_session_to_dict(_standalone_session())
    document["stage"]["preset"] = {
        "path": "other.stage.json",
        "digest": _DIGEST_B,
    }

    with pytest.raises(ViewerSessionError, match=r"preset\.digest must match"):
        viewer_session_from_dict(document)


def test_run_bundle_requires_run_manifest_digest() -> None:
    document = viewer_session_to_dict(_standalone_session())
    document["input"]["kind"] = "run-bundle"

    with pytest.raises(ViewerSessionError, match="missing keys"):
        viewer_session_from_dict(document)


def test_standalone_forbids_run_manifest_digest() -> None:
    document = viewer_session_to_dict(_standalone_session())
    document["input"]["run_manifest_sha256"] = _DIGEST_B

    with pytest.raises(ViewerSessionError, match="unknown keys"):
        viewer_session_from_dict(document)


@pytest.mark.parametrize(
    "path",
    [
        "",
        ".",
        "/absolute/model.zarr",
        "../model.zarr",
        "catalog/../model.zarr",
        "catalog\\model.zarr",
        "catalog//model.zarr",
        "catalog/model.zarr/",
        "catalog/\x00model.zarr",
    ],
)
def test_reader_rejects_nonportable_input_paths(path: str) -> None:
    document = viewer_session_to_dict(_standalone_session())
    document["input"]["path"] = path

    with pytest.raises(ViewerSessionError, match=r"input\.path"):
        viewer_session_from_dict(document)


def test_reader_rejects_nonportable_preset_path() -> None:
    config = StageConfig()
    session = _standalone_session(
        config=config,
        preset=SessionPresetRef(
            path="valid.stage.json", digest=stage_config_digest(config)
        ),
    )
    document = viewer_session_to_dict(session)
    document["stage"]["preset"]["path"] = "../escape.stage.json"

    with pytest.raises(ViewerSessionError, match=r"preset\.path"):
        viewer_session_from_dict(document)


def test_json_reader_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.session.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ViewerSessionError, match="not valid JSON"):
        viewer_session_from_json(path)


def test_create_and_resolve_standalone_session(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    _write_optical(input_root / "model.zarr")
    candidate = resolve_candidate(input_root, Path("model.zarr"))
    config = StageConfig(render=RenderSettings(max_depth=14))

    session = create_viewer_session(
        candidate, config, "llvm_ad_rgb", 99
    )
    resolved = resolve_viewer_session(session, input_root)

    assert resolved.input_candidate == candidate
    assert resolved.optical_zarr == candidate.optical_zarr
    assert resolved.stage_config == config
    assert resolved.variant == "llvm_ad_rgb"
    assert resolved.seed == 99
    assert resolved.preset_candidate is None


def test_create_and_resolve_bundle_session(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    _write_bundle(input_root / "bundle")
    candidate = resolve_candidate(input_root, Path("bundle"))

    session = create_viewer_session(
        candidate, StageConfig(), "cuda_ad_rgb", 0
    )
    document = viewer_session_to_dict(session)
    resolved = resolve_viewer_session(session, input_root)

    assert session.input.kind is session_module.InputKind.RUN_BUNDLE
    assert session.input.run_manifest_sha256 is not None
    assert document["input"]["run_manifest_sha256"].startswith("sha256:")
    assert resolved.optical_zarr == input_root / "bundle" / "optical.zarr"


def test_resolver_rejects_modified_optical_store(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    _write_optical(input_root / "model.zarr")
    candidate = resolve_candidate(input_root, Path("model.zarr"))
    session = create_viewer_session(
        candidate, StageConfig(), "llvm_ad_rgb", 1
    )
    (candidate.optical_zarr / "changed-marker").write_text("changed")

    with pytest.raises(ViewerSessionError, match="input optical digest mismatch"):
        resolve_viewer_session(session, input_root)


def test_resolver_rejects_modified_run_manifest(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    _write_bundle(input_root / "bundle")
    candidate = resolve_candidate(input_root, Path("bundle"))
    session = create_viewer_session(
        candidate, StageConfig(), "llvm_ad_rgb", 1
    )
    (candidate.path / "run.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ViewerSessionError, match="run manifest digest mismatch"):
        resolve_viewer_session(session, input_root)


def test_resolver_rejects_input_kind_mismatch(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    _write_bundle(input_root / "candidate")
    optical = input_root / "candidate" / "optical.zarr"
    digest = session_module.zarr_store_sha256(optical)
    session = ViewerSession(
        input=SessionInputRef(
            kind=session_module.InputKind.OPTICAL_ZARR,
            path="candidate",
            optical_sha256=digest,
        ),
        stage_config=StageConfig(),
        effective_digest=stage_config_digest(StageConfig()),
        variant="llvm_ad_rgb",
        seed=1,
    )

    with pytest.raises(ViewerSessionError, match="input kind mismatch"):
        resolve_viewer_session(session, input_root)


def test_resolver_rejects_input_symlink_escaping_root(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    outside = tmp_path / "outside.zarr"
    _write_optical(outside)
    (input_root / "escape.zarr").symlink_to(outside, target_is_directory=True)
    session = ViewerSession(
        input=SessionInputRef(
            kind=session_module.InputKind.OPTICAL_ZARR,
            path="escape.zarr",
            optical_sha256=session_module.zarr_store_sha256(outside),
        ),
        stage_config=StageConfig(),
        effective_digest=stage_config_digest(StageConfig()),
        variant="llvm_ad_rgb",
        seed=1,
    )

    with pytest.raises(ViewerSessionError, match="outside --input-root"):
        resolve_viewer_session(session, input_root)


def test_resolver_rejects_missing_input(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    session = ViewerSession(
        input=SessionInputRef(
            kind=session_module.InputKind.OPTICAL_ZARR,
            path="missing.zarr",
            optical_sha256=_DIGEST_A,
        ),
        stage_config=StageConfig(),
        effective_digest=stage_config_digest(StageConfig()),
        variant="llvm_ad_rgb",
        seed=1,
    )

    with pytest.raises(ViewerSessionError, match="input does not exist"):
        resolve_viewer_session(session, input_root)


def test_resolver_verifies_preset_semantically(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    preset_root = tmp_path / "presets"
    input_root.mkdir()
    _write_optical(input_root / "model.zarr")
    preset_config = StageConfig(render=RenderSettings(max_depth=12))
    preset_path = preset_root / "highkey.stage.json"
    _write_preset(preset_path, preset_config)
    candidate = resolve_candidate(input_root, Path("model.zarr"))
    preset_ref = SessionPresetRef(
        path="highkey.stage.json",
        digest=stage_config_digest(preset_config),
    )
    session = create_viewer_session(
        candidate,
        preset_config,
        "llvm_ad_rgb",
        10,
        preset=preset_ref,
    )

    preset_path.write_text(
        json.dumps(stage_config_to_dict(preset_config), indent=4),
        encoding="utf-8",
    )
    resolved = resolve_viewer_session(session, input_root, preset_root)

    assert resolved.preset_candidate is not None
    assert resolved.preset_candidate.root_relative == "highkey.stage.json"


def test_resolver_rejects_modified_preset_value(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    preset_root = tmp_path / "presets"
    input_root.mkdir()
    _write_optical(input_root / "model.zarr")
    original = StageConfig()
    preset_path = preset_root / "preset.stage.json"
    _write_preset(preset_path, original)
    session = create_viewer_session(
        resolve_candidate(input_root, Path("model.zarr")),
        original,
        "llvm_ad_rgb",
        1,
        preset=SessionPresetRef(
            path="preset.stage.json", digest=stage_config_digest(original)
        ),
    )
    _write_preset(
        preset_path,
        StageConfig(render=RenderSettings(max_depth=32)),
    )

    with pytest.raises(ViewerSessionError, match="stage preset digest mismatch"):
        resolve_viewer_session(session, input_root, preset_root)


def test_resolver_requires_preset_root_for_reference(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    _write_optical(input_root / "model.zarr")
    session = create_viewer_session(
        resolve_candidate(input_root, Path("model.zarr")),
        StageConfig(),
        "llvm_ad_rgb",
        1,
        preset=SessionPresetRef(
            path="preset.stage.json", digest=stage_config_digest(StageConfig())
        ),
    )

    with pytest.raises(ViewerSessionError, match="requires --preset-root"):
        resolve_viewer_session(session, input_root)


def test_resolver_rejects_missing_preset(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    preset_root = tmp_path / "presets"
    input_root.mkdir()
    preset_root.mkdir()
    _write_optical(input_root / "model.zarr")
    config = StageConfig()
    session = create_viewer_session(
        resolve_candidate(input_root, Path("model.zarr")),
        config,
        "llvm_ad_rgb",
        1,
        preset=SessionPresetRef(
            path="missing.stage.json", digest=stage_config_digest(config)
        ),
    )

    with pytest.raises(ViewerSessionError, match="preset does not exist"):
        resolve_viewer_session(session, input_root, preset_root)


# -- mapping (format 1.1) -----------------------------------------------------------


def test_mapping_round_trips_and_survives_write_read() -> None:
    original = _bundle_session_with_mapping()

    document = viewer_session_to_dict(original)
    restored = viewer_session_from_dict(document)

    assert restored == original
    assert document["mapping"] == {
        "path": "tinted.optical-mapping.json",
        "digest": _DIGEST_A,
        "derived_optical_sha256": _DIGEST_B,
    }


def test_mapping_requires_run_bundle_input_kind() -> None:
    with pytest.raises(ValueError, match=r"mapping requires input\.kind"):
        ViewerSession(
            input=SessionInputRef(
                kind=session_module.InputKind.OPTICAL_ZARR,
                path="catalog/model.zarr",
                optical_sha256=_DIGEST_A,
            ),
            stage_config=StageConfig(),
            effective_digest=stage_config_digest(StageConfig()),
            variant="llvm_ad_rgb",
            seed=1,
            mapping=SessionMappingRef(
                path="tinted.optical-mapping.json",
                digest=_DIGEST_A,
                derived_optical_sha256=_DIGEST_B,
            ),
        )


def test_reader_rejects_unknown_mapping_keys() -> None:
    document = viewer_session_to_dict(_bundle_session_with_mapping())
    document["mapping"]["unknown"] = True

    with pytest.raises(ViewerSessionError, match=r"mapping.*unknown keys"):
        viewer_session_from_dict(document)


@pytest.mark.parametrize(
    "path",
    ["", ".", "/absolute.optical-mapping.json", "../escape.optical-mapping.json"],
)
def test_reader_rejects_nonportable_mapping_path(path: str) -> None:
    document = viewer_session_to_dict(_bundle_session_with_mapping())
    document["mapping"]["path"] = path

    with pytest.raises(ViewerSessionError, match=r"mapping\.path"):
        viewer_session_from_dict(document)


def test_create_viewer_session_with_mapping(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    _write_bundle(input_root / "bundle")
    candidate = resolve_candidate(input_root, Path("bundle"))
    mapping_ref = SessionMappingRef(
        path="tinted.optical-mapping.json",
        digest=_DIGEST_A,
        derived_optical_sha256=_DIGEST_B,
    )

    session = create_viewer_session(
        candidate, StageConfig(), "llvm_ad_rgb", 1, mapping=mapping_ref
    )

    assert session.mapping == mapping_ref


def test_create_viewer_session_rejects_mapping_with_optical_zarr_candidate(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    _write_optical(input_root / "model.zarr")
    candidate = resolve_candidate(input_root, Path("model.zarr"))

    with pytest.raises(ViewerSessionError, match=r"mapping requires input\.kind"):
        create_viewer_session(
            candidate,
            StageConfig(),
            "llvm_ad_rgb",
            1,
            mapping=SessionMappingRef(
                path="tinted.optical-mapping.json",
                digest=_DIGEST_A,
                derived_optical_sha256=_DIGEST_B,
            ),
        )


def test_resolve_and_verify_derived_optical_round_trip(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    mapping_root = tmp_path / "mappings"
    input_root.mkdir()
    _write_bundle(input_root / "bundle")
    candidate = resolve_candidate(input_root, Path("bundle"))

    mapping_path = mapping_root / "tinted.optical-mapping.json"
    _write_mapping_document(mapping_path)
    mapping_digest = phase0_provisional_mapping().digest

    derived_path = tmp_path / "derived" / "optical.zarr"
    _write_optical(derived_path)
    derived_digest = zarr_store_sha256(derived_path)

    session = create_viewer_session(
        candidate,
        StageConfig(),
        "llvm_ad_rgb",
        1,
        mapping=SessionMappingRef(
            path="tinted.optical-mapping.json",
            digest=mapping_digest,
            derived_optical_sha256=derived_digest,
        ),
    )

    resolved = resolve_viewer_session(session, input_root, mapping_root=mapping_root)

    assert resolved.mapping_candidate is not None
    assert resolved.mapping_candidate.root_relative == "tinted.optical-mapping.json"
    verify_derived_optical(resolved, derived_path)


def test_verify_derived_optical_rejects_mismatched_bytes(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    mapping_root = tmp_path / "mappings"
    input_root.mkdir()
    _write_bundle(input_root / "bundle")
    candidate = resolve_candidate(input_root, Path("bundle"))

    mapping_path = mapping_root / "tinted.optical-mapping.json"
    _write_mapping_document(mapping_path)
    mapping_digest = phase0_provisional_mapping().digest

    derived_path = tmp_path / "derived" / "optical.zarr"
    _write_optical(derived_path)
    derived_digest = zarr_store_sha256(derived_path)

    session = create_viewer_session(
        candidate,
        StageConfig(),
        "llvm_ad_rgb",
        1,
        mapping=SessionMappingRef(
            path="tinted.optical-mapping.json",
            digest=mapping_digest,
            derived_optical_sha256=derived_digest,
        ),
    )
    resolved = resolve_viewer_session(session, input_root, mapping_root=mapping_root)
    (derived_path / "changed-marker").write_text("changed")

    with pytest.raises(ViewerSessionError, match="derived optical digest mismatch"):
        verify_derived_optical(resolved, derived_path)


def test_verify_derived_optical_requires_mapping_reference(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    _write_optical(input_root / "model.zarr")
    candidate = resolve_candidate(input_root, Path("model.zarr"))
    session = create_viewer_session(candidate, StageConfig(), "llvm_ad_rgb", 1)
    resolved = resolve_viewer_session(session, input_root)

    with pytest.raises(ViewerSessionError, match="no mapping reference to verify"):
        verify_derived_optical(resolved, input_root / "model.zarr")


def test_resolver_requires_mapping_root_for_reference(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    _write_bundle(input_root / "bundle")
    candidate = resolve_candidate(input_root, Path("bundle"))
    session = create_viewer_session(
        candidate,
        StageConfig(),
        "llvm_ad_rgb",
        1,
        mapping=SessionMappingRef(
            path="tinted.optical-mapping.json",
            digest=_DIGEST_A,
            derived_optical_sha256=_DIGEST_B,
        ),
    )

    with pytest.raises(ViewerSessionError, match="requires --mapping-root"):
        resolve_viewer_session(session, input_root)


def test_resolver_rejects_modified_mapping_file(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    mapping_root = tmp_path / "mappings"
    input_root.mkdir()
    _write_bundle(input_root / "bundle")
    candidate = resolve_candidate(input_root, Path("bundle"))
    mapping_path = mapping_root / "tinted.optical-mapping.json"
    _write_mapping_document(mapping_path)
    mapping_digest = phase0_provisional_mapping().digest
    session = create_viewer_session(
        candidate,
        StageConfig(),
        "llvm_ad_rgb",
        1,
        mapping=SessionMappingRef(
            path="tinted.optical-mapping.json",
            digest=mapping_digest,
            derived_optical_sha256=_DIGEST_B,
        ),
    )
    document = json.loads(mapping_path.read_text())
    document["materials"][0]["sigma_a_rgb_per_m"] = [9.0, 9.0, 9.0]
    mapping_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ViewerSessionError, match="mapping digest mismatch"):
        resolve_viewer_session(session, input_root, mapping_root=mapping_root)


def test_atomic_writer_round_trips_and_creates_parent(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "viewer.session.json"
    session = _standalone_session()

    write_viewer_session(path, session)

    assert viewer_session_from_json(path) == session
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert list(path.parent.glob(".*.tmp")) == []


def test_atomic_writer_preserves_existing_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "viewer.session.json"
    path.write_text("old\n", encoding="utf-8")

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(session_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_viewer_session(path, _standalone_session())

    assert path.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.glob(".*.tmp")) == []
