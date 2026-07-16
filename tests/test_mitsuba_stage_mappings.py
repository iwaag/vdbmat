from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).parents[1] / "examples" / "pipeline_run" / "demo"
sys.path.insert(0, str(DEMO_DIR))

from mitsuba_stage_mappings import (  # noqa: E402
    MappingCandidate,
    MappingCatalogError,
    describe_mapping,
    load_mapping,
    resolve_mapping_candidate,
    resolve_mapping_root,
    scan_mapping_catalog,
)

_BUILTIN_MAPPING = (
    Path(__file__).parents[1]
    / "examples"
    / "pipeline_run"
    / "mappings"
    / "phase0-provisional-materials-v1.optical-mapping.json"
)


def _document(**updates: object) -> dict[str, object]:
    document = json.loads(_BUILTIN_MAPPING.read_text(encoding="utf-8"))
    document.update(updates)
    return document


def _write(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_scan_lists_only_mapping_json_in_relative_path_order(tmp_path: Path) -> None:
    _write(tmp_path / "zeta.optical-mapping.json", _document())
    _write(tmp_path / "alpha" / "nested.optical-mapping.json", _document())
    _write(tmp_path / "ignored.json", _document())
    (tmp_path / "not-a-file.optical-mapping.json").mkdir()

    catalog = scan_mapping_catalog(tmp_path)

    assert [candidate.root_relative for candidate in catalog] == [
        "alpha/nested.optical-mapping.json",
        "zeta.optical-mapping.json",
    ]


def test_scan_keeps_invalid_document_for_deferred_diagnostics(tmp_path: Path) -> None:
    path = tmp_path / "broken.optical-mapping.json"
    path.write_text("{", encoding="utf-8")

    catalog = scan_mapping_catalog(tmp_path)

    assert [candidate.root_relative for candidate in catalog] == [
        "broken.optical-mapping.json"
    ]
    with pytest.raises(MappingCatalogError, match="invalid JSON"):
        describe_mapping(catalog[0])


def test_scan_excludes_file_symlink_escaping_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.optical-mapping.json"
    _write(outside, _document())
    (root / "escape.optical-mapping.json").symlink_to(outside)

    assert scan_mapping_catalog(root) == []


def test_scan_and_resolve_accept_file_symlink_staying_inside_root(
    tmp_path: Path,
) -> None:
    target = tmp_path / "actual.optical-mapping.json"
    _write(target, _document())
    link = tmp_path / "alias.optical-mapping.json"
    link.symlink_to(target)

    catalog = scan_mapping_catalog(tmp_path)
    candidate = resolve_mapping_candidate(tmp_path, Path("alias.optical-mapping.json"))

    assert [item.root_relative for item in catalog] == [
        "actual.optical-mapping.json",
        "alias.optical-mapping.json",
    ]
    assert candidate == MappingCandidate(
        root_relative="alias.optical-mapping.json", path=target.resolve()
    )


@pytest.mark.parametrize(
    ("user_path", "message"),
    [
        (Path("/tmp/absolute.optical-mapping.json"), "must be relative"),
        (Path("../parent.optical-mapping.json"), "parent traversal"),
        (Path("wrong.json"), "must end with"),
        (Path("."), "must not be empty"),
    ],
)
def test_resolve_rejects_invalid_lexical_paths(
    tmp_path: Path, user_path: Path, message: str
) -> None:
    with pytest.raises(MappingCatalogError, match=message):
        resolve_mapping_candidate(tmp_path, user_path)


def test_resolve_rejects_missing_mapping(tmp_path: Path) -> None:
    with pytest.raises(MappingCatalogError, match="does not exist"):
        resolve_mapping_candidate(tmp_path, Path("missing.optical-mapping.json"))


def test_resolve_rejects_symlink_escaping_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.optical-mapping.json"
    _write(outside, _document())
    (root / "escape.optical-mapping.json").symlink_to(outside)

    with pytest.raises(MappingCatalogError, match="outside --mapping-root"):
        resolve_mapping_candidate(root, Path("escape.optical-mapping.json"))


def test_load_rejects_wrong_format_and_version(tmp_path: Path) -> None:
    wrong_format = tmp_path / "wrong-format.optical-mapping.json"
    wrong_version = tmp_path / "wrong-version.optical-mapping.json"
    _write(wrong_format, _document(format="not-a-mapping"))
    _write(wrong_version, _document(format_version="9.0.0"))

    with pytest.raises(
        MappingCatalogError, match=r"must be 'vdbmat\.optical-mapping'"
    ):
        load_mapping(resolve_mapping_candidate(tmp_path, Path(wrong_format.name)))
    with pytest.raises(MappingCatalogError, match="unsupported major version"):
        load_mapping(resolve_mapping_candidate(tmp_path, Path(wrong_version.name)))


def test_load_rejects_external_id(tmp_path: Path) -> None:
    document = _document()
    document["materials"][0]["external_id"] = "catalog-sku-123"
    path = tmp_path / "external-id.optical-mapping.json"
    _write(path, document)

    with pytest.raises(MappingCatalogError, match="external_id"):
        load_mapping(resolve_mapping_candidate(tmp_path, Path(path.name)))


def test_load_rejects_missing_material_field(tmp_path: Path) -> None:
    document = _document()
    del document["materials"][0]["ior"]
    path = tmp_path / "missing-field.optical-mapping.json"
    _write(path, document)

    with pytest.raises(MappingCatalogError, match="missing fields"):
        load_mapping(resolve_mapping_candidate(tmp_path, Path(path.name)))


def test_describe_reports_summary_fields(tmp_path: Path) -> None:
    path = tmp_path / "builtin.optical-mapping.json"
    _write(path, _document())

    summary = describe_mapping(resolve_mapping_candidate(tmp_path, Path(path.name)))

    assert summary.configuration_id == "phase0-provisional-materials-v1"
    assert summary.version == "1.0.0"
    assert summary.calibration_status == "provisional-uncalibrated"
    assert summary.digest.startswith("sha256:")
    assert (0, "air") in summary.materials
    assert (1, "transparent-resin") in summary.materials


def test_describe_reflects_current_file_content_across_calls(tmp_path: Path) -> None:
    path = tmp_path / "editable.optical-mapping.json"
    _write(path, _document())
    candidate = resolve_mapping_candidate(tmp_path, Path(path.name))
    first = describe_mapping(candidate)

    edited = _document()
    edited["materials"][1]["sigma_a_rgb_per_m"] = [9.0, 9.0, 9.0]
    _write(path, edited)
    second = describe_mapping(candidate)

    assert first.digest != second.digest


def test_resolve_root_prefers_explicit_directory(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit"
    explicit.mkdir()

    assert resolve_mapping_root(explicit) == explicit.resolve()


def test_resolve_root_defaults_to_builtin_mappings() -> None:
    builtin = resolve_mapping_root(None)
    assert builtin.name == "mappings"
    assert (builtin / "phase0-provisional-materials-v1.optical-mapping.json").is_file()


def test_resolve_root_rejects_non_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")

    with pytest.raises(MappingCatalogError, match="not a directory"):
        resolve_mapping_root(file_path)
