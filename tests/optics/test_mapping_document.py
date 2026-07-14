"""Tests for the ``vdbmat.optical-mapping`` external document (ADR-009 D3/D4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vdbmat.optics import (
    OpticalMappingError,
    load_optical_mapping,
    optical_mapping_from_json_dict,
    optical_mapping_to_json_dict,
    phase0_provisional_mapping,
    write_optical_mapping,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMMITTED = (
    _REPO_ROOT
    / "examples"
    / "pipeline_run"
    / "mappings"
    / "phase0-provisional-materials-v1.optical-mapping.json"
)


def _document() -> dict[str, Any]:
    return optical_mapping_to_json_dict(phase0_provisional_mapping())


def test_roundtrip_preserves_identity(tmp_path: Path) -> None:
    builtin = phase0_provisional_mapping()
    path = write_optical_mapping(tmp_path / "mapping.json", builtin)
    restored = load_optical_mapping(path)
    assert restored == builtin
    assert restored.digest == builtin.digest


def test_committed_example_equals_the_builtin() -> None:
    restored = load_optical_mapping(_COMMITTED)
    assert restored.digest == phase0_provisional_mapping().digest


def test_digest_is_independent_of_document_formatting(tmp_path: Path) -> None:
    # Reserialize with different whitespace/key order: identity must not change.
    shuffled = dict(reversed(list(_document().items())))
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(shuffled, indent=None), encoding="utf-8")
    assert load_optical_mapping(path).digest == phase0_provisional_mapping().digest


def test_external_id_is_rejected() -> None:
    document = _document()
    document["materials"][0]["external_id"] = "vendor-sku-1"
    with pytest.raises(OpticalMappingError) as info:
        optical_mapping_from_json_dict(document)
    assert "external_id" in info.value.field_path


@pytest.mark.parametrize(
    "mutate, field_fragment",
    [
        (lambda d: d.update(format="other"), "format"),
        (lambda d: d.update(format_version="2.0.0"), "format_version"),
        (lambda d: d.update(surprise=True), "mapping"),
        (lambda d: d.pop("materials"), "mapping"),
        (lambda d: d["optical_basis"].update(transfer="srgb"), "optical_basis"),
        (lambda d: d["materials"][0].update(g=2.0), "materials[0]"),
        (lambda d: d["materials"][0].pop("ior"), "materials[0]"),
        (lambda d: d.update(mixing_rule="unknown-rule"), "mapping"),
    ],
)
def test_field_violations_fail_loudly(mutate: Any, field_fragment: str) -> None:
    document = _document()
    mutate(document)
    with pytest.raises(OpticalMappingError) as info:
        optical_mapping_from_json_dict(document)
    assert field_fragment in info.value.field_path


def test_missing_file_reports_io_shape(tmp_path: Path) -> None:
    with pytest.raises(OpticalMappingError, match="file not found"):
        load_optical_mapping(tmp_path / "missing.json")
