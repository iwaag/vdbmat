"""Strict reader/writer for the ``vdbmat.optical-mapping`` interchange (ADR-009 D3).

An optical mapping supplied as data carries exactly the fields of
:class:`OpticalMappingConfig`, so a mapping's canonical JSON and digest are
independent of whether it was compiled in or loaded from a file. The reader never
repairs or defaults scientific values; every violation is a field-oriented failure.

Per ADR-009 D4, ``external_id`` (the physical printer-material catalog layer) must
never appear in a mapping document; materials are keyed by ``material_id`` with a
human-readable ``name`` only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from vdbmat.core import OpticalBasis, SchemaVersion

from .config import (
    MaterialOpticalProperties,
    OpticalMappingConfig,
    RGBTriple,
)
from .errors import OpticalMappingError

FORMAT_NAME = "vdbmat.optical-mapping"
SUPPORTED_MAJOR = 1

_ALLOWED_TOP_LEVEL = frozenset(
    {
        "format",
        "format_version",
        "configuration_id",
        "version",
        "optical_basis",
        "mixing_rule",
        "calibration_status",
        "materials",
    }
)
_ALLOWED_BASIS = frozenset(
    {"kind", "identifier", "coordinates", "reference_white", "observer", "transfer"}
)
_ALLOWED_MATERIAL = frozenset(
    {"material_id", "name", "sigma_a_rgb_per_m", "sigma_s_rgb_per_m", "g", "ior"}
)


def optical_mapping_to_json_dict(config: OpticalMappingConfig) -> dict[str, Any]:
    """Return the portable ``vdbmat.optical-mapping/1.0.0`` document of a mapping."""
    basis = config.optical_basis
    return {
        "format": FORMAT_NAME,
        "format_version": "1.0.0",
        "configuration_id": config.configuration_id,
        "version": str(config.version),
        "optical_basis": {
            "kind": basis.kind.value,
            "identifier": basis.identifier,
            "coordinates": list(basis.coordinates),
            "reference_white": basis.reference_white,
            "observer": basis.observer,
            "transfer": basis.transfer,
        },
        "mixing_rule": config.mixing_rule.value,
        "calibration_status": config.calibration_status.value,
        "materials": [
            {
                "material_id": item.material_id,
                "name": item.name,
                "sigma_a_rgb_per_m": list(item.sigma_a_rgb_per_m),
                "sigma_s_rgb_per_m": list(item.sigma_s_rgb_per_m),
                "g": item.g,
                "ior": item.ior,
            }
            for item in config.materials
        ],
    }


def write_optical_mapping(path: str | Path, config: OpticalMappingConfig) -> Path:
    """Write a mapping as a reviewable ``vdbmat.optical-mapping`` JSON document."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(optical_mapping_to_json_dict(config), indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def load_optical_mapping(path: str | Path) -> OpticalMappingConfig:
    """Read and validate a ``vdbmat.optical-mapping`` document into a config."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise OpticalMappingError("mapping", f"file not found: {source}") from error
    except UnicodeDecodeError as error:
        raise OpticalMappingError("mapping", "document must be UTF-8") from error
    except OSError as error:
        raise OpticalMappingError(
            "mapping", f"cannot read {source}: {error}"
        ) from error

    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise OpticalMappingError("mapping", f"invalid JSON: {error}") from error
    return optical_mapping_from_json_dict(document)


def optical_mapping_from_json_dict(document: Any) -> OpticalMappingConfig:
    """Validate a parsed mapping document and build the immutable config."""
    if not isinstance(document, Mapping):
        raise OpticalMappingError("mapping", "must be a JSON object")

    unknown = sorted(set(document) - _ALLOWED_TOP_LEVEL)
    if unknown:
        raise OpticalMappingError("mapping", f"unknown fields: {unknown}")
    missing = sorted(_ALLOWED_TOP_LEVEL - set(document))
    if missing:
        raise OpticalMappingError("mapping", f"missing fields: {missing}")

    if document["format"] != FORMAT_NAME:
        raise OpticalMappingError(
            "format", f"must be {FORMAT_NAME!r}, got {document['format']!r}"
        )
    format_version = _version(document["format_version"], field="format_version")
    if format_version.major != SUPPORTED_MAJOR:
        raise OpticalMappingError(
            "format_version",
            f"unsupported major version {format_version.major}; "
            f"this reader supports major {SUPPORTED_MAJOR}",
        )

    configuration_id = document["configuration_id"]
    if not isinstance(configuration_id, str) or not configuration_id.strip():
        raise OpticalMappingError(
            "configuration_id", "must be a non-empty string"
        )
    version = _version(document["version"], field="version")
    basis = _basis(document["optical_basis"])
    materials = _materials(document["materials"])

    mixing_rule = document["mixing_rule"]
    calibration_status = document["calibration_status"]
    if not isinstance(mixing_rule, str):
        raise OpticalMappingError("mixing_rule", "must be a string")
    if not isinstance(calibration_status, str):
        raise OpticalMappingError("calibration_status", "must be a string")

    try:
        return OpticalMappingConfig(
            configuration_id=configuration_id,
            version=version,
            materials=materials,
            optical_basis=basis,
            mixing_rule=mixing_rule,  # type: ignore[arg-type]
            calibration_status=calibration_status,  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as error:
        raise OpticalMappingError("mapping", str(error)) from error


def _version(raw: Any, *, field: str) -> SchemaVersion:
    if not isinstance(raw, str):
        raise OpticalMappingError(field, "must be a string")
    try:
        return SchemaVersion.parse(raw)
    except (TypeError, ValueError) as error:
        raise OpticalMappingError(field, str(error)) from error


def _basis(raw: Any) -> OpticalBasis:
    if not isinstance(raw, Mapping):
        raise OpticalMappingError("optical_basis", "must be a JSON object")
    unknown = sorted(set(raw) - _ALLOWED_BASIS)
    if unknown:
        raise OpticalMappingError("optical_basis", f"unknown fields: {unknown}")
    expected = OpticalBasis.phase0_rgb()
    declared = {
        "kind": raw.get("kind"),
        "identifier": raw.get("identifier"),
        "coordinates": tuple(raw.get("coordinates", ())),
        "reference_white": raw.get("reference_white"),
        "observer": raw.get("observer"),
        "transfer": raw.get("transfer"),
    }
    actual = {
        "kind": expected.kind.value,
        "identifier": expected.identifier,
        "coordinates": expected.coordinates,
        "reference_white": expected.reference_white,
        "observer": expected.observer,
        "transfer": expected.transfer,
    }
    for field, value in actual.items():
        if declared[field] != value:
            raise OpticalMappingError(
                f"optical_basis.{field}",
                f"schema 1.x supports only the Phase 0 RGB basis; must be "
                f"{value!r}, got {declared[field]!r}",
            )
    return expected


def _materials(raw: Any) -> tuple[MaterialOpticalProperties, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise OpticalMappingError("materials", "must be an array")
    materials: list[MaterialOpticalProperties] = []
    for index, entry in enumerate(raw):
        field = f"materials[{index}]"
        if not isinstance(entry, Mapping):
            raise OpticalMappingError(field, "must be an object")
        if "external_id" in entry:
            raise OpticalMappingError(
                f"{field}.external_id",
                "physical catalog identifiers are not allowed in an optical "
                "mapping (ADR-009 D4)",
            )
        unknown = sorted(set(entry) - _ALLOWED_MATERIAL)
        if unknown:
            raise OpticalMappingError(field, f"unknown fields: {unknown}")
        missing = sorted(_ALLOWED_MATERIAL - set(entry))
        if missing:
            raise OpticalMappingError(field, f"missing fields: {missing}")
        try:
            materials.append(
                MaterialOpticalProperties(
                    material_id=entry["material_id"],
                    name=entry["name"],
                    sigma_a_rgb_per_m=_triple(entry["sigma_a_rgb_per_m"]),
                    sigma_s_rgb_per_m=_triple(entry["sigma_s_rgb_per_m"]),
                    g=entry["g"],
                    ior=entry["ior"],
                )
            )
        except (TypeError, ValueError) as error:
            raise OpticalMappingError(field, str(error)) from error
    return tuple(materials)


def _triple(raw: Any) -> RGBTriple:
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise TypeError("RGB coefficients must be an array of 3 numbers")
    return cast(RGBTriple, tuple(raw))
