"""Strict reader for the ``vbdmat.voxels`` direct material-voxel interchange.

Implements ADR-006: a UTF-8 JSON manifest plus one ``uint16[z, y, x]`` NumPy
payload is adapted into a canonical :class:`MaterialLabelVolume`. The reader never
transposes, casts, infers units, or remaps material IDs; every such situation is a
field-oriented failure rather than a silent repair.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np

from vbdmat.core import (
    GridGeometry,
    MaterialDefinition,
    MaterialLabelVolume,
    MaterialPalette,
    MaterialRole,
    Provenance,
    SchemaVersion,
)
from vbdmat.core.transforms import Matrix4

from .errors import VoxelManifestError

FORMAT_NAME = "vbdmat.voxels"
SUPPORTED_MAJOR = 1
ASSET_TYPE = "material-label"

_UNIT_TO_METRES = {"m": 1.0, "mm": 1.0e-3}
_ALLOWED_TOP_LEVEL = frozenset(
    {
        "format",
        "format_version",
        "asset_type",
        "payload",
        "shape_zyx",
        "voxel_size_xyz_m",
        "voxel_size",
        "local_to_world",
        "materials",
        "source",
    }
)
_ALLOWED_PAYLOAD = frozenset({"path", "sha256", "dtype", "dimensions"})
_ALLOWED_VOXEL_SIZE = frozenset({"value", "unit"})
_ALLOWED_SOURCE = frozenset({"generator", "generator_version", "identity", "notes"})
_ALLOWED_MATERIAL = frozenset(
    {"material_id", "name", "role", "external_id", "metadata"}
)


@dataclass(frozen=True, slots=True)
class ManifestInspection:
    """Metadata-only view of a direct-voxel manifest (no payload load)."""

    format_version: SchemaVersion
    shape_zyx: tuple[int, int, int]
    voxel_size_xyz_m: tuple[float, float, float]
    material_ids: tuple[int, ...]
    payload_path: str
    payload_sha256: str
    source_identity: str | None


def inspect_material_label_manifest(path: str | Path) -> ManifestInspection:
    """Return declared manifest metadata without reading or verifying the payload."""
    manifest, _ = _load_manifest(path)
    version = _parse_version(manifest)
    shape = _shape_zyx(manifest)
    voxel_size = _voxel_size_xyz_m(manifest)
    palette = _palette(manifest)
    payload = _mapping(manifest, "payload")
    return ManifestInspection(
        format_version=version,
        shape_zyx=shape,
        voxel_size_xyz_m=voxel_size,
        material_ids=palette.material_ids,
        payload_path=_string(payload, "payload.path"),
        payload_sha256=_hex_digest(payload, "payload.sha256"),
        source_identity=_source_identity(manifest),
    )


def read_material_label_manifest(path: str | Path) -> MaterialLabelVolume:
    """Read and validate a ``vbdmat.voxels`` manifest into a canonical volume."""
    manifest, manifest_path = _load_manifest(path)
    version = _parse_version(manifest)
    shape = _shape_zyx(manifest)
    voxel_size = _voxel_size_xyz_m(manifest)
    local_to_world = _local_to_world(manifest)
    palette = _palette(manifest)

    payload = _mapping(manifest, "payload")
    _require_dimensions(payload)
    declared_dtype = _string(payload, "payload.dtype")
    if declared_dtype != "uint16":
        raise VoxelManifestError(
            "payload.dtype", f"must be 'uint16', got {declared_dtype!r}"
        )
    declared_sha = _hex_digest(payload, "payload.sha256")

    payload_bytes = _read_payload_bytes(manifest_path.parent, payload)
    actual_sha = hashlib.sha256(payload_bytes).hexdigest()
    if actual_sha != declared_sha:
        raise VoxelManifestError(
            "payload.sha256",
            f"payload SHA-256 mismatch: declared {declared_sha}, actual {actual_sha}",
        )

    material_id = _load_payload_array(payload_bytes, shape)

    try:
        geometry = GridGeometry(
            shape_zyx=shape,
            voxel_size_xyz_m=voxel_size,
            local_to_world=local_to_world,
        )
    except (ValueError, TypeError) as error:
        raise VoxelManifestError("local_to_world", str(error)) from error
    provenance = _provenance(manifest, version, declared_sha)
    return MaterialLabelVolume(
        geometry=geometry,
        palette=palette,
        provenance=provenance,
        material_id=material_id,
    )


def write_material_label_manifest(
    directory: str | Path,
    name: str,
    volume: MaterialLabelVolume,
    *,
    identity: str | None = None,
) -> Path:
    """Write a canonical volume as a ``vbdmat.voxels`` manifest plus payload.

    This is the shared emitter for external input generators (ADR-009 D2): the
    payload is written as ``<name>.material_id.npy`` and the manifest as
    ``<name>.voxels.json`` under ``directory``, with the payload SHA-256 recorded
    in the manifest. Generator identity comes from the volume's provenance;
    ``identity`` optionally records the source-data identity. Output of this
    writer round-trips through :func:`read_material_label_manifest`.
    """
    if not isinstance(volume, MaterialLabelVolume):
        raise VoxelManifestError("volume", "must be a MaterialLabelVolume")
    if not name or "/" in name or "\\" in name:
        raise VoxelManifestError("name", "must be a non-empty single path component")
    if identity is not None and (not isinstance(identity, str) or not identity.strip()):
        raise VoxelManifestError("source.identity", "must be a non-empty string")

    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    payload_name = f"{name}.material_id.npy"

    buffer = io.BytesIO()
    np.save(buffer, np.asarray(volume.material_id, dtype=np.uint16))
    payload_bytes = buffer.getvalue()
    payload_sha = hashlib.sha256(payload_bytes).hexdigest()
    (base / payload_name).write_bytes(payload_bytes)

    materials: list[dict[str, Any]] = []
    for definition in volume.palette:
        entry: dict[str, Any] = {
            "material_id": definition.material_id,
            "name": definition.name,
            "role": definition.role.value,
        }
        if definition.external_id is not None:
            entry["external_id"] = definition.external_id
        if definition.metadata:
            entry["metadata"] = dict(definition.metadata)
        materials.append(entry)

    source: dict[str, Any] = {
        "generator": volume.provenance.generator,
        "generator_version": volume.provenance.generator_version,
    }
    if identity is not None:
        source["identity"] = identity
    if volume.provenance.notes is not None:
        source["notes"] = volume.provenance.notes

    manifest = {
        "format": FORMAT_NAME,
        "format_version": "1.0.0",
        "asset_type": ASSET_TYPE,
        "payload": {
            "path": payload_name,
            "sha256": payload_sha,
            "dtype": "uint16",
            "dimensions": ["z", "y", "x"],
        },
        "shape_zyx": list(volume.geometry.shape_zyx),
        "voxel_size_xyz_m": list(volume.geometry.voxel_size_xyz_m),
        "local_to_world": [list(row) for row in volume.geometry.local_to_world],
        "materials": materials,
        "source": source,
    }
    manifest_path = base / f"{name}.voxels.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def _load_manifest(path: str | Path) -> tuple[Mapping[str, Any], Path]:
    manifest_path = Path(path)
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise VoxelManifestError(
            "manifest", f"file not found: {manifest_path}"
        ) from error
    except UnicodeDecodeError as error:
        raise VoxelManifestError("manifest", "manifest must be UTF-8") from error
    except OSError as error:
        raise VoxelManifestError(
            "manifest", f"cannot read {manifest_path}: {error}"
        ) from error

    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise VoxelManifestError("manifest", f"invalid JSON: {error}") from error

    if not isinstance(document, dict):
        raise VoxelManifestError("manifest", "must be a JSON object")

    unknown = set(document) - _ALLOWED_TOP_LEVEL
    if unknown:
        raise VoxelManifestError("manifest", f"unknown fields: {sorted(unknown)}")
    required = {
        "format",
        "format_version",
        "asset_type",
        "payload",
        "shape_zyx",
        "local_to_world",
        "materials",
        "source",
    }
    missing = required - set(document)
    if missing:
        raise VoxelManifestError("manifest", f"missing fields: {sorted(missing)}")

    if document["format"] != FORMAT_NAME:
        raise VoxelManifestError(
            "format", f"must be {FORMAT_NAME!r}, got {document['format']!r}"
        )
    if document["asset_type"] != ASSET_TYPE:
        raise VoxelManifestError(
            "asset_type", f"must be {ASSET_TYPE!r}, got {document['asset_type']!r}"
        )
    return document, manifest_path


def _parse_version(manifest: Mapping[str, Any]) -> SchemaVersion:
    raw = manifest["format_version"]
    if not isinstance(raw, str):
        raise VoxelManifestError("format_version", "must be a string")
    try:
        version = SchemaVersion.parse(raw)
    except ValueError as error:
        raise VoxelManifestError("format_version", str(error)) from error
    if version.major != SUPPORTED_MAJOR:
        raise VoxelManifestError(
            "format_version",
            f"unsupported major version {version.major}; "
            f"this reader supports major {SUPPORTED_MAJOR}",
        )
    return version


def _shape_zyx(manifest: Mapping[str, Any]) -> tuple[int, int, int]:
    values = _sequence(manifest, "shape_zyx")
    if len(values) != 3:
        raise VoxelManifestError("shape_zyx", "must contain exactly 3 integers")
    shape: list[int] = []
    for axis, item in zip(("z", "y", "x"), values, strict=True):
        if isinstance(item, bool) or not isinstance(item, int):
            raise VoxelManifestError(f"shape_zyx.{axis}", "must be an integer")
        if item <= 0:
            raise VoxelManifestError(f"shape_zyx.{axis}", "must be greater than zero")
        shape.append(item)
    return (shape[0], shape[1], shape[2])


def _voxel_size_xyz_m(manifest: Mapping[str, Any]) -> tuple[float, float, float]:
    has_metres = "voxel_size_xyz_m" in manifest
    has_convenience = "voxel_size" in manifest
    if has_metres == has_convenience:
        raise VoxelManifestError(
            "voxel_size",
            "declare exactly one of 'voxel_size_xyz_m' or 'voxel_size'",
        )
    if has_metres:
        values = _sequence(manifest, "voxel_size_xyz_m")
        return _voxel_triplet(values, field="voxel_size_xyz_m", factor=1.0)

    block = _mapping(manifest, "voxel_size")
    unknown = set(block) - _ALLOWED_VOXEL_SIZE
    if unknown:
        raise VoxelManifestError("voxel_size", f"unknown fields: {sorted(unknown)}")
    unit = block.get("unit")
    if not isinstance(unit, str) or unit not in _UNIT_TO_METRES:
        raise VoxelManifestError(
            "voxel_size.unit",
            f"must be one of {sorted(_UNIT_TO_METRES)}, got {unit!r}",
        )
    values = _sequence(block, "voxel_size.value")
    return _voxel_triplet(
        values, field="voxel_size.value", factor=_UNIT_TO_METRES[unit]
    )


def _voxel_triplet(
    values: Sequence[Any], *, field: str, factor: float
) -> tuple[float, float, float]:
    if len(values) != 3:
        raise VoxelManifestError(field, "must contain exactly 3 numbers")
    result: list[float] = []
    for axis, item in zip(("x", "y", "z"), values, strict=True):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise VoxelManifestError(f"{field}.{axis}", "must be a number")
        value = float(item) * factor
        if not np.isfinite(value) or value <= 0.0:
            raise VoxelManifestError(
                f"{field}.{axis}", "must be finite and greater than zero"
            )
        result.append(value)
    return (result[0], result[1], result[2])


def _local_to_world(manifest: Mapping[str, Any]) -> Matrix4:
    rows = _sequence(manifest, "local_to_world")
    if len(rows) != 4:
        raise VoxelManifestError("local_to_world", "must be a 4x4 array")
    matrix: list[tuple[float, ...]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Sequence) or isinstance(row, str) or len(row) != 4:
            raise VoxelManifestError(
                f"local_to_world[{row_index}]", "must contain 4 numbers"
            )
        converted: list[float] = []
        for column_index, item in enumerate(row):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise VoxelManifestError(
                    f"local_to_world[{row_index}][{column_index}]",
                    "must be a number",
                )
            converted.append(float(item))
        matrix.append(tuple(converted))
    return cast(Matrix4, tuple(matrix))


def _palette(manifest: Mapping[str, Any]) -> MaterialPalette:
    entries = _sequence(manifest, "materials")
    definitions: list[MaterialDefinition] = []
    for index, entry in enumerate(entries):
        field = f"materials[{index}]"
        if not isinstance(entry, Mapping):
            raise VoxelManifestError(field, "must be an object")
        unknown = set(entry) - _ALLOWED_MATERIAL
        if unknown:
            raise VoxelManifestError(field, f"unknown fields: {sorted(unknown)}")
        material_id = entry.get("material_id")
        name = entry.get("name")
        role = entry.get("role")
        if isinstance(material_id, bool) or not isinstance(material_id, int):
            raise VoxelManifestError(f"{field}.material_id", "must be an integer")
        if not isinstance(name, str):
            raise VoxelManifestError(f"{field}.name", "must be a string")
        if not isinstance(role, str):
            raise VoxelManifestError(f"{field}.role", "must be a string")
        try:
            role_value = MaterialRole(role)
        except ValueError as error:
            raise VoxelManifestError(
                f"{field}.role", f"unsupported material role: {role!r}"
            ) from error
        external_id = entry.get("external_id")
        metadata = entry.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise VoxelManifestError(f"{field}.metadata", "must be an object")
        try:
            definitions.append(
                MaterialDefinition(
                    material_id=material_id,
                    name=name,
                    role=role_value,
                    external_id=external_id,
                    metadata=dict(metadata),
                )
            )
        except (ValueError, TypeError) as error:
            raise VoxelManifestError(field, str(error)) from error
    try:
        return MaterialPalette.from_sequence(definitions)
    except (ValueError, TypeError) as error:
        raise VoxelManifestError("materials", str(error)) from error


def _provenance(
    manifest: Mapping[str, Any], version: SchemaVersion, payload_sha: str
) -> Provenance:
    source = _mapping(manifest, "source")
    unknown = set(source) - _ALLOWED_SOURCE
    if unknown:
        raise VoxelManifestError("source", f"unknown fields: {sorted(unknown)}")
    generator = source.get("generator")
    generator_version = source.get("generator_version")
    if not isinstance(generator, str) or not generator.strip():
        raise VoxelManifestError("source.generator", "must be a non-empty string")
    if not isinstance(generator_version, str) or not generator_version.strip():
        raise VoxelManifestError(
            "source.generator_version", "must be a non-empty string"
        )
    notes = source.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise VoxelManifestError("source.notes", "must be a string")
    identity = _source_identity(manifest)
    sources: tuple[str, ...] = (
        f"{FORMAT_NAME}/{version}",
        f"sha256:{payload_sha}",
    )
    if identity is not None:
        sources = (*sources, f"identity:{identity}")
    try:
        return Provenance(
            generator=generator,
            generator_version=generator_version,
            sources=sources,
            notes=notes,
        )
    except (ValueError, TypeError) as error:
        raise VoxelManifestError("source", str(error)) from error


def _source_identity(manifest: Mapping[str, Any]) -> str | None:
    source = _mapping(manifest, "source")
    identity = source.get("identity")
    if identity is None:
        return None
    if not isinstance(identity, str) or not identity.strip():
        raise VoxelManifestError("source.identity", "must be a non-empty string")
    return identity


def _require_dimensions(payload: Mapping[str, Any]) -> None:
    unknown = set(payload) - _ALLOWED_PAYLOAD
    if unknown:
        raise VoxelManifestError("payload", f"unknown fields: {sorted(unknown)}")
    dimensions = payload.get("dimensions")
    if dimensions != ["z", "y", "x"]:
        raise VoxelManifestError(
            "payload.dimensions", "must be exactly ['z', 'y', 'x']"
        )


def _read_payload_bytes(manifest_dir: Path, payload: Mapping[str, Any]) -> bytes:
    relative = _string(payload, "payload.path")
    resolved = _resolve_payload_path(manifest_dir, relative)
    try:
        return resolved.read_bytes()
    except FileNotFoundError as error:
        raise VoxelManifestError(
            "payload.path", f"payload file not found: {relative}"
        ) from error
    except OSError as error:
        raise VoxelManifestError(
            "payload.path", f"cannot read payload {relative}: {error}"
        ) from error


def _resolve_payload_path(manifest_dir: Path, relative: str) -> Path:
    if not relative:
        raise VoxelManifestError("payload.path", "must be a non-empty relative path")
    if "\x00" in relative or "\\" in relative:
        raise VoxelManifestError("payload.path", "must be a POSIX relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or (len(relative) >= 2 and relative[1] == ":"):
        raise VoxelManifestError("payload.path", "must not be an absolute path")
    if any(part == ".." for part in pure.parts):
        raise VoxelManifestError(
            "payload.path", "must not traverse outside the manifest directory"
        )
    base = manifest_dir.resolve()
    resolved = (manifest_dir / pure).resolve()
    if resolved != base and base not in resolved.parents:
        raise VoxelManifestError(
            "payload.path", "resolved payload path escapes the manifest directory"
        )
    return resolved


def _load_payload_array(
    payload_bytes: bytes, shape: tuple[int, int, int]
) -> np.ndarray[Any, np.dtype[np.uint16]]:
    try:
        array = np.load(io.BytesIO(payload_bytes), allow_pickle=False)
    except ValueError as error:
        raise VoxelManifestError(
            "payload", f"payload is not a valid pickle-free .npy: {error}"
        ) from error
    if not isinstance(array, np.ndarray):
        raise VoxelManifestError("payload", "payload must be a NumPy array")
    if array.dtype != np.dtype(np.uint16):
        raise VoxelManifestError(
            "payload.dtype",
            f"payload array must be uint16, got {array.dtype}",
        )
    if array.shape != shape:
        raise VoxelManifestError(
            "payload.shape",
            f"payload shape {array.shape} does not match shape_zyx {shape}",
        )
    return array


def _mapping(container: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = container.get(field.rsplit(".", 1)[-1])
    if not isinstance(value, Mapping):
        raise VoxelManifestError(field, "must be an object")
    return value


def _sequence(container: Mapping[str, Any], field: str) -> Sequence[Any]:
    value = container.get(field.rsplit(".", 1)[-1])
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise VoxelManifestError(field, "must be an array")
    return value


def _string(container: Mapping[str, Any], field: str) -> str:
    value = container.get(field.rsplit(".", 1)[-1])
    if not isinstance(value, str) or not value:
        raise VoxelManifestError(field, "must be a non-empty string")
    return value


def _hex_digest(container: Mapping[str, Any], field: str) -> str:
    value = _string(container, field)
    lowered = value.lower()
    if len(lowered) != 64 or any(char not in "0123456789abcdef" for char in lowered):
        raise VoxelManifestError(field, "must be 64 lowercase hex digits")
    return lowered
