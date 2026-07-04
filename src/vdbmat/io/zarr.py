"""Failure-safe Zarr persistence for canonical Phase 0 volumes."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

from vbdmat.core.geometry import GridGeometry
from vbdmat.core.materials import MaterialDefinition, MaterialPalette, MaterialRole
from vbdmat.core.metadata import VOLUME_SCHEMA, Provenance, SchemaVersion
from vbdmat.core.optical_basis import OpticalBasis, OpticalBasisKind
from vbdmat.core.transforms import Matrix4
from vbdmat.core.volumes import (
    MaterialLabelVolume,
    MaterialMixtureVolume,
    OpticalPropertyVolume,
    VolumeAssetType,
)

from .errors import VolumeIOError

CanonicalVolume: TypeAlias = (
    MaterialLabelVolume | MaterialMixtureVolume | OpticalPropertyVolume
)
PathLike: TypeAlias = str | os.PathLike[str]
RegionZYX: TypeAlias = tuple[slice, slice, slice]

_MANIFEST_ATTRIBUTE = "vbdmat"
_LENGTH_UNIT = "m"
_FORMAT_NAME = "zarr-v3-directory"
_COMPRESSOR = BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.bitshuffle)


@dataclass(frozen=True, slots=True)
class ArrayInspection:
    """Metadata-only description of one persisted array."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    dimensions: tuple[str, ...]
    chunks: tuple[int, ...]
    unit: str


@dataclass(frozen=True, slots=True)
class VolumeInspection:
    """Metadata-only description of a persisted volume."""

    asset_type: VolumeAssetType
    schema_name: str
    schema_version: str
    geometry: GridGeometry
    arrays: tuple[ArrayInspection, ...]


def write_volume(
    path: PathLike, volume: CanonicalVolume, *, overwrite: bool = False
) -> None:
    """Write a validated volume through a temporary sibling directory."""
    if not isinstance(
        volume, (MaterialLabelVolume, MaterialMixtureVolume, OpticalPropertyVolume)
    ):
        raise TypeError("volume must be a canonical volume object")

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = target.with_name(f".{target.name}.tmp-{token}")
    backup = target.with_name(f".{target.name}.bak-{token}")

    try:
        root = zarr.open_group(temporary, mode="w", zarr_format=3)
        manifest = _manifest_for(volume)
        root.attrs[_MANIFEST_ATTRIBUTE] = manifest
        arrays = root.create_group("arrays")
        for name, data in _volume_arrays(volume).items():
            declaration = cast(dict[str, Any], manifest["arrays"])[name]
            arrays.create_array(
                name,
                data=data,
                chunks=_chunk_shape(data.shape),
                compressors=[_COMPRESSOR],
                dimension_names=tuple(declaration["dimensions"]),
                attributes={
                    "dimensions": declaration["dimensions"],
                    "unit": declaration["unit"],
                },
            )
        inspect_volume(temporary)

        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(temporary, target)
        except BaseException:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists() and target.exists():
            shutil.rmtree(backup)


def inspect_volume(path: PathLike) -> VolumeInspection:
    """Inspect schema, geometry, declarations, and chunk metadata only."""
    root = _open_root(path)
    manifest = _read_manifest(root)
    asset_type = _asset_type(manifest)
    schema_name, schema_version = _schema_metadata(manifest)
    geometry = _geometry(manifest)
    declarations = _mapping(manifest.get("arrays"), "vbdmat.arrays")
    arrays_group = _required_group(root, "arrays")

    inspected: list[ArrayInspection] = []
    for name, expected in _required_fields(asset_type).items():
        declaration = _mapping(declarations.get(name), f"vbdmat.arrays.{name}")
        array = _required_array(arrays_group, name)
        dimensions = _string_tuple(
            declaration.get("dimensions"), f"vbdmat.arrays.{name}.dimensions"
        )
        unit = _string(declaration.get("unit"), f"vbdmat.arrays.{name}.unit")
        dtype = _string(declaration.get("dtype"), f"vbdmat.arrays.{name}.dtype")
        declared_shape = _integer_tuple(
            declaration.get("shape"), f"vbdmat.arrays.{name}.shape"
        )
        if dimensions != expected[0]:
            raise VolumeIOError(
                f"vbdmat.arrays.{name}.dimensions", f"expected {expected[0]}"
            )
        if unit != expected[1]:
            raise VolumeIOError(
                f"vbdmat.arrays.{name}.unit", f"expected {expected[1]!r}"
            )
        if dtype != expected[2] or np.dtype(array.dtype).name != expected[2]:
            raise VolumeIOError(
                f"arrays.{name}.dtype", f"expected canonical dtype {expected[2]}"
            )
        actual_shape = tuple(int(item) for item in array.shape)
        expected_shape = _expected_shape(asset_type, name, geometry, manifest)
        if declared_shape != actual_shape or actual_shape != expected_shape:
            raise VolumeIOError(
                f"arrays.{name}.shape",
                f"declared, stored, and expected shapes must equal {expected_shape}",
            )
        attrs_dimensions = _string_tuple(
            array.attrs.get("dimensions"), f"arrays.{name}.attrs.dimensions"
        )
        attrs_unit = _string(array.attrs.get("unit"), f"arrays.{name}.attrs.unit")
        if attrs_dimensions != dimensions or attrs_unit != unit:
            raise VolumeIOError(
                f"arrays.{name}.attrs", "must match the manifest declaration"
            )
        inspected.append(
            ArrayInspection(
                name=name,
                shape=actual_shape,
                dtype=np.dtype(array.dtype).name,
                dimensions=dimensions,
                chunks=tuple(int(item) for item in array.chunks),
                unit=unit,
            )
        )

    return VolumeInspection(
        asset_type=asset_type,
        schema_name=schema_name,
        schema_version=schema_version,
        geometry=geometry,
        arrays=tuple(inspected),
    )


def read_volume(path: PathLike) -> CanonicalVolume:
    """Read and fully validate one canonical volume."""
    root, manifest, inspection = _validated_asset(path)
    _require_runtime_schema(inspection.schema_name, inspection.schema_version)
    arrays = _required_group(root, "arrays")
    provenance = _provenance(manifest)
    if inspection.asset_type is VolumeAssetType.MATERIAL_LABEL:
        return MaterialLabelVolume(
            geometry=inspection.geometry,
            provenance=provenance,
            palette=_palette(manifest),
            material_id=np.asarray(_required_array(arrays, "material_id")[:]),
        )
    if inspection.asset_type is VolumeAssetType.MATERIAL_MIXTURE:
        return MaterialMixtureVolume(
            geometry=inspection.geometry,
            provenance=provenance,
            palette=_palette(manifest),
            fractions=np.asarray(_required_array(arrays, "fractions")[:]),
            material_ids=np.asarray(_required_array(arrays, "material_ids")[:]),
        )
    return OpticalPropertyVolume(
        geometry=inspection.geometry,
        provenance=provenance,
        optical_basis=_optical_basis(manifest),
        sigma_a=np.asarray(_required_array(arrays, "sigma_a")[:]),
        sigma_s=np.asarray(_required_array(arrays, "sigma_s")[:]),
        g=np.asarray(_required_array(arrays, "g")[:]),
        ior=np.asarray(_required_array(arrays, "ior")[:]),
    )


def read_optical_region(path: PathLike, region_zyx: RegionZYX) -> OpticalPropertyVolume:
    """Read a non-empty unit-stride optical region and preserve world placement."""
    root, manifest, inspection = _validated_asset(path)
    _require_runtime_schema(inspection.schema_name, inspection.schema_version)
    if inspection.asset_type is not VolumeAssetType.OPTICAL_PROPERTY:
        raise VolumeIOError("vbdmat.asset_type", "must be 'optical-property'")
    normalized, starts, shape = _normalize_region(
        region_zyx, inspection.geometry.shape_zyx
    )
    arrays = _required_group(root, "arrays")
    coefficient_region = (*normalized, slice(None))
    geometry = _region_geometry(inspection.geometry, starts, shape)
    return OpticalPropertyVolume(
        geometry=geometry,
        provenance=_provenance(manifest),
        optical_basis=_optical_basis(manifest),
        sigma_a=np.asarray(_required_array(arrays, "sigma_a")[coefficient_region]),
        sigma_s=np.asarray(_required_array(arrays, "sigma_s")[coefficient_region]),
        g=np.asarray(_required_array(arrays, "g")[normalized]),
        ior=np.asarray(_required_array(arrays, "ior")[normalized]),
    )


def _validated_asset(
    path: PathLike,
) -> tuple[zarr.Group, Mapping[str, Any], VolumeInspection]:
    root = _open_root(path)
    manifest = _read_manifest(root)
    return root, manifest, inspect_volume(path)


def _open_root(path: PathLike) -> zarr.Group:
    try:
        return zarr.open_group(Path(path), mode="r", zarr_format=3)
    except Exception as error:
        raise VolumeIOError("store", f"cannot open Zarr v3 group: {error}") from error


def _read_manifest(root: zarr.Group) -> Mapping[str, Any]:
    return _mapping(root.attrs.get(_MANIFEST_ATTRIBUTE), _MANIFEST_ATTRIBUTE)


def _manifest_for(volume: CanonicalVolume) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "format": _FORMAT_NAME,
        "schema": {"name": volume.schema.name, "version": str(volume.schema.version)},
        "asset_type": volume.asset_type.value,
        "geometry": {
            "shape_zyx": list(volume.geometry.shape_zyx),
            "voxel_size_xyz_m": list(volume.geometry.voxel_size_xyz_m),
            "local_to_world": [list(row) for row in volume.geometry.local_to_world],
            "length_unit": _LENGTH_UNIT,
        },
        "provenance": _provenance_json(volume.provenance),
        "arrays": {},
    }
    if isinstance(volume, (MaterialLabelVolume, MaterialMixtureVolume)):
        manifest["palette"] = [_material_json(item) for item in volume.palette]
    else:
        manifest["optical_basis"] = _basis_json(volume.optical_basis)

    declarations: dict[str, Any] = {}
    required = _required_fields(volume.asset_type)
    for name, data in _volume_arrays(volume).items():
        dimensions, unit, dtype = required[name]
        declarations[name] = {
            "dtype": dtype,
            "dimensions": list(dimensions),
            "shape": list(data.shape),
            "unit": unit,
        }
    manifest["arrays"] = declarations
    return manifest


def _required_fields(
    asset_type: VolumeAssetType,
) -> dict[str, tuple[tuple[str, ...], str, str]]:
    if asset_type is VolumeAssetType.MATERIAL_LABEL:
        return {"material_id": (("z", "y", "x"), "1", "uint16")}
    if asset_type is VolumeAssetType.MATERIAL_MIXTURE:
        return {
            "fractions": (("z", "y", "x", "material"), "1", "float32"),
            "material_ids": (("material",), "1", "uint16"),
        }
    return {
        "sigma_a": (("z", "y", "x", "basis"), "m^-1", "float32"),
        "sigma_s": (("z", "y", "x", "basis"), "m^-1", "float32"),
        "g": (("z", "y", "x"), "1", "float32"),
        "ior": (("z", "y", "x"), "1", "float32"),
    }


def _volume_arrays(volume: CanonicalVolume) -> dict[str, np.ndarray[Any, Any]]:
    if isinstance(volume, MaterialLabelVolume):
        return {"material_id": volume.material_id}
    if isinstance(volume, MaterialMixtureVolume):
        return {"fractions": volume.fractions, "material_ids": volume.material_ids}
    return {
        "sigma_a": volume.sigma_a,
        "sigma_s": volume.sigma_s,
        "g": volume.g,
        "ior": volume.ior,
    }


def _expected_shape(
    asset_type: VolumeAssetType,
    name: str,
    geometry: GridGeometry,
    manifest: Mapping[str, Any],
) -> tuple[int, ...]:
    spatial = geometry.shape_zyx
    if name == "material_ids":
        return (len(_palette(manifest)),)
    if name == "fractions":
        return (*spatial, len(_palette(manifest)))
    if name in {"sigma_a", "sigma_s"}:
        return (*spatial, _optical_basis(manifest).size)
    return spatial


def _chunk_shape(shape: Sequence[int]) -> tuple[int, ...]:
    spatial_count = min(3, len(shape))
    spatial = tuple(min(int(item), 2) for item in shape[:spatial_count])
    return (*spatial, *(int(item) for item in shape[spatial_count:]))


def _asset_type(manifest: Mapping[str, Any]) -> VolumeAssetType:
    value = _string(manifest.get("asset_type"), "vbdmat.asset_type")
    try:
        return VolumeAssetType(value)
    except ValueError as error:
        raise VolumeIOError(
            "vbdmat.asset_type", f"unsupported value {value!r}"
        ) from error


def _schema_metadata(manifest: Mapping[str, Any]) -> tuple[str, str]:
    schema = _mapping(manifest.get("schema"), "vbdmat.schema")
    name = _string(schema.get("name"), "vbdmat.schema.name")
    version_text = _string(schema.get("version"), "vbdmat.schema.version")
    try:
        version = SchemaVersion.parse(version_text)
    except (TypeError, ValueError) as error:
        raise VolumeIOError("vbdmat.schema.version", str(error)) from error
    if name != VOLUME_SCHEMA.name:
        raise VolumeIOError("vbdmat.schema.name", f"unsupported schema {name!r}")
    if version.major != VOLUME_SCHEMA.version.major:
        raise VolumeIOError(
            "vbdmat.schema.version",
            f"incompatible major version {version.major}; "
            f"expected {VOLUME_SCHEMA.version.major}",
        )
    return name, version_text


def _require_runtime_schema(name: str, version_text: str) -> None:
    version = SchemaVersion.parse(version_text)
    if name != VOLUME_SCHEMA.name or version.minor != VOLUME_SCHEMA.version.minor:
        raise VolumeIOError(
            "vbdmat.schema.version",
            f"runtime supports {VOLUME_SCHEMA.name} 1.0.x; found {name} {version}",
        )


def _geometry(manifest: Mapping[str, Any]) -> GridGeometry:
    value = _mapping(manifest.get("geometry"), "vbdmat.geometry")
    unit = _string(value.get("length_unit"), "vbdmat.geometry.length_unit")
    if unit != _LENGTH_UNIT:
        raise VolumeIOError("vbdmat.geometry.length_unit", "must be 'm'")
    try:
        return GridGeometry(
            shape_zyx=cast(
                tuple[int, int, int],
                _integer_tuple(value.get("shape_zyx"), "geometry.shape_zyx"),
            ),
            voxel_size_xyz_m=cast(
                tuple[float, float, float],
                _number_tuple(
                    value.get("voxel_size_xyz_m"), "geometry.voxel_size_xyz_m"
                ),
            ),
            local_to_world=cast(
                Matrix4,
                tuple(
                    _number_tuple(row, f"geometry.local_to_world[{index}]")
                    for index, row in enumerate(
                        _sequence(
                            value.get("local_to_world"), "geometry.local_to_world"
                        )
                    )
                ),
            ),
        )
    except (TypeError, ValueError) as error:
        raise VolumeIOError("vbdmat.geometry", str(error)) from error


def _provenance(manifest: Mapping[str, Any]) -> Provenance:
    value = _mapping(manifest.get("provenance"), "vbdmat.provenance")
    created = value.get("created_utc")
    try:
        return Provenance(
            generator=_string(value.get("generator"), "provenance.generator"),
            generator_version=_string(
                value.get("generator_version"), "provenance.generator_version"
            ),
            created_utc=datetime.fromisoformat(created)
            if created is not None
            else None,
            configuration_digest=cast(str | None, value.get("configuration_digest")),
            sources=_string_tuple(value.get("sources", []), "provenance.sources"),
            notes=cast(str | None, value.get("notes")),
        )
    except (TypeError, ValueError) as error:
        raise VolumeIOError("vbdmat.provenance", str(error)) from error


def _palette(manifest: Mapping[str, Any]) -> MaterialPalette:
    entries = _sequence(manifest.get("palette"), "vbdmat.palette")
    materials: list[MaterialDefinition] = []
    try:
        for index, item in enumerate(entries):
            value = _mapping(item, f"vbdmat.palette[{index}]")
            materials.append(
                MaterialDefinition(
                    material_id=_integer(
                        value.get("material_id"), f"palette[{index}].material_id"
                    ),
                    name=_string(value.get("name"), f"palette[{index}].name"),
                    role=MaterialRole(
                        _string(value.get("role"), f"palette[{index}].role")
                    ),
                    external_id=cast(str | None, value.get("external_id")),
                    metadata=cast(Mapping[str, object], value.get("metadata", {})),
                )
            )
        return MaterialPalette.from_sequence(materials)
    except (TypeError, ValueError) as error:
        raise VolumeIOError("vbdmat.palette", str(error)) from error


def _optical_basis(manifest: Mapping[str, Any]) -> OpticalBasis:
    value = _mapping(manifest.get("optical_basis"), "vbdmat.optical_basis")
    try:
        return OpticalBasis(
            kind=OpticalBasisKind(_string(value.get("kind"), "basis.kind")),
            identifier=_string(value.get("identifier"), "basis.identifier"),
            coordinates=tuple(_sequence(value.get("coordinates"), "basis.coordinates")),
            reference_white=cast(str | None, value.get("reference_white")),
            observer=cast(str | None, value.get("observer")),
            transfer=cast(str | None, value.get("transfer")),
        )
    except (TypeError, ValueError) as error:
        raise VolumeIOError("vbdmat.optical_basis", str(error)) from error


def _normalize_region(
    region: RegionZYX, shape: tuple[int, int, int]
) -> tuple[RegionZYX, tuple[int, int, int], tuple[int, int, int]]:
    if not isinstance(region, tuple) or len(region) != 3:
        raise TypeError("region_zyx must be a tuple of three slices")
    normalized: list[slice] = []
    starts: list[int] = []
    sizes: list[int] = []
    for axis, item, extent in zip(("z", "y", "x"), region, shape, strict=True):
        if not isinstance(item, slice):
            raise TypeError(f"region_zyx.{axis} must be a slice")
        start, stop, step = item.indices(extent)
        if step != 1:
            raise ValueError(f"region_zyx.{axis} must have unit positive stride")
        if start >= stop:
            raise ValueError(f"region_zyx.{axis} must select at least one cell")
        normalized.append(slice(start, stop))
        starts.append(start)
        sizes.append(stop - start)
    return (
        cast(RegionZYX, tuple(normalized)),
        cast(tuple[int, int, int], tuple(starts)),
        cast(tuple[int, int, int], tuple(sizes)),
    )


def _region_geometry(
    geometry: GridGeometry,
    starts_zyx: tuple[int, int, int],
    shape_zyx: tuple[int, int, int],
) -> GridGeometry:
    z, y, x = starts_zyx
    sx, sy, sz = geometry.voxel_size_xyz_m
    matrix = np.asarray(geometry.local_to_world, dtype=np.float64)
    offset = matrix[:3, :3] @ np.asarray((x * sx, y * sy, z * sz))
    translated = matrix.copy()
    translated[:3, 3] += offset
    return GridGeometry(
        shape_zyx=shape_zyx,
        voxel_size_xyz_m=geometry.voxel_size_xyz_m,
        local_to_world=cast(
            Matrix4, tuple(tuple(float(v) for v in row) for row in translated)
        ),
    )


def _provenance_json(value: Provenance) -> dict[str, Any]:
    result: dict[str, Any] = {
        "generator": value.generator,
        "generator_version": value.generator_version,
        "sources": list(value.sources),
    }
    if value.created_utc is not None:
        result["created_utc"] = value.created_utc.isoformat()
    if value.configuration_digest is not None:
        result["configuration_digest"] = value.configuration_digest
    if value.notes is not None:
        result["notes"] = value.notes
    return result


def _material_json(value: MaterialDefinition) -> dict[str, Any]:
    return {
        "material_id": value.material_id,
        "name": value.name,
        "role": value.role.value,
        "external_id": value.external_id,
        "metadata": _json_value(value.metadata),
    }


def _basis_json(value: OpticalBasis) -> dict[str, Any]:
    return {
        "kind": value.kind.value,
        "identifier": value.identifier,
        "coordinates": list(value.coordinates),
        "reference_white": value.reference_white,
        "observer": value.observer,
        "transfer": value.transfer,
    }


def _json_value(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _required_group(root: zarr.Group, name: str) -> zarr.Group:
    try:
        item = root[name]
    except KeyError as error:
        raise VolumeIOError(name, "required group is missing") from error
    if not isinstance(item, zarr.Group):
        raise VolumeIOError(name, "must be a group")
    return item


def _required_array(group: zarr.Group, name: str) -> zarr.Array[Any]:
    try:
        item = group[name]
    except KeyError as error:
        raise VolumeIOError(f"arrays.{name}", "required array is missing") from error
    if not isinstance(item, zarr.Array):
        raise VolumeIOError(f"arrays.{name}", "must be an array")
    return item


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VolumeIOError(field, "must be an object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise VolumeIOError(field, "must be an array")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise VolumeIOError(field, "must be a string")
    return value


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    items = _sequence(value, field)
    if any(not isinstance(item, str) for item in items):
        raise VolumeIOError(field, "must contain only strings")
    return cast(tuple[str, ...], tuple(items))


def _integer_tuple(value: object, field: str) -> tuple[int, ...]:
    items = _sequence(value, field)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in items):
        raise VolumeIOError(field, "must contain only integers")
    return tuple(int(item) for item in items)


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VolumeIOError(field, "must be an integer")
    return value


def _number_tuple(value: object, field: str) -> tuple[float, ...]:
    items = _sequence(value, field)
    try:
        return tuple(float(item) for item in items)
    except (TypeError, ValueError) as error:
        raise VolumeIOError(field, "must contain only numbers") from error
