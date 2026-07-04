"""Cross-consumer contract checks for the Phase 0 renderer proofs."""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from vbdmat.boundaries import CapabilityStatus, derive_ior_interfaces
from vbdmat.core.volumes import OpticalPropertyVolume
from vbdmat.exporters.mitsuba import (
    MitsubaFieldConversion,
    convert_optical_fields,
    mitsuba_capability_report,
)
from vbdmat.exporters.openvdb import (
    OpenVDBFieldConversion,
    convert_openvdb_fields,
    openvdb_capability_report,
)


class ConformanceLayer(StrEnum):
    """Layer blamed when one shared contract check fails."""

    CANONICAL = "canonical"
    SERIALIZATION = "serialization"
    ADAPTER_CONVERSION = "adapter_conversion"
    IMAGE_SANITY = "image_sanity"


@dataclass(frozen=True, slots=True)
class ConformanceCheck:
    """One independently attributable conformance assertion."""

    name: str
    layer: ConformanceLayer
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "layer": self.layer.value,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class FixtureConformance:
    """Conformance result for one canonical fixture."""

    fixture: str
    checks: tuple[ConformanceCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "fixture": self.fixture,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True, slots=True)
class CrossConsumerConformanceReport:
    """Stable machine-readable report for all fixture checks."""

    fixtures: tuple[FixtureConformance, ...]
    expected_adapter_differences: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return bool(self.fixtures) and all(item.passed for item in self.fixtures)

    def to_dict(self) -> dict[str, object]:
        failures = [
            {
                "fixture": fixture.fixture,
                **check.to_dict(),
            }
            for fixture in self.fixtures
            for check in fixture.checks
            if not check.passed
        ]
        return {
            "passed": self.passed,
            "fixture_count": len(self.fixtures),
            "failure_count": len(failures),
            "failures": failures,
            "expected_adapter_differences": list(self.expected_adapter_differences),
            "fixtures": [item.to_dict() for item in self.fixtures],
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


EXPECTED_ADAPTER_DIFFERENCES = (
    "Mitsuba consumes RGB sigma_t/albedo; Cycles consumes equal-weight scalar "
    "absorption/scattering grids while OpenVDB retains the RGB components.",
    "Both consumers reduce spatial g to the same scattering-weighted scalar.",
    "Mitsuba emits derived dielectric IOR interface meshes; the Cycles proof "
    "retains the IOR grid but does not consume internal interfaces.",
    "Image pixels are not compared across renderers because their scene, camera, "
    "lighting, boundary, and transport parameterizations differ.",
)


def check_fixture_conformance(
    fixture: str,
    canonical: OpticalPropertyVolume,
    restored: OpticalPropertyVolume,
) -> FixtureConformance:
    """Check serialization and both pure adapter mappings for one fixture."""
    checks: list[ConformanceCheck] = []
    checks.append(
        _check(
            "canonical-metadata",
            ConformanceLayer.CANONICAL,
            canonical.coefficient_unit == "m^-1"
            and canonical.dimensionless_unit == "1"
            and canonical.scalar_dimensions == ("z", "y", "x"),
            "canonical coefficients are m^-1, scalar fields are dimensionless, "
            "and storage axes are ZYX",
        )
    )
    checks.append(
        _check(
            "zarr-round-trip",
            ConformanceLayer.SERIALIZATION,
            _volumes_equal(canonical, restored),
            "geometry, schema, basis, provenance, and all arrays round-trip exactly",
        )
    )

    mitsuba = convert_optical_fields(restored)
    openvdb = convert_openvdb_fields(restored)
    checks.extend(_adapter_checks(restored, mitsuba, openvdb))
    return FixtureConformance(fixture, tuple(checks))


def check_png_sanity(
    path: str | Path, *, expected_size: tuple[int, int]
) -> tuple[bool, str]:
    """Decode a basic non-interlaced PNG and reject empty or flat proof output."""
    try:
        width, height, channels, pixels = _read_png(Path(path))
    except (OSError, ValueError, zlib.error) as error:
        return False, f"PNG could not be inspected: {error}"
    if (width, height) != expected_size:
        return False, f"PNG dimensions {(width, height)} != expected {expected_size}"
    rgb = pixels[:, :, : min(channels, 3)]
    value_range = int(np.max(rgb)) - int(np.min(rgb))
    if value_range == 0:
        return False, "PNG is spatially flat"
    return True, f"{width}x{height}, channels={channels}, byte range={value_range}"


def png_pixel_sha256(path: str | Path) -> str:
    """Hash decoded PNG pixels, excluding encoder metadata and compression bytes."""
    width, height, channels, pixels = _read_png(Path(path))
    digest = hashlib.sha256()
    digest.update(struct.pack(">III", width, height, channels))
    digest.update(pixels.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def image_sanity_check(
    fixture: str,
    path: str | Path,
    *,
    expected_size: tuple[int, int],
    consumer: str,
) -> ConformanceCheck:
    passed, detail = check_png_sanity(path, expected_size=expected_size)
    return _check(
        f"{consumer}-image-{fixture}",
        ConformanceLayer.IMAGE_SANITY,
        passed,
        detail,
    )


def _adapter_checks(
    volume: OpticalPropertyVolume,
    mitsuba: MitsubaFieldConversion,
    openvdb: OpenVDBFieldConversion,
) -> list[ConformanceCheck]:
    checks: list[ConformanceCheck] = []
    vdb_a = _rgb_from_openvdb(openvdb, "sigma_a")
    vdb_s = _rgb_from_openvdb(openvdb, "sigma_s")
    vdb_g = _zyx(openvdb.fields_xyz["g"])
    vdb_ior = _zyx(openvdb.fields_xyz["ior"])
    fields_match = all(
        np.array_equal(left, right)
        for left, right in (
            (vdb_a, volume.sigma_a),
            (vdb_s, volume.sigma_s),
            (vdb_g, volume.g),
            (vdb_ior, volume.ior),
        )
    )
    checks.append(
        _check(
            "openvdb-pre-consumer-fields",
            ConformanceLayer.ADAPTER_CONVERSION,
            fields_match,
            "XYZ component grids transpose exactly back to canonical ZYX fields",
        )
    )

    expected_t = volume.sigma_a + volume.sigma_s
    expected_albedo = np.zeros_like(expected_t)
    np.divide(volume.sigma_s, expected_t, out=expected_albedo, where=expected_t > 0)
    checks.append(
        _check(
            "mitsuba-pre-consumer-fields",
            ConformanceLayer.ADAPTER_CONVERSION,
            np.array_equal(mitsuba.sigma_t, expected_t)
            and np.array_equal(mitsuba.albedo, expected_albedo),
            "sigma_t and albedo equal the documented componentwise conversion",
        )
    )

    expected_corners = _canonical_corners(volume)
    checks.append(
        _check(
            "volume-bounds-and-transform",
            ConformanceLayer.ADAPTER_CONVERSION,
            np.allclose(_mitsuba_corners(mitsuba), expected_corners, atol=1e-12)
            and np.allclose(
                _openvdb_corners(openvdb, volume.geometry.shape_xyz),
                expected_corners,
                atol=1e-12,
            ),
            "both adapter transforms map their complete domains to canonical "
            "world bounds",
        )
    )

    canonical_interfaces = derive_ior_interfaces(volume)
    boundary_match = (
        np.array_equal(vdb_ior, volume.ior)
        and mitsuba.interfaces.faces == canonical_interfaces.faces
        and all(
            np.allclose(
                mitsuba.interfaces.world_corners(face),
                canonical_interfaces.world_corners(expected),
            )
            for face, expected in zip(
                mitsuba.interfaces.faces,
                canonical_interfaces.faces,
                strict=True,
            )
        )
    )
    checks.append(
        _check(
            "material-regions-and-boundaries",
            ConformanceLayer.ADAPTER_CONVERSION,
            boundary_match,
            "OpenVDB preserves region IOR values and Mitsuba interface locations "
            "match canonical face corners",
        )
    )

    background = np.all(volume.sigma_a == 0, axis=-1) & np.all(
        volume.sigma_s == 0, axis=-1
    )
    vdb_background = np.all(vdb_a == 0, axis=-1) & np.all(vdb_s == 0, axis=-1)
    mitsuba_background = np.all(mitsuba.sigma_t == 0, axis=-1)
    checks.append(
        _check(
            "background-treatment",
            ConformanceLayer.ADAPTER_CONVERSION,
            np.array_equal(background, vdb_background)
            and np.array_equal(background, mitsuba_background),
            "zero-extinction cells remain zero in both consumer mappings",
        )
    )

    checks.append(
        _check(
            "coefficient-units-and-scale",
            ConformanceLayer.ADAPTER_CONVERSION,
            _unit_contracts_match(volume, mitsuba, openvdb),
            "both reports declare metre scenes and no coefficient numeric scaling",
        )
    )
    checks.append(
        _check(
            "global-phase-reduction",
            ConformanceLayer.ADAPTER_CONVERSION,
            np.isclose(mitsuba.phase_g, openvdb.phase_g, rtol=0.0, atol=1e-12),
            "both adapters use the same scattering-weighted global g",
        )
    )
    checks.append(
        _check(
            "capability-reports",
            ConformanceLayer.ADAPTER_CONVERSION,
            _capabilities_match_contract(volume, mitsuba, openvdb),
            "shared fields are complete and expected renderer differences are explicit",
        )
    )
    return checks


def _check(
    name: str, layer: ConformanceLayer, passed: bool | np.bool[Any], detail: str
) -> ConformanceCheck:
    return ConformanceCheck(name, layer, bool(passed), detail)


def _volumes_equal(left: OpticalPropertyVolume, right: OpticalPropertyVolume) -> bool:
    return (
        left.geometry == right.geometry
        and left.provenance == right.provenance
        and left.schema == right.schema
        and left.optical_basis == right.optical_basis
        and all(
            np.array_equal(getattr(left, name), getattr(right, name))
            for name in ("sigma_a", "sigma_s", "g", "ior")
        )
    )


def _rgb_from_openvdb(
    conversion: OpenVDBFieldConversion, prefix: str
) -> npt.NDArray[np.float32]:
    return np.stack(
        tuple(_zyx(conversion.fields_xyz[f"{prefix}_{channel}"]) for channel in "rgb"),
        axis=-1,
    )


def _zyx(array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return np.transpose(array, (2, 1, 0))


def _canonical_corners(volume: OpticalPropertyVolume) -> npt.NDArray[np.float64]:
    nx, ny, nz = volume.geometry.shape_xyz
    return np.asarray(
        [
            volume.geometry.continuous_index_to_world((x, y, z))
            for x in (0, nx)
            for y in (0, ny)
            for z in (0, nz)
        ],
        dtype=np.float64,
    )


def _transformed_corners(
    matrix: object,
    xs: tuple[float, float],
    ys: tuple[float, float],
    zs: tuple[float, float],
) -> npt.NDArray[np.float64]:
    transform = np.asarray(matrix, dtype=np.float64)
    return np.asarray(
        [
            (transform @ np.asarray((x, y, z, 1.0)))[:3]
            for x in xs
            for y in ys
            for z in zs
        ]
    )


def _mitsuba_corners(conversion: MitsubaFieldConversion) -> npt.NDArray[np.float64]:
    return _transformed_corners(
        conversion.volume_to_world, (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
    )


def _openvdb_corners(
    conversion: OpenVDBFieldConversion, shape_xyz: tuple[int, int, int]
) -> npt.NDArray[np.float64]:
    nx, ny, nz = shape_xyz
    return _transformed_corners(
        conversion.index_to_world,
        (-0.5, nx - 0.5),
        (-0.5, ny - 0.5),
        (-0.5, nz - 0.5),
    )


def _unit_contracts_match(
    volume: OpticalPropertyVolume,
    mitsuba: MitsubaFieldConversion,
    openvdb: OpenVDBFieldConversion,
) -> bool:
    reports = (
        mitsuba_capability_report(volume, mitsuba),
        openvdb_capability_report(volume, openvdb),
    )
    return all(
        report.by_field("coefficient_units").status is CapabilityStatus.REPRESENTED
        and "no numeric" in report.by_field("coefficient_units").detail
        for report in reports
    )


def _capabilities_match_contract(
    volume: OpticalPropertyVolume,
    mitsuba: MitsubaFieldConversion,
    openvdb: OpenVDBFieldConversion,
) -> bool:
    required = {
        "geometry",
        "coefficient_units",
        "optical_basis",
        "sigma_a",
        "sigma_s",
        "g",
        "ior",
        "derived_ior_interfaces",
        "provenance",
    }
    mi = mitsuba_capability_report(volume, mitsuba)
    ov = openvdb_capability_report(volume, openvdb)
    return (
        {entry.field for entry in mi.entries} == required
        and {entry.field for entry in ov.entries} == required
        and mi.by_field("ior").status is CapabilityStatus.UNSUPPORTED
        and ov.by_field("ior").status is CapabilityStatus.UNSUPPORTED
        and mi.by_field("derived_ior_interfaces").status is CapabilityStatus.TRANSFORMED
        and ov.by_field("derived_ior_interfaces").status is CapabilityStatus.UNSUPPORTED
    )


def _read_png(path: Path) -> tuple[int, int, int, npt.NDArray[np.uint8]]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("invalid PNG signature")
    position = 8
    compressed = bytearray()
    width = height = bit_depth = color_type = interlace = -1
    while position < len(data):
        length = struct.unpack(">I", data[position : position + 4])[0]
        kind = data[position + 4 : position + 8]
        payload = data[position + 8 : position + 8 + length]
        position += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if channels is None or bit_depth != 8 or interlace != 0:
        raise ValueError(
            "only non-interlaced 8-bit grayscale/RGB/RGBA PNG is supported"
        )
    stride = width * channels
    raw = zlib.decompress(bytes(compressed))
    if len(raw) != height * (stride + 1):
        raise ValueError("unexpected PNG payload size")
    rows = np.empty((height, stride), dtype=np.uint8)
    previous = np.zeros(stride, dtype=np.uint8)
    for row_index in range(height):
        offset = row_index * (stride + 1)
        filter_type = raw[offset]
        current = np.frombuffer(
            raw[offset + 1 : offset + 1 + stride], dtype=np.uint8
        ).copy()
        _unfilter(current, previous, filter_type, channels)
        rows[row_index] = current
        previous = current
    return width, height, channels, rows.reshape(height, width, channels)


def _unfilter(
    row: npt.NDArray[np.uint8],
    previous: npt.NDArray[np.uint8],
    filter_type: int,
    bytes_per_pixel: int,
) -> None:
    for index in range(len(row)):
        left = int(row[index - bytes_per_pixel]) if index >= bytes_per_pixel else 0
        above = int(previous[index])
        upper_left = (
            int(previous[index - bytes_per_pixel]) if index >= bytes_per_pixel else 0
        )
        value = int(row[index])
        if filter_type == 1:
            value += left
        elif filter_type == 2:
            value += above
        elif filter_type == 3:
            value += (left + above) // 2
        elif filter_type == 4:
            value += _paeth(left, above, upper_left)
        elif filter_type != 0:
            raise ValueError(f"unsupported PNG filter {filter_type}")
        row[index] = value & 0xFF


def _paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    distances = (
        abs(estimate - left),
        abs(estimate - above),
        abs(estimate - upper_left),
    )
    return (left, above, upper_left)[distances.index(min(distances))]
