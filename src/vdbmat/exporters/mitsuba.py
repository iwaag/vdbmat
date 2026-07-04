"""Optional Mitsuba 3 proof adapter for canonical optical volumes."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import ModuleType
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from vbdmat.boundaries import (
    BoundaryAxis,
    BoundaryDerivationConfig,
    CapabilityStatus,
    DerivedInterfaceSet,
    derive_ior_interfaces,
)
from vbdmat.core.axes import PointXYZ
from vbdmat.core.geometry import GridGeometry
from vbdmat.core.transforms import Matrix4
from vbdmat.core.volumes import OpticalPropertyVolume

from .diagnostics import CapabilityEntry, CapabilityReport

MITSUBA_ADAPTER = "vbdmat.exporters.mitsuba"
MITSUBA_ADAPTER_VERSION = "1.0.0"

FloatArray: TypeAlias = npt.NDArray[np.float32]
MeshQuad: TypeAlias = tuple[PointXYZ, PointXYZ, PointXYZ, PointXYZ]


class MitsubaExportError(RuntimeError):
    """Mitsuba conversion, scene construction, or rendering failed."""


class MitsubaDependencyError(MitsubaExportError):
    """The requested Mitsuba runtime or variant is unavailable."""


@dataclass(frozen=True, slots=True)
class MitsubaExportConfig:
    """Fixed and explicit scene settings for the Phase 0 proof."""

    width: int = 64
    height: int = 64
    spp: int = 32
    seed: int = 20260628
    max_depth: int = 8
    fov_degrees: float = 35.0
    ambient_ior: float = 1.0
    ior_absolute_tolerance: float = 1e-6
    attenuation_diagnostic_gain: float = 128.0
    variant: str = "llvm_ad_rgb"

    def __post_init__(self) -> None:
        for field in ("width", "height", "spp", "max_depth"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{field} must be an integer")
            number = int(value)
            if number <= 0:
                raise ValueError(f"{field} must be greater than zero")
            object.__setattr__(self, field, number)
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must not be negative")
        object.__setattr__(self, "seed", int(self.seed))
        for field in (
            "fov_degrees",
            "ambient_ior",
            "ior_absolute_tolerance",
            "attenuation_diagnostic_gain",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field} must be a real number")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{field} must be finite")
            object.__setattr__(self, field, number)
        if not 1.0 <= self.fov_degrees < 179.0:
            raise ValueError("fov_degrees must lie in [1, 179)")
        if self.ambient_ior <= 0.0:
            raise ValueError("ambient_ior must be greater than zero")
        if self.ior_absolute_tolerance < 0.0:
            raise ValueError("ior_absolute_tolerance must not be negative")
        if self.attenuation_diagnostic_gain <= 0.0:
            raise ValueError("attenuation_diagnostic_gain must be greater than zero")
        if not isinstance(self.variant, str) or not self.variant.strip():
            raise ValueError("variant must be a non-empty string")


DEFAULT_MITSUBA_CONFIG = MitsubaExportConfig()


@dataclass(frozen=True, slots=True, eq=False)
class MitsubaFieldConversion:
    """Renderer fields derived deterministically before Mitsuba object creation."""

    sigma_t: FloatArray
    albedo: FloatArray
    phase_g: float
    interfaces: DerivedInterfaceSet
    volume_to_world: Matrix4


@dataclass(frozen=True, slots=True, eq=False)
class PreparedMitsubaScene:
    """Loadable Mitsuba scene dictionary plus reproducibility artifacts."""

    scene_dict: dict[str, Any]
    conversion: MitsubaFieldConversion
    capability_report: CapabilityReport
    artifact_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class MitsubaRenderResult:
    """Paths and deterministic summary of one completed proof render."""

    exr_path: Path
    png_path: Path
    attenuation_png_path: Path
    png_sha256: str
    attenuation_png_sha256: str
    mean_linear_rgb: tuple[float, float, float]
    minimum: float
    maximum: float
    capability_report: CapabilityReport


def convert_optical_fields(
    volume: OpticalPropertyVolume,
    config: MitsubaExportConfig = DEFAULT_MITSUBA_CONFIG,
) -> MitsubaFieldConversion:
    """Convert canonical coefficients and interfaces without importing Mitsuba."""
    if not isinstance(volume, OpticalPropertyVolume):
        raise TypeError("volume must be an OpticalPropertyVolume")
    if not isinstance(config, MitsubaExportConfig):
        raise TypeError("config must be a MitsubaExportConfig")

    sigma_t64 = volume.sigma_a.astype(np.float64) + volume.sigma_s.astype(np.float64)
    float32_max = np.finfo(np.float32).max
    if not np.all(np.isfinite(sigma_t64)) or np.any(sigma_t64 > float32_max):
        raise MitsubaExportError("sigma_t conversion exceeds finite float32 range")
    sigma_t = np.asarray(sigma_t64, dtype=np.float32)
    albedo = np.zeros_like(sigma_t)
    np.divide(volume.sigma_s, sigma_t, out=albedo, where=sigma_t > 0.0)

    scattering_weights = np.mean(volume.sigma_s, axis=-1, dtype=np.float64)
    total_weight = float(np.sum(scattering_weights, dtype=np.float64))
    phase_g = (
        float(
            np.sum(volume.g.astype(np.float64) * scattering_weights, dtype=np.float64)
            / total_weight
        )
        if total_weight > 0.0
        else 0.0
    )
    if not -1.0 < phase_g < 1.0:
        raise MitsubaExportError(
            "scattering-weighted phase g must lie strictly inside (-1, 1) for Mitsuba"
        )

    for array in (sigma_t, albedo):
        array.setflags(write=False)
    interfaces = derive_ior_interfaces(
        volume,
        BoundaryDerivationConfig(
            ambient_ior=config.ambient_ior,
            ior_absolute_tolerance=config.ior_absolute_tolerance,
        ),
    )
    return MitsubaFieldConversion(
        sigma_t=sigma_t,
        albedo=albedo,
        phase_g=phase_g,
        interfaces=interfaces,
        volume_to_world=_volume_to_world_matrix(volume.geometry),
    )


def mitsuba_capability_report(
    volume: OpticalPropertyVolume, conversion: MitsubaFieldConversion
) -> CapabilityReport:
    """Describe every canonical field and the derived interface disposition."""
    entries = (
        CapabilityEntry(
            "geometry",
            CapabilityStatus.TRANSFORMED,
            "canonical grid transform to Mitsuba volume and world-space PLY meshes",
            "ZYX tensor order and anisotropic metric transform are retained",
        ),
        CapabilityEntry(
            "coefficient_units",
            CapabilityStatus.REPRESENTED,
            "scene units are metres and heterogeneous medium scale is 1",
            "canonical m^-1 coefficients need no numeric unit scaling",
        ),
        CapabilityEntry(
            "optical_basis",
            CapabilityStatus.APPROXIMATED,
            "raw three-channel tensor in Mitsuba RGB transport mode",
            "linear-srgb-effective-v1 is effective RGB, not a measured spectrum",
        ),
        CapabilityEntry(
            "sigma_a",
            CapabilityStatus.TRANSFORMED,
            "sigma_t=sigma_a+sigma_s and albedo=sigma_s/sigma_t",
            "lossless algebra except canonical float32 output rounding",
        ),
        CapabilityEntry(
            "sigma_s",
            CapabilityStatus.TRANSFORMED,
            "sigma_t=sigma_a+sigma_s and albedo=sigma_s/sigma_t",
            "zero extinction maps to exactly zero albedo",
        ),
        CapabilityEntry(
            "g",
            CapabilityStatus.APPROXIMATED,
            "one scattering-weighted global Henyey-Greenstein g",
            f"spatial field reduced to g={conversion.phase_g:.9g}",
        ),
        CapabilityEntry(
            "ior",
            CapabilityStatus.UNSUPPORTED,
            "not passed as a heterogeneous medium volume",
            "Mitsuba heterogeneous media do not accept a spatial IOR field",
        ),
        CapabilityEntry(
            "derived_ior_interfaces",
            CapabilityStatus.TRANSFORMED,
            "oriented PLY patches with null or dielectric BSDFs",
            f"{len(conversion.interfaces.faces)} non-index-matched faces; complete "
            "domain containment is emitted separately",
        ),
        CapabilityEntry(
            "provenance",
            CapabilityStatus.REPRESENTED,
            "capability JSON source metadata",
            f"source generator {volume.provenance.generator} "
            f"{volume.provenance.generator_version}",
        ),
    )
    return CapabilityReport(
        consumer="mitsuba-3",
        adapter=MITSUBA_ADAPTER,
        adapter_version=MITSUBA_ADAPTER_VERSION,
        schema_name=volume.schema.name,
        schema_version=str(volume.schema.version),
        entries=entries,
    )


def prepare_mitsuba_scene(
    volume: OpticalPropertyVolume,
    output_directory: str | Path,
    config: MitsubaExportConfig = DEFAULT_MITSUBA_CONFIG,
) -> PreparedMitsubaScene:
    """Create a loadable fixed Mitsuba scene and write mesh/report artifacts."""
    conversion = convert_optical_fields(volume, config)
    report = mitsuba_capability_report(volume, conversion)
    output = Path(output_directory)
    mi = _load_mitsuba(config.variant)
    output.mkdir(parents=True, exist_ok=True)
    transform = mi.ScalarTransform4f(
        np.asarray(conversion.volume_to_world, dtype=np.float32)
    )

    scene: dict[str, Any] = {
        "type": "scene",
        "integrator": {"type": "volpath", "max_depth": config.max_depth},
        "sensor": _sensor_dict(mi, volume.geometry, config),
        "vbdmat_medium": {
            "type": "heterogeneous",
            "sigma_t": {
                "type": "gridvolume",
                "data": mi.TensorXf(np.asarray(conversion.sigma_t)),
                "raw": True,
                "filter_type": "nearest",
                "to_world": transform,
            },
            "albedo": {
                "type": "gridvolume",
                "data": mi.TensorXf(np.asarray(conversion.albedo)),
                "raw": True,
                "filter_type": "nearest",
                "to_world": transform,
            },
            "phase": {"type": "hg", "g": conversion.phase_g},
            "scale": 1.0,
        },
        "backlight": _backlight_dict(mi, volume.geometry),
    }

    artifacts: list[Path] = []
    shape_index = 0
    for descriptor, quads in _exterior_mesh_groups(volume, config).items():
        interior_ior, index_matched = descriptor
        path = output / f"exterior-{shape_index:03d}.ply"
        _write_ply(path, quads)
        artifacts.append(path)
        shape = _mesh_shape(path)
        shape["bsdf"] = (
            {"type": "null"}
            if index_matched
            else {
                "type": "dielectric",
                "int_ior": interior_ior,
                "ext_ior": config.ambient_ior,
            }
        )
        shape["interior"] = {"type": "ref", "id": "vbdmat_medium"}
        scene[f"exterior_{shape_index:03d}"] = shape
        shape_index += 1

    internal_groups = _internal_mesh_groups(conversion.interfaces)
    for (negative_ior, positive_ior), quads in internal_groups.items():
        path = output / f"interior-{shape_index:03d}.ply"
        _write_ply(path, quads)
        artifacts.append(path)
        shape = _mesh_shape(path)
        shape["bsdf"] = {
            "type": "dielectric",
            "int_ior": negative_ior,
            "ext_ior": positive_ior,
        }
        shape["interior"] = {"type": "ref", "id": "vbdmat_medium"}
        shape["exterior"] = {"type": "ref", "id": "vbdmat_medium"}
        scene[f"interior_{shape_index:03d}"] = shape
        shape_index += 1

    report_path = output / "capabilities.json"
    report.write_json(report_path)
    artifacts.append(report_path)
    summary_path = output / "scene-summary.json"
    summary_path.write_text(
        json.dumps(
            _scene_summary(volume, conversion, config, artifacts),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts.append(summary_path)
    return PreparedMitsubaScene(
        scene_dict=scene,
        conversion=conversion,
        capability_report=report,
        artifact_paths=tuple(artifacts),
    )


def render_mitsuba(
    volume: OpticalPropertyVolume,
    output_directory: str | Path,
    *,
    name: str = "render",
    config: MitsubaExportConfig = DEFAULT_MITSUBA_CONFIG,
) -> MitsubaRenderResult:
    """Prepare, load, render, and write deterministic EXR and PNG proof outputs."""
    prepared = prepare_mitsuba_scene(volume, output_directory, config)
    mi = _load_mitsuba(config.variant)
    try:
        scene = mi.load_dict(prepared.scene_dict)
        image = mi.render(scene, seed=config.seed, spp=config.spp)
    except Exception as error:
        raise MitsubaExportError(
            f"Mitsuba scene load/render failed: {error}"
        ) from error
    output = Path(output_directory)
    exr_path = output / f"{name}.exr"
    png_path = output / f"{name}.png"
    attenuation_path = output / f"{name}-attenuation.png"
    mi.util.write_bitmap(str(exr_path), image, write_async=False)
    mi.util.write_bitmap(str(png_path), image, write_async=False)
    pixels = np.asarray(image, dtype=np.float32)
    attenuation = np.asarray(
        np.clip(
            (1.0 - pixels) * config.attenuation_diagnostic_gain,
            0.0,
            1.0,
        ),
        dtype=np.float32,
    )
    mi.util.write_bitmap(str(attenuation_path), attenuation, write_async=False)
    mean = np.asarray(np.mean(pixels, axis=(0, 1), dtype=np.float64))
    return MitsubaRenderResult(
        exr_path=exr_path,
        png_path=png_path,
        attenuation_png_path=attenuation_path,
        png_sha256="sha256:" + hashlib.sha256(png_path.read_bytes()).hexdigest(),
        attenuation_png_sha256=(
            "sha256:" + hashlib.sha256(attenuation_path.read_bytes()).hexdigest()
        ),
        mean_linear_rgb=(float(mean[0]), float(mean[1]), float(mean[2])),
        minimum=float(np.min(pixels)),
        maximum=float(np.max(pixels)),
        capability_report=prepared.capability_report,
    )


def _load_mitsuba(variant: str) -> ModuleType:
    try:
        mi = importlib.import_module("mitsuba")
    except ImportError as error:
        raise MitsubaDependencyError(
            "Mitsuba bindings are unavailable; run with `uv run --group mitsuba`"
        ) from error
    if variant not in mi.variants():
        raise MitsubaDependencyError(f"Mitsuba variant is unavailable: {variant}")
    if mi.variant() != variant:
        mi.set_variant(variant)
    return mi


def _volume_to_world_matrix(geometry: GridGeometry) -> Matrix4:
    transform = np.asarray(geometry.local_to_world, dtype=np.float64)
    scale = np.diag([*geometry.local_extent_xyz_m, 1.0])
    result = transform @ scale
    return cast(Matrix4, tuple(tuple(float(value) for value in row) for row in result))


def _sensor_dict(
    mi: ModuleType, geometry: GridGeometry, config: MitsubaExportConfig
) -> dict[str, Any]:
    center, radius, camera_direction = _scene_frame(geometry)
    distance = radius * 1.6 / math.tan(math.radians(config.fov_degrees) * 0.5)
    origin = center + camera_direction * distance
    return {
        "type": "perspective",
        "fov": config.fov_degrees,
        "near_clip": max(radius * 0.001, 1e-9),
        "far_clip": max(radius * 20.0, 1e-6),
        "to_world": mi.ScalarTransform4f.look_at(
            origin=origin.tolist(), target=center.tolist(), up=[0.0, 0.0, 1.0]
        ),
        "sampler": {"type": "independent", "sample_count": config.spp},
        "film": {
            "type": "hdrfilm",
            "width": config.width,
            "height": config.height,
            "pixel_format": "rgb",
            "component_format": "float32",
            "rfilter": {"type": "box"},
        },
    }


def _scene_frame(
    geometry: GridGeometry,
) -> tuple[npt.NDArray[np.float64], float, npt.NDArray[np.float64]]:
    corners = np.asarray(
        [
            geometry.continuous_index_to_world((x, y, z))
            for x in (0, geometry.shape_xyz[0])
            for y in (0, geometry.shape_xyz[1])
            for z in (0, geometry.shape_xyz[2])
        ],
        dtype=np.float64,
    )
    minimum = np.min(corners, axis=0)
    maximum = np.max(corners, axis=0)
    center = (minimum + maximum) * 0.5
    radius = float(np.linalg.norm(maximum - minimum) * 0.5)
    camera_direction = np.asarray((1.6, -2.2, 1.4), dtype=np.float64)
    camera_direction /= np.linalg.norm(camera_direction)
    return center, radius, camera_direction


def _backlight_dict(mi: ModuleType, geometry: GridGeometry) -> dict[str, Any]:
    center, radius, camera_direction = _scene_frame(geometry)
    position = center - camera_direction * radius * 4.0
    return {
        "type": "rectangle",
        "to_world": mi.ScalarTransform4f.look_at(
            origin=position.tolist(), target=center.tolist(), up=[0.0, 0.0, 1.0]
        )
        @ mi.ScalarTransform4f.scale([radius * 3.5, radius * 3.5, 1.0]),
        "emitter": {
            "type": "area",
            "radiance": {"type": "rgb", "value": [1.0, 1.0, 1.0]},
        },
    }


def _exterior_mesh_groups(
    volume: OpticalPropertyVolume, config: MitsubaExportConfig
) -> dict[tuple[float, bool], tuple[MeshQuad, ...]]:
    groups: defaultdict[tuple[float, bool], list[MeshQuad]] = defaultdict(list)
    geometry = volume.geometry
    nz, ny, nx = geometry.shape_zyx
    for axis, extent in (
        (BoundaryAxis.X, nx),
        (BoundaryAxis.Y, ny),
        (BoundaryAxis.Z, nz),
    ):
        for upper in (False, True):
            coordinate = extent - 1 if upper else 0
            for cell in _boundary_cells(axis, coordinate, geometry.shape_zyx):
                ior = float(volume.ior[cell])
                matched = abs(ior - config.ambient_ior) <= config.ior_absolute_tolerance
                corners = _boundary_world_corners(geometry, axis, cell, upper=upper)
                groups[(ior, matched)].append(corners)
    return {key: tuple(value) for key, value in sorted(groups.items())}


def _internal_mesh_groups(
    interfaces: DerivedInterfaceSet,
) -> dict[tuple[float, float], tuple[MeshQuad, ...]]:
    groups: defaultdict[tuple[float, float], list[MeshQuad]] = defaultdict(list)
    for face in interfaces.interior_faces:
        groups[(face.ior_negative, face.ior_positive)].append(
            interfaces.world_corners(face)
        )
    return {key: tuple(value) for key, value in sorted(groups.items())}


def _boundary_cells(
    axis: BoundaryAxis, coordinate: int, shape_zyx: tuple[int, int, int]
) -> tuple[tuple[int, int, int], ...]:
    nz, ny, nx = shape_zyx
    if axis is BoundaryAxis.X:
        return tuple((z, y, coordinate) for z in range(nz) for y in range(ny))
    if axis is BoundaryAxis.Y:
        return tuple((z, coordinate, x) for z in range(nz) for x in range(nx))
    return tuple((coordinate, y, x) for y in range(ny) for x in range(nx))


def _boundary_world_corners(
    geometry: GridGeometry,
    axis: BoundaryAxis,
    cell: tuple[int, int, int],
    *,
    upper: bool,
) -> MeshQuad:
    z, y, x = cell
    indices: tuple[tuple[int, int, int], ...]
    if axis is BoundaryAxis.X:
        plane = x + int(upper)
        indices = (
            (plane, y, z),
            (plane, y + 1, z),
            (plane, y + 1, z + 1),
            (plane, y, z + 1),
        )
    elif axis is BoundaryAxis.Y:
        plane = y + int(upper)
        indices = (
            (x, plane, z),
            (x, plane, z + 1),
            (x + 1, plane, z + 1),
            (x + 1, plane, z),
        )
    else:
        plane = z + int(upper)
        indices = (
            (x, y, plane),
            (x + 1, y, plane),
            (x + 1, y + 1, plane),
            (x, y + 1, plane),
        )
    if not upper:
        indices = tuple(reversed(indices))
    return cast(
        MeshQuad,
        tuple(geometry.continuous_index_to_world(index) for index in indices),
    )


def _write_ply(path: Path, quads: Sequence[MeshQuad]) -> None:
    vertices = [vertex for quad in quads for vertex in quad]
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(quads) * 2}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines.extend(
        " ".join(f"{coordinate:.17g}" for coordinate in vertex) for vertex in vertices
    )
    for index in range(len(quads)):
        base = index * 4
        lines.append(f"3 {base} {base + 1} {base + 2}")
        lines.append(f"3 {base} {base + 2} {base + 3}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _mesh_shape(path: Path) -> dict[str, Any]:
    return {"type": "ply", "filename": str(path.resolve()), "face_normals": True}


def _scene_summary(
    volume: OpticalPropertyVolume,
    conversion: MitsubaFieldConversion,
    config: MitsubaExportConfig,
    artifacts: Sequence[Path],
) -> dict[str, object]:
    return {
        "adapter": MITSUBA_ADAPTER,
        "adapter_version": MITSUBA_ADAPTER_VERSION,
        "schema": {"name": volume.schema.name, "version": str(volume.schema.version)},
        "geometry": {
            "shape_zyx": list(volume.geometry.shape_zyx),
            "voxel_size_xyz_m": list(volume.geometry.voxel_size_xyz_m),
            "local_to_world": [list(row) for row in volume.geometry.local_to_world],
        },
        "render": {
            "variant": config.variant,
            "width": config.width,
            "height": config.height,
            "spp": config.spp,
            "seed": config.seed,
            "max_depth": config.max_depth,
            "fov_degrees": config.fov_degrees,
            "attenuation_diagnostic_gain": config.attenuation_diagnostic_gain,
        },
        "conversion": {
            "sigma_t_min": float(np.min(conversion.sigma_t)),
            "sigma_t_max": float(np.max(conversion.sigma_t)),
            "albedo_min": float(np.min(conversion.albedo)),
            "albedo_max": float(np.max(conversion.albedo)),
            "phase_g": conversion.phase_g,
            "derived_interface_faces": len(conversion.interfaces.faces),
            "volume_to_world": [list(row) for row in conversion.volume_to_world],
        },
        "artifacts": [path.name for path in artifacts],
    }
