"""Optional OpenVDB export and Cycles field conversion proof."""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import ModuleType
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

from vbdmat.boundaries import CapabilityStatus
from vbdmat.core.transforms import Matrix4
from vbdmat.core.volumes import OpticalPropertyVolume

from .diagnostics import CapabilityEntry, CapabilityReport

OPENVDB_ADAPTER = "vbdmat.exporters.openvdb"
OPENVDB_ADAPTER_VERSION = "1.0.0"

FloatArray: TypeAlias = npt.NDArray[np.float32]


class OpenVDBExportError(RuntimeError):
    """OpenVDB conversion or file export failed."""


class OpenVDBDependencyError(OpenVDBExportError):
    """Compatible OpenVDB Python bindings are unavailable."""


@dataclass(frozen=True, slots=True)
class OpenVDBExportConfig:
    """Explicit Cycles proof settings and scalar RGB reduction policy."""

    width: int = 64
    height: int = 64
    samples: int = 32
    seed: int = 20260629
    max_bounces: int = 8
    rgb_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)

    def __post_init__(self) -> None:
        for field in ("width", "height", "samples", "max_bounces"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{field} must be an integer")
            if int(value) <= 0:
                raise ValueError(f"{field} must be greater than zero")
            object.__setattr__(self, field, int(value))
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")
        if int(self.seed) < 0:
            raise ValueError("seed must not be negative")
        object.__setattr__(self, "seed", int(self.seed))
        if len(self.rgb_weights) != 3:
            raise ValueError("rgb_weights must contain three values")
        weights: list[float] = []
        for value in self.rgb_weights:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("rgb_weights values must be real numbers")
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError("rgb_weights values must be finite and non-negative")
            weights.append(number)
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("rgb_weights must sum to one")
        object.__setattr__(
            self, "rgb_weights", cast(tuple[float, float, float], tuple(weights))
        )


DEFAULT_OPENVDB_CONFIG = OpenVDBExportConfig()


@dataclass(frozen=True, slots=True, eq=False)
class OpenVDBFieldConversion:
    """Dense XYZ fields and the cell-centred index-to-world transform."""

    fields_xyz: dict[str, FloatArray]
    index_to_world: Matrix4
    phase_g: float


@dataclass(frozen=True, slots=True)
class OpenVDBExportResult:
    """OpenVDB and reproducibility artifacts produced by one export."""

    vdb_path: Path
    manifest_path: Path
    capability_path: Path
    capability_report: CapabilityReport
    grid_names: tuple[str, ...]


def convert_openvdb_fields(
    volume: OpticalPropertyVolume,
    config: OpenVDBExportConfig = DEFAULT_OPENVDB_CONFIG,
) -> OpenVDBFieldConversion:
    """Convert canonical ZYX fields to OpenVDB XYZ arrays without bindings."""
    if not isinstance(volume, OpticalPropertyVolume):
        raise TypeError("volume must be an OpticalPropertyVolume")
    if not isinstance(config, OpenVDBExportConfig):
        raise TypeError("config must be an OpenVDBExportConfig")

    fields: dict[str, FloatArray] = {}
    for prefix, source in (("sigma_a", volume.sigma_a), ("sigma_s", volume.sigma_s)):
        for channel, index in zip("rgb", range(3), strict=True):
            fields[f"{prefix}_{channel}"] = _xyz(source[..., index])
    fields["g"] = _xyz(volume.g)
    fields["ior"] = _xyz(volume.ior)

    weights = np.asarray(config.rgb_weights, dtype=np.float32)
    absorption = np.asarray(
        np.sum(volume.sigma_a * weights, axis=-1, dtype=np.float32),
        dtype=np.float32,
    )
    scattering = np.asarray(
        np.sum(volume.sigma_s * weights, axis=-1, dtype=np.float32),
        dtype=np.float32,
    )
    fields["cycles_absorption"] = _xyz(absorption)
    fields["cycles_scattering"] = _xyz(scattering)

    scattering_weights = np.mean(volume.sigma_s, axis=-1, dtype=np.float64)
    total = float(np.sum(scattering_weights, dtype=np.float64))
    phase_g = (
        float(np.sum(volume.g * scattering_weights, dtype=np.float64) / total)
        if total > 0.0
        else 0.0
    )
    for array in fields.values():
        array.setflags(write=False)
    return OpenVDBFieldConversion(
        fields_xyz=fields,
        index_to_world=_cell_center_index_to_world(volume),
        phase_g=phase_g,
    )


def openvdb_capability_report(
    volume: OpticalPropertyVolume, conversion: OpenVDBFieldConversion
) -> CapabilityReport:
    """Describe canonical-to-OpenVDB/Cycles field dispositions."""
    entries = (
        CapabilityEntry(
            "geometry",
            CapabilityStatus.TRANSFORMED,
            "XYZ OpenVDB indices with a cell-centred affine transform in metres",
            "canonical ZYX arrays are transposed; rotation, translation, and "
            "anisotropic spacing are retained",
        ),
        CapabilityEntry(
            "coefficient_units",
            CapabilityStatus.REPRESENTED,
            "Blender unit scale is 1 metre and Cycles densities consume m^-1 values",
            "no numeric coefficient scale is applied",
        ),
        CapabilityEntry(
            "optical_basis",
            CapabilityStatus.APPROXIMATED,
            "RGB component grids are preserved; Cycles proof uses an explicit "
            "weighted scalar reduction",
            "linear-srgb-effective-v1 is not a measured spectrum",
        ),
        CapabilityEntry(
            "sigma_a",
            CapabilityStatus.APPROXIMATED,
            "sigma_a_{r,g,b} grids plus cycles_absorption weighted scalar grid",
            "Cycles Volume Absorption density consumes the scalar reduction",
        ),
        CapabilityEntry(
            "sigma_s",
            CapabilityStatus.APPROXIMATED,
            "sigma_s_{r,g,b} grids plus cycles_scattering weighted scalar grid",
            "Cycles Volume Scatter density consumes the scalar reduction",
        ),
        CapabilityEntry(
            "g",
            CapabilityStatus.APPROXIMATED,
            "g grid is preserved; Cycles node uses one scattering-weighted scalar",
            f"spatial field reduced to g={conversion.phase_g:.9g}",
        ),
        CapabilityEntry(
            "ior",
            CapabilityStatus.UNSUPPORTED,
            "ior grid is preserved in OpenVDB but is not connected to Cycles "
            "volume shading",
            "Cycles does not expose heterogeneous volume IOR",
        ),
        CapabilityEntry(
            "derived_ior_interfaces",
            CapabilityStatus.UNSUPPORTED,
            "none",
            "the proof uses the volume domain boundary only; internal dielectric "
            "transitions are not represented",
        ),
        CapabilityEntry(
            "provenance",
            CapabilityStatus.REPRESENTED,
            "OpenVDB file metadata and capability JSON",
            f"source generator {volume.provenance.generator} "
            f"{volume.provenance.generator_version}",
        ),
    )
    return CapabilityReport(
        consumer="openvdb-blender-cycles",
        adapter=OPENVDB_ADAPTER,
        adapter_version=OPENVDB_ADAPTER_VERSION,
        schema_name=volume.schema.name,
        schema_version=str(volume.schema.version),
        entries=entries,
    )


def export_openvdb(
    volume: OpticalPropertyVolume,
    output_directory: str | Path,
    *,
    name: str = "volume",
    config: OpenVDBExportConfig = DEFAULT_OPENVDB_CONFIG,
) -> OpenVDBExportResult:
    """Write named FloatGrids and stable diagnostics using optional pyopenvdb."""
    if not name or Path(name).name != name:
        raise ValueError("name must be a non-empty filename stem")
    conversion = convert_openvdb_fields(volume, config)
    report = openvdb_capability_report(volume, conversion)
    vdb = _load_openvdb()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    transform = vdb.createLinearTransform(
        matrix=_openvdb_row_matrix(conversion.index_to_world)
    )
    grids: list[object] = []
    for grid_name, array in conversion.fields_xyz.items():
        grid = vdb.FloatGrid()
        grid.name = grid_name
        grid.creator = f"{OPENVDB_ADAPTER} {OPENVDB_ADAPTER_VERSION}"
        grid.gridClass = "fog volume"
        grid.transform = transform.copy() if hasattr(transform, "copy") else transform
        grid.copyFromArray(np.asarray(array), tolerance=0.0)
        grid["vbdmat:unit"] = (
            "m^-1" if "sigma" in grid_name or grid_name.startswith("cycles_") else "1"
        )
        grid["vbdmat:dimensions"] = "x,y,z"
        grids.append(grid)

    vdb_path = output / f"{name}.vdb"
    metadata = {
        "creator": f"{OPENVDB_ADAPTER} {OPENVDB_ADAPTER_VERSION}",
        "vbdmat:schema": f"{volume.schema.name} {volume.schema.version}",
        "vbdmat:world_unit": "m",
    }
    try:
        vdb.write(str(vdb_path), grids=grids, metadata=metadata)
    except Exception as error:
        raise OpenVDBExportError(f"failed to write {vdb_path}: {error}") from error

    capability_path = output / "capabilities.json"
    report.write_json(capability_path)
    manifest_path = output / "openvdb-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "vdb": vdb_path.name,
                "grid_names": list(conversion.fields_xyz),
                "grid_type": "FloatGrid",
                "dimensions_xyz": list(volume.geometry.shape_xyz),
                "index_to_world_column_matrix": [
                    list(row) for row in conversion.index_to_world
                ],
                "phase_g": conversion.phase_g,
                "cycles": {
                    "engine": "CYCLES",
                    "width": config.width,
                    "height": config.height,
                    "samples": config.samples,
                    "seed": config.seed,
                    "max_bounces": config.max_bounces,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return OpenVDBExportResult(
        vdb_path, manifest_path, capability_path, report, tuple(conversion.fields_xyz)
    )


def _xyz(array: npt.NDArray[np.float32]) -> FloatArray:
    return np.ascontiguousarray(np.transpose(array, (2, 1, 0)), dtype=np.float32)


def _cell_center_index_to_world(volume: OpticalPropertyVolume) -> Matrix4:
    geometry = volume.geometry
    rigid = np.asarray(geometry.local_to_world, dtype=np.float64)
    affine = rigid.copy()
    affine[:3, :3] = rigid[:3, :3] @ np.diag(geometry.voxel_size_xyz_m)
    half = np.asarray(geometry.voxel_size_xyz_m, dtype=np.float64) * 0.5
    affine[:3, 3] = rigid[:3, 3] + rigid[:3, :3] @ half
    return cast(Matrix4, tuple(tuple(float(value) for value in row) for row in affine))


def _openvdb_row_matrix(matrix: Matrix4) -> tuple[tuple[float, ...], ...]:
    """Convert our column-vector affine convention to OpenVDB's row-vector Mat4."""
    return tuple(tuple(float(value) for value in row) for row in np.asarray(matrix).T)


def _load_openvdb() -> ModuleType:
    for module_name in ("openvdb", "pyopenvdb"):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            pass
    raise OpenVDBDependencyError(
        "OpenVDB Python bindings are unavailable; run in Blender's Python or "
        "the pinned tools/phase0/Dockerfile.openvdb-cycles environment"
    )
