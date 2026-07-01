"""Dense, cell-centre reference voxelization for watertight single solids.

Implements the ADR-006 mesh path: topology inspection, deterministic domain
construction with one cell of padding, and a signed-winding +X ray classification of
cell centres. Correctness and inspectability are favoured over speed; the Phase 1 cell
bound keeps the dense method tractable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor
from typing import cast

import numpy as np
import numpy.typing as npt

from vbdmat.core import (
    GridGeometry,
    MaterialDefinition,
    MaterialLabelVolume,
    MaterialPalette,
    MaterialRole,
    Provenance,
)
from vbdmat.core.transforms import IDENTITY_MATRIX_4, Matrix4
from vbdmat.io.mesh import RawMesh

from .errors import MeshTopologyError, VoxelizationError

_UNIT_TO_METRES = {"m": 1.0, "mm": 1.0e-3}
_BARYCENTRIC_TOLERANCE = 1e-9
# Distinct sub-voxel Y/Z offsets (fractions of a voxel). They must differ so that
# cell centres with equal Y and Z do not stay on a 45-degree triangulation diagonal.
_SAMPLE_JITTER_Y = 7.3e-5
_SAMPLE_JITTER_Z = 3.1e-5
_MAX_AXIS_CELLS = 128
_MAX_TOTAL_CELLS = 2_000_000
# A span landing within this fraction of a cell of an integer is treated as exactly
# that many cells, so float32 STL round-off does not add a spurious padded cell and
# equivalent unit expressions of one solid yield the same grid.
_DOMAIN_SNAP_EPS = 1e-6

BoundsXYZ = tuple[tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True, slots=True)
class VoxelizationDiagnostics:
    """Structured, machine-readable voxelization findings (ADR-006 D-diagnostics)."""

    triangle_count: int
    bounds_min_xyz_m: tuple[float, float, float]
    bounds_max_xyz_m: tuple[float, float, float]
    shape_zyx: tuple[int, int, int]
    occupied_cells: int
    material_id: int


@dataclass(frozen=True, slots=True)
class VoxelizationResult:
    """A canonical label volume plus its voxelization diagnostics."""

    volume: MaterialLabelVolume
    diagnostics: VoxelizationDiagnostics


def voxelize_mesh(
    mesh: RawMesh,
    *,
    source_unit: str,
    voxel_size_xyz_m: Sequence[float],
    material_id: int,
    material_name: str = "material",
    placement: Matrix4 = IDENTITY_MATRIX_4,
    padding_cells: int = 1,
    generator: str = "vbdmat.voxelize",
    generator_version: str = "1.0.0",
    identity: str | None = None,
) -> VoxelizationResult:
    """Voxelize a watertight single-solid mesh into a ``MaterialLabelVolume``."""
    factor = _unit_factor(source_unit)
    voxel_size = _voxel_size(voxel_size_xyz_m)
    _require_interior_material(material_id)
    if not isinstance(padding_cells, int) or padding_cells < 0:
        raise VoxelizationError("padding_cells", "must be a non-negative integer")

    inspect_topology(mesh)
    triangles_m = mesh.triangles * factor

    origin, shape_zyx = _domain(triangles_m, voxel_size, padding_cells)
    local_to_world = _compose_placement(placement, origin)

    label = _classify(triangles_m, origin, voxel_size, shape_zyx, material_id)
    occupied = int(np.count_nonzero(label))

    palette = _palette(material_id, material_name)
    geometry = GridGeometry(
        shape_zyx=shape_zyx,
        voxel_size_xyz_m=voxel_size,
        local_to_world=local_to_world,
    )
    sources: tuple[str, ...] = (f"mesh:stl:{mesh.triangle_count}-triangles",)
    if identity is not None:
        sources = (*sources, f"identity:{identity}")
    provenance = Provenance(
        generator=generator,
        generator_version=generator_version,
        sources=sources,
        notes=(
            f"dense cell-centre voxelization; source_unit={source_unit}; "
            f"padding_cells={padding_cells}"
        ),
    )
    volume = MaterialLabelVolume(
        geometry=geometry,
        palette=palette,
        provenance=provenance,
        material_id=label,
    )

    vertices = triangles_m.reshape(-1, 3)
    bounds_min = cast(
        "tuple[float, float, float]", tuple(vertices.min(axis=0).tolist())
    )
    bounds_max = cast(
        "tuple[float, float, float]", tuple(vertices.max(axis=0).tolist())
    )
    diagnostics = VoxelizationDiagnostics(
        triangle_count=mesh.triangle_count,
        bounds_min_xyz_m=bounds_min,
        bounds_max_xyz_m=bounds_max,
        shape_zyx=shape_zyx,
        occupied_cells=occupied,
        material_id=material_id,
    )
    return VoxelizationResult(volume=volume, diagnostics=diagnostics)


def inspect_topology(mesh: RawMesh) -> None:
    """Reject any mesh that is not a watertight, consistently oriented single solid."""
    triangles = mesh.triangles
    count = int(triangles.shape[0])
    if count == 0:
        raise MeshTopologyError("mesh", "mesh has no triangles")

    scale = float(np.max(np.abs(triangles))) if triangles.size else 1.0
    tol = max(scale, 1.0) * 1e-9
    vertices = triangles.reshape(-1, 3)
    keys = np.round(vertices / tol).astype(np.int64)
    _unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    welded = np.asarray(inverse).reshape(-1)[: count * 3].reshape(count, 3)

    # Degenerate faces: repeated welded vertex or near-zero area.
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    area_eps = (max(scale, 1.0) ** 2) * 1e-18
    for tri_index in range(count):
        a, b, c = (int(value) for value in welded[tri_index])
        if len({a, b, c}) != 3:
            raise MeshTopologyError(
                f"triangle[{tri_index}]", "has a repeated (degenerate) vertex"
            )
        if float(areas[tri_index]) <= area_eps:
            raise MeshTopologyError(
                f"triangle[{tri_index}]", "has near-zero area (degenerate)"
            )

    directed: dict[tuple[int, int], int] = {}
    undirected: dict[tuple[int, int], int] = {}
    parent: list[int] = list(range(count))
    edge_owner: dict[tuple[int, int], int] = {}

    for tri_index in range(count):
        a, b, c = (int(value) for value in welded[tri_index])
        for u, v in ((a, b), (b, c), (c, a)):
            directed[(u, v)] = directed.get((u, v), 0) + 1
            key = (u, v) if u < v else (v, u)
            undirected[key] = undirected.get(key, 0) + 1
            if key in edge_owner:
                _union(parent, edge_owner[key], tri_index)
            else:
                edge_owner[key] = tri_index

    for (u, v), total in undirected.items():
        if total == 1:
            raise MeshTopologyError(
                "mesh", "surface is open / not watertight (an edge borders one face)"
            )
        if total > 2:
            raise MeshTopologyError(
                "mesh",
                "surface is non-manifold (an edge borders more than two faces)",
                count=total,
            )
        forward = directed.get((u, v), 0)
        backward = directed.get((v, u), 0)
        if forward != 1 or backward != 1:
            raise MeshTopologyError(
                "mesh",
                "inconsistent triangle orientation across a shared edge",
            )

    roots = {_find(parent, index) for index in range(count)}
    if len(roots) != 1:
        raise MeshTopologyError(
            "mesh",
            "mesh contains multiple disconnected solids",
            count=len(roots),
        )


def _classify(
    triangles_m: npt.NDArray[np.float64],
    origin: tuple[float, float, float],
    voxel_size: tuple[float, float, float],
    shape_zyx: tuple[int, int, int],
    material_id: int,
) -> npt.NDArray[np.uint16]:
    nz, ny, nx = shape_zyx
    sx, sy, sz = voxel_size
    ox, oy, oz = origin

    x0 = triangles_m[:, 0, 0]
    y0 = triangles_m[:, 0, 1]
    z0 = triangles_m[:, 0, 2]
    x1 = triangles_m[:, 1, 0]
    y1 = triangles_m[:, 1, 1]
    z1 = triangles_m[:, 1, 2]
    x2 = triangles_m[:, 2, 0]
    y2 = triangles_m[:, 2, 1]
    z2 = triangles_m[:, 2, 2]

    denom = (y1 - y0) * (z2 - z0) - (y2 - y0) * (z1 - z0)
    denom_scale = float(np.max(np.abs(denom))) if denom.size else 0.0
    area_eps = denom_scale * 1e-12
    facing = np.abs(denom) > area_eps
    sign = np.sign(denom)
    safe_denom = np.where(facing, denom, 1.0)

    # Deterministic sub-voxel offset of the YZ sample point (ADR-006 D8). It only
    # perturbs a centre off an interior triangulation diagonal (a non-physical shared
    # edge); real surfaces are at least half a voxel from any cell centre for
    # well-posed Phase 1 inputs, so classification is unchanged.
    jitter_y = sy * _SAMPLE_JITTER_Y
    jitter_z = sz * _SAMPLE_JITTER_Z

    xc = ox + (np.arange(nx) + 0.5) * sx
    label = np.zeros((nz, ny, nx), dtype=np.uint16)

    for k in range(nz):
        zc = oz + (k + 0.5) * sz + jitter_z
        for j in range(ny):
            yc = oy + (j + 0.5) * sy + jitter_y
            w1 = ((yc - y0) * (z2 - z0) - (y2 - y0) * (zc - z0)) / safe_denom
            w2 = ((y1 - y0) * (zc - z0) - (yc - y0) * (z1 - z0)) / safe_denom
            w0 = 1.0 - w1 - w2
            inside = (
                facing
                & (w0 >= -_BARYCENTRIC_TOLERANCE)
                & (w1 >= -_BARYCENTRIC_TOLERANCE)
                & (w2 >= -_BARYCENTRIC_TOLERANCE)
            )
            if not np.any(inside):
                continue
            x_int = w0 * x0 + w1 * x1 + w2 * x2
            xi = x_int[inside]
            si = sign[inside]
            ahead = xi[None, :] > xc[:, None]
            winding = (ahead * si[None, :]).sum(axis=1)
            occupied = np.abs(winding) >= 0.5
            if np.any(occupied):
                label[k, j, occupied] = material_id
    return label


def _domain(
    triangles_m: npt.NDArray[np.float64],
    voxel_size: tuple[float, float, float],
    padding_cells: int,
) -> tuple[tuple[float, float, float], tuple[int, int, int]]:
    vertices = triangles_m.reshape(-1, 3)
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    origin: list[float] = []
    extents_xyz: list[int] = []
    for axis in range(3):
        size = voxel_size[axis]
        base = floor(float(minimum[axis]) / size + _DOMAIN_SNAP_EPS) * size
        span = float(maximum[axis]) - base
        cells = max(1, ceil(span / size - _DOMAIN_SNAP_EPS))
        base -= padding_cells * size
        cells += 2 * padding_cells
        origin.append(base)
        extents_xyz.append(cells)

    total = extents_xyz[0] * extents_xyz[1] * extents_xyz[2]
    if any(cells > _MAX_AXIS_CELLS for cells in extents_xyz):
        raise VoxelizationError(
            "voxel_size",
            f"grid extent {tuple(extents_xyz)} exceeds the Phase 1 axis bound "
            f"of {_MAX_AXIS_CELLS} cells; use a coarser voxel size",
        )
    if total > _MAX_TOTAL_CELLS:
        raise VoxelizationError(
            "voxel_size",
            f"grid has {total} cells, exceeding the Phase 1 bound of "
            f"{_MAX_TOTAL_CELLS}; use a coarser voxel size",
        )

    shape_zyx = (extents_xyz[2], extents_xyz[1], extents_xyz[0])
    origin_xyz = (origin[0], origin[1], origin[2])
    return origin_xyz, shape_zyx


def _compose_placement(
    placement: Matrix4, origin: tuple[float, float, float]
) -> Matrix4:
    placement_matrix = np.asarray(placement, dtype=np.float64)
    translation = np.eye(4, dtype=np.float64)
    translation[0, 3] = origin[0]
    translation[1, 3] = origin[1]
    translation[2, 3] = origin[2]
    composed = placement_matrix @ translation
    return cast(
        Matrix4, tuple(tuple(float(value) for value in row) for row in composed)
    )


def _palette(material_id: int, material_name: str) -> MaterialPalette:
    return MaterialPalette.from_sequence(
        (
            MaterialDefinition(0, "background", MaterialRole.BACKGROUND),
            MaterialDefinition(material_id, material_name, MaterialRole.MATERIAL),
        )
    )


def _unit_factor(source_unit: str) -> float:
    if source_unit not in _UNIT_TO_METRES:
        raise VoxelizationError(
            "source_unit",
            f"must be one of {sorted(_UNIT_TO_METRES)}, got {source_unit!r}",
        )
    return _UNIT_TO_METRES[source_unit]


def _voxel_size(voxel_size_xyz_m: Sequence[float]) -> tuple[float, float, float]:
    values = tuple(voxel_size_xyz_m)
    if len(values) != 3:
        raise VoxelizationError(
            "voxel_size_xyz_m", "must contain exactly 3 numbers"
        )
    result: list[float] = []
    for axis, item in zip(("x", "y", "z"), values, strict=True):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise VoxelizationError(f"voxel_size_xyz_m.{axis}", "must be a number")
        value = float(item)
        if not np.isfinite(value) or value <= 0.0:
            raise VoxelizationError(
                f"voxel_size_xyz_m.{axis}", "must be finite and greater than zero"
            )
        result.append(value)
    return (result[0], result[1], result[2])


def _require_interior_material(material_id: int) -> None:
    if isinstance(material_id, bool) or not isinstance(material_id, int):
        raise VoxelizationError("material_id", "must be an integer")
    if not 1 <= material_id <= 65535:
        raise VoxelizationError(
            "material_id",
            "must be a non-background material ID in [1, 65535]",
        )


def _union(parent: list[int], left: int, right: int) -> None:
    root_left = _find(parent, left)
    root_right = _find(parent, right)
    if root_left != root_right:
        parent[root_right] = root_left


def _find(parent: list[int], node: int) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node
