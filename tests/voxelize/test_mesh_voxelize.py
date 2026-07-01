"""Tests for the STL reader and dense reference voxelizer (ADR-006, Step 3)."""

from __future__ import annotations

import struct
from collections.abc import Sequence

import numpy as np
import pytest

from vbdmat.core.transforms import IDENTITY_MATRIX_4
from vbdmat.io import read_stl_bytes
from vbdmat.io.errors import MeshReadError
from vbdmat.voxelize import (
    MeshTopologyError,
    VoxelizationError,
    voxelize_mesh,
)

Vertex = tuple[float, float, float]


def _cube_triangles(lo: Vertex, hi: Vertex) -> list[tuple[Vertex, Vertex, Vertex]]:
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    corners = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    # Quads wound counter-clockwise as seen from outside (outward normals).
    quads = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    triangles: list[tuple[Vertex, Vertex, Vertex]] = []
    for a, b, c, d in quads:
        triangles.append((corners[a], corners[b], corners[c]))
        triangles.append((corners[a], corners[c], corners[d]))
    return triangles


def _binary_stl(triangles: Sequence[tuple[Vertex, Vertex, Vertex]]) -> bytes:
    data = struct.pack("<80sI", b"", len(triangles))
    for tri in triangles:
        data += struct.pack(
            "<12fH", 0.0, 0.0, 0.0, *tri[0], *tri[1], *tri[2], 0
        )
    return data


def _ascii_stl(triangles: Sequence[tuple[Vertex, Vertex, Vertex]]) -> bytes:
    lines = ["solid test"]
    for tri in triangles:
        lines.append("facet normal 0 0 0")
        lines.append("outer loop")
        for vertex in tri:
            lines.append(f"vertex {vertex[0]} {vertex[1]} {vertex[2]}")
        lines.append("endloop")
        lines.append("endfacet")
    lines.append("endsolid test")
    return ("\n".join(lines) + "\n").encode("ascii")


def _cube_stl(lo: Vertex, hi: Vertex) -> bytes:
    return _binary_stl(_cube_triangles(lo, hi))


def _voxelize_cube(
    lo: Vertex,
    hi: Vertex,
    *,
    unit: str = "mm",
    voxel_size: Sequence[float] = (0.001, 0.001, 0.001),
    placement: object = IDENTITY_MATRIX_4,
):
    mesh = read_stl_bytes(_cube_stl(lo, hi))
    return voxelize_mesh(
        mesh,
        source_unit=unit,
        voxel_size_xyz_m=voxel_size,
        material_id=1,
        placement=placement,  # type: ignore[arg-type]
    )


def test_axis_aligned_cube_occupancy() -> None:
    result = _voxelize_cube((0, 0, 0), (3, 3, 3))
    assert result.diagnostics.shape_zyx == (5, 5, 5)
    assert result.diagnostics.occupied_cells == 27
    label = np.asarray(result.volume.material_id)
    # Interior block is exactly padded indices 1..3 on each axis.
    assert label[1:4, 1:4, 1:4].sum() == 27
    assert label.sum() == 27


def test_translated_cube_preserves_count() -> None:
    result = _voxelize_cube((1, 1, 1), (4, 4, 4))
    assert result.diagnostics.occupied_cells == 27


def test_cell_centres_on_mesh_boundary_use_closed_solid_convention() -> None:
    result = _voxelize_cube((0.5, 0.5, 0.5), (2.5, 2.5, 2.5))
    label = np.asarray(result.volume.material_id)

    assert result.diagnostics.occupied_cells == 27
    assert np.all(label[1:4, 1:4, 1:4] == 1)
    assert int(label.sum()) == 27


def test_non_cubic_box_occupancy() -> None:
    result = _voxelize_cube((0, 0, 0), (5, 3, 2))
    assert result.diagnostics.occupied_cells == 5 * 3 * 2


def test_millimetre_and_metre_agree() -> None:
    in_mm = _voxelize_cube((0, 0, 0), (3, 3, 3), unit="mm")
    in_m = _voxelize_cube(
        (0.0, 0.0, 0.0), (0.003, 0.003, 0.003), unit="m"
    )
    assert in_mm.diagnostics.shape_zyx == in_m.diagnostics.shape_zyx
    assert np.array_equal(
        np.asarray(in_mm.volume.material_id),
        np.asarray(in_m.volume.material_id),
    )


def test_anisotropic_voxel_size() -> None:
    result = _voxelize_cube(
        (0, 0, 0), (6, 6, 6), voxel_size=(0.002, 0.001, 0.001)
    )
    # 6 mm cube: 3 cells along X (2 mm), 6 along Y and Z.
    assert result.diagnostics.occupied_cells == 3 * 6 * 6


def test_rigid_rotation_preserves_occupancy_and_moves_world() -> None:
    # 90-degree rotation about Z: (x, y) -> (-y, x).
    rotation = (
        (0.0, -1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    plain = _voxelize_cube((0, 0, 0), (3, 3, 3))
    rotated = _voxelize_cube((0, 0, 0), (3, 3, 3), placement=rotation)
    assert rotated.diagnostics.occupied_cells == plain.diagnostics.occupied_cells
    assert np.array_equal(
        np.asarray(rotated.volume.material_id),
        np.asarray(plain.volume.material_id),
    )
    # The rotated world centre is the plain world centre turned 90 deg about Z.
    plain_world = plain.volume.geometry.cell_center_world((2, 2, 2))
    rotated_world = rotated.volume.geometry.cell_center_world((2, 2, 2))
    assert rotated_world == pytest.approx(
        (-plain_world[1], plain_world[0], plain_world[2])
    )


def test_triangle_reordering_is_invariant() -> None:
    triangles = _cube_triangles((0, 0, 0), (3, 3, 3))
    forward = read_stl_bytes(_binary_stl(triangles))
    reversed_mesh = read_stl_bytes(_binary_stl(list(reversed(triangles))))
    a = voxelize_mesh(
        forward, source_unit="mm", voxel_size_xyz_m=(0.001,) * 3, material_id=1
    )
    b = voxelize_mesh(
        reversed_mesh, source_unit="mm", voxel_size_xyz_m=(0.001,) * 3, material_id=1
    )
    assert np.array_equal(
        np.asarray(a.volume.material_id), np.asarray(b.volume.material_id)
    )


def test_ascii_and_binary_agree() -> None:
    triangles = _cube_triangles((0, 0, 0), (3, 3, 3))
    binary = voxelize_mesh(
        read_stl_bytes(_binary_stl(triangles)),
        source_unit="mm",
        voxel_size_xyz_m=(0.001,) * 3,
        material_id=1,
    )
    ascii_result = voxelize_mesh(
        read_stl_bytes(_ascii_stl(triangles)),
        source_unit="mm",
        voxel_size_xyz_m=(0.001,) * 3,
        material_id=1,
    )
    assert np.array_equal(
        np.asarray(binary.volume.material_id),
        np.asarray(ascii_result.volume.material_id),
    )


def test_diagnostics_bounds() -> None:
    result = _voxelize_cube((0, 0, 0), (3, 3, 3))
    assert result.diagnostics.triangle_count == 12
    assert result.diagnostics.bounds_min_xyz_m == pytest.approx((0.0, 0.0, 0.0))
    assert result.diagnostics.bounds_max_xyz_m == pytest.approx(
        (0.003, 0.003, 0.003)
    )


def test_open_mesh_is_rejected() -> None:
    triangles = _cube_triangles((0, 0, 0), (3, 3, 3))[:-1]
    with pytest.raises(MeshTopologyError) as info:
        voxelize_mesh(
            read_stl_bytes(_binary_stl(triangles)),
            source_unit="mm",
            voxel_size_xyz_m=(0.001,) * 3,
            material_id=1,
        )
    assert "open" in str(info.value).lower()


def test_non_manifold_mesh_is_rejected() -> None:
    # Add a duplicate face so one edge borders three triangles.
    triangles = _cube_triangles((0, 0, 0), (3, 3, 3))
    triangles.append(triangles[0])
    with pytest.raises(MeshTopologyError):
        voxelize_mesh(
            read_stl_bytes(_binary_stl(triangles)),
            source_unit="mm",
            voxel_size_xyz_m=(0.001,) * 3,
            material_id=1,
        )


def test_degenerate_triangle_is_rejected() -> None:
    triangles = _cube_triangles((0, 0, 0), (3, 3, 3))
    triangles[0] = ((0, 0, 0), (0, 0, 0), (1, 0, 0))
    with pytest.raises(MeshTopologyError):
        voxelize_mesh(
            read_stl_bytes(_binary_stl(triangles)),
            source_unit="mm",
            voxel_size_xyz_m=(0.001,) * 3,
            material_id=1,
        )


def test_empty_mesh_is_rejected() -> None:
    with pytest.raises(MeshTopologyError):
        voxelize_mesh(
            read_stl_bytes(_binary_stl([])),
            source_unit="mm",
            voxel_size_xyz_m=(0.001,) * 3,
            material_id=1,
        )


def test_multi_solid_is_rejected() -> None:
    triangles = _cube_triangles((0, 0, 0), (3, 3, 3))
    triangles += _cube_triangles((10, 10, 10), (13, 13, 13))
    with pytest.raises(MeshTopologyError) as info:
        voxelize_mesh(
            read_stl_bytes(_binary_stl(triangles)),
            source_unit="mm",
            voxel_size_xyz_m=(0.001,) * 3,
            material_id=1,
        )
    assert "disconnected" in str(info.value).lower()


def test_missing_unit_is_rejected() -> None:
    with pytest.raises(VoxelizationError) as info:
        voxelize_mesh(
            read_stl_bytes(_cube_stl((0, 0, 0), (3, 3, 3))),
            source_unit="inch",
            voxel_size_xyz_m=(0.001,) * 3,
            material_id=1,
        )
    assert info.value.field_path == "source_unit"


def test_background_material_id_is_rejected() -> None:
    with pytest.raises(VoxelizationError) as info:
        voxelize_mesh(
            read_stl_bytes(_cube_stl((0, 0, 0), (3, 3, 3))),
            source_unit="mm",
            voxel_size_xyz_m=(0.001,) * 3,
            material_id=0,
        )
    assert info.value.field_path == "material_id"


def test_cell_bound_is_enforced() -> None:
    with pytest.raises(VoxelizationError) as info:
        voxelize_mesh(
            read_stl_bytes(_cube_stl((0, 0, 0), (200, 1, 1))),
            source_unit="mm",
            voxel_size_xyz_m=(0.001, 0.001, 0.001),
            material_id=1,
        )
    assert info.value.field_path == "voxel_size"


def test_truncated_binary_stl_is_rejected() -> None:
    data = _cube_stl((0, 0, 0), (3, 3, 3))
    with pytest.raises(MeshReadError):
        read_stl_bytes(data[:-10])


def test_no_renderer_dependency_imported() -> None:
    from pathlib import Path

    import vbdmat.voxelize.mesh as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "mitsuba" not in source
    assert "openvdb" not in source
