"""Canonical voxel-grid geometry and coordinate conversion."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import floor
from typing import cast

from .axes import (
    IndexZYX,
    PointXYZ,
    ShapeZYX,
    VoxelSizeXYZ,
    normalize_index_zyx,
    normalize_point_xyz,
    normalize_shape_zyx,
    normalize_voxel_size_xyz_m,
)
from .transforms import (
    IDENTITY_MATRIX_4,
    Matrix4,
    apply_inverse_rigid_transform,
    apply_rigid_transform,
    normalize_rigid_transform,
)


@dataclass(frozen=True, slots=True)
class GridGeometry:
    """Immutable geometry for a dense, cell-centred voxel grid."""

    shape_zyx: ShapeZYX
    voxel_size_xyz_m: VoxelSizeXYZ
    local_to_world: Matrix4 = IDENTITY_MATRIX_4

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape_zyx", normalize_shape_zyx(self.shape_zyx))
        object.__setattr__(
            self,
            "voxel_size_xyz_m",
            normalize_voxel_size_xyz_m(self.voxel_size_xyz_m),
        )
        object.__setattr__(
            self,
            "local_to_world",
            normalize_rigid_transform(self.local_to_world),
        )

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        """Return grid extents in semantic ``(nx, ny, nz)`` order."""
        nz, ny, nx = self.shape_zyx
        return (nx, ny, nz)

    @property
    def local_extent_xyz_m(self) -> PointXYZ:
        """Return the grid's local physical extent in metres."""
        return cast(
            PointXYZ,
            tuple(
                float(extent * size)
                for extent, size in zip(
                    self.shape_xyz, self.voxel_size_xyz_m, strict=True
                )
            ),
        )

    def cell_center_local(self, index_zyx: Sequence[int]) -> PointXYZ:
        """Return the local-metric centre of a contained cell."""
        z, y, x = self._contained_cell_index(index_zyx)
        sx, sy, sz = self.voxel_size_xyz_m
        return (sx * (x + 0.5), sy * (y + 0.5), sz * (z + 0.5))

    def cell_center_world(self, index_zyx: Sequence[int]) -> PointXYZ:
        """Return the world-space centre of a contained cell."""
        return apply_rigid_transform(
            self.local_to_world, self.cell_center_local(index_zyx)
        )

    def continuous_index_to_local(self, index_xyz: Sequence[float]) -> PointXYZ:
        """Convert a continuous corner-relative XYZ index to local metres."""
        index = normalize_point_xyz(index_xyz, field="index_xyz")
        return cast(
            PointXYZ,
            tuple(
                coordinate * size
                for coordinate, size in zip(index, self.voxel_size_xyz_m, strict=True)
            ),
        )

    def continuous_index_to_world(self, index_xyz: Sequence[float]) -> PointXYZ:
        """Convert a continuous corner-relative XYZ index to world metres."""
        return apply_rigid_transform(
            self.local_to_world, self.continuous_index_to_local(index_xyz)
        )

    def world_to_continuous_index(self, world_xyz_m: Sequence[float]) -> PointXYZ:
        """Convert a world-space point to continuous corner-relative XYZ index."""
        local = apply_inverse_rigid_transform(self.local_to_world, world_xyz_m)
        return cast(
            PointXYZ,
            tuple(
                coordinate / size
                for coordinate, size in zip(local, self.voxel_size_xyz_m, strict=True)
            ),
        )

    def contains_continuous_index(self, index_xyz: Sequence[float]) -> bool:
        """Return whether a continuous XYZ index lies in half-open grid bounds."""
        index = normalize_point_xyz(index_xyz, field="index_xyz")
        return all(
            0.0 <= coordinate < extent
            for coordinate, extent in zip(index, self.shape_xyz, strict=True)
        )

    def continuous_index_to_cell(self, index_xyz: Sequence[float]) -> IndexZYX:
        """Return the containing cell in array order without clamping."""
        index = normalize_point_xyz(index_xyz, field="index_xyz")
        if not self.contains_continuous_index(index):
            raise IndexError("index_xyz lies outside the half-open grid bounds")
        x, y, z = (floor(coordinate) for coordinate in index)
        return (z, y, x)

    def world_to_cell(self, world_xyz_m: Sequence[float]) -> IndexZYX:
        """Return the cell containing a world-space point without clamping."""
        return self.continuous_index_to_cell(
            self.world_to_continuous_index(world_xyz_m)
        )

    def _contained_cell_index(self, index_zyx: Sequence[int]) -> IndexZYX:
        index = normalize_index_zyx(index_zyx)
        for axis, item, extent in zip(
            ("z", "y", "x"), index, self.shape_zyx, strict=True
        ):
            if not 0 <= item < extent:
                raise IndexError(f"index_zyx.{axis}={item} is outside [0, {extent})")
        return index
