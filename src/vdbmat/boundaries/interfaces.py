"""Derived sharp interfaces for canonical cell-centred refractive index fields."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from typing import TypeAlias, cast

import numpy as np

from vbdmat.core.axes import IndexZYX, PointXYZ
from vbdmat.core.geometry import GridGeometry
from vbdmat.core.metadata import Provenance, SchemaIdentity
from vbdmat.core.transforms import apply_rigid_transform
from vbdmat.core.volumes import OpticalPropertyVolume


class BoundaryAxis(StrEnum):
    """Semantic axis normal to an interface face."""

    X = "x"
    Y = "y"
    Z = "z"


CellOrExterior: TypeAlias = IndexZYX | None
CornerIndexXYZ: TypeAlias = tuple[int, int, int]
WorldCorners: TypeAlias = tuple[PointXYZ, PointXYZ, PointXYZ, PointXYZ]


@dataclass(frozen=True, slots=True)
class BoundaryDerivationConfig:
    """Explicit threshold and exterior medium for sharp-interface derivation."""

    ambient_ior: float = 1.0
    ior_absolute_tolerance: float = 1e-6
    include_exterior: bool = True

    def __post_init__(self) -> None:
        for field in ("ambient_ior", "ior_absolute_tolerance"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field} must be a real number")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{field} must be finite")
            object.__setattr__(self, field, number)
        if self.ambient_ior <= 0.0:
            raise ValueError("ambient_ior must be greater than zero")
        if self.ior_absolute_tolerance < 0.0:
            raise ValueError("ior_absolute_tolerance must not be negative")
        if not isinstance(self.include_exterior, bool):
            raise TypeError("include_exterior must be a bool")


DEFAULT_BOUNDARY_CONFIG = BoundaryDerivationConfig()


@dataclass(frozen=True, slots=True)
class InterfaceFace:
    """One oriented unit-cell face separating two piecewise-constant IOR regions.

    The face normal points from the negative semantic-axis side to the positive side.
    A ``None`` cell denotes the exterior medium. ``corner_index_xyz`` is the minimum
    continuous grid corner of the face; the coordinate on ``axis`` is its plane index.
    """

    axis: BoundaryAxis
    corner_index_xyz: CornerIndexXYZ
    negative_cell_zyx: CellOrExterior
    positive_cell_zyx: CellOrExterior
    ior_negative: float
    ior_positive: float

    @property
    def is_exterior(self) -> bool:
        """Return whether either side is outside the canonical grid."""
        return self.negative_cell_zyx is None or self.positive_cell_zyx is None


@dataclass(frozen=True, slots=True)
class DerivedInterfaceSet:
    """Renderer-neutral sharp interfaces derived from one optical volume."""

    geometry: GridGeometry
    source_schema: SchemaIdentity
    source_provenance: Provenance
    config: BoundaryDerivationConfig
    faces: tuple[InterfaceFace, ...]

    @property
    def interior_faces(self) -> tuple[InterfaceFace, ...]:
        """Return faces whose two sides are canonical cells."""
        return tuple(face for face in self.faces if not face.is_exterior)

    @property
    def exterior_faces(self) -> tuple[InterfaceFace, ...]:
        """Return faces between a canonical cell and the configured ambient medium."""
        return tuple(face for face in self.faces if face.is_exterior)

    def world_corners(self, face: InterfaceFace) -> WorldCorners:
        """Return four world-space corners with winding toward the positive axis."""
        x, y, z = face.corner_index_xyz
        if face.axis is BoundaryAxis.X:
            indices = ((x, y, z), (x, y + 1, z), (x, y + 1, z + 1), (x, y, z + 1))
        elif face.axis is BoundaryAxis.Y:
            indices = ((x, y, z), (x, y, z + 1), (x + 1, y, z + 1), (x + 1, y, z))
        else:
            indices = ((x, y, z), (x + 1, y, z), (x + 1, y + 1, z), (x, y + 1, z))
        return cast(
            WorldCorners,
            tuple(self.geometry.continuous_index_to_world(index) for index in indices),
        )

    def world_normal(self, face: InterfaceFace) -> PointXYZ:
        """Return the unit world normal from negative to positive side."""
        local_normal = {
            BoundaryAxis.X: (1.0, 0.0, 0.0),
            BoundaryAxis.Y: (0.0, 1.0, 0.0),
            BoundaryAxis.Z: (0.0, 0.0, 1.0),
        }[face.axis]
        origin = apply_rigid_transform(self.geometry.local_to_world, (0.0, 0.0, 0.0))
        endpoint = apply_rigid_transform(self.geometry.local_to_world, local_normal)
        return cast(
            PointXYZ,
            tuple(end - start for start, end in zip(origin, endpoint, strict=True)),
        )


def derive_ior_interfaces(
    volume: OpticalPropertyVolume,
    config: BoundaryDerivationConfig = DEFAULT_BOUNDARY_CONFIG,
) -> DerivedInterfaceSet:
    """Derive oriented sharp faces wherever adjacent piecewise-constant IOR differs."""
    if not isinstance(volume, OpticalPropertyVolume):
        raise TypeError("volume must be an OpticalPropertyVolume")
    if not isinstance(config, BoundaryDerivationConfig):
        raise TypeError("config must be a BoundaryDerivationConfig")

    faces: list[InterfaceFace] = []
    ior = volume.ior
    _append_interior_faces(faces, ior, BoundaryAxis.X, config)
    _append_interior_faces(faces, ior, BoundaryAxis.Y, config)
    _append_interior_faces(faces, ior, BoundaryAxis.Z, config)
    if config.include_exterior:
        _append_exterior_faces(faces, ior, config)
    return DerivedInterfaceSet(
        geometry=volume.geometry,
        source_schema=volume.schema,
        source_provenance=volume.provenance,
        config=config,
        faces=tuple(faces),
    )


def _append_interior_faces(
    faces: list[InterfaceFace],
    ior: np.ndarray[tuple[int, ...], np.dtype[np.float32]],
    axis: BoundaryAxis,
    config: BoundaryDerivationConfig,
) -> None:
    array_axis = {BoundaryAxis.X: 2, BoundaryAxis.Y: 1, BoundaryAxis.Z: 0}[axis]
    negative_slices = [slice(None), slice(None), slice(None)]
    positive_slices = [slice(None), slice(None), slice(None)]
    negative_slices[array_axis] = slice(None, -1)
    positive_slices[array_axis] = slice(1, None)
    negative = ior[tuple(negative_slices)]
    positive = ior[tuple(positive_slices)]
    mask = np.abs(positive - negative) > config.ior_absolute_tolerance
    for z, y, x in np.argwhere(mask):
        negative_cell = (int(z), int(y), int(x))
        positive_cell_list = [int(z), int(y), int(x)]
        positive_cell_list[array_axis] += 1
        positive_cell = cast(IndexZYX, tuple(positive_cell_list))
        corner = _face_corner(axis, negative_cell, upper=True)
        faces.append(
            InterfaceFace(
                axis=axis,
                corner_index_xyz=corner,
                negative_cell_zyx=negative_cell,
                positive_cell_zyx=positive_cell,
                ior_negative=float(ior[negative_cell]),
                ior_positive=float(ior[positive_cell]),
            )
        )


def _append_exterior_faces(
    faces: list[InterfaceFace],
    ior: np.ndarray[tuple[int, ...], np.dtype[np.float32]],
    config: BoundaryDerivationConfig,
) -> None:
    nz, ny, nx = ior.shape
    for axis, array_axis, extent in (
        (BoundaryAxis.X, 2, nx),
        (BoundaryAxis.Y, 1, ny),
        (BoundaryAxis.Z, 0, nz),
    ):
        for upper in (False, True):
            cell_coordinate = extent - 1 if upper else 0
            slices: list[int | slice] = [slice(None), slice(None), slice(None)]
            slices[array_axis] = cell_coordinate
            boundary_values = ior[tuple(slices)]
            mask = (
                np.abs(boundary_values - config.ambient_ior)
                > config.ior_absolute_tolerance
            )
            for reduced_index in np.argwhere(mask):
                cell = _expand_boundary_index(
                    axis, cell_coordinate, tuple(int(item) for item in reduced_index)
                )
                faces.append(
                    InterfaceFace(
                        axis=axis,
                        corner_index_xyz=_face_corner(axis, cell, upper=upper),
                        negative_cell_zyx=cell if upper else None,
                        positive_cell_zyx=None if upper else cell,
                        ior_negative=float(ior[cell]) if upper else config.ambient_ior,
                        ior_positive=config.ambient_ior if upper else float(ior[cell]),
                    )
                )


def _face_corner(
    axis: BoundaryAxis, cell_zyx: IndexZYX, *, upper: bool
) -> CornerIndexXYZ:
    z, y, x = cell_zyx
    coordinate = [x, y, z]
    semantic_axis = {BoundaryAxis.X: 0, BoundaryAxis.Y: 1, BoundaryAxis.Z: 2}[axis]
    if upper:
        coordinate[semantic_axis] += 1
    return cast(CornerIndexXYZ, tuple(coordinate))


def _expand_boundary_index(
    axis: BoundaryAxis, cell_coordinate: int, reduced_index: tuple[int, ...]
) -> IndexZYX:
    if axis is BoundaryAxis.X:
        z, y = reduced_index
        return (z, y, cell_coordinate)
    if axis is BoundaryAxis.Y:
        z, x = reduced_index
        return (z, cell_coordinate, x)
    y, x = reduced_index
    return (cell_coordinate, y, x)
