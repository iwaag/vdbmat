"""Canonical axis-order types and numeric normalization helpers."""

import math
from collections.abc import Sequence
from numbers import Integral, Real
from typing import TypeAlias, cast

ShapeZYX: TypeAlias = tuple[int, int, int]
IndexZYX: TypeAlias = tuple[int, int, int]
PointXYZ: TypeAlias = tuple[float, float, float]
VoxelSizeXYZ: TypeAlias = tuple[float, float, float]


def normalize_shape_zyx(value: Sequence[int]) -> ShapeZYX:
    """Return a validated positive ``(nz, ny, nx)`` shape."""
    if len(value) != 3:
        raise ValueError("shape_zyx must contain exactly 3 values")

    normalized: list[int] = []
    for axis, item in zip(("z", "y", "x"), value, strict=True):
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise TypeError(f"shape_zyx.{axis} must be an integer")
        integer = int(item)
        if integer <= 0:
            raise ValueError(f"shape_zyx.{axis} must be greater than zero")
        normalized.append(integer)
    return cast(ShapeZYX, tuple(normalized))


def normalize_index_zyx(value: Sequence[int]) -> IndexZYX:
    """Return a validated integer ``(z, y, x)`` cell index."""
    if len(value) != 3:
        raise ValueError("index_zyx must contain exactly 3 values")

    normalized: list[int] = []
    for axis, item in zip(("z", "y", "x"), value, strict=True):
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise TypeError(f"index_zyx.{axis} must be an integer")
        normalized.append(int(item))
    return cast(IndexZYX, tuple(normalized))


def normalize_point_xyz(value: Sequence[float], *, field: str) -> PointXYZ:
    """Return three finite real values in semantic ``(x, y, z)`` order."""
    if len(value) != 3:
        raise ValueError(f"{field} must contain exactly 3 values")

    normalized: list[float] = []
    for axis, item in zip(("x", "y", "z"), value, strict=True):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(f"{field}.{axis} must be a real number")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{field}.{axis} must be finite")
        normalized.append(number)
    return cast(PointXYZ, tuple(normalized))


def normalize_voxel_size_xyz_m(value: Sequence[float]) -> VoxelSizeXYZ:
    """Return three finite, positive voxel dimensions in metres."""
    normalized = normalize_point_xyz(value, field="voxel_size_xyz_m")
    for axis, item in zip(("x", "y", "z"), normalized, strict=True):
        if item <= 0.0:
            raise ValueError(f"voxel_size_xyz_m.{axis} must be greater than zero")
    return normalized
