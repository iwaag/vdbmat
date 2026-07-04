"""Validation and application of canonical rigid transforms."""

from collections.abc import Sequence
from numbers import Real
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

from .axes import PointXYZ, normalize_point_xyz

Matrix4: TypeAlias = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]

IDENTITY_MATRIX_4: Matrix4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

TRANSFORM_ABSOLUTE_TOLERANCE = 1e-9


def normalize_rigid_transform(value: Sequence[Sequence[float]]) -> Matrix4:
    """Return an immutable, finite, right-handed rigid ``4 x 4`` transform."""
    if len(value) != 4 or any(len(row) != 4 for row in value):
        raise ValueError("local_to_world must have shape (4, 4)")

    rows: list[tuple[float, float, float, float]] = []
    for row_index, row in enumerate(value):
        converted: list[float] = []
        for column_index, item in enumerate(row):
            if isinstance(item, bool) or not isinstance(item, Real):
                raise TypeError(
                    f"local_to_world[{row_index}][{column_index}] must be a real number"
                )
            number = float(item)
            if not np.isfinite(number):
                raise ValueError(
                    f"local_to_world[{row_index}][{column_index}] must be finite"
                )
            converted.append(number)
        rows.append(cast(tuple[float, float, float, float], tuple(converted)))

    matrix = np.asarray(rows, dtype=np.float64)
    if not np.allclose(
        matrix[3],
        np.array([0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=TRANSFORM_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError("local_to_world must have last row [0, 0, 0, 1]")

    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=0.0,
        atol=TRANSFORM_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError("local_to_world rotation must be orthonormal")

    determinant = float(np.linalg.det(rotation))
    if not np.isclose(
        determinant,
        1.0,
        rtol=0.0,
        atol=TRANSFORM_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError("local_to_world rotation must have determinant +1")

    return cast(Matrix4, tuple(rows))


def apply_rigid_transform(matrix: Matrix4, point_xyz: Sequence[float]) -> PointXYZ:
    """Apply a canonical local-to-world transform to a metric XYZ point."""
    point = normalize_point_xyz(point_xyz, field="point_xyz")
    transform = _as_array(matrix)
    homogeneous = np.array([*point, 1.0], dtype=np.float64)
    result = transform @ homogeneous
    return cast(PointXYZ, tuple(float(item) for item in result[:3]))


def apply_inverse_rigid_transform(
    matrix: Matrix4, point_xyz: Sequence[float]
) -> PointXYZ:
    """Apply the analytic inverse of a canonical rigid transform."""
    point = np.asarray(
        normalize_point_xyz(point_xyz, field="point_xyz"), dtype=np.float64
    )
    transform = _as_array(matrix)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    result = rotation.T @ (point - translation)
    return cast(PointXYZ, tuple(float(item) for item in result))


def _as_array(matrix: Matrix4) -> npt.NDArray[np.float64]:
    return np.asarray(matrix, dtype=np.float64)
