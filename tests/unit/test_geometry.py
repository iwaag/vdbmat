"""Tests for the coordinate contract in ADR-001."""

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from vdbmat.core import GridGeometry

TRANSLATED = (
    (1.0, 0.0, 0.0, 0.010),
    (0.0, 1.0, 0.0, 0.020),
    (0.0, 0.0, 1.0, 0.030),
    (0.0, 0.0, 0.0, 1.0),
)


@pytest.fixture
def anisotropic_geometry() -> GridGeometry:
    """Return the normative `4 x 3 x 2` example from ADR-001."""
    return GridGeometry(
        shape_zyx=(2, 3, 4),
        voxel_size_xyz_m=(0.00004, 0.00005, 0.00003),
        local_to_world=TRANSLATED,
    )


def test_anisotropic_worked_example(anisotropic_geometry: GridGeometry) -> None:
    assert anisotropic_geometry.shape_xyz == (4, 3, 2)
    assert anisotropic_geometry.local_extent_xyz_m == pytest.approx(
        (0.00016, 0.00015, 0.00006)
    )
    assert anisotropic_geometry.cell_center_local((1, 2, 3)) == pytest.approx(
        (0.00014, 0.000125, 0.000045)
    )
    assert anisotropic_geometry.cell_center_world((1, 2, 3)) == pytest.approx(
        (0.01014, 0.020125, 0.030045)
    )


def test_input_sequences_are_normalized_to_immutable_tuples() -> None:
    geometry = GridGeometry(  # type: ignore[arg-type]
        shape_zyx=[2, 3, 4],
        voxel_size_xyz_m=[0.1, 0.2, 0.3],
        local_to_world=[list(row) for row in TRANSLATED],
    )

    assert isinstance(geometry.shape_zyx, tuple)
    assert isinstance(geometry.voxel_size_xyz_m, tuple)
    assert all(isinstance(row, tuple) for row in geometry.local_to_world)
    with pytest.raises(FrozenInstanceError):
        geometry.shape_zyx = (1, 1, 1)  # type: ignore[misc]


def test_continuous_index_world_round_trip_with_rotation() -> None:
    rotation_and_translation = (
        (0.0, -1.0, 0.0, 10.0),
        (1.0, 0.0, 0.0, 20.0),
        (0.0, 0.0, 1.0, 30.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    geometry = GridGeometry(
        shape_zyx=(5, 6, 7),
        voxel_size_xyz_m=(1.0, 2.0, 3.0),
        local_to_world=rotation_and_translation,
    )

    assert geometry.cell_center_world((0, 0, 0)) == pytest.approx((9.0, 20.5, 31.5))
    index_xyz = (3.25, 2.5, 4.75)
    world = geometry.continuous_index_to_world(index_xyz)
    assert geometry.world_to_continuous_index(world) == pytest.approx(
        index_xyz, abs=1e-9
    )


def test_half_open_bounds_and_internal_boundary_ownership(
    anisotropic_geometry: GridGeometry,
) -> None:
    assert anisotropic_geometry.contains_continuous_index((0.0, 0.0, 0.0))
    assert anisotropic_geometry.contains_continuous_index((3.999, 2.999, 1.999))
    assert not anisotropic_geometry.contains_continuous_index((4.0, 3.0, 2.0))
    assert not anisotropic_geometry.contains_continuous_index((-1e-12, 0.0, 0.0))
    assert anisotropic_geometry.continuous_index_to_cell((3.0, 2.0, 1.0)) == (
        1,
        2,
        3,
    )


@pytest.mark.parametrize(
    ("shape", "error"),
    [
        ((0, 2, 3), "greater than zero"),
        ((1, -2, 3), "greater than zero"),
        ((1, 2), "exactly 3"),
        ((True, 2, 3), "must be an integer"),
    ],
)
def test_invalid_shapes_are_rejected(shape: object, error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        GridGeometry(
            shape_zyx=shape,  # type: ignore[arg-type]
            voxel_size_xyz_m=(1.0, 1.0, 1.0),
        )


@pytest.mark.parametrize(
    "voxel_size",
    [
        (0.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (math.nan, 1.0, 1.0),
        (math.inf, 1.0, 1.0),
    ],
)
def test_invalid_voxel_sizes_are_rejected(voxel_size: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        GridGeometry(
            shape_zyx=(1, 1, 1),
            voxel_size_xyz_m=voxel_size,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "transform",
    [
        ((1.0, 0.0), (0.0, 1.0)),
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0, 1.0),
        ),
        (
            (2.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        (
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        (
            (1.0, 0.1, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    ],
)
def test_non_rigid_or_left_handed_transforms_are_rejected(transform: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        GridGeometry(
            shape_zyx=(1, 1, 1),
            voxel_size_xyz_m=(1.0, 1.0, 1.0),
            local_to_world=transform,  # type: ignore[arg-type]
        )


def test_out_of_bounds_cells_are_rejected(
    anisotropic_geometry: GridGeometry,
) -> None:
    with pytest.raises(IndexError, match="outside"):
        anisotropic_geometry.cell_center_world((2, 0, 0))
    with pytest.raises(IndexError, match="outside"):
        anisotropic_geometry.continuous_index_to_cell((4.0, 0.0, 0.0))


def test_nan_continuous_index_is_rejected(anisotropic_geometry: GridGeometry) -> None:
    with pytest.raises(ValueError, match="finite"):
        anisotropic_geometry.continuous_index_to_world((np.nan, 0.0, 0.0))
