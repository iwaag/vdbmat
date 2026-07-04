"""Small deterministic canonical volumes for regression and adapter proofs."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import product
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

from vbdmat.core import (
    GridGeometry,
    MaterialDefinition,
    MaterialLabelVolume,
    MaterialMixtureVolume,
    MaterialPalette,
    MaterialRole,
    Provenance,
)
from vbdmat.core.axes import IndexZYX, PointXYZ, ShapeZYX, VoxelSizeXYZ

CanonicalMaterialVolume: TypeAlias = MaterialLabelVolume | MaterialMixtureVolume
BoundsXYZ: TypeAlias = tuple[PointXYZ, PointXYZ]

FIXTURE_GENERATOR = "vbdmat.synthetic-fixtures"
FIXTURE_GENERATOR_VERSION = "1.0.0"

_TRANSLATED_LOCAL_TO_WORLD = (
    (1.0, 0.0, 0.0, 0.010),
    (0.0, 1.0, 0.0, 0.020),
    (0.0, 0.0, 1.0, 0.030),
    (0.0, 0.0, 0.0, 1.0),
)
_DEFAULT_VOXEL_SIZE_XYZ_M: VoxelSizeXYZ = (0.00004, 0.00005, 0.00003)


@dataclass(frozen=True, slots=True)
class SelectedCellExpectation:
    """Expected canonical value and position for one fixture cell."""

    index_zyx: IndexZYX
    world_center_xyz_m: PointXYZ
    material_id: int | None = None
    fractions: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if (self.material_id is None) == (self.fractions is None):
            raise ValueError(
                "selected cell must define exactly one of material_id or fractions"
            )


@dataclass(frozen=True, slots=True)
class FixtureManifest:
    """Machine-readable expected summaries for a synthetic fixture."""

    name: str
    description: str
    shape_zyx: ShapeZYX
    local_bounds_xyz_m: BoundsXYZ
    world_bounds_xyz_m: BoundsXYZ
    material_voxel_counts: tuple[tuple[int, int], ...]
    material_fraction_totals: tuple[tuple[int, float], ...]
    selected_cells: tuple[SelectedCellExpectation, ...]


@dataclass(frozen=True, slots=True, eq=False)
class SyntheticFixture:
    """A canonical material volume paired with deterministic expectations."""

    volume: CanonicalMaterialVolume
    manifest: FixtureManifest


FixtureFactory: TypeAlias = Callable[[], SyntheticFixture]


def homogeneous_transparent() -> SyntheticFixture:
    """Return 24 cells of the transparent-resin material command."""
    name = "homogeneous-transparent"
    geometry = _geometry((2, 3, 4))
    labels = np.full(geometry.shape_zyx, 1, dtype=np.uint16)
    return _label_fixture(
        name=name,
        description="Homogeneous transparent-resin material volume.",
        geometry=geometry,
        labels=labels,
        selected_indices=((0, 0, 0), (1, 2, 3)),
    )


def homogeneous_scattering_white() -> SyntheticFixture:
    """Return 24 cells of the scattering-white material command."""
    name = "homogeneous-scattering-white"
    geometry = _geometry((2, 3, 4))
    labels = np.full(geometry.shape_zyx, 2, dtype=np.uint16)
    return _label_fixture(
        name=name,
        description="Homogeneous white-resin material volume for later mapping.",
        geometry=geometry,
        labels=labels,
        selected_indices=((0, 0, 0), (1, 2, 3)),
    )


def transparent_opaque_interface() -> SyntheticFixture:
    """Return equal transparent and opaque regions separated on an X plane."""
    name = "transparent-opaque-interface"
    geometry = _geometry((2, 4, 6))
    labels = np.empty(geometry.shape_zyx, dtype=np.uint16)
    labels[:, :, :3] = 1
    labels[:, :, 3:] = 3
    return _label_fixture(
        name=name,
        description="Sharp X-normal interface between transparent and opaque resin.",
        geometry=geometry,
        labels=labels,
        selected_indices=((0, 0, 2), (0, 0, 3), (1, 3, 5)),
    )


def layered_material_slab() -> SyntheticFixture:
    """Return four Z layers with the sequence transparent, white, black, clear."""
    name = "layered-material-slab"
    geometry = _geometry((4, 3, 5))
    labels = np.empty(geometry.shape_zyx, dtype=np.uint16)
    for z, material_id in enumerate((1, 2, 3, 1)):
        labels[z, :, :] = material_id
    return _label_fixture(
        name=name,
        description="Four Z-normal material layers with a non-palindromic sequence.",
        geometry=geometry,
        labels=labels,
        selected_indices=((0, 1, 2), (1, 1, 2), (2, 1, 2), (3, 1, 2)),
    )


def two_material_mixture_ramp() -> SyntheticFixture:
    """Return a linear X ramp from transparent resin to white resin."""
    name = "two-material-mixture-ramp"
    geometry = _geometry((2, 3, 5))
    palette = _mixture_palette()
    material_ids = np.asarray(palette.material_ids, dtype=np.uint16)

    fractions = np.zeros((*geometry.shape_zyx, len(palette)), dtype=np.float32)
    white_fraction = np.linspace(0.0, 1.0, geometry.shape_xyz[0], dtype=np.float32)
    fractions[..., 1] = 1.0 - white_fraction
    fractions[..., 2] = white_fraction

    volume = MaterialMixtureVolume(
        geometry=geometry,
        palette=palette,
        provenance=_provenance(name),
        fractions=fractions,
        material_ids=material_ids,
    )
    return SyntheticFixture(
        volume=volume,
        manifest=_mixture_manifest(
            name=name,
            description="Linear X ramp from transparent to white resin.",
            volume=volume,
            selected_indices=((0, 1, 0), (0, 1, 2), (0, 1, 4)),
        ),
    )


def anisotropic_axis_marker() -> SyntheticFixture:
    """Return unequal X/Y/Z marker lengths on an anisotropic `4 x 3 x 2` grid."""
    name = "anisotropic-axis-marker"
    geometry = _geometry((2, 3, 4))
    labels = np.zeros(geometry.shape_zyx, dtype=np.uint16)

    labels[0, 0, :] = 10  # Four-cell X marker.
    labels[1, :, 0] = 20  # Three-cell Y marker on the other Z plane.
    labels[:, 2, 3] = 30  # Two-cell Z marker, disjoint from X and Y.

    return _label_fixture(
        name=name,
        description=(
            "Axis marker with X/Y/Z lengths 4/3/2 and distinct voxel dimensions."
        ),
        geometry=geometry,
        labels=labels,
        selected_indices=(
            (0, 0, 0),
            (0, 0, 3),
            (1, 0, 0),
            (1, 2, 0),
            (0, 2, 3),
            (1, 2, 3),
            (0, 1, 1),
        ),
        palette=_axis_palette(),
    )


SYNTHETIC_FIXTURE_FACTORIES: tuple[FixtureFactory, ...] = (
    homogeneous_transparent,
    homogeneous_scattering_white,
    transparent_opaque_interface,
    layered_material_slab,
    two_material_mixture_ramp,
    anisotropic_axis_marker,
)


def all_synthetic_fixtures() -> tuple[SyntheticFixture, ...]:
    """Generate every canonical Phase 0 synthetic fixture in stable order."""
    return tuple(factory() for factory in SYNTHETIC_FIXTURE_FACTORIES)


def _geometry(shape_zyx: ShapeZYX) -> GridGeometry:
    return GridGeometry(
        shape_zyx=shape_zyx,
        voxel_size_xyz_m=_DEFAULT_VOXEL_SIZE_XYZ_M,
        local_to_world=_TRANSLATED_LOCAL_TO_WORLD,
    )


def _common_palette() -> MaterialPalette:
    return MaterialPalette(
        (
            MaterialDefinition(0, "air", MaterialRole.BACKGROUND),
            MaterialDefinition(1, "transparent-resin", MaterialRole.MATERIAL),
            MaterialDefinition(2, "white-resin", MaterialRole.MATERIAL),
            MaterialDefinition(3, "black-opaque-resin", MaterialRole.MATERIAL),
        )
    )


def _mixture_palette() -> MaterialPalette:
    return MaterialPalette(
        (
            MaterialDefinition(0, "air", MaterialRole.BACKGROUND),
            MaterialDefinition(1, "transparent-resin", MaterialRole.MATERIAL),
            MaterialDefinition(2, "white-resin", MaterialRole.MATERIAL),
        )
    )


def _axis_palette() -> MaterialPalette:
    return MaterialPalette(
        (
            MaterialDefinition(0, "air", MaterialRole.BACKGROUND),
            MaterialDefinition(10, "axis-x-diagnostic", MaterialRole.MATERIAL),
            MaterialDefinition(20, "axis-y-diagnostic", MaterialRole.MATERIAL),
            MaterialDefinition(30, "axis-z-diagnostic", MaterialRole.MATERIAL),
        )
    )


def _provenance(name: str) -> Provenance:
    return Provenance(
        generator=FIXTURE_GENERATOR,
        generator_version=FIXTURE_GENERATOR_VERSION,
        sources=(f"fixture:{name}",),
    )


def _label_fixture(
    *,
    name: str,
    description: str,
    geometry: GridGeometry,
    labels: npt.NDArray[np.uint16],
    selected_indices: Sequence[IndexZYX],
    palette: MaterialPalette | None = None,
) -> SyntheticFixture:
    volume = MaterialLabelVolume(
        geometry=geometry,
        palette=palette if palette is not None else _common_palette(),
        provenance=_provenance(name),
        material_id=labels,
    )
    return SyntheticFixture(
        volume=volume,
        manifest=_label_manifest(
            name=name,
            description=description,
            volume=volume,
            selected_indices=selected_indices,
        ),
    )


def _label_manifest(
    *,
    name: str,
    description: str,
    volume: MaterialLabelVolume,
    selected_indices: Sequence[IndexZYX],
) -> FixtureManifest:
    material_ids, counts = np.unique(volume.material_id, return_counts=True)
    material_voxel_counts = tuple(
        (int(material_id), int(count))
        for material_id, count in zip(material_ids, counts, strict=True)
    )
    selected_cells = tuple(
        SelectedCellExpectation(
            index_zyx=index,
            world_center_xyz_m=volume.geometry.cell_center_world(index),
            material_id=int(volume.material_id[index]),
        )
        for index in selected_indices
    )
    return _manifest(
        name=name,
        description=description,
        volume=volume,
        material_voxel_counts=material_voxel_counts,
        material_fraction_totals=(),
        selected_cells=selected_cells,
    )


def _mixture_manifest(
    *,
    name: str,
    description: str,
    volume: MaterialMixtureVolume,
    selected_indices: Sequence[IndexZYX],
) -> FixtureManifest:
    fraction_totals = cast(
        npt.NDArray[np.float64],
        np.sum(volume.fractions, axis=(0, 1, 2), dtype=np.float64),
    )
    material_fraction_totals = tuple(
        (int(material_id), float(total))
        for material_id, total in zip(volume.material_ids, fraction_totals, strict=True)
    )
    selected_cells = tuple(
        SelectedCellExpectation(
            index_zyx=index,
            world_center_xyz_m=volume.geometry.cell_center_world(index),
            fractions=tuple(float(item) for item in volume.fractions[index]),
        )
        for index in selected_indices
    )
    return _manifest(
        name=name,
        description=description,
        volume=volume,
        material_voxel_counts=(),
        material_fraction_totals=material_fraction_totals,
        selected_cells=selected_cells,
    )


def _manifest(
    *,
    name: str,
    description: str,
    volume: CanonicalMaterialVolume,
    material_voxel_counts: tuple[tuple[int, int], ...],
    material_fraction_totals: tuple[tuple[int, float], ...],
    selected_cells: tuple[SelectedCellExpectation, ...],
) -> FixtureManifest:
    geometry = volume.geometry
    return FixtureManifest(
        name=name,
        description=description,
        shape_zyx=geometry.shape_zyx,
        local_bounds_xyz_m=((0.0, 0.0, 0.0), geometry.local_extent_xyz_m),
        world_bounds_xyz_m=_world_bounds(geometry),
        material_voxel_counts=material_voxel_counts,
        material_fraction_totals=material_fraction_totals,
        selected_cells=selected_cells,
    )


def _world_bounds(geometry: GridGeometry) -> BoundsXYZ:
    nx, ny, nz = geometry.shape_xyz
    corners = tuple(
        geometry.continuous_index_to_world(index_xyz)
        for index_xyz in product((0.0, float(nx)), (0.0, float(ny)), (0.0, float(nz)))
    )
    minimum = cast(
        PointXYZ, tuple(min(point[axis] for point in corners) for axis in range(3))
    )
    maximum = cast(
        PointXYZ, tuple(max(point[axis] for point in corners) for axis in range(3))
    )
    return (minimum, maximum)
