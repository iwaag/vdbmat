"""Regression tests for deterministic Phase 0 synthetic fixtures."""

from itertools import combinations

import numpy as np
import pytest

from vdbmat.core import (
    MaterialLabelVolume,
    MaterialMixtureVolume,
    VolumeValidationError,
)
from vdbmat.fixtures import (
    FIXTURE_GENERATOR,
    FIXTURE_GENERATOR_VERSION,
    SYNTHETIC_FIXTURE_FACTORIES,
    SyntheticFixture,
    all_synthetic_fixtures,
    anisotropic_axis_marker,
    homogeneous_scattering_white,
    homogeneous_transparent,
    layered_material_slab,
    transparent_opaque_interface,
    two_material_mixture_ramp,
)


def test_all_fixture_order_and_names_are_stable() -> None:
    fixtures = all_synthetic_fixtures()

    assert tuple(item.manifest.name for item in fixtures) == (
        "homogeneous-transparent",
        "homogeneous-scattering-white",
        "transparent-opaque-interface",
        "layered-material-slab",
        "two-material-mixture-ramp",
        "anisotropic-axis-marker",
    )
    assert len(SYNTHETIC_FIXTURE_FACTORIES) == 6


@pytest.mark.parametrize("factory", SYNTHETIC_FIXTURE_FACTORIES)
def test_generation_is_deterministic(factory) -> None:  # type: ignore[no-untyped-def]
    first = factory()
    second = factory()

    assert first.manifest == second.manifest
    assert type(first.volume) is type(second.volume)
    assert first.volume.geometry == second.volume.geometry
    assert first.volume.provenance == second.volume.provenance
    assert first.volume.provenance.generator == FIXTURE_GENERATOR
    assert first.volume.provenance.generator_version == FIXTURE_GENERATOR_VERSION
    assert first.volume.provenance.created_utc is None

    if isinstance(first.volume, MaterialLabelVolume):
        assert isinstance(second.volume, MaterialLabelVolume)
        assert np.array_equal(first.volume.material_id, second.volume.material_id)
    else:
        assert isinstance(first.volume, MaterialMixtureVolume)
        assert isinstance(second.volume, MaterialMixtureVolume)
        assert np.array_equal(first.volume.material_ids, second.volume.material_ids)
        assert np.array_equal(first.volume.fractions, second.volume.fractions)


@pytest.mark.parametrize("factory", SYNTHETIC_FIXTURE_FACTORIES)
def test_manifest_recomputes_from_canonical_volume(factory) -> None:  # type: ignore[no-untyped-def]
    fixture = factory()
    volume = fixture.volume
    manifest = fixture.manifest

    assert manifest.shape_zyx == volume.geometry.shape_zyx
    assert manifest.local_bounds_xyz_m == (
        (0.0, 0.0, 0.0),
        volume.geometry.local_extent_xyz_m,
    )
    assert sum(count for _, count in manifest.material_voxel_counts) in (
        0,
        int(np.prod(volume.geometry.shape_zyx)),
    )

    for expected in manifest.selected_cells:
        assert expected.world_center_xyz_m == pytest.approx(
            volume.geometry.cell_center_world(expected.index_zyx)
        )
        if isinstance(volume, MaterialLabelVolume):
            assert expected.material_id == int(volume.material_id[expected.index_zyx])
            assert expected.fractions is None
        else:
            assert expected.material_id is None
            assert expected.fractions == pytest.approx(
                tuple(float(item) for item in volume.fractions[expected.index_zyx])
            )

    if isinstance(volume, MaterialLabelVolume):
        ids, counts = np.unique(volume.material_id, return_counts=True)
        recomputed = tuple(
            (int(material_id), int(count))
            for material_id, count in zip(ids, counts, strict=True)
        )
        assert manifest.material_voxel_counts == recomputed
        assert manifest.material_fraction_totals == ()
    else:
        totals = np.sum(volume.fractions, axis=(0, 1, 2), dtype=np.float64)
        recomputed_totals = tuple(
            (int(material_id), float(total))
            for material_id, total in zip(volume.material_ids, totals, strict=True)
        )
        assert manifest.material_voxel_counts == ()
        assert manifest.material_fraction_totals == recomputed_totals


@pytest.mark.parametrize(
    ("factory", "expected_counts"),
    [
        (homogeneous_transparent, ((1, 24),)),
        (homogeneous_scattering_white, ((2, 24),)),
        (transparent_opaque_interface, ((1, 24), (3, 24))),
        (layered_material_slab, ((1, 30), (2, 15), (3, 15))),
        (anisotropic_axis_marker, ((0, 15), (10, 4), (20, 3), (30, 2))),
    ],
)
def test_label_fixture_material_counts(factory, expected_counts) -> None:  # type: ignore[no-untyped-def]
    fixture = factory()

    assert fixture.manifest.material_voxel_counts == expected_counts


def test_world_bounds_include_translation_and_anisotropic_extent() -> None:
    fixture = anisotropic_axis_marker()

    assert np.asarray(fixture.manifest.local_bounds_xyz_m) == pytest.approx(
        np.asarray(((0.0, 0.0, 0.0), (0.00016, 0.00015, 0.00006)))
    )
    assert np.asarray(fixture.manifest.world_bounds_xyz_m) == pytest.approx(
        np.asarray(((0.010, 0.020, 0.030), (0.01016, 0.02015, 0.03006)))
    )


def test_homogeneous_fixtures_are_uniform_and_distinct() -> None:
    transparent = homogeneous_transparent()
    white = homogeneous_scattering_white()

    assert isinstance(transparent.volume, MaterialLabelVolume)
    assert isinstance(white.volume, MaterialLabelVolume)
    assert np.all(transparent.volume.material_id == 1)
    assert np.all(white.volume.material_id == 2)
    assert transparent.manifest.selected_cells[0].material_id == 1
    assert white.manifest.selected_cells[-1].material_id == 2


def test_interface_has_one_sharp_x_boundary() -> None:
    fixture = transparent_opaque_interface()
    assert isinstance(fixture.volume, MaterialLabelVolume)
    labels = fixture.volume.material_id

    assert labels.shape == (2, 4, 6)
    assert np.all(labels[:, :, :3] == 1)
    assert np.all(labels[:, :, 3:] == 3)
    left_world = fixture.volume.geometry.cell_center_world((0, 0, 2))
    right_world = fixture.volume.geometry.cell_center_world((0, 0, 3))
    assert left_world[0] == pytest.approx(0.01010)
    assert right_world[0] == pytest.approx(0.01014)
    assert (left_world[0] + right_world[0]) / 2 == pytest.approx(0.01012)


def test_layered_slab_sequence_is_not_palindromic() -> None:
    fixture = layered_material_slab()
    assert isinstance(fixture.volume, MaterialLabelVolume)

    layer_values = tuple(int(fixture.volume.material_id[z, 1, 2]) for z in range(4))
    assert layer_values == (1, 2, 3, 1)
    assert layer_values != tuple(reversed(layer_values))
    assert (
        tuple(item.material_id for item in fixture.manifest.selected_cells)
        == layer_values
    )


def test_mixture_ramp_is_linear_and_has_expected_totals() -> None:
    fixture = two_material_mixture_ramp()
    assert isinstance(fixture.volume, MaterialMixtureVolume)
    volume = fixture.volume

    expected = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.75, 0.25],
            [0.0, 0.5, 0.5],
            [0.0, 0.25, 0.75],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    assert volume.material_ids.tolist() == [0, 1, 2]
    assert np.array_equal(volume.fractions[0, 1, :, :], expected)
    assert np.array_equal(volume.fractions[1, 2, :, :], expected)
    assert fixture.manifest.material_fraction_totals == (
        (0, 0.0),
        (1, 15.0),
        (2, 15.0),
    )
    assert tuple(item.fractions for item in fixture.manifest.selected_cells) == (
        (0.0, 1.0, 0.0),
        (0.0, 0.5, 0.5),
        (0.0, 0.0, 1.0),
    )


def test_axis_marker_encodes_distinct_x_y_z_lengths_and_world_deltas() -> None:
    fixture = anisotropic_axis_marker()
    assert isinstance(fixture.volume, MaterialLabelVolume)
    volume = fixture.volume
    labels = volume.material_id

    assert labels.shape == (2, 3, 4)
    assert np.array_equal(labels[0, 0, :], np.full(4, 10, dtype=np.uint16))
    assert np.array_equal(labels[1, :, 0], np.full(3, 20, dtype=np.uint16))
    assert np.array_equal(labels[:, 2, 3], np.full(2, 30, dtype=np.uint16))
    assert labels[0, 1, 1] == 0

    x_start = np.asarray(volume.geometry.cell_center_world((0, 0, 0)))
    x_end = np.asarray(volume.geometry.cell_center_world((0, 0, 3)))
    y_start = np.asarray(volume.geometry.cell_center_world((1, 0, 0)))
    y_end = np.asarray(volume.geometry.cell_center_world((1, 2, 0)))
    z_start = np.asarray(volume.geometry.cell_center_world((0, 2, 3)))
    z_end = np.asarray(volume.geometry.cell_center_world((1, 2, 3)))
    assert x_end - x_start == pytest.approx((0.00012, 0.0, 0.0))
    assert y_end - y_start == pytest.approx((0.0, 0.00010, 0.0))
    assert z_end - z_start == pytest.approx((0.0, 0.0, 0.00003))


@pytest.mark.parametrize("axes", [(1, 0, 2), (2, 1, 0), (0, 2, 1)])
def test_any_spatial_axis_swap_is_rejected_visibly(axes: tuple[int, int, int]) -> None:
    fixture = anisotropic_axis_marker()
    assert isinstance(fixture.volume, MaterialLabelVolume)
    volume = fixture.volume
    swapped = np.transpose(volume.material_id, axes).copy()

    assert swapped.shape != fixture.manifest.shape_zyx
    with pytest.raises(VolumeValidationError, match="shape"):
        MaterialLabelVolume(
            geometry=volume.geometry,
            palette=volume.palette,
            provenance=volume.provenance,
            material_id=swapped,
        )


def test_all_three_axis_pairs_are_covered() -> None:
    assert tuple(combinations(("z", "y", "x"), 2)) == (
        ("z", "y"),
        ("z", "x"),
        ("y", "x"),
    )


def test_fixture_wrapper_uses_identity_equality_for_numpy_safety() -> None:
    fixture = homogeneous_transparent()

    assert isinstance(fixture, SyntheticFixture)
    assert fixture == fixture
    assert fixture != homogeneous_transparent()
