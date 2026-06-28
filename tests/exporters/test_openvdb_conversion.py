from dataclasses import replace

import numpy as np
import pytest

from vbdmat.boundaries import CapabilityStatus
from vbdmat.exporters.openvdb import (
    OpenVDBExportConfig,
    convert_openvdb_fields,
    openvdb_capability_report,
)
from vbdmat.fixtures import anisotropic_axis_marker, layered_material_slab
from vbdmat.optics import map_material_volume_to_optical, phase0_provisional_mapping


def _mapped(fixture):  # type: ignore[no-untyped-def]
    return map_material_volume_to_optical(fixture.volume, phase0_provisional_mapping())


def test_fields_are_named_float32_xyz_grids_with_selected_values() -> None:
    volume = _mapped(anisotropic_axis_marker())
    converted = convert_openvdb_fields(volume)
    assert tuple(converted.fields_xyz) == (
        "sigma_a_r",
        "sigma_a_g",
        "sigma_a_b",
        "sigma_s_r",
        "sigma_s_g",
        "sigma_s_b",
        "g",
        "ior",
        "cycles_absorption",
        "cycles_scattering",
    )
    assert all(array.shape == (4, 3, 2) for array in converted.fields_xyz.values())
    assert all(array.dtype == np.float32 for array in converted.fields_xyz.values())
    assert all(not array.flags.writeable for array in converted.fields_xyz.values())
    assert converted.fields_xyz["sigma_a_g"][3, 0, 0] == volume.sigma_a[0, 0, 3, 1]
    assert converted.fields_xyz["sigma_a_r"][0, 2, 1] == volume.sigma_a[1, 2, 0, 0]


def test_index_transform_maps_integer_indices_to_canonical_cell_centres() -> None:
    volume = _mapped(anisotropic_axis_marker())
    matrix = np.asarray(convert_openvdb_fields(volume).index_to_world)
    for zyx in ((0, 0, 0), (1, 2, 3)):
        z, y, x = zyx
        actual = matrix @ np.asarray((x, y, z, 1.0))
        np.testing.assert_allclose(actual[:3], volume.geometry.cell_center_world(zyx))


def test_cycles_reduction_is_explicit_and_phase_is_weighted() -> None:
    volume = _mapped(layered_material_slab())
    converted = convert_openvdb_fields(volume)
    expected_a = np.mean(volume.sigma_a, axis=-1).transpose(2, 1, 0)
    np.testing.assert_allclose(converted.fields_xyz["cycles_absorption"], expected_a)
    weights = np.mean(volume.sigma_s, axis=-1, dtype=np.float64)
    expected_g = np.sum(volume.g * weights) / np.sum(weights)
    assert converted.phase_g == pytest.approx(expected_g)


def test_capabilities_are_complete_and_do_not_hide_ior() -> None:
    volume = _mapped(layered_material_slab())
    converted = convert_openvdb_fields(volume)
    report = openvdb_capability_report(volume, converted)
    assert {entry.field for entry in report.entries} == {
        "geometry",
        "coefficient_units",
        "optical_basis",
        "sigma_a",
        "sigma_s",
        "g",
        "ior",
        "derived_ior_interfaces",
        "provenance",
    }
    assert report.by_field("sigma_a").status is CapabilityStatus.APPROXIMATED
    assert report.by_field("ior").status is CapabilityStatus.UNSUPPORTED
    assert (
        report.by_field("derived_ior_interfaces").status is CapabilityStatus.UNSUPPORTED
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0},
        {"samples": True},
        {"seed": -1},
        {"rgb_weights": (0.2, 0.2, 0.2)},
        {"rgb_weights": (1.0, -1.0, 1.0)},
    ],
)
def test_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        OpenVDBExportConfig(**kwargs)  # type: ignore[arg-type]


def test_zero_scattering_has_zero_phase() -> None:
    volume = _mapped(layered_material_slab())
    converted = convert_openvdb_fields(
        replace(
            volume, sigma_s=np.zeros_like(volume.sigma_s), g=np.zeros_like(volume.g)
        )
    )
    assert converted.phase_g == 0.0
