from dataclasses import replace

import numpy as np
import pytest

from vdbmat.boundaries import CapabilityStatus
from vdbmat.exporters.mitsuba import (
    MitsubaExportConfig,
    convert_optical_fields,
    mitsuba_capability_report,
)
from vdbmat.fixtures import (
    anisotropic_axis_marker,
    homogeneous_transparent,
    layered_material_slab,
)
from vdbmat.optics import (
    map_material_volume_to_optical,
    phase0_provisional_mapping,
)


def _map(fixture):  # type: ignore[no-untyped-def]
    return map_material_volume_to_optical(fixture.volume, phase0_provisional_mapping())


def test_sigma_t_and_albedo_conversion_is_exact() -> None:
    volume = _map(layered_material_slab())
    conversion = convert_optical_fields(volume)
    np.testing.assert_array_equal(conversion.sigma_t, volume.sigma_a + volume.sigma_s)
    expected = np.zeros_like(volume.sigma_s)
    np.divide(
        volume.sigma_s,
        conversion.sigma_t,
        out=expected,
        where=conversion.sigma_t > 0.0,
    )
    np.testing.assert_array_equal(conversion.albedo, expected)
    assert not conversion.sigma_t.flags.writeable
    assert not conversion.albedo.flags.writeable


def test_zero_extinction_maps_to_zero_albedo() -> None:
    volume = _map(homogeneous_transparent())
    zero = np.zeros_like(volume.sigma_a)
    converted = convert_optical_fields(
        replace(volume, sigma_a=zero, sigma_s=zero, g=np.zeros_like(volume.g))
    )
    assert not np.any(converted.sigma_t)
    assert not np.any(converted.albedo)
    assert converted.phase_g == 0.0


def test_phase_g_is_scattering_weighted_global_mean() -> None:
    volume = _map(layered_material_slab())
    converted = convert_optical_fields(volume)
    weights = np.mean(volume.sigma_s, axis=-1, dtype=np.float64)
    expected = np.sum(volume.g * weights, dtype=np.float64) / np.sum(weights)
    assert converted.phase_g == pytest.approx(expected)


def test_axis_marker_tensor_order_and_metric_transform() -> None:
    volume = _map(anisotropic_axis_marker())
    converted = convert_optical_fields(volume)
    np.testing.assert_array_equal(converted.sigma_t, volume.sigma_a + volume.sigma_s)
    assert converted.sigma_t.shape == (2, 3, 4, 3)
    np.testing.assert_array_equal(converted.sigma_t[0, 0, 3], (0.0, 100.0, 100.0))
    np.testing.assert_array_equal(converted.sigma_t[1, 2, 0], (100.0, 0.0, 100.0))
    np.testing.assert_array_equal(converted.sigma_t[1, 2, 3], (100.0, 100.0, 0.0))
    matrix = np.asarray(converted.volume_to_world)
    np.testing.assert_allclose(matrix[:3, 3], (0.01, 0.02, 0.03))
    np.testing.assert_allclose(
        matrix[:3, :3], np.diag(volume.geometry.local_extent_xyz_m)
    )


def test_capability_report_covers_every_required_semantic() -> None:
    volume = _map(layered_material_slab())
    conversion = convert_optical_fields(volume)
    report = mitsuba_capability_report(volume, conversion)
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
    assert report.by_field("sigma_a").status is CapabilityStatus.TRANSFORMED
    assert report.by_field("g").status is CapabilityStatus.APPROXIMATED
    assert report.by_field("ior").status is CapabilityStatus.UNSUPPORTED
    assert (
        report.by_field("derived_ior_interfaces").status is CapabilityStatus.TRANSFORMED
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"width": 0}, ValueError, "width"),
        ({"spp": True}, TypeError, "spp"),
        ({"seed": -1}, ValueError, "seed"),
        ({"fov_degrees": 179.0}, ValueError, "fov"),
        ({"ambient_ior": 0.0}, ValueError, "ambient_ior"),
        ({"ior_absolute_tolerance": -1.0}, ValueError, "tolerance"),
        ({"attenuation_diagnostic_gain": 0.0}, ValueError, "diagnostic_gain"),
        ({"variant": ""}, ValueError, "variant"),
    ],
)
def test_export_config_rejects_invalid_values(
    kwargs: dict[str, object], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        MitsubaExportConfig(**kwargs)  # type: ignore[arg-type]
