from dataclasses import replace

import numpy as np
import pytest

from vdbmat.boundaries import (
    BoundaryAxis,
    BoundaryDerivationConfig,
    CapabilityStatus,
    ConsumerKind,
    derive_ior_interfaces,
    ior_policy,
)
from vdbmat.core import GridGeometry
from vdbmat.fixtures import all_synthetic_fixtures, transparent_opaque_interface
from vdbmat.optics import (
    map_material_volume_to_optical,
    phase0_provisional_mapping,
)


def _mapped_fixture(name: str):  # type: ignore[no-untyped-def]
    fixture = next(
        item for item in all_synthetic_fixtures() if item.manifest.name == name
    )
    return map_material_volume_to_optical(fixture.volume, phase0_provisional_mapping())


@pytest.mark.parametrize(
    ("name", "total", "interior", "exterior"),
    [
        ("homogeneous-transparent", 52, 0, 52),
        ("homogeneous-scattering-white", 52, 0, 52),
        ("transparent-opaque-interface", 96, 8, 88),
        ("layered-material-slab", 124, 30, 94),
        ("two-material-mixture-ramp", 86, 24, 62),
        ("anisotropic-axis-marker", 0, 0, 0),
    ],
)
def test_fixture_interface_counts(
    name: str, total: int, interior: int, exterior: int
) -> None:
    interfaces = derive_ior_interfaces(_mapped_fixture(name))
    assert len(interfaces.faces) == total
    assert len(interfaces.interior_faces) == interior
    assert len(interfaces.exterior_faces) == exterior


def test_interface_fixture_has_oriented_sharp_x_faces() -> None:
    interfaces = derive_ior_interfaces(
        _mapped_fixture("transparent-opaque-interface"),
        BoundaryDerivationConfig(include_exterior=False),
    )
    assert len(interfaces.faces) == 8
    assert {face.axis for face in interfaces.faces} == {BoundaryAxis.X}
    assert {face.corner_index_xyz[0] for face in interfaces.faces} == {3}
    for face in interfaces.faces:
        assert face.negative_cell_zyx is not None
        assert face.positive_cell_zyx is not None
        assert face.negative_cell_zyx[2] == 2
        assert face.positive_cell_zyx[2] == 3
        assert face.ior_negative == pytest.approx(1.48)
        assert face.ior_positive == pytest.approx(1.52)


def test_equal_ior_material_regions_do_not_create_false_interfaces() -> None:
    interfaces = derive_ior_interfaces(
        _mapped_fixture("layered-material-slab"),
        BoundaryDerivationConfig(include_exterior=False),
    )
    assert len(interfaces.faces) == 30
    assert {face.corner_index_xyz[2] for face in interfaces.faces} == {1, 3}
    assert 2 not in {face.corner_index_xyz[2] for face in interfaces.faces}


def test_tolerance_controls_piecewise_constant_interface_detection() -> None:
    volume = _mapped_fixture("homogeneous-transparent")
    ior = np.array(volume.ior, copy=True)
    ior[:, :, 2:] += np.float32(5e-7)
    perturbed = replace(volume, ior=ior)
    assert not derive_ior_interfaces(
        perturbed, BoundaryDerivationConfig(include_exterior=False)
    ).faces
    assert (
        len(
            derive_ior_interfaces(
                perturbed,
                BoundaryDerivationConfig(
                    include_exterior=False, ior_absolute_tolerance=1e-8
                ),
            ).faces
        )
        == 6
    )


def test_exterior_faces_compare_cells_with_explicit_ambient() -> None:
    volume = _mapped_fixture("homogeneous-transparent")
    interfaces = derive_ior_interfaces(
        volume, BoundaryDerivationConfig(ambient_ior=1.48)
    )
    assert not interfaces.faces


def test_world_geometry_preserves_rotation_winding_and_scale() -> None:
    volume = map_material_volume_to_optical(
        transparent_opaque_interface().volume, phase0_provisional_mapping()
    )
    geometry = GridGeometry(
        shape_zyx=volume.geometry.shape_zyx,
        voxel_size_xyz_m=volume.geometry.voxel_size_xyz_m,
        local_to_world=(
            (0.0, -1.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 2.0),
            (0.0, 0.0, 1.0, 3.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    rotated = replace(volume, geometry=geometry)
    interfaces = derive_ior_interfaces(
        rotated, BoundaryDerivationConfig(include_exterior=False)
    )
    face = interfaces.faces[0]
    corners = interfaces.world_corners(face)
    assert interfaces.world_normal(face) == pytest.approx((0.0, 1.0, 0.0))
    assert corners[0] == pytest.approx(geometry.continuous_index_to_world((3, 0, 0)))
    assert np.linalg.norm(np.subtract(corners[1], corners[0])) == pytest.approx(
        geometry.voxel_size_xyz_m[1]
    )
    assert np.linalg.norm(np.subtract(corners[3], corners[0])) == pytest.approx(
        geometry.voxel_size_xyz_m[2]
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"ambient_ior": 0.0}, ValueError, "ambient_ior"),
        ({"ior_absolute_tolerance": -1.0}, ValueError, "tolerance"),
        ({"include_exterior": 1}, TypeError, "include_exterior"),
    ],
)
def test_derivation_config_rejects_invalid_values(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        BoundaryDerivationConfig(**kwargs)  # type: ignore[arg-type]


def test_mitsuba_ior_policy_is_explicit() -> None:
    policy = ior_policy(ConsumerKind.MITSUBA_3)
    assert policy.spatial_ior is CapabilityStatus.UNSUPPORTED
    assert policy.derived_interfaces is CapabilityStatus.TRANSFORMED
    assert "ext_ior/int_ior" in policy.exterior_boundary
    assert "never interpolate IOR" in policy.interpolation


def test_cycles_ior_policy_is_explicit() -> None:
    policy = ior_policy(ConsumerKind.BLENDER_CYCLES)
    assert policy.spatial_ior is CapabilityStatus.UNSUPPORTED
    assert policy.derived_interfaces is CapabilityStatus.APPROXIMATED
    assert "OpenVDB" in policy.internal_boundary
    assert "never interpolate IOR" in policy.interpolation
