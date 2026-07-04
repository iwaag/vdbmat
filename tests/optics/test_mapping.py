"""Tests for deterministic material-to-optical conversion."""

import numpy as np
import pytest

from vbdmat.core import (
    MaterialLabelVolume,
    MaterialMixtureVolume,
    OpticalPropertyVolume,
    Provenance,
    VolumeValidationError,
)
from vbdmat.fixtures import (
    SYNTHETIC_FIXTURE_FACTORIES,
    anisotropic_axis_marker,
    homogeneous_scattering_white,
    homogeneous_transparent,
    transparent_opaque_interface,
    two_material_mixture_ramp,
)
from vbdmat.optics import (
    MAPPING_GENERATOR,
    MAPPING_GENERATOR_VERSION,
    MaterialOpticalProperties,
    OpticalMappingConfig,
    OpticalMappingError,
    map_material_volume_to_optical,
    phase0_provisional_mapping,
)


@pytest.mark.parametrize("factory", SYNTHETIC_FIXTURE_FACTORIES)
def test_every_synthetic_fixture_maps_to_valid_optical_volume(factory) -> None:  # type: ignore[no-untyped-def]
    fixture = factory()

    output = map_material_volume_to_optical(
        fixture.volume, phase0_provisional_mapping()
    )

    assert isinstance(output, OpticalPropertyVolume)
    assert output.geometry is fixture.volume.geometry
    assert output.schema is fixture.volume.schema
    assert output.sigma_a.shape == (*fixture.volume.geometry.shape_zyx, 3)
    assert output.sigma_s.shape == (*fixture.volume.geometry.shape_zyx, 3)
    assert output.g.shape == fixture.volume.geometry.shape_zyx
    assert output.ior.shape == fixture.volume.geometry.shape_zyx


def test_label_mapping_is_direct_lookup() -> None:
    fixture = homogeneous_transparent()
    config = phase0_provisional_mapping()
    expected = config.by_id(1)

    output = map_material_volume_to_optical(fixture.volume, config)

    assert np.all(output.sigma_a == np.asarray(expected.sigma_a_rgb_per_m))
    assert np.all(output.sigma_s == np.asarray(expected.sigma_s_rgb_per_m))
    assert np.all(output.g == expected.g)
    assert np.all(output.ior == expected.ior)


def test_white_mapping_uses_provisional_scattering_coefficients() -> None:
    output = map_material_volume_to_optical(
        homogeneous_scattering_white().volume, phase0_provisional_mapping()
    )

    assert np.all(output.sigma_a == (1.0, 1.0, 1.0))
    assert np.all(output.sigma_s == (1000.0, 1000.0, 1000.0))
    assert np.all(output.g == np.float32(0.2))
    assert np.all(output.ior == np.float32(1.52))


def test_interface_mapping_preserves_material_regions() -> None:
    fixture = transparent_opaque_interface()
    config = phase0_provisional_mapping()
    output = map_material_volume_to_optical(fixture.volume, config)

    assert np.all(output.sigma_a[:, :, :3] == (2.0, 1.0, 0.5))
    assert np.all(output.sigma_a[:, :, 3:] == (4000.0, 5000.0, 6000.0))
    assert np.all(output.sigma_s[:, :, :3] == 0.0)
    assert np.all(output.sigma_s[:, :, 3:] == 100.0)


def test_mixture_ramp_matches_hand_calculated_midpoint() -> None:
    fixture = two_material_mixture_ramp()
    output = map_material_volume_to_optical(
        fixture.volume, phase0_provisional_mapping()
    )

    assert output.sigma_a[0, 1, 2] == pytest.approx((1.5, 1.0, 0.75))
    assert output.sigma_s[0, 1, 2] == pytest.approx((500.0, 500.0, 500.0))
    assert float(output.g[0, 1, 2]) == pytest.approx(0.1)
    assert float(output.ior[0, 1, 2]) == pytest.approx(1.5)


def test_pure_mixture_endpoints_equal_direct_label_lookup() -> None:
    fixture = two_material_mixture_ramp()
    assert isinstance(fixture.volume, MaterialMixtureVolume)
    mixture = fixture.volume
    config = phase0_provisional_mapping()
    mixture_output = map_material_volume_to_optical(mixture, config)

    transparent_labels = np.full(mixture.geometry.shape_zyx, 1, dtype=np.uint16)
    white_labels = np.full(mixture.geometry.shape_zyx, 2, dtype=np.uint16)
    transparent = MaterialLabelVolume(
        mixture.geometry,
        mixture.palette,
        mixture.provenance,
        transparent_labels,
    )
    white = MaterialLabelVolume(
        mixture.geometry,
        mixture.palette,
        mixture.provenance,
        white_labels,
    )
    transparent_output = map_material_volume_to_optical(transparent, config)
    white_output = map_material_volume_to_optical(white, config)

    for field in ("sigma_a", "sigma_s", "g", "ior"):
        mixture_field = getattr(mixture_output, field)
        transparent_field = getattr(transparent_output, field)
        white_field = getattr(white_output, field)
        assert np.array_equal(mixture_field[:, :, 0], transparent_field[:, :, 0])
        assert np.array_equal(mixture_field[:, :, -1], white_field[:, :, -1])


def test_axis_marker_receives_distinct_diagnostic_coefficients() -> None:
    fixture = anisotropic_axis_marker()
    output = map_material_volume_to_optical(
        fixture.volume, phase0_provisional_mapping()
    )

    assert np.array_equal(output.sigma_a[0, 0, 0], (0.0, 100.0, 100.0))
    assert np.array_equal(output.sigma_a[1, 0, 0], (100.0, 0.0, 100.0))
    assert np.array_equal(output.sigma_a[0, 2, 3], (100.0, 100.0, 0.0))
    assert np.array_equal(output.sigma_a[0, 1, 1], (0.0, 0.0, 0.0))


def test_output_provenance_preserves_config_and_source_identity() -> None:
    fixture = homogeneous_transparent()
    config = phase0_provisional_mapping()

    output = map_material_volume_to_optical(fixture.volume, config)

    assert output.provenance.generator == MAPPING_GENERATOR
    assert output.provenance.generator_version == MAPPING_GENERATOR_VERSION
    assert output.provenance.configuration_digest == config.digest
    assert output.provenance.created_utc is None
    assert "fixture:homogeneous-transparent" in output.provenance.sources
    assert any(
        item.startswith("source-generator:vbdmat.synthetic-fixtures@")
        for item in output.provenance.sources
    )
    assert any(
        item.startswith("source-provenance-sha256:")
        for item in output.provenance.sources
    )
    assert (
        "mapping-config:phase0-provisional-materials-v1@1.0.0"
        in output.provenance.sources
    )
    assert output.provenance.notes is not None
    assert "Provisional uncalibrated" in output.provenance.notes


def test_repeated_mapping_is_identical_and_side_effect_free() -> None:
    fixture = two_material_mixture_ramp()
    assert isinstance(fixture.volume, MaterialMixtureVolume)
    config = phase0_provisional_mapping()
    input_before = fixture.volume.fractions.copy()

    first = map_material_volume_to_optical(fixture.volume, config)
    second = map_material_volume_to_optical(fixture.volume, config)

    assert first.provenance == second.provenance
    assert first.optical_basis == second.optical_basis
    for field in ("sigma_a", "sigma_s", "g", "ior"):
        assert np.array_equal(getattr(first, field), getattr(second, field))
    assert np.array_equal(fixture.volume.fractions, input_before)
    assert not np.shares_memory(first.sigma_a, fixture.volume.fractions)


def test_missing_declared_palette_mapping_fails_before_conversion() -> None:
    fixture = homogeneous_transparent()
    full = phase0_provisional_mapping()
    missing_black = OpticalMappingConfig(
        configuration_id="missing-black",
        version=full.version,
        materials=tuple(item for item in full.materials if item.material_id != 3),
    )

    with pytest.raises(OpticalMappingError) as captured:
        map_material_volume_to_optical(fixture.volume, missing_black)

    assert captured.value.field_path == "palette.material_ids"
    assert "(3,)" in captured.value.message


def test_palette_and_mapping_name_disagreement_fails(tmp_path: object) -> None:
    # ADR-009 D4: a shared material_id whose names disagree is an error, never a
    # silent wrong-coefficient application.
    fixture = homogeneous_transparent()
    full = phase0_provisional_mapping()
    renamed = OpticalMappingConfig(
        configuration_id="renamed-id-1",
        version=full.version,
        materials=tuple(
            item
            if item.material_id != 1
            else MaterialOpticalProperties(
                1,
                "some-other-resin",
                sigma_a_rgb_per_m=item.sigma_a_rgb_per_m,
                sigma_s_rgb_per_m=item.sigma_s_rgb_per_m,
                g=item.g,
                ior=item.ior,
            )
            for item in full.materials
        ),
    )

    with pytest.raises(OpticalMappingError) as captured:
        map_material_volume_to_optical(fixture.volume, renamed)

    assert captured.value.field_path == "palette.materials"
    assert "some-other-resin" in captured.value.message


def test_invalid_mixture_fractions_fail_at_volume_boundary_before_mapping() -> None:
    fixture = two_material_mixture_ramp()
    assert isinstance(fixture.volume, MaterialMixtureVolume)
    volume = fixture.volume
    invalid = volume.fractions.copy()
    invalid[0, 0, 0] = (0.0, -0.1, 1.1)

    with pytest.raises(VolumeValidationError, match=r"arrays\.fractions"):
        MaterialMixtureVolume(
            geometry=volume.geometry,
            palette=volume.palette,
            provenance=volume.provenance,
            fractions=invalid,
            material_ids=volume.material_ids,
        )


def test_input_provenance_changes_output_fingerprint_deterministically() -> None:
    fixture = homogeneous_transparent()
    assert isinstance(fixture.volume, MaterialLabelVolume)
    changed_source = MaterialLabelVolume(
        geometry=fixture.volume.geometry,
        palette=fixture.volume.palette,
        provenance=Provenance("different-source", "1.0.0"),
        material_id=fixture.volume.material_id,
    )
    config = phase0_provisional_mapping()

    original = map_material_volume_to_optical(fixture.volume, config)
    changed = map_material_volume_to_optical(changed_source, config)

    original_fingerprint = next(
        item
        for item in original.provenance.sources
        if item.startswith("source-provenance-sha256:")
    )
    changed_fingerprint = next(
        item
        for item in changed.provenance.sources
        if item.startswith("source-provenance-sha256:")
    )
    assert original_fingerprint != changed_fingerprint
