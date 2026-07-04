"""Tests for provisional optical mapping configuration."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from vdbmat.core import OpticalBasis, SchemaVersion
from vdbmat.optics import (
    CalibrationStatus,
    MaterialOpticalProperties,
    MixingRule,
    OpticalMappingConfig,
    phase0_provisional_mapping,
)


def transparent_properties() -> MaterialOpticalProperties:
    return MaterialOpticalProperties(
        material_id=1,
        name="transparent-resin",
        sigma_a_rgb_per_m=(2.0, 1.0, 0.5),
        sigma_s_rgb_per_m=(0.0, 0.0, 0.0),
        g=0.0,
        ior=1.48,
    )


def test_phase0_configuration_is_explicit_and_uncalibrated() -> None:
    config = phase0_provisional_mapping()

    assert config.configuration_id == "phase0-provisional-materials-v1"
    assert config.version == SchemaVersion(1, 0, 0)
    assert config.optical_basis == OpticalBasis.phase0_rgb()
    assert config.mixing_rule is MixingRule.LINEAR_VOLUME_FRACTION_V1
    assert config.calibration_status is CalibrationStatus.PROVISIONAL_UNCALIBRATED
    assert config.material_ids == (0, 1, 2, 3, 10, 20, 30)
    assert config.by_id(1) == transparent_properties()
    assert config.by_id(2).sigma_s_rgb_per_m == (1000.0, 1000.0, 1000.0)


def test_configuration_order_is_normalized_before_digesting() -> None:
    original = phase0_provisional_mapping()
    reversed_config = OpticalMappingConfig(
        configuration_id=original.configuration_id,
        version=original.version,
        materials=tuple(reversed(original.materials)),
    )

    assert reversed_config.materials == original.materials
    assert reversed_config.canonical_json() == original.canonical_json()
    assert reversed_config.digest == original.digest


def test_configuration_digest_is_valid_sha256_and_content_sensitive() -> None:
    config = phase0_provisional_mapping()
    changed = replace(config, configuration_id="changed-configuration")

    assert config.digest.startswith("sha256:")
    assert len(config.digest) == len("sha256:") + 64
    assert changed.digest != config.digest
    payload = json.loads(config.canonical_json())
    assert payload["calibration_status"] == "provisional-uncalibrated"
    assert payload["mixing_rule"] == "linear-volume-fraction-v1"
    assert payload["materials"][1]["material_id"] == 1


def test_configuration_and_properties_are_immutable() -> None:
    config = phase0_provisional_mapping()

    with pytest.raises(FrozenInstanceError):
        config.configuration_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        config.materials[0].ior = 2.0  # type: ignore[misc]
    with pytest.raises(KeyError):
        config.by_id(999)


@pytest.mark.parametrize(
    "override",
    [
        {"material_id": -1},
        {"material_id": 65536},
        {"material_id": True},
        {"name": ""},
        {"sigma_a_rgb_per_m": (1.0, 2.0)},
        {"sigma_a_rgb_per_m": (-1.0, 0.0, 0.0)},
        {"sigma_s_rgb_per_m": (0.0, float("nan"), 0.0)},
        {"sigma_s_rgb_per_m": (0.0, -1.0, 0.0)},
        {"g": -1.01},
        {"g": 1.01},
        {"ior": 0.0},
        {"ior": float("inf")},
    ],
)
def test_invalid_material_properties_are_rejected(override: dict[str, object]) -> None:
    values: dict[str, object] = {
        "material_id": 1,
        "name": "material",
        "sigma_a_rgb_per_m": (1.0, 1.0, 1.0),
        "sigma_s_rgb_per_m": (1.0, 1.0, 1.0),
        "g": 0.0,
        "ior": 1.5,
    }
    values.update(override)

    with pytest.raises((TypeError, ValueError)):
        MaterialOpticalProperties(**values)  # type: ignore[arg-type]


def test_duplicate_material_properties_are_rejected() -> None:
    material = transparent_properties()
    with pytest.raises(ValueError, match="duplicate optical material_id"):
        OpticalMappingConfig(
            "duplicate",
            SchemaVersion(1, 0, 0),
            materials=(material, material),
        )


def test_empty_mapping_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        OpticalMappingConfig("empty", SchemaVersion(1, 0, 0), materials=())


def test_spectral_basis_is_rejected_for_phase0_mapping() -> None:
    with pytest.raises(ValueError, match="linear-srgb-effective-v1"):
        OpticalMappingConfig(
            "spectral",
            SchemaVersion(1, 0, 0),
            materials=(transparent_properties(),),
            optical_basis=OpticalBasis.spectral_wavelengths_nm((450.0, 550.0)),
        )
