"""Tests for optical, schema, and provenance metadata."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from vdbmat.core import (
    VOLUME_SCHEMA,
    OpticalBasis,
    OpticalBasisKind,
    Provenance,
    SchemaIdentity,
    SchemaVersion,
)


def test_phase0_rgb_basis_is_exact_and_immutable() -> None:
    basis = OpticalBasis.phase0_rgb()

    assert basis.kind is OpticalBasisKind.RGB
    assert basis.identifier == "linear-srgb-effective-v1"
    assert basis.coordinates == ("R", "G", "B")
    assert basis.reference_white == "D65"
    assert basis.observer == "CIE-1931-2deg"
    assert basis.transfer == "linear"
    assert basis.size == 3


def test_reserved_spectral_metadata_has_ordered_numeric_coordinates() -> None:
    basis = OpticalBasis.spectral_wavelengths_nm([450, 550.0, 650])

    assert basis.kind is OpticalBasisKind.SPECTRAL
    assert basis.identifier == "wavelength-nm"
    assert basis.coordinates == (450.0, 550.0, 650.0)
    assert basis.size == 3


@pytest.mark.parametrize(
    "coordinates",
    [
        (),
        (550.0, 450.0),
        (450.0, 450.0),
        (450.0, float("nan")),
        ("450", "550"),
    ],
)
def test_invalid_spectral_coordinates_are_rejected(coordinates: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        OpticalBasis.spectral_wavelengths_nm(coordinates)  # type: ignore[arg-type]


def test_rgb_metadata_cannot_be_reinterpreted() -> None:
    with pytest.raises(ValueError, match="coordinates"):
        OpticalBasis(
            kind=OpticalBasisKind.RGB,
            identifier="linear-srgb-effective-v1",
            coordinates=("B", "G", "R"),
            reference_white="D65",
            observer="CIE-1931-2deg",
            transfer="linear",
        )


def test_schema_version_is_strict_and_major_compatible() -> None:
    version = SchemaVersion.parse("1.2.3")

    assert version == SchemaVersion(1, 2, 3)
    assert str(version) == "1.2.3"
    assert version.has_compatible_major(SchemaVersion(1, 9, 0))
    assert not version.has_compatible_major(SchemaVersion(2, 0, 0))
    assert SchemaIdentity("vdbmat.volume", SchemaVersion(1, 0, 0)) == VOLUME_SCHEMA


@pytest.mark.parametrize("value", ["1", "1.2", "01.2.3", "1.2.3-dev", "a.b.c"])
def test_invalid_schema_versions_are_rejected(value: str) -> None:
    with pytest.raises(ValueError, match=r"MAJOR\.MINOR\.PATCH"):
        SchemaVersion.parse(value)


def test_provenance_normalizes_sources_and_accepts_utc() -> None:
    created = datetime(2026, 6, 28, 12, 0, tzinfo=UTC)
    provenance = Provenance(
        generator="vdbmat",
        generator_version="0.1.0",
        created_utc=created,
        configuration_digest="sha256:" + "a" * 64,
        sources=["fixture:anisotropic"],  # type: ignore[arg-type]
    )

    assert provenance.created_utc == created
    assert provenance.sources == ("fixture:anisotropic",)


@pytest.mark.parametrize(
    "created",
    [
        datetime(2026, 6, 28, 12, 0),
        datetime(2026, 6, 28, 12, 0, tzinfo=timezone(timedelta(hours=9))),
    ],
)
def test_provenance_requires_utc(created: datetime) -> None:
    with pytest.raises(ValueError, match=r"UTC|timezone-aware"):
        Provenance("vdbmat", "0.1.0", created_utc=created)


def test_invalid_configuration_digest_is_rejected() -> None:
    with pytest.raises(ValueError, match="sha256"):
        Provenance("vdbmat", "0.1.0", configuration_digest="ABC")


def test_provenance_rejects_one_string_as_sources_sequence() -> None:
    with pytest.raises(TypeError, match="sequence of strings"):
        Provenance("vdbmat", "0.1.0", sources="fixture")  # type: ignore[arg-type]
