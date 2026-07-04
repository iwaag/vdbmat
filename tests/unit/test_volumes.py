"""Tests for canonical NumPy-backed volume containers."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from vdbmat.core import (
    VOLUME_SCHEMA,
    GridGeometry,
    MaterialDefinition,
    MaterialLabelVolume,
    MaterialMixtureVolume,
    MaterialPalette,
    MaterialRole,
    OpticalBasis,
    OpticalPropertyVolume,
    Provenance,
    SchemaIdentity,
    SchemaVersion,
    VolumeAssetType,
    VolumeValidationError,
)


def geometry() -> GridGeometry:
    return GridGeometry((2, 2, 3), (0.00004, 0.00005, 0.00003))


def palette() -> MaterialPalette:
    return MaterialPalette(
        (
            MaterialDefinition(0, "air", MaterialRole.BACKGROUND),
            MaterialDefinition(7, "resin", MaterialRole.MATERIAL),
        )
    )


def provenance() -> Provenance:
    return Provenance("vdbmat-test", "1.0.0")


def valid_labels() -> npt.NDArray[np.uint16]:
    labels = np.zeros(geometry().shape_zyx, dtype=np.uint16)
    labels[:, :, 1:] = 7
    return labels


def valid_fractions() -> npt.NDArray[np.float32]:
    fractions = np.empty((*geometry().shape_zyx, 2), dtype=np.float32)
    fractions[..., 0] = 0.25
    fractions[..., 1] = 0.75
    return fractions


def valid_optical_arrays() -> dict[str, npt.NDArray[np.float32]]:
    spatial_shape = geometry().shape_zyx
    coefficient_shape = (*spatial_shape, 3)
    return {
        "sigma_a": np.full(coefficient_shape, 1.0, dtype=np.float32),
        "sigma_s": np.full(coefficient_shape, 2.0, dtype=np.float32),
        "g": np.zeros(spatial_shape, dtype=np.float32),
        "ior": np.full(spatial_shape, 1.49, dtype=np.float32),
    }


def optical_volume(
    **overrides: object,
) -> OpticalPropertyVolume:
    values: dict[str, object] = {
        "geometry": geometry(),
        "provenance": provenance(),
        "optical_basis": OpticalBasis.phase0_rgb(),
        **valid_optical_arrays(),
    }
    values.update(overrides)
    return OpticalPropertyVolume(**values)  # type: ignore[arg-type]


def test_valid_label_volume_copies_and_freezes_array() -> None:
    source = valid_labels()[:, :, ::-1]
    assert not source.flags.c_contiguous
    volume = MaterialLabelVolume(geometry(), palette(), provenance(), source)

    assert volume.asset_type is VolumeAssetType.MATERIAL_LABEL
    assert volume.schema == VOLUME_SCHEMA
    assert volume.material_id_dimensions == ("z", "y", "x")
    assert np.array_equal(volume.material_id, source)
    assert volume.material_id is not source
    assert volume.material_id.flags.c_contiguous
    assert not volume.material_id.flags.writeable

    source.fill(0)
    assert np.count_nonzero(volume.material_id == 7) > 0
    with pytest.raises(ValueError, match="read-only"):
        volume.material_id[0, 0, 0] = 7


def test_label_volume_requires_ndarray_without_casting() -> None:
    with pytest.raises(VolumeValidationError) as captured:
        MaterialLabelVolume(
            geometry(), palette(), provenance(), [[[0, 0, 0], [0, 0, 0]]]
        )  # type: ignore[arg-type]

    assert captured.value.field_path == "arrays.material_id"
    assert "NumPy ndarray" in captured.value.message


def test_label_volume_requires_exact_dtype() -> None:
    labels = valid_labels().astype(np.int32)
    with pytest.raises(VolumeValidationError) as captured:
        MaterialLabelVolume(geometry(), palette(), provenance(), labels)  # type: ignore[arg-type]

    assert captured.value.field_path == "arrays.material_id"
    assert "uint16" in captured.value.message
    assert labels.dtype == np.int32


def test_label_volume_requires_geometry_shape() -> None:
    labels = np.zeros((2, 3, 2), dtype=np.uint16)
    with pytest.raises(VolumeValidationError) as captured:
        MaterialLabelVolume(geometry(), palette(), provenance(), labels)

    assert captured.value.field_path == "arrays.material_id"
    assert "(2, 2, 3)" in captured.value.message


def test_label_volume_reports_unknown_material_ids() -> None:
    labels = valid_labels()
    labels[0, 0, 1] = 99
    labels[1, 1, 2] = 99

    with pytest.raises(VolumeValidationError) as captured:
        MaterialLabelVolume(geometry(), palette(), provenance(), labels)

    error = captured.value
    assert error.field_path == "arrays.material_id"
    assert error.invalid_count == 2
    assert error.first_index == (0, 0, 1)
    assert error.first_value == 99
    assert "first_index=(0, 0, 1)" in str(error)


def test_valid_mixture_volume_copies_and_freezes_arrays() -> None:
    fractions = valid_fractions()
    material_ids = np.array([0, 7], dtype=np.uint16)
    volume = MaterialMixtureVolume(
        geometry(), palette(), provenance(), fractions, material_ids
    )

    assert volume.asset_type is VolumeAssetType.MATERIAL_MIXTURE
    assert volume.fractions_dimensions == ("z", "y", "x", "material")
    assert volume.material_ids_dimensions == ("material",)
    assert np.array_equal(volume.fractions, fractions)
    assert np.array_equal(volume.material_ids, material_ids)
    assert volume.fractions is not fractions
    assert volume.material_ids is not material_ids
    assert not volume.fractions.flags.writeable
    assert not volume.material_ids.flags.writeable


@pytest.mark.parametrize(
    ("material_ids", "expected_message"),
    [
        (np.array([0, 7], dtype=np.int32), "dtype must be uint16"),
        (np.array([0], dtype=np.uint16), "shape must be (2,)"),
        (np.array([7, 0], dtype=np.uint16), "match palette order"),
    ],
)
def test_mixture_requires_exact_material_axis(
    material_ids: npt.NDArray[np.generic], expected_message: str
) -> None:
    with pytest.raises(VolumeValidationError) as captured:
        MaterialMixtureVolume(
            geometry(), palette(), provenance(), valid_fractions(), material_ids
        )  # type: ignore[arg-type]

    assert captured.value.field_path == "arrays.material_ids"
    assert expected_message in captured.value.message


def test_mixture_requires_exact_fraction_dtype() -> None:
    fractions = valid_fractions().astype(np.float64)
    with pytest.raises(VolumeValidationError) as captured:
        MaterialMixtureVolume(
            geometry(),
            palette(),
            provenance(),
            fractions,  # type: ignore[arg-type]
            np.array([0, 7], dtype=np.uint16),
        )

    assert captured.value.field_path == "arrays.fractions"
    assert "float32" in captured.value.message
    assert fractions.dtype == np.float64


def test_mixture_requires_fraction_shape() -> None:
    fractions = np.ones((*geometry().shape_zyx, 1), dtype=np.float32)
    with pytest.raises(VolumeValidationError, match=r"arrays\.fractions: shape"):
        MaterialMixtureVolume(
            geometry(),
            palette(),
            provenance(),
            fractions,
            np.array([0, 7], dtype=np.uint16),
        )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_mixture_requires_finite_fractions(invalid: float) -> None:
    fractions = valid_fractions()
    fractions[1, 0, 2, 1] = invalid
    with pytest.raises(VolumeValidationError) as captured:
        MaterialMixtureVolume(
            geometry(),
            palette(),
            provenance(),
            fractions,
            np.array([0, 7], dtype=np.uint16),
        )

    assert captured.value.field_path == "arrays.fractions"
    assert captured.value.first_index == (1, 0, 2, 1)
    assert "finite" in captured.value.message


@pytest.mark.parametrize("invalid", [-0.01, 1.01])
def test_mixture_requires_closed_fraction_range(invalid: float) -> None:
    fractions = valid_fractions()
    fractions[0, 1, 2, 0] = invalid
    before = fractions.copy()

    with pytest.raises(VolumeValidationError) as captured:
        MaterialMixtureVolume(
            geometry(),
            palette(),
            provenance(),
            fractions,
            np.array([0, 7], dtype=np.uint16),
        )

    assert captured.value.field_path == "arrays.fractions"
    assert captured.value.first_index == (0, 1, 2, 0)
    assert np.array_equal(fractions, before)


def test_mixture_requires_normalization_without_repair() -> None:
    fractions = valid_fractions()
    fractions[1, 1, 2] = (0.2, 0.7)
    before = fractions.copy()

    with pytest.raises(VolumeValidationError) as captured:
        MaterialMixtureVolume(
            geometry(),
            palette(),
            provenance(),
            fractions,
            np.array([0, 7], dtype=np.uint16),
        )

    error = captured.value
    assert error.field_path == "arrays.fractions.sum"
    assert error.invalid_count == 1
    assert error.first_index == (1, 1, 2)
    assert float(error.first_value) == pytest.approx(0.9)
    assert np.array_equal(fractions, before)


def test_mixture_accepts_normalization_within_absolute_tolerance() -> None:
    fractions = valid_fractions()
    fractions[0, 0, 0] = (0.2, 0.8000005)

    volume = MaterialMixtureVolume(
        geometry(),
        palette(),
        provenance(),
        fractions,
        np.array([0, 7], dtype=np.uint16),
    )

    assert float(volume.fractions[0, 0, 0].sum()) == pytest.approx(1.0000005)


def test_valid_optical_volume_keeps_separate_read_only_fields() -> None:
    arrays = valid_optical_arrays()
    volume = optical_volume(**arrays)

    assert volume.asset_type is VolumeAssetType.OPTICAL_PROPERTY
    assert volume.coefficient_unit == "m^-1"
    assert volume.dimensionless_unit == "1"
    assert volume.coefficient_dimensions == ("z", "y", "x", "basis")
    assert volume.scalar_dimensions == ("z", "y", "x")
    for field, source in arrays.items():
        stored = getattr(volume, field)
        assert np.array_equal(stored, source)
        assert stored is not source
        assert stored.flags.c_contiguous
        assert not stored.flags.writeable


@pytest.mark.parametrize("field", ["sigma_a", "sigma_s", "g", "ior"])
def test_optical_fields_require_exact_float32_dtype(field: str) -> None:
    invalid = valid_optical_arrays()[field].astype(np.float64)
    with pytest.raises(VolumeValidationError) as captured:
        optical_volume(**{field: invalid})

    assert captured.value.field_path == f"arrays.{field}"
    assert "float32" in captured.value.message


@pytest.mark.parametrize("field", ["sigma_a", "sigma_s", "g", "ior"])
def test_optical_fields_require_geometry_and_basis_shape(field: str) -> None:
    source = valid_optical_arrays()[field]
    invalid = source[:-1]
    with pytest.raises(VolumeValidationError) as captured:
        optical_volume(**{field: invalid})

    assert captured.value.field_path == f"arrays.{field}"
    assert "shape" in captured.value.message


@pytest.mark.parametrize("field", ["sigma_a", "sigma_s", "g", "ior"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_optical_fields_require_finite_values(field: str, invalid: float) -> None:
    source = valid_optical_arrays()[field]
    index = (1, 0, 2, 1) if source.ndim == 4 else (1, 0, 2)
    source[index] = invalid

    with pytest.raises(VolumeValidationError) as captured:
        optical_volume(**{field: source})

    assert captured.value.field_path == f"arrays.{field}"
    assert captured.value.first_index == index
    assert "finite" in captured.value.message


@pytest.mark.parametrize("field", ["sigma_a", "sigma_s"])
def test_optical_coefficients_must_be_non_negative(field: str) -> None:
    source = valid_optical_arrays()[field]
    source[1, 1, 2, 2] = -0.01

    with pytest.raises(VolumeValidationError) as captured:
        optical_volume(**{field: source})

    assert captured.value.field_path == f"arrays.{field}"
    assert captured.value.first_index == (1, 1, 2, 2)
    assert "non-negative" in captured.value.message


@pytest.mark.parametrize("invalid", [-1.0001, 1.0001])
def test_anisotropy_must_be_in_closed_range(invalid: float) -> None:
    g = valid_optical_arrays()["g"]
    g[0, 1, 2] = invalid

    with pytest.raises(VolumeValidationError) as captured:
        optical_volume(g=g)

    assert captured.value.field_path == "arrays.g"
    assert captured.value.first_index == (0, 1, 2)
    assert "[-1, 1]" in captured.value.message


def test_anisotropy_endpoints_are_canonical() -> None:
    g = valid_optical_arrays()["g"]
    g[0, 0, 0] = -1.0
    g[1, 1, 2] = 1.0

    volume = optical_volume(g=g)

    assert volume.g[0, 0, 0] == -1.0
    assert volume.g[1, 1, 2] == 1.0


@pytest.mark.parametrize("invalid", [0.0, -0.01])
def test_ior_must_be_positive(invalid: float) -> None:
    ior = valid_optical_arrays()["ior"]
    ior[1, 0, 1] = invalid

    with pytest.raises(VolumeValidationError) as captured:
        optical_volume(ior=ior)

    assert captured.value.field_path == "arrays.ior"
    assert captured.value.first_index == (1, 0, 1)
    assert "greater than zero" in captured.value.message


def test_phase0_optical_volume_rejects_spectral_basis() -> None:
    spectral = OpticalBasis.spectral_wavelengths_nm((450.0, 550.0, 650.0))

    with pytest.raises(VolumeValidationError) as captured:
        optical_volume(optical_basis=spectral)

    assert captured.value.field_path == "optical_basis.kind"
    assert captured.value.first_value == "spectral"


@pytest.mark.parametrize(
    ("override", "path"),
    [
        ({"geometry": object()}, "geometry"),
        ({"provenance": object()}, "provenance"),
        (
            {
                "schema": SchemaIdentity(
                    "vdbmat.volume", SchemaVersion(major=1, minor=1, patch=0)
                )
            },
            "schema",
        ),
    ],
)
def test_common_metadata_is_validated(override: dict[str, object], path: str) -> None:
    with pytest.raises(VolumeValidationError) as captured:
        optical_volume(**override)

    assert captured.value.field_path == path


def test_volume_errors_report_count_index_and_value() -> None:
    sigma_a = valid_optical_arrays()["sigma_a"]
    sigma_a[0, 0, 1, 2] = -3.0
    sigma_a[1, 1, 2, 0] = -4.0

    with pytest.raises(VolumeValidationError) as captured:
        optical_volume(sigma_a=sigma_a)

    error = captured.value
    assert error.invalid_count == 2
    assert error.first_index == (0, 0, 1, 2)
    assert error.first_value == -3.0
    assert "invalid_count=2" in str(error)


def test_all_volume_builders_are_callable() -> None:
    builders: tuple[Callable[[], object], ...] = (
        lambda: MaterialLabelVolume(
            geometry(), palette(), provenance(), valid_labels()
        ),
        lambda: MaterialMixtureVolume(
            geometry(),
            palette(),
            provenance(),
            valid_fractions(),
            np.array([0, 7], dtype=np.uint16),
        ),
        optical_volume,
    )

    assert all(builder() is not None for builder in builders)
