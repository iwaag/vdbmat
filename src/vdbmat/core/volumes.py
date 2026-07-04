"""Validated NumPy-backed canonical volume containers."""

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, cast

import numpy as np
import numpy.typing as npt

from .errors import VolumeValidationError
from .geometry import GridGeometry
from .materials import MaterialPalette
from .metadata import VOLUME_SCHEMA, Provenance, SchemaIdentity
from .optical_basis import OpticalBasis, OpticalBasisKind
from .validation import (
    Float32Array,
    UInt16Array,
    raise_for_mask,
    readonly_copy,
    require_float32_array,
    require_uint16_array,
)

MIXTURE_NORMALIZATION_TOLERANCE = 1e-6


class VolumeAssetType(StrEnum):
    """Logical volume asset types defined by schema 1.0."""

    MATERIAL_LABEL = "material-label"
    MATERIAL_MIXTURE = "material-mixture"
    OPTICAL_PROPERTY = "optical-property"


@dataclass(frozen=True, slots=True, eq=False)
class MaterialLabelVolume:
    """One declared material identifier per canonical voxel cell."""

    geometry: GridGeometry
    palette: MaterialPalette
    provenance: Provenance
    material_id: UInt16Array
    schema: SchemaIdentity = VOLUME_SCHEMA

    asset_type: ClassVar[VolumeAssetType] = VolumeAssetType.MATERIAL_LABEL
    material_id_dimensions: ClassVar[tuple[str, ...]] = ("z", "y", "x")

    def __post_init__(self) -> None:
        _validate_common_metadata(self.geometry, self.provenance, self.schema)
        if not isinstance(self.palette, MaterialPalette):
            raise VolumeValidationError("palette", "must be a MaterialPalette")

        material_id = require_uint16_array(
            self.material_id,
            field_path="arrays.material_id",
            shape=self.geometry.shape_zyx,
        )
        invalid = ~np.isin(material_id, self.palette.material_ids)
        raise_for_mask(
            "arrays.material_id",
            material_id,
            invalid,
            "values must reference declared palette material IDs",
        )
        object.__setattr__(self, "material_id", readonly_copy(material_id))


@dataclass(frozen=True, slots=True, eq=False)
class MaterialMixtureVolume:
    """Normalized material volume fractions for every canonical voxel cell."""

    geometry: GridGeometry
    palette: MaterialPalette
    provenance: Provenance
    fractions: Float32Array
    material_ids: UInt16Array
    schema: SchemaIdentity = VOLUME_SCHEMA

    asset_type: ClassVar[VolumeAssetType] = VolumeAssetType.MATERIAL_MIXTURE
    fractions_dimensions: ClassVar[tuple[str, ...]] = (
        "z",
        "y",
        "x",
        "material",
    )
    material_ids_dimensions: ClassVar[tuple[str, ...]] = ("material",)

    def __post_init__(self) -> None:
        _validate_common_metadata(self.geometry, self.provenance, self.schema)
        if not isinstance(self.palette, MaterialPalette):
            raise VolumeValidationError("palette", "must be a MaterialPalette")

        material_count = len(self.palette)
        material_ids = require_uint16_array(
            self.material_ids,
            field_path="arrays.material_ids",
            shape=(material_count,),
        )
        expected_ids = np.asarray(self.palette.material_ids, dtype=np.uint16)
        raise_for_mask(
            "arrays.material_ids",
            material_ids,
            material_ids != expected_ids,
            "values must exactly match palette order",
        )

        fractions = require_float32_array(
            self.fractions,
            field_path="arrays.fractions",
            shape=(*self.geometry.shape_zyx, material_count),
        )
        raise_for_mask(
            "arrays.fractions",
            fractions,
            (fractions < 0.0) | (fractions > 1.0),
            "values must lie in [0, 1]",
        )

        fraction_sums = cast(
            npt.NDArray[np.float64],
            np.sum(fractions, axis=-1, dtype=np.float64),
        )
        invalid_sums = ~np.isclose(
            fraction_sums,
            1.0,
            rtol=0.0,
            atol=MIXTURE_NORMALIZATION_TOLERANCE,
        )
        raise_for_mask(
            "arrays.fractions.sum",
            fraction_sums,
            invalid_sums,
            f"material fractions must sum to 1 within absolute tolerance "
            f"{MIXTURE_NORMALIZATION_TOLERANCE}",
        )

        object.__setattr__(self, "fractions", readonly_copy(fractions))
        object.__setattr__(self, "material_ids", readonly_copy(material_ids))


@dataclass(frozen=True, slots=True, eq=False)
class OpticalPropertyVolume:
    """Effective optical transport fields on a canonical voxel grid."""

    geometry: GridGeometry
    provenance: Provenance
    optical_basis: OpticalBasis
    sigma_a: Float32Array
    sigma_s: Float32Array
    g: Float32Array
    ior: Float32Array
    schema: SchemaIdentity = VOLUME_SCHEMA

    asset_type: ClassVar[VolumeAssetType] = VolumeAssetType.OPTICAL_PROPERTY
    coefficient_unit: ClassVar[str] = "m^-1"
    dimensionless_unit: ClassVar[str] = "1"
    coefficient_dimensions: ClassVar[tuple[str, ...]] = (
        "z",
        "y",
        "x",
        "basis",
    )
    scalar_dimensions: ClassVar[tuple[str, ...]] = ("z", "y", "x")

    def __post_init__(self) -> None:
        _validate_common_metadata(self.geometry, self.provenance, self.schema)
        if not isinstance(self.optical_basis, OpticalBasis):
            raise VolumeValidationError("optical_basis", "must be an OpticalBasis")
        if self.optical_basis.kind is not OpticalBasisKind.RGB:
            raise VolumeValidationError(
                "optical_basis.kind",
                "Phase 0 optical volumes require the declared RGB basis",
                first_value=self.optical_basis.kind.value,
            )
        if self.optical_basis != OpticalBasis.phase0_rgb():
            raise VolumeValidationError(
                "optical_basis",
                "must exactly match linear-srgb-effective-v1",
            )

        coefficient_shape = (*self.geometry.shape_zyx, self.optical_basis.size)
        sigma_a = require_float32_array(
            self.sigma_a,
            field_path="arrays.sigma_a",
            shape=coefficient_shape,
        )
        sigma_s = require_float32_array(
            self.sigma_s,
            field_path="arrays.sigma_s",
            shape=coefficient_shape,
        )
        g = require_float32_array(
            self.g,
            field_path="arrays.g",
            shape=self.geometry.shape_zyx,
        )
        ior = require_float32_array(
            self.ior,
            field_path="arrays.ior",
            shape=self.geometry.shape_zyx,
        )

        raise_for_mask(
            "arrays.sigma_a",
            sigma_a,
            sigma_a < 0.0,
            "absorption coefficients must be non-negative",
        )
        raise_for_mask(
            "arrays.sigma_s",
            sigma_s,
            sigma_s < 0.0,
            "scattering coefficients must be non-negative",
        )
        raise_for_mask(
            "arrays.g",
            g,
            (g < -1.0) | (g > 1.0),
            "anisotropy values must lie in [-1, 1]",
        )
        raise_for_mask(
            "arrays.ior",
            ior,
            ior <= 0.0,
            "refractive index values must be greater than zero",
        )

        object.__setattr__(self, "sigma_a", readonly_copy(sigma_a))
        object.__setattr__(self, "sigma_s", readonly_copy(sigma_s))
        object.__setattr__(self, "g", readonly_copy(g))
        object.__setattr__(self, "ior", readonly_copy(ior))


def _validate_common_metadata(
    geometry: object, provenance: object, schema: object
) -> None:
    if not isinstance(geometry, GridGeometry):
        raise VolumeValidationError("geometry", "must be a GridGeometry")
    if not isinstance(provenance, Provenance):
        raise VolumeValidationError("provenance", "must be Provenance")
    if not isinstance(schema, SchemaIdentity):
        raise VolumeValidationError("schema", "must be a SchemaIdentity")
    if schema != VOLUME_SCHEMA:
        raise VolumeValidationError(
            "schema",
            f"must be {VOLUME_SCHEMA.name} version {VOLUME_SCHEMA.version}",
            first_value=f"{schema.name} {schema.version}",
        )
