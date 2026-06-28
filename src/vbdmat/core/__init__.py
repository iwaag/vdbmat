"""Canonical data definitions shared by all VBDMAT modules."""

from .errors import VolumeValidationError
from .geometry import GridGeometry
from .materials import MaterialDefinition, MaterialPalette, MaterialRole
from .metadata import VOLUME_SCHEMA, Provenance, SchemaIdentity, SchemaVersion
from .optical_basis import OpticalBasis, OpticalBasisKind
from .volumes import (
    MIXTURE_NORMALIZATION_TOLERANCE,
    MaterialLabelVolume,
    MaterialMixtureVolume,
    OpticalPropertyVolume,
    VolumeAssetType,
)

__all__ = [
    "MIXTURE_NORMALIZATION_TOLERANCE",
    "VOLUME_SCHEMA",
    "GridGeometry",
    "MaterialDefinition",
    "MaterialLabelVolume",
    "MaterialMixtureVolume",
    "MaterialPalette",
    "MaterialRole",
    "OpticalBasis",
    "OpticalBasisKind",
    "OpticalPropertyVolume",
    "Provenance",
    "SchemaIdentity",
    "SchemaVersion",
    "VolumeAssetType",
    "VolumeValidationError",
]
