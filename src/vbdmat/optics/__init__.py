"""Material-to-optical mapping configuration and reference conversion."""

from .config import (
    CalibrationStatus,
    MaterialOpticalProperties,
    MixingRule,
    OpticalMappingConfig,
    phase0_provisional_mapping,
)
from .document import (
    load_optical_mapping,
    optical_mapping_from_json_dict,
    optical_mapping_to_json_dict,
    write_optical_mapping,
)
from .errors import OpticalMappingError
from .mapping import (
    MAPPING_GENERATOR,
    MAPPING_GENERATOR_VERSION,
    map_material_volume_to_optical,
)

__all__ = [
    "MAPPING_GENERATOR",
    "MAPPING_GENERATOR_VERSION",
    "CalibrationStatus",
    "MaterialOpticalProperties",
    "MixingRule",
    "OpticalMappingConfig",
    "OpticalMappingError",
    "load_optical_mapping",
    "map_material_volume_to_optical",
    "optical_mapping_from_json_dict",
    "optical_mapping_to_json_dict",
    "phase0_provisional_mapping",
    "write_optical_mapping",
]
