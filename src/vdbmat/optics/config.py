"""Immutable, digestible optical mapping configuration."""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral, Real
from types import MappingProxyType
from typing import TypeAlias, cast

from vbdmat.core import OpticalBasis, SchemaVersion

RGBTriple: TypeAlias = tuple[float, float, float]


class MixingRule(StrEnum):
    """Supported material-mixture rules."""

    LINEAR_VOLUME_FRACTION_V1 = "linear-volume-fraction-v1"


class CalibrationStatus(StrEnum):
    """Evidence status attached to an optical mapping configuration."""

    PROVISIONAL_UNCALIBRATED = "provisional-uncalibrated"


@dataclass(frozen=True, slots=True)
class MaterialOpticalProperties:
    """Provisional optical values associated with one material ID."""

    material_id: int
    name: str
    sigma_a_rgb_per_m: RGBTriple
    sigma_s_rgb_per_m: RGBTriple
    g: float
    ior: float

    def __post_init__(self) -> None:
        if isinstance(self.material_id, bool) or not isinstance(
            self.material_id, Integral
        ):
            raise TypeError("material_id must be an integer")
        material_id = int(self.material_id)
        if not 0 <= material_id <= 65535:
            raise ValueError("material_id must be in [0, 65535]")
        object.__setattr__(self, "material_id", material_id)

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("material optical name must be a non-empty string")

        sigma_a = _normalize_rgb(self.sigma_a_rgb_per_m, field="sigma_a_rgb_per_m")
        sigma_s = _normalize_rgb(self.sigma_s_rgb_per_m, field="sigma_s_rgb_per_m")
        if any(value < 0.0 for value in sigma_a):
            raise ValueError("sigma_a_rgb_per_m values must be non-negative")
        if any(value < 0.0 for value in sigma_s):
            raise ValueError("sigma_s_rgb_per_m values must be non-negative")
        object.__setattr__(self, "sigma_a_rgb_per_m", sigma_a)
        object.__setattr__(self, "sigma_s_rgb_per_m", sigma_s)

        g = _normalize_scalar(self.g, field="g")
        if not -1.0 <= g <= 1.0:
            raise ValueError("g must lie in [-1, 1]")
        object.__setattr__(self, "g", g)

        ior = _normalize_scalar(self.ior, field="ior")
        if ior <= 0.0:
            raise ValueError("ior must be greater than zero")
        object.__setattr__(self, "ior", ior)


@dataclass(frozen=True, slots=True)
class OpticalMappingConfig:
    """A versioned set of material properties and one explicit mixing rule."""

    configuration_id: str
    version: SchemaVersion
    materials: tuple[MaterialOpticalProperties, ...]
    optical_basis: OpticalBasis = field(default_factory=OpticalBasis.phase0_rgb)
    mixing_rule: MixingRule = MixingRule.LINEAR_VOLUME_FRACTION_V1
    calibration_status: CalibrationStatus = CalibrationStatus.PROVISIONAL_UNCALIBRATED
    _by_id: Mapping[int, MaterialOpticalProperties] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.configuration_id, str)
            or not self.configuration_id.strip()
        ):
            raise ValueError("configuration_id must be a non-empty string")
        if not isinstance(self.version, SchemaVersion):
            raise TypeError("mapping version must be a SchemaVersion")
        if self.optical_basis != OpticalBasis.phase0_rgb():
            raise ValueError("Phase 0 mapping requires linear-srgb-effective-v1")
        try:
            mixing_rule = MixingRule(self.mixing_rule)
        except ValueError as error:
            raise ValueError(
                f"unsupported mixing rule: {self.mixing_rule!r}"
            ) from error
        object.__setattr__(self, "mixing_rule", mixing_rule)
        try:
            status = CalibrationStatus(self.calibration_status)
        except ValueError as error:
            raise ValueError(
                f"unsupported calibration status: {self.calibration_status!r}"
            ) from error
        object.__setattr__(self, "calibration_status", status)

        materials = tuple(self.materials)
        if not materials:
            raise ValueError("mapping materials must not be empty")
        if any(not isinstance(item, MaterialOpticalProperties) for item in materials):
            raise TypeError(
                "mapping materials must contain MaterialOpticalProperties objects"
            )
        ordered = tuple(sorted(materials, key=lambda item: item.material_id))
        by_id: dict[int, MaterialOpticalProperties] = {}
        for item in ordered:
            if item.material_id in by_id:
                raise ValueError(f"duplicate optical material_id: {item.material_id}")
            by_id[item.material_id] = item
        object.__setattr__(self, "materials", ordered)
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))

    @property
    def material_ids(self) -> tuple[int, ...]:
        """Return configured IDs in canonical sorted order."""
        return tuple(item.material_id for item in self.materials)

    def by_id(self, material_id: int) -> MaterialOpticalProperties:
        """Return configured properties for one material ID."""
        return self._by_id[material_id]

    def canonical_json(self) -> str:
        """Return stable JSON used to identify this exact configuration."""
        payload = {
            "calibration_status": self.calibration_status.value,
            "configuration_id": self.configuration_id,
            "materials": [
                {
                    "g": item.g,
                    "ior": item.ior,
                    "material_id": item.material_id,
                    "name": item.name,
                    "sigma_a_rgb_per_m": item.sigma_a_rgb_per_m,
                    "sigma_s_rgb_per_m": item.sigma_s_rgb_per_m,
                }
                for item in self.materials
            ],
            "mixing_rule": self.mixing_rule.value,
            "optical_basis": {
                "coordinates": self.optical_basis.coordinates,
                "identifier": self.optical_basis.identifier,
                "kind": self.optical_basis.kind.value,
                "observer": self.optical_basis.observer,
                "reference_white": self.optical_basis.reference_white,
                "transfer": self.optical_basis.transfer,
            },
            "version": str(self.version),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        """Return the SHA-256 identity of the canonical configuration."""
        encoded = self.canonical_json().encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def phase0_provisional_mapping() -> OpticalMappingConfig:
    """Return the explicit uncalibrated configuration used by Phase 0 proofs."""
    return OpticalMappingConfig(
        configuration_id="phase0-provisional-materials-v1",
        version=SchemaVersion(1, 0, 0),
        materials=(
            MaterialOpticalProperties(
                0,
                "air",
                sigma_a_rgb_per_m=(0.0, 0.0, 0.0),
                sigma_s_rgb_per_m=(0.0, 0.0, 0.0),
                g=0.0,
                ior=1.0,
            ),
            MaterialOpticalProperties(
                1,
                "transparent-resin",
                sigma_a_rgb_per_m=(2.0, 1.0, 0.5),
                sigma_s_rgb_per_m=(0.0, 0.0, 0.0),
                g=0.0,
                ior=1.48,
            ),
            MaterialOpticalProperties(
                2,
                "white-resin",
                sigma_a_rgb_per_m=(1.0, 1.0, 1.0),
                sigma_s_rgb_per_m=(1000.0, 1000.0, 1000.0),
                g=0.2,
                ior=1.52,
            ),
            MaterialOpticalProperties(
                3,
                "black-opaque-resin",
                sigma_a_rgb_per_m=(4000.0, 5000.0, 6000.0),
                sigma_s_rgb_per_m=(100.0, 100.0, 100.0),
                g=0.1,
                ior=1.52,
            ),
            MaterialOpticalProperties(
                10,
                "axis-x-diagnostic",
                sigma_a_rgb_per_m=(0.0, 100.0, 100.0),
                sigma_s_rgb_per_m=(0.0, 0.0, 0.0),
                g=0.0,
                ior=1.0,
            ),
            MaterialOpticalProperties(
                20,
                "axis-y-diagnostic",
                sigma_a_rgb_per_m=(100.0, 0.0, 100.0),
                sigma_s_rgb_per_m=(0.0, 0.0, 0.0),
                g=0.0,
                ior=1.0,
            ),
            MaterialOpticalProperties(
                30,
                "axis-z-diagnostic",
                sigma_a_rgb_per_m=(100.0, 100.0, 0.0),
                sigma_s_rgb_per_m=(0.0, 0.0, 0.0),
                g=0.0,
                ior=1.0,
            ),
        ),
    )


def _normalize_rgb(value: Sequence[float], *, field: str) -> RGBTriple:
    if len(value) != 3:
        raise ValueError(f"{field} must contain exactly 3 values")
    normalized = tuple(
        _normalize_scalar(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    return cast(RGBTriple, normalized)


def _normalize_scalar(value: float, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be finite")
    return normalized
