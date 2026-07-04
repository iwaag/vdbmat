"""Explicit optical-basis metadata for transport coefficients."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from numbers import Real
from typing import TypeAlias

BasisCoordinate: TypeAlias = str | float


class OpticalBasisKind(StrEnum):
    """Supported metadata forms for an optical coefficient basis."""

    RGB = "rgb"
    SPECTRAL = "spectral"


@dataclass(frozen=True, slots=True)
class OpticalBasis:
    """Immutable metadata that defines the last optical-array dimension."""

    kind: OpticalBasisKind
    identifier: str
    coordinates: tuple[BasisCoordinate, ...]
    reference_white: str | None = None
    observer: str | None = None
    transfer: str | None = None

    def __post_init__(self) -> None:
        try:
            kind = OpticalBasisKind(self.kind)
        except ValueError as error:
            raise ValueError(
                f"unsupported optical basis kind: {self.kind!r}"
            ) from error
        object.__setattr__(self, "kind", kind)

        if not isinstance(self.identifier, str) or not self.identifier.strip():
            raise ValueError("optical basis identifier must be a non-empty string")

        coordinates = tuple(self.coordinates)
        if not coordinates:
            raise ValueError("optical basis coordinates must not be empty")
        object.__setattr__(self, "coordinates", coordinates)

        if kind is OpticalBasisKind.RGB:
            self._validate_phase0_rgb()
        else:
            self._validate_spectral_metadata()

    @classmethod
    def phase0_rgb(cls) -> "OpticalBasis":
        """Return the sole RGB transport basis defined by schema 1.0."""
        return cls(
            kind=OpticalBasisKind.RGB,
            identifier="linear-srgb-effective-v1",
            coordinates=("R", "G", "B"),
            reference_white="D65",
            observer="CIE-1931-2deg",
            transfer="linear",
        )

    @classmethod
    def spectral_wavelengths_nm(cls, coordinates_nm: Sequence[float]) -> "OpticalBasis":
        """Create reserved spectral metadata without enabling spectral processing."""
        return cls(
            kind=OpticalBasisKind.SPECTRAL,
            identifier="wavelength-nm",
            coordinates=tuple(coordinates_nm),
        )

    @property
    def size(self) -> int:
        """Return the required extent of an optical array's basis dimension."""
        return len(self.coordinates)

    def _validate_phase0_rgb(self) -> None:
        expected = {
            "identifier": "linear-srgb-effective-v1",
            "coordinates": ("R", "G", "B"),
            "reference_white": "D65",
            "observer": "CIE-1931-2deg",
            "transfer": "linear",
        }
        for field, expected_value in expected.items():
            if getattr(self, field) != expected_value:
                raise ValueError(
                    f"Phase 0 RGB basis requires {field}={expected_value!r}"
                )

    def _validate_spectral_metadata(self) -> None:
        if self.identifier != "wavelength-nm":
            raise ValueError("spectral basis identifier must be 'wavelength-nm'")
        if any(
            item is not None
            for item in (self.reference_white, self.observer, self.transfer)
        ):
            raise ValueError(
                "spectral wavelength metadata must not declare RGB-only fields"
            )

        wavelengths: list[float] = []
        for index, coordinate in enumerate(self.coordinates):
            if isinstance(coordinate, bool) or not isinstance(coordinate, Real):
                raise TypeError(f"spectral coordinate {index} must be a real number")
            wavelength = float(coordinate)
            if not math.isfinite(wavelength):
                raise ValueError(f"spectral coordinate {index} must be finite")
            wavelengths.append(wavelength)
        if any(left >= right for left, right in pairwise(wavelengths)):
            raise ValueError(
                "spectral wavelength coordinates must be strictly increasing"
            )
        object.__setattr__(self, "coordinates", tuple(wavelengths))
