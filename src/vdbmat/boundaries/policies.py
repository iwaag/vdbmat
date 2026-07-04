"""Explicit target-consumer policies for canonical and derived IOR semantics."""

from dataclasses import dataclass
from enum import StrEnum


class ConsumerKind(StrEnum):
    """Phase 0 rendering consumers."""

    MITSUBA_3 = "mitsuba-3"
    BLENDER_CYCLES = "blender-cycles"


class CapabilityStatus(StrEnum):
    """Required exporter diagnostic dispositions from ADR-003."""

    REPRESENTED = "represented"
    TRANSFORMED = "transformed"
    APPROXIMATED = "approximated"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class IORConsumerPolicy:
    """Testable contract for handling canonical IOR and derived interfaces."""

    consumer: ConsumerKind
    spatial_ior: CapabilityStatus
    derived_interfaces: CapabilityStatus
    interpolation: str
    exterior_boundary: str
    internal_boundary: str


MITSUBA_3_IOR_POLICY = IORConsumerPolicy(
    consumer=ConsumerKind.MITSUBA_3,
    spatial_ior=CapabilityStatus.UNSUPPORTED,
    derived_interfaces=CapabilityStatus.TRANSFORMED,
    interpolation="nearest-cell coefficients; never interpolate IOR",
    exterior_boundary=(
        "oriented faces map ambient/cell IOR to dielectric ext_ior/int_ior"
    ),
    internal_boundary=(
        "oriented faces map negative/positive IOR to dielectric ext_ior/int_ior; "
        "the exporter must build compatible closed region meshes"
    ),
)

CYCLES_IOR_POLICY = IORConsumerPolicy(
    consumer=ConsumerKind.BLENDER_CYCLES,
    spatial_ior=CapabilityStatus.UNSUPPORTED,
    derived_interfaces=CapabilityStatus.APPROXIMATED,
    interpolation="nearest-cell coefficients; never interpolate IOR",
    exterior_boundary=(
        "closed derived region surface uses Glass or transmissive Principled IOR"
    ),
    internal_boundary=(
        "closed derived region surfaces approximate adjacent-medium transitions; "
        "OpenVDB volume grids alone cannot represent bulk refraction"
    ),
)


def ior_policy(consumer: ConsumerKind) -> IORConsumerPolicy:
    """Return the mandatory IOR policy for a Phase 0 consumer."""
    try:
        normalized = ConsumerKind(consumer)
    except ValueError as error:
        raise ValueError(f"unsupported consumer: {consumer!r}") from error
    if normalized is ConsumerKind.MITSUBA_3:
        return MITSUBA_3_IOR_POLICY
    return CYCLES_IOR_POLICY
