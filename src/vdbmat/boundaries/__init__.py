"""Renderer-neutral derived boundary assets and consumer policies."""

from .interfaces import (
    DEFAULT_BOUNDARY_CONFIG,
    BoundaryAxis,
    BoundaryDerivationConfig,
    DerivedInterfaceSet,
    InterfaceFace,
    derive_ior_interfaces,
)
from .policies import (
    CYCLES_IOR_POLICY,
    MITSUBA_3_IOR_POLICY,
    CapabilityStatus,
    ConsumerKind,
    IORConsumerPolicy,
    ior_policy,
)

__all__ = [
    "CYCLES_IOR_POLICY",
    "DEFAULT_BOUNDARY_CONFIG",
    "MITSUBA_3_IOR_POLICY",
    "BoundaryAxis",
    "BoundaryDerivationConfig",
    "CapabilityStatus",
    "ConsumerKind",
    "DerivedInterfaceSet",
    "IORConsumerPolicy",
    "InterfaceFace",
    "derive_ior_interfaces",
    "ior_policy",
]
