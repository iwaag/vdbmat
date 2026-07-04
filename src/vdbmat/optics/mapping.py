"""Deterministic reference conversion from materials to optical fields."""

import hashlib
import json
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

from vbdmat.core import (
    MaterialLabelVolume,
    MaterialMixtureVolume,
    OpticalPropertyVolume,
    Provenance,
)

from .config import OpticalMappingConfig
from .errors import OpticalMappingError

MaterialVolume: TypeAlias = MaterialLabelVolume | MaterialMixtureVolume

MAPPING_GENERATOR = "vbdmat.optics.reference-mapping"
MAPPING_GENERATOR_VERSION = "1.0.0"


def map_material_volume_to_optical(
    volume: MaterialVolume, config: OpticalMappingConfig
) -> OpticalPropertyVolume:
    """Map a validated material volume using explicit provisional rules."""
    if not isinstance(volume, (MaterialLabelVolume, MaterialMixtureVolume)):
        raise TypeError("volume must be a canonical material volume")
    if not isinstance(config, OpticalMappingConfig):
        raise TypeError("config must be an OpticalMappingConfig")

    _require_palette_coverage(volume, config)
    if isinstance(volume, MaterialLabelVolume):
        sigma_a, sigma_s, g, ior = _map_labels(volume, config)
    else:
        sigma_a, sigma_s, g, ior = _map_mixture(volume, config)

    return OpticalPropertyVolume(
        geometry=volume.geometry,
        provenance=_output_provenance(volume.provenance, config),
        optical_basis=config.optical_basis,
        sigma_a=sigma_a,
        sigma_s=sigma_s,
        g=g,
        ior=ior,
        schema=volume.schema,
    )


def _require_palette_coverage(
    volume: MaterialVolume, config: OpticalMappingConfig
) -> None:
    configured = set(config.material_ids)
    missing = tuple(
        material_id
        for material_id in volume.palette.material_ids
        if material_id not in configured
    )
    if missing:
        raise OpticalMappingError(
            "palette.material_ids",
            f"mapping config is missing declared material IDs {missing}",
        )
    # ADR-009 D4: the simulation-side contract is material_id plus name. A palette
    # and mapping that disagree on the name of a shared ID are describing different
    # materials; failing here prevents silently applying the wrong coefficients.
    mismatched = tuple(
        (item.material_id, item.name, config.by_id(item.material_id).name)
        for item in volume.palette
        if item.name != config.by_id(item.material_id).name
    )
    if mismatched:
        details = "; ".join(
            f"id {material_id}: palette {palette_name!r} vs mapping {mapping_name!r}"
            for material_id, palette_name, mapping_name in mismatched
        )
        raise OpticalMappingError(
            "palette.materials",
            f"material names disagree with the mapping for shared IDs ({details})",
        )


def _map_labels(
    volume: MaterialLabelVolume, config: OpticalMappingConfig
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    coefficient_shape = (*volume.geometry.shape_zyx, config.optical_basis.size)
    sigma_a = np.empty(coefficient_shape, dtype=np.float32)
    sigma_s = np.empty(coefficient_shape, dtype=np.float32)
    g = np.empty(volume.geometry.shape_zyx, dtype=np.float32)
    ior = np.empty(volume.geometry.shape_zyx, dtype=np.float32)

    for material_id in volume.palette.material_ids:
        properties = config.by_id(material_id)
        selected = volume.material_id == material_id
        sigma_a[selected] = properties.sigma_a_rgb_per_m
        sigma_s[selected] = properties.sigma_s_rgb_per_m
        g[selected] = properties.g
        ior[selected] = properties.ior
    return (sigma_a, sigma_s, g, ior)


def _map_mixture(
    volume: MaterialMixtureVolume, config: OpticalMappingConfig
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    properties = tuple(config.by_id(int(item)) for item in volume.material_ids)
    sigma_a_matrix = np.asarray(
        [item.sigma_a_rgb_per_m for item in properties], dtype=np.float64
    )
    sigma_s_matrix = np.asarray(
        [item.sigma_s_rgb_per_m for item in properties], dtype=np.float64
    )
    g_vector = np.asarray([item.g for item in properties], dtype=np.float64)
    ior_vector = np.asarray([item.ior for item in properties], dtype=np.float64)
    fractions = volume.fractions.astype(np.float64, copy=False)

    sigma_a = _as_float32(fractions @ sigma_a_matrix)
    sigma_s = _as_float32(fractions @ sigma_s_matrix)
    g = _as_float32(fractions @ g_vector)
    ior = _as_float32(fractions @ ior_vector)
    return (sigma_a, sigma_s, g, ior)


def _as_float32(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
    return cast(npt.NDArray[np.float32], value.astype(np.float32, copy=False))


def _output_provenance(source: Provenance, config: OpticalMappingConfig) -> Provenance:
    source_fingerprint = _provenance_fingerprint(source)
    sources = (
        *source.sources,
        f"source-generator:{source.generator}@{source.generator_version}",
        f"source-provenance-{source_fingerprint}",
        f"mapping-config:{config.configuration_id}@{config.version}",
    )
    return Provenance(
        generator=MAPPING_GENERATOR,
        generator_version=MAPPING_GENERATOR_VERSION,
        configuration_digest=config.digest,
        sources=sources,
        notes=(
            "Provisional uncalibrated optical coefficients; "
            f"mixtures use {config.mixing_rule.value}."
        ),
    )


def _provenance_fingerprint(provenance: Provenance) -> str:
    payload = {
        "configuration_digest": provenance.configuration_digest,
        "created_utc": (
            provenance.created_utc.isoformat()
            if provenance.created_utc is not None
            else None
        ),
        "generator": provenance.generator,
        "generator_version": provenance.generator_version,
        "notes": provenance.notes,
        "sources": provenance.sources,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
