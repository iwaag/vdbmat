"""Persistence adapters for canonical volume assets."""

from .errors import MeshReadError, VolumeIOError, VoxelManifestError
from .mesh import RawMesh, read_stl, read_stl_bytes
from .voxel_manifest import (
    ManifestInspection,
    inspect_material_label_manifest,
    read_material_label_manifest,
)
from .zarr import (
    ArrayInspection,
    CanonicalVolume,
    RegionZYX,
    VolumeInspection,
    inspect_volume,
    read_optical_region,
    read_volume,
    write_volume,
)

__all__ = [
    "ArrayInspection",
    "CanonicalVolume",
    "ManifestInspection",
    "MeshReadError",
    "RawMesh",
    "RegionZYX",
    "VolumeIOError",
    "VolumeInspection",
    "VoxelManifestError",
    "inspect_material_label_manifest",
    "inspect_volume",
    "read_material_label_manifest",
    "read_optical_region",
    "read_stl",
    "read_stl_bytes",
    "read_volume",
    "write_volume",
]
