"""Print the metadata-only summary of one canonical Zarr asset."""

import argparse
from pathlib import Path

from vdbmat.io import inspect_volume


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    inspection = inspect_volume(args.path)
    geometry = inspection.geometry
    print(f"asset_type: {inspection.asset_type.value}")
    print(f"schema: {inspection.schema_name} {inspection.schema_version}")
    print(f"shape_zyx: {geometry.shape_zyx}")
    print(f"voxel_size_xyz_m: {geometry.voxel_size_xyz_m}")
    print("fields:")
    for field in inspection.arrays:
        print(
            f"  {field.name}: shape={field.shape} dtype={field.dtype} "
            f"dimensions={field.dimensions} chunks={field.chunks} unit={field.unit}"
        )


if __name__ == "__main__":
    main()
