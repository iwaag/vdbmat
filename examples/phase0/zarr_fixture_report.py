"""Write every fixture and report Zarr size and optical partial-read behavior."""

import tempfile
from pathlib import Path

import numpy as np

from vdbmat.fixtures import all_synthetic_fixtures
from vdbmat.io import read_optical_region, write_volume
from vdbmat.optics import (
    map_material_volume_to_optical,
    phase0_provisional_mapping,
)


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    config = phase0_provisional_mapping()
    with tempfile.TemporaryDirectory(prefix="vdbmat-zarr-report-") as directory:
        root = Path(directory)
        print("fixture,canonical_bytes,optical_bytes,region_zyx,partial_exact")
        for fixture in all_synthetic_fixtures():
            canonical_path = root / f"{fixture.manifest.name}.zarr"
            optical_path = root / f"{fixture.manifest.name}-optical.zarr"
            optical = map_material_volume_to_optical(fixture.volume, config)
            write_volume(canonical_path, fixture.volume)
            write_volume(optical_path, optical)
            nz, ny, nx = optical.geometry.shape_zyx
            region = (
                slice(0, max(1, nz // 2)),
                slice(0, max(1, ny // 2)),
                slice(0, max(1, nx // 2)),
            )
            partial = read_optical_region(optical_path, region)
            exact = all(
                np.array_equal(getattr(partial, name), getattr(optical, name)[region])
                for name in ("sigma_a", "sigma_s", "g", "ior")
            )
            print(
                f"{fixture.manifest.name},{_directory_size(canonical_path)},"
                f"{_directory_size(optical_path)},"
                f"{partial.geometry.shape_zyx},{str(exact).lower()}"
            )


if __name__ == "__main__":
    main()
