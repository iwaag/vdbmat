"""Print sharp IOR-interface summaries for every mapped synthetic fixture."""

from collections import Counter

from vdbmat.boundaries import derive_ior_interfaces
from vdbmat.fixtures import all_synthetic_fixtures
from vdbmat.optics import (
    map_material_volume_to_optical,
    phase0_provisional_mapping,
)


def main() -> None:
    config = phase0_provisional_mapping()
    for fixture in all_synthetic_fixtures():
        optical = map_material_volume_to_optical(fixture.volume, config)
        interfaces = derive_ior_interfaces(optical)
        interior_axes = Counter(face.axis.value for face in interfaces.interior_faces)
        print(
            f"{fixture.manifest.name}: total={len(interfaces.faces)} "
            f"interior={len(interfaces.interior_faces)} "
            f"exterior={len(interfaces.exterior_faces)} "
            f"interior_axes={dict(sorted(interior_axes.items()))}"
        )


if __name__ == "__main__":
    main()
