"""Print deterministic summaries of every Phase 0 synthetic fixture."""

from vdbmat.fixtures import all_synthetic_fixtures


def main() -> None:
    for fixture in all_synthetic_fixtures():
        manifest = fixture.manifest
        print(f"{manifest.name}: shape_zyx={manifest.shape_zyx}")
        if manifest.material_voxel_counts:
            print(f"  material_voxel_counts={manifest.material_voxel_counts}")
        if manifest.material_fraction_totals:
            print(f"  material_fraction_totals={manifest.material_fraction_totals}")
        print(f"  world_bounds_xyz_m={manifest.world_bounds_xyz_m}")


if __name__ == "__main__":
    main()
