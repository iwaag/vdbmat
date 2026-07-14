"""Export every Phase 0 optical fixture to OpenVDB for the Cycles proof."""

import argparse
import json
from pathlib import Path

from vdbmat.exporters.openvdb import OpenVDBExportConfig, export_openvdb
from vdbmat.fixtures import all_synthetic_fixtures
from vdbmat.optics import map_material_volume_to_optical, phase0_provisional_mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()
    config = OpenVDBExportConfig(
        width=args.width, height=args.height, samples=args.samples, seed=args.seed
    )
    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    mapping = phase0_provisional_mapping()
    for fixture in all_synthetic_fixtures():
        optical = map_material_volume_to_optical(fixture.volume, mapping)
        result = export_openvdb(
            optical,
            args.output / fixture.manifest.name,
            name=fixture.manifest.name,
            config=config,
        )
        records.append(
            {
                "fixture": fixture.manifest.name,
                "vdb": str(result.vdb_path.relative_to(args.output)),
                "manifest": str(result.manifest_path.relative_to(args.output)),
                "capabilities": str(result.capability_path.relative_to(args.output)),
            }
        )
        print(f"{fixture.manifest.name}: {result.vdb_path}")
    (args.output / "export-report.json").write_text(
        json.dumps(
            {"consumer": "openvdb-blender-cycles", "fixtures": records},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
