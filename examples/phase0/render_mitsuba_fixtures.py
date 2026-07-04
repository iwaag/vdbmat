"""Render every mapped Phase 0 fixture with the fixed Mitsuba proof scene."""

import argparse
import json
from pathlib import Path

from vdbmat.exporters.mitsuba import MitsubaExportConfig, render_mitsuba
from vdbmat.fixtures import all_synthetic_fixtures
from vdbmat.optics import (
    map_material_volume_to_optical,
    phase0_provisional_mapping,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--spp", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    render_config = MitsubaExportConfig(
        width=args.width,
        height=args.height,
        spp=args.spp,
        seed=args.seed,
    )
    mapping_config = phase0_provisional_mapping()
    records: list[dict[str, object]] = []
    for fixture in all_synthetic_fixtures():
        optical = map_material_volume_to_optical(fixture.volume, mapping_config)
        result = render_mitsuba(
            optical,
            args.output / fixture.manifest.name,
            name=fixture.manifest.name,
            config=render_config,
        )
        record: dict[str, object] = {
            "fixture": fixture.manifest.name,
            "png": str(result.png_path.relative_to(args.output)),
            "attenuation_png": str(
                result.attenuation_png_path.relative_to(args.output)
            ),
            "exr": str(result.exr_path.relative_to(args.output)),
            "png_sha256": result.png_sha256,
            "attenuation_png_sha256": result.attenuation_png_sha256,
            "mean_linear_rgb": list(result.mean_linear_rgb),
            "minimum": result.minimum,
            "maximum": result.maximum,
        }
        records.append(record)
        print(
            f"{fixture.manifest.name}: {result.png_sha256} "
            f"mean={result.mean_linear_rgb}"
        )

    report = {
        "renderer": "mitsuba-3",
        "variant": render_config.variant,
        "width": render_config.width,
        "height": render_config.height,
        "spp": render_config.spp,
        "seed": render_config.seed,
        "fixtures": records,
    }
    (args.output / "render-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
