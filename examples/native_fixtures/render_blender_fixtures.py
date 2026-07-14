"""Render all previously exported Phase 0 VDB fixtures with Blender."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exports", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    script = Path(__file__).with_name("blender_cycles_volume.py").resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    manifests = sorted(args.exports.glob("*/openvdb-manifest.json"))
    if not manifests:
        raise SystemExit(f"no fixture manifests found below {args.exports}")
    for manifest in manifests:
        name = manifest.parent.name
        png = (args.output / f"{name}.png").resolve()
        subprocess.run(
            [
                args.blender,
                "--background",
                "--python",
                str(script),
                "--",
                str(manifest.resolve()),
                str(png),
            ],
            check=True,
        )
        if not png.is_file():
            raise RuntimeError(
                f"Blender exited without producing the expected render: {png}"
            )
        records.append(
            {
                "fixture": name,
                "png": png.name,
                "blend": png.with_suffix(".blend").name,
                "png_sha256": _sha256(png),
            }
        )
        print(f"{name}: {records[-1]['png_sha256']}")
    (args.output / "render-report.json").write_text(
        json.dumps(
            {"renderer": "blender-cycles", "fixtures": records},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
