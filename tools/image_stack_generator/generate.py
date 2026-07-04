"""External input generator: layered 2D image stack → ``vdbmat.voxels`` manifest.

This is the reference demonstration of the ADR-009 D2 input-generator contract: a
tool *outside* the core pipeline that turns non-voxel source data into the
manifest the core accepts. It deliberately depends on ``vdbmat`` only for the
canonical types and the shared manifest writer (generator → core, never the
reverse).

Input is a directory of grayscale PGM slices (binary ``P5`` or ASCII ``P2``,
8-bit) plus a stack configuration JSON:

```json
{
  "voxel_size_xyz_m": [0.001, 0.001, 0.001],
  "levels": [
    {"gray": 0,   "material_id": 0, "name": "air",               "role": "background"},
    {"gray": 255, "material_id": 1, "name": "transparent-resin", "role": "material"}
  ]
}
```

Slices are stacked in ascending filename order as z = 0, 1, ...; image rows map to
+Y and columns to +X. Every gray value present in the slices must be declared in
``levels`` — an undeclared value is an error, never a guess.

Usage::

    uv run python tools/image_stack_generator/generate.py SLICES_DIR CONFIG OUT_DIR NAME
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

from vdbmat.core import (
    GridGeometry,
    MaterialDefinition,
    MaterialLabelVolume,
    MaterialPalette,
    MaterialRole,
    Provenance,
)
from vdbmat.io import write_material_label_manifest

GENERATOR = "vdbmat-image-stack"
GENERATOR_VERSION = "0.1.0"


class StackError(ValueError):
    """A slice stack or its configuration violates the generator contract."""


def read_pgm(path: Path) -> npt.NDArray[np.uint8]:
    """Read one 8-bit grayscale PGM (P5 binary or P2 ASCII) slice."""
    data = path.read_bytes()
    tokens: list[bytes] = []
    index = 0
    # PGM header tokens may be separated by whitespace and '#' comments.
    while len(tokens) < 4 and index < len(data):
        if data[index : index + 1].isspace():
            index += 1
            continue
        if data[index : index + 1] == b"#":
            end = data.find(b"\n", index)
            index = len(data) if end == -1 else end + 1
            continue
        start = index
        while index < len(data) and not data[index : index + 1].isspace():
            index += 1
        tokens.append(data[start:index])
    if len(tokens) < 4:
        raise StackError(f"{path.name}: truncated PGM header")
    magic, width_token, height_token, maxval_token = tokens
    try:
        width, height, maxval = (
            int(width_token),
            int(height_token),
            int(maxval_token),
        )
    except ValueError as error:
        raise StackError(f"{path.name}: non-numeric PGM header") from error
    if width <= 0 or height <= 0:
        raise StackError(f"{path.name}: image dimensions must be positive")
    if maxval != 255:
        raise StackError(f"{path.name}: only 8-bit PGM (maxval 255) is supported")

    if magic == b"P5":
        pixels = data[index + 1 :]
        expected = width * height
        if len(pixels) != expected:
            raise StackError(
                f"{path.name}: expected {expected} pixel bytes, got {len(pixels)}"
            )
        return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width)
    if magic == b"P2":
        values = data[index:].split()
        if len(values) != width * height:
            raise StackError(
                f"{path.name}: expected {width * height} pixels, got {len(values)}"
            )
        try:
            flat = np.asarray([int(item) for item in values], dtype=np.int64)
        except ValueError as error:
            raise StackError(f"{path.name}: non-numeric P2 pixel") from error
        if flat.min() < 0 or flat.max() > 255:
            raise StackError(f"{path.name}: P2 pixels must lie in [0, 255]")
        return flat.astype(np.uint8).reshape(height, width)
    raise StackError(f"{path.name}: unsupported PGM magic {magic!r}")


def load_stack_config(
    path: Path,
) -> tuple[tuple[float, float, float], dict[int, int], MaterialPalette]:
    """Parse the stack configuration into voxel size, gray→ID map, and palette."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StackError(f"config: cannot parse {path}: {error}") from error
    if not isinstance(document, dict):
        raise StackError("config: must be a JSON object")
    unknown = sorted(set(document) - {"voxel_size_xyz_m", "levels"})
    if unknown:
        raise StackError(f"config: unknown fields: {unknown}")

    raw_size = document.get("voxel_size_xyz_m")
    if not isinstance(raw_size, list) or len(raw_size) != 3:
        raise StackError("config.voxel_size_xyz_m: must contain exactly 3 numbers")
    size = tuple(float(item) for item in raw_size)
    if any(not np.isfinite(item) or item <= 0.0 for item in size):
        raise StackError("config.voxel_size_xyz_m: must be finite and positive")

    levels = document.get("levels")
    if not isinstance(levels, list) or not levels:
        raise StackError("config.levels: must be a non-empty array")
    gray_to_id: dict[int, int] = {}
    definitions: list[MaterialDefinition] = []
    for index, entry in enumerate(levels):
        field = f"config.levels[{index}]"
        if not isinstance(entry, dict):
            raise StackError(f"{field}: must be an object")
        unknown = sorted(set(entry) - {"gray", "material_id", "name", "role"})
        if unknown:
            raise StackError(f"{field}: unknown fields: {unknown}")
        gray = entry.get("gray")
        if not isinstance(gray, int) or isinstance(gray, bool) or not 0 <= gray <= 255:
            raise StackError(f"{field}.gray: must be an integer in [0, 255]")
        if gray in gray_to_id:
            raise StackError(f"{field}.gray: duplicate gray level {gray}")
        material_id = entry.get("material_id")
        name = entry.get("name")
        role = entry.get("role")
        if not isinstance(material_id, int) or isinstance(material_id, bool):
            raise StackError(f"{field}.material_id: must be an integer")
        if not isinstance(name, str):
            raise StackError(f"{field}.name: must be a string")
        if not isinstance(role, str):
            raise StackError(f"{field}.role: must be a string")
        try:
            definition = MaterialDefinition(
                material_id=material_id,
                name=name,
                role=MaterialRole(role),
            )
        except (TypeError, ValueError) as error:
            raise StackError(f"{field}: {error}") from error
        gray_to_id[gray] = definition.material_id
        definitions.append(definition)
    try:
        palette = MaterialPalette.from_sequence(definitions)
    except (TypeError, ValueError) as error:
        raise StackError(f"config.levels: {error}") from error
    return (size[0], size[1], size[2]), gray_to_id, palette


def stack_to_volume(
    slices_dir: Path, config_path: Path
) -> tuple[MaterialLabelVolume, str]:
    """Build a canonical label volume from a slice directory and configuration."""
    voxel_size, gray_to_id, palette = load_stack_config(config_path)
    slice_paths = sorted(slices_dir.glob("*.pgm"))
    if not slice_paths:
        raise StackError(f"slices: no .pgm files under {slices_dir}")

    identity_hash = hashlib.sha256()
    layers: list[npt.NDArray[np.uint8]] = []
    for path in slice_paths:
        identity_hash.update(path.name.encode("utf-8"))
        identity_hash.update(path.read_bytes())
        layer = read_pgm(path)
        if layers and layer.shape != layers[0].shape:
            raise StackError(
                f"{path.name}: slice shape {layer.shape} differs from "
                f"{slice_paths[0].name} {layers[0].shape}"
            )
        layers.append(layer)
    stack = np.stack(layers, axis=0)  # [z, y, x] grayscale

    present = np.unique(stack)
    undeclared = sorted(int(v) for v in present if int(v) not in gray_to_id)
    if undeclared:
        raise StackError(
            f"slices: gray values {undeclared} are not declared in config.levels"
        )

    lookup = np.zeros(256, dtype=np.uint16)
    for gray, material_id in gray_to_id.items():
        lookup[gray] = material_id
    label = lookup[stack]

    identity = f"pgm-stack:sha256:{identity_hash.hexdigest()}"
    geometry = GridGeometry(
        shape_zyx=(int(stack.shape[0]), int(stack.shape[1]), int(stack.shape[2])),
        voxel_size_xyz_m=voxel_size,
    )
    provenance = Provenance(
        generator=GENERATOR,
        generator_version=GENERATOR_VERSION,
        sources=(f"slices:{len(slice_paths)}-pgm",),
        notes=f"layered image stack from {slices_dir.name}; rows=+Y, columns=+X",
    )
    volume = MaterialLabelVolume(
        geometry=geometry,
        palette=palette,
        provenance=provenance,
        material_id=label,
    )
    return volume, identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stack grayscale PGM slices into a vdbmat.voxels manifest."
    )
    parser.add_argument("slices_dir", type=Path, metavar="SLICES_DIR")
    parser.add_argument("config", type=Path, metavar="CONFIG")
    parser.add_argument("out_dir", type=Path, metavar="OUT_DIR")
    parser.add_argument("name", metavar="NAME")
    arguments = parser.parse_args(argv)
    try:
        volume, identity = stack_to_volume(arguments.slices_dir, arguments.config)
        manifest_path = write_material_label_manifest(
            arguments.out_dir, arguments.name, volume, identity=identity
        )
    except (StackError, ValueError) as error:
        sys.stderr.write(f"image-stack: {error}\n")
        return 3
    print(
        json.dumps(
            {
                "status": "ok",
                "manifest": str(manifest_path),
                "shape_zyx": list(volume.geometry.shape_zyx),
                "identity": identity,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
