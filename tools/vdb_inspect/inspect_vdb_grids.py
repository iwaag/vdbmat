"""Report per-grid active-voxel value ranges of an exported OpenVDB asset.

A small diagnostic for the demo/debug workflow. It answers the question that made the
2026-07-02 black-render investigation click: "what is actually inside this volume?"
For example it reveals that the ``stepped_wedge`` fixture has an all-zero
``cycles_scattering`` grid and only a tiny uniform absorption, i.e. it is pure
transparent resin and cannot be seen as a bare Cycles medium.

Requires the OpenVDB Python bindings, so run it inside the pinned native container:

    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONPATH=/work/src \
        -v "$PWD:/work" -w /work vdbmat-openvdb-cycles \
        python3 tools/phase0/inspect_vdb_grids.py \
        .local/phase1/step10/runs/stepped_wedge/exports/openvdb/openvdb-manifest.json

Accepts either an ``openvdb-manifest.json`` or a ``volume.vdb`` path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import openvdb as vdb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - depends on runtime
    import pyopenvdb as vdb  # type: ignore


def _resolve_vdb(target: Path) -> tuple[Path, dict[str, object] | None]:
    if target.suffix == ".vdb":
        return target, None
    manifest = json.loads(target.read_text(encoding="utf-8"))
    return (target.parent / str(manifest["vdb"])).resolve(), manifest


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: inspect_vdb_grids.py MANIFEST_OR_VDB")
    path, manifest = _resolve_vdb(Path(sys.argv[1]).resolve())
    print(f"file: {path}")
    if manifest is not None:
        print(f"dims: {manifest.get('dimensions_xyz')}  phase_g: {manifest.get('phase_g')}")

    for meta in vdb.readAllGridMetadata(str(path)):
        grid = vdb.read(str(path), meta.name)
        values = [item.value for item in grid.citerOnValues()]
        if not values:
            print(f"  {meta.name:20s} active=0 (empty grid)")
            continue
        lo = min(values)
        hi = max(values)
        mean = sum(values) / len(values)
        print(
            f"  {meta.name:20s} active={len(values):7d} "
            f"min={lo:<12.6g} max={hi:<12.6g} mean={mean:<12.6g}"
        )


if __name__ == "__main__":
    main()
