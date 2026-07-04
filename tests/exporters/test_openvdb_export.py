import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vdbmat.exporters.openvdb import export_openvdb
from vdbmat.fixtures import anisotropic_axis_marker
from vdbmat.optics import map_material_volume_to_optical, phase0_provisional_mapping


class _Transform:
    def __init__(self, matrix):  # type: ignore[no-untyped-def]
        self.matrix = matrix

    def copy(self):  # type: ignore[no-untyped-def]
        return _Transform(self.matrix)


class _Grid:
    def __init__(self) -> None:
        self.metadata: dict[str, str] = {}
        self.array: np.ndarray | None = None

    def __setitem__(self, key: str, value: str) -> None:
        self.metadata[key] = value

    def copyFromArray(self, array: np.ndarray, tolerance: float) -> None:
        assert tolerance == 0.0
        self.array = array.copy()


def test_export_uses_named_float_grids_common_transform_and_metadata(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    written: dict[str, object] = {}

    def write(path: str, *, grids, metadata):  # type: ignore[no-untyped-def]
        written.update(path=path, grids=grids, metadata=metadata)
        Path(path).write_bytes(b"fake-vdb")

    fake = SimpleNamespace(
        FloatGrid=_Grid,
        createLinearTransform=lambda *, matrix: _Transform(matrix),
        write=write,
    )
    monkeypatch.setitem(sys.modules, "openvdb", fake)
    fixture = anisotropic_axis_marker()
    volume = map_material_volume_to_optical(
        fixture.volume, phase0_provisional_mapping()
    )
    result = export_openvdb(volume, tmp_path, name=fixture.manifest.name)

    grids = written["grids"]
    assert [grid.name for grid in grids] == list(result.grid_names)  # type: ignore[attr-defined]
    assert all(grid.array.shape == (4, 3, 2) for grid in grids)  # type: ignore[attr-defined]
    assert all(grid.metadata["vdbmat:dimensions"] == "x,y,z" for grid in grids)  # type: ignore[attr-defined]
    transform = grids[0].transform  # type: ignore[index,union-attr]
    assert all(grid.transform.matrix == transform.matrix for grid in grids)  # type: ignore[attr-defined]
    # OpenVDB right-multiplies row vectors, hence the serialized matrix transpose.
    index = np.asarray((3.0, 2.0, 1.0, 1.0))
    actual = index @ np.asarray(transform.matrix)
    np.testing.assert_allclose(actual[:3], volume.geometry.cell_center_world((1, 2, 3)))
    assert result.vdb_path.read_bytes() == b"fake-vdb"
    assert result.manifest_path.is_file()
    assert result.capability_path.is_file()
