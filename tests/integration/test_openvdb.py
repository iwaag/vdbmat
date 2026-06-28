from pathlib import Path

import numpy as np
import pytest

from vbdmat.exporters.openvdb import export_openvdb
from vbdmat.fixtures import anisotropic_axis_marker
from vbdmat.optics import map_material_volume_to_optical, phase0_provisional_mapping

vdb = pytest.importorskip("openvdb")

pytestmark = pytest.mark.openvdb


def test_written_grids_round_trip_names_values_dimensions_and_transform(
    tmp_path: Path,
) -> None:
    fixture = anisotropic_axis_marker()
    volume = map_material_volume_to_optical(
        fixture.volume, phase0_provisional_mapping()
    )
    result = export_openvdb(volume, tmp_path, name="axis-marker")
    grids = vdb.readAll(str(result.vdb_path))[0]
    assert tuple(grid.name for grid in grids) == result.grid_names
    assert all(grid.__class__.__name__ == "FloatGrid" for grid in grids)
    assert all(grid.evalActiveVoxelBoundingBox()[1] <= (3, 2, 1) for grid in grids)
    by_name = {grid.name: grid for grid in grids}
    assert by_name["sigma_a_g"].getConstAccessor().getValue((3, 0, 0)) == pytest.approx(
        float(volume.sigma_a[0, 0, 3, 1])
    )
    for zyx in ((0, 0, 0), (1, 2, 3)):
        z, y, x = zyx
        np.testing.assert_allclose(
            by_name["ior"].transform.indexToWorld((x, y, z)),
            volume.geometry.cell_center_world(zyx),
        )
