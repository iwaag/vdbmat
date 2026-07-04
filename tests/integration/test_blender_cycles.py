import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from vdbmat.exporters.openvdb import OpenVDBExportConfig, export_openvdb
from vdbmat.fixtures import all_synthetic_fixtures
from vdbmat.optics import map_material_volume_to_optical, phase0_provisional_mapping

try:
    import openvdb  # noqa: F401
except ModuleNotFoundError:
    pytest.importorskip("pyopenvdb")
BLENDER = shutil.which("blender")
if BLENDER is None:
    pytest.skip("Blender executable is unavailable", allow_module_level=True)

pytestmark = [pytest.mark.openvdb, pytest.mark.blender]


def test_every_fixture_loads_and_renders_without_scene_edits(tmp_path: Path) -> None:
    exports = tmp_path / "exports"
    renders = tmp_path / "renders"
    config = OpenVDBExportConfig(width=8, height=8, samples=1, seed=17)
    mapping = phase0_provisional_mapping()
    fixtures = all_synthetic_fixtures()
    for fixture in fixtures:
        volume = map_material_volume_to_optical(fixture.volume, mapping)
        export_openvdb(
            volume,
            exports / fixture.manifest.name,
            name=fixture.manifest.name,
            config=config,
        )
    runner = Path("examples/phase0/render_blender_fixtures.py").resolve()
    subprocess.run(
        [
            sys.executable,
            str(runner),
            str(exports),
            str(renders),
            "--blender",
            BLENDER,
        ],
        check=True,
    )
    for fixture in fixtures:
        assert (renders / f"{fixture.manifest.name}.png").stat().st_size > 0
        assert (renders / f"{fixture.manifest.name}.blend").stat().st_size > 0
    assert (renders / "render-report.json").is_file()
