from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vdbmat.exporters.mitsuba import MitsubaExportConfig, prepare_mitsuba_scene
from vdbmat.fixtures import homogeneous_transparent
from vdbmat.optics import map_material_volume_to_optical, phase0_provisional_mapping

DEMO_DIR = Path(__file__).parents[2] / "examples" / "pipeline_run" / "demo"
sys.path.insert(0, str(DEMO_DIR))

from mitsuba_stage import (  # noqa: E402
    BacklightOverride,
    CameraOverride,
    StageConfig,
    apply_stage,
)
from mitsuba_stage_viewer import TraversedPreviewScene  # noqa: E402

mi = pytest.importorskip("mitsuba")

pytestmark = pytest.mark.mitsuba


def _fresh_render(prepared, volume, config: StageConfig, spp: int) -> np.ndarray:  # type: ignore[no-untyped-def]
    scene_dict = dict(prepared.scene_dict)
    apply_stage(mi, scene_dict, volume.geometry, config)
    return np.asarray(mi.render(mi.load_dict(scene_dict), seed=17, spp=spp))


def test_traverse_updates_match_fresh_rebuild_and_structure_falls_back(
    tmp_path: Path,
) -> None:
    volume = map_material_volume_to_optical(
        homogeneous_transparent().volume, phase0_provisional_mapping()
    )
    render = MitsubaExportConfig(width=12, height=12, spp=2, seed=17)
    prepared = prepare_mitsuba_scene(volume, tmp_path / "scene", render)
    config = StageConfig(
        camera=CameraOverride(), backlight=BacklightOverride()
    ).with_cli_overrides(width=12, height=12, spp=2)
    preview = TraversedPreviewScene(mi, prepared, volume.geometry, config, seed=17)

    configs = [
        replace(
            config,
            backdrop=replace(
                config.backdrop,
                color0=(0.2, 0.1, 0.1),
                checker_scale=5,
                distance_factor=2.5,
            ),
        ),
        replace(
            config,
            floor=replace(
                config.floor,
                color1=(0.4, 0.5, 0.1),
                drop_factor=0.2,
                scale_factor=5.0,
            ),
        ),
        replace(
            config,
            key_light=replace(
                config.key_light,
                direction=(-1.0, -1.0, 3.0),
                distance_factor=4.0,
                scale_factor=0.8,
                radiance=(8.0, 4.0, 2.0),
            ),
        ),
        replace(config, backlight=BacklightOverride((2.0, 1.0, 1.0))),
        replace(config, camera=CameraOverride(-40.0, 35.0, 4.5, 42.0)),
    ]
    for changed in configs:
        traversed, route = preview.render(changed, spp=2)
        fresh = _fresh_render(prepared, volume, changed, spp=2)
        assert route == "traverse"
        np.testing.assert_allclose(np.asarray(traversed), fresh, rtol=0.0, atol=1e-5)
        config = changed

    structural = replace(config, floor=replace(config.floor, pattern="solid"))
    _, route = preview.render(structural, spp=2)
    assert route == "rebuild"
    continuous_again = replace(
        structural, floor=replace(structural.floor, color0=(0.1, 0.2, 0.3))
    )
    traversed, route = preview.render(continuous_again, spp=2)
    fresh = _fresh_render(prepared, volume, continuous_again, spp=2)
    assert route == "traverse"
    np.testing.assert_allclose(np.asarray(traversed), fresh, rtol=0.0, atol=1e-5)
