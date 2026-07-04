from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vdbmat.exporters.mitsuba import (
    MitsubaExportConfig,
    prepare_mitsuba_scene,
    render_mitsuba,
)
from vdbmat.fixtures import all_synthetic_fixtures, homogeneous_transparent
from vdbmat.optics import (
    map_material_volume_to_optical,
    phase0_provisional_mapping,
)

mi = pytest.importorskip("mitsuba")

pytestmark = pytest.mark.mitsuba


def _mapped(fixture):  # type: ignore[no-untyped-def]
    return map_material_volume_to_optical(fixture.volume, phase0_provisional_mapping())


@pytest.mark.parametrize(
    "fixture", all_synthetic_fixtures(), ids=lambda item: item.manifest.name
)
def test_every_fixture_scene_loads_and_renders(
    tmp_path: Path,
    fixture,  # type: ignore[no-untyped-def]
) -> None:
    config = MitsubaExportConfig(width=12, height=12, spp=1, seed=17)
    result = render_mitsuba(
        _mapped(fixture), tmp_path / fixture.manifest.name, config=config
    )
    assert result.exr_path.is_file()
    assert result.png_path.is_file()
    assert result.attenuation_png_path.is_file()
    assert result.png_sha256.startswith("sha256:")
    assert result.attenuation_png_sha256.startswith("sha256:")
    assert np.isfinite(result.mean_linear_rgb).all()
    assert result.maximum > 0.0


def test_prepared_scene_has_nearest_raw_grids_and_loads(tmp_path: Path) -> None:
    config = MitsubaExportConfig(width=8, height=8, spp=1)
    prepared = prepare_mitsuba_scene(
        _mapped(homogeneous_transparent()), tmp_path, config
    )
    medium = prepared.scene_dict["vdbmat_medium"]
    assert medium["sigma_t"]["filter_type"] == "nearest"
    assert medium["sigma_t"]["raw"] is True
    assert medium["albedo"]["filter_type"] == "nearest"
    assert medium["scale"] == 1.0
    scene = mi.load_dict(prepared.scene_dict)
    assert scene is not None
    assert (tmp_path / "capabilities.json").is_file()
    assert (tmp_path / "scene-summary.json").is_file()


@pytest.mark.parametrize("field", ["sigma_a", "sigma_s"])
def test_increasing_extinction_has_monotonic_backlit_response(
    tmp_path: Path, field: str
) -> None:
    base = _mapped(homogeneous_transparent())
    one_ior = np.ones_like(base.ior)
    config = MitsubaExportConfig(width=24, height=24, spp=8, seed=11)
    means: list[float] = []
    for coefficient in (0.0, 10_000.0, 50_000.0):
        sigma_a = np.zeros_like(base.sigma_a)
        sigma_s = np.zeros_like(base.sigma_s)
        if field == "sigma_a":
            sigma_a.fill(coefficient)
        else:
            sigma_s.fill(coefficient)
        volume = replace(
            base,
            sigma_a=sigma_a,
            sigma_s=sigma_s,
            g=np.zeros_like(base.g),
            ior=one_ior,
        )
        result = render_mitsuba(
            volume,
            tmp_path / f"{field}-{int(coefficient)}",
            config=config,
        )
        means.append(float(np.mean(result.mean_linear_rgb)))
    assert means[0] > means[1] > means[2]


def test_fixed_seed_render_is_reproducible(tmp_path: Path) -> None:
    volume = _mapped(homogeneous_transparent())
    config = MitsubaExportConfig(width=16, height=16, spp=4, seed=23)
    first = render_mitsuba(volume, tmp_path / "first", config=config)
    second = render_mitsuba(volume, tmp_path / "second", config=config)
    assert first.png_sha256 == second.png_sha256
    assert first.mean_linear_rgb == second.mean_linear_rgb
