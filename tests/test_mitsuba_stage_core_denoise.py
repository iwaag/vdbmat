"""Unit tests for the denoise helpers in ``mitsuba_stage_core`` (denoise plan Step 2).

These exercise ``denoise_image`` / ``finalize_render_image`` /
``require_denoise_variant`` against a stub ``mi`` module (no dependency on the
real Mitsuba package, per the plan's "cut Mitsuba dependency at the function
boundary" verification approach) to check the raw-first / stats-from-raw
contract without needing a CUDA host.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).parents[1] / "examples" / "pipeline_run" / "demo"
sys.path.insert(0, str(DEMO_DIR))

from mitsuba_stage import RenderSettings  # noqa: E402
from mitsuba_stage_core import (  # noqa: E402
    DenoiseVariantError,
    denoise_image,
    finalize_render_image,
    require_denoise_variant,
)


class _StubDenoiser:
    def __init__(self, input_size: list[int]) -> None:
        self.input_size = tuple(input_size)
        self.calls: list[object] = []

    def __call__(self, image: object) -> object:
        self.calls.append(image)
        return f"denoised({image!r})"


class _StubUtil:
    def __init__(self) -> None:
        self.written: list[tuple[str, object]] = []

    def write_bitmap(self, path: str, image: object, write_async: bool) -> None:
        assert write_async is False
        self.written.append((path, image))


class _StubMi:
    def __init__(self, variant: str) -> None:
        self._variant = variant
        self.util = _StubUtil()
        self.denoisers: list[_StubDenoiser] = []

    def variant(self) -> str:
        return self._variant

    def OptixDenoiser(self, input_size: list[int]) -> _StubDenoiser:
        denoiser = _StubDenoiser(input_size)
        self.denoisers.append(denoiser)
        return denoiser


@pytest.mark.parametrize("variant", ["cuda_ad_rgb", "cuda_ad_rgb_double"])
def test_require_denoise_variant_accepts_cuda_family(variant: str) -> None:
    require_denoise_variant(variant)  # must not raise


def test_require_denoise_variant_rejects_cpu_variant() -> None:
    with pytest.raises(DenoiseVariantError, match="cuda_ad_rgb-family"):
        require_denoise_variant("llvm_ad_rgb")


def test_denoise_image_reuses_cache_per_resolution() -> None:
    mi = _StubMi("cuda_ad_rgb")
    cache: dict[tuple[int, int], object] = {}

    first = denoise_image(mi, "image-a", 64, 48, cache)
    second = denoise_image(mi, "image-b", 64, 48, cache)

    assert first == "denoised('image-a')"
    assert second == "denoised('image-b')"
    assert len(mi.denoisers) == 1
    assert mi.denoisers[0].input_size == (64, 48)


def test_denoise_image_builds_separate_denoiser_per_resolution() -> None:
    mi = _StubMi("cuda_ad_rgb")
    cache: dict[tuple[int, int], object] = {}

    denoise_image(mi, "image-a", 64, 48, cache)
    denoise_image(mi, "image-b", 32, 32, cache)

    assert len(mi.denoisers) == 2


def test_finalize_render_image_denoise_off_writes_only_final_pixel_identical(
    tmp_path: Path,
) -> None:
    mi = _StubMi("llvm_ad_rgb")
    output_png = tmp_path / "final.png"

    stats = finalize_render_image(
        mi, "raw-image", RenderSettings(denoise=False), output_png, "STATS", {}
    )

    assert stats == "STATS"
    assert mi.util.written == [(str(output_png), "raw-image")]
    assert not (tmp_path / "final.raw.png").exists()


def test_finalize_render_image_denoise_on_writes_raw_then_denoised(
    tmp_path: Path,
) -> None:
    mi = _StubMi("cuda_ad_rgb")
    output_png = tmp_path / "final.png"
    cache: dict[tuple[int, int], object] = {}

    stats = finalize_render_image(
        mi,
        "raw-image",
        RenderSettings(width=64, height=48, denoise=True),
        output_png,
        "STATS",
        cache,
    )

    assert stats == "STATS denoise=optix"
    assert mi.util.written == [
        (str(tmp_path / "final.raw.png"), "raw-image"),
        (str(output_png), "denoised('raw-image')"),
    ]


def test_finalize_render_image_denoise_on_requires_cuda_variant(
    tmp_path: Path,
) -> None:
    mi = _StubMi("llvm_ad_rgb")
    output_png = tmp_path / "final.png"

    with pytest.raises(DenoiseVariantError):
        finalize_render_image(
            mi, "raw-image", RenderSettings(denoise=True), output_png, "STATS", {}
        )

    assert mi.util.written == []
