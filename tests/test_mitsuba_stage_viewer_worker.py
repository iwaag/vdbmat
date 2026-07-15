from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).parents[1] / "examples" / "pipeline_run" / "demo"
sys.path.insert(0, str(DEMO_DIR))

from mitsuba_stage import StageConfig  # noqa: E402
from mitsuba_stage_viewer import (  # noqa: E402
    RenderWorker,
    _fit_preview_to_aspect,
    _parse_args,
)


def _wait_until(predicate, timeout: float = 2.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for render worker")
        time.sleep(0.005)


def test_render_worker_discards_stale_and_settles_latest() -> None:
    rendered: list[tuple[int, str]] = []
    published: list[tuple[int, str]] = []
    first_started = threading.Event()
    release_first = threading.Event()

    def render(config: StageConfig, quality: str) -> int:
        rendered.append((config.render.width, quality))
        if len(rendered) == 1:
            first_started.set()
            assert release_first.wait(1.0)
        return config.render.width

    worker = RenderWorker(settle_delay=0.02)
    worker.configure(
        render,
        lambda result, quality: published.append((int(result), quality)),
        lambda message: (_ for _ in ()).throw(AssertionError(message)),
    )
    worker.start()
    worker.request_preview(StageConfig())
    assert first_started.wait(1.0)
    latest = StageConfig().with_cli_overrides(width=640)
    worker.request_preview(latest)
    release_first.set()

    _wait_until(lambda: (640, "settled") in published)
    assert (512, "interactive") not in published
    assert published[-2:] == [(640, "interactive"), (640, "settled")]


def test_render_worker_serializes_final_job_and_latest_preview() -> None:
    events: list[str] = []
    final_started = threading.Event()
    release_final = threading.Event()

    def render(_config: StageConfig, quality: str) -> str:
        events.append(f"render-{quality}")
        return quality

    def final_job() -> None:
        events.append("final-start")
        final_started.set()
        assert release_final.wait(1.0)
        events.append("final-end")

    worker = RenderWorker(settle_delay=0.01)
    worker.configure(render, lambda _result, _quality: None, pytest_fail)
    worker.start()
    worker.submit(final_job)
    assert final_started.wait(1.0)
    worker.request_preview(StageConfig())
    release_final.set()

    _wait_until(lambda: "render-settled" in events)
    assert events.index("final-end") < events.index("render-interactive")


def pytest_fail(message: str) -> None:
    raise AssertionError(message)


def test_render_worker_recovers_after_preview_error() -> None:
    calls = 0
    errors: list[str] = []
    published: list[str] = []

    def render(_config: StageConfig, quality: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("expected failure")
        return quality

    worker = RenderWorker(settle_delay=0.01)
    worker.configure(
        render,
        lambda result, _quality: published.append(str(result)),
        errors.append,
    )
    worker.start()
    worker.request_preview(StageConfig())
    _wait_until(lambda: bool(errors))
    worker.request_preview(StageConfig().with_cli_overrides(width=640))
    _wait_until(lambda: "settled" in published)
    assert "expected failure" in errors[0]


def test_viewer_cli_rejects_interactive_spp_above_preview_spp() -> None:
    try:
        _parse_args(["input.zarr", "--preview-spp", "4", "--interactive-spp", "8"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("invalid spp combination was accepted")


def test_preview_is_padded_to_wide_and_tall_viewports_without_distortion() -> None:
    pixels = np.full((4, 4, 3), 255, dtype=np.uint8)

    wide = _fit_preview_to_aspect(pixels, 2.0)
    assert wide.shape == (4, 8, 3)
    assert np.array_equal(wide[:, 2:6], pixels)
    assert not np.any(wide[:, :2])
    assert not np.any(wide[:, 6:])

    tall = _fit_preview_to_aspect(pixels, 0.5)
    assert tall.shape == (8, 4, 3)
    assert np.array_equal(tall[2:6], pixels)
    assert not np.any(tall[:2])
    assert not np.any(tall[6:])
