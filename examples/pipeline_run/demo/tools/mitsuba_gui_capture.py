"""Capture stage-viewer GUI panel screenshots for manual authoring.

This is a docs/tooling script, not a test: it drives the real
``mitsuba_stage_viewer.py`` (viser + Mitsuba) in a subprocess, opens it in
headless Chromium via Playwright, clicks through every GUI tab, and writes
one screenshot per state plus a JSON manifest describing exactly what
produced each image. The manifest is meant to be handed to an AI agent
together with the images so a manual can be written from recorded facts
(tab names, panel text, actions taken) instead of guessed from pixels.

It intentionally does *not* validate rendering correctness: the 3D
viewport content is out of scope and may be a low-spp placeholder. Pixel
correctness of the render is already covered by the Python-level
unit/integration tests under ``vdbmat/tests``; this script only checks that
the GUI *panel* was reachable and legible enough to screenshot.

Setup (one-time browser download):

    cd vdbmat
    uv sync --group mitsuba-viewer --group gui-capture
    uv run --group gui-capture playwright install chromium

Usage:

    cd vdbmat
    uv run --group mitsuba-viewer --group gui-capture \\
        python examples/pipeline_run/demo/tools/mitsuba_gui_capture.py \\
        --out .local/gui_image_export/captures \\
        [--port 8090] [--keep-viewer] [--viewport-size 1400x900] \\
        [--ready-timeout 180]

Output layout (under ``--out/<timestamp>/``):

    manifest.json       -- see MANIFEST_SCHEMA below
    01-input-panel.png  -- one pair of full/panel screenshots per tab, plus
    01-input-full.png       a couple of representative interaction states
    ...

The manifest schema (``vdbmat.gui-capture-manifest/1.0``):

    {
      "schema": "vdbmat.gui-capture-manifest/1.0",
      "captured_at": "<ISO-8601 UTC>",
      "viewer_command": ["<argv...>"],
      "viewport_size": [1400, 900],
      "captures": [
        {
          "image_full": "01-input-full.png",
          "image_panel": "01-input-panel.png",
          "tab": "Input",
          "actions": ["click tab 'Input'"],
          "status_text": "preview settled/traverse 0.01s ...",
          "panel_text": ["Mitsuba stage viewer", "...", ...],
          "error": null
        },
        ...
      ]
    }

A capture with a non-null ``error`` means that tab/action could not be
reached (for example a renamed tab label); the run continues so one broken
tab does not lose the rest of the captures.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

MANIFEST_SCHEMA = "vdbmat.gui-capture-manifest/1.0"

# Kept in sync by hand with the tab labels built in
# ``mitsuba_stage_binder.py`` (``StageBinder.__init__``) and the Input/Output
# tabs added in ``mitsuba_stage_viewer.py`` (``ViewerApp._build_input_tab`` /
# ``_build_preset_tab`` / the Output tab in ``ViewerApp.__init__``). If the
# GUI adds/renames a tab, update this tuple -- a mismatch shows up as a
# per-tab "error" entry in the manifest rather than failing the whole run.
TAB_NAMES: tuple[str, ...] = (
    "Input",
    "Preset",
    "Render",
    "Backdrop",
    "Floor",
    "Key light",
    "Camera",
    "Backlight",
    "Output",
)

# The viser control panel is the unique ``.mantine-Paper-root`` that contains
# the panel label set via ``gui.set_panel_label("Mitsuba stage viewer")`` in
# ``ViewerApp.__init__``. Filtering by that text (rather than DOM order) is
# what makes this selector survive unrelated viser DOM churn (see report for
# this tool: two other Paper-root elements exist -- an empty positioning
# wrapper and a hidden dev-options popover -- and their order is not
# guaranteed across viser versions).
PANEL_LABEL = "Mitsuba stage viewer"

_READY_LOG_MARKER = "viewer ready:"


@dataclass
class Capture:
    image_full: str
    image_panel: str | None
    tab: str
    actions: list[str]
    status_text: str | None
    panel_text: list[str]
    error: str | None = None


@dataclass
class CaptureRun:
    viewer_command: list[str]
    viewport_size: tuple[int, int]
    captures: list[Capture] = field(default_factory=list)

    def to_manifest(self) -> dict:
        return {
            "schema": MANIFEST_SCHEMA,
            "captured_at": datetime.now(UTC).isoformat(),
            "viewer_command": self.viewer_command,
            "viewport_size": list(self.viewport_size),
            "captures": [
                {
                    "image_full": c.image_full,
                    "image_panel": c.image_panel,
                    "tab": c.tab,
                    "actions": c.actions,
                    "status_text": c.status_text,
                    "panel_text": c.panel_text,
                    "error": c.error,
                }
                for c in self.captures
            ],
        }


def prepare_fixture_input(fixture_root: Path) -> Path:
    """Build (or reuse) a minimal canonical run bundle for the capture run.

    Reuses the existing pipeline and checked-in ``nested_material_cube``
    voxel manifest -- the same recipe
    ``tests/integration/test_mitsuba_stage_viewer.py::_write_nested_material_cube_bundle``
    uses -- rather than inventing a new fixture. Returns the bundle
    directory (suitable as the viewer's positional input argument).
    """
    from vdbmat.pipeline import PipelineConfig, run_pipeline

    bundle_dir = fixture_root / "nested_material_cube"
    if (bundle_dir / "run.json").exists():
        return bundle_dir

    repo_root = Path(__file__).resolve().parents[4]
    inputs_dir = repo_root / "examples" / "pipeline_run" / "inputs"
    fixture_root.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig(
        input_kind="direct-voxel",
        input_path="nested_material_cube.voxels.json",
        output_path=str(bundle_dir),
        mapping_name="phase0-provisional-materials-v1",
    )
    result = run_pipeline(config, base_dir=str(inputs_dir))
    return result.output_path


def _checked_in_mapping_root() -> Path:
    return Path(__file__).resolve().parents[3] / "pipeline_run" / "mappings"


def _checked_in_preset_root() -> Path:
    return Path(__file__).resolve().parent.parent / "presets"


def _demo_viewer_script() -> Path:
    return Path(__file__).resolve().parent.parent / "mitsuba_stage_viewer.py"


def _start_viewer(
    *,
    bundle_dir: Path,
    input_root: Path,
    work_dir: Path,
    port: int,
    ready_timeout: float,
) -> tuple[subprocess.Popen, list[str], list[str]]:
    """Launch the viewer subprocess and block until it reports readiness.

    Readiness is judged from the ``viewer ready: ...`` stdout line the
    viewer's own ``main()`` prints (see ``mitsuba_stage_viewer.py``), so this
    stays truthful to what the viewer itself considers "up" instead of
    guessing from a fixed sleep. ``PYTHONUNBUFFERED=1`` is required: without
    it, that print sits in the child's stdout buffer indefinitely because
    stdout is a pipe, not a TTY.
    """
    command = [
        sys.executable,
        str(_demo_viewer_script()),
        str(bundle_dir),
        "--input-root",
        str(input_root),
        "--mapping-root",
        str(_checked_in_mapping_root()),
        "--preset-root",
        str(_checked_in_preset_root()),
        "--port",
        str(port),
        "--work-dir",
        str(work_dir),
        "--preview-size",
        "64",
        "--preview-spp",
        "2",
        "--interactive-spp",
        "1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert process.stdout is not None
    lines: list[str] = []
    deadline = time.monotonic() + ready_timeout
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if line:
            lines.append(line.rstrip("\n"))
            if _READY_LOG_MARKER in line:
                return process, command, lines
        if process.poll() is not None:
            break
    process.kill()
    process.wait(timeout=10)
    raise RuntimeError(
        "viewer did not report readiness within "
        f"{ready_timeout:.0f}s; stdout so far:\n" + "\n".join(lines)
    )


def _dismiss_webgl_warning(page) -> None:
    """Close the headless-Chromium 'software WebGL' toast if present.

    It is a browser artifact of running Chromium without a GPU, not part of
    the GUI under manual, and would otherwise sit on top of every viewport
    screenshot.
    """
    toast = page.get_by_text("Software WebGL rendering detected", exact=False)
    if toast.count() == 0:
        return
    close_button = page.get_by_role("button").first
    with contextlib.suppress(Exception):
        close_button.click(timeout=1000)


def _panel_locator(page):
    return page.locator(".mantine-Paper-root").filter(has_text=PANEL_LABEL)


def _panel_text_lines(page) -> list[str]:
    panel = _panel_locator(page)
    if panel.count() == 0:
        return []
    text = panel.first.inner_text()
    return [line for line in text.splitlines() if line.strip()]


def _status_text(panel_lines: list[str]) -> str | None:
    # panel_lines[0] is always the panel label itself (see PANEL_LABEL);
    # the status markdown (``ViewerApp.status``) is the next non-empty line.
    return panel_lines[1] if len(panel_lines) > 1 else None


def run_captures(
    page,
    out_dir: Path,
    *,
    viewport_size: tuple[int, int],
) -> list[Capture]:
    captures: list[Capture] = []

    def _snapshot(*, index: int, tab: str, slug: str, actions: list[str]) -> Capture:
        page.wait_for_timeout(200)
        _dismiss_webgl_warning(page)
        panel_lines = _panel_text_lines(page)
        full_name = f"{index:02d}-{slug}-full.png"
        panel_name = f"{index:02d}-{slug}-panel.png"
        page.screenshot(path=str(out_dir / full_name), full_page=False)
        panel = _panel_locator(page)
        panel_image: str | None = panel_name
        if panel.count() == 0:
            panel_image = None
        else:
            panel.first.screenshot(path=str(out_dir / panel_name))
        return Capture(
            image_full=full_name,
            image_panel=panel_image,
            tab=tab,
            actions=actions,
            status_text=_status_text(panel_lines),
            panel_text=panel_lines,
        )

    for index, tab_name in enumerate(TAB_NAMES, start=1):
        slug = tab_name.lower().replace(" ", "-")
        actions = [f"click tab '{tab_name}'"]
        try:
            page.get_by_role("tab", name=tab_name, exact=True).click(timeout=5000)
        except Exception as error:
            captures.append(
                Capture(
                    image_full="",
                    image_panel=None,
                    tab=tab_name,
                    actions=actions,
                    status_text=None,
                    panel_text=[],
                    error=f"tab click failed: {error}",
                )
            )
            continue
        captures.append(
            _snapshot(index=index, tab=tab_name, slug=slug, actions=actions)
        )

    # Representative interaction states (initial version limited to two, per
    # plan.md Step 2): the Input dropdown open, and the Preset tab with a
    # selection made (summary populated) but not applied.
    try:
        page.get_by_role("tab", name="Input", exact=True).click(timeout=5000)
        page.get_by_role("textbox").first.click(timeout=5000)
        captures.append(
            _snapshot(
                index=len(TAB_NAMES) + 1,
                tab="Input",
                slug="input-dropdown-open",
                actions=[
                    "click tab 'Input'",
                    "click input dropdown (open candidate list)",
                ],
            )
        )
        # Close the dropdown so it doesn't bleed into later captures.
        page.keyboard.press("Escape")
    except Exception as error:
        captures.append(
            Capture(
                image_full="",
                image_panel=None,
                tab="Input",
                actions=["click tab 'Input'", "click input dropdown"],
                status_text=None,
                panel_text=[],
                error=f"input dropdown interaction failed: {error}",
            )
        )

    try:
        page.get_by_role("tab", name="Preset", exact=True).click(timeout=5000)
        captures.append(
            _snapshot(
                index=len(TAB_NAMES) + 2,
                tab="Preset",
                slug="preset-summary",
                actions=["click tab 'Preset'"],
            )
        )
    except Exception as error:
        captures.append(
            Capture(
                image_full="",
                image_panel=None,
                tab="Preset",
                actions=["click tab 'Preset'"],
                status_text=None,
                panel_text=[],
                error=f"preset tab capture failed: {error}",
            )
        )

    return captures


def _self_check(out_dir: Path, run: CaptureRun) -> None:
    """Fail loudly if captures are missing, empty, or wrong-sized.

    This is the only automated verification this script does -- it is a
    docs tool, not a correctness test (see module docstring) -- so it must
    at least guarantee the manual-writing agent gets real, viewport-sized
    images rather than 0-byte files.
    """
    expected = len(TAB_NAMES) + 2
    if len(run.captures) != expected:
        raise RuntimeError(
            f"expected {expected} capture entries, got {len(run.captures)}"
        )
    failures: list[str] = []
    for capture in run.captures:
        if capture.error is not None:
            failures.append(f"{capture.tab}: {capture.error}")
            continue
        for name in (capture.image_full, capture.image_panel):
            if not name:
                continue
            path = out_dir / name
            if not path.exists() or path.stat().st_size == 0:
                failures.append(f"{capture.tab}: missing or empty image {name}")
    if failures:
        raise RuntimeError("gui capture self-check failed:\n" + "\n".join(failures))


def _parse_viewport_size(raw: str) -> tuple[int, int]:
    try:
        width_str, height_str = raw.lower().split("x", 1)
        return int(width_str), int(height_str)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"--viewport-size must look like WIDTHxHEIGHT, got {raw!r}"
        ) from error


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="capture output root; a timestamped subdirectory is created "
        "under it (example: .local/gui_image_export/captures)",
    )
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument(
        "--keep-viewer",
        action="store_true",
        help="leave the viewer subprocess running after capture (for "
        "interactive follow-up); default is to stop it",
    )
    parser.add_argument(
        "--viewport-size", type=_parse_viewport_size, default=(1400, 900)
    )
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=180.0,
        help="seconds to wait for 'viewer ready' on a cold start (Mitsuba "
        "JIT compilation can be slow the first time)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    args.out = args.out.resolve()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    fixture_root = args.out / "fixture"
    bundle_dir = prepare_fixture_input(fixture_root)
    input_root = bundle_dir.parent
    work_dir = args.out / "work" / timestamp

    process, command, startup_log = _start_viewer(
        bundle_dir=bundle_dir,
        input_root=input_root,
        work_dir=work_dir,
        port=args.port,
        ready_timeout=args.ready_timeout,
    )
    print("\n".join(startup_log))

    run = CaptureRun(viewer_command=command, viewport_size=args.viewport_size)
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(
                    viewport={
                        "width": args.viewport_size[0],
                        "height": args.viewport_size[1],
                    }
                )
                page.goto(
                    f"http://127.0.0.1:{args.port}",
                    wait_until="load",
                    timeout=30000,
                )
                page.get_by_role("tab").first.wait_for(state="visible", timeout=30000)
                run.captures = run_captures(
                    page, out_dir, viewport_size=args.viewport_size
                )
            finally:
                browser.close()
    finally:
        if args.keep_viewer:
            print(f"viewer left running: pid={process.pid} port={args.port}")
        else:
            process.kill()
            process.wait(timeout=10)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(run.to_manifest(), indent=2) + "\n", encoding="utf-8"
    )

    _self_check(out_dir, run)
    print(f"captures written to {out_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
