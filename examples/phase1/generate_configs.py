"""Regenerate the committed Phase 1 example pipeline run configurations.

Deterministic: rerunning this script reproduces byte-identical, human-reviewable
``config.json`` documents under ``examples/phase1/configs/``. Paths are portable and
relative to each config file's own directory; ``vbdmat run CONFIG`` resolves them
against the config's directory (see ADR-007/ADR-008 and plan Step 5).

Usage::

    uv run python examples/phase1/generate_configs.py
"""

from __future__ import annotations

import json
from pathlib import Path

from vbdmat.pipeline import InputKind, PipelineConfig

CONFIGS = Path(__file__).parent / "configs"


def window_coupon_config() -> PipelineConfig:
    """Direct-voxel window coupon → validated material + optical run bundle."""
    return PipelineConfig(
        input_kind=InputKind.DIRECT_VOXEL,
        input_path="../inputs/window_coupon.voxels.json",
        output_path="../../../.local/phase1/quickstart/window_coupon",
    )


def stepped_wedge_config() -> PipelineConfig:
    """Single-material stepped wedge supplied as a direct-voxel manifest."""
    return PipelineConfig(
        input_kind=InputKind.DIRECT_VOXEL,
        input_path="../inputs/stepped_wedge.voxels.json",
        output_path="../../../.local/phase1/quickstart/stepped_wedge",
    )


def _write(path: Path, config: PipelineConfig) -> None:
    text = json.dumps(config.to_json_dict(), indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path} (digest {config.digest})")


def main() -> None:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    _write(CONFIGS / "window_coupon.run.json", window_coupon_config())
    _write(CONFIGS / "stepped_wedge.run.json", stepped_wedge_config())


if __name__ == "__main__":
    main()
