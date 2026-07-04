"""End-to-end demonstration of the ADR-009 input-generator contract.

The image-stack tool lives outside the core package (``tools/``); these tests
prove that a non-mesh external generator can produce a ``vdbmat.voxels`` manifest
that the unmodified core pipeline accepts, including with a file-based optical
mapping — the Phase 1-side1 exit criterion.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from vdbmat.io import read_material_label_manifest
from vdbmat.optics import phase0_provisional_mapping, write_optical_mapping
from vdbmat.pipeline import PipelineConfig, run_pipeline

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENERATOR = _REPO_ROOT / "tools" / "image_stack_generator" / "generate.py"


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("image_stack_generate", _GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_pgm_p5(path: Path, pixels: np.ndarray) -> None:
    height, width = pixels.shape
    path.write_bytes(
        f"P5\n{width} {height}\n255\n".encode("ascii")
        + pixels.astype(np.uint8).tobytes()
    )


def _stack_fixture(base: Path) -> tuple[Path, Path]:
    """Three 6x8 slices: a transparent block with a white core in the middle z."""
    slices = base / "slices"
    slices.mkdir()
    plain = np.full((6, 8), 255, dtype=np.uint8)  # transparent-resin
    cored = plain.copy()
    cored[2:4, 3:6] = 128  # white-resin core
    _write_pgm_p5(slices / "slice_000.pgm", plain)
    _write_pgm_p5(slices / "slice_001.pgm", cored)
    _write_pgm_p5(slices / "slice_002.pgm", plain)

    config = base / "stack.json"
    config.write_text(
        json.dumps(
            {
                "voxel_size_xyz_m": [0.001, 0.001, 0.001],
                "levels": [
                    {"gray": 0, "material_id": 0, "name": "air",
                     "role": "background"},
                    {"gray": 255, "material_id": 1, "name": "transparent-resin",
                     "role": "material"},
                    {"gray": 128, "material_id": 2, "name": "white-resin",
                     "role": "material"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return slices, config


def test_stack_produces_a_conforming_manifest(
    generator: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    slices, config = _stack_fixture(tmp_path)
    exit_code = generator.main(
        [str(slices), str(config), str(tmp_path / "out"), "cored_block"]
    )
    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["shape_zyx"] == [3, 6, 8]

    volume = read_material_label_manifest(tmp_path / "out" / "cored_block.voxels.json")
    label = np.asarray(volume.material_id)
    assert int(np.count_nonzero(label == 2)) == 2 * 3  # the white core, middle z only
    assert int(np.count_nonzero(label == 1)) == 3 * 6 * 8 - 6
    assert volume.provenance.generator == "vdbmat-image-stack"
    assert any(
        source.startswith("identity:pgm-stack:sha256:")
        for source in volume.provenance.sources
    )


def test_generator_output_runs_through_the_core_pipeline_with_a_mapping_file(
    generator: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    slices, config_path = _stack_fixture(tmp_path)
    assert generator.main(
        [str(slices), str(config_path), str(tmp_path), "cored_block"]
    ) == 0
    capsys.readouterr()

    mapping = phase0_provisional_mapping()
    write_optical_mapping(tmp_path / "mapping.optical-mapping.json", mapping)
    config = PipelineConfig(
        input_kind="direct-voxel",
        input_path="cored_block.voxels.json",
        output_path="runs/cored_block",
        mapping_path="mapping.optical-mapping.json",
        mapping_digest=mapping.digest,
    )

    result = run_pipeline(config, base_dir=str(tmp_path))
    assert all(
        stage.status.value in ("ok", "skipped") for stage in result.stages
    )
    summary = result.summary
    counts = {int(k): v for k, v in summary["material"]["counts"].items()}
    assert counts[2] == 6  # material conservation through the whole pipeline
    assert counts[1] == 3 * 6 * 8 - 6
    assert result.mapping_digest == mapping.digest


def test_undeclared_gray_value_fails(
    generator: ModuleType, tmp_path: Path
) -> None:
    slices, config = _stack_fixture(tmp_path)
    rogue = np.full((6, 8), 7, dtype=np.uint8)
    _write_pgm_p5(slices / "slice_003.pgm", rogue)
    with pytest.raises(generator.StackError, match="not declared"):
        generator.stack_to_volume(slices, config)


def test_mismatched_slice_shapes_fail(
    generator: ModuleType, tmp_path: Path
) -> None:
    slices, config = _stack_fixture(tmp_path)
    _write_pgm_p5(slices / "slice_003.pgm", np.full((4, 8), 255, dtype=np.uint8))
    with pytest.raises(generator.StackError, match="differs"):
        generator.stack_to_volume(slices, config)
