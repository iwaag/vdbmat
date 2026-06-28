"""Load one exported VDB and render the fixed Phase 0 Cycles proof scene.

Invoke with: blender --background --python this_file.py -- MANIFEST OUTPUT_PNG
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _arguments() -> tuple[Path, Path]:
    try:
        separator = sys.argv.index("--")
        manifest, output = sys.argv[separator + 1 : separator + 3]
    except (ValueError, IndexError) as error:
        raise SystemExit("expected -- MANIFEST OUTPUT_PNG") from error
    return Path(manifest).resolve(), Path(output).resolve()


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def _world_bounds(manifest: dict[str, object]) -> tuple[Vector, Vector]:
    matrix = manifest["index_to_world_column_matrix"]
    dimensions = manifest["dimensions_xyz"]
    corners = []
    for x in (-0.5, float(dimensions[0]) - 0.5):
        for y in (-0.5, float(dimensions[1]) - 0.5):
            for z in (-0.5, float(dimensions[2]) - 0.5):
                point = Vector((x, y, z, 1.0))
                corners.append(
                    Vector(
                        tuple(
                            sum(
                                matrix[row][column] * point[column]
                                for column in range(4)
                            )
                            for row in range(3)
                        )
                    )
                )
    return (
        Vector(tuple(min(point[axis] for point in corners) for axis in range(3))),
        Vector(tuple(max(point[axis] for point in corners) for axis in range(3))),
    )


def _attribute(nodes: bpy.types.Nodes, name: str) -> bpy.types.Node:
    node = nodes.new("ShaderNodeAttribute")
    node.attribute_name = name
    return node


def main() -> None:
    manifest_path, output_path = _arguments()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vdb_path = manifest_path.parent / manifest["vdb"]
    cycles = manifest["cycles"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = int(cycles["samples"])
    scene.cycles.seed = int(cycles["seed"])
    scene.cycles.max_bounces = int(cycles["max_bounces"])
    scene.render.resolution_x = int(cycles["width"])
    scene.render.resolution_y = int(cycles["height"])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output_path)
    scene.render.film_transparent = False
    scene.world.color = (0.02, 0.02, 0.02)
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0

    volume = bpy.data.volumes.new("vbdmat-volume")
    volume.filepath = str(vdb_path)
    volume.is_sequence = False
    volume_obj = bpy.data.objects.new("vbdmat-volume", volume)
    scene.collection.objects.link(volume_obj)

    material = bpy.data.materials.new("vbdmat-cycles-volume")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    absorption = nodes.new("ShaderNodeVolumeAbsorption")
    scatter = nodes.new("ShaderNodeVolumeScatter")
    absorption.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    scatter.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    scatter.inputs["Anisotropy"].default_value = float(manifest["phase_g"])
    absorption_attribute = _attribute(nodes, "cycles_absorption")
    scattering_attribute = _attribute(nodes, "cycles_scattering")
    links.new(absorption_attribute.outputs["Fac"], absorption.inputs["Density"])
    links.new(scattering_attribute.outputs["Fac"], scatter.inputs["Density"])
    add = nodes.new("ShaderNodeAddShader")
    links.new(absorption.outputs["Volume"], add.inputs[0])
    links.new(scatter.outputs["Volume"], add.inputs[1])
    links.new(add.outputs[0], output.inputs["Volume"])
    volume.materials.append(material)

    lower, upper = _world_bounds(manifest)
    center = (lower + upper) * 0.5
    extent = upper - lower
    radius = max(extent) * 0.5
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = center + Vector((2.8, -3.5, 2.4)) * max(radius, 1e-4)
    camera_data.lens = 50.0
    _look_at(camera, center)
    scene.camera = camera

    light_data = bpy.data.lights.new("backlight", "AREA")
    light_data.energy = 1000.0
    light_data.shape = "DISK"
    light_data.size = max(2.5 * radius, 1e-3)
    light = bpy.data.objects.new("backlight", light_data)
    scene.collection.objects.link(light)
    light.location = center + Vector((0.0, 2.5, 0.5)) * max(radius, 1e-4)
    _look_at(light, center)

    bpy.context.view_layer.update()
    available = {grid.name for grid in volume.grids}
    required = {"cycles_absorption", "cycles_scattering"}
    if not required <= available:
        raise RuntimeError(f"missing VDB grids: {sorted(required - available)}")
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path.with_suffix(".blend")))
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
