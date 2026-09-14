"""Export the donut from DONA.blend as PLY surfaces (run inside Blender).

Two files are written into the output directory:

* ``donut_points.ply`` – the torus body ("Rosca") only, with its
  subdivision-surface modifier applied.  This is the fitting target.
* ``donut_scene.ply`` – every mesh object (body, icing shell, sprinkles)
  with modifiers applied and per-vertex RGB taken from each material's
  Principled BSDF base colour, for visualisation.

Usage (headless)::

    /Applications/Blender.app/Contents/MacOS/Blender -b DONA.blend \
        --python export_donut_points.py -- <output directory>

`fit_donut.py` runs this automatically when the PLY files are missing.
"""
import os
import sys

import bpy
import numpy as np

BODY_OBJECT = "Rosca"
DEFAULT_COLOUR = (0.8, 0.8, 0.8)  # Blender's material-less grey

out_dir = sys.argv[sys.argv.index("--") + 1]


def linear_to_srgb(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.0031308, 12.92 * c,
                    1.055 * np.power(c, 1 / 2.4) - 0.055)


def material_colour(obj):
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                return tuple(node.inputs["Base Color"].default_value)[:3]
    return DEFAULT_COLOUR


def evaluated_triangles(obj, depsgraph):
    """World-space vertices and triangle indices with modifiers applied."""
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    mesh.calc_loop_triangles()
    matrix = np.array(evaluated.matrix_world)
    verts = np.array([v.co[:] for v in mesh.vertices])
    verts = verts @ matrix[:3, :3].T + matrix[:3, 3]
    tris = np.array([t.vertices[:] for t in mesh.loop_triangles],
                    dtype=np.int64)
    evaluated.to_mesh_clear()
    return verts, tris


def write_coloured_ply(path, verts, colours, tris):
    vertex = np.empty(len(verts), dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = verts.T.astype(np.float32)
    rgb = np.clip(np.round(colours * 255), 0, 255).astype(np.uint8)
    vertex["red"], vertex["green"], vertex["blue"] = rgb.T
    face = np.empty(len(tris), dtype=[("n", "u1"), ("idx", "<i4", (3,))])
    face["n"] = 3
    face["idx"] = tris
    header = "\n".join([
        "ply", "format binary_little_endian 1.0",
        f"element vertex {len(verts)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        f"element face {len(tris)}",
        "property list uchar int vertex_indices",
        "end_header", ""])
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(vertex.tobytes())
        fh.write(face.tobytes())


# --- body only: fitting target -------------------------------------------
bpy.ops.object.select_all(action="DESELECT")
body = bpy.data.objects[BODY_OBJECT]
body.select_set(True)
bpy.context.view_layer.objects.active = body
bpy.ops.wm.ply_export(
    filepath=os.path.join(out_dir, "donut_points.ply"),
    export_selected_objects=True,
    apply_modifiers=True,
    export_normals=True,
    export_uv=False,
    export_colors="NONE",
    ascii_format=False,
)
print(f"Exported {BODY_OBJECT} to donut_points.ply")

# --- whole scene with material colours: visualisation --------------------
depsgraph = bpy.context.evaluated_depsgraph_get()
all_verts, all_colours, all_tris = [], [], []
offset = 0
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    verts, tris = evaluated_triangles(obj, depsgraph)
    colour = linear_to_srgb(material_colour(obj))
    all_verts.append(verts)
    all_colours.append(np.tile(colour, (len(verts), 1)))
    all_tris.append(tris + offset)
    offset += len(verts)

write_coloured_ply(os.path.join(out_dir, "donut_scene.ply"),
                   np.concatenate(all_verts), np.concatenate(all_colours),
                   np.concatenate(all_tris))
print(f"Exported {offset} coloured vertices to donut_scene.ply")
