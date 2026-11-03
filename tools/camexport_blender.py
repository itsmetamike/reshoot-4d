"""Camera export for the Virtual Camera, run inside Blender (headless) by camfile.export_camera:

  blender -b --factory-startup -P camexport_blender.py -- <frames.json> <out.fbx|.usdc|.abc>

frames.json (metres, Blender's z-up world, the subject at the origin): fps, aspect, frames [{pos, rot (camera-to-world 3x3, Blender camera axes),
lens (mm on a 36 mm tall frame), fstop, focus}], source {pos, rot, lens} (the real camera at frame 0), subject (an empty at the origin).
"""
import json, sys
from pathlib import Path

import bpy
from mathutils import Matrix

argv = sys.argv[sys.argv.index("--") + 1:]
J = json.loads(Path(argv[0]).read_text()); out = Path(argv[1])
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
fps = float(J["fps"]); sc.render.fps = max(1, round(fps)); sc.render.fps_base = sc.render.fps / fps
sc.render.resolution_y = 1920; sc.render.resolution_x = max(2, round(1920 * float(J.get("aspect", 9 / 16)) / 2) * 2)
n = len(J["frames"]); sc.frame_start, sc.frame_end = 1, n


def camera(name, lens):
    cd = bpy.data.cameras.new(name); cd.sensor_fit = "VERTICAL"; cd.sensor_height = 36.0; cd.sensor_width = 36.0 * max(1.0, J.get("aspect", 1.0))
    cd.lens = lens; ob = bpy.data.objects.new(name, cd); sc.collection.objects.link(ob); return ob


def place(ob, fr):
    m = Matrix([list(r) + [p] for r, p in zip(fr["rot"], fr["pos"])] + [[0, 0, 0, 1]]); ob.matrix_world = m


subject = bpy.data.objects.new("Subject", None); subject.empty_display_type = "SPHERE"; subject.empty_display_size = 0.15
sc.collection.objects.link(subject)
if J.get("source"):
    src = camera("SourceCamera", J["source"]["lens"]); place(src, J["source"])
cam = camera("ReshootCamera", J["frames"][0]["lens"]); cam.rotation_mode = "XYZ"; sc.camera = cam
cd = cam.data; cd.dof.use_dof = any(f.get("fstop") for f in J["frames"])
for i, fr in enumerate(J["frames"]):
    f = i + 1; place(cam, fr)
    cam.keyframe_insert("location", frame=f); cam.keyframe_insert("rotation_euler", frame=f)
    cd.lens = fr["lens"]; cd.keyframe_insert("lens", frame=f)
    if cd.dof.use_dof:
        cd.dof.aperture_fstop = fr.get("fstop") or 2.8; cd.dof.focus_distance = fr.get("focus") or 3.0
        cd.dof.keyframe_insert("aperture_fstop", frame=f); cd.dof.keyframe_insert("focus_distance", frame=f)
ext = out.suffix.lower()   # every frame is keyed, so the exporters bake the exact per-frame camera
if ext == ".fbx":
    bpy.ops.export_scene.fbx(filepath=str(out), object_types={"CAMERA", "EMPTY"}, bake_anim=True, bake_anim_simplify_factor=0.0)
elif ext in (".usd", ".usdc", ".usda"):
    bpy.ops.wm.usd_export(filepath=str(out), export_animation=True, selected_objects_only=False)
elif ext == ".abc":
    bpy.ops.wm.alembic_export(filepath=str(out), start=1, end=n)
else:
    raise SystemExit(f"unsupported export: {ext}")
