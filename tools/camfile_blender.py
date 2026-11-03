"""Camera file reader for the Virtual Camera import, run inside Blender (headless) by app_routes.py:

  blender -b --factory-startup -P camfile_blender.py -- <file.fbx|.abc|.usd|.usda|.usdc|.usdz> <out.json> [camera name]

Without a camera name it lists what the file holds (cameras with their frame range and lens, plus the other objects). With one it
bakes that camera at every frame: world pose (metres, Blender's z-up world, camera looks down -z with +y up), lens, sensor and fit,
lens shift, depth of field (focus distance, f-stop) and type. Animation from parents, constraints and drivers is included, because
each frame is evaluated, not read from the camera's own keys.
"""
import json, math, sys
from pathlib import Path

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
src, out = Path(argv[0]), Path(argv[1]); want = argv[2] if len(argv) > 2 else None

bpy.ops.wm.read_factory_settings(use_empty=True)
ext = src.suffix.lower()
if ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath=str(src), use_anim=True)
elif ext == ".abc":
    bpy.ops.wm.alembic_import(filepath=str(src), as_background_job=False)
elif ext in (".usd", ".usda", ".usdc", ".usdz"):
    bpy.ops.wm.usd_import(filepath=str(src))
else:
    raise SystemExit(f"unsupported camera file: {ext} (fbx, abc, usd, usda, usdc, usdz)")

sc = bpy.context.scene
unit = sc.unit_settings.scale_length or 1.0   # metres per Blender unit after import
fps = sc.render.fps / (sc.render.fps_base or 1.0)


def frame_range(ob):
    """The frames the object animates over: its own action, an Alembic/USD cache, else the scene range."""
    lo, hi = [], []
    chain, p = [ob], ob.parent
    while p:
        chain.append(p); p = p.parent
    for o in chain:
        ad = o.animation_data
        if ad and ad.action:
            a, b = ad.action.frame_range; lo.append(a); hi.append(b)
        if o.data and getattr(o.data, "animation_data", None) and o.data.animation_data.action:
            a, b = o.data.animation_data.action.frame_range; lo.append(a); hi.append(b)
        for c in o.constraints:
            if c.type == "TRANSFORM_CACHE" and c.cache_file:
                cf = c.cache_file
                if getattr(cf, "frame_start", None) is not None:
                    lo.append(cf.frame_start); hi.append(cf.frame_end)
    if not lo:
        return sc.frame_start, sc.frame_end
    return int(math.floor(min(lo))), int(math.ceil(max(hi)))


cams = [o for o in sc.objects if o.type == "CAMERA"]
if want is None:
    info = {"file": src.name, "fps": fps, "metres_per_unit": unit, "scene_range": [sc.frame_start, sc.frame_end],
            "cameras": [{"name": o.name, "range": list(frame_range(o)), "lens": round(o.data.lens, 3), "type": o.data.type,
                         "sensor": [o.data.sensor_width, o.data.sensor_height], "fit": o.data.sensor_fit} for o in cams],
            "objects": sorted({o.type for o in sc.objects if o.type != "CAMERA"})}
    out.write_text(json.dumps(info, indent=1)); raise SystemExit(0)

ob = next((o for o in cams if o.name == want), None)
if ob is None:
    raise SystemExit(f"no camera named {want!r} in {src.name}: {[o.name for o in cams]}")
f0, f1 = frame_range(ob)
# vertical field of view needs the render aspect: an imported file sets it when it carries one, else the camera's sensor shape
W, H = sc.render.resolution_x * sc.render.pixel_aspect_x, sc.render.resolution_y * sc.render.pixel_aspect_y
frames = []
for f in range(f0, f1 + 1):
    sc.frame_set(f)
    ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())   # caches (Alembic/USD) and drivers applied
    cam = ev.data; M = ev.matrix_world
    fit = cam.sensor_fit if cam.sensor_fit != "AUTO" else ("HORIZONTAL" if W >= H else "VERTICAL")
    half = (cam.sensor_height if cam.sensor_fit == "VERTICAL" else cam.sensor_width) / 2 / cam.lens   # tan(half fov), fitted side
    tan_v = half if fit == "VERTICAL" else half * H / W
    dof = cam.dof
    frames.append({"f": f, "t": (f - f0) / fps,
                   "pos": [v * unit for v in M.translation], "rot": [list(r)[:3] for r in M.to_3x3().normalized()],
                   "lens": cam.lens, "tan_half": half, "fov_y": math.degrees(2 * math.atan(tan_v)), "shift": [cam.shift_x, cam.shift_y],
                   "dof": {"use": bool(dof.use_dof), "focus": (dof.focus_distance * unit if not dof.focus_object else
                                                               (dof.focus_object.matrix_world.translation - M.translation).length * unit),
                           "fstop": dof.aperture_fstop},
                   "type": cam.type, "ortho_scale": cam.ortho_scale * unit, "clip": [cam.clip_start * unit, cam.clip_end * unit]})
out.write_text(json.dumps({"file": src.name, "camera": ob.name, "fps": fps, "metres_per_unit": unit, "aspect": W / H,
                           "sensor": [ob.data.sensor_width, ob.data.sensor_height], "fit": ob.data.sensor_fit, "frames": frames}))
