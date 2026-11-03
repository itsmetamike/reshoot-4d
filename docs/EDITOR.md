# The 4D camera editor

Virtual Camera is a 4D camera editor inside ComfyUI. Direct a camera over the moving capture, inspect the path and save it with the workflow.

## Start a shot

Open a current Capture in Virtual Camera. Start from the real camera at the first frame. Pick a preset (arc left or right, push in, crane up, bullet time, orbit left or right, zoom in or out, or the face move), then adjust eye position, target, lens, roll and key time. Switch between camera view for composition and editor view for seeing the path. Save back to the node before running Retake.

The capture is the editor preview and a reusable asset. Retake generates the video that feeds the final LTX 2.3 pass.

## Navigation and selection

| Action | Control |
| --- | --- |
| Orbit editor view | Drag viewport or view gizmo |
| Pan | Shift-drag or middle drag |
| Zoom | Mouse wheel |
| Front / right / top | Numpad 1 / 3 / 7 |
| Opposite views | Ctrl with those numpad keys; gizmo also exposes back / left / bottom |
| Recenter editor | Period or numpad decimal |
| Toggle camera/editor view | C |
| Select multiple keys | Shift+click in viewport or timeline |
| Move / rotate selected group | G / R in editor context |
| Snap moves and turns | Snap button, or hold Ctrl while dragging to flip it |

Camera navigation and key transforms depend on focus. Click the viewport before using shortcuts.

## Keys and editing

| Action | Control |
| --- | --- |
| Key the camera at the playhead | I |
| Key what the editor view sees | V in editor view |
| Delete key | X in editor view, Delete or Backspace |
| Duplicate key | Shift+D |
| Copy / paste key | Ctrl+C / Ctrl+V (Cmd on Mac) |
| Undo / redo | Ctrl+Z / Ctrl+Shift+Z or Ctrl+Y (Cmd on Mac) |
| Cancel a drag or close a menu | Esc |

## Timeline and transport

Use start/end (Home / End), frame step (Left / Right) and previous/next-key (Shift+Left / Shift+Right) to inspect exact moments. Space toggles playback. J/K/L provide reverse, stop and forward shuttle. Loop and preview speed help review a move; preview speed is separate from the capture's authored performance speed.

The editor has two clocks. **Shot time** is the position in your new camera move (the camera lane, with its keys). **Capture time** is the moment of the original performance shown at that point (the capture lane); freeze bands mark where the performance pauses while the camera keeps moving. Shift-select keys and drag to retime them together. Whole-shot ease remaps camera time across the shot.

## Space, interpolation and time

**Fixed in background** keeps the camera and look-at point in place in the background scene. **Follows subject** makes them travel with the tracked person.

Choose orbit or spline interpolation between keys. Circular orbit is available as a preset. Whole-shot ease smooths the authored path.

Bullet-time authoring holds capture time while camera time advances in the editor preview. See the Retake channel limits below.

## Camera settings

Each key has a **Camera** section: **Aperture** (f/0.7 to f/22), **Focus** (a distance in metres, or `subject` to follow the person), **Vignette** and **Grain** (0 to 1). Leave a field blank and the value carries on from the keys around it, so a rack focus is two keys with different focus and an aperture pull is two keys with different apertures. Focus moves between keys in diopters, like a focus pull. **Shot > Camera FX > Zoom aims at** chooses where lens changes zoom to: the subject (kept in frame) or the centre (a true lens).

These settings and the keys' lens are applied by the Camera FX node after the reshoot, so changing them needs no new Retake. A lens wider than the Retake was rendered at needs new picture: queue the Retake again.

## Focus preview, lens character and shutter

The camera view previews depth of field from the keys' aperture and focus: points away from the focus plane grow and thin out like bokeh (a rough guide; Camera FX renders the real thing). Turn it off in **Shot > Camera FX > Preview focus**. **Alt+click** a point in the camera view to set the focus of the key at the playhead to that distance.

**Shot > Camera FX** also holds the shot's lens character, applied by Camera FX: **Distortion** (+ barrel, - pincushion), **Chromatic ab.**, **Bloom**, **Halation** and **Anamorphic** (oval bokeh, 1 = round), and the **Shutter** angle that Finish turns into motion blur (180 = film look). Blank = off.

## Camera files

**Import** also takes `.fbx`, `.abc`, `.usd`, `.usda`, `.usdc` and `.usdz`. Blender reads the file in the background (install it and set `blender` in `config.local.json` if it isn't on PATH). If the file holds several cameras, pick one. The camera is baked at every frame (parents, constraints and caches included) and placed so its first frame sits on the clip's real camera, locked in the setting, at real scale (her height = 1.7 m). Roll and focal length carry over; focus distance and f-stop land on the keys. Long files are thinned to at most 240 keys.

If the file holds several cameras, **All cameras: save each as a path** downloads one path file per camera (multicam) and loads the first; import the others when you want them.

**Export** saves the move as a path (`.json`) or as an animated camera: `.usdc` (pose, lens, focus and f-stop), `.fbx` (pose and lens) or `.abc` (pose). It is in metres, z-up, with the subject at the origin, and includes the real source camera (`SourceCamera`) and a `Subject` marker. Re-importing an export gives the same move. Exported focus and f-stop change linearly between keys; key easing and whole-shot ease apply to the pose and lens only.

Limits: Alembic files don't carry animated focal length or focus through Blender (pose only); FBX has no f-stop; only import files you trust (a USD file can reference other files on the machine running ComfyUI).

## Retake channel limits

Retake samples camera position, target and roll. It selects source frames uniformly and uses source-camera intrinsics. For a lens change, Retake renders the whole move at its widest field of view and Camera FX crops those frames to make the zoom, so a zoom or dolly zoom needs no extra render (a long zoom crops in, so it softens a little). Editor freezes and performance rewinds are not transferred to the generated retake.

Start the move near the real source camera. Begin with a modest arc or push; unseen regions can change in generated footage.

## Portable paths

Export a JSON path and import it on another capture with a source-camera solve. The `cam` array describes keys relative to the real first camera. Review composition after import. See [Path format](PATH_FORMAT.md).
