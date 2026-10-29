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

The camera lane shows keys; the capture lane shows scene time and freeze bands. Shift-select keys and drag to retime them together. Whole-shot ease remaps camera time across the shot.

## Space, interpolation and time

**Locked** keeps the camera and look-at point in the setting. **With her** makes them travel with the tracked person.

Choose orbit or spline interpolation between keys. Circular orbit is available as a preset. Whole-shot ease smooths the authored path.

Bullet-time authoring holds capture time while camera time advances in the editor preview. See the Retake channel limits below.

## Retake channel limits

Retake samples camera position, target and roll. It selects source frames uniformly and uses source-camera intrinsics. Editor freezes, performance rewinds and per-key lens animation are not transferred to the generated retake.

Start the move near the real source camera. Begin with a modest arc or push; unseen regions can change in generated footage.

## Portable paths

Export a JSON path and import it on another capture with a source-camera solve. The `cam` array describes keys relative to the real first camera. Review composition after import. See [Path format](PATH_FORMAT.md).
