# Camera path files

This guide describes exported camera path JSON files used by the 4D camera editor.

## Envelope

An exported path uses `dropcam_path: 1`, a `state` object and, when metadata permits, `rel` and `cam` arrays corresponding one-to-one with `state.keys`.

| Field | Meaning |
| --- | --- |
| state.duration / fps | Authored shot duration in seconds and output sampling rate. |
| state.keys | Camera keys; `t` in seconds, `eye` and `target` vectors, `lens`, `roll`, `scene`, `ease`; some keys also carry aim/freeze behavior. |
| state.path | Orbit or spline interpolation. |
| state.shot_ease | `smooth` enables whole-shot camera time remapping. Other/absent uses per-key behavior. |
| state.follow | False fixes the camera in the background scene; enabled follow travels with the subject track when available. |
| state.scene_mode / speed / scene_end | Performance time mapping and end behavior. Distinct from transport preview speed. |
| state.space_dir | Capture space used to author the absolute keys; not a portable download location. |
| state.aspect | Shot aspect when present; importing onto another capture keeps that capture's aspect. |
| rel | Older subject-relative portability data. Keep for compatibility if exported. |
| cam | Camera-relative portability data, preferred for cross-capture replay when available. |

Keys store absolute capture-space eye and target values. Copying only those values to a different capture does not preserve the intended relative shot.

## The portable cam array

Each `cam` entry describes a key relative to the video's real first camera and aim distance. A source-camera solve is required to generate and use it.

| Field | Meaning in current implementation |
| --- | --- |
| az / el | Azimuth and elevation changes in degrees from the real camera's eye-to-aim direction. |
| dist | Eye-to-target distance divided by real-camera aim distance. |
| aim | Target offset in the camera-relative horizontal direction, side and up basis, divided by real-camera aim distance. |
| lens | Lens ratio relative to the real source lens. |
| roll | Roll value carried from the key. |
| source | When true, import snaps that key to the new capture's real source eye, target and lens. |

The importer uses absolute keys for the same capture. For a different capture it reconstructs keys from `cam` when lengths and metadata are valid, otherwise falls back to `rel`. It rejects another capture's path if no supported portable copy exists. The new capture's aspect is retained.

The exporter normalizes azimuth changes around the source angle. Multi-turn winding is not encoded by individual normalized angles.

## Import and export checks

Export paths from Virtual Camera and keep the JSON with the workflow and render settings.

Check matching key/portable-entry counts, finite vectors and times, positive FPS/duration, source-camera start and playback after import on a second capture. Review composition for each capture.

The Retake adapter currently consumes target poses but not every saved channel. Lens and capture-time/freeze transfer are limited as described in [Editor](EDITOR.md). Keep the path file with each render even where those channels are unused.

## Camera files (.fbx, .abc, .usd, .usda, .usdc, .usdz)

Import in the Virtual Camera also takes camera files. Blender (headless, `blender` in config) reads the file on the server and lists its cameras; with more than one you pick which. The picked camera is baked at every frame, so parents, constraints and caches are included.

The move is kept relative to the camera's own first frame and starts on this clip's real camera, fixed in the background scene (no shared scene or round trip needed). Translation uses the capture's metric scale (her height = 1.7 m, the same scale Retake uses). Roll carries over. Focal length changes become lens changes relative to the real lens. Keys use spline interpolation, one per frame, up to 240.

Focus distance and f-stop are stored on each key (`focus` in capture units, `fstop`) when the file's camera has depth of field on. Nothing renders them yet.

Known limits: Blender's Alembic importer does not read animated focal length (the pose is fine). Files need a real camera solve on the capture.

## Camera settings on keys (Camera FX)

Keys may carry `fstop` (0.7-22), `focus` (metres, or `"subject"` to follow the person), `vignette` and `grain` (0-1). A key without a setting carries it on from the keys around it. `state.zoom_aim` (`subject` or `centre`) sets where lens changes zoom to. Camera FX reads these from the path (its `camera_path` input), timed across the clip; focus moves in diopters between keys.

Lens changes: the Retake holds the move's widest lens (`retake_zoom` in path.npz) and Camera FX applies the rest as a crop (`post_zoom`, via its `retake` input), so a lens change is applied once and changing it needs no new re-shoot. Imported camera files put their focus distance (metres) and f-stop on the keys.
