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
| state.follow | False locks the camera in the setting; enabled follow travels with the subject track when available. |
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
