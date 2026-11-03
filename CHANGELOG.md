# Changelog

## 0.1.1

- **Camera-only edits reuse the render.** Changing aperture, focus, vignette, grain, lens character, shutter or a lens inside the move's widest one no longer re-renders: Retake and the Photoreal Pass keep their results and return them when the move renders the same, so only Camera FX and Finish do new work.
- **Virtual Camera:** Minimise is off when a capture can't open (it used to throw an error).
- **Docs:** imported cameras' focus is in metres and Camera FX renders it.

## 0.1.0

First release.

- **Capture, Virtual Camera, Retake, Photoreal Pass:** reshoot a video of a person along a new camera move, with LTX 2.3 rendering every final frame.
- **Camera files:** the Virtual Camera imports cameras from `.fbx`, `.abc` and `.usd` (Blender, optional). Pick a camera; it starts on the source camera at approximate real scale (assuming a 1.7 m person) with its lens, roll, focus and aperture.
- **Camera settings per key:** aperture, focus (metres or the subject), vignette and grain, plus where zooms aim.
- **Camera FX:** applies the keys' lens and camera settings after the reshoot (depth of field from metric depth with matte-sharp edges, zoom, roll, vignette, grain), so changing them needs no new render.
- **Focus preview and Alt+click focus** in the camera view.
- **Lens character** per shot: distortion, chromatic aberration, bloom, halation, anamorphic bokeh; depth and matte passes.
- **Camera export** to `.fbx` / `.usd` / `.abc`, and multicam files saved as one path per camera.
- **Finish:** standard frame rates (RIFE), motion blur from a shutter angle, delivery resize, source audio, and the camera path saved beside each output.
