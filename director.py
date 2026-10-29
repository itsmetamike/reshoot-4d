"""4D Director keyframe sampler. The browser (web/director.js) runs the same maths for its preview; keep them in step.

State: {"duration": s, "fps": f, "shake": 0..1, "keys": [{"t": s, "eye": [x,y,z], "target": [x,y,z], "lens": mm,
         "roll": deg, "scene": 0..1, "ease": "smooth"|"linear"|"hold"}, ...]}
Segment i (key i -> key i+1) maps output time to u in [0,1], eased by key i's ease. Eye and target follow a uniform
Catmull-Rom spline through the keys; lens, roll and scene time are interpolated linearly in the eased u (no overshoot,
so scene time stays monotone where the keys are). Lens is mm on a 36 mm tall (portrait full-frame) sensor.
Path mode "orbit" (default) interpolates the eye around the target in spherical coordinates (azimuth unwrapped key to key,
elevation, distance), so a handful of keys gives a true circle; "spline" runs Catmull-Rom on the eye position directly.
Scene time (state "scene_mode": "auto", the default for new shots) is not keyed: the capture plays at "speed" x real time,
except across segments whose start key has "freeze": true (bullet time: the camera moves, the capture stops). Past the
capture's end it ping-pongs, loops or holds ("scene_end"). Legacy states (scene_mode "keys") interpolate a per-key scene value.
"Hold" ease only affects the camera channels.
Whole-shot ease (state "shot_ease": "smooth"): one smootherstep over the camera's time from the first key to the last, with the keys
in between passed through at speed (their own ease counts as linear, "hold" excepted). Per-key smooth eases stop the camera at
every key; this gives one ease-in at the start and one ease-out at the end. Scene time is not remapped: the capture keeps real time.
Aim: a key's "aim" is "fixed" (its target point) or "head" / "body" (that point on the subject at the current scene time, from
the capture's track). In orbit mode a tracked key's spherical offset is measured from the subject at the key's own time, so
the camera travels with the subject as well as looking at it.
Rows: (scene 0..1, eye xyz, target xyz, roll deg, fov_y deg).
"""
import math

SENSOR_MM = 36.0


def ease(kind, u):
    if kind == "linear":
        return u
    if kind == "hold":
        return 0.0 if u < 1.0 else 1.0
    return u * u * (3 - 2 * u)


def lens_to_fov(mm):
    return math.degrees(2 * math.atan(SENSOR_MM / (2 * mm)))


def _cr(p0, p1, p2, p3, u):
    u2, u3 = u * u, u * u * u
    return [0.5 * (2 * b + (-a + c) * u + (2 * a - 5 * b + 4 * c - d) * u2 + (-a + 3 * b - 3 * c + d) * u3)
            for a, b, c, d in zip(p0, p1, p2, p3)]


def _shake(t, amp):
    if amp <= 0:
        return [0.0, 0.0, 0.0]
    w = 2 * math.pi
    return [amp * (0.5 * math.sin(w * 1.3 * t) + 0.3 * math.sin(w * 2.9 * t + 1.7) + 0.2 * math.sin(w * 5.1 * t + 0.4)),
            amp * (0.5 * math.sin(w * 1.1 * t + 2.1) + 0.3 * math.sin(w * 3.3 * t + 0.9) + 0.2 * math.sin(w * 4.7 * t + 2.6)),
            amp * (0.5 * math.sin(w * 0.9 * t + 4.2) + 0.3 * math.sin(w * 2.3 * t + 3.1) + 0.2 * math.sin(w * 6.1 * t + 1.2))]


def _sph(eye, tgt, basis):
    up, front, side = basis
    rel = [a - b for a, b in zip(eye, tgt)]
    r = math.sqrt(sum(x * x for x in rel)) or 1e-9
    dot = lambda u, v: sum(a * b for a, b in zip(u, v))
    return math.atan2(dot(rel, side), dot(rel, front)), math.asin(max(-1.0, min(1.0, dot(rel, up) / r))), r


def _unwrapped(keys, basis):
    out, prev = [], None
    for k in keys:
        az, el, r = _sph(k["eye"], k["target"], basis)
        if prev is not None:
            az += 2 * math.pi * round((prev - az) / (2 * math.pi))
        out.append((az, el, r)); prev = az
    return out


def _cr1(a, b, c, d, u):
    return _cr([a], [b], [c], [d], u)[0]


def scene_auto(state, keys, t):
    frozen = 0.0
    for a, b in zip(keys, keys[1:]):
        if a.get("freeze"):
            frozen += max(0.0, min(t, b["t"]) - a["t"]) if t > a["t"] else 0.0
    secs = max(float(state.get("capture_seconds", 1.0)), 1e-6)
    x = max(0.0, float(state.get("speed", 1.0)) * (t - frozen) / secs)
    end = state.get("scene_end", "pingpong")
    if end == "loop":
        return x % 1.0
    if end == "pingpong":
        x = x % 2.0
        return x if x <= 1.0 else 2.0 - x
    return min(1.0, x)


def _track_at(track, name, scene):
    pts = track[name]; x = min(1.0, max(0.0, scene)) * (len(pts) - 1); i = min(int(x), len(pts) - 2); f = x - i
    return [a + (b - a) * f for a, b in zip(pts[i], pts[i + 1])]


def _scene(state, keys, t, i, j, e):
    if state.get("scene_mode", "keys") == "auto":
        return scene_auto(state, keys, t)
    return min(1.0, max(0.0, keys[i]["scene"] + (keys[j]["scene"] - keys[i]["scene"]) * e))


def shot_time(state, keys, t):
    """Camera time under the whole-shot ease (scene time stays t)."""
    if state.get("shot_ease") != "smooth" or len(keys) < 2:
        return t
    t0, t1 = keys[0]["t"], keys[-1]["t"]
    if t <= t0 or t >= t1 or t1 - t0 < 1e-6:
        return t
    x = (t - t0) / (t1 - t0)
    return t0 + (t1 - t0) * x * x * x * (x * (6 * x - 15) + 10)


def sample(state, t, dist0=1.0, basis=None, track=None):
    keys = sorted(state["keys"], key=lambda k: k["t"])
    if not keys:
        raise ValueError("Virtual Camera: add at least one key")
    tr, t = t, shot_time(state, keys, t)   # tr: real time (scene, shake); t: camera time
    if t <= keys[0]["t"] or len(keys) == 1:
        i, u = 0, 0.0
    elif t >= keys[-1]["t"]:
        i, u = len(keys) - 2, 1.0
    else:
        i = max(j for j in range(len(keys) - 1) if keys[j]["t"] <= t)
        span = max(keys[i + 1]["t"] - keys[i]["t"], 1e-6)
        u = (t - keys[i]["t"]) / span
    j = min(i + 1, len(keys) - 1)
    kind = keys[i].get("ease", "smooth")
    e = ease("linear" if state.get("shot_ease") == "smooth" and kind != "hold" else kind, u)
    scene = _scene(state, keys, tr, i, j, e)
    K = lambda k: keys[max(0, min(len(keys) - 1, k))]

    def aim(k, sc):   # a key's look-at point at scene time sc
        name = k.get("aim", "fixed")
        return _track_at(track, name, sc) if track and name in track else k["target"]

    def key_scene(k):
        kk = keys.index(k); return _scene(state, keys, k["t"], kk, kk, 0.0)

    tgt = _cr(*(aim(K(n), scene) for n in (i - 1, i, j, j + 1)), e)
    if state.get("path", "orbit") == "orbit" and basis is not None:
        sph, prev = [], None
        for n in (i - 1, i, j, j + 1):
            k = K(n); az, el, r = _sph(k["eye"], aim(k, key_scene(k)), basis)
            sph.append([az, el, r])
        for n in range(1, 4):   # unwrap azimuth across the four control keys
            sph[n][0] += 2 * math.pi * round((sph[n - 1][0] - sph[n][0]) / (2 * math.pi))
        az, el, r = (_cr1(sph[0][m], sph[1][m], sph[2][m], sph[3][m], e) for m in range(3))
        up, front, side = basis
        eye = [tg + r * (math.cos(el) * (math.cos(az) * f + math.sin(az) * sd) + math.sin(el) * uu)
               for tg, f, sd, uu in zip(tgt, front, side, up)]
    else:
        eye = _cr(K(i - 1)["eye"], K(i)["eye"], K(j)["eye"], K(j + 1)["eye"], e)
    lerp = lambda name: keys[i][name] + (keys[j][name] - keys[i][name]) * e
    sh = _shake(tr, state.get("shake", 0.0) * 0.012 * dist0)
    eye = [a + b for a, b in zip(eye, sh)]; tgt = [a + 0.5 * b for a, b in zip(tgt, sh)]
    fl = (track or {}).get("follow")
    if fl and state.get("follow", True) and not (track and keys[i].get("aim", "fixed") in track):
        # follow: a fixed-target camera travels with her through the setting (as the source camera did); keys are relative to the middle of
        # the clip, where the capture's centre is
        d = [a - b for a, b in zip(_track_at({"f": fl}, "f", scene), _track_at({"f": fl}, "f", 0.5))]
        eye = [a + b for a, b in zip(eye, d)]; tgt = [a + b for a, b in zip(tgt, d)]
    return [scene, *eye, *tgt, lerp("roll"), lens_to_fov(lerp("lens"))]


def rows(state, dist0=1.0, basis=None, track=None):
    fps = float(state.get("fps", 24)); n = max(1, int(round(float(state["duration"]) * fps)))
    return [sample(state, i / fps, dist0, basis, track) for i in range(n)]
