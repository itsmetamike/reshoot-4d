"""Beat-synced virtual camera for a 4D capture. Pure numpy; rows = (t_src 0..1, yaw_deg, pitch_deg, dist_scale, roll_deg).

Grammar (all on the music grid):
  orbit      - yaw advances orbit_deg per bar, but in steps: each beat eases forward (camera "steps" on the kick)
  push       - on every downbeat the camera lunges in by `push` and settles back
  snare roll - beats 2 and 4 kick a small roll that decays
  bullet     - every `bullet_every` bars (and at `drop_s`) source time freezes for one bar while yaw sweeps `bullet_deg`
  time       - the capture plays at `speed` (x real time), ping-ponging so a 5 s capture fills any length of music
"""
import math

import numpy as np


def ease(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def beat_path(bpm, first_beat_s, duration_s, src_seconds, fps=24.0, orbit_deg=45.0, push=0.12, roll_deg=3.0,
              bullet_every=4, bullet_deg=180.0, drop_s=-1.0, speed=0.75, pitch_deg=6.0, dist=1.15, start_yaw=0.0):
    spb = 60.0 / bpm
    n = int(round(duration_s * fps))
    t = np.arange(n) / fps
    beat = (t - first_beat_s) / spb                      # beat index (float); negative before the first beat
    bar = np.floor(beat / 4.0)
    in_beat = beat - np.floor(beat)
    beat_in_bar = np.mod(np.floor(beat), 4)

    # stepped orbit: whole beats add orbit_deg/4, the current beat eases in over its first half
    yaw = start_yaw + (np.floor(beat) + ease(in_beat * 2.0)) * orbit_deg / 4.0

    bullets = set()
    if bullet_every > 0:
        bullets |= {int(b) for b in np.unique(bar) if b > 0 and int(b) % bullet_every == bullet_every - 1}
    if drop_s >= 0:
        bullets.add(int(math.floor((drop_s - first_beat_s) / spb / 4.0)))
    is_bullet = np.isin(bar, list(bullets))
    # bullet sweep: extra yaw that eases in across the bar; it stays added afterwards
    extra = np.zeros(n)
    for b in sorted(bullets):
        x = (beat - 4 * b) / 4.0
        extra += bullet_deg * ease(x)
    yaw = yaw + extra

    # source time: advance at `speed` except during bullet bars, then ping-pong over the capture
    dt = np.where(is_bullet, 0.0, speed / fps)
    s = np.cumsum(dt) - dt[0]
    period = 2.0 * src_seconds
    ph = np.mod(s, period)
    src = np.where(ph <= src_seconds, ph, period - ph) / src_seconds

    since_down = np.where(beat >= 0, np.mod(beat, 4.0) * spb, 99.0)
    d = dist * (1.0 - push * np.exp(-since_down / (0.35 * spb)) * (since_down < 4 * spb))
    since_beat = np.where(beat >= 0, in_beat * spb, 99.0)
    snare = np.isin(beat_in_bar, [1, 3])
    roll = np.where(snare, roll_deg * np.exp(-since_beat / (0.3 * spb)) * np.where(np.mod(bar, 2) == 0, 1, -1), 0.0)
    pitch = pitch_deg * np.sin(2 * math.pi * np.maximum(beat, 0) / 8.0)
    rows = np.stack([src, yaw, pitch, d, roll], 1)
    markers = {"bullet_bars": sorted(bullets), "bars": int(bar.max()) + 1 if n else 0}
    return rows.tolist(), markers
