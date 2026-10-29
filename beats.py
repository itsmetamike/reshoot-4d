"""Tempo + beat phase from an audio excerpt, numpy only (ComfyUI's venv has no librosa).

Onsets come from spectral flux in the 300-2000 Hz band: that band holds the kick's click, and fitting on it avoids the
rolling sub-bass that drags low-band onset fits late (found on the Berlin NTM retime). Tempo = autocorrelation peak of
the onset envelope in [bpm_min, bpm_max]; phase = comb filter over one beat period. Returns the first beat at/after `start`.
"""
import subprocess

import numpy as np

SR, HOP, WIN = 22050, 256, 2048


def _load(path, start, dur):
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(dur), "-i", path, "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(pcm, np.float32)


def onset_envelope(x, lo=300, hi=2000):
    n = 1 + max(0, len(x) - WIN) // HOP
    frames = np.lib.stride_tricks.sliding_window_view(x, WIN)[::HOP][:n] * np.hanning(WIN)
    mag = np.abs(np.fft.rfft(frames, axis=1))
    f = np.fft.rfftfreq(WIN, 1 / SR); band = (f >= lo) & (f <= hi)
    lm = np.log1p(mag[:, band] * 10)
    flux = np.maximum(lm[1:] - lm[:-1], 0).sum(1)
    flux = np.concatenate([[0], flux])
    flux -= np.convolve(flux, np.ones(16) / 16, "same")          # remove slow trend
    return np.maximum(flux, 0)


def _comb(env, fps, period):
    """Best phase and mean onset strength on a beat grid of this period."""
    t_end = len(env) / fps
    t = np.arange(len(env)) / fps
    phases = np.linspace(0, period, 160, endpoint=False)
    means = [np.interp(np.arange(ph, t_end, period), t, env).mean() for ph in phases]
    i = int(np.argmax(means))
    return phases[i], means[i]


def detect(path, start=0.0, dur=30.0, bpm_min=90.0, bpm_max=180.0):
    x = _load(path, start, dur)
    env = onset_envelope(x)
    fps = SR / HOP
    ac = np.correlate(env, env, "full")[len(env) - 1:]
    lags = np.arange(len(ac)) / fps
    ok = (lags >= 60 / bpm_max) & (lags <= 60 / bpm_min)
    lag = np.argmax(np.where(ok, ac, -np.inf))
    # parabolic refinement of the peak for sub-frame tempo
    if 0 < lag < len(ac) - 1:
        a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]; lag = lag + 0.5 * (a - c) / (a - 2 * b + c)
    period = lag / fps
    # metrical errors (2/3, 3/4, 1/2 of the real tempo) are common on techno; keep the multiple whose beat grid hits hardest
    cands = [period / m for m in (1, 1.5, 2, 4 / 3, 2 / 3, 0.75, 0.5) if bpm_min <= 60 * m / period <= bpm_max]
    period = max(cands, key=lambda p: _comb(env, fps, p)[1])
    bpm = 60 / period
    # phase from 150-300 Hz (kick body above the sub): on the Berlin tracks this sat within ~20-65 ms of the hand-checked grid,
    # where 40-150 Hz was pulled by rolling sub-bass and 300-2000 Hz by off-beat hats. The app adds nudge buttons for the rest.
    phase, _ = _comb(onset_envelope(x, 150, 300), fps, period)
    phase = (phase + WIN / 2 / SR) % period          # frame stamps are window starts; onsets sit near the window centre
    return {"bpm": round(float(bpm), 2), "first_beat_s": round(float(start + phase), 3), "period_s": round(float(period), 4)}


if __name__ == "__main__":
    import json, sys
    print(json.dumps(detect(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.0)))
