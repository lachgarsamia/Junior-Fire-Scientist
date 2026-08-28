"""Real FDS SOOT DENSITY as the smoke visualization's density source
(FireLab roadmap: Architecture C -- replacing cinema/smoke.py's
synthetic SmokeSimulator for public mode, per the architecture audit and
its follow-up alignment investigation this session).

No transport model here, and deliberately so: no buoyancy, no drift, no
decay, no ceiling reservoir. The density this module hands to the
renderer is the real simulation output at the real simulation time
being displayed, linearly interpolated between the two real SOOT
DENSITY frames bracketing that time (SOOT DENSITY is recorded on its
own, denser output schedule -- ~1001 frames vs TEMPERATURE's ~481 over
the same real interval -- so frame *index* alignment would silently
pair mismatched instants; only real timestamps, from
ScenarioStore.get_times(), give a correct mapping). See the session's
architecture audit for why the previous synthetic model -- however much
its buoyancy/decay/reservoir constants were re-tuned -- kept finding a
new scenario it homogenized, misplaced (its ceiling band never reached
the real structural ceiling), or otherwise got wrong: every one of
those was a property of *inventing* a transport model, not of any
single wrong constant.
"""

from __future__ import annotations

import numpy as np

# The 99.5th percentile of every non-zero SOOT DENSITY value across all
# 24 scenarios (107,348,610 samples; computed once offline -- see the
# scan this constant is derived from, not re-run at import time or
# during playback). Deliberately a single, fixed, dataset-wide value,
# not a per-scenario or per-frame one: a per-scenario reference (e.g.
# each scenario's own max or 99th percentile) would make a mild and a
# severe scenario read as equally "full," destroying the exact
# comparability a public-mode "what changed?" exhibit depends on.
#
# Chosen over the other candidates in the same scan by directly
# measuring saturation and cross-scenario contrast at each frame's own
# real data (case 0 = a mild, 1-candle scenario; case 21 = a severe,
# 2-candle/both-vents-closed scenario; both at their own last, most-
# developed real frame):
#
#   reference   mild mean_opacity   severe mean_opacity   severe sat.%
#   p95  (24.5)   0.499                0.862                 33.6%
#   p99  (35.8)   0.342                0.665                  5.1%
#   p99.5(45.9)   0.267                0.523                  0.5%   <- chosen
#   p99.9(61.8)   0.199                0.389                  0.0%
#
# p95/p99 saturate a visually significant fraction of the severe
# scenario's own field to flat, detail-losing full opacity (33.6%/5.1%
# of cells) -- exactly the "artificially saturating most of the field"
# outcome to avoid. p99.9 avoids saturation entirely but pushes overall
# brightness down across every scenario (both means drop below their
# p99.5 values) without materially improving the severe/mild contrast
# ratio (1.96 vs 1.95). p99.5 gives the best balance: negligible
# saturation (0.5%, only at the single most-developed frame of the most
# severe scenario in the dataset) while keeping visible dynamic range,
# and preserves the mild/severe distinction just as well as the more
# conservative p99.9 choice would.
REFERENCE_DENSITY = 45.865  # mg/m3 (matches load_data.SOOT_DISPLAY_SCALE's display units)


def soot_at_time(soot_frames: np.ndarray, soot_times: np.ndarray, target_time: float) -> np.ndarray:
    """Linearly-interpolated raw SOOT DENSITY (mg/m3) at `target_time`
    (real simulation seconds), from the two real SOOT DENSITY frames
    bracketing it. Clamped to [soot_times[0], soot_times[-1]] -- never
    extrapolates beyond the real recorded interval (both TEMPERATURE and
    SOOT DENSITY cover the same [0, run_end] interval for every scenario
    in this dataset, so normal playback never hits this clamp; it exists
    for correctness at exactly the first/last frame and any future
    dataset where the two quantities' recorded intervals might not
    match exactly).

    Interpolates the physical density values themselves, before any
    normalization or alpha mapping -- see normalize_soot_density(), kept
    as a distinct, later step so the display-facing REFERENCE_DENSITY
    clamp is never itself interpolated (that would blend two different
    saturation points, not two density values)."""
    n = len(soot_times)
    if n == 0:
        raise ValueError("soot_times is empty")
    if n == 1:
        return soot_frames[0]
    t = float(np.clip(target_time, soot_times[0], soot_times[-1]))
    i = int(np.searchsorted(soot_times, t, side="right") - 1)
    i = max(0, min(i, n - 2))
    t0, t1 = soot_times[i], soot_times[i + 1]
    if t1 == t0:
        return soot_frames[i]
    f = (t - t0) / (t1 - t0)
    return soot_frames[i] * (1.0 - f) + soot_frames[i + 1] * f


def normalize_soot_density(raw_density: np.ndarray) -> np.ndarray:
    """Raw SOOT DENSITY (mg/m3) -> [0, 1] alpha-ready density, via the
    fixed dataset-wide REFERENCE_DENSITY -- the only normalization step
    in this module, applied after temporal interpolation (see
    soot_at_time), never before it."""
    return np.clip(raw_density / REFERENCE_DENSITY, 0.0, 1.0)
