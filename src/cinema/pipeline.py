"""Per-frame temperature -> RGBA rendering (FireLab roadmap, Phase 2 task 1).

EffectsPipeline.render() replaces matplotlib's Normalize+cmap step: it
normalizes against an adaptive (auto-exposure) upper bound, applies a
filmic tone curve so hot cores saturate gracefully instead of clipping,
and looks the result up in the FireLUT's black-body-with-alpha ramp.
"""

from __future__ import annotations

import time

import numpy as np

from cinema.bloom import apply_bloom
from cinema.luts import FIRE_RGBA_LUT
from cinema.noise import FLICKER_TRACK
from cinema.shimmer import HeatShimmer
from cinema.smoke import SmokeSimulator, composite_over, smoke_rgba

# Absolute (not auto-exposure-relative) ceiling below which a pixel is
# "warm gas/plume," never "still-burning flame" -- comfortably above
# smoke.py's own SOURCE_THRESHOLD_C=60 (where smoke starts accumulating
# at all) and well under this kind of dataset's real flame-core
# temperatures (300-450C, e.g. scene.py's own _flame_lean_for and
# TestCandleMarker's candle-position checks both use >150C as "clearly
# hot," and a real measured flame peak sits well above that). Used only
# to gate _suppress_haze_over_smoke below -- everywhere else in this
# module still reads the auto-exposure-normalized `t`, not raw degrees.
HAZE_TEMP_CEILING_C = 150.0

# 1/f flicker amplitude: fraction of tonemapped intensity the pink-noise
# track can add/subtract per frame -- candle-like breathing, not a
# strobing screen. Bumped slightly from the original 0.05 for a more
# visibly "alive" flame per the fire-realism pass.
FLICKER_AMPLITUDE = 0.07

# Bloom strength at hrr_intensity=1.0 (see EffectsPipeline.render).
# Bumped from the original 0.8 -- a stronger glow reads as a hotter,
# more incandescent flame instead of a flat-edged hot spot.
BLOOM_STRENGTH = 1.3

# Ambient backdrop (fire-realism pass): a very dim, warm radial falloff
# behind the fire so it reads as "floating in a dim room" rather than a
# pure black void -- centered low/wide like ambient floor-bounce light,
# composited under smoke and fire.
AMBIENT_STRENGTH = 0.05
AMBIENT_TINT = np.array([46.0, 32.0, 24.0], dtype=np.float32)  # dim warm ember-brown


def _suppress_haze_over_smoke(fire_rgba: np.ndarray, frame: np.ndarray,
                               density: np.ndarray) -> np.ndarray:
    """Damp fire_rgba's alpha wherever real smoke sits over a pixel that
    isn't actually still-burning (frame < HAZE_TEMP_CEILING_C), so that
    warm-but-not-flame gas reads as the smoke's own (grey) tint instead
    of the FireLUT's orange.

    Why absolute temperature, not alpha/opacity: a first version of this
    gated on how transparent fire_rgba's own alpha already was (protect
    near-opaque pixels, suppress near-transparent ones), reasoning that
    the true flame body is the opaque part. That works for a typical
    scenario, but a real, measured counter-example broke it: a strongly-
    ventilated flame (candles=1, vod=2/HVAC, voc=1, both vents open)
    burns at a genuinely lower peak temperature, which pulls the auto-
    exposure ceiling down with it -- and on that compressed a scale, even
    a modest 60-130C haze pixel (real smoke, nowhere near combustion)
    lands at alpha=255, fully opaque, indistinguishable from true flame
    by opacity alone. Raw degrees don't have that problem: 100C is 100C
    regardless of what the rest of the frame is doing, so it's what
    actually separates "haze the smoke should win" from "core the flame
    should keep" in every scenario, not just the typical one."""
    alpha = fire_rgba[..., 3].astype(np.float32) / 255.0
    haze = frame < HAZE_TEMP_CEILING_C
    suppress = np.where(haze, 1.0 - np.clip(density, 0.0, 1.0), 1.0)
    out = fire_rgba.copy()
    out[..., 3] = np.clip(alpha * suppress * 255.0, 0.0, 255.0).astype(np.uint8)
    return out


def _ambient_backdrop(shape: tuple) -> np.ndarray:
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    cx, cy = nx / 2.0, ny * 0.85
    dist = np.hypot((xx - cx) / (nx * 0.65), (yy - cy) / (ny * 0.65))
    falloff = np.clip(1.0 - dist, 0.0, 1.0) ** 2
    out = np.empty(shape + (4,), dtype=np.uint8)
    out[..., 0] = AMBIENT_TINT[0]
    out[..., 1] = AMBIENT_TINT[1]
    out[..., 2] = AMBIENT_TINT[2]
    out[..., 3] = (falloff * AMBIENT_STRENGTH * 255.0).astype(np.uint8)
    return out


class AutoExposure:
    """Camera-iris-style adaptive vmax: an EMA of a high percentile of
    each incoming frame, so faint early plumes stay visible and later
    flashover frames don't clip. `locked=True` freezes vmax (science-mode
    parity / manual slider control)."""

    def __init__(self, vmax_init: float, tau_frames: float = 8.0, percentile: float = 99.5):
        self.vmax = float(vmax_init)
        self._alpha = 1.0 / max(tau_frames, 1.0)
        self._percentile = percentile
        self.locked = False
        self._snap_next = False

    def snap_next(self) -> None:
        """The next update() jumps straight to that frame's own target
        instead of easing toward it over ~tau_frames calls -- used on a
        real scenario switch (see EffectsPipeline.reset_exposure), where
        the EMA would otherwise keep coasting on the *previous*
        scenario's brightness ceiling for several frames, a real,
        measured ~30% vmax swing that reads as the whole scene gradually
        dimming/brightening right after the switch instead of snapping
        to the new scenario's own level immediately."""
        self._snap_next = True

    def update(self, frame: np.ndarray) -> float:
        if self.locked:
            return self.vmax
        target = float(np.percentile(frame, self._percentile))
        if self._snap_next:
            self.vmax = target
            self._snap_next = False
        else:
            self.vmax += self._alpha * (target - self.vmax)
        return self.vmax


def filmic_tonemap(t: np.ndarray, shoulder: float = 0.6) -> np.ndarray:
    """Reinhard-style shoulder curve on already-normalized [0, 1] input:
    compresses highlights toward 1.0 gracefully instead of clipping, while
    staying close to linear at low values."""
    return t * (1.0 + t * shoulder) / (1.0 + t)


class EffectsPipeline:
    """Owns the auto-exposure state for one view cell; render() turns a
    raw temperature array into an RGBA uint8 image of the same shape."""

    def __init__(self, vmin: float, vmax_init: float):
        self.vmin = float(vmin)
        self.exposure = AutoExposure(vmax_init)
        self.last_cost_ms = 0.0
        self._flicker_i = 0
        self._smoke: SmokeSimulator = None
        self._shimmer = HeatShimmer()
        self._ambient_backdrop: np.ndarray = None

    def reset_smoke(self) -> None:
        """Drop the smoke density buffer so the next render() call starts
        it fresh (it lazily recreates when None, same as the shape-change
        branch below already relies on) -- called on a real scenario
        switch (see views.SliceView.reset_cinema_simulators) so smoke
        that accumulated against the *previous* scenario doesn't keep
        drifting/decaying on screen after the temperature field
        underneath it has already changed (e.g. a public-mode candle-
        count change leaving a residual haze at the removed candle's
        position). Flicker/shimmer state is untouched -- those track the
        image's own motion generically, not a specific scenario's
        spatial layout. Exposure is a different story (see
        reset_exposure): its vmax ceiling is a real EMA of *this*
        scenario's own brightness, and carrying the old one over is
        exactly the kind of scenario-specific staleness this method
        exists to clear -- callers reset both together (see
        views.SliceView.reset_cinema_simulators)."""
        self._smoke = None

    def reset_exposure(self) -> None:
        """Make the very next render() snap its auto-exposure ceiling
        straight to the new scenario's own first frame instead of
        coasting in from the *previous* scenario's vmax over the usual
        ~tau_frames EMA window -- called alongside reset_smoke() on a
        real scenario switch. Measured effect of skipping this: vmax
        swinging ~145 -> ~86 -> back to ~110 over the next ~10 frames
        after a 2-candle -> 1-candle switch, reading as the whole scene
        gradually dimming/brightening right after the change rather than
        the new scenario's own brightness simply appearing outright."""
        self.exposure.snap_next()

    def render(self, frame: np.ndarray, hrr_intensity: float = 1.0,
               velocity_frame: np.ndarray = None) -> np.ndarray:
        """hrr_intensity: a scenario's current HRR(t) normalized to its own
        peak (1.0 = at-or-near peak), or 1.0 (neutral) if no HRR data is
        available -- scales both the flicker amplitude and the bloom
        strength, so the glow physically tracks the real heat-release
        curve instead of being a constant cosmetic overlay.

        velocity_frame: this cell's VELOCITY data at the same timestep, or
        None -- drives the smoke layer's Tier 2 advection (see
        cinema/smoke.py); Tier 1 (fixed upward drift) is used when it's
        absent."""
        t0 = time.perf_counter()
        vmax = self.exposure.update(frame)
        span = max(vmax - self.vmin, 1e-6)
        t = np.clip((frame - self.vmin) / span, 0.0, 1.0)
        t = filmic_tonemap(t)

        flicker = FLICKER_TRACK[self._flicker_i % len(FLICKER_TRACK)]
        self._flicker_i += 1
        t = np.clip(t * (1.0 + FLICKER_AMPLITUDE * hrr_intensity * flicker), 0.0, 1.0)

        if self._smoke is None or self._smoke.buffer.shape != frame.shape:
            self._smoke = SmokeSimulator(frame.shape, ambient_c=self.vmin)
        if self._ambient_backdrop is None or self._ambient_backdrop.shape[:2] != frame.shape:
            self._ambient_backdrop = _ambient_backdrop(frame.shape)
        density = self._smoke.step(frame, velocity_frame)

        idx = (t * (len(FIRE_RGBA_LUT) - 1)).astype(np.uint8)
        fire_rgba = FIRE_RGBA_LUT[idx]
        fire_rgba = apply_bloom(fire_rgba, t, strength=BLOOM_STRENGTH * hrr_intensity)
        # Damp whatever's left of fire_rgba's alpha over real smoke that
        # isn't actually still-burning (see _suppress_haze_over_smoke's
        # own docstring for the real, measured combo that needed this) --
        # applied to the fully-composed fire layer (LUT + bloom both
        # already in) so one pass covers both an orange bloom-glow spill
        # and an orange *intrinsic* LUT color, rather than two separate,
        # partial fixes.
        fire_rgba = _suppress_haze_over_smoke(fire_rgba, frame, density)

        composited = composite_over(smoke_rgba(density), self._ambient_backdrop)
        composited = composite_over(fire_rgba, composited)
        composited = self._shimmer.warp(composited, frame, self.vmin)

        self.last_cost_ms = (time.perf_counter() - t0) * 1000.0
        return composited
