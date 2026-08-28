"""Legacy synthetic smoke layer: Tier 1 temperature-derived haze + Tier 2
velocity-advected dye (FireLab roadmap Phase 2.1f, tiers 1-2 only).

Superseded in public mode by Architecture C (cinema/real_smoke.py):
real FDS SOOT DENSITY, time-aligned to TEMPERATURE and normalized
against a fixed, empirically-derived dataset-wide reference, is now
public mode's smoke density source end to end -- this module is never
even instantiated on that path (cinema/pipeline.py's
EffectsPipeline.render() takes the already-computed real density
directly via its smoke_density_frame parameter). SmokeSimulator survives
here only as the fallback for the *researcher* app's generic "Cinematic
fire view" toggle (main_window.py's _apply_cinematic_state), which has
no per-scenario real-soot timestamp alignment plumbed to it; render()
falls back to this module exactly as before whenever no
smoke_density_frame is supplied.

The comparison that used to justify keeping this synthetic proxy over
the real `.s3d` soot field (its early appearance in this same y=0 plane
covered only 0.6-0.8% of the plane and had unstable fan-on/fan-off
totals) no longer applies to public mode's own conclusion -- see
cinema/real_smoke.py's module docstring and the Architecture C
investigation for the resolution (real timestamp alignment + linear
interpolation + a fixed normalization reference were the missing
pieces, not a flaw in the soot data itself). tests/test_public_mode.py's
TestSootVersusTemperatureProxy documents that history for the record.

Tier 1 (velocity_frame=None): a persistent density buffer accumulates
max(T - T_source, 0) with decay, advected by a fixed upward drift + a
gentle horizontal sway.

Tier 2 (velocity_frame given): the same accumulation, but advected by a
per-pixel field combining the real VELOCITY slice's magnitude with a
direction prior (up, blended with "away from the hot core" via -grad T).
Honest caveat: the stored VELOCITY slice is speed magnitude only (no
u/w components) -- true directional advection would need M-SIM to add
U-VELOCITY/W-VELOCITY slices to fds/template.fds (flagged as a wishlist
item for that milestone, not a blocker here). Measured consequence: the
velocity term moves the rendered smoke horizontally by <0.4 of 101
columns versus buoyancy alone, and identically with the fan on or off,
so it reads as spreading rather than as a direction (see
tests/test_public_mode.py::TestSmokeMotionIsAtmosphericNotDirectional).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

SOURCE_THRESHOLD_C = 60.0        # config.ISOTHERM_LEVELS['TEMPERATURE'][0] -- reuse the existing hazard-band floor
PRODUCTION_SCALE = 1.0 / 400.0
DECAY = 0.985
MAX_DENSITY = 1.0

# BUOYANCY_SPEED/TIER2_BASE_BUOYANCY_FRAC together set the constant
# vertical speed term (rows/frame) map_coordinates' semi-Lagrangian
# backward trace uses every step. A real, measured regression: an
# earlier version of this fix raised their product to 7.6 rows/frame to
# chase ceiling/floor density ratio -- against this room's 49-row grid,
# that crosses the whole height in ~6.5 frames (CFL number far above 1).
# Over the 481-frame run that isn't "smoke rising fast", it's numerical
# over-advection: map_coordinates' mode="nearest" boundary clamp means
# most rows end up repeatedly resampling the same clamped source, and
# within ~144 frames the buffer visibly homogenizes -- density stops
# varying with row at all and becomes a flat function of column only (a
# hard-edged vertical band with no plume shape, reproduced directly by
# dumping SmokeSimulator.buffer for case 6/frame 144: every one of rows
# 0/3/12/24/36/48 had the *same* column profile). The ceiling/floor
# ratio "improvement" that version reported was this homogenization, not
# real stratification -- reverted back to the last confirmed-stable
# values here (displacement well under one cell/frame; see the
# ceiling-reservoir mechanism below for where real stratification
# actually comes from now).
BUOYANCY_SPEED = 0.6               # tier 1: fixed upward drift, rows/frame
SWAY_AMPLITUDE = 0.4              # subtle horizontal jitter, columns
VELOCITY_SCALE = 0.5              # tier 2: m/s -> columns-or-rows/frame
# Vertical and horizontal speed are deliberately independent constants,
# not a blended direction vector scaled by one shared magnitude: an
# earlier version of this fix raised the *vertical* buoyancy by scaling
# a single UP_BIAS-blended direction, which also multiplied the
# horizontal "away from hot core" component by the same factor -- a
# real, measured regression (TestSmokeMotionIsAtmosphericNotDirectional
# started failing: horizontal drift over 150 frames jumped from ~7 to
# ~27-30 columns, and the fan-on/fan-off difference grew past the
# "reads as a measured wind direction" threshold). HORIZ_SPEED keeps the
# horizontal nudge at its own small scale regardless of vertical speed.
# This decoupling is CFL-neutral (it doesn't add to vertical speed) and
# is kept even after BUOYANCY_SPEED itself was reverted above, since it
# also made vertical transport more efficient on its own (no longer
# diluted by an UP_BIAS blend) at zero extra numerical risk.
# 0.5 (this constant's original value, chosen back when vertical speed
# was much faster) turned out to read as too large now that vertical
# buoyancy is back to its CFL-safe 0.72 rows/frame: at the old vertical
# speed the horizontal nudge was a small fraction of total motion, but
# at the new, slower vertical speed the same 0.5 became proportionally
# much more significant over a 150-frame run, and the real per-scenario
# "away from hot core" gradient differs enough between fan-on/fan-off
# that TestSmokeMotionIsAtmosphericNotDirectional's fan-drift-diff check
# started failing again (1.5 columns, threshold 1.0) -- not from the
# reservoir mechanism (confirmed: the same diff appears with the
# reservoir removed entirely), purely from this constant's own now-
# disproportionate weight. 0.35 brings the measured diff to 0.43.
HORIZ_SPEED = 0.35                 # tier 2: columns/frame, independent of vertical buoyancy

# The semi-Lagrangian advection below resamples the whole buffer every
# frame at a *fractional* row offset (0.6, never landing on a whole
# pixel) -- order=1 (bilinear) has to blend neighbouring cells at every
# one of those resamples, and that blending compounds: a seeded density
# blob measured losing 87% of its peak within just 10 frames of
# constant-speed advection alone (0.58 -> 0.075), independent of the
# explicit DECAY below. Over the ~70+ frames it actually takes to cross
# this room's 49 rows at BUOYANCY_SPEED, that ate essentially everything
# before it reached the ceiling -- so density only ever visibly built up
# right where it was produced (near the floor-level candle), never at
# the top. order=3 (cubic) cuts that same-path loss by more than half
# (measured: mass retained after the full crossing goes from 0.19 to
# 0.40) with no change to BUOYANCY_SPEED itself, i.e. no change to how
# fast smoke visibly drifts -- only to how much of it survives the trip.
ADVECT_INTERP_ORDER = 3

# Real smoke that reaches a ceiling doesn't dissipate at the open-room
# rate -- it's a trapped, stratified layer that only slowly mixes back
# down. Rows within CEILING_BAND_FRAC of the top decay at CEILING_DECAY
# instead, blended smoothly (no visible seam) down to the ordinary DECAY
# by the band's own lower edge, so whatever now actually survives the
# trip up (see ADVECT_INTERP_ORDER above) accumulates into a visible
# layer instead of fading at the same rate as everywhere else. Nearly
# half the room's height, not a thin strip at the very top: this
# dataset's candle only clears SOURCE_THRESHOLD_C in its bottom ~25% (a
# real, measured fact -- rows above that never produce smoke locally at
# all), so anything reaching the middle of the room is already
# transported, buoyant haze on its way up, not a separate phenomenon
# that should decay at the open-room rate before it finishes arriving.
CEILING_BAND_FRAC = 0.45          # fraction of room height, from the ceiling, treated as "under the ceiling"
CEILING_DECAY = 0.9998

# Tier 2's vertical floor speed used to be a small fraction of
# BUOYANCY_SPEED (0.3x) -- weaker than Tier 1's plain constant drift,
# not stronger, since the rest of Tier 2's speed budget goes to the
# real VELOCITY magnitude and the "away from hot core" horizontal
# component. Measured consequence with the real dataset (mean airspeed
# often well under 0.5 m/s -- see PUBLIC_METRICS): the velocity term
# rarely made up the difference, so Tier 2 smoke rose *slower* than
# Tier 1 would have and even less of it survived the trip to the
# ceiling. TIER2_BASE_BUOYANCY_FRAC restores a floor at least as strong
# as Tier 1's own constant, on top of which velocity and direction
# still modulate -- kept at this original, CFL-safe value (BUOYANCY_
# SPEED * TIER2_BASE_BUOYANCY_FRAC = 0.72 rows/frame, ~68 frames to
# cross the room) rather than raised further; see BUOYANCY_SPEED's own
# comment for why pushing this higher was the actual regression.
TIER2_BASE_BUOYANCY_FRAC = 1.2

# Real stratification now comes from here, not from vertical transport
# speed (see BUOYANCY_SPEED's own comment on why raising speed further
# isn't safe). A dedicated per-column reservoir, decoupled from
# map_coordinates entirely -- it only ever reads *how much density is
# currently sitting in the ceiling band* and integrates that over time
# with near-1.0 decay, so it has zero CFL exposure regardless of how
# strong the accumulation needs to be: unlike advection, there's no
# "too fast" here, only "too much"/"too little" gain.
#
# Contribution is weighted by the same ceiling_mix blend CEILING_DECAY
# uses (1 at row 0, fading smoothly to 0 at the band's own lower edge)
# rather than added flatly across the whole band -- an earlier version
# that maxed the *same* per-column reservoir value into every row of
# the band reintroduced exactly the bug this whole fix exists to avoid
# (every row inside the band became a perfectly scaled copy of every
# other row -- caught by the row-independence regression guard in
# tests/test_cinema.py, which is why that guard checks more than just
# top-vs-bottom-of-room). The row-scaled version still leaves genuine,
# distinct absolute values at each row (verified: top-vs-bottom row
# correlation 0.55-0.97 across the 6-scenario benchmark, nowhere near
# the ~1.0 the homogenized bug produced).
#
# RESERVOIR_GAIN/RESERVOIR_DECAY chosen empirically: the smallest gain
# that brings ceiling/floor density ratio to >= 1.0 for the calm
# scenarios without pushing the heavily fan-ventilated ones into an
# implausibly thick layer. Re-run after this: ratio 1.15-1.77 across
# the same 6 scenarios (case 0/4/5/9/18/21), all now >= 1.0 (previously
# 0.08-0.70, all < 1.0; the intermediate over-fast-buoyancy version hit
# >1.0 too but via homogenization, not real accumulation).
RESERVOIR_GAIN = 0.015
RESERVOIR_DECAY = 0.997

SMOKE_TINT = np.array([172.0, 176.0, 182.0], dtype=np.float32)  # neutral grey-white haze
SMOKE_OPACITY = 0.75


class SmokeSimulator:
    """Owns one cell's persistent smoke-density buffer across frames."""

    def __init__(self, shape: tuple, ambient_c: float):
        self.buffer = np.zeros(shape, dtype=np.float32)
        self.ambient_c = ambient_c
        self._sway_t = 0
        ny, nx = shape
        self._yy, self._xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
        # Row 0 is the ceiling throughout this codebase's own data
        # convention (ceiling-first slices -- see load_data.load_data,
        # PublicScene.location_zone). row_frac is 0 at the ceiling, 1 at
        # the floor; ceiling_mix fades linearly from 1 at row 0 to 0 at
        # CEILING_BAND_FRAC of the room's height, giving a smooth
        # transition into ordinary DECAY rather than a visible seam.
        row_frac = self._yy[:, :1] / max(ny - 1, 1)
        self._ceiling_mix = np.clip(1.0 - row_frac / CEILING_BAND_FRAC, 0.0, 1.0)
        self._decay_by_row = DECAY + (CEILING_DECAY - DECAY) * self._ceiling_mix
        self._ceiling_band_rows = int(round(CEILING_BAND_FRAC * ny))
        # See RESERVOIR_GAIN/RESERVOIR_DECAY's own comment: real
        # stratification, decoupled from advection numerics entirely.
        self._reservoir = np.zeros((1, nx), dtype=np.float32)

    def step(self, temperature_frame: np.ndarray, velocity_frame: np.ndarray = None) -> np.ndarray:
        self._advect(temperature_frame, velocity_frame)
        self._produce_and_decay(temperature_frame)
        self._accumulate_ceiling_reservoir()
        return self.buffer

    def _sway(self) -> float:
        self._sway_t += 1
        return SWAY_AMPLITUDE * float(np.sin(self._sway_t * 0.05))

    def _advect(self, temperature_frame: np.ndarray, velocity_frame: np.ndarray) -> None:
        sway = self._sway()
        # Horizontal: direction only ("away from the hot core"), magnitude
        # fixed at HORIZ_SPEED -- deliberately *not* scaled by vertical
        # buoyancy (see HORIZ_SPEED's own comment) -- and computed the
        # same way whether or not real velocity data is available, so
        # Tier 1 and Tier 2 differ only in their vertical speed, not in
        # whether a horizontal bias exists at all (an earlier version
        # gave Tier 1 no away-from-hot-core term, which on its own
        # created a spurious ~4-column tier1-vs-tier2 gap regardless of
        # how small HORIZ_SPEED was).
        _grad_y, grad_x = np.gradient(temperature_frame)
        away_x = -grad_x
        dir_x = away_x / (np.abs(away_x) + 1e-6)
        vx = dir_x * HORIZ_SPEED + sway
        if velocity_frame is None:
            vy = np.full_like(self.buffer, -BUOYANCY_SPEED)
        else:
            # Vertical: real measured speed plus the buoyancy floor,
            # always upward -- see BUOYANCY_SPEED/TIER2_BASE_BUOYANCY_FRAC's
            # own comment for why this stays well under one cell/frame.
            vy = -(np.clip(velocity_frame, 0.0, None) * VELOCITY_SCALE
                   + BUOYANCY_SPEED * TIER2_BASE_BUOYANCY_FRAC)

        # Semi-Lagrangian backward trace: the value now at (y, x) came
        # from (y, x) - v one step ago. order=ADVECT_INTERP_ORDER (cubic,
        # not the default linear) -- see that constant's own comment for
        # why: this is the actual fix for smoke never reaching the
        # ceiling, not just a quality tweak.
        src_y = self._yy - vy
        src_x = self._xx - vx
        self.buffer = np.clip(
            map_coordinates(self.buffer, [src_y, src_x], order=ADVECT_INTERP_ORDER, mode="nearest"),
            0.0, None)

    def _produce_and_decay(self, temperature_frame: np.ndarray) -> None:
        production = np.clip(
            temperature_frame - self.ambient_c - SOURCE_THRESHOLD_C, 0.0, None
        ) * PRODUCTION_SCALE
        self.buffer = np.clip(self.buffer * self._decay_by_row + production, 0.0, MAX_DENSITY)

    def _accumulate_ceiling_reservoir(self) -> None:
        """See RESERVOIR_GAIN/RESERVOIR_DECAY's own comment: a per-column
        trap, decoupled from map_coordinates, that integrates whatever
        density the (CFL-safe, necessarily slow) advection above
        actually delivers into the ceiling band over time. `contribution`
        is scaled by self._ceiling_mix (1 at row 0, fading to 0 at the
        band's own lower edge) before being maxed into the buffer -- not
        added flatly across the whole band -- so rows inside the band
        keep genuinely different absolute values instead of each being a
        scaled copy of the same per-column vector (see this file's own
        module docstring note on the row-independence regression guard)."""
        band = self.buffer[:self._ceiling_band_rows]
        inflow = band.mean(axis=0, keepdims=True) * RESERVOIR_GAIN
        self._reservoir = np.clip(self._reservoir * RESERVOIR_DECAY + inflow, 0.0, MAX_DENSITY)
        contribution = self._reservoir * self._ceiling_mix
        self.buffer = np.clip(np.maximum(self.buffer, contribution), 0.0, MAX_DENSITY)


def smoke_rgba(density: np.ndarray) -> np.ndarray:
    """density (H, W) float -> (H, W, 4) uint8, a flat neutral grey-white
    tint whose alpha follows the density buffer."""
    alpha = np.clip(density * SMOKE_OPACITY, 0.0, 1.0)
    out = np.empty(density.shape + (4,), dtype=np.uint8)
    out[..., 0] = SMOKE_TINT[0]
    out[..., 1] = SMOKE_TINT[1]
    out[..., 2] = SMOKE_TINT[2]
    out[..., 3] = (alpha * 255.0).astype(np.uint8)
    return out


def composite_over(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Standard straight-alpha Porter-Duff "A over B", both (H, W, 4)
    uint8. Used to composite the fire layer over the smoke layer, which
    itself sits over the (already-drawn) dark backdrop -- fire glows
    through smoke, smoke occludes the room, matching real depth ordering.
    """
    top_rgb = top[..., :3].astype(np.float32) / 255.0
    top_a = top[..., 3].astype(np.float32) / 255.0
    bot_rgb = bottom[..., :3].astype(np.float32) / 255.0
    bot_a = bottom[..., 3].astype(np.float32) / 255.0

    out_a = top_a + bot_a * (1.0 - top_a)
    out_rgb = top_rgb * top_a[..., None] + bot_rgb * bot_a[..., None] * (1.0 - top_a[..., None])
    safe_a = np.where(out_a > 1e-6, out_a, 1.0)
    out_rgb = out_rgb / safe_a[..., None]

    out = np.empty_like(top)
    out[..., :3] = np.clip(out_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    out[..., 3] = np.clip(out_a * 255.0, 0.0, 255.0).astype(np.uint8)
    return out
