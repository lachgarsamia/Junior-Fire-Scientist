"""Smoke layer: Tier 1 temperature-derived haze + Tier 2 velocity-advected
dye (FireLab roadmap Phase 2.1f, tiers 1-2 only). Tier 3 (reading the
unused .s3d real soot data) is a separate, timeboxed spike -- not
implemented here.

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

Why not the real SOOT DENSITY (Tier 3), even though it is on disk
-----------------------------------------------------------------
The `.s3d` volumetric soot *is* parsed, cached and available through
ScenarioStore (SliceKey('SOOT DENSITY', 1, 0, plane_pos=0.0), ~6-9 ms
warm). It was measured against this proxy on both fan scenarios before
being rejected as the visual source:

  * it covers only 0.6-0.8% of the y=0 plane -- a ~5-column thread
    directly above the candle (cols 91-95, rows 30-48 of 49x101);
  * it shows no ceiling layer at all, so it would contradict the
    smoke-layer narration and leave a visitor nothing to look at;
  * its fan-on/fan-off totals are unstable (an order of magnitude apart
    mid-run, near-equal by the end), so it discriminates the experiment
    worse than the proxy does.

The same comparison *validates* this module: 94-98% of the cells where
real soot exists are also flagged by the temperature threshold below, so
the proxy is a superset that agrees with the measurement everywhere the
measurement exists. tests/test_public_mode.py::TestSootVersusTemperatureProxy
re-checks those numbers, and fails if a future re-run makes the real
soot field dense enough to reconsider.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

SOURCE_THRESHOLD_C = 60.0        # config.ISOTHERM_LEVELS['TEMPERATURE'][0] -- reuse the existing hazard-band floor
PRODUCTION_SCALE = 1.0 / 400.0
DECAY = 0.985
MAX_DENSITY = 1.0

BUOYANCY_SPEED = 0.6              # tier 1: fixed upward drift, rows/frame
SWAY_AMPLITUDE = 0.4              # subtle horizontal jitter, columns
VELOCITY_SCALE = 0.5              # tier 2: m/s -> columns-or-rows/frame
UP_BIAS = 0.55                    # tier 2: blend weight of "straight up" vs "away from hot core"

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

    def step(self, temperature_frame: np.ndarray, velocity_frame: np.ndarray = None) -> np.ndarray:
        self._advect(temperature_frame, velocity_frame)
        self._produce_and_decay(temperature_frame)
        return self.buffer

    def _sway(self) -> float:
        self._sway_t += 1
        return SWAY_AMPLITUDE * float(np.sin(self._sway_t * 0.05))

    def _advect(self, temperature_frame: np.ndarray, velocity_frame: np.ndarray) -> None:
        sway = self._sway()
        if velocity_frame is None:
            vy = np.full_like(self.buffer, -BUOYANCY_SPEED)
            vx = np.full_like(self.buffer, sway)
        else:
            grad_y, grad_x = np.gradient(temperature_frame)
            away_y, away_x = -grad_y, -grad_x  # points away from the hot core
            norm = np.hypot(away_y, away_x) + 1e-6
            dir_y, dir_x = away_y / norm, away_x / norm
            dir_y = dir_y * (1.0 - UP_BIAS) - UP_BIAS  # blend in a constant "straight up" bias
            dnorm = np.hypot(dir_y, dir_x) + 1e-6
            dir_y, dir_x = dir_y / dnorm, dir_x / dnorm
            speed = np.clip(velocity_frame, 0.0, None) * VELOCITY_SCALE + BUOYANCY_SPEED * 0.3
            vy, vx = dir_y * speed, dir_x * speed
            vx = vx + sway

        # Semi-Lagrangian backward trace: the value now at (y, x) came
        # from (y, x) - v one step ago.
        src_y = self._yy - vy
        src_x = self._xx - vx
        self.buffer = map_coordinates(self.buffer, [src_y, src_x], order=1, mode="nearest")

    def _produce_and_decay(self, temperature_frame: np.ndarray) -> None:
        production = np.clip(
            temperature_frame - self.ambient_c - SOURCE_THRESHOLD_C, 0.0, None
        ) * PRODUCTION_SCALE
        self.buffer = np.clip(self.buffer * DECAY + production, 0.0, MAX_DENSITY)


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
