"""The public canvas: one SliceView, always cinematic, no chrome.

This is a *thin* wrapper on purpose. All the fire, smoke, ember,
shimmer, bloom and 30 Hz sub-frame interpolation work already exists in
cinema/ and views.py and is driven by real FDS data -- the public
experience's visual quality comes from turning that on and getting the
scientific chrome (colorbar, ticks, titles) out of the way, not from a
second rendering path.

Frames are pulled from ScenarioStore exactly the way the Live Viewer
pulls them (a warm get() is ~1-6 ms), so switching scenarios mid-
experiment is effectively instant for this dataset, whose fields are all
already on disk in the .npy cache.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle
from PyQt5 import QtWidgets

from public import i18n
from registry import get_quantity
from schematic import room_overlay_geometry, _CANDLE_X, ROOM_X, ROOM_Z
from slice_key import SliceKey, DEFAULT_SLICE_KEY
from summary_stats import _read_hrr_csv
from views import SliceView, CINEMA_BG

logger = logging.getLogger(__name__)

VELOCITY_KEY = SliceKey("VELOCITY", 1, 0)

# The scene renders into the left SCENE_WIDTH_FRAC of this widget, not
# the full width -- the remaining right-hand strip is real, reserved
# space the axes never draws into at all, for the thermometer to dock
# in without ever overlapping scene content (the room's right wall,
# where the second vent lives, is the domain's own right edge -- i.e.
# the *previous* full-bleed axes' right edge too, leaving no genuine
# gap there for anything to dock into; several rounds of shrinking the
# vent glyph and nudging the thermometer's anchor offset never fixed
# that, because there was no slack to nudge into). _strip_chrome uses
# this to position the axes; widget_fraction_for and probe_at both use
# it too, to keep data<->pixel conversion consistent with wherever the
# axes actually is on screen -- getting these three out of sync is
# exactly the "probe mismapping" bug this session already fixed once
# (see _strip_chrome's own docstring).
SCENE_WIDTH_FRAC = 0.76

# Candle body drawn at the real burner location (schematic._CANDLE_X,
# the floor). Sized small relative to the 1.0 x 0.48 m domain -- an
# anchor for "the fire starts here", not a second flame; the actual fire
# is still the cinema pipeline's rendering of the real temperature field.
_CANDLE_WIDTH_M = 0.028
_CANDLE_HEIGHT_M = 0.026
_WICK_HEIGHT_M = 0.012
# A small schematic flame sits right on the wick tip -- same idea as
# schematic.py's SchematicWidget flame (three warm-to-hot layers plus a
# soft glow), scaled to this widget's physical units instead of pixels.
# This is what makes the candle read as "burning" rather than "an unlit
# object beside the real fire": the layered flame's own base overlaps
# the wick tip, and the FDS-rendered heat field (already bright there)
# shows through the glow, visually fusing the two.
#
# Sized to reach well up into where the real (data-driven) hot plume
# starts reading as fire-coloured -- a first pass at this (0.030 m, a
# tight glow) left a visible gap that read as "a small toy candle next
# to a separate, bigger fire" rather than one continuous flame (caught
# in live testing, not a hypothetical). Taller layers plus a wider,
# softer glow close that gap without changing what either layer
# represents: the layers are still the same "here's the candle" prop,
# the glow still just blends its edge into whatever real colour sits
# above it.
_FLAME_HEIGHT_M = 0.052
_FLAME_LAYERS = (   # (height fraction, colour, lift fraction) outer -> core
    (1.00, "#D93415", 0.00),
    (0.70, "#FF8A1E", 0.06),
    (0.40, "#FFDD57", 0.12),
)
# Tap tolerance around a candle's (x, z) anchor -- generous enough for a
# child's finger, small enough to stay clearly "the candle" rather than
# "the whole left half of the room".
_CANDLE_TAP_RADIUS_M = 0.05

# Fan housing + blades (Design Review §6 Priority 1: the fan should read
# as a physical object at rest, not an effect that only exists while
# active). Sized against the same 1.0 x 0.48 m domain the candle/flame
# constants above are scaled to.
#
# Blades are thin spoke *lines*, not filled wedges: this view's axes use
# aspect="auto" (views.py's imshow_kwargs) so a data-space shape is
# stretched by whatever the widget's own pixel aspect happens to be.
# Solid kite-shaped blades survived that fine individually but merged
# into one dart/star silhouette once stretched -- caught in a live
# 800x600 screenshot, not a hypothetical. Four thin spokes in a "+"
# read as a fan under any stretch; three solid wedges did not.
_FAN_HOUSING_RADIUS_M = 0.032
_FAN_BLADE_LENGTH_M = 0.026
_FAN_N_BLADES = 4
# Purely decorative, like _jitter_flame's own flicker (see its
# docstring) -- FDS does not simulate blade RPM, so the spin is a
# symbolic "this is on" state, not a claimed measurement, the same way
# the flame's wobble is not read off the HRR curve. Real per-frame
# velocity already drives the separate activity glow (_jitter_vent_
# activity), which *is* tied to a measured quantity.
_FAN_SPIN_RADIANS_PER_FRAME = 0.5

# The second vent's own object: a small louvered flap, not a powered
# fan (voc has no HVAC state, just open/closed -- schematic._VOC_STATES),
# so it has no blades to spin. Two discrete states rather than a
# per-frame animation: closed = flat slats flush with the frame
# (blocking), open = slats tilted with visible gaps between them.
#
# Deliberately smaller than the fan housing: this vent's real position
# (x=0.86-0.94, near the room's far/right wall) sits right where the
# thermometer docks (PublicScene.room_wall_anchor is the same wall, at
# the domain's own right edge -- there is no "outside the wall" margin
# at this resolution, so the thermometer is always pinned as far right
# as its own width allows and the two are always close). Physical
# position is never adjusted to dodge that (it would misrepresent the
# real vent location); only the rendered size is reduced, twice now --
# a first pass still touched the thermometer's gold ring with no gap in
# an 800x600 screenshot.
_VENT2_FRAME_HALF_WIDTH_M = 0.011
_VENT2_SLAT_LENGTH_M = 0.018
_VENT2_N_SLATS = 3
_VENT2_CLOSED_ANGLE = 0.0                 # flush, horizontal
_VENT2_OPEN_ANGLE = math.pi / 5           # tilted open


def _flame_vertices(cx: float, base_z: float, height: float, width_ratio: float = 0.42) -> list:
    """A rounded teardrop outline, `height` above (cx, base_z) -- the data-
    coordinate equivalent of schematic.py's QPainterPath flame shape."""
    w = height * width_ratio
    return [
        (cx, base_z + height),
        (cx + w * 0.55, base_z + height * 0.55),
        (cx + w, base_z + height * 0.12),
        (cx + w * 0.32, base_z),
        (cx - w * 0.32, base_z),
        (cx - w, base_z + height * 0.12),
        (cx - w * 0.55, base_z + height * 0.55),
    ]


def _fan_blade_endpoints(cx: float, cz: float, angle: float, length: float) -> tuple:
    """(xs, ys) for a single spoke from the hub (cx, cz) outward at
    `angle` radians -- feeds a Line2D's set_data the same way
    _flame_vertices feeds a Polygon's set_xy."""
    dx, dz = math.cos(angle), math.sin(angle)
    return ([cx, cx + dx * length], [cz, cz + dz * length])


def _vent2_slat_endpoints(cx: float, cz: float, row: int, angle: float, length: float) -> tuple:
    """(xs, ys) for one horizontal louvre slat of the second vent,
    `row` steps above/below the frame centre (cz), tilted by `angle`
    radians off horizontal -- 0 (closed) is flush, _VENT2_OPEN_ANGLE
    (open) shows a gap to the slat above/below it."""
    rz = cz + row * (_VENT2_FRAME_HALF_WIDTH_M * 0.75)
    half = length / 2
    dx, dz = math.cos(angle), math.sin(angle)
    return ([cx - dx * half, cx + dx * half], [rz - dz * half, rz + dz * half])


class PublicScene(QtWidgets.QWidget):
    """Owns one SliceView and the data lookups that feed it.

    `store` is a ScenarioSource (real or demo), `manifest` the scenario
    entry list (may be empty in demo mode -- the scene degrades to no
    room outline and neutral HRR rather than failing).
    """

    def __init__(self, store, manifest: list, fps: int, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = manifest or []
        self._fps = fps
        self._case_index = None
        self._quantity_key = DEFAULT_SLICE_KEY
        self._temperature = None   # (frames, h, w) for the current case
        self._velocity = None      # same, or None if unavailable
        self._hrr_cache = {}
        # Static candle body/wick patches for the current scenario --
        # not part of SliceView's animated-artist set (they never change
        # frame to frame), so they're tracked here and baked into the
        # blit background on load, not redrawn every tick.
        self._candle_patches: list = []
        self._candle_signature: Optional[tuple] = None
        # The flame layers/glow (a subset of _candle_patches) are the only
        # part of the candle that flickers -- tracked separately so
        # _jitter_flame() can update just those, and so they can be
        # unregistered from SliceView's animated set before removal (see
        # _draw_candles). Each entry carries what its own jitter formula
        # needs to recompute geometry from scratch every call.
        #
        # Deliberately driven from show_frame()'s own per-frame call, not
        # an independent QTimer: a separate timer's blit_update() has to
        # redraw the *entire* animated-artist set (heatmap image included,
        # not just the flame -- restore_region() only recovers the frozen
        # background, so anything not explicitly redrawn on top would
        # revert to the load-time frame), measured at ~15 ms per tick.
        # Piggybacking on the render that already happens every real
        # frame costs nothing extra -- and it means the flame correctly
        # freezes along with everything else when playback is paused,
        # rather than visibly flickering over a frozen scene.
        self._flame_layers_anim: list = []   # (patch, cx, base_z, base_height)
        self._flame_glows: list = []         # (patch, base_radius)
        # A brief extra-amplitude boost applied on top of the normal
        # flicker -- decays over the next few real frames. Purely a "the
        # child touched this" acknowledgement (Phase 3 section 1); never
        # changes what the flame *represents*, only how it moves for a
        # moment.
        self._flame_pulse_frames_left = 0
        # Phase 12 section 2: a small activity glow at the real vent
        # position, visible only while the fan is ON -- driven every
        # frame by the real per-frame velocity magnitude (never a
        # direction; this dataset only supports magnitude), the same
        # animated-extra piggyback _jitter_flame already uses so it costs
        # nothing beyond the redraw show_frame() was already doing.
        self._vent_activity_patch = None
        self._vent_activity_base_r = 0.0
        # The fan itself: a housing (static, baked into the background
        # like the candle body/wick) plus blades (animated like the flame
        # layers) -- present at the vent position in every scenario that
        # has one, regardless of on/off, so it reads as a physical object
        # at rest rather than an effect that only exists while active
        # (Design Review §6 Priority 1). _fan_signature is the vent
        # position last drawn at, the same redundant-redraw guard
        # _candle_signature gives _draw_candles.
        self._fan_housing_patch = None
        self._fan_blades: list = []
        self._fan_blade_angle = 0.0
        self._fan_signature: Optional[tuple] = None
        # The second vent (voc, "vent2"): a frame + louvre slats, all
        # static -- no activity glow and no per-frame spin, since voc has
        # no HVAC state to drive one (schematic._VOC_STATES is just
        # open/closed). Only redrawn when the position or open/closed
        # state actually changes (_vent2_signature covers both, unlike
        # _fan_signature which is position alone -- the fan's own state
        # is shown by whether it spins, this vent's only by slat angle).
        self._vent2_frame_patch = None
        self._vent2_slats: list = []
        self._vent2_signature: Optional[tuple] = None
        # Hot/Cold's two simultaneous markers -- separate scatter artists
        # from SliceView.hover_highlight (which the researcher app's
        # Context Panel hover also uses), so this never touches shared
        # behaviour. Created lazily on first use, reused after.
        self._hot_marker = None
        self._cool_marker = None
        # Phase 5: the temperature trail's own numbered markers (a
        # scatter ring + a text label per tap), and the baseline "ghost"
        # contour outline. Both static overlays baked into the blit
        # background once per change, the same way the candle patches
        # already are -- matplotlib has no cheap per-frame "update in
        # place" primitive for either (see views.py's own isotherms).
        self._trail_markers: list = []   # [(scatter, text), ...]
        # Phase 9 section 7: a thin static line joining the *current*
        # scenario's own trail points in tap order -- "where I measured",
        # never drawn across a scenario switch (a preserved dim "before"
        # point never joins it; see add_trail_marker's `dim` gate), so it
        # can never read as implying airflow between two conditions.
        self._trail_points: list = []    # [(x, z), ...] current-scenario taps only
        self._trail_line = None
        self._ghost_contour = None
        # Phase 9 section 3: the vent's own drawn line gets a brief
        # boosted-linewidth flash on a direct tap (see pulse_vent) -- the
        # base width is captured lazily the first time so the restore
        # never hardcodes a value that could drift from views.py's own
        # _ROOM_VENT_LW tuning.
        self._vent_pulse_base_lw = None

        self.view = SliceView(self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view.widget())

    # -- setup ----------------------------------------------------------
    def load_case(self, case_index: int, quantity_key: SliceKey = None) -> int:
        """Point the scene at a scenario. Returns its frame count.

        Safe to call repeatedly: the first call builds the plot, later
        ones just swap the data. Cinematic mode is enabled once, on the
        first load, and never turned off -- this scene has no science
        mode to fall back to.
        """
        key = quantity_key or self._quantity_key
        self._case_index = case_index
        self._quantity_key = key

        self._temperature = np.asarray(self._store.get(case_index, key))
        self._velocity = self._load_velocity(case_index)

        info = get_quantity(key.quantity)
        if self.view.ax is None:
            extent = self._extent_for(case_index, key)
            self.view.init_plot(
                self._temperature[0], cmap=info.cmap, interpolation="bilinear",
                vmin=info.vmin, vmax=float(info.slider_default),
                colorbar_label="", extent=extent)
            self._strip_chrome()
            self.view.set_cinematic_mode(
                True, vmin=info.vmin, vmax_init=float(info.slider_default))
            # Direction-free flow glyphs. The arrow quiver the research
            # views use infers its direction from a temperature gradient
            # (the signed U/W components are gated), and a child pointing
            # at an arrow would be reading an inference as a measurement.
            # Size still tracks the real measured speed, so the fan
            # contrast is unchanged.
            self.view.set_flow_mode("activity")
        self.view.set_room_outline(self._room_outline_for(case_index))
        self._draw_candles(case_index)
        self._draw_fan(case_index)
        self._update_vent_activity(case_index)
        self._draw_vent2(case_index)
        self.clear_probe()
        return int(self._temperature.shape[0])

    def _draw_candles(self, case_index: int) -> None:
        """One candle body + wick per real burner in this scenario, at
        the actual FDS burner x-position (schematic._CANDLE_X) and the
        floor (schematic.ROOM_Z[0]) -- a visual anchor for "the fire
        starts here", not a second competing flame; the flame itself is
        still the cinema pipeline's rendering of the real temperature
        field. Count comes from the manifest's own `candles` factor
        (0 -> 1 burner, 1 -> 2 -- schematic.py's own convention), never
        assumed.
        """
        entry = self._entry(case_index)
        # Only meaningful on the y-normal side view this study's geometry
        # is defined against -- same gate _room_outline_for uses.
        if entry is None or self._quantity_key.direction != 1:
            signature = None
        else:
            n_candles = 2 if entry.candles == 1 else 1
            cx_mid = sum(_CANDLE_X) / 2
            span = _CANDLE_X[1] - _CANDLE_X[0]
            xs = ([cx_mid] if n_candles == 1
                  else [cx_mid - span * 0.35, cx_mid + span * 0.35])
            signature = tuple(xs)

        # A scenario switch that doesn't change the candle count (e.g.
        # toggling the fan) would otherwise pay a full canvas redraw for
        # visually identical patches -- measured at ~10 ms, on top of the
        # cinema pipeline's own per-frame cost, right on the interactive
        # path a visitor's tap takes. Checked *before* touching any
        # existing patch, so the early return leaves the current (still
        # correct) candles exactly as they were.
        if signature == self._candle_signature:
            return
        self._candle_signature = signature

        for patch, *_ in self._flame_layers_anim:
            self.view.remove_animated_extra(patch)
        for patch, *_ in self._flame_glows:
            self.view.remove_animated_extra(patch)
        self._flame_layers_anim = []
        self._flame_glows = []
        for patch in self._candle_patches:
            patch.remove()
        self._candle_patches = []

        if signature is None:
            self.view.canvas.capture_background()
            return

        z0 = ROOM_Z[0]
        for cx in xs:
            body = Rectangle(
                (cx - _CANDLE_WIDTH_M / 2, z0), _CANDLE_WIDTH_M, _CANDLE_HEIGHT_M,
                facecolor="#F4E8D8", edgecolor="#B8A888", linewidth=0.8, zorder=9)
            wick_top = z0 + _CANDLE_HEIGHT_M + _WICK_HEIGHT_M
            wick = Line2D(
                [cx, cx], [z0 + _CANDLE_HEIGHT_M, wick_top],
                color="#3A2E22", linewidth=1.4, zorder=9)
            self.view.ax.add_patch(body)
            self.view.ax.add_line(wick)
            self._candle_patches += [body, wick]
            self._candle_patches += self._draw_flame(cx, wick_top)
        # Static patches aren't in SliceView's animated-artist set, so a
        # full draw+recapture is what makes them show up at all -- the
        # same reason set_room_outline's siblings (the LineCollections)
        # don't need this: those live in the animated set and blit every
        # tick, these are baked into the background once per scenario.
        self.view.canvas.capture_background()

    def _draw_flame(self, cx: float, base_z: float) -> list:
        """A small layered flame (glow + three warm-to-hot polygons)
        rooted at the wick tip -- schematic.py's SchematicWidget draws
        the same three-layer idea with QPainter; this is the matplotlib
        equivalent, in physical (x, z) data units, drawn on top of the
        cinema pipeline's own flame rendering (zorder 9) so the two
        visually fuse into one fire rather than sitting side by side."""
        patches = []
        glow_r = _FLAME_HEIGHT_M * 1.15
        glow = Circle((cx, base_z + _FLAME_HEIGHT_M * 0.5), glow_r,
                      facecolor="#FF7A18", edgecolor="none", alpha=0.22, zorder=8)
        self.view.ax.add_patch(glow)
        patches.append(glow)
        self.view.add_animated_extra(glow)
        self._flame_glows.append((glow, glow_r))
        for height_frac, color, lift_frac in _FLAME_LAYERS:
            height = _FLAME_HEIGHT_M * height_frac
            lift = _FLAME_HEIGHT_M * lift_frac
            layer = Polygon(_flame_vertices(cx, base_z + lift, height),
                            closed=True, facecolor=color, edgecolor="none", zorder=10)
            self.view.ax.add_patch(layer)
            patches.append(layer)
            self.view.add_animated_extra(layer)
            self._flame_layers_anim.append((layer, cx, base_z + lift, height))
        return patches

    def _jitter_flame(self, index: int) -> None:
        """A subtle, continuous flame flicker -- purely decorative (the
        real fire is still the cinema pipeline's own rendering of the
        measured temperature field; this only jitters the small schematic
        anchor drawn on top of it). A deterministic function of the frame
        index (two out-of-phase sine waves, not randomness) rather than a
        wall-clock timer: replaying the same frame always looks the same,
        and -- more importantly -- it means this costs nothing beyond the
        redraw show_frame() was already going to do. An independent timer
        was tried first and measured at ~15 ms per tick, because
        blit_update() has to redraw the *entire* animated-artist set
        (restore_region() only recovers the frozen background, so the
        heatmap image itself would revert to the load-time frame if it
        weren't redrawn too) -- piggybacking on the tick that redraws
        everything anyway is free by comparison.
        """
        if not self._flame_layers_anim and not self._flame_glows:
            return
        boost = 0.0
        if self._flame_pulse_frames_left > 0:
            boost = 0.22 * (self._flame_pulse_frames_left / self._FLAME_PULSE_FRAMES)
            self._flame_pulse_frames_left -= 1
        wobble = (1.0 + boost + 0.07 * math.sin(index * 0.7)
                  + 0.04 * math.sin(index * 1.9 + 1.3))
        for patch, cx, base_z, base_height in self._flame_layers_anim:
            patch.set_xy(_flame_vertices(cx, base_z, base_height * wobble))
        glow_wobble = 1.0 + boost + 0.05 * math.sin(index * 1.1 + 2.0)
        for patch, base_r in self._flame_glows:
            patch.set_radius(base_r * glow_wobble)

    # How many real frames a tap's flame pulse lasts -- a handful of
    # frames at the ~24 fps cinema rate is well under a second, a
    # deliberate "I felt that" flash rather than a lingering effect.
    _FLAME_PULSE_FRAMES = 8

    def _draw_fan(self, case_index: int) -> None:
        """A small housing + blades at the real vent position, drawn
        whenever a scenario has one to draw at all -- unlike the activity
        glow below, this does not depend on on/off state (Design Review
        §6 Priority 1): a fan that only exists in the ON scenes reads as
        an effect, not an object. The housing is static (baked into the
        background like the candle body/wick); the blades sit in
        SliceView's animated set so _jitter_vent_activity can turn them.

        Same redundant-redraw guard _draw_candles uses (a signature
        check before touching any existing patch): the vent position is
        the same across every scenario on this plane, so in practice this
        only ever does real work once, on the first load.
        """
        position = self.vent_marker_position(case_index)
        if position == self._fan_signature:
            return
        self._fan_signature = position
        if self._fan_housing_patch is not None:
            self._fan_housing_patch.remove()
            self._fan_housing_patch = None
        for blade in self._fan_blades:
            self.view.remove_animated_extra(blade)
            blade.remove()
        self._fan_blades = []
        if position is None:
            self.view.canvas.capture_background()
            return
        vx, vz = position
        # A ring, not a filled disc -- a solid housing sat on top of the
        # blades' own colour and read as a heavy button rather than a
        # fan's outer frame (same live-screenshot check as the blade
        # shape above).
        housing = Circle((vx, vz), _FAN_HOUSING_RADIUS_M,
                         facecolor="#232B38", edgecolor="#8B96A8",
                         linewidth=1.4, alpha=0.85, zorder=7)
        self.view.ax.add_patch(housing)
        self._fan_housing_patch = housing
        self._fan_blade_angle = 0.0
        for k in range(_FAN_N_BLADES):
            angle = k * (2 * math.pi / _FAN_N_BLADES)
            xs, ys = _fan_blade_endpoints(vx, vz, angle, _FAN_BLADE_LENGTH_M)
            blade = Line2D(xs, ys, color="#C3CCD9", linewidth=2.4,
                           solid_capstyle="round", zorder=9)
            self.view.ax.add_line(blade)
            self.view.add_animated_extra(blade)
            self._fan_blades.append(blade)
        self.view.canvas.capture_background()

    def _draw_vent2(self, case_index: int) -> None:
        """The second vent's own object: a frame + louvre slats at the
        real voc opening, drawn whenever a scenario has one -- same
        "always present" reasoning _draw_fan gives for the fan (Design
        Review §6 Priority 1 generalizes to any tappable device here, not
        just the powered one). All static: voc has only open/closed
        states (schematic._VOC_STATES), no HVAC speed to animate, so
        unlike the fan's blades these never need a per-frame update --
        only a redraw when the position or open/closed state changes.

        Same redundant-redraw guard _draw_fan/_draw_candles use, keyed on
        (position, is_open) rather than position alone, since this is
        what has to change to justify touching the patches.
        """
        entry = self._entry(case_index)
        position = self.vent2_marker_position(case_index)
        is_open = entry is not None and entry.voc == 0
        signature = (position, is_open)
        if signature == self._vent2_signature:
            return
        self._vent2_signature = signature
        if self._vent2_frame_patch is not None:
            self._vent2_frame_patch.remove()
            self._vent2_frame_patch = None
        for slat in self._vent2_slats:
            slat.remove()
        self._vent2_slats = []
        if position is None:
            self.view.canvas.capture_background()
            return
        vx, vz = position
        frame = Rectangle(
            (vx - _VENT2_FRAME_HALF_WIDTH_M, vz - _VENT2_FRAME_HALF_WIDTH_M),
            _VENT2_FRAME_HALF_WIDTH_M * 2, _VENT2_FRAME_HALF_WIDTH_M * 2,
            facecolor="#232B38", edgecolor="#8B96A8", linewidth=1.4, zorder=7)
        self.view.ax.add_patch(frame)
        self._vent2_frame_patch = frame
        angle = _VENT2_OPEN_ANGLE if is_open else _VENT2_CLOSED_ANGLE
        color = "#7DD3FC" if is_open else "#C3CCD9"
        for row in range(-1, _VENT2_N_SLATS - 1):
            xs, ys = _vent2_slat_endpoints(vx, vz, row, angle, _VENT2_SLAT_LENGTH_M)
            slat = Line2D(xs, ys, color=color, linewidth=2.2,
                         solid_capstyle="round", zorder=9)
            self.view.ax.add_line(slat)
            self._vent2_slats.append(slat)
        # Static like the fan housing/candle body -- baked into the
        # background once per state change, not part of the per-frame
        # animated set.
        self.view.canvas.capture_background()

    def _update_vent_activity(self, case_index: int) -> None:
        """A small activity glow at the vent's real position -- present
        only while the fan is actually ON (entry.vod == 2), matching
        "quiet when OFF" (Phase 12 section 2). Recreated only when the
        on/off state changes, the same signature-check idea _draw_
        candles already uses to skip redundant redraws."""
        entry = self._entry(case_index)
        position = self.vent_marker_position(case_index)
        on = entry is not None and entry.vod == 2
        if not on or position is None:
            if self._vent_activity_patch is not None:
                self.view.remove_animated_extra(self._vent_activity_patch)
                self._vent_activity_patch.remove()
                self._vent_activity_patch = None
            return
        vx, vz = position
        if self._vent_activity_patch is not None:
            self._vent_activity_patch.center = (vx, vz)
            return
        self._vent_activity_base_r = 0.035
        patch = Circle((vx, vz), self._vent_activity_base_r,
                      facecolor="#7DD3FC", edgecolor="none", alpha=0.2, zorder=8)
        self.view.ax.add_patch(patch)
        self.view.add_animated_extra(patch)
        self._vent_activity_patch = patch

    def _jitter_vent_activity(self, index: int, vel) -> None:
        """The vent's own "this thing is doing something" -- driven by
        the real per-frame mean velocity magnitude (never a direction;
        _VELOCITY_ACTIVITY_SCALE is just a display normalization against
        the real ~0.5 m/s this dataset's own fan-on scenarios measure,
        not an invented threshold). Piggybacks on the same show_frame()
        tick _jitter_flame already rides for free -- no independent
        timer, and the patch simply doesn't exist while the fan is OFF,
        so there is nothing to update (and nothing drawn) then."""
        self._spin_fan_blades()
        if self._vent_activity_patch is None:
            return
        speed = float(vel.mean()) if vel is not None else 0.0
        intensity = max(0.0, min(1.0, speed / self._VELOCITY_ACTIVITY_SCALE))
        pulse = 0.5 + 0.5 * math.sin(index * 0.6)
        self._vent_activity_patch.set_radius(
            self._vent_activity_base_r * (1.0 + 0.6 * intensity * pulse))
        self._vent_activity_patch.set_alpha(0.12 + 0.28 * intensity)

    _VELOCITY_ACTIVITY_SCALE = 0.5

    def _spin_fan_blades(self) -> None:
        """Turn the blades while the fan is ON, hold them still while
        OFF -- the same "does the activity patch exist" proxy for on/off
        _jitter_vent_activity's own docstring already relies on, so this
        never needs its own copy of the entry.vod check."""
        if not self._fan_blades or self._fan_signature is None:
            return
        if self._vent_activity_patch is not None:
            self._fan_blade_angle += _FAN_SPIN_RADIANS_PER_FRAME
        vx, vz = self._fan_signature
        for k, blade in enumerate(self._fan_blades):
            angle = self._fan_blade_angle + k * (2 * math.pi / _FAN_N_BLADES)
            xs, ys = _fan_blade_endpoints(vx, vz, angle, _FAN_BLADE_LENGTH_M)
            blade.set_data(xs, ys)

    def pulse_flame(self) -> None:
        """Trigger a brief extra-amplitude flicker -- the candle's own
        "I felt that" acknowledgement for a tap (see PublicExperience.
        _on_candle_tapped). Purely decorative; the next _jitter_flame()
        call (already happening every real frame) picks it up for free."""
        self._flame_pulse_frames_left = self._FLAME_PULSE_FRAMES

    def candle_hit(self, x: float, z: float) -> bool:
        """Whether a tapped physical point (x, z) lands on a candle --
        the burner body/wick/flame, not the surrounding air -- so a tap
        there can be answered with "this is the fire source" instead of
        a bare temperature reading. None/empty signature (candles not
        drawn for this scenario/plane) never matches."""
        if not self._candle_signature:
            return False
        z0 = ROOM_Z[0]
        top = z0 + _CANDLE_HEIGHT_M + _WICK_HEIGHT_M + _FLAME_HEIGHT_M
        if not (z0 <= z <= top):
            return False
        return any(abs(x - cx) <= _CANDLE_TAP_RADIUS_M for cx in self._candle_signature)

    def vent_marker_position(self, case_index: int) -> Optional[tuple]:
        """Physical (x, z) at the fan/HVAC vent's real ceiling opening --
        the "vod" vent from schematic.room_overlay_geometry, the same
        factor the Fan explore control drives (see experiments.py). None
        off the y-normal plane or with no manifest entry, the same gate
        _room_outline_for and _draw_candles use."""
        entry = self._entry(case_index)
        if entry is None or self._quantity_key.direction != 1:
            return None
        geometry = room_overlay_geometry(entry.door, entry.vod, entry.voc)
        (x0, z0, x1, _z1), _state = geometry["vents"][0]
        return ((x0 + x1) / 2, z0)

    def vent2_marker_position(self, case_index: int) -> Optional[tuple]:
        """Physical (x, z) at the second (candle-side) vent's real
        ceiling opening -- the "voc" vent from schematic.room_overlay_
        geometry, the same factor the "vent2" explore control drives
        (see experiments.py). Same gating as vent_marker_position, just
        against geometry["vents"][1] instead of [0]."""
        entry = self._entry(case_index)
        if entry is None or self._quantity_key.direction != 1:
            return None
        geometry = room_overlay_geometry(entry.door, entry.vod, entry.voc)
        (x0, z0, x1, _z1), _state = geometry["vents"][1]
        return ((x0 + x1) / 2, z0)

    def room_wall_anchor(self, case_index: int) -> Optional[tuple]:
        """Physical (x, z) just outside the room's own real right-hand
        wall (ROOM_X[1]) at mid-height -- lets a Qt overlay widget (the
        thermometer) dock beside the actual room geometry instead of a
        fixed pixel offset from the window edge, the same idea
        vent_marker_position already uses for the fan label. None off
        the y-normal plane or with no manifest entry, same gate that
        one uses."""
        entry = self._entry(case_index)
        if entry is None or self._quantity_key.direction != 1:
            return None
        return (ROOM_X[1], (ROOM_Z[0] + ROOM_Z[1]) / 2)

    # Tap tolerance for the vent -- same physical scale as the candle's
    # own tap radius, making the vent a real tappable object in the scene
    # rather than a decorative label (Phase 3 section 1/2).
    _VENT_TAP_RADIUS_M = 0.06

    def vent_hit(self, x: float, z: float, case_index) -> bool:
        """Whether a tapped physical point (x, z) lands on the fan/HVAC
        vent -- see PublicExperience._on_overlay_tapped, which toggles
        the fan when this is true instead of treating it as a plain air
        reading."""
        position = self.vent_marker_position(case_index)
        if position is None:
            return False
        vx, vz = position
        return (abs(x - vx) <= self._VENT_TAP_RADIUS_M
                and abs(z - vz) <= self._VENT_TAP_RADIUS_M)

    def vent2_hit(self, x: float, z: float, case_index) -> bool:
        """Same idea as vent_hit, against the second (voc) vent -- see
        PublicExperience._on_overlay_tapped, which toggles the "vent2"
        explore control when this is true."""
        position = self.vent2_marker_position(case_index)
        if position is None:
            return False
        vx, vz = position
        return (abs(x - vx) <= self._VENT_TAP_RADIUS_M
                and abs(z - vz) <= self._VENT_TAP_RADIUS_M)

    def pulse_vent(self, vent_index: int = 0) -> None:
        """A brief boosted-linewidth flash on one of the vent's own drawn
        lines -- its "I felt that" acknowledgement for a direct tap (see
        PublicExperience._on_vent_tapped/_on_vent2_tapped), the same idea
        as the candle's pulse_flame(). A one-shot flash rather than a
        per-frame effect: unlike the flame, the vent segments don't
        otherwise change between ticks, so there's nothing to piggyback
        the boost onto -- the caller restores it with reset_vent_width()
        after a short delay (PublicExperience._defer(), not a new timer
        type).

        room_vents is one LineCollection holding both vent segments
        (index 0 = vod/fan, 1 = voc/vent2), so the boosted width is set
        as a per-segment array with only `vent_index` raised -- a plain
        scalar set_linewidth would have pulsed both vents on either tap.
        """
        if self.view.ax is None or self.view.room_vents is None:
            return
        if self._vent_pulse_base_lw is None:
            widths = self.view.room_vents.get_linewidths()
            self._vent_pulse_base_lw = widths[0] if len(widths) else 4.0
        n = len(self.view.room_vents.get_segments())
        widths = [self._vent_pulse_base_lw] * n
        if 0 <= vent_index < n:
            widths[vent_index] = self._vent_pulse_base_lw * 1.8
        self.view.room_vents.set_linewidth(widths)
        self.view.canvas.capture_background()
        self.view.canvas.blit_update(self.view._animated_artists())

    def reset_vent_width(self) -> None:
        if (self.view.ax is None or self.view.room_vents is None
                or self._vent_pulse_base_lw is None):
            return
        self.view.room_vents.set_linewidth(self._vent_pulse_base_lw)
        self.view.canvas.capture_background()
        self.view.canvas.blit_update(self.view._animated_artists())

    def location_zone(self, x: float, z: float) -> str:
        """Stable, language-independent zone id for a probed point
        ("flame"/"ceiling"/"floor"/"middle") -- used for game logic that
        needs to compare *where* a point is (e.g. the mystery game's
        ceiling-vs-floor tracking), which must keep working the same way
        regardless of the current UI language. See location_phrase() for
        the translated, human-readable version of the same judgement."""
        if self.candle_hit(x, z):
            return "flame"
        top, bottom = ROOM_Z[1], ROOM_Z[0]
        span = top - bottom
        if z >= top - span * 0.15:
            return "ceiling"
        if z <= bottom + span * 0.10:
            return "floor"
        return "middle"

    def location_phrase(self, x: float, z: float) -> str:
        """A short, honest description of *where* a probed point
        physically is -- computed from the real room geometry
        (schematic.ROOM_Z) and the real candle position, never a
        decorative label. Used to caption a reading with something more
        grounded than "here" (e.g. "near the ceiling" for a point in the
        smoke layer, "at the flame" for the candle itself)."""
        zone = self.location_zone(x, z)
        return i18n.tr({
            "flame": "location_at_flame",
            "ceiling": "location_near_ceiling",
            "floor": "location_near_floor",
            "middle": "location_middle_of_room",
        }[zone])

    def widget_fraction_for(self, x: float, z: float) -> Optional[tuple]:
        """Inverse of probe_at's pixel->physical mapping: a physical
        (x, z) -> this widget's own (fraction_x, fraction_y_from_top),
        for placing a Qt overlay element aligned with real FDS geometry
        (e.g. a fan-state label sitting at the actual vent). None outside
        the plotted extent.

        The x fraction is scaled by SCENE_WIDTH_FRAC: the axes only
        occupies that much of the widget's width (see _strip_chrome), so
        a data fraction of 1.0 (the room's own right wall) has to land at
        widget fraction SCENE_WIDTH_FRAC, not 1.0, or every real-geometry
        marker (the vent labels, this widget's own thermometer anchor)
        would sit past the axes' actual right edge, out in the reserved
        thermometer column."""
        if self.view._extent is None:
            return None
        x0, x1, z0, z1 = self.view._extent
        if x1 == x0 or z1 == z0:
            return None
        return (SCENE_WIDTH_FRAC * (x - x0) / (x1 - x0), 1.0 - (z - z0) / (z1 - z0))

    def _strip_chrome(self) -> None:
        """Hide every scientific affordance: colorbar, axis frame, title.
        Done here rather than by changing SliceView's defaults so the
        research views are untouched."""
        self.view.colorbar.ax.set_visible(False)
        self.view.ax.set_xticks([])
        self.view.ax.set_yticks([])
        for spine in self.view.ax.spines.values():
            spine.set_visible(False)
        # Full bleed vertically, SCENE_WIDTH_FRAC horizontally -- the fire
        # should reach the top/bottom/left edges of the screen, not sit in
        # a plot box with margins, but the right SCENE_WIDTH_FRAC..1 strip
        # is deliberately real reserved space for the thermometer (see
        # SCENE_WIDTH_FRAC's own comment), never drawn into.
        # subplots_adjust() alone can't get either the old full bleed or
        # this: init_plot()'s fig.colorbar(fraction=0.04, pad=0.02) already
        # shrank this axes' position directly (colorbar's own make_axes
        # sets it explicitly, bypassing the subplot grid subplots_adjust
        # recomputes from), and hiding the colorbar above never gives that
        # width back -- it leaves a permanent blank strip on the right.
        # Only an explicit set_position() overrides an explicitly-set
        # position.
        self.view.ax.set_position([0, 0, SCENE_WIDTH_FRAC, 1])
        # That reserved strip is outside the axes' own bounds, so it
        # paints with the *figure's* background, not the axes' CINEMA_BG
        # (set by set_cinematic_mode, called right after this) -- left at
        # its research-view default (MplCanvas.PLOT_BG, white) it showed
        # up as a stark white column behind the thermometer. Matched to
        # the same near-black the axes itself uses so the reserved column
        # reads as "part of this dark screen", not a rendering glitch.
        self.view.canvas.fig.set_facecolor(CINEMA_BG)
        self.view.canvas.capture_background()

    def _load_velocity(self, case_index: int):
        try:
            return np.asarray(self._store.get(case_index, VELOCITY_KEY))
        except Exception as e:  # noqa: BLE001 - velocity is an enhancement, never fatal
            logger.info("velocity unavailable for case %s (%s); "
                        "smoke will use its fixed-drift tier", case_index, e)
            return None

    def _extent_for(self, case_index: int, key: SliceKey):
        try:
            extent = self._store.get_extent(case_index, key)
        except Exception:  # noqa: BLE001 - geometry is a nice-to-have
            return None
        return tuple(extent) if extent is not None else None

    def _room_outline_for(self, case_index: int):
        entry = self._entry(case_index)
        if entry is None or self._quantity_key.direction != 1:
            return None
        return room_overlay_geometry(entry.door, entry.vod, entry.voc)

    def _entry(self, case_index: int):
        return next((e for e in self._manifest if e.case_index == case_index), None)

    # -- playback -------------------------------------------------------
    def show_frame(self, index: int) -> None:
        """Render frame `index`, with the lookahead frame and HRR-derived
        bloom intensity the cinema pipeline wants."""
        if self._temperature is None:
            return
        n = self._temperature.shape[0]
        i = max(0, min(index, n - 1))
        nxt = self._temperature[i + 1] if i + 1 < n else None
        vel = None
        if self._velocity is not None and i < self._velocity.shape[0]:
            vel = self._velocity[i]
        self._jitter_flame(i)
        self._jitter_vent_activity(i, vel)
        self.view.show_frame(self._temperature[i], velocity_frame=vel,
                             next_frame=nxt, bloom_intensity=self._hrr_intensity(i))

    def _hrr_intensity(self, index: int) -> float:
        """Current HRR normalized to this scenario's own peak, so the
        flicker and bloom track the real heat-release curve. 1.0 when
        there is no CSV -- same neutral fallback the Live Viewer uses."""
        cached = self._hrr_cache.get(self._case_index)
        if cached is None:
            entry = self._entry(self._case_index)
            data = _read_hrr_csv(entry.path) if entry else None
            cached = data if data is not None else ()
            self._hrr_cache[self._case_index] = cached
        if cached == ():
            return 1.0
        times, hrr_kw = cached
        peak = float(np.max(hrr_kw)) if len(hrr_kw) else 0.0
        if peak <= 0:
            return 1.0
        current = float(np.interp(index / self._fps, times, hrr_kw))
        return max(0.15, min(1.5, current / peak))

    # -- measurements the overlay reads --------------------------------
    def frame_count(self) -> int:
        return int(self._temperature.shape[0]) if self._temperature is not None else 0

    def peak_temperature_at(self, index: int) -> float:
        """Hottest cell in the current frame, C -- a real measurement,
        not an interpretation."""
        if self._temperature is None:
            return 0.0
        i = max(0, min(index, self._temperature.shape[0] - 1))
        return float(self._temperature[i].max())

    def mean_temperature_at(self, index: int) -> float:
        """Frame-average temperature, C. This -- not the peak -- is what
        actually responds to the fan, since the flame core is roughly the
        same temperature in every scenario."""
        if self._temperature is None:
            return 0.0
        i = max(0, min(index, self._temperature.shape[0] - 1))
        return float(self._temperature[i].mean())

    def mean_airspeed_at(self, index: int) -> float:
        """Frame-average |v|, m/s, or 0.0 if velocity is unavailable."""
        if self._velocity is None:
            return 0.0
        i = max(0, min(index, self._velocity.shape[0] - 1))
        return float(self._velocity[i].mean())

    def max_airspeed_at(self, index: int) -> float:
        if self._velocity is None:
            return 0.0
        i = max(0, min(index, self._velocity.shape[0] - 1))
        return float(self._velocity[i].max())

    # -- tap-to-inspect ---------------------------------------------------
    def probe_at(self, widget_pos) -> Optional[tuple]:
        """The real measured temperature at a tapped point, or None
        outside the plotted data.

        `widget_pos` is a QPoint in this scene's own widget-pixel space
        (top-left origin) -- the overlay sitting on top forwards its own
        tap position unchanged, which is valid because the overlay, this
        scene, and its canvas all fill the exact same rectangle (see
        PublicExperience's StackAll layout).

        The pixel -> physical mapping here is deliberately *not*
        matplotlib's own transform/event pipeline (which needs a real
        QMouseEvent delivered to the canvas, and the overlay sits above
        it and would have to give that up first). Instead it uses the
        two geometric facts PublicScene itself guarantees: `_strip_chrome`
        always sets the axes to (0,0)-(SCENE_WIDTH_FRAC,1) of the figure,
        so a widget-fraction position maps to the known physical `extent`
        directly once rescaled by SCENE_WIDTH_FRAC -- no devicePixelRatio
        or transform-API version assumptions involved. A tap past
        SCENE_WIDTH_FRAC lands in the reserved thermometer column, not on
        the scene, and returns None the same way a tap outside the
        widget's own bounds already did.
        """
        if self._temperature is None or self.view._extent is None:
            return None
        canvas = self.view.canvas
        width, height = canvas.width(), canvas.height()
        if width <= 0 or height <= 0:
            return None
        frac_x = (widget_pos.x() / width) / SCENE_WIDTH_FRAC
        frac_y = 1.0 - (widget_pos.y() / height)   # Qt top-left -> plot bottom-left
        if not (0.0 <= frac_x <= 1.0 and 0.0 <= frac_y <= 1.0):
            return None
        x0, x1, z0, z1 = self.view._extent
        x = x0 + frac_x * (x1 - x0)
        z = z0 + frac_y * (z1 - z0)
        value = self.probe_value_at(x, z)
        if value is None:
            return None
        return (x, z, value)

    def probe_value_at(self, x: float, z: float) -> Optional[float]:
        """The real measured value at a physical (x, z), moving the tap
        ring there too. Used both by probe_at (a fresh tap) and by a
        pinned "watch this spot" location being resampled on every new
        frame -- neither ever interpolates or invents a value between
        cells; both go through SliceView.value_at()."""
        if self._temperature is None:
            return None
        value = self.view.value_at(x, z)
        if value is None:
            return None
        self.view.set_hover_highlight((x, z))
        self.view.canvas.blit_update(self.view._animated_artists())
        return float(value)

    # -- Phase 2: mini-games and cross-scenario comparison ---------------
    # Tap tolerance for "found the hottest place" -- generous enough for a
    # child's finger, on the same physical scale as the candle's own tap
    # radius (_CANDLE_TAP_RADIUS_M).
    _HOTTEST_TAP_RADIUS_M = 0.06

    def _extreme_point_at(self, index: int, mode: str) -> Optional[tuple]:
        """Real (x, z, value) of the single hottest or coolest cell in
        this frame -- the actual measured max/min, never a hardcoded
        screen position. Used only to judge a "find the hottest/coolest
        place" guess (see hottest_point_at/coolest_point_at); nothing
        about this is drawn (the fire rendering already *is* the real
        field)."""
        if self._temperature is None or self.view._extent is None:
            return None
        i = max(0, min(index, self._temperature.shape[0] - 1))
        frame = self._temperature[i]
        finder = np.argmax if mode == "hottest" else np.argmin
        row, col = np.unravel_index(finder(frame), frame.shape)
        x0, x1, z0, z1 = self.view._extent
        n_z, n_x = frame.shape
        x = x0 + (col / max(n_x - 1, 1)) * (x1 - x0)
        z = z1 - (row / max(n_z - 1, 1)) * (z1 - z0)
        return (x, z, float(frame[row, col]))

    def hottest_point_at(self, index: int) -> Optional[tuple]:
        return self._extreme_point_at(index, "hottest")

    def coolest_point_at(self, index: int) -> Optional[tuple]:
        return self._extreme_point_at(index, "coolest")

    def _extreme_guess_is_close(self, x: float, z: float, index: int, mode: str) -> bool:
        point = self._extreme_point_at(index, mode)
        if point is None:
            return False
        target_x, target_z, _value = point
        return (abs(x - target_x) <= self._HOTTEST_TAP_RADIUS_M
                and abs(z - target_z) <= self._HOTTEST_TAP_RADIUS_M)

    def hottest_guess_is_close(self, x: float, z: float, index: int) -> bool:
        """Whether a tap at physical (x, z) counts as "found it" --
        within a real physical tolerance of the actual measured hottest
        cell, not a pixel-distance guess."""
        return self._extreme_guess_is_close(x, z, index, "hottest")

    def coolest_guess_is_close(self, x: float, z: float, index: int) -> bool:
        return self._extreme_guess_is_close(x, z, index, "coolest")

    def measure_case_at(self, case_index, x: float, z: float, frame_index: int = -1) -> Optional[float]:
        """The real measured temperature at physical (x, z) in a scenario
        OTHER than (or the same as) the one currently on screen -- for
        "same place, two conditions" comparisons. Reuses value_at()'s own
        physical -> array-index mapping (every scenario in a study shares
        one domain/extent), against an array pulled straight from the
        store rather than whatever frame happens to be plotted right now.
        `frame_index` defaults to -1 (the settled/final frame), the same
        reference point experiments.measure()'s mean_temperature_end
        uses, so a comparison here can never disagree with the guided
        experiment's own numbers for the same scenario.
        """
        if case_index is None or self.view._extent is None:
            return None
        try:
            arr = np.asarray(self._store.get(case_index, self._quantity_key))
        except Exception:  # noqa: BLE001 - an unavailable case is just "no reading"
            return None
        if arr.size == 0:
            return None
        i = max(-arr.shape[0], min(frame_index, arr.shape[0] - 1))
        frame = arr[i]
        x0, x1, z0, z1 = self.view._extent
        n_z, n_x = frame.shape
        if x1 == x0 or z1 == z0:
            return None
        col = int(round((x - x0) / (x1 - x0) * (n_x - 1)))
        row = int(round((z1 - z) / (z1 - z0) * (n_z - 1)))
        if 0 <= row < n_z and 0 <= col < n_x:
            return float(frame[row, col])
        return None

    def current_entry(self):
        """The manifest entry for whatever scenario is currently loaded,
        or None (demo mode / no manifest)."""
        return self._entry(self._case_index)

    def clear_probe(self, keep_trail: bool = False) -> None:
        """Hide the tap ring -- called on every scenario load and phase
        change so a reading never survives onto a screen it no longer
        describes.

        `keep_trail` protects the temperature trail and the "show
        before" ghost from this same blanket clear: both should persist
        across an incidental UI change (opening/closing the Compare or
        Discovery Notebook card while still in OBSERVE looking at the
        exact same scenario) -- Phase 6 section 2 is explicit that a
        child's own measurements must feel persistent, not wiped just
        because they looked at something else for a moment. They are
        still cleared -- via PublicExperience._clear_game_state(), not
        here -- the moment the scenario itself actually changes (explore
        toggle, restart, reset) or OBSERVE is left entirely, since only
        then would either genuinely describe data no longer on screen.
        """
        if self.view.ax is None:
            return
        self.view.set_hover_highlight(None)
        self.clear_dual_markers()
        if not keep_trail:
            self.clear_trail_markers()
            self.hide_baseline_ghost()
        self.view.canvas.blit_update(self.view._animated_artists())

    def show_dual_markers(self, hot_xz: tuple, cool_xz: tuple) -> None:
        """Two ring markers on screen at once -- the Hot/Cold game's own
        artists (separate from SliceView.hover_highlight, which the
        researcher app's Context Panel hover also uses), so the child
        can see both locations they chose simultaneously rather than one
        overwriting the other."""
        if self.view.ax is None:
            return
        if self._hot_marker is None:
            self._hot_marker = self.view.ax.scatter(
                [], [], s=260, marker="o", facecolors="none",
                edgecolors="#FF5A36", linewidths=2.4, zorder=11)
            self.view.add_animated_extra(self._hot_marker)
        if self._cool_marker is None:
            self._cool_marker = self.view.ax.scatter(
                [], [], s=260, marker="o", facecolors="none",
                edgecolors="#5AA9E6", linewidths=2.4, zorder=11)
            self.view.add_animated_extra(self._cool_marker)
        self._hot_marker.set_offsets([hot_xz])
        self._cool_marker.set_offsets([cool_xz])
        self.view.canvas.blit_update(self.view._animated_artists())

    def clear_dual_markers(self) -> None:
        empty = np.empty((0, 2))
        if self._hot_marker is not None:
            self._hot_marker.set_offsets(empty)
        if self._cool_marker is not None:
            self._cool_marker.set_offsets(empty)

    # -- Phase 5: the temperature trail (build-your-own map) -------------
    def add_trail_marker(self, x: float, z: float, number: int, dim: bool = False,
                         redraw: bool = True) -> None:
        """A small numbered ring at a real tapped point -- "let the
        child build their own temperature map" (Phase 5 section 5).
        Static once drawn (the tap it records doesn't change), so it's
        baked into the blit background like the candle patches rather
        than tracked as an animated extra -- there is nothing about it
        that needs updating frame to frame.

        `dim` draws a muted version of the same marker -- used for a
        point measured *before* the child changed the experiment, kept
        on screen after a scenario switch instead of being wiped (Phase
        8 section 7: "I measured HERE before"), so it stays visually
        distinguishable from a fresh measurement without a second marker
        style/shape to maintain.

        `redraw=False` skips the capture_background() call -- each one
        is a full, ~140 ms canvas redraw through the cinema pipeline's
        layered artists, so restoring a whole "before" trail one point
        at a time (see PublicExperience._on_explore_changed) used to pay
        that cost once per point. A caller adding several markers in a
        batch should pass this and call the scene's own capture once,
        after the loop, itself.
        """
        if self.view.ax is None:
            return
        color = "#9AA0A6" if dim else "#FFD166"
        alpha = 0.55 if dim else 1.0
        marker = self.view.ax.scatter(
            [x], [z], s=170, marker="o", facecolors="none",
            edgecolors=color, linewidths=2.0, zorder=11, alpha=alpha)
        label = self.view.ax.text(
            x, z, str(number), color=color, fontsize=9, fontweight="bold",
            ha="center", va="center", zorder=12, alpha=alpha)
        self._trail_markers.append((marker, label))
        if not dim:
            # A dim "before" point never joins the line -- it belongs to
            # a scenario that is no longer on screen, so connecting it to
            # a fresh measurement would visually claim a single walk
            # across a change that never happened.
            self._trail_points.append((x, z))
            self._update_trail_line()
        if redraw:
            self.view.canvas.capture_background()

    def _update_trail_line(self) -> None:
        if self.view.ax is None:
            return
        if len(self._trail_points) < 2:
            if self._trail_line is not None:
                self._trail_line.set_data([], [])
            return
        if self._trail_line is None:
            self._trail_line, = self.view.ax.plot(
                [], [], color="#FFD166", alpha=0.45, linewidth=1.4, zorder=10)
        xs = [p[0] for p in self._trail_points]
        zs = [p[1] for p in self._trail_points]
        self._trail_line.set_data(xs, zs)

    def clear_trail_markers(self) -> None:
        # This runs on every scenario load/phase change via clear_probe()
        # (see its own docstring), not just when the child has actually
        # drawn a trail -- an unconditional capture_background() here paid
        # a full ~145 ms redraw on every single Fan/Candle toggle even
        # when there was nothing on screen to clear, which was most of
        # the measured "why is switching slow" cost. Skip entirely when
        # there is nothing to remove.
        if not self._trail_markers and self._trail_line is None:
            return
        for marker, label in self._trail_markers:
            marker.remove()
            label.remove()
        self._trail_markers = []
        self._trail_points = []
        if self._trail_line is not None:
            self._trail_line.set_data([], [])
        if self.view.ax is not None:
            self.view.canvas.capture_background()

    # -- Phase 5: "show before" ghost overlay -----------------------------
    def show_baseline_ghost(self, baseline_case_index) -> None:
        """A faint dashed contour outline of the baseline scenario's own
        settled (last-frame) temperature field, over the current
        heatmap -- "before/after" without ever rendering a second full
        heatmap (Phase 5 section 6). Static once drawn (baked into the
        background exactly like views.py's own isotherms are, since
        matplotlib contours have no cheap per-frame "update in place"
        primitive) -- recomputed only when re-toggled, not per tick.
        """
        self.hide_baseline_ghost()
        if self.view.ax is None or self._store is None or self.view._extent is None:
            return
        try:
            arr = np.asarray(self._store.get(baseline_case_index, self._quantity_key))
        except Exception:  # noqa: BLE001 - an unavailable baseline just skips the ghost
            return
        if arr.size == 0:
            return
        frame = arr[-1]
        x0, x1, z0, z1 = self.view._extent
        n_z, n_x = frame.shape
        xs = np.linspace(x0, x1, n_x)
        zs = np.linspace(z1, z0, n_z)   # row 0 = z1 (top), matching origin='upper'
        try:
            # Violet, not white: room_walls (views.py) is also a white
            # dashed line, and a "before" contour in the same color reads
            # as a duplicated/offset room wall rather than a temperature
            # comparison. Violet is the one hue this scene doesn't already
            # use for something else (fire is red/orange/yellow, door/vents
            # are blue/green/amber, trail is gold).
            self._ghost_contour = self.view.ax.contour(
                xs, zs, frame, levels=4, colors="#A78BFA", alpha=0.6,
                linewidths=1.2, linestyles="dashed", zorder=9)
        except Exception:  # noqa: BLE001 - a degenerate (uniform) frame has no contours
            self._ghost_contour = None
            return
        self.view.canvas.capture_background()

    def hide_baseline_ghost(self) -> None:
        if self._ghost_contour is not None:
            self._ghost_contour.remove()
            self._ghost_contour = None
            if self.view.ax is not None:
                self.view.canvas.capture_background()

    def refresh(self) -> None:
        """Repaint the canvas at the current frame.

        The overlay above this scene is translucent, so when playback is
        paused and an overlay widget is deleted (a button row swapped out
        on a phase change), nothing repaints the pixels underneath and the
        old widget stays visible as a ghost. Re-blitting the canvas damages
        that whole region, which makes Qt repaint the overlay's *current*
        children over a clean background. Cheap -- it is exactly the same
        blit that runs every frame during playback.
        """
        if self._temperature is None:
            return
        self.view.canvas.blit_update(self.view._animated_artists())

    def set_ui_scale(self, scale: float) -> None:
        self.view.set_ui_scale(scale)
