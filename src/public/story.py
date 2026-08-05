"""Narrated story beats, driven by real detected events.

events.detect_events() already finds ignition, hazard crossings, fastest
heating, the peak, smoke-layer descent and stabilization in a scenario,
each carrying the real time it happened. StoryController translates those
into visitor-facing sentences and hands back whichever beat belongs to
the current frame.

Nothing here invents a beat: if a detector didn't fire for this scenario,
the corresponding sentence is simply never shown. That's the same rule
auto_summary.py follows ("all numbers computed, none generated"), applied
to narration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from descriptors import compute_descriptors
from events import detect_events
from insight import Insight
from public.mascot import CURIOUS, EXCITED, WATCHING

# --- smoke layer, as it actually behaves in this study ------------------
# Measured on the two fan-experiment scenarios: layer_height starts at the
# frame-0 sentinel (the ceiling, meaning "no distinguishable layer yet" --
# see layer_height.py), collapses to near the floor the instant the candle
# lights, then *rises* as the plume pools under the room ceiling and
# plateaus (0.217 m with the fan off, 0.180 m with it on; the room ceiling
# is at 0.22 m). This is a 0.73 x 0.22 m tabletop box, so the physical
# story is a layer building up under the ceiling -- not the descending
# layer of a tall compartment.
CEILING_PLATEAU_FRACTION = 0.90
CEILING_SMOOTH_FRAMES = 9
# The layer must genuinely climb by at least this much (metres) for the
# "gathering under the ceiling" beat to be emitted at all. A run where it
# never rises gets no beat rather than a fabricated one.
MIN_CEILING_RISE_M = 0.05
# A layer-descent event inside the first few frames is the frame-0
# sentinel decaying as the fire ignites, not a physical descent. events.py
# compares against layer[0], which is that sentinel, so the detector fires
# at frame 1 in every scenario here (verified across both fan cases).
# Narrating it would tell a visitor the opposite of what the data shows.
SENTINEL_FRAMES = 4


@dataclass(frozen=True)
class Beat:
    frame_index: int
    # i18n.py translation key, not literal display text -- Beat is built
    # once per scenario and cached, so the caller must tr() this at
    # display time to pick up the language active *then*, not at build
    # time.
    text: str
    icon: str
    # The Insight this came from, so a science view can show the real
    # statement and its basis.
    source: object = None
    # How the guide should react when this beat fires. Declared with the
    # beat so the mascot's expression is tied to a detected event rather
    # than to a timer.
    mood: str = CURIOUS
    # What the guide *says* when it fires: also an i18n.py translation
    # key. Deliberately not the beat text: the banner already states the
    # finding, and having the mascot repeat it word-for-word wasted the
    # one element that can react like a person. Empty falls back to the
    # beat text.
    reaction: str = ""


# Maps a detected event to kid-facing wording. Matching is by substring
# against the Insight's own statement, because events.py builds those
# statements from templates -- keying on them keeps this module from
# duplicating the detector logic. Order matters: first match wins.
# text/reaction here are i18n.py translation keys, not literal display
# text -- Beat objects are built once per scenario during prewarm and
# cached (StoryController is expensive to rebuild), so baking in a
# literal English string would freeze it in whatever language was active
# at prewarm time. Callers must tr() these keys at display time instead.
_BEAT_RULES = (
    ("Ignition", "beat_ignition_text", "🔥", CURIOUS, "beat_ignition_reaction"),
    ("Fastest heating", "beat_fastest_heating_text", "📈", EXCITED,
     "beat_fastest_heating_reaction"),
    ("Peak", "beat_peak_text", "🌡️", EXCITED, "beat_peak_reaction"),
    ("Conditions stabilize", "beat_stabilize_text", "🌡️", WATCHING, "beat_stabilize_reaction"),
)

# The smoke beat is deliberately absent from _BEAT_RULES: events.py's
# "Smoke layer begins descending" is a sentinel artifact here (see
# SENTINEL_FRAMES), and the honest observation for this geometry is the
# layer *gathering* under the ceiling, detected by _ceiling_beat() below.
CEILING_BEAT_TEXT = "beat_ceiling_text"
# 💨 rather than 🌫️ (fog): the fog glyph is missing from the default font
# fallback on the exhibition build and renders as an empty box, which is
# worse than a slightly less literal icon.
CEILING_BEAT_ICON = "💨"


class StoryController:
    """Per-scenario story beats. Construction runs the detectors once
    (a few hundred ms for a 481-frame scenario), so callers should build
    one per case_index and keep it."""

    def __init__(self, data, extent, fps: int):
        self._fps = max(1, fps)
        self._beats: list = []
        self._ceiling_beat: Beat = None
        try:
            table = compute_descriptors(data, extent, self._fps)
            insights = detect_events(table)
        except Exception:  # noqa: BLE001 - narration must never break playback
            table = None
            insights = []
        for insight in insights:
            beat = self._to_beat(insight)
            if beat is not None:
                self._beats.append(beat)
        if table is not None:
            self._ceiling_beat = self._detect_ceiling_beat(table)
            if self._ceiling_beat is not None:
                self._beats.append(self._ceiling_beat)
        self._beats.sort(key=lambda b: b.frame_index)

    def _to_beat(self, insight):
        frame = insight.frame_index(self._fps)
        if frame is None:
            return None
        if ("smoke layer" in insight.statement.lower() and frame <= SENTINEL_FRAMES):
            # The frame-0 "no layer yet" sentinel decaying, not physics.
            return None
        for needle, text, icon, mood, reaction in _BEAT_RULES:
            if needle.lower() in insight.statement.lower():
                return Beat(frame_index=frame, text=text, icon=icon,
                            source=insight, mood=mood, reaction=reaction)
        return None

    def _detect_ceiling_beat(self, table):
        """The frame where the smoke layer has settled under the ceiling.

        Defined as the first frame at which the (smoothed) layer height
        reaches CEILING_PLATEAU_FRACTION of the run's own late-time
        plateau. Keyed to the run's plateau rather than to the room's
        geometry so it stays a property of the data, and gated on the
        layer having actually risen by MIN_CEILING_RISE_M -- a run where
        it never does produces no beat, and the UI simply omits the
        narration.
        """
        try:
            layer = np.asarray(table.column("layer_height"), dtype=float)
        except Exception:  # noqa: BLE001
            return None
        # Frame 0 is the "no distinguishable layer" sentinel (the ceiling
        # itself); including it would poison both the plateau and the rise.
        body = layer[1:]
        if body.size < CEILING_SMOOTH_FRAMES * 2:
            return None

        plateau = float(np.median(body[int(body.size * 0.75):]))
        start = float(np.min(body[:CEILING_SMOOTH_FRAMES]))
        if plateau <= 0.0 or (plateau - start) < MIN_CEILING_RISE_M:
            return None

        kernel = np.ones(CEILING_SMOOTH_FRAMES) / CEILING_SMOOTH_FRAMES
        smoothed = np.convolve(body, kernel, mode="same")
        hits = np.flatnonzero(smoothed >= CEILING_PLATEAU_FRACTION * plateau)
        if not hits.size:
            return None
        frame = int(hits[0]) + 1  # undo the frame-0 offset

        source = Insight(
            statement=f"Smoke layer settles under the ceiling at {plateau:.2f} m.",
            category="event", quantity="TEMPERATURE",
            time_s=frame / self._fps, value=plateau, unit="m",
            basis=(f"first frame where the smoothed smoke-layer height reaches "
                   f"{CEILING_PLATEAU_FRACTION:.0%} of its late-time plateau "
                   f"({plateau:.3f} m)"))
        return Beat(frame_index=frame, text=CEILING_BEAT_TEXT,
                    icon=CEILING_BEAT_ICON, source=source, mood=WATCHING)

    def ceiling_beat(self):
        """The smoke-gathering beat, or None if this run never shows one.
        The observe phase uses its frame to decide when it may end."""
        return self._ceiling_beat

    def beats(self) -> list:
        return list(self._beats)

    def beat_at(self, frame_index: int, window: int = 12):
        """The beat that should be on screen at `frame_index`, i.e. the
        most recent one starting within `window` frames behind it, or None.

        A window rather than an exact match so playback cannot skip a beat
        between ticks. `window` is the caller's to choose because the
        right size depends on playback speed, not on the data: the public
        experience passes a window worth several seconds of wall clock at
        its own 6x rate (see experience.BEAT_VISIBLE_FRAMES)."""
        current = None
        for beat in self._beats:
            if beat.frame_index <= frame_index <= beat.frame_index + window:
                current = beat
        return current

    def first_beat_after(self, frame_index: int):
        return next((b for b in self._beats if b.frame_index > frame_index), None)
