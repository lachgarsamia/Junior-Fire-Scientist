"""PublicExperience: the root widget for the Fire Explorer.

Owns the phase machine (PublicState), the scene, the overlay, and its
*own* TimeController instance. That last point is deliberate -- it reuses
the TimeController class but not MainWindow's instance, so public
playback can run at exhibition speed without touching the researcher
app's clock, timeline widget, or selection bus.

Pacing: scenarios are 481 frames at 4 fps (120 s of simulated time), far
too long for a visitor. Playback runs at PLAYBACK_SPEED so the observe
step and the experiment each take a handful of seconds, and the whole
journey fits in 60-90 s.
"""

from __future__ import annotations

import contextlib
import logging
import random
import sys
import time
import traceback

from PyQt5 import QtCore, QtWidgets

from manifest import foreign_path_entries
from public import experiments as experiments_mod
from public import kid_language as kid
from public.mascot import (CURIOUS, EXCITED, EXPLAINING, IDLE, POINTING,
                           SURPRISED, THINKING, WATCHING)
from public import i18n
from public.i18n import tr
from public.overlay import PublicOverlay
from public.scene import PublicScene, VELOCITY_KEY
from public.sound import play_success_chime
from public.widgets import BarCompare, DELIGHT, HeroMetric, SecondaryMetric, VerdictBadge
from public.state import Phase, PublicState
from public.story import StoryController
from slice_key import DEFAULT_SLICE_KEY
from time_controller import TimeController

logger = logging.getLogger(__name__)

# Playback multiplier over the data's native 4 fps. 6x turns a 120 s
# scenario into ~20 s of screen time.
PLAYBACK_SPEED = 6
# Frames of playback per second of wall clock, at PLAYBACK_SPEED.
_FRAMES_PER_WALL_SECOND = 4 * PLAYBACK_SPEED

# How long a narration beat stays on screen, in wall-clock seconds. The
# original 12-frame window was half a second at 6x -- the message flashed
# past before it could be read.
BEAT_READ_SECONDS = 4.0
BEAT_VISIBLE_FRAMES = int(BEAT_READ_SECONDS * _FRAMES_PER_WALL_SECOND)

# The observe phase ends a readable moment *after* the smoke-gathering
# beat, so the visitor always sees it before being asked to predict --
# see _observe_end_frame(). These are the bounds around that, not the
# boundary itself.
OBSERVE_MIN_FRAMES = 120                  # never cut the fire off in its first 30 s
OBSERVE_FALLBACK_END_FRAME = 200          # when a run has no smoke beat at all
OBSERVE_TAIL_FRAMES = BEAT_VISIBLE_FRAMES  # reading time after the beat fires
# How far ahead of the smoke beat the guide says "look closely" -- about
# a second and a half of wall clock at PLAYBACK_SPEED.
NUDGE_LEAD_FRAMES = 36

# The experiment run stops here: far enough for the airflow difference to
# be unmistakable and the room temperature to have settled.
EXPERIMENT_END_FRAME = 400

# EXPERIMENT plays for EXPERIMENT_END_FRAME / _FRAMES_PER_WALL_SECOND real
# seconds (~17s at the current constants) with nothing but the fire on
# screen and one line said back at the very start -- long enough that a
# child watching for "did I get it right?" reasonably reads the silence
# as broken (games UX pass: a real "still doesn't say if right or wrong"
# report during exactly this wait). One reassurance partway through,
# timed off the same real constants rather than a guessed number of ms,
# so it always lands with room to read it before the reveal actually
# arrives.
EXPERIMENT_ALMOST_THERE_MS = int(EXPERIMENT_END_FRAME / _FRAMES_PER_WALL_SECOND * 1000 * 0.65)

# The countdown before a run: four beats of ~400 ms.
COUNTDOWN_STEP_MS = 400
_COUNTDOWN_STEPS = ("3", "2", "1", "🕯️")

# Hot/Cold (games UX pass): how far apart the two tapped readings must be
# to count as a genuine "big difference," not the room's own ordinary
# spatial noise. Deliberately its own constant, not PUBLIC_METRICS'
# room_temp.noticeable_delta (0.3 C) -- that metric guards a *whole-scene
# average* comparing two full scenario runs, where a fraction of a degree
# is real and worth reporting; this compares two single-point taps in one
# frame, where measured on this dataset a typical pair of "ordinary" room
# spots already differs by several degrees just from thermal layering
# (frame-to-frame p50-to-p90 spread runs ~7-10 C with candles=1, fan off).
# Reusing 0.3 C here made nearly every tap pair "succeed" regardless of
# whether the child had actually found a hot spot versus a cool one --
# the bug behind "hot/cold always says big difference." 15 C clears that
# ordinary spread while staying reachable by a child aiming for "near the
# flame/smoke" versus "away from it" without needing to hit the hottest
# ~1% of pixels exactly.
HOTCOLD_BIG_DIFFERENCE_C = 15.0

# Phases where tapping the heatmap measures a real temperature there.
# Not the guided question/countdown/reveal-transition screens, where a
# card or the countdown glyph occupies most of the fire anyway.
# GAME_PLAY is included: every tap-based Games/Challenges activity reuses
# this exact same probe.
PROBE_PHASES = (Phase.OBSERVE, Phase.EXPERIMENT, Phase.REVEAL, Phase.SCIENCE,
                Phase.GAME_PLAY)

# Dr. Funke's fact bank -- real, general fire-science statements (never
# a claim about this specific simulation's own measurements, which stay
# the exclusive job of the honesty-gated PUBLIC_METRICS/kid_language
# machinery elsewhere). Keys, not literal text, same convention every
# other translatable content bank in this module already follows.
FIRE_FACT_KEYS = (
    "fact_hot_air_rises",
    "fact_smoke_ceiling_first",
    "fact_fire_triangle",
    "fact_moving_air_oxygen",
    "fact_cool_air_sinks",
    "fact_blue_flame_hottest",
    "fact_closed_door_slows_fire",
    "fact_smoke_more_dangerous",
    "fact_firefighters_study_smoke",
)
# 7000 -> 14000: per explicit feedback ("let the message pop for a
# longer while so user can read it slowly, no rush") -- doubled, not
# just nudged, so a child reading at their own pace never has it
# disappear mid-sentence.
_FACT_SHOW_MS = 14000
# 15000 -> 24000: an explicitly-requested slower, more ambient cadence
# (~20-30s between facts, not ~15-22s) -- she should read as filling
# occasional quiet moments, not chattering on a short loop.
_FACT_GAP_MS = 24000
_FACT_FIRST_DELAY_MS = 12000
# How soon _show_next_fact retries after finding it isn't a lull (see
# _is_a_lull) -- short, since this is "try again shortly", not a new
# full-length wait; _defer_if_current's own phase guard is what stops
# these retries once OBSERVE is actually left, so this can't spin
# forever on some other phase.
_FACT_RETRY_MS = 4000
# How recently the primary mascot must have spoken, or the child must
# have interacted, for a fact to be withheld -- "a genuine lull," not
# merely "no fact currently showing." Same window for both checks (see
# _is_a_lull): the ask was "a few seconds" for either, not two
# different tunings.
_LULL_GRACE_S = 4.0

# Which stage of the WATCH -> GUESS -> TEST -> DISCOVER strip each phase
# belongs to. -1 means "no journey under way", so the strip is hidden --
# that includes GAMES_HOME/GAME_PLAY, which aren't part of this guided
# journey at all (the strip would claim a "step number" a Games activity
# doesn't have).
_STAGE_FOR_PHASE = {
    Phase.OBSERVE: -1,
    Phase.PREDICTION: 1,
    Phase.COUNTDOWN: 1,
    Phase.EXPERIMENT: 2,
    Phase.REVEAL: 3,
    Phase.SCIENCE: 3,
    Phase.COMPLETE: 3,
}


class PublicExperience(QtWidgets.QWidget):
    """Full-screen public experience. Construct once; call enter() when
    it becomes visible and leave() when it stops being visible."""

    exit_requested = QtCore.pyqtSignal()

    def __init__(self, sim_data, parent=None):
        super().__init__(parent)
        self.sim_data = sim_data
        self.state = PublicState()
        self._story: StoryController = None
        self._stories: dict = {}
        self._measurements: dict = {}
        # Free-play "what changed?" comparison (baseline vs whatever the
        # child explored to) -- keyed by case_index rather than by choice
        # key like _measurements, since an explored scenario need not be
        # either side of the guided Experiment's own Choices.
        self._case_measurements: dict = {}
        # The compare card is a pause-and-look detour, exactly like the
        # help card (_help_open/_resume_after_help below) -- it must not
        # advance the phase or disturb OBSERVE's own state.
        self._compare_open = False
        self._resume_after_compare = False
        # Phase 2: the two Fire Lab mini-games and the same-place
        # comparison, all layered on the same real tap handler
        # (_on_overlay_tapped) rather than a separate input mode. None
        # means "no game in progress" -- a plain tap. Cleared on any
        # scenario switch (_clear_game_state) so an in-progress guess
        # never gets silently compared against a scenario it wasn't
        # made in.
        self._active_game: Optional[str] = None   # None | "hottest" | "hotcold"
        self._hotcold_stage = 0
        self._hotcold_readings: dict = {}
        # Which extreme "hottest"'s game is hunting for this round --
        # chosen dynamically per activation (Phase 4 section 13), not
        # two separate menu items.
        self._temp_hunt_target = "hottest"
        # "Test an idea" quiz mode: the prediction buttons currently on
        # screen, kept so _flash_prediction_choice can highlight the one
        # actually tapped and disable the rest for the beat before the
        # phase advances (see _render_prediction).
        self._prediction_buttons: list = []
        # Phase 4: "Can you figure it out?" (the Mystery game) -- the fan
        # affecting the ceiling and the floor differently. Tracked across
        # separate "compare this place" taps so either order counts.
        # Phase 8 section 9: detection itself no longer needs the "Figure
        # it out" button pressed first -- every same-place comparison
        # opportunistically checks for this pattern (see
        # _show_same_place_comparison/_note_mystery_progress), so the
        # discovery can emerge from ordinary measuring instead of only
        # from an armed mini-game.
        #
        # Games UX pass: this used to hardcode "ceiling must read cooler,
        # floor must read warmer" as the only recognized pattern -- true
        # for the dataset this was written against, but re-measuring
        # against the current one (see _note_mystery_progress) found the
        # ceiling's own direction genuinely varies by exact tap position
        # (recirculation, not a uniform layer), while the floor reliably
        # warms. Hardcoding a direction that isn't reliably true anymore
        # made most ceiling taps silently not count as progress at all --
        # the real bug behind "games aren't interacting as they should."
        # The fix tracks whichever direction each zone *actually* showed
        # and only claims the "aha" when the two genuinely disagree,
        # which is the actual condition the "same fan, different place"
        # story depends on -- never assumed, always the measured sign.
        self._mystery_found_high = False
        self._mystery_found_low = False
        self._mystery_high_warmer: Optional[bool] = None
        self._mystery_low_warmer: Optional[bool] = None
        self._mystery_solved = False
        # Phase 12 section 11: the real physical (x, z) of whichever
        # ceiling/floor taps satisfied the mystery pattern -- kept only
        # so the "aha" moment can highlight the *actual* two places the
        # child measured (see show_dual_markers in _show_same_place_
        # comparison), never invented positions.
        self._mystery_high_xz: Optional[tuple] = None
        self._mystery_low_xz: Optional[tuple] = None
        # Phase 4: the Discovery Notebook -- real, measured moments the
        # child has found, capped and deduplicated by title (see
        # _record_discovery). Never a score; just a small persistent
        # record of what was actually discovered.
        self._discoveries: list = []
        # Bumped on every scenario load and phase render (see _load_case,
        # _render_phase) -- a _defer_if_current callback captures this at
        # schedule time and refuses to run if it has since changed, so a
        # delayed thermometer sweep can never
        # stomp a caption/value that no longer describes what's on
        # screen (Phase 4 section 15).
        self._interaction_generation = 0
        # Tracks the phase (and, for GAME_PLAY, which game) _render_phase()
        # last actually rendered, so it can tell "the same screen
        # re-rendering because a detour closed" (the trail/ghost must
        # survive) apart from "rendering fresh after a real scenario
        # switch or a genuine phase/game change" (they must not) -- see
        # _render_phase's own same_screen_rerender.
        self._last_rendered_phase: Optional[Phase] = None
        self._last_rendered_game: Optional[str] = None
        # Phase 5: the temperature trail -- real taps the child has
        # placed on the current scenario, cleared on a genuine scenario
        # switch/restart/reset or on leaving OBSERVE (see
        # _clear_game_state), but *not* by an incidental UI change like
        # opening/closing a card (Phase 6 section 2).
        self._temp_trail: list = []
        # Phase 8 section 7: the trail measured *before* the child's most
        # recent Fan/Candle change, kept on screen (dimmed) instead of
        # being wiped by the scenario switch -- so tapping approximately
        # the same spot again reads as "revisiting a place", not opening
        # a new tool (see _match_before_trail/_on_explore_changed).
        self._before_trail: list = []
        # Phase 5: the "experiment board" -- the most recent real change
        # and the most recent real measured finding, surfaced inside the
        # Discovery Notebook (see _on_notebook_requested) rather than a
        # second always-on panel this 800x600 layout has no room for.
        self._last_change: Optional[str] = None
        self._last_watched: Optional[str] = None
        # Phase 7: a one-time nudge after the first candle tap, only while
        # the child hasn't changed anything yet -- see _on_candle_tapped.
        self._candle_nudge_shown = False
        # Dr. Funke's fact bank, shuffled and drawn down without repeats
        # until exhausted, then reshuffled -- see _show_next_fact. Not
        # reset by _restart_narration/scenario switches: unlike the
        # story beats, a fact isn't *about* the currently-loaded run, so
        # there's no reason to make a visitor sit through the same one
        # twice just because they flipped the fan.
        self._fact_queue: list = []
        # Bumped once per _arm_fact_timer() call, own guard the whole
        # fact-timer chain checks against -- deliberately separate from
        # _interaction_generation (see _arm_fact_timer's own comment on
        # why reusing that one is wrong here). Lets a same-screen
        # OBSERVE re-render (e.g. returning from the Games hub) cleanly
        # retire whatever chain was already pending instead of running
        # two in parallel.
        self._fact_chain_token = 0
        # 0.0, not time.monotonic(): reads as "long ago" from the first
        # check without a sentinel -- see _last_say_at's own comment
        # (PublicOverlay) for why the same idiom is used there. Bumped
        # in _on_overlay_tapped and _on_explore_changed, the same two
        # handlers _idle_vent_timer already hooks -- see _show_next_fact
        # for why this and PublicOverlay.seconds_since_say() are checked
        # together before a fact is allowed to appear.
        self._last_interaction_at = 0.0
        # TEMPORARY diagnostic instrumentation for the real, live SIGABRT
        # crash under investigation (a 3-4 level deep Qt signal/slot
        # chain ending in abort(), per the crash report) -- tracks
        # re-entrancy depth per named call site so a repro run can show
        # whether any of these are genuinely being re-entered while
        # already on the call stack, not just guessed at from the C-level
        # trace. See _diag_reentrancy_guard. Remove once root-caused.
        self._diag_depths: dict = {}
        self._active = False
        # Set by _play_until(): the frame playback should stop at, and
        # what to run when it gets there. None means "no scheduled stop"
        # (the attract loop).
        self._play_end_frame = None
        self._play_finished_cb = None
        # Frame indices of beats already narrated this run, so a beat
        # fires once rather than on every frame inside its window (and
        # never twice when the attract loop wraps). Cleared by
        # _restart_narration() on enter / reset / replay / scenario swap.
        self._spoken_beats: set = set()
        # True once this run's airflow has crossed into a band the
        # kid-language layer calls strong -- reacted to once, from the
        # measured field, not from knowing which scenario is loaded.
        self._airflow_reacted = False
        # True once this run's "look closely" nudge has been spoken.
        self._nudged = False
        # The help card ('What am I seeing?') is a pause-and-explain
        # detour, not a phase: it must not disturb the journey.
        self._help_open = False
        self._resume_after_help = False
        # Countdown ticker. Parented to this widget so it cannot outlive
        # it -- the same rule the banner flash and mascot timers follow.
        self._countdown_step = 0
        self._countdown_timer = QtCore.QTimer(self)
        self._countdown_timer.timeout.connect(self._advance_countdown)

        # Thermometer / probe state. "mean" is the resting state: the
        # thermometer always means something, even before any tap.
        # "point" is a single tapped/current reading, replaced by the
        # next tap -- there is no persisted multi-point trail here (see
        # "Map It" for that, entered intentionally from the Games hub).
        self._probe_mode = "mean"
        self._probe_xz = None
        # Recreated per phase-render (see _render_phase); guarded so a
        # stale reference from a phase that had no such button is never
        # touched by the TimeController's playing_changed signal.
        self._play_pause_button = None
        # Phase 11 section 2: a one-time, one-shot nudge toward the vent
        # if the child has been inactive for a while during OBSERVE --
        # armed on entry, cancelled by any real interaction, never
        # re-armed once it has fired (see _arm_idle_vent_hint/
        # _on_idle_vent_timeout). A restarted QTimer, not a per-frame
        # animation loop -- idle-vent-hint fires at most once per visit.
        self._idle_vent_hint_shown = False
        self._idle_vent_timer = QtCore.QTimer(self)
        self._idle_vent_timer.setSingleShot(True)
        self._idle_vent_timer.timeout.connect(self._on_idle_vent_timeout)

        self.experiment = experiments_mod.FAN_EXPERIMENT
        self._available = experiments_mod.is_available(
            sim_data.manifest, self.experiment)
        # Free-play controls (fan, candle count), resolved from whatever
        # this manifest actually contains -- see experiments.py for why
        # only these two, and why each is one-factor-at-a-time.
        self._explore_controls = experiments_mod.available_explore_controls(
            sim_data.manifest)

        self.setStyleSheet("background: #07090E;")
        self.setAutoFillBackground(True)

        self.scene = PublicScene(sim_data.store, sim_data.manifest or [],
                                 sim_data.timesteps_per_second, self)
        self.overlay = PublicOverlay(self)
        # Reserve real screen space for the mascots at the bottom of the
        # scene, rather than letting them float on top of the flame/room
        # they'd otherwise overlap (a real, screenshotted complaint) --
        # both mascot sizes are fixed for the life of the app, so this is
        # set once here rather than recomputed per scenario switch.
        self.scene.set_bottom_reserve_px(self.overlay.mascot_band_height_px())

        # StackAll keeps the scene painted underneath and the overlay on
        # top, both filling the widget -- the overlay is translucent, so
        # the fire shows through everywhere it has no chrome.
        layout = QtWidgets.QStackedLayout(self)
        layout.setStackingMode(QtWidgets.QStackedLayout.StackAll)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.overlay)
        layout.addWidget(self.scene)
        layout.setCurrentWidget(self.overlay)

        self.time_controller = TimeController(
            self._frame_count, sim_data.timesteps_per_second, parent=self)
        self.time_controller.set_speed(PLAYBACK_SPEED)
        self.time_controller.set_loop(False)
        self.time_controller.time_changed.connect(self._on_time_changed)
        self.time_controller.playing_changed.connect(self._sync_play_pause_label)

        # Everything else (Help, Games, the Games hub tiles, each game's
        # own action buttons) is wired directly from whichever _render_*
        # method adds that button (see overlay.add_button) rather than a
        # persistent signal connected once here -- those buttons don't
        # exist for the life of the app the way these do.
        self.overlay.start_requested.connect(self._begin_journey)
        self.overlay.prediction_made.connect(self._on_prediction)
        self.overlay.choice_made.connect(self._on_choice)
        self.overlay.science_toggled.connect(self._on_science_toggled)
        self.overlay.help_requested.connect(self._on_help_requested)
        self.overlay.replay_requested.connect(self._on_replay)
        self.overlay.exit_requested.connect(self.exit_requested)
        self.overlay.tapped.connect(self._on_overlay_tapped)
        self.overlay.explore_changed.connect(self._on_explore_changed)
        self.overlay.language_requested.connect(self._on_language_requested)

        self.overlay.set_explore_controls(self._explore_controls)
        self._validate_data()
        self._resolve_baseline()

    # -- data integrity -------------------------------------------------
    def _validate_data(self) -> None:
        """Public mode must not silently serve another checkout's data.

        The manifest stores absolute paths, so a copied study keeps
        pointing at the original directory; on an exhibition machine that
        surfaces as a mysterious fallback to demo data. Log loudly here
        rather than trusting a manifest this process did not write.
        """
        from load_data import SIM_ROOT
        entries = self.sim_data.manifest or []
        if not entries:
            return
        foreign = foreign_path_entries(entries, SIM_ROOT)
        if foreign:
            logger.error(
                "public mode: %d of %d manifest entries point outside %s "
                "(first: %s). Delete fds/sim/manifest.json to regenerate.",
                len(foreign), len(entries), SIM_ROOT, foreign[0].path)

    def _resolve_baseline(self) -> None:
        """The scenario the intro/observe steps show: the experiment's own
        baseline choice, so 'observe' and 'fan off' are literally the same
        run and the comparison is honest."""
        if not self._available:
            self.state.baseline_case_index = 0
            self.state.case_index = 0
            return
        case_index = experiments_mod.resolve_choice(
            self.sim_data.manifest, self.experiment, self.experiment.baseline_choice)
        self.state.baseline_case_index = case_index
        self.state.case_index = case_index
        self.state.experiment = self.experiment

    # -- lifecycle ------------------------------------------------------
    def enter(self) -> None:
        self._active = True
        self.state.reset()
        self._load_case(self.state.case_index)
        self._render_phase()

    def leave(self) -> None:
        self._active = False
        self.time_controller.pause()

    def shutdown(self) -> None:
        """Stop every timer this experience owns. Called from
        MainWindow.closeEvent -- the same place the kiosk filter and the
        busy-cursor stack are unwound, for the same reason: Qt timers and
        filters outlive Python references."""
        self._countdown_timer.stop()
        self.time_controller.pause()
        self.overlay.shutdown()

    def reset(self) -> None:
        """Back to attract -- what the kiosk idle timer calls."""
        self._countdown_timer.stop()
        self.time_controller.pause()
        self.state.reset()
        self._idle_vent_hint_shown = False
        self._idle_vent_timer.stop()
        self._reset_mystery_and_notebook()
        self._load_case(self.state.baseline_case_index)
        self._render_phase()

    def _reset_mystery_and_notebook(self) -> None:
        """A new visitor's own notebook and mystery progress -- kept
        separate from _clear_game_state(), which fires on every ordinary
        scenario switch and must *not* forget mystery progress (the
        "compare this place" taps that solve it deliberately span more
        than one fan toggle)."""
        self._mystery_found_high = False
        self._mystery_found_low = False
        self._mystery_high_warmer = None
        self._mystery_low_warmer = None
        self._mystery_solved = False
        self._mystery_high_xz = None
        self._mystery_low_xz = None
        self._discoveries = []
        self._candle_nudge_shown = False

    # -- playback -------------------------------------------------------
    def _frame_count(self) -> int:
        return self.scene.frame_count()

    def _restart_narration(self) -> None:
        """Let every beat play again from the top of a fresh run."""
        self._spoken_beats.clear()
        self._airflow_reacted = False
        self._nudged = False

    def _reset_probe_state(self) -> None:
        """Back to the resting "mean" reading -- called on every scenario
        load so a tapped point from the previous run (or the previous
        scenario) never survives onto this one."""
        self._probe_mode = "mean"
        self._probe_xz = None

    def _clear_game_state(self) -> None:
        """Abandon any in-progress Fire Lab mini-game -- called on every
        scenario switch/restart/reset so a "hot" reading taken under Fan
        OFF can never end up compared against a "cool" one taken after
        switching to Fan ON (see _handle_hotcold_tap)."""
        self._active_game = None
        self._hotcold_stage = 0
        self._hotcold_readings = {}
        self.scene.clear_dual_markers()
        # The trail's own scene artists are cleared by PublicScene.
        # clear_probe() (called from _load_case/_render_phase); this
        # resets the *experience's* tracking of it so a re-render's own
        # button declarations stay in sync.
        self._temp_trail = []
        self._before_trail = []

    # -- Discovery Notebook (Phase 4) -------------------------------------
    _MAX_DISCOVERIES = 4

    def _record_discovery(self, icon: str, title: str, text: str) -> None:
        """A real, measured moment the child found -- never a score, an
        achievement threshold, or anything invented. Deduplicated by
        title (repeating the same mini-game win doesn't spam duplicate
        entries) and capped so the notebook stays a handful of real
        highlights, not a log."""
        self._discoveries = [d for d in self._discoveries if d["title"] != title]
        self._discoveries.append({"icon": icon, "title": title, "text": text})
        self._discoveries = self._discoveries[-self._MAX_DISCOVERIES:]
        # "My Discoveries" appears on the Games hub as soon as there's
        # something in it -- computed at render time (_render_games_home)
        # rather than a persistent visibility flag to keep in sync here.
        self.scene.pulse_flame()   # the same small "aha" flourish a candle tap gets

    # A short pool so "you got it" doesn't say the exact same line every
    # time -- layered onto the real, specific finding, never replacing it.
    # Looked up fresh (not a frozen tuple) so a language switch changes
    # what gets randomly picked next, same as everything else on screen.
    _CELEBRATION_LINE_KEYS = (
        "celebration_line_1", "celebration_line_2", "celebration_line_3", "celebration_line_4")

    def _celebrate(self, message: str) -> None:
        """The one place every "correct answer" moment in the Games
        section goes through: a confetti/star burst (CelebrationOverlay),
        a short synthesized chime, and an encouraging line appended to
        the real, specific finding -- never a bare "Wrong"/"Correct" and
        never replacing the actual measured result with generic praise."""
        self.overlay.celebrate()
        play_success_chime()
        line = tr(random.choice(self._CELEBRATION_LINE_KEYS))
        self.overlay.say(f"{message} {line}", EXCITED)

    def _on_notebook_requested(self) -> None:
        """📓 My discoveries -- reuses the exact same pause-and-look
        detour "🔎 What changed?" already uses (_open_compare_detour/
        _close_compare); this is a card, not a new dashboard widget.

        Leads with a small "🔬 Right now" experiment-board section (Phase
        5 section 3: I CHANGED / I WATCHED / I DISCOVERED) built from
        whatever the child has *actually* done most recently -- not a
        second always-on panel this 800x600 layout has no spare room
        for, just the same real state the rest of Fire Lab already
        tracks, surfaced here.
        """
        if self.state.phase not in (Phase.GAMES_HOME, Phase.GAME_PLAY) or self._compare_open:
            return
        if not self._discoveries:
            return
        self._open_compare_detour()
        board = []
        if self._last_change:
            board.append(tr("board_i_changed", value=self._last_change))
        if self._temp_trail or self._before_trail:
            # Phase 10 section 10: numbered and arrow-chained -- "I
            # changed this, measured here, then measured there" should
            # be reconstructable at a glance, in the same numbering the
            # in-scene markers already use (before-points first, since
            # they were measured first).
            all_points = self._before_trail + self._temp_trail
            readings = " → ".join(
                f"{self._circled_number(i + 1)}{p['value']:.0f}°C"
                for i, p in enumerate(all_points))
            board.append(tr("board_i_measured", value=readings))
        elif self._last_watched:
            board.append(tr("board_i_watched", value=self._last_watched))
        if self._discoveries:
            board.append(tr("board_i_discovered", value=self._discoveries[-1]['text']))
        history = [f"{d['icon']} {d['title']}\n{d['text']}"
                  for d in reversed(self._discoveries)]
        self.overlay.show_card(tr("notebook_title"), board, dim_lines=history)
        self.overlay.say(tr("notebook_say"), EXCITED)
        self.overlay.add_button(tr("keep_exploring"), "🕯️", self._close_compare, primary=True)

    # -- Phase 5: temperature trail (build-your-own map) -----------------
    _MAX_TRAIL_POINTS = 6

    @staticmethod
    def _circled_number(n: int) -> str:
        """"①"/"②"/... matching the in-scene trail markers' own
        numbering -- falls back to a plain "12." past the 20 Unicode
        circled digits exist for, which the trail's own 6-point cap
        never gets close to."""
        return chr(0x2460 + n - 1) if 1 <= n <= 20 else f"{n}."

    def _add_trail_point(self, x: float, z: float, value_c: float) -> None:
        if len(self._temp_trail) >= self._MAX_TRAIL_POINTS:
            self.overlay.say(tr("trail_full"), CURIOUS)
            return
        # Numbering continues past any preserved "before" points rather
        # than restarting at (1) -- two dim ①② already on screen and a
        # fresh ① for a genuinely new spot would look like a mistake.
        number = len(self._before_trail) + len(self._temp_trail) + 1
        self._temp_trail.append({"x": x, "z": z, "value": value_c})
        self.scene.add_trail_marker(x, z, number)
        if len(self._temp_trail) == 1:
            self.overlay.say(
                tr("trail_first_reading", value=value_c, where=self.scene.location_phrase(x, z)),
                CURIOUS)
        else:
            chain = " → ".join(f"{p['value']:.0f}°C" for p in self._temp_trail)
            self.overlay.say(f"📍 {chain}", CURIOUS)
        # "Clear map" only appears once there's something to clear --
        # this is the moment that first becomes true.
        self._refresh_game_buttons()

    # -- Phase 8 section 7: revisiting a place measured before a change --
    _REVISIT_TOLERANCE_M = 0.08

    def _match_before_trail(self, x: float, z: float):
        """The preserved "before" point nearest a new tap, if the tap
        lands close enough to count as the same physical spot -- same
        tolerance scale as PublicScene's own vent tap radius. Returns
        None (a genuinely new location) rather than ever guessing."""
        for point in self._before_trail:
            if (abs(x - point["x"]) <= self._REVISIT_TOLERANCE_M
                    and abs(z - point["z"]) <= self._REVISIT_TOLERANCE_M):
                return point
        return None

    def _on_clear_trail_requested(self) -> None:
        if self.state.phase is not Phase.GAME_PLAY:
            return
        self._temp_trail = []
        self._before_trail = []
        self.scene.clear_trail_markers()
        self.scene.refresh()
        self.overlay.say(tr("trail_cleared"), CURIOUS)
        self._refresh_game_buttons()

    def _load_case(self, case_index) -> None:
        if case_index is None:
            return
        # Invalidates any pending _defer_if_current callback (a
        # thermometer sweep scheduled by the *previous* scenario/frame) --
        # see _defer_if_current's own docstring for the bug this closes.
        self._interaction_generation += 1
        self.scene.load_case(case_index, DEFAULT_SLICE_KEY)
        self.state.case_index = case_index
        self._story = self._story_for(case_index)
        # A new scenario is a new run: its beats have not been heard yet,
        # even if the previous scenario's beat at the same frame index had.
        self._restart_narration()
        self._reset_probe_state()
        self.time_controller.seek(0)
        if not self._active:
            # seek(0) above only reaches the scene through
            # _on_time_changed, which no-ops while inactive (e.g. during
            # construction, before enter() first runs) -- paint the first
            # frame directly so the scene is never left blank. When
            # active (every real call site today), the cinema pipeline
            # render this would trigger is already the one seek(0) just
            # did through that signal, and doing it twice was a real,
            # measured ~14 ms per scenario switch for nothing.
            self.scene.show_frame(0)

    def _story_for(self, case_index: int) -> StoryController:
        """One StoryController per scenario, cached -- building it runs
        the descriptor + event detectors over the whole run."""
        if case_index not in self._stories:
            extent = self.scene._extent_for(case_index, DEFAULT_SLICE_KEY)
            data = self.sim_data.store.get(case_index, DEFAULT_SLICE_KEY)
            self._stories[case_index] = StoryController(
                data, extent, self.sim_data.timesteps_per_second)
        return self._stories[case_index]

    def _play_until(self, end_frame: int, on_finished) -> None:
        # Known small inefficiency, left alone rather than risked this
        # late: TimeController.seek() unconditionally emits time_changed
        # (see time_controller.py), so a caller that already sought to 0
        # itself just before this (e.g. _load_case, on an explore-control
        # switch) pays a second real cinema-pipeline render of frame 0
        # here (~14 ms, measured). Making seek() a no-op for an unchanged
        # index would fix it but is shared with the researcher app's own
        # TimeController use, which is out of scope for a public-only
        # latency pass.
        self._play_end_frame = end_frame
        self._play_finished_cb = on_finished
        self.time_controller.seek(0)
        self.time_controller.play()

    def _on_time_changed(self, index: int) -> None:
        if not self._active:
            return
        self.state.frame_index = index
        self.scene.show_frame(index)
        self.overlay.update_meters(
            self.scene.mean_temperature_at(index),
            self.scene.mean_airspeed_at(index),
            has_velocity=self.scene._velocity is not None)
        self._update_thermometer(index)
        self._react_to_airflow(index)

        self._nudge_before_beat(index)
        self._narrate(index)

        end = self._play_end_frame
        if end is not None and index >= min(end, self.scene.frame_count() - 1):
            self.time_controller.pause()
            self._play_end_frame = None
            callback = self._play_finished_cb
            self._play_finished_cb = None
            if callback is not None:
                callback()

    def _react_to_airflow(self, index: int) -> None:
        """Have the guide react the first time the air genuinely starts
        moving. Driven by the measured field through the same band table
        the meter uses -- not by knowing which scenario is loaded, so it
        simply never fires in a run where the air stays still."""
        if self._airflow_reacted or self.state.phase is not Phase.EXPERIMENT:
            return
        if self.scene._velocity is None:
            return
        speed = self.scene.mean_airspeed_at(index)
        if kid.airflow_band_key(speed) not in kid.STRONG_AIRFLOW_KEYS:
            return
        self._airflow_reacted = True
        self.overlay.say(tr("airflow_reacted_say"), SURPRISED)

    def _update_thermometer(self, index: int) -> None:
        """Keep the thermometer honest each frame: the resting "mean"
        reading tracks the whole-scene average as the run plays. A plain
        "point" tap is left alone: it is a single measurement of one
        instant, not something that should silently drift back to the
        average a frame later.
        """
        if self.state.phase not in PROBE_PHASES:
            return
        if self._probe_mode == "mean":
            self.overlay.update_thermometer(
                self.scene.mean_temperature_at(index), "🌡️ Whole room average")

    # -- "what am I seeing?" --------------------------------------------
    def _what_am_i_seeing(self) -> list:
        """Short, child-readable notes on what is actually on screen.

        Each line is gated on the thing being there: the arrow line only
        appears when velocity data is loaded, the smoke line only once
        the detected smoke beat has actually fired. Nothing describes a
        visual the visitor cannot currently see.
        """
        # Ordered by what is most interesting *right now*, not as a fixed
        # glossary: the smoke line leads once the detected smoke beat has
        # fired, the airflow line leads while the air is measurably
        # moving, and otherwise it falls back to the general explanation.
        # One short line each -- the card is height-constrained at
        # 800x600, and a wrapped line pushes the last item out of view.
        index = self.state.frame_index
        smoke_beat = self._story.ceiling_beat() if self._story is not None else None
        # Checked against the current frame, not _spoken_beats: this card
        # only opens on an explicit tap (a pull, never pushed at the
        # child -- see _narrate's own docstring), so it must describe
        # what is truly on screen right now even in OBSERVE, where the
        # beat's own mascot narration deliberately never fires.
        smoke_now = (smoke_beat is not None and index >= smoke_beat.frame_index)
        has_velocity = self.scene._velocity is not None
        airflow_now = (has_velocity
                       and kid.airflow_band_key(self.scene.mean_airspeed_at(index))
                       in kid.STRONG_AIRFLOW_KEYS)

        lines = []
        if smoke_now:
            lines.append(tr("seeing_smoke"))
        if airflow_now:
            lines.append(tr("seeing_airflow_now"))
        if has_velocity and not airflow_now:
            lines.append(tr("seeing_airflow_dots"))
        lines.append(tr("seeing_colors"))
        lines.append(tr("seeing_flame"))
        return lines[:4]

    def _on_help_requested(self) -> None:
        """Toggle the help card, pausing playback while it is open so a
        visitor reading it does not miss the run."""
        if self._help_open:
            self._close_help()
            return
        self._help_open = True
        self._resume_after_help = self.time_controller.is_playing()
        self.time_controller.pause()
        self.overlay.clear_buttons()
        self.overlay.show_card(tr("help_title"), self._what_am_i_seeing(), [])
        self.overlay.say(tr("help_say"), EXPLAINING)
        self.overlay.add_button(tr("help_got_it"), "👍", self.overlay.help_requested.emit,
                                primary=True)
        self.scene.refresh()

    def _close_help(self) -> None:
        """Put the phase's own chrome back and carry on from where the
        visitor paused.

        _render_phase() re-runs the phase handler, and the watching
        phases start playback from frame 0 -- so without restoring the
        index here, tapping "What am I seeing?" silently sent the run
        back to an unlit candle.
        """
        self._help_open = False
        resume_at = self.time_controller.index
        was_playing = self._resume_after_help
        self._resume_after_help = False
        self._render_phase()
        self.time_controller.pause()
        self.time_controller.seek(resume_at)
        if was_playing:
            self.time_controller.play()

    def _add_help_button(self) -> None:
        self.overlay.add_button(tr("help_button"), "🔍",
                                self.overlay.help_requested.emit)

    def _add_play_pause_button(self) -> None:
        """"Stop time" (see the NEXT LEVEL spec's section on it): a plain
        play/pause over the same TimeController everything else already
        drives. Its label is kept in sync by _sync_play_pause_label,
        connected once to TimeController.playing_changed in __init__, so
        it is always right regardless of *what* paused or resumed
        playback (this button, the help card, a replay) -- see
        _render_phase's note on why _play_pause_button is reset to None
        before every phase handler runs.
        """
        self._play_pause_button = self.overlay.add_button(
            tr("pause_button"), "⏸️", self._on_play_pause_clicked)
        self._sync_play_pause_label(self.time_controller.is_playing())

    def _on_play_pause_clicked(self) -> None:
        if self.time_controller.is_playing():
            self.time_controller.pause()
        else:
            self.time_controller.play()

    def _sync_play_pause_label(self, playing: bool) -> None:
        if self._play_pause_button is None:
            return
        self._play_pause_button.setText(
            f"⏸️  {tr('pause_button')}" if playing else f"▶️  {tr('play_button')}")

    def _nudge_before_beat(self, index: int) -> None:
        """One "look closely" a moment before the smoke beat lands.

        EXPERIMENT-only: the guided run is a deliberately entered
        activity the child is actively watching, so a heads-up before the
        beat lands is a fair courtesy there. Free exploration (OBSERVE)
        must stay calm and un-narrated -- see _narrate below.

        Timed off the detected beat rather than a fixed frame, so it
        builds anticipation for something that is genuinely about to
        happen -- and never fires at all in a run where the beat does
        not exist.
        """
        if self._nudged or self.state.phase is not Phase.EXPERIMENT:
            return
        beat = self._story.ceiling_beat() if self._story is not None else None
        if beat is None or beat.frame_index in self._spoken_beats:
            return
        if index < beat.frame_index - NUDGE_LEAD_FRAMES:
            return
        self._nudged = True
        self.overlay.say(tr("look_closely"), CURIOUS)

    def _narrate(self, index: int) -> None:
        """Speak the story beat for this frame, at most once per run.

        EXPERIMENT-only. Free exploration (OBSERVE) is meant to hold one
        sentence in a child's head -- "I am exploring a fire experiment"
        -- with nothing arriving unprompted; a beat firing mid-loop,
        independent of anything the child had just done, competed with
        that (see docs/HANDOFF-PUBLIC-MODE-REDESIGN.md §3/§4). The
        guided experiment is different: the child chose to run it, is
        actively watching one specific scenario, and the beat is real
        commentary on what they are watching, not ambient noise.

        Beats are tracked by frame index in `_spoken_beats` rather than by
        comparing bubble text: seeking backwards would otherwise
        re-trigger a beat every time it came round again. Once spoken, a
        message is left on screen until the next beat replaces it --
        clearing it after its window would blank the bubble mid-sentence
        for a slow reader.
        """
        if self._story is None or self.state.phase is not Phase.EXPERIMENT:
            return
        beat = self._story.beat_at(index, window=BEAT_VISIBLE_FRAMES)
        if beat is None or beat.frame_index in self._spoken_beats:
            return
        self._spoken_beats.add(beat.frame_index)
        beat_line = f"{beat.icon}  {tr(beat.text)}"
        self.overlay.say(tr(beat.reaction) if beat.reaction else beat_line, beat.mood)

    # -- phase rendering ------------------------------------------------
    def _render_phase(self) -> None:
        phase = self.state.phase
        # Any re-render invalidates a pending _defer_if_current callback
        # scheduled by whatever was on screen before -- see that method's
        # docstring for the stale-caption bug this prevents.
        self._interaction_generation += 1
        # A pending _play_until(...) is invalidated the same way: leaving
        # EXPERIMENT early (e.g. "Back to games" mid-run, before playback
        # ever reaches EXPERIMENT_END_FRAME) used to leave _play_end_frame/
        # _play_finished_cb set, armed and forgotten -- _on_time_changed
        # has no idea the run was abandoned, so the *next* unrelated
        # playback to cross that same frame index (a different game
        # entirely, potentially minutes later) fired the stale
        # _experiment_finished callback anyway: an uninvited REVEAL card
        # yanking the child out of whatever they were actually doing, and
        # (once quiz scoring existed) a phantom extra round added to the
        # score. A couple of render_* methods already defensively
        # cleared these two fields by hand (_start_free_play,
        # _render_attract); centralizing it here covers every phase
        # change instead of only the ones someone remembered to guard.
        self._play_end_frame = None
        self._play_finished_cb = None
        self.overlay.clear_buttons()
        self.overlay.clear_card()
        # Recreated (or not) by whichever handler runs below -- a stale
        # reference here would let playing_changed touch a button Qt has
        # already scheduled for deletion.
        self._play_pause_button = None
        handler = {
            Phase.ATTRACT: self._render_attract,
            Phase.INTRO: self._render_intro,
            Phase.OBSERVE: self._render_observe,
            Phase.PREDICTION: self._render_prediction,
            Phase.COUNTDOWN: self._render_countdown,
            Phase.EXPERIMENT: self._render_experiment,
            Phase.REVEAL: self._render_reveal,
            Phase.SCIENCE: self._render_science,
            Phase.COMPLETE: self._render_complete,
            Phase.GAMES_HOME: self._render_games_home,
            Phase.GAME_PLAY: self._render_game_play,
        }[phase]
        self.overlay.set_stage(_STAGE_FOR_PHASE.get(phase, -1))
        if phase is not Phase.COUNTDOWN:
            self.overlay.hide_countdown()
        # A tap reading or highlight ring from the previous screen must
        # never survive onto this one -- it would describe a point on a
        # frame the visitor is no longer looking at. The trail and the
        # "show before" ghost are different: they must survive a same-
        # screen re-render (closing Compare/Notebook/Help while still
        # looking at the exact same OBSERVE scenario, or the same Games
        # activity) -- a child's own measurements should feel persistent,
        # not wiped by an incidental UI change. They still reset the
        # moment the scenario actually changes or the screen is left
        # entirely -- both go through _clear_game_state(), called either
        # right here or from _on_explore_changed directly.
        self.overlay.set_probe_enabled(phase in PROBE_PHASES)
        same_screen_rerender = (
            phase == self._last_rendered_phase
            and (phase is Phase.OBSERVE
                 or (phase is Phase.GAME_PLAY
                     and self._active_game == self._last_rendered_game)))
        self._last_rendered_phase = phase
        self._last_rendered_game = self._active_game if phase is Phase.GAME_PLAY else None
        self.scene.clear_probe(keep_trail=same_screen_rerender)
        self._reset_probe_state()
        if phase not in (Phase.OBSERVE, Phase.GAME_PLAY):
            self.overlay.set_explore_visible(False)
            self._clear_game_state()
        if phase is not Phase.GAMES_HOME:
            self.overlay.set_games_home_visible(False)
        if phase not in (Phase.OBSERVE, Phase.GAMES_HOME, Phase.GAME_PLAY):
            self.overlay.set_nav_buttons([])
        handler()
        # Thermometer visibility/positioning runs *after* the handler,
        # not before: _position_thermometer reads the meter chips' real
        # geometry, and set_meters_visible is the handler's own call --
        # doing this earlier measured the *previous* phase's meter
        # visibility instead, a real bug that put the thermometer at the
        # wrong height (see set_thermometer_visible's docstring).
        self.overlay.set_thermometer_visible(phase in PROBE_PHASES)
        # Dr. Funke only stands in OBSERVE (see set_scientist_visible's
        # own docstring for why: the one phase the stage strip is
        # guaranteed hidden in, which her spot below the thermometer
        # relies on). Positioned after the thermometer for the same
        # reason the line above runs here and not earlier.
        self.overlay.set_scientist_visible(phase is Phase.OBSERVE)
        if phase is Phase.OBSERVE:
            self._arm_fact_timer()
        # Re-flow the mascot bubble now that the thermometer's real
        # visibility/position for *this* phase is known. handler() above
        # already called overlay.say() once (most phases open with one),
        # and that call's own bubble-width clamp ran against whatever
        # phase was on screen *before* this one -- e.g. entering OBSERVE
        # from INTRO, where the thermometer is hidden, sized the bubble
        # with no thermometer clamp at all, and nothing re-flowed it
        # afterward. A real, measured overlap: the bubble's own sizeHint
        # reached ~160px past the thermometer's left edge in an 800x600
        # screenshot.
        self.overlay._position_mascot()
        if phase in PROBE_PHASES:
            # REVEAL/SCIENCE are static (no playback tick to refresh it
            # through _on_time_changed), so without this the thermometer
            # could keep showing a stale reading from whatever phase ran
            # before -- a real but no-longer-current number.
            self._update_thermometer(self.state.frame_index)
        # Clear any ghost of the widgets just deleted -- see
        # PublicScene.refresh() for why the overlay cannot do this itself.
        self.scene.refresh()

    def _on_overlay_tapped(self, pos) -> None:
        """A tap (or drag-scan move) on the fire itself. Measures the
        real value at that point -- never an interpolated or invented
        one, see PublicScene.probe_at() -- unless the point lands on the
        candle itself, which gets its own explanation rather than a bare
        temperature reading. Explore and a Games activity react quite
        differently to the same tap, so this only measures and then
        dispatches by phase."""
        with self._diag_reentrancy_guard("_on_overlay_tapped"):
            if self.state.phase not in PROBE_PHASES:
                return
            result = self.scene.probe_at(pos)
            if result is None:
                return
            self._idle_vent_timer.stop()
            self._last_interaction_at = time.monotonic()
            x, z, value = result
            if self.state.phase is Phase.GAME_PLAY:
                self._on_game_tap(x, z, value)
            else:
                self._on_explore_tap(x, z, value)

    def _on_explore_tap(self, x: float, z: float, value: float) -> None:
        """A plain Explore tap: always just the real reading at that
        point (or, on the candle, what that object is) -- never a
        mini-game reaction, since a game can't be active outside
        Phase.GAME_PLAY. The in-diagram vent icons are a passive display
        of the real vent state (see PublicScene._draw_vent_object/
        _animate_vent), not a second control -- only the Vent 1/Vent 2
        buttons in the control bar change that state (_on_explore_changed)."""
        if self.scene.candle_hit(x, z):
            self.scene.pulse_flame()
            self._on_candle_tapped(value)
            return
        self._probe_mode = "point"
        self._probe_xz = (x, z)
        self.overlay.update_thermometer(value, f"🌡️ Air — {self.scene.location_phrase(x, z)}")
        self.overlay.update_thermometer_trail([])
        self.overlay.say(f"🌡️ {value:.0f}°C — {self.scene.location_phrase(x, z)}", CURIOUS)

    def _on_game_tap(self, x: float, z: float, value: float) -> None:
        """A tap while a Games/Challenges activity is on screen --
        dispatches on which one (self._active_game), reusing exactly the
        same handlers Explore's own tap used to call directly."""
        game = self._active_game
        if self.scene.candle_hit(x, z):
            self.scene.pulse_flame()
            # The real hottest cell is almost always at (or right beside)
            # the candle itself -- a Temp Hunt guess aimed squarely at
            # the flame is real game input, not something the plain
            # candle callout below should swallow silently.
            if game == "hottest" and self._temp_hunt_target == "hottest":
                self.overlay.update_thermometer(value, tr("thermometer_at_flame"))
                # The candle sprite is a fixed visual anchor, not a
                # guarantee the real fire is actually hot there *right
                # now* -- early in a run (or at some frames) the real
                # measured value at that exact point can still be near
                # ambient. A real, reported bug: tapping the icon
                # unconditionally celebrated "you found the hottest
                # place!" even for a 23 C reading. Same honesty check as
                # an ordinary (non-candle) tap -- see
                # _HOTTEST_TAP_RADIUS_M/_HOTTEST_VALUE_TOLERANCE_FRAC's
                # own comment -- so tapping the icon is never a free win.
                if self.scene.hottest_guess_is_close(x, z, value, self.state.frame_index):
                    self._celebrate(tr("found_flame_say", value=value))
                    self._record_discovery(
                        "🌡️", tr("found_flame_title"), tr("found_flame_discovery", value=value))
                else:
                    self.overlay.say(kid.heat_guess_reaction(value), CURIOUS)
                return
            if game == "hottest":
                # Hunting for the *coolest* place: the candle is
                # certainly not it -- explain what it is, but don't end
                # the hunt.
                self.overlay.update_thermometer(value, tr("thermometer_at_flame"))
                self.overlay.say(tr("candle_hottest_try_elsewhere"), CURIOUS)
                return
            if game == "hotcold":
                self.overlay.update_thermometer(value, tr("thermometer_at_flame"))
                self._handle_hotcold_tap(x, z, value)
                return
            if game != "map":
                self._on_candle_tapped(value)
                return
            # Map It's whole point is reading the real temperature
            # anywhere in the room, including right at the fire -- a tap
            # on the candle icon used to be swallowed by the generic
            # "here's what a candle is" callout above instead of adding
            # a real trail point the way every other tap does. Falls
            # through to the ordinary tap handling below, unchanged.
        self._probe_mode = "point"
        self._probe_xz = (x, z)
        self.overlay.update_thermometer(value, f"🌡️ Air — {self.scene.location_phrase(x, z)}")
        self.overlay.update_thermometer_trail([])
        if game == "hottest":
            self._handle_hottest_guess(x, z, value)
            return
        if game == "hotcold":
            self._handle_hotcold_tap(x, z, value)
            return
        if game == "mystery":
            # The whole point of this screen is comparing this exact
            # place before/after the fan -- every tap is that comparison,
            # no separate "arm" step needed. But before the child has
            # touched HVAC at all, case_index still equals
            # baseline_case_index -- a "same place" comparison against
            # itself is trivially "almost the same" and used to pop the
            # full before/after card on
            # literally the first curious tap, before there was
            # anything to compare (a real, reproduced dead end: the
            # card's only exit swallows the tap that should have gone
            # to the HVAC toggle). The plain thermometer reading already
            # given above, plus a nudge toward HVAC, is the right
            # response until there is a genuine "after."
            if self.state.case_index == self.state.baseline_case_index:
                self.overlay.say(
                    tr("mystery_try_the_fan_first", fan=self._fan_control_hint_phrase()), CURIOUS)
                return
            self._show_same_place_comparison(x, z)
            return
        revisit = self._match_before_trail(x, z)
        if revisit is not None:
            # Phase 8 section 7/9: tapping approximately where the child
            # already measured before their last Fan/Candle change reads
            # as revisiting that spot -- the verdict-first before/after
            # card pops, reached by returning to a place rather than
            # arming a tool first.
            self._show_same_place_comparison(revisit["x"], revisit["z"])
        elif game == "map":
            self._add_trail_point(x, z, value)

    def _on_candle_tapped(self, value_c: float) -> None:
        """The child touched the candle itself, not the surrounding air --
        answer with what it *is* rather than a bare number. `value_c` is
        still the real flame-cell temperature at that exact point (the
        same measurement any other tap would report), just phrased as an
        introduction instead of a reading."""
        self._probe_mode = "point"
        self._probe_xz = None
        self.overlay.update_thermometer(value_c, tr("thermometer_at_flame"))
        self.overlay.update_thermometer_trail([])
        # Draw the eye toward the real candle-count control right where
        # the child just touched the fire -- "here's how you change it",
        # without building a second floating picker (Phase 8 section 2).
        self.overlay.pulse_explore_toggle("candles")
        # A one-time nudge toward the real candle-count control, only
        # while nothing has been changed yet -- once the child has tried
        # the fan or candles, the invitation has already been answered
        # and repeating it would just be noise (Phase 7 section 1).
        if (not self._candle_nudge_shown
                and self.state.case_index == self.state.baseline_case_index):
            self._candle_nudge_shown = True
            self.overlay.say(tr("candle_burning_with_nudge"), EXCITED)
        else:
            self.overlay.say(tr("candle_burning"), EXCITED)

    def _current_factors(self) -> dict:
        """The real, currently-loaded scenario's own factor values for
        every exposed explore control (candles, vent1/vod, vent2/voc,
        door -- see EXPLORE_CONTROLS). `factors["door"] = 1` is only a
        fallback for a study where door isn't an available control (its
        own ExploreControl.held default); the loop below overwrites it
        with the real value whenever door *is* exposed, same as any
        other control. This is the starting point _on_explore_changed
        composes a single change on top of, so flipping one control
        keeps whatever the others are already set to instead of
        resetting them to a baseline the child may have already left
        (see _on_explore_changed's own docstring). Falls back to every
        control's own default if nothing is loaded yet."""
        entry = self.scene.current_entry()
        factors = {"door": 1}
        for control in self._explore_controls:
            factors[control.factor] = (getattr(entry, control.factor)
                                       if entry is not None else control.default_value)
        return factors

    def _on_explore_changed(self, control_key: str, value) -> None:
        """A free-play control was flipped: switch to the real matching
        scenario immediately -- combined with whatever the *other*
        controls are already set to, not reset against a fixed baseline.
        Flipping Fan ON and then Vent SHUT resolves the real fan-on +
        vent-shut scenario (the manifest has all combinations this
        study's factors produce); it does not silently discard the fan
        choice the way resolving against each control's own `held`
        baseline used to. See _current_factors().

        Meaningful during OBSERVE and during a Games activity that shows
        the Fan/Candle toggles (Mystery/Compare/Map It); a signal from a
        hidden panel (Temp Hunt/Hot-Cold don't show the toggles at all)
        is ignored. The toggle itself already gave the visitor its click
        feedback (Qt flips a checkable button's pressed state before this
        slot runs), so nothing here needs to fake an "acknowledged" step.
        """
        with self._diag_reentrancy_guard("_on_explore_changed"):
            if self.state.phase not in (Phase.OBSERVE, Phase.GAME_PLAY):
                return
            self._idle_vent_timer.stop()
            self._last_interaction_at = time.monotonic()
            control = next((c for c in self._explore_controls if c.key == control_key), None)
            if control is None:
                return
            factors = self._current_factors()
            factors[control.factor] = value
            case_index = experiments_mod.resolve_case_index(self.sim_data.manifest, factors)
            if case_index is None or case_index == self.state.case_index:
                return
            # Force the toggle's own new visual state onto the screen *now*,
            # before the potentially-expensive scene switch below (a full
            # canvas redraw through the cinema pipeline's layered artists can
            # take several hundred ms on this dataset). Qt only flips the
            # checked *state* synchronously here -- the repaint that shows it
            # is otherwise deferred until this whole slot returns, so without
            # this the button visually does nothing for as long as the scene
            # switch takes, reading as an unresponsive tap rather than an
            # instant one.
            #
            # TEMPORARY diagnostic prints bracket this specific call: a
            # synchronous repaint() is the one call in this whole slot that
            # can pump the event loop and re-enter application code before
            # this slot itself returns -- the prime suspect for the real
            # crash's reentrant Qt signal/slot chain.
            print("[DIAG] before overlay.repaint()", file=sys.stderr)
            self.overlay.repaint()
            print("[DIAG] after overlay.repaint()", file=sys.stderr)
            # Phase 8 section 7: whatever the child had already measured
            # becomes the "before" record for this change -- kept on screen
            # (dimmed) rather than wiped, so revisiting one of these spots
            # afterward reads as "I measured HERE before" (see
            # _match_before_trail) instead of requiring "Compare this place"
            # to be armed first.
            before_trail = list(self._temp_trail)
            active_game = self._active_game
            self._clear_game_state()
            self._active_game = active_game   # which Games activity this is stays put
            self._load_case(case_index)
            if before_trail:
                self._before_trail = before_trail
                # redraw=False on every point but the last: each add_trail_
                # marker() redraw is a full ~140 ms cinema-pipeline canvas
                # draw, so restoring N "before" points used to pay N of them
                # back-to-back -- one real, measured contributor to "why does
                # switching feel slow" for a child who had already tapped
                # around before flipping the Fan/Candle toggle.
                last = len(self._before_trail) - 1
                for index, point in enumerate(self._before_trail):
                    self.scene.add_trail_marker(point["x"], point["z"], index + 1, dim=True,
                                                redraw=index == last)
            # _label_for_case can now list more than one differing factor
            # (see its own docstring) but only ever names them in words, no
            # icon -- this control's own icon leads, same as before, since
            # it is the one the child just touched.
            self._last_change = f"{control.icon} {self._label_for_case(case_index)}"
            # A short, physical reaction rather than a caption -- "the child
            # should see the consequence before reading any number" (Phase 3
            # section 2). The banner briefly takes over in the accent colour
            # (same flash mechanism a detected story beat already uses) so it
            # reads as a felt event, not a settings confirmation.
            if control_key == "vent1":
                # value is the real vod level (0=open, 1=closed, 2=HVAC
                # fan) -- only the fan state itself is a "whoosh"; closed
                # is not a truthy leftover of the old open/HVAC-only
                # control.
                self.overlay.banner.flash(tr("banner_whoosh") if value == 2 else tr("banner_quiet_again"))
                self.overlay.say(tr("explore_changed_fan"), CURIOUS)
            else:
                self.overlay.say(tr("explore_changed_other"), CURIOUS)
            self._start_free_play()
            if self.state.phase is Phase.GAME_PLAY:
                self._refresh_game_buttons()

    _VENT_PULSE_MS = 260

    # TEMPORARY -- crash investigation instrumentation, see _diag_depths'
    # own comment in __init__. Wraps a suspected call site so a repro run
    # can show whether it's genuinely re-entered while already on the
    # call stack (the signature the real crash's C-level trace showed:
    # several chained Qt signal emits before an abort()), rather than
    # guessing from binary offsets alone. Prints to stderr, not the
    # logging module, so it survives even if something about logging
    # setup is itself implicated.
    @contextlib.contextmanager
    def _diag_reentrancy_guard(self, name: str):
        depth = self._diag_depths.get(name, 0) + 1
        self._diag_depths[name] = depth
        if depth > 1:
            print(f"[DIAG] REENTRANT: {name} depth={depth}", file=sys.stderr)
            traceback.print_stack(file=sys.stderr)
        else:
            print(f"[DIAG] enter {name}", file=sys.stderr)
        try:
            yield
        finally:
            print(f"[DIAG] exit {name} (was depth {depth})", file=sys.stderr)
            self._diag_depths[name] = depth - 1

    def _defer(self, delay_ms: int, callback) -> None:
        """A single-shot delay, parented to this experience -- never the
        bare QtCore.QTimer.singleShot(...) staticmethod, whose timer has
        no parent to tie its lifetime to. A bare singleShot's callback
        still fires even after the window (and everything the callback
        touches) has been destroyed -- e.g. between one test's teardown
        and the next test's setup -- which segfaults the whole process
        rather than raising a catchable exception. TitleBanner.flash()
        already established this same fix (a QTimer parented to the
        widget it restores); this generalizes it for callers that don't
        own a persistent timer of their own.
        """
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(callback)
        timer.start(delay_ms)

    def _defer_if_current(self, delay_ms: int, callback) -> None:
        """Like _defer, but the callback only runs if nothing has
        re-rendered or loaded a new scenario since this was scheduled
        (see _interaction_generation) -- prevents a delayed thermometer
        sweep from stomping a caption/value that no longer describes
        what's on screen. A real, reproducible bug this closes: tap ->
        same-place sweep scheduled -> switch scenario within the sweep's
        ~300 ms delay -> the old sweep's second half still fired and
        overwrote the thermometer with a stale reading from a comparison
        the child had already moved on from.
        """
        generation = self._interaction_generation

        def guarded():
            if generation == self._interaction_generation:
                callback()
        self._defer(delay_ms, guarded)

    def _start_free_play(self) -> None:
        """Loop the scenario currently loaded, indefinitely.

        Free exploration must never hand the child back to the guided
        narrative on a timer -- see the NEXT MILESTONE note on removing
        the old _play_until(..., self._observe_finished) auto-advance.
        The child leaves OBSERVE only by pressing "Test an idea"
        themselves (_on_test_idea_clicked). Looping (rather than just
        playing once to the last frame and stopping) means the fire
        keeps burning for as long as they want to watch, tap, or drag
        the probe, exactly like the ATTRACT loop before anyone arrives.
        """
        # _play_end_frame/_play_finished_cb are cleared centrally in
        # _render_phase now (see its own comment); no longer done here.
        self.time_controller.set_loop(True)
        # Skipped when already at 0 -- TimeController.seek() unconditionally
        # emits time_changed regardless of whether the index actually moved
        # (see its own docstring/_play_until's comment on the same
        # inefficiency), so every single Vent/Candle/Door tap used to pay
        # for two full cinema-pipeline renders of frame 0 in a row: one
        # from _load_case's own seek(0) (run just before this, on the
        # explore-control-change path -- see _on_explore_changed), and an
        # unconditional second one here. _load_case doesn't need the fix
        # itself since it's the *first* seek in that sequence; this is the
        # redundant *second* one. Harmless to check unconditionally here
        # (every other call site -- _render_observe, _render_attract's own
        # loop-restart -- genuinely does need the seek, and a plain index
        # comparison costs nothing when it does).
        if self.time_controller.index != 0:
            self.time_controller.seek(0)
        self.time_controller.play()

    def _sync_explore_controls_to_case(self) -> None:
        """Snap every toggle's visual state to the scenario actually
        loaded right now -- read directly off its own factor value
        (entry.vod, entry.candles, entry.voc), not a blind reset to
        defaults, which would desync the toggles from an already-
        explored scenario every time OBSERVE re-renders without a fresh
        baseline load (closing the Help or Compare detour). Reads the
        real entry rather than re-resolving through each control's own
        case_for/held: since _on_explore_changed composes changes now
        (see its own docstring), a loaded scenario can have more than
        one factor away from baseline at once, which case_for -- built
        to resolve a single factor against a fixed baseline -- has no
        way to recognise."""
        entry = self.scene.current_entry()
        if entry is None:
            return
        values = {control.key: getattr(entry, control.factor, control.default_value)
                 for control in self._explore_controls}
        self.overlay.set_explore_values(values)

    def _on_test_idea_clicked(self) -> None:
        """The child's own decision to leave free play for the guided
        prediction -- see _start_free_play's docstring for why nothing
        else may call _observe_finished on a timer any more."""
        self.time_controller.pause()
        self.time_controller.set_loop(False)
        self._observe_finished()

    # -- "what changed?" compare (free play, not the guided reveal) -----
    def _case_measurement(self, case_index):
        if case_index not in self._case_measurements:
            self._case_measurements[case_index] = experiments_mod.measure_case(
                self.sim_data.store, case_index)
        return self._case_measurements[case_index]

    def _label_for_case(self, case_index) -> str:
        """A short, honest label for one explored scenario, listing every
        factor that differs from the shared baseline (not just the one
        the child last touched -- _on_explore_changed composes changes
        now, so a case can be away from baseline on more than one
        factor at once, e.g. "Fan ON, Vent SHUT"). Never a hardcoded
        "Before"/"After". Falls back to a neutral label when nothing
        differs (the baseline scenario itself)."""
        entry = next((e for e in self.sim_data.manifest if e.case_index == case_index), None)
        if entry is None:
            return tr("this_scenario_fallback")
        parts = []
        for control in self._explore_controls:
            value = getattr(entry, control.factor, None)
            if value == control.default_value:
                continue
            option = next((o for o in control.options if o.value == value), None)
            if option is not None:
                parts.append(f"{control.label} {option.label}")
        return ", ".join(parts) if parts else tr("this_scenario_fallback")

    def _add_why_button(self, explanation: str) -> None:
        """A one-shot "❓ Why?" reveal, offered only after a real
        discovery/comparison and only on request (Phase 6 section 13) --
        explanations stay out of the way until the child has already
        seen the *effect* for themselves. `explanation` is always a
        declared string (a Metric.explanation field, or one of the small
        set written for the mystery/same-place moments below) -- never
        composed from the current experiment's identity, so nothing here
        special-cases which factor changed."""
        if not explanation:
            return
        self.overlay.add_button(tr("why_button"), "❓", lambda: self.overlay.say(explanation, EXPLAINING))

    def _open_compare_detour(self) -> None:
        """Shared pause-and-look setup for every real comparison surface
        (the Discovery Notebook, Mystery's same-place comparison) -- a
        detour modelled on _on_help_requested, not a phase change:
        closing it (_close_compare) must return to exactly where
        OBSERVE was.

        The comparison card carries real bar charts (same widgets the
        science card uses) and is genuinely tall -- the meters and
        explore toggles must step aside the way REVEAL/
        SCIENCE already do, or the card collides with them at 800x600 (a
        real, screenshotted overlap this replaced). The banner (still
        showing whatever the game last flashed) is the same row the
        meters shared and easily costs ~95 px on its own -- with two real
        bar charts to fit into 600 px total, that alone was enough to
        compress the card below its own sizeHint (also a real, measured
        clip, not a hypothetical one). _close_compare's own
        _render_phase() call restores all of it.
        """
        self._compare_open = True
        self._resume_after_compare = self.time_controller.is_playing()
        self.time_controller.pause()
        self.overlay.clear_buttons()
        self.overlay.set_meters_visible(False)
        self.overlay.set_explore_visible(False)
        self.overlay.set_prompt("")

    def _close_compare(self) -> None:
        self._compare_open = False
        was_playing = self._resume_after_compare
        self._resume_after_compare = False
        resume_at = self.time_controller.index
        self._render_phase()
        self.time_controller.pause()
        self.time_controller.seek(resume_at)
        if was_playing:
            self.time_controller.play()

    def _note_mystery_progress(self, zone: str, delta: float, x: float, z: float) -> str:
        """Called with a real same-place comparison's own location/delta
        (already gated noticeable by the caller). Returns a short
        reaction line, or "" to fall back to the normal change_line.
        Only ever reports what has *actually* been measured -- never
        assumes the fan is what changed (a candle-count same-place
        comparison simply never matches zone "ceiling"/"floor" at all).

        `zone` is the stable, language-independent id from
        PublicScene.location_zone() ("ceiling"/"floor"/...), not the
        translated display phrase -- this logic must keep working
        identically regardless of the current UI language.

        Every noticeable ceiling or floor reading counts as progress in
        that zone, in whichever direction it actually went -- the
        direction is *not* assumed (see this method's own state-field
        comments on why an assumed direction was the actual bug). The
        "aha" only fires once both zones have a reading AND their
        directions genuinely disagree (the real condition "same fan,
        different place" depends on); two readings that happen to agree
        get an honest, still-encouraging line instead of a false
        "solved". Records the real (x, z) of each zone (Phase 12 section
        11) so the "aha" moment can highlight the *actual* two spots the
        child measured, not invented ones."""
        warmer = delta > 0
        if zone == "ceiling":
            self._mystery_found_high = True
            self._mystery_high_xz = (x, z)
            self._mystery_high_warmer = warmer
        elif zone == "floor":
            self._mystery_found_low = True
            self._mystery_low_xz = (x, z)
            self._mystery_low_warmer = warmer
        else:
            return ""
        direction = tr("direction_warmer") if warmer else tr("direction_cooler")
        if self._mystery_found_high and self._mystery_found_low:
            if self._mystery_high_warmer != self._mystery_low_warmer:
                self._mystery_solved = True
                self._record_discovery(
                    "🔎", tr("mystery_solved_discovery_title"),
                    tr("mystery_solved_discovery",
                       high=tr("direction_warmer") if self._mystery_high_warmer
                       else tr("direction_cooler"),
                       low=tr("direction_warmer") if self._mystery_low_warmer
                       else tr("direction_cooler")))
                return tr("mystery_solved_title")
            return tr("mystery_both_same_direction", direction=direction)
        icon = "🌡️" if warmer else "❄️"
        where = tr("location_near_ceiling" if zone == "ceiling" else "location_near_floor")
        return tr("mystery_progress_zone", icon=icon, direction=direction, where=where)

    def _show_same_place_comparison(self, x: float, z: float) -> None:
        """The real measured temperature at physical (x, z), read from
        BOTH the baseline scenario and whatever the child has switched
        to -- PublicScene.measure_case_at() samples each scenario's own
        stored array directly, so this never needs to actually reload
        either scenario into the scene just to read one point.

        The card leads with a qualitative verdict (much hotter / much
        cooler / almost the same) rather than the raw degrees -- "the
        main visual should be the difference, not the numbers" (Phase 7
        section 3). The verdict comes straight out of the *same*
        noticeable_delta threshold the rest of Fire Lab already uses to
        decide whether a change is worth mentioning at all; no new
        magnitude tier is invented, and the exact °C values stay visible
        in the BarCompare below for anyone who wants them."""
        baseline_case = self.state.baseline_case_index
        current_case = self.state.case_index
        baseline_val = self.scene.measure_case_at(baseline_case, x, z)
        current_val = self.scene.measure_case_at(current_case, x, z)
        if baseline_val is None or current_val is None:
            return
        self._open_compare_detour()
        zone = self.scene.location_zone(x, z)
        where = self.scene.location_phrase(x, z)
        before_label = self._label_for_case(baseline_case)
        after_label = self._label_for_case(current_case)
        room_temp = next(m for m in experiments_mod.PUBLIC_METRICS if m.key == "room_temp")
        delta = current_val - baseline_val
        noticeable = abs(delta) >= room_temp.noticeable_delta
        if noticeable:
            direction = tr("direction_warmer") if delta > 0 else tr("direction_cooler")
            verdict = tr("verdict_much_hotter") if delta > 0 else tr("verdict_much_cooler")
            change_line = tr("change_line_delta", delta=abs(delta), direction=direction)
            self._last_watched = tr("last_watched_temp", where=where, delta=abs(delta),
                                    direction=direction)
        else:
            verdict = tr("verdict_almost_same")
            change_line = tr("change_line_barely")
        bar = BarCompare(before_label, baseline_val, after_label, current_val, "°C", 1)
        mascot_line = verdict
        title = verdict
        solved_before = self._mystery_solved
        if noticeable:
            mascot_line = self._note_mystery_progress(zone, delta, x, z) or verdict
        mystery_just_solved = self._mystery_solved and not solved_before
        if mystery_just_solved:
            title = tr("mystery_solved_title")
            # The scene itself is the climax, not just the card (Phase 12
            # section 11): both real measured spots glow together for a
            # couple of seconds, reusing the exact dual-marker artists
            # the Hot/Cold mini-game already draws with -- no new marker
            # style to invent, and the highlighted points are the real
            # (x, z) pairs the child actually tapped.
            self.scene.show_dual_markers(self._mystery_low_xz, self._mystery_high_xz)
            self._defer_if_current(2000, self.scene.clear_dual_markers)
        lines = [f"📍 {where}", change_line]
        self.overlay.show_card(
            title, lines,
            hero=[HeroMetric(tr("hero_temp_at_spot"), bar)])
        if mystery_just_solved:
            self._celebrate(mascot_line)
        else:
            self.overlay.say(mascot_line, EXCITED if noticeable else CURIOUS)
        self.overlay.add_button(tr("keep_exploring"), "🕯️", self._close_compare, primary=True)
        if mystery_just_solved:
            self._add_why_button(tr("why_moving_air"))
        elif noticeable:
            self._add_why_button(tr("why_hot_air_rises"))
        # The thermometer sweeps baseline -> current -- "I didn't compare
        # two random places, I compared the SAME place before and after"
        # (Phase 3 section 6) is best felt by watching one instrument
        # move between the two real values, not just reading two numbers.
        self.overlay.update_thermometer(baseline_val, f"🌡️ {before_label}")
        self._defer_if_current(
            340, lambda: self.overlay.update_thermometer(current_val, f"🌡️ {after_label}"))

    # -- Games/Challenges: Temp Hunt, Hot/Cold --------------------------
    # (entry points are _enter_game_hottest/_enter_game_hotcold, below;
    # these two just handle the tap once that game is on screen)
    def _handle_hottest_guess(self, x: float, z: float, value_c: float) -> None:
        target = self._temp_hunt_target
        if target == "hottest":
            close = self.scene.hottest_guess_is_close(x, z, value_c, self.state.frame_index)
        else:
            close = self.scene.coolest_guess_is_close(x, z, value_c, self.state.frame_index)
        if close:
            icon = "🌡️" if target == "hottest" else "🧊"
            target_word = tr("target_hottest" if target == "hottest" else "target_coolest")
            self._celebrate(tr("found_spot_say", target=target_word, icon=icon, value=value_c))
            self._record_discovery(
                icon, tr("found_spot_title", target=target_word.upper()),
                tr("found_spot_discovery", target=target_word, value=value_c))
        else:
            self.overlay.say(kid.heat_guess_reaction(value_c), CURIOUS)

    def _handle_hotcold_tap(self, x: float, z: float, value_c: float) -> None:
        # A tap right after a completed round (_hotcold_stage == 2) starts
        # a fresh one rather than being ignored -- "tap anywhere to try
        # again" (see _render_game_hotcold).
        if self._hotcold_stage in (0, 2):
            if self._hotcold_stage == 2:
                self.scene.clear_dual_markers()
            self._hotcold_readings = {"hot": (x, z, value_c)}
            self._hotcold_stage = 1
            self.overlay.say(tr("hotcold_find_second"), CURIOUS)
            return
        hot_x, hot_z, hot_c = self._hotcold_readings.get("hot", (x, z, value_c))
        cool_c = value_c
        self._hotcold_stage = 2
        # Both locations marked on the heatmap at once, and the
        # thermometer swept from one to the other -- "find somewhere hot,
        # find somewhere cool" only reads as a comparison once both are
        # visible together (Phase 3 section 5), not as two separate taps.
        self.scene.show_dual_markers((hot_x, hot_z), (x, z))
        self.overlay.update_thermometer(hot_c, tr("thermometer_hot_spot"))
        self._defer_if_current(
            320, lambda: self.overlay.update_thermometer(cool_c, tr("thermometer_cool_spot")))
        big_difference = abs(hot_c - cool_c) >= HOTCOLD_BIG_DIFFERENCE_C
        diff_word = tr("hotcold_diff_big" if big_difference else "hotcold_diff_small")
        if big_difference:
            self._celebrate(tr("hotcold_say", hot=hot_c, cool=cool_c, diff=diff_word))
            self._record_discovery(
                "🌡️", tr("hotcold_discovery_title"),
                tr("hotcold_discovery_text", hot=hot_c, cool=cool_c))
        else:
            # No confetti/chime/discovery for this one -- the whole point
            # of the game is a *big* contrast, not any two numbers that
            # happen to differ (see HOTCOLD_BIG_DIFFERENCE_C). The nudge
            # names what to do differently rather than just repeating
            # the numbers back.
            self.overlay.say(
                tr("hotcold_say", hot=hot_c, cool=cool_c, diff=diff_word), CURIOUS)
            self._defer_if_current(
                1400, lambda: self.overlay.say(tr("hotcold_try_bigger_gap"), THINKING))

    def _render_attract(self) -> None:
        self.overlay.set_meters_visible(False)
        self.overlay.set_prompt(tr("attract_prompt"))
        if not self._available:
            self.overlay.say(tr("attract_unavailable"), IDLE)
            self.overlay.add_button(tr("back_to_app"), "←",
                                    self.exit_requested.emit, primary=True)
            return
        # The attract loop is the pitch: real fire, real smoke, real
        # airflow, already moving behind a single enormous invitation. A
        # visitor arriving mid-loop should need no instructions.
        self.overlay.set_meters_visible(True)
        self.overlay.say(tr("attract_say"), CURIOUS)
        self.overlay.add_button(tr("attract_button"), "🕯️",
                                self.overlay.start_requested.emit, primary=True, tall=True)
        # Idle attract: loop the baseline fire quietly behind the invite.
        # (_play_end_frame/_play_finished_cb are cleared centrally in
        # _render_phase now.)
        self.time_controller.set_loop(True)
        self.time_controller.seek(0)
        self.time_controller.play()

    def _begin_journey(self) -> None:
        self.time_controller.pause()
        self.time_controller.set_loop(False)
        self.state.go_to(Phase.INTRO)
        self._render_phase()
        self._prewarm_stories()

    def _prewarm_stories(self) -> None:
        """Build the StoryController for every scenario this visit could
        reach, before any of them are needed -- and, in the same pass,
        warm the VELOCITY array ScenarioStore.get() will otherwise only
        fetch cold the first time that scenario is actually switched to.

        Measured: ScenarioStore's own read is sub-millisecond once warm
        (this whole dataset already is), but the descriptor/event
        detection StoryController runs costs ~20-30 ms per scenario and
        is cached per case_index -- so it was the actual dominant cost
        behind a perceptible hitch the first time a visitor tapped an
        explore control or reached the guided experiment's own scenario.
        Run once here, while the intro card is still being read (a Qt
        event-loop return already happens for the tap that got here, so
        this delays the *next* paint by ~50-75 ms rather than anything
        already on screen), so that path is already warm by the time it
        matters.

        _story_for() only ever reads DEFAULT_SLICE_KEY (temperature) --
        PublicScene.load_case() separately loads VELOCITY_KEY every
        scenario switch (see PublicScene._load_velocity), which this loop
        never used to touch. That left the *first* real Fan-toggle tap
        paying a full, uncached parse of the raw velocity slice file --
        the actual "feels slow" moment this closes. Wrapped in try/except
        the same way _load_velocity already treats velocity as optional:
        a missing/corrupt velocity file must never block prewarming or
        entry into OBSERVE, only skip this one scenario's warm-up.
        """
        seen = set()
        for control in self._explore_controls:
            for option in control.options:
                case = control.case_for(self.sim_data.manifest, option.value)
                if case is not None and case not in seen:
                    seen.add(case)
                    self._story_for(case)
        for choice in self.experiment.choices:
            case = experiments_mod.resolve_choice(
                self.sim_data.manifest, self.experiment, choice.key)
            if case is not None and case not in seen:
                seen.add(case)
                self._story_for(case)
        for case in seen:
            try:
                self.sim_data.store.get(case, VELOCITY_KEY)
            except Exception as e:  # noqa: BLE001 - velocity is an enhancement, never fatal
                logger.info("prewarm: velocity unavailable for case %s (%s)", case, e)

    def _render_intro(self) -> None:
        self.overlay.set_meters_visible(False)
        self.overlay.set_prompt("")
        # One sentence, not four. The previous version was ~55 words of
        # adult prose on a dark, static screen -- the single least
        # child-readable moment in the journey.
        self.overlay.show_card(
            tr("intro_title"),
            [tr("intro_body")],
            [tr("intro_source")])
        self.overlay.say(tr("intro_say"), CURIOUS)
        self.overlay.add_button(tr("watch_the_fire"), "🕯️", self._start_observe, primary=True)

    def _start_observe(self) -> None:
        self.state.go_to(Phase.OBSERVE)
        self._render_phase()

    def _observe_end_frame(self) -> int:
        """When the observe phase may hand over to the prediction.

        Driven by the data, not a fixed frame: the visitor must have seen
        the smoke gather under the ceiling, and had time to read the line
        about it, before being asked what the fan will do. Measured on
        the two fan scenarios the beat lands at frame 64 (16 s) with the
        fan off and frame 39 (9.8 s) with it on, so this typically ends
        around frame 160 -- sooner than the old fixed 200, and unlike it,
        guaranteed to include the beat.

        A run with no smoke beat falls back to a fixed window rather than
        waiting for something that never happens.
        """
        last_frame = max(self.scene.frame_count() - 1, 0)
        beat = self._story.ceiling_beat() if self._story is not None else None
        end = (beat.frame_index + OBSERVE_TAIL_FRAMES if beat is not None
               else OBSERVE_FALLBACK_END_FRAME)
        return max(OBSERVE_MIN_FRAMES, min(end, last_frame))

    def _render_observe(self) -> None:
        """Explore: the always-available free-play screen. Three things
        a child can always do here -- tap anywhere to measure, flip
        Fan/Candles, watch the simulation respond. Every optional
        activity (the mini-games, comparisons, the guided "Test an
        idea" journey, the discovery notebook) lives one tap away in
        Games, entered intentionally via the small button below rather
        than surfacing here unasked.

        Dr. Funke (see PublicOverlay.Scientist -- ambient scoping via
        _arm_fact_timer, below) is a deliberate, explicitly-requested
        exception to that "nothing arrives unprompted" rule the rest of
        this phase still follows: her facts are general fire-science
        content, not a claim about this run, standing quietly in the
        instrument sidebar rather than competing with the scene itself
        for the centre of the screen.
        """
        self.overlay.set_prompt(self.experiment.observe_prompt)
        self._nudged = False
        self.overlay.set_meters_visible(True)
        self.overlay.say(tr("explore_say"), WATCHING)
        self._add_help_button()
        self._add_play_pause_button()
        # A small nav pill, not a third BigButton: two BigButtons (Help,
        # Play/Pause) plus a third measured 729 px wide at 800x600, 81 px
        # past the thermometer's own reserved column -- a real overlap
        # neither button_row nor the thermometer's own button-avoidance
        # clamp can resolve (see PublicOverlay.nav_row's own comment).
        self.overlay.set_nav_buttons([(tr("games_button"), "🎮", self._on_games_requested, True)])
        self.overlay.set_explore_visible(True)
        self._sync_explore_controls_to_case()
        self._start_free_play()
        self._arm_idle_vent_hint()

    def _arm_fact_timer(self) -> None:
        """Schedule Dr. Funke's next fact -- a real delay before the
        first one (_FACT_FIRST_DELAY_MS) so it doesn't fire the instant
        OBSERVE appears, before a child has looked at anything.

        _render_phase calls this every time it renders OBSERVE,
        including a same-screen re-render (e.g. _on_back_to_explore,
        returning from the Games hub without ever leaving OBSERVE) --
        so this can run more than once per visit while a previous
        chain's timers are still pending. _fact_chain_token is what
        keeps those from stacking: each call claims a fresh token, and
        every step below only proceeds if its own token is still the
        current one, so an old chain quietly stops the moment a new one
        is armed instead of running in parallel with it.

        Plain _defer, deliberately not _defer_if_current: the latter's
        own generation guard exists to stop a *stale computed value*
        (e.g. a thermometer sweep computed against a scenario that's
        since changed) from landing late -- wrong here, since _load_case
        bumps that same generation on every Fan/Candle/Vent toggle, and
        this chain never carries a stale value forward; each step
        re-reads live state (phase, the lull check) the instant it
        actually runs. Using _defer_if_current here was a real, latent
        bug: the very first toggle during OBSERVE would silently and
        permanently stop the whole fact cycle for the rest of that
        visit, since nothing else re-arms it short of leaving and
        re-entering OBSERVE. Plain _defer still avoids the bare-
        QTimer.singleShot crash risk (see _defer's own docstring) --
        it's just not tied to that unrelated generation; _fact_chain_
        token is the guard that actually belongs to this chain."""
        self._fact_chain_token += 1
        token = self._fact_chain_token
        self._defer(_FACT_FIRST_DELAY_MS, lambda: self._show_next_fact(token))

    def _is_a_lull(self) -> bool:
        """Whether this is a genuinely quiet moment for Dr. Funke to
        speak into -- neither mascot's own bubble system distinguishes
        "on screen" from "just said something," so this checks *recency*
        of each: the primary guide's own OBSERVE invitation stays
        visible for the whole phase once said (PublicOverlay.say has no
        auto-hide), so a bare `bubble.isVisible()` check would block her
        forever. Both conditions, not either -- a fact landing right on
        top of a fresh reaction line, or right as a child's finger is
        still on the glass, is exactly the "second character adding to
        the overload" this whole deferral scheme exists to avoid."""
        return (self.overlay.seconds_since_say() >= _LULL_GRACE_S
                and time.monotonic() - self._last_interaction_at >= _LULL_GRACE_S)

    def _show_next_fact(self, token: int) -> None:
        if token != self._fact_chain_token or self.state.phase is not Phase.OBSERVE:
            return
        if not self._is_a_lull():
            # Not silently dropped -- retried shortly (plain _defer, see
            # _arm_fact_timer's own comment on why not _defer_if_current),
            # so a fact merely postponed by a stray tap still gets its
            # ambient turn once things go quiet, rather than needing to
            # wait a full _FACT_GAP_MS for the next scheduled attempt.
            self._defer(_FACT_RETRY_MS, lambda: self._show_next_fact(token))
            return
        if not self._fact_queue:
            self._fact_queue = list(FIRE_FACT_KEYS)
            random.shuffle(self._fact_queue)
        key = self._fact_queue.pop()
        self.overlay.say_fact(tr(key))
        self._defer(_FACT_SHOW_MS, lambda: self._hide_fact_and_wait_for_more(token))

    def _hide_fact_and_wait_for_more(self, token: int) -> None:
        if token != self._fact_chain_token or self.state.phase is not Phase.OBSERVE:
            return
        self.overlay.clear_fact()
        self._defer(_FACT_GAP_MS, lambda: self._show_next_fact(token))

    def _on_games_requested(self) -> None:
        self.time_controller.pause()
        self.state.go_to(Phase.GAMES_HOME)
        self._render_phase()

    def _on_language_requested(self, lang: str) -> None:
        """Switch languages instantly, no restart: _render_phase() already
        fully re-declares every phase's text from live tr()/kid_language
        calls on every render (see its own module docstring), so the only
        extra step is the handful of labels set once at construction and
        never revisited (overlay.retranslate()) plus the always-visible
        meters, which _render_phase() doesn't touch every frame on its
        own.

        The Fan/Candles explore toggles are a second such case:
        set_explore_controls() builds each ExploreToggle's caption/option
        buttons once, from ExploreControl/ExploreOption.label at that
        moment -- unlike everything else here, it is never called again
        by a normal render (_render_observe just resyncs *values* via
        _sync_explore_controls_to_case). Rebuilding them here picks up
        the new language; the resync that follows via _render_phase()
        restores whichever option is actually checked for the loaded
        scenario, which a fresh build always resets to the default."""
        i18n.set_language(lang)
        self.overlay.set_language_active(lang)
        self.overlay.retranslate()
        self.overlay.set_explore_controls(self._explore_controls)
        self._render_phase()
        if self.state.phase in PROBE_PHASES:
            self.overlay.update_meters(
                self.scene.mean_temperature_at(self.state.frame_index),
                self.scene.mean_airspeed_at(self.state.frame_index),
                has_velocity=self.scene._velocity is not None)

    def _on_back_to_explore(self) -> None:
        self.state.go_to(Phase.OBSERVE)
        self._render_phase()

    def _on_back_to_games(self) -> None:
        self.state.go_to(Phase.GAMES_HOME)
        self._render_phase()

    # -- Games/Challenges hub -------------------------------------------
    def _render_games_home(self) -> None:
        self.overlay.set_meters_visible(False)
        self.overlay.set_explore_visible(False)
        self.overlay.set_prompt(tr("games_hub_prompt"))
        self.overlay.say(tr("games_hub_say"), CURIOUS)
        self.overlay.set_game_tiles([
            ("🌡️", tr("tile_temp_hunt"), self._enter_game_hottest),
            ("🧊", tr("tile_hot_cold"), self._enter_game_hotcold),
            ("🔎", tr("tile_mystery"), self._enter_game_mystery),
            ("🧪", tr("tile_test_idea"), self._enter_game_test_idea),
            ("🗺️", tr("tile_map_it"), self._enter_game_map),
        ])
        nav = []
        if self._discoveries:
            nav.append((tr("my_discoveries"), "📓", self._on_notebook_requested))
        nav.append((tr("back_to_exploring"), "🔙", self._on_back_to_explore))
        self.overlay.set_nav_buttons(nav)

    # -- Games/Challenges: one active game ------------------------------
    def _enter_game_hottest(self) -> None:
        self._clear_game_state()
        self._active_game = "hottest"
        self._temp_hunt_target = random.choice(("hottest", "coolest"))
        self.state.go_to(Phase.GAME_PLAY)
        self._render_phase()

    def _enter_game_hotcold(self) -> None:
        self._clear_game_state()
        self._active_game = "hotcold"
        self.state.go_to(Phase.GAME_PLAY)
        self._render_phase()

    def _enter_game_mystery(self) -> None:
        self._clear_game_state()
        self._active_game = "mystery"
        self.state.go_to(Phase.GAME_PLAY)
        self._render_phase()

    def _enter_game_map(self) -> None:
        self._clear_game_state()
        self._active_game = "map"
        self.state.go_to(Phase.GAME_PLAY)
        self._render_phase()

    def _enter_game_test_idea(self) -> None:
        """"Test an idea" never renders its own Games chrome -- it jumps
        straight into the existing, untouched guided journey (PREDICTION
        -> COUNTDOWN -> EXPERIMENT -> REVEAL -> SCIENCE), the same one a
        "Test an idea" button used to reach directly from Explore.
        return_phase makes "Try again"/"Start again" land back on the
        Games hub instead of dropping the child into free play."""
        self.state.return_phase = Phase.GAMES_HOME
        self._on_test_idea_clicked()

    def _render_game_play(self) -> None:
        self.overlay.set_meters_visible(True)
        handler = {
            "hottest": self._render_game_hottest,
            "hotcold": self._render_game_hotcold,
            "mystery": self._render_game_mystery,
            "map": self._render_game_map,
        }.get(self._active_game)
        if handler is None:
            # Shouldn't happen, but never show a blank screen.
            self.state.go_to(Phase.GAMES_HOME)
            self._render_phase()
            return
        handler()
        self._refresh_game_buttons()
        # Resume wherever the scenario's own loop already was -- never a
        # fresh _start_free_play() here, which would reseek to frame 0
        # and silently invalidate whatever frame-dependent measurement
        # (e.g. Temp Hunt's real hottest-point-at-this-frame target) was
        # already in progress. The Games hub only pauses playback
        # (_on_games_requested); entering a game just un-pauses it.
        self.time_controller.play()

    def _refresh_game_buttons(self) -> None:
        """(Re-)declare the current game's own action buttons -- called
        both by _render_game_play (a full re-render) and by
        _on_explore_changed/_add_trail_point/_on_clear_trail_requested
        (a partial refresh, so a Fan/Candle change or a trail edit
        doesn't force a full _render_phase() and risk disturbing the
        trail still drawn on the scene).

        button_row keeps just the two universal BigButtons (Help, Play/
        Pause) every game screen offers -- "Back to games" and any
        game-specific action (Clear map) are small nav pills instead,
        the same overlap this class already avoids for Explore's own
        "Games" entry (see PublicOverlay.nav_row)."""
        self.overlay.clear_buttons()
        self._add_help_button()
        self._add_play_pause_button()
        nav = [(tr("back_to_games"), "🔙", self._on_back_to_games)]
        if self._active_game == "map":
            nav += self._map_game_nav_items()
        self.overlay.set_nav_buttons(nav)

    def _render_game_hottest(self) -> None:
        self.overlay.set_explore_visible(False)
        self.overlay.set_prompt(tr("prompt_temp_hunt"))
        if self._temp_hunt_target == "hottest":
            self.overlay.say(tr("temp_hunt_find_hottest"), CURIOUS)
        else:
            self.overlay.say(tr("temp_hunt_find_coolest"), CURIOUS)

    def _render_game_hotcold(self) -> None:
        self.overlay.set_explore_visible(False)
        self.overlay.set_prompt(tr("prompt_hot_cold"))
        if self._hotcold_stage == 0:
            self.overlay.say(tr("hotcold_find_first"), CURIOUS)
        elif self._hotcold_stage == 1:
            self.overlay.say(tr("hotcold_find_second"), CURIOUS)
        else:
            self.overlay.say(tr("hotcold_try_again"), CURIOUS)

    def _fan_control_hint_phrase(self) -> str:
        """The real on-screen control + option name for "turn the fan
        on" -- e.g. "Vent 1 to HVAC" -- looked up from the fan_on
        choice's own factor override and the matching explore control,
        never hardcoded. Games UX pass, item 5: the mystery's hint used
        to just say "turn the fan ON", which names no on-screen control
        at all (there is no button labelled "fan" -- see EXPLORE_
        CONTROLS' own comment on why the fan is Vent 1's HVAC option,
        not a dedicated control). A visitor acting on that generic
        wording could tap Vent 2, or Vent 1's plain OPEN, believe they'd
        "turned the fan on", and then get an apparently-contradicting
        "try HVAC first" nudge -- the same real case_index guard, just
        never satisfied because the wrong control was touched. Naming
        the actual control removes that ambiguity at the source, rather
        than the nudge needing to explain a mismatch after the fact.
        Falls back to the generic wording if this study's data doesn't
        have a fan_on choice/control (never assumed)."""
        fan_choice = next(
            (c for c in self.experiment.choices if c.key == "fan_on"), None)
        if fan_choice is None or len(fan_choice.factors) != 1:
            return tr("mystery_fan_fallback_name")
        (factor, value), = fan_choice.factors.items()
        control = next((c for c in self._explore_controls if c.factor == factor), None)
        option = next((o for o in (control.options if control else ()) if o.value == value), None)
        if control is None or option is None:
            return tr("mystery_fan_fallback_name")
        return tr("mystery_fan_control_and_option", control=control.label, option=option.label)

    def _render_game_mystery(self) -> None:
        """"Can you figure it out?": a real, measured, genuinely
        counter-intuitive effect this dataset supports -- switching the
        fan ON cools the ceiling (the smoke layer gets vented out) but
        *warms* the floor (real 481-frame, 24-scenario data, held
        candles/door/voc at baseline, checked against the last frame):

            near ceiling: 27.2 C (fan off) -> 22.9 C (fan on), -4.4 C
            near floor:   25.7 C (fan off) -> 31.6 C (fan on), +5.9 C

        Both comfortably clear PUBLIC_METRICS' room_temp.noticeable_delta
        (0.3 C). Every tap here is a same-place comparison against the
        baseline (see _on_game_tap) -- the Fan toggle is shown because
        testing the mystery requires actually switching it."""
        self.overlay.set_explore_visible(True)
        self._sync_explore_controls_to_case()
        self.overlay.set_prompt(tr("prompt_mystery"))
        self.overlay.say(tr("mystery_hint", fan=self._fan_control_hint_phrase()), CURIOUS)

    def _render_game_map(self) -> None:
        self.overlay.set_explore_visible(True)
        self._sync_explore_controls_to_case()
        self.overlay.set_prompt(tr("prompt_map_it"))
        self.overlay.say(tr("map_it_say"), CURIOUS)

    def _map_game_nav_items(self) -> list:
        if self._temp_trail or self._before_trail:
            return [(tr("action_clear_map"), "🧹", self._on_clear_trail_requested)]
        return []

    def _arm_idle_vent_hint(self) -> None:
        """Section 2: a real child left alone for a few seconds without
        touching anything gets one quiet nudge toward the vent -- never
        text, just the same pulse a direct tap already gives it (see
        PublicScene.pulse_vent). Cancelled by _on_overlay_tapped/
        _on_explore_changed the moment there's any real interaction, and
        never re-armed once fired (_idle_vent_hint_shown)."""
        if not self._idle_vent_hint_shown:
            self._idle_vent_timer.start(9000)

    def _on_idle_vent_timeout(self) -> None:
        if self._idle_vent_hint_shown or self.state.phase is not Phase.OBSERVE:
            return
        self._idle_vent_hint_shown = True
        self.scene.pulse_vent()
        self._defer(self._VENT_PULSE_MS, self.scene.reset_vent_width)

    def _observe_finished(self) -> None:
        self.state.go_to(Phase.PREDICTION)
        self._render_phase()

    def _add_return_nav(self) -> None:
        """"Back to games" for PREDICTION/COUNTDOWN/EXPERIMENT/REVEAL --
        SCIENCE deliberately does not call this (see its own comment):
        its card already fits its own sizeHint with zero spare budget at
        800x600, and this laid-out nav_row pill costs enough height to
        push it below that. Safe to call unconditionally on the other
        four: _enter_game_test_idea is the only live entry into this
        chain, and it always sets return_phase = Phase.GAMES_HOME before
        it ever gets here (confirmed by grepping every caller of
        _on_test_idea_clicked, not assumed from a docstring -- the
        direct-from-Explore entry this chain's own comments still
        describe no longer exists)."""
        self.overlay.set_nav_buttons([(tr("back_to_games"), "🔙", self._on_back_to_games)])

    def _render_prediction(self) -> None:
        # Meters off: while deciding, the question and the two choices
        # are the whole screen. Live numbers here also compete for the
        # height the touch targets need at 800x600.
        self.overlay.set_meters_visible(False)
        # The big prompt goes in the banner so the question reads from
        # across a room; the card carries the experiment's own wording.
        self.overlay.set_prompt(tr("prediction_prompt"))
        self.overlay.show_card(self.experiment.question, [])
        self._prediction_buttons = []
        for prediction in self.experiment.predictions:
            button = self.overlay.add_button(
                prediction.label, prediction.icon,
                lambda _checked=False, k=prediction.key: self._flash_prediction_choice(k),
                tall=True)
            self._prediction_buttons.append(button)
        self._add_return_nav()
        # say() last, not first: it positions the mascot's speech bubble
        # synchronously (PublicOverlay._say_inner -> _position_mascot),
        # clamped against button_row's own real width so the bubble
        # can't render over PREDICTION's three tall choice buttons (see
        # _position_mascot's own comment) -- clear_buttons() (run
        # centrally by _render_phase before this handler) leaves
        # button_row empty until the loop above runs, so calling say()
        # any earlier positions the bubble against a stale, empty row
        # and never gets called again once the real buttons exist. A
        # real, screenshotted bug: the bubble's own "Make a guess..."
        # text rendered underneath the choice buttons instead of beside
        # the mascot.
        self.overlay.say(tr("prediction_say"), THINKING)

    _PREDICTION_FLASH_MS = 260

    def _flash_prediction_choice(self, prediction_key: str) -> None:
        """Instant, visible confirmation that the tap registered: the
        chosen button gets a bright ring and every button disables for a
        beat, before the phase actually advances to the countdown.

        Games UX pass: a prediction button that silently disappears
        straight into the countdown reads as "nothing happened" to a
        first-time visitor -- this is that missing acknowledgement, kept
        separate from the *correct/incorrect* verdict (see
        _prediction_verdict), which only the data can answer, after the
        experiment has actually run."""
        for button, prediction in zip(self._prediction_buttons, self.experiment.predictions):
            # Disabled either way (a second tap during the flash must not
            # schedule a second advance) -- the border is what marks
            # which one was actually chosen; :disabled only restyles
            # background/color, so it stays visible.
            button.setEnabled(False)
            if prediction.key == prediction_key:
                button.setStyleSheet(
                    button.styleSheet() + f"QPushButton {{ border: 4px solid {DELIGHT}; }}")
        self._defer_if_current(
            self._PREDICTION_FLASH_MS,
            lambda k=prediction_key: self.overlay.prediction_made.emit(k))

    def _on_prediction(self, prediction_key: str) -> None:
        self.state.record_prediction(prediction_key)
        self.state.go_to(Phase.COUNTDOWN)
        self._render_phase()

    def _start_chosen_experiment(self) -> None:
        # First visit runs the fan (the experiment's first-declared, most
        # striking choice); a replay runs the side they have not seen, so
        # "try again" completes the comparison instead of repeating it.
        self._on_choice(self.experiment.next_untried_choice(self.state.tried_choices))

    def _on_choice(self, choice_key: str) -> None:
        if choice_key is None:
            return
        self.state.record_choice(choice_key)
        self.state.go_to(Phase.EXPERIMENT)
        case_index = experiments_mod.resolve_choice(
            self.sim_data.manifest, self.experiment, choice_key)
        if case_index is not None:
            self._load_case(case_index)
        self._render_phase()

    def _render_countdown(self) -> None:
        """A beat of anticipation between choosing and running.

        Short on purpose (~1.6 s total): long enough that the run feels
        like a consequence of the visitor's choice, short enough that a
        queue keeps moving. Driven by the widget's own QTimer, parented
        to this experience, so it dies with the window.
        """
        self.overlay.set_meters_visible(False)
        # Name the idea being tested, so the countdown reads as
        # "your guess -> let's test it" rather than a bare number.
        guess = self.experiment.prediction(self.state.prediction)
        self.overlay.set_prompt(
            tr("countdown_testing", icon=guess.icon, label=guess.label) if guess
            else tr("countdown_lets_test"))
        self.overlay.say(tr("countdown_say"), CURIOUS)
        self._countdown_step = 0
        self.overlay.show_countdown(_COUNTDOWN_STEPS[0])
        self._countdown_timer.start(COUNTDOWN_STEP_MS)
        self._add_return_nav()

    def _advance_countdown(self) -> None:
        self._countdown_step += 1
        if self._countdown_step < len(_COUNTDOWN_STEPS):
            self.overlay.show_countdown(_COUNTDOWN_STEPS[self._countdown_step])
            self.scene.refresh()
            return
        self._countdown_timer.stop()
        self.overlay.hide_countdown()
        self._start_chosen_experiment()

    def _render_experiment(self) -> None:
        choice = self.experiment.choice(self.state.choice)
        label = choice.short if choice else "Running"
        self.overlay.set_meters_visible(True)
        self.overlay.set_prompt(f"{choice.icon if choice else ''}  {label}")
        self.overlay.say(tr("experiment_say"), CURIOUS)
        self._add_help_button()
        self._add_play_pause_button()
        # Games UX pass: EXPERIMENT plays for real (~17-20s at the
        # current pacing constants) before the verdict shows -- a real,
        # reported "I give up waiting before it ever tells me anything"
        # complaint, even with the mid-run reassurance line below. The
        # measured comparison numbers already come from the full stored
        # arrays regardless of which frame is on screen (see
        # experiments.measure/_science_comparisons), so skipping ahead
        # changes nothing about the reveal's own correctness -- only how
        # much of the animation a visitor chose to watch first.
        self.overlay.add_button(tr("skip_to_results"), "⏩", self._on_skip_experiment)
        self._announce_expected_change()
        self._defer_if_current(
            EXPERIMENT_ALMOST_THERE_MS,
            lambda: self.overlay.say(tr("experiment_almost_there_say"), THINKING))
        self._play_until(EXPERIMENT_END_FRAME, self._experiment_finished)
        self._add_return_nav()

    def _on_skip_experiment(self) -> None:
        # Just seeking is enough: _play_until already armed _play_end_frame
        # at EXPERIMENT_END_FRAME, and TimeController.seek() unconditionally
        # emits time_changed -- _on_time_changed's own end-frame check
        # fires _experiment_finished() from that exact same path a natural
        # run reaches it through. Calling _experiment_finished() again
        # here directly would double-fire it (double-counting the quiz
        # score, double-rendering REVEAL).
        self.time_controller.seek(EXPERIMENT_END_FRAME)

    def _announce_expected_change(self) -> None:
        """Flash the change this run is about, but only when the measured
        data actually contains one.

        The emphasis is earned, not decorative: the banner shouts only if
        the leading metric clears its own noticeable_delta *and* the run
        on screen is the side that differs from the baseline. Running the
        baseline itself, or an experiment whose numbers barely move, gets
        no whoosh -- which is why room temperature (1.5 C) never triggers
        this and airflow (6.4x) does.
        """
        if self.state.choice == self.experiment.baseline_choice:
            return
        hero = experiments_mod.strongest_metric(self._science_comparisons())
        if hero is None:
            return
        self.overlay.banner.flash(
            tr("announce_whoosh", icon=hero.metric.icon, label=hero.metric.label.lower()),
            duration_ms=1800)

    def _experiment_finished(self) -> None:
        self.state.go_to(Phase.REVEAL)
        self._render_phase()

    # -- reveal ---------------------------------------------------------
    def _measurement(self, choice_key: str):
        if choice_key not in self._measurements:
            self._measurements[choice_key] = experiments_mod.measure(
                self.sim_data.store, self.sim_data.manifest, self.experiment, choice_key)
        return self._measurements[choice_key]

    @staticmethod
    def _guess_is_borne_out(guess, comparisons) -> bool:
        """Whether the data actually backs a guess this run.

        `supported_prediction` is a static declaration about the
        experiment; this checks the measurement. Without it a run where
        nothing cleared its threshold could still be congratulated with
        "You got it!" directly above "Almost nothing changed" -- the
        headline has to answer to the same numbers as the findings. A
        guess with no metric attached keeps the declared answer, since
        there is nothing to check it against.
        """
        if not guess.metric_key:
            return True
        match = next((c for c in comparisons if c.metric.key == guess.metric_key), None)
        return match is not None and match.is_noticeable

    def _reveal_shows_verdict(self) -> bool:
        """Whether this reveal has a real guess-vs-outcome verdict to
        show at all. False only for a run of the baseline on its own
        before the other side has ever been tried -- nothing was
        predicted *about* that specific run yet (see _reveal_lines'
        "ran baseline" branch), so there is nothing to score."""
        return not (self.state.choice == self.experiment.baseline_choice
                    and not self.experiment.all_choices_tried(self.state.tried_choices))

    def _prediction_verdict(self, comparisons) -> Optional[bool]:
        """True/False if the child's guess can be checked against this
        run's measured comparison, None if there is nothing to check
        (no guess made, no comparisons available, or this reveal doesn't
        show a verdict at all -- see _reveal_shows_verdict). The single
        source of truth for both the reveal's headline and its badge, so
        they can never disagree."""
        guess = self.experiment.prediction(self.state.prediction)
        if guess is None or not comparisons or not self._reveal_shows_verdict():
            return None
        return (self.state.prediction == self.experiment.supported_prediction
                and self._guess_is_borne_out(guess, comparisons))

    @staticmethod
    def _sentence(clause: str) -> str:
        """A clause from kid_language as a standalone sentence. str's own
        capitalize() would lowercase the rest of the string."""
        return clause[0].upper() + clause[1:] if clause else ""

    def _reveal_lines(self) -> tuple:
        """(headline, lines, dim_lines, verdict) for the reveal card.

        Structured as: what you predicted (the headline), what happened
        (the measured findings), why it matters (the leading metric's
        declared explanation). Nothing here names a factor, a quantity or
        a direction -- the findings come from the same generic metric
        comparison the science card and the guide use, and the side names
        come from the experiment's own choice labels, so this reads
        correctly for any experiment whose strongest metric is something
        else entirely.

        The measured-vs-explanatory split is deliberate: a finding
        sentence is composed mechanically from a measured direction and
        size, while an explanation is a physical claim that has to be
        declared and defended per metric (see Metric.explanation).

        `verdict` (Optional[bool]) is the same True/False/None
        _prediction_verdict already scored this run with -- carried
        through here so the headline text and the reveal's explicit
        Correct/Incorrect badge can never disagree about the answer.
        """
        comparisons = self._science_comparisons()
        hero = experiments_mod.strongest_metric(comparisons)
        if not comparisons:
            return tr("reveal_headline_default"), [tr("reveal_no_comparisons_line")], [], None

        _, contrast_label = self._choice_labels()
        dim = []
        if hero is not None:
            dim.append(tr(
                "reveal_measured_dim", label=hero.metric.label.lower(),
                baseline=hero.value_text(hero.baseline), contrast=hero.value_text(hero.contrast),
                unit=hero.metric.unit))

        findings = []
        for comparison in (hero, experiments_mod.secondary_metric(comparisons, hero)):
            if comparison is None:
                continue
            clause = kid.finding_sentence(comparison)
            if clause:
                findings.append(f"{comparison.metric.icon} {self._sentence(clause)}.")
        if not findings:
            # Nothing cleared its own threshold: say so rather than
            # promoting a rounding difference into a discovery.
            findings = [tr("reveal_almost_nothing")]

        explanation = hero.metric.explanation if hero is not None else ""
        verdict = self._prediction_verdict(comparisons)

        if self.experiment.all_choices_tried(self.state.tried_choices):
            baseline_label, _ = self._choice_labels()
            return (tr("reveal_both_tested_headline"),
                    [tr("reveal_seen_both", baseline=baseline_label, contrast=contrast_label)]
                    + findings[:1] + ([explanation] if explanation else []),
                    dim, verdict)

        # A run of the baseline on its own: the findings still describe
        # what the *other* side does, so name it and invite the comparison.
        if self.state.choice == self.experiment.baseline_choice:
            lines = [tr("reveal_with_contrast", contrast=contrast_label,
                       clause=kid.finding_sentence(hero))
                     if hero is not None else findings[0]]
            return (tr("reveal_headline_default"),
                    lines + ([explanation] if explanation else [])
                    + [tr("reveal_try_again_line")],
                    dim, None)

        # Full quiz mode: the headline states the verdict plainly, and
        # _render_reveal adds an explicit Correct/Incorrect badge to
        # match -- both driven by the same `verdict` so they can't drift
        # apart.
        guess = self.experiment.prediction(self.state.prediction)
        if guess is None:
            headline = tr("reveal_headline_no_guess")
        elif verdict:
            headline = tr("reveal_headline_matched")
        else:
            headline = tr("reveal_headline_incorrect")

        return (headline, findings + ([explanation] if explanation else []), dim, verdict)

    def _render_reveal(self) -> None:
        # Meters off: the card below carries the same numbers, and on a
        # short display the two compete for the height the explanation
        # needs in order to wrap instead of clipping.
        self.overlay.set_meters_visible(False)
        self.overlay.set_prompt("")
        headline, lines, dim, verdict = self._reveal_lines()
        badge = None
        if verdict is not None:
            # Full quiz mode (games UX pass): an explicit, unmissable
            # Correct/Incorrect badge above the findings, plus the same
            # confetti/chime every other "you got it" moment in Games
            # uses -- only for a genuine correct guess, never for a miss.
            badge = VerdictBadge(
                tr("verdict_correct_badge") if verdict else tr("verdict_incorrect_badge"), verdict)
            if verdict:
                self.overlay.celebrate()
                play_success_chime()
        self.overlay.show_card(headline, lines, dim, hero=badge)
        # The guide reacts to the side that was actually just watched --
        # asking "did you see how much more the air was moving?" after a
        # fan-*off* run contradicts the screen.
        baseline_label, contrast_label = self._choice_labels()
        hero = experiments_mod.strongest_metric(self._science_comparisons())
        clause = kid.finding_sentence(hero) if hero is not None else ""
        if self.experiment.all_choices_tried(self.state.tried_choices):
            self.overlay.say(tr("reveal_both_tried_say", contrast=contrast_label), EXCITED)
        elif self.state.choice == self.experiment.baseline_choice:
            self.overlay.say(
                tr("reveal_baseline_say", baseline=baseline_label, contrast=contrast_label),
                THINKING)
        elif clause:
            self.overlay.say(tr("reveal_spotted_it", clause=self._sentence(clause)), EXCITED)
        else:
            self.overlay.say(tr("reveal_spot_what_changed"), EXCITED)
        self.overlay.add_button(tr("science_button"), "🔬",
                                self.overlay.science_toggled.emit, primary=True)
        self._add_return_nav()

    def _on_science_toggled(self) -> None:
        self.state.toggle_science()
        self.state.go_to(Phase.SCIENCE if self.state.science_visible else Phase.REVEAL)
        self._render_phase()

    def _science_comparisons(self) -> list:
        """Every public metric, measured baseline -> contrast. Normalized
        to that direction, so watching ON then OFF and OFF then ON give
        an identical comparison."""
        baseline = self._measurement(self.experiment.baseline_choice)
        contrast_key = self.experiment.contrast_choice()
        contrast = self._measurement(contrast_key) if contrast_key else None
        return experiments_mod.compare_metrics(baseline, contrast)

    def _choice_labels(self) -> tuple:
        """Short column headings for the two sides, from the experiment's
        own choice labels rather than anything fan-specific."""
        baseline = self.experiment.choice(self.experiment.baseline_choice)
        contrast = self.experiment.choice(self.experiment.contrast_choice())
        return (baseline.short if baseline else "Before",
                contrast.short if contrast else "After")

    def _prediction_line(self, comparison) -> str:
        """Tie the secondary finding back to the visitor's own guess, but
        only when the guess was genuinely about this metric and the data
        actually supports it. Both facts are declared on the Prediction
        itself, so nothing here is specific to any one experiment; a
        guess with no metric attached falls back to what it means."""
        guess = self.experiment.prediction(self.state.prediction)
        if (guess is not None and guess.confirmation
                and guess.metric_key == comparison.metric.key
                and self.state.prediction == self.experiment.supported_prediction):
            return guess.confirmation
        return comparison.metric.meaning

    def _render_science(self) -> None:
        """The same run, with the numbers a researcher would look at.

        Ordered conclusion-first: what changed most, then a bar
        comparison sized so the difference is visible before any decimal
        is read, then the exact measured values, then one line saying
        what the leading quantity actually means. Every number comes from
        Experiment.measure(); nothing here recomputes physics.
        """
        self.overlay.set_meters_visible(False)
        self.overlay.set_prompt("")

        comparisons = self._science_comparisons()
        hero = experiments_mod.strongest_metric(comparisons)
        both_tried = self.experiment.all_choices_tried(self.state.tried_choices)
        baseline_label, contrast_label = self._choice_labels()

        secondary = experiments_mod.secondary_metric(comparisons, hero)

        blocks = []
        dim = []
        if hero is not None:
            # Headline leads with the size of the change and stays on one
            # line -- a second line here pushes the rest out of the card
            # at 800x600. The bars label their own values, so no separate
            # "0.085 -> 0.540" line is needed.
            blocks.append(HeroMetric(
                f"{hero.metric.icon}  {hero.change_text()} {hero.metric.label.lower()}",
                BarCompare(baseline_label, hero.baseline, contrast_label, hero.contrast,
                           hero.metric.unit, hero.metric.decimals)))
        if secondary is not None:
            blocks.append(SecondaryMetric(
                f"{secondary.metric.icon}  {secondary.metric.label} — {secondary.change_text()}",
                BarCompare(baseline_label, secondary.baseline,
                           contrast_label, secondary.contrast,
                           secondary.metric.unit, secondary.metric.decimals,
                           compact=True)))
            dim.append(self._prediction_line(secondary))

        # Whatever got no chart still gets its measured numbers, so the
        # evidence stays complete even for a value that barely moved.
        rows = [
            f"{c.metric.icon} {c.metric.label:<12}"
            f"{c.value_text(c.baseline):>10} {c.metric.unit:<4}"
            f"{c.value_text(c.contrast):>9} {c.metric.unit:<4}"
            f"  {c.change_text()}"
            for c in experiments_mod.unchanged_metrics(comparisons, [hero, secondary])
        ]

        title = tr("science_title_both") if both_tried else tr("science_title_default")
        # The detected story beats deliberately do not appear here. They
        # are narrated during playback, where they are tied to the moment
        # they happen, and repeating them turns this card into a wall of
        # text that overflows on a 600 px-tall display.
        self.overlay.show_card(title, [], [d for d in dim if d],
                               block="\n".join(rows), hero=blocks)
        # The guide interprets the *same* hero comparison the card just
        # charted, so the two can never disagree about what mattered.
        # Falls back to the generic line only when nothing moved enough
        # to be worth a claim.
        finding = kid.mascot_finding(hero, both_tried) if hero is not None else ""
        self.overlay.say(finding or tr("science_fallback_say"),
                         EXPLAINING)
        self.overlay.add_button(tr("science_back_button"), "←", self.overlay.science_toggled.emit,
                                primary=True)
        # No "Back to games" pill on SCIENCE: its card already fits its
        # own sizeHint with zero spare budget at 800x600 -- any chrome
        # added here (laid-out or absolutely positioned; both were tried)
        # collided with either the card or button_row. "← Back" above
        # returns to REVEAL, which has the pill (_add_return_nav) --
        # confirmed via science_toggled -> _on_science_toggled -> the
        # REVEAL branch of Phase.SCIENCE if science_visible else REVEAL,
        # so the games hub stays reachable from here in one extra tap.

    def _render_complete(self) -> None:
        self.overlay.set_prompt("")
        self.overlay.say(tr("complete_say"), IDLE)
        self.overlay.add_button(tr("replay_start_again"), "🔁",
                                self.overlay.replay_requested.emit, primary=True)

    def _on_replay(self) -> None:
        self._countdown_timer.stop()
        self.time_controller.pause()
        self.state.replay()
        self._idle_vent_hint_shown = False
        self._idle_vent_timer.stop()
        self._reset_mystery_and_notebook()
        self._load_case(self.state.baseline_case_index)
        self._render_phase()
