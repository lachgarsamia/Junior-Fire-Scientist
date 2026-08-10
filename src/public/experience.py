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

import logging
import random

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
from public.widgets import BarCompare, HeroMetric, SecondaryMetric
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

# The countdown before a run: four beats of ~400 ms.
COUNTDOWN_STEP_MS = 400
_COUNTDOWN_STEPS = ("3", "2", "1", "🔥")

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
    "fact_flame_temperature",
    "fact_moving_air_oxygen",
    "fact_cool_air_sinks",
    "fact_blue_flame_hottest",
    "fact_closed_door_slows_fire",
    "fact_smoke_more_dangerous",
    "fact_firefighters_study_smoke",
)
_FACT_SHOW_MS = 7000
_FACT_GAP_MS = 15000
_FACT_FIRST_DELAY_MS = 9000

# Phases where the fan/vent marker (anchored to the real vent position)
# is worth showing -- the phases where the scene itself, not a card, is
# what fills the screen.
FAN_MARKER_PHASES = (Phase.OBSERVE, Phase.EXPERIMENT, Phase.GAME_PLAY)

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
        self._compare_place_armed = False
        # Which extreme "hottest"'s game is hunting for this round --
        # chosen dynamically per activation (Phase 4 section 13), not
        # two separate menu items.
        self._temp_hunt_target = "hottest"
        # Phase 4: "Can you figure it out?" (the Mystery game) -- the one
        # genuinely surprising real effect this dataset supports (see
        # _render_game_mystery's docstring for the measured numbers):
        # the fan cools the ceiling but *warms* the floor. Tracked across
        # separate "compare this place" taps so either order counts.
        # Phase 8 section 9: detection itself no longer needs the "Figure
        # it out" button pressed first -- every same-place comparison
        # opportunistically checks for this pattern (see
        # _show_same_place_comparison/_note_mystery_progress), so the
        # discovery can emerge from ordinary measuring instead of only
        # from an armed mini-game.
        self._mystery_found_high = False
        self._mystery_found_low = False
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
        # Phase 5: "show before" -- a static contour of the baseline's
        # own settled frame over the current heatmap.
        self._ghost_visible = False
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
        switching to Fan ON (see _handle_hotcold_tap), and a stale
        "compare this place" arm-state never fires against the wrong
        scenario."""
        self._active_game = None
        self._hotcold_stage = 0
        self._hotcold_readings = {}
        self._compare_place_armed = False
        self.scene.clear_dual_markers()
        # The trail/ghost's own scene artists are cleared by
        # PublicScene.clear_probe() (called from _load_case/_render_
        # phase); this resets the *experience's* tracking of them so a
        # re-render's own button declarations stay in sync.
        self._temp_trail = []
        self._before_trail = []
        self._ghost_visible = False

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
        self.overlay.add_button(tr("keep_exploring"), "🔥", self._close_compare, primary=True)

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

    # -- Phase 5: "show before" ghost overlay -----------------------------
    def _on_ghost_toggle_requested(self) -> None:
        """A static dashed contour of the baseline's own settled field
        over the current heatmap -- real data (PublicScene.
        show_baseline_ghost reads the same store the rest of the app
        does), never a second full heatmap or an invented overlay."""
        if self.state.phase is not Phase.GAME_PLAY:
            return
        if self.state.case_index == self.state.baseline_case_index:
            return
        self._ghost_visible = not self._ghost_visible
        if self._ghost_visible:
            self.scene.show_baseline_ghost(self.state.baseline_case_index)
            self.overlay.say(tr("ghost_shown_say"), CURIOUS)
        else:
            self.scene.hide_baseline_ghost()
        self.scene.refresh()
        self._refresh_game_buttons()

    def _update_fan_marker(self) -> None:
        """Anchor the fan label to the real vent position for whatever
        scenario is now loaded -- None/hidden when this study has no
        manifest entry or the current plane isn't the one the geometry
        is defined against (PublicScene.vent_marker_position's own gate).
        Updates the thermometer's room-wall anchor in the same breath --
        both are the scene's own real-geometry positions and always go
        stale together on a scenario switch."""
        position = self.scene.vent_marker_position(self.state.case_index)
        if position is None:
            self.overlay.set_fan_marker(None, False)
        else:
            frac = self.scene.widget_fraction_for(*position)
            entry = self.scene.current_entry()
            self.overlay.set_fan_marker(frac, entry is not None and entry.vod == 2)
        wall_position = self.scene.room_wall_anchor(self.state.case_index)
        if wall_position is None:
            self.overlay.set_thermometer_anchor(None)
        else:
            wall_frac_x, _wall_frac_y = self.scene.widget_fraction_for(*wall_position)
            self.overlay.set_thermometer_anchor(wall_frac_x)

    def _hide_fan_marker(self) -> None:
        """Hides the fan label -- stealing vertical space from a phase
        (SCIENCE, a compare detour) that has no vent to show at all, on
        the strength of a now-stale frac from whatever OBSERVE visit
        last set it, caused a real, measured HeroMetric clip on the
        SCIENCE card before this."""
        self.overlay.set_fan_marker(None, False)

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
        self._update_fan_marker()
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
        if phase not in FAN_MARKER_PHASES:
            self._hide_fan_marker()
        else:
            self._update_fan_marker()
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
        if self.state.phase not in PROBE_PHASES:
            return
        result = self.scene.probe_at(pos)
        if result is None:
            return
        self._idle_vent_timer.stop()
        x, z, value = result
        if self.state.phase is Phase.GAME_PLAY:
            self._on_game_tap(x, z, value)
        else:
            self._on_explore_tap(x, z, value)

    def _on_explore_tap(self, x: float, z: float, value: float) -> None:
        """A plain Explore tap: always just the real reading at that
        point (or, on the candle/vent, what that object is) -- never a
        mini-game reaction, since a game can't be active outside
        Phase.GAME_PLAY."""
        if self.scene.vent_hit(x, z, self.state.case_index):
            self._on_vent_tapped()
            return
        if self.scene.vent2_hit(x, z, self.state.case_index):
            self._on_vent2_tapped()
            return
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
        if game in ("compare", "map", "mystery"):
            if self.scene.vent_hit(x, z, self.state.case_index):
                self._on_vent_tapped()
                return
            if self.scene.vent2_hit(x, z, self.state.case_index):
                self._on_vent2_tapped()
                return
        if self.scene.candle_hit(x, z):
            self.scene.pulse_flame()
            # The real hottest cell is almost always at (or right beside)
            # the candle itself -- a Temp Hunt guess aimed squarely at
            # the flame is real game input, not something the plain
            # candle callout below should swallow silently.
            if game == "hottest" and self._temp_hunt_target == "hottest":
                self.overlay.update_thermometer(value, tr("thermometer_at_flame"))
                self._celebrate(tr("found_flame_say", value=value))
                self._record_discovery(
                    "🔥", tr("found_flame_title"), tr("found_flame_discovery", value=value))
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
            self._on_candle_tapped(value)
            return
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
            # no separate "arm" step needed (unlike the Compare game's
            # own "Compare this place" button, see _on_compare_place_
            # requested).
            self._show_same_place_comparison(x, z)
            return
        revisit = self._match_before_trail(x, z)
        if self._compare_place_armed:
            self._compare_place_armed = False
            self._show_same_place_comparison(x, z)
        elif revisit is not None:
            # Phase 8 section 7/9: tapping approximately where the child
            # already measured before their last Fan/Candle change reads
            # as revisiting that spot -- the same verdict-first
            # before/after card "Compare this place" shows, but reached
            # by returning to a place instead of arming a tool first.
            self._show_same_place_comparison(revisit["x"], revisit["z"])
        elif game == "map":
            self._add_trail_point(x, z, value)
        # game == "compare" with nothing armed and no revisit: the
        # reading just shown above is the whole reaction.

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
        every exposed explore control, plus door -- the one factor none
        of them varies, held at the same wide-open value every
        ExploreControl.held dict already fixes it to. This is the
        starting point _on_explore_changed composes a single change on
        top of, so flipping one control keeps whatever the others are
        already set to instead of resetting them to a baseline the
        child may have already left (see _on_explore_changed's own
        docstring). Falls back to every control's own default if
        nothing is loaded yet."""
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
        if self.state.phase not in (Phase.OBSERVE, Phase.GAME_PLAY):
            return
        self._idle_vent_timer.stop()
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
        self.overlay.repaint()
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
        if control_key == "fan":
            self.overlay.banner.flash(tr("banner_whoosh") if value else tr("banner_quiet_again"))
            self.overlay.say(tr("explore_changed_fan"), CURIOUS)
        else:
            self.overlay.say(tr("explore_changed_other"), CURIOUS)
        self._start_free_play()
        if self.state.phase is Phase.GAME_PLAY:
            self._refresh_game_buttons()

    _VENT_PULSE_MS = 260

    def _on_vent_tapped(self) -> None:
        """The vent is a real tappable object in the scene, not just an
        external toggle -- see _toggle_explore_control_by_tap, which
        this and _on_vent2_tapped both share."""
        self._toggle_explore_control_by_tap("fan", vent_index=0)
        # Deferred to the next event-loop turn so the marker's geometry
        # is already the *new* (post-switch) one when the pulse captures
        # it -- pulsing before the click would animate back to a rect
        # that set_fan_marker's own reposition immediately overrides.
        self._defer(0, self.overlay.pulse_fan_marker)

    def _on_vent2_tapped(self) -> None:
        """The second (candle-side) vent is a real tappable object too --
        same idea as _on_vent_tapped, against the "vent2" control and the
        second vent line (index 1 in the shared room_vents
        LineCollection, see PublicScene.pulse_vent). No floating marker
        label: unlike the fan, this vent's open/closed state is already
        visible in the vent glyph itself (PublicScene._draw_vent2's slat
        angle/colour) and in the ExploreToggle's own checked state, so it
        doesn't need a second copy of set_fan_marker's whole subsystem
        (see the redesign brief's "do not create duplicate systems" rule)."""
        self._toggle_explore_control_by_tap("vent2", vent_index=1)

    def _toggle_explore_control_by_tap(self, control_key: str, vent_index: int) -> None:
        """Flip the named explore control by clicking its actual toggle
        widget (so its checked state, and everything _on_explore_changed
        does, stays the single source of truth rather than a second code
        path that could drift from it), then flash the tapped vent's own
        drawn line so the tap reads as "I touched a machine and it
        reacted", not "a setting changed" (Phase 9 section 3/11)."""
        control = next((c for c in self._explore_controls if c.key == control_key), None)
        toggle = self.overlay._explore_toggles.get(control_key)
        entry = self.scene.current_entry()
        if control is None or toggle is None or entry is None:
            return
        options = [opt.value for opt in control.options]
        current = getattr(entry, control.factor, None)
        if current not in options:
            return
        toggle._group.button(1 - options.index(current)).click()
        self.scene.pulse_vent(vent_index)
        self._defer(self._VENT_PULSE_MS, self.scene.reset_vent_width)

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
        self._play_end_frame = None
        self._play_finished_cb = None
        self.time_controller.set_loop(True)
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
        """Shared pause-and-look setup for every compare flavour ("What
        changed?" and "Compare this place") -- a detour modelled on
        _on_help_requested, not a phase change: closing it (_close_
        compare) must return to exactly where OBSERVE was.

        The comparison card carries real bar charts (same widgets the
        science card uses) and is genuinely tall -- the meters, fan
        marker and explore toggles must step aside the way REVEAL/
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
        self._hide_fan_marker()
        self.overlay.set_prompt("")

    def _on_compare_requested(self) -> None:
        """"What changed?": the baseline the child started from versus
        whatever they have since explored to -- reuses the exact same
        compare_metrics/strongest_metric/secondary_metric machinery the
        guided science card uses, so free play and the guided experiment
        can never disagree about what a "noticeable" change is.
        """
        if self.state.phase is not Phase.GAME_PLAY or self._compare_open:
            return
        if self.state.case_index == self.state.baseline_case_index:
            return
        self._open_compare_detour()

        baseline = self._case_measurement(self.state.baseline_case_index)
        contrast = self._case_measurement(self.state.case_index)
        comparisons = experiments_mod.compare_metrics(baseline, contrast)
        hero = experiments_mod.strongest_metric(comparisons)
        secondary = experiments_mod.secondary_metric(comparisons, hero)
        before_label = self._label_for_case(self.state.baseline_case_index)
        after_label = self._label_for_case(self.state.case_index)

        blocks = []
        if hero is not None:
            blocks.append(HeroMetric(
                f"{hero.metric.icon}  {hero.change_text()} {hero.metric.label.lower()}",
                BarCompare(before_label, hero.baseline, after_label, hero.contrast,
                          hero.metric.unit, hero.metric.decimals)))
        if secondary is not None:
            blocks.append(SecondaryMetric(
                f"{secondary.metric.icon}  {secondary.metric.label} — {secondary.change_text()}",
                BarCompare(before_label, secondary.baseline, after_label, secondary.contrast,
                          secondary.metric.unit, secondary.metric.decimals, compact=True)))
        lines = [] if blocks else [tr("what_changed_nothing")]
        self.overlay.show_card(tr("what_changed_title"), lines, hero=blocks or None)
        if hero is not None:
            self._last_watched = f"{hero.metric.icon} {hero.metric.label} — {hero.change_text()}"
        finding = kid.mascot_finding(hero) if hero is not None else ""
        self.overlay.say(finding or tr("keep_exploring"), EXCITED if finding else CURIOUS)
        self.overlay.add_button(tr("keep_exploring"), "🔥", self._close_compare, primary=True)
        if hero is not None:
            self._add_why_button(hero.metric.explanation)

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

    # -- "compare this place": same physical point, two conditions ------
    def _on_compare_place_requested(self) -> None:
        """Arm the next real tap as a same-location comparison instead of
        a plain reading -- see _show_same_place_comparison, dispatched
        from _on_overlay_tapped. Only meaningful once the child has
        actually explored something different from the baseline (gated
        the same way "🔎 Compare" is); comparing a scenario against
        itself has nothing to say."""
        if self.state.phase is not Phase.GAME_PLAY:
            return
        if self.state.case_index == self.state.baseline_case_index:
            return
        self._compare_place_armed = True
        self.overlay.say(tr("compare_this_place_say"), CURIOUS)

    def _note_mystery_progress(self, zone: str, delta: float, x: float, z: float) -> str:
        """Called with a real same-place comparison's own location/delta.
        Returns a short reaction line, or "" to fall back to the normal
        change_line. Only ever reports what has *actually* been measured
        -- never assumes the fan is what changed (a candle-count same-
        place comparison simply never matches either zone/sign pair).

        `zone` is the stable, language-independent id from
        PublicScene.location_zone() ("ceiling"/"floor"/...), not the
        translated display phrase -- this logic must keep working
        identically regardless of the current UI language.

        Staged as its own small discovery, not a single "solved" card
        (Phase 7 section 7): the first zone found gets a reaction plus a
        nudge toward the other one, and only the second zone reveals the
        "same fan, different place" punchline. Records the real (x, z)
        of each zone (Phase 12 section 11) so the "aha" moment can
        highlight the *actual* two spots the child measured, not
        invented ones."""
        if zone == "ceiling" and delta < 0:
            self._mystery_found_high = True
            self._mystery_high_xz = (x, z)
        elif zone == "floor" and delta > 0:
            self._mystery_found_low = True
            self._mystery_low_xz = (x, z)
        else:
            return ""
        if self._mystery_found_high and self._mystery_found_low:
            self._record_discovery(
                "🔎", tr("mystery_solved_discovery_title"), tr("mystery_solved_discovery"))
            return tr("mystery_solved_title")
        if zone == "ceiling":
            return tr("mystery_progress_ceiling")
        return tr("mystery_progress_floor")

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
        solved_before = self._mystery_found_high and self._mystery_found_low
        if noticeable:
            mascot_line = self._note_mystery_progress(zone, delta, x, z) or verdict
        mystery_just_solved = (not solved_before
                               and self._mystery_found_high and self._mystery_found_low)
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
        self.overlay.add_button(tr("keep_exploring"), "🔥", self._close_compare, primary=True)
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
            close = self.scene.hottest_guess_is_close(x, z, self.state.frame_index)
        else:
            close = self.scene.coolest_guess_is_close(x, z, self.state.frame_index)
        if close:
            icon = "🔥" if target == "hottest" else "🧊"
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
            self.overlay.say(tr("hotcold_find_cool"), CURIOUS)
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
        room_temp = next(m for m in experiments_mod.PUBLIC_METRICS if m.key == "room_temp")
        big_difference = abs(hot_c - cool_c) >= room_temp.noticeable_delta
        diff_word = tr("hotcold_diff_big" if big_difference else "hotcold_diff_small")
        if big_difference:
            self._celebrate(tr("hotcold_say", hot=hot_c, cool=cool_c, diff=diff_word))
            self._record_discovery(
                "🌡️", tr("hotcold_discovery_title"),
                tr("hotcold_discovery_text", hot=hot_c, cool=cool_c))
        else:
            self.overlay.say(tr("hotcold_say", hot=hot_c, cool=cool_c, diff=diff_word), CURIOUS)

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
        self.overlay.add_button(tr("attract_button"), "🔥",
                                self.overlay.start_requested.emit, primary=True, tall=True)
        # Idle attract: loop the baseline fire quietly behind the invite.
        self.time_controller.set_loop(True)
        self._play_end_frame = None
        self._play_finished_cb = None
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
        self.overlay.add_button(tr("watch_the_fire"), "🔥", self._start_observe, primary=True)

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
        _defer_if_current, not a persistent QTimer: the callback simply
        no-ops if the phase has changed by the time it fires (leaving
        OBSERVE for a moment and coming back re-arms fresh via
        _render_phase, rather than this timer needing its own cancel/
        restart bookkeeping), the same guard _show_next_fact and
        _hide_fact_and_wait_for_more both rely on too."""
        self._defer_if_current(_FACT_FIRST_DELAY_MS, self._show_next_fact)

    def _show_next_fact(self) -> None:
        if self.state.phase is not Phase.OBSERVE:
            return
        if not self._fact_queue:
            self._fact_queue = list(FIRE_FACT_KEYS)
            random.shuffle(self._fact_queue)
        key = self._fact_queue.pop()
        self.overlay.say_fact(tr(key))
        self._defer_if_current(_FACT_SHOW_MS, self._hide_fact_and_wait_for_more)

    def _hide_fact_and_wait_for_more(self) -> None:
        if self.state.phase is not Phase.OBSERVE:
            return
        self.overlay.clear_fact()
        self._defer_if_current(_FACT_GAP_MS, self._show_next_fact)

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
            ("🔥", tr("tile_temp_hunt"), self._enter_game_hottest),
            ("🧊", tr("tile_hot_cold"), self._enter_game_hotcold),
            ("🔎", tr("tile_mystery"), self._enter_game_mystery),
            ("🧪", tr("tile_test_idea"), self._enter_game_test_idea),
            ("📊", tr("tile_compare"), self._enter_game_compare),
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

    def _enter_game_compare(self) -> None:
        self._clear_game_state()
        self._active_game = "compare"
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
            "compare": self._render_game_compare,
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
        _on_explore_changed/_add_trail_point/_on_clear_trail_requested/
        _on_ghost_toggle_requested (a partial refresh, so a Fan/Candle
        change or a trail edit doesn't force a full _render_phase() and
        risk disturbing the trail/ghost still drawn on the scene).

        button_row keeps just the two universal BigButtons (Help, Play/
        Pause) every game screen offers -- "Back to games" and any
        game-specific action (What changed?/Compare this place/Show
        before/Clear map) are small nav pills instead, the same
        overlap this class already avoids for Explore's own "Games"
        entry (see PublicOverlay.nav_row)."""
        self.overlay.clear_buttons()
        self._add_help_button()
        self._add_play_pause_button()
        nav = [(tr("back_to_games"), "🔙", self._on_back_to_games)]
        if self._active_game == "compare":
            nav += self._compare_game_nav_items()
        elif self._active_game == "map":
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
            self.overlay.say(tr("hotcold_find_hot"), CURIOUS)
        elif self._hotcold_stage == 1:
            self.overlay.say(tr("hotcold_find_cool"), CURIOUS)
        else:
            self.overlay.say(tr("hotcold_try_again"), CURIOUS)

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
        self.overlay.say(tr("mystery_hint"), CURIOUS)

    def _render_game_compare(self) -> None:
        self.overlay.set_explore_visible(True)
        self._sync_explore_controls_to_case()
        self.overlay.set_prompt(tr("prompt_compare"))
        self.overlay.say(tr("compare_say"), CURIOUS)

    def _compare_game_nav_items(self) -> list:
        if self.state.case_index == self.state.baseline_case_index:
            return []
        return [
            (tr("action_what_changed"), "🔎", self._on_compare_requested),
            (tr("action_compare_this_place"), "📍", self._on_compare_place_requested),
            (tr("action_hide_before") if self._ghost_visible else tr("action_show_before"), "👻",
             self._on_ghost_toggle_requested),
        ]

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
        self.overlay.say(tr("prediction_say"), THINKING)
        # The big prompt goes in the banner so the question reads from
        # across a room; the card carries the experiment's own wording.
        self.overlay.set_prompt(tr("prediction_prompt"))
        self.overlay.show_card(self.experiment.question, [])
        for prediction in self.experiment.predictions:
            self.overlay.add_button(
                prediction.label, prediction.icon,
                lambda _checked=False, k=prediction.key: self.overlay.prediction_made.emit(k),
                tall=True)
        self._add_return_nav()

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
        self._announce_expected_change()
        self._play_until(EXPERIMENT_END_FRAME, self._experiment_finished)
        self._add_return_nav()

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

    @staticmethod
    def _sentence(clause: str) -> str:
        """A clause from kid_language as a standalone sentence. str's own
        capitalize() would lowercase the rest of the string."""
        return clause[0].upper() + clause[1:] if clause else ""

    def _reveal_lines(self) -> tuple:
        """(headline, lines, dim_lines) for the reveal card.

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
        """
        comparisons = self._science_comparisons()
        hero = experiments_mod.strongest_metric(comparisons)
        if not comparisons:
            return tr("reveal_headline_default"), [tr("reveal_no_comparisons_line")], []

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

        if self.experiment.all_choices_tried(self.state.tried_choices):
            baseline_label, _ = self._choice_labels()
            return (tr("reveal_both_tested_headline"),
                    [tr("reveal_seen_both", baseline=baseline_label, contrast=contrast_label)]
                    + findings[:1] + ([explanation] if explanation else []),
                    dim)

        # A run of the baseline on its own: the findings still describe
        # what the *other* side does, so name it and invite the comparison.
        if self.state.choice == self.experiment.baseline_choice:
            lines = [tr("reveal_with_contrast", contrast=contrast_label,
                       clause=kid.finding_sentence(hero))
                     if hero is not None else findings[0]]
            return (tr("reveal_headline_default"),
                    lines + ([explanation] if explanation else [])
                    + [tr("reveal_try_again_line")],
                    dim)

        # Predicting is the thing being rewarded, not being right: a
        # child who guessed differently ran exactly the same experiment
        # and learned exactly as much. No wording here implies a wrong
        # answer or a failure.
        guess = self.experiment.prediction(self.state.prediction)
        if guess is None:
            headline = tr("reveal_headline_no_guess")
        elif (self.state.prediction == self.experiment.supported_prediction
              and self._guess_is_borne_out(guess, comparisons)):
            headline = tr("reveal_headline_matched")
        else:
            headline = tr("reveal_headline_great_guess")

        return (headline, findings + ([explanation] if explanation else []), dim)

    def _replay_invitation(self) -> tuple:
        """(label, icon) for the replay button, phrased from what this
        visitor has actually tried. Naming the untested side is what
        makes a second run feel like an invitation rather than a repeat;
        `tried_choices` stays the single source of truth."""
        if self.experiment.all_choices_tried(self.state.tried_choices):
            return (tr("replay_start_again"), "🔁")
        next_key = self.experiment.next_untried_choice(self.state.tried_choices)
        other = self.experiment.choice(next_key)
        if other is None or next_key == self.state.choice:
            return (tr("replay_try_again"), "🔁")
        return (tr("replay_try_other", label=other.short), "🔎")

    def _render_reveal(self) -> None:
        # Meters off: the card below carries the same numbers, and on a
        # short display the two compete for the height the explanation
        # needs in order to wrap instead of clipping.
        self.overlay.set_meters_visible(False)
        self.overlay.set_prompt("")
        headline, lines, dim = self._reveal_lines()
        self.overlay.show_card(headline, lines, dim)
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
                                self.overlay.science_toggled.emit)
        self.overlay.add_button(*self._replay_invitation(),
                                self.overlay.replay_requested.emit, primary=True)
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
        self.overlay.add_button(tr("science_back_button"), "←", self.overlay.science_toggled.emit)
        self.overlay.add_button(*self._replay_invitation(),
                                self.overlay.replay_requested.emit, primary=True)
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
