"""Explicit state model for the public experience.

Deliberately a plain dataclass + enum with no Qt dependency: the phase
machine is the part most worth testing, and it stays testable without a
QApplication. Widgets read this; they never each keep their own copy of
"has the visitor predicted yet".

Phase order is the visitor's journey, and `advance()` encodes the only
legal forward path. Anything else (a stray click during a transition,
a kiosk reset mid-experiment) goes through reset() or a direct
`go_to()`, so an illegal jump is a loud programming error rather than a
half-updated screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Phase(Enum):
    ATTRACT = "attract"        # nobody is here; loop the fire, invite a touch
    INTRO = "intro"            # "this is the real box on the table"
    OBSERVE = "observe"        # watch the baseline fire run
    PREDICTION = "prediction"  # the question + choice buttons
    COUNTDOWN = "countdown"    # a beat of anticipation before the run
    EXPERIMENT = "experiment"  # the chosen scenario is running
    REVEAL = "reveal"          # what actually happened, in plain language
    SCIENCE = "science"        # the same moment, with real numbers
    COMPLETE = "complete"      # replay / try the other choice
    GAMES_HOME = "games_home"  # the Games/Challenges menu, entered intentionally
    GAME_PLAY = "game_play"    # one focused game/tool, live scene still underneath


# The only forward transitions the experience itself drives. SCIENCE is
# reachable from REVEAL and returns there, so it's a detour rather than a
# step -- a visitor who never presses "Show me the science" still has a
# complete journey. GAMES_HOME/GAME_PLAY are reachable from OBSERVE at any
# time (not just at this point in the chain); the entries below are just
# for `advance()`'s completeness, since the experience always drives these
# two via explicit `go_to()` calls rather than `advance()`.
_NEXT = {
    Phase.ATTRACT: Phase.INTRO,
    Phase.INTRO: Phase.OBSERVE,
    Phase.OBSERVE: Phase.PREDICTION,
    Phase.PREDICTION: Phase.COUNTDOWN,
    Phase.COUNTDOWN: Phase.EXPERIMENT,
    Phase.EXPERIMENT: Phase.REVEAL,
    Phase.REVEAL: Phase.COMPLETE,
    Phase.SCIENCE: Phase.COMPLETE,
    Phase.COMPLETE: Phase.ATTRACT,
    Phase.GAMES_HOME: Phase.GAME_PLAY,
    Phase.GAME_PLAY: Phase.GAMES_HOME,
}


@dataclass
class PublicState:
    """Everything the public experience needs to know about where the
    visitor is. One instance per PublicExperience."""

    phase: Phase = Phase.ATTRACT
    # Which scenario is on screen right now (a manifest case_index), and
    # the baseline the experiment started from.
    case_index: Optional[int] = None
    baseline_case_index: Optional[int] = None
    frame_index: int = 0
    # The experiment currently loaded (an experiments.Experiment) and the
    # key of the choice the visitor picked ("fan_on" / "fan_off" / ...).
    experiment: object = None
    prediction: Optional[str] = None
    choice: Optional[str] = None
    # True once the visitor has pressed "Show me the science" at least
    # once this run -- the button changes label rather than toggling
    # invisibly.
    science_visible: bool = False
    # Choice keys already tried this session, so "try the other one" can
    # offer something new instead of repeating.
    tried_choices: list = field(default_factory=list)
    # Where replay() should land -- OBSERVE by default, but GAMES_HOME
    # when the run being replayed ("Test an idea") was entered from the
    # Games hub rather than directly from free play, so "Try again"
    # returns the visitor to the hub instead of dropping them into
    # Explore. Reset alongside everything else in reset().
    return_phase: Phase = Phase.OBSERVE

    # -- transitions ----------------------------------------------------
    def advance(self) -> Phase:
        """Move to the next phase in the visitor's journey."""
        self.phase = _NEXT[self.phase]
        return self.phase

    def go_to(self, phase: Phase) -> Phase:
        self.phase = phase
        return self.phase

    def record_prediction(self, prediction_key: str) -> None:
        """The visitor's *guess*, recorded before the experiment runs.
        Kept separate from `choice` (what actually got simulated) so the
        reveal can honestly say whether the guess matched."""
        self.prediction = prediction_key

    def record_choice(self, choice_key: str) -> None:
        self.choice = choice_key
        if choice_key not in self.tried_choices:
            self.tried_choices.append(choice_key)

    def toggle_science(self) -> bool:
        self.science_visible = not self.science_visible
        return self.science_visible

    def reset(self) -> None:
        """Back to attract, for the next visitor. Deliberately keeps
        nothing: an exhibit that remembers the previous person's guess is
        confusing, and `tried_choices` is per-visitor by definition."""
        self.phase = Phase.ATTRACT
        self.frame_index = 0
        self.prediction = None
        self.choice = None
        self.science_visible = False
        self.tried_choices = []
        self.case_index = self.baseline_case_index
        self.return_phase = Phase.OBSERVE

    def replay(self) -> None:
        """Start the run again from the observation step.

        Deliberately OBSERVE and not PREDICTION. "Try again" at an exhibit
        is usually pressed by the *next* person in the queue, who has seen
        none of this; dropping them straight on the question left the
        prediction card sitting over a scene paused at frame 0, i.e. an
        unlit candle. Re-running the observation costs a few seconds and
        makes a replay indistinguishable from a fresh session.

        `tried_choices` is kept so the UI can still tell which option this
        visitor has already seen; everything else about the run resets.

        Lands on `return_phase` rather than hardcoding OBSERVE: a run
        entered from the Games hub ("Test an idea") replays back to the
        hub, not into free play.
        """
        self.phase = self.return_phase
        self.frame_index = 0
        self.prediction = None
        self.choice = None
        self.science_visible = False
        self.case_index = self.baseline_case_index
