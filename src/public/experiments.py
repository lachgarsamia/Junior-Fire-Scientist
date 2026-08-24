"""Experiments: a declarative prediction -> run -> reveal unit.

An Experiment names the factor it changes, the factors it holds fixed,
and the choices a visitor can make. Scenario case indices are *resolved
from the manifest at runtime* -- never hardcoded -- so an experiment
keeps working against any study that has the same factor axes, and fails
cleanly (returns None) against one that doesn't.

Why the fan and not the door
----------------------------
Measured across all 24 scenarios, matched on every other factor:

  door narrow -> wide : mean air speed changes by +-0.03 m/s, and the
                        sign flips depending on vent state; end-state
                        temperature moves 0.0-1.7 C.
  vents -> HVAC (fan) : mean air speed goes 0.09 -> 0.62 m/s (~7x), peak
                        speed ~0.5 -> ~3.0 m/s, and the room ends cooler.

The door is a genuinely weak factor in this dataset. Leading with it
would show a visitor a null result and teach them the opposite of what
the data says, so the fan is the first experiment. `measure()` below
exists to keep that honest over time: it computes the real effect size
from the store, and the test suite asserts a shipped experiment actually
clears a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from slice_key import DEFAULT_SLICE_KEY, SliceKey
from public.i18n import tr

VELOCITY_KEY = SliceKey("VELOCITY", 1, 0)


# Every "text" field in this module (Choice.label, Experiment.title, ...)
# actually stores an i18n.py translation KEY, not literal display text --
# these dataclasses are built once at import time, frozen, and reused for
# the life of the process, so baking in a literal string would freeze it
# in whatever language happened to be active at import. Each class below
# exposes the *displayed* text through an ordinary @property with the
# same name a caller would otherwise expect the field to have (e.g.
# `choice.label`), calling tr() fresh on every access -- so every
# existing call site across the app keeps working unchanged, and a
# language switch is picked up the next time anything reads the
# property, with no separate refresh step of its own.
@dataclass(frozen=True)
class Choice:
    """One option the visitor can pick. `factors` is the factor override
    applied on top of the experiment's `held` values to identify a real
    scenario."""
    key: str
    label_key: str
    icon: str
    factors: dict
    # Column/bar heading form. The button label is a sentence ("Switch
    # the fan ON") and is far too wide for a chart axis, so charts use
    # this instead; it falls back to `label` when not given.
    short_label_key: str = ""

    @property
    def label(self) -> str:
        return tr(self.label_key)

    @property
    def short(self) -> str:
        return tr(self.short_label_key) if self.short_label_key else self.label


@dataclass(frozen=True)
class Prediction:
    """A guess the visitor makes before seeing the result."""
    key: str
    label_key: str
    icon: str
    # Which measured metric this guess is actually about, and the line to
    # show when the data backs it. Declared here rather than inferred, so
    # the science card can tie a result back to the guess without any
    # experiment-specific logic in the presentation layer. Left empty for
    # guesses no metric can confirm.
    metric_key: str = ""
    confirmation_key: str = ""

    @property
    def label(self) -> str:
        return tr(self.label_key)

    @property
    def confirmation(self) -> str:
        return tr(self.confirmation_key) if self.confirmation_key else ""


@dataclass(frozen=True)
class Experiment:
    key: str
    title_key: str
    # Shown during OBSERVE, before the question.
    observe_prompt_key: str
    question_key: str
    predictions: tuple
    choices: tuple
    held: dict = field(default_factory=dict)
    # The choice key that represents "leave it as it was" -- the
    # reference the reveal compares against.
    baseline_choice: str = ""
    # Which prediction the data actually supports. Used only to tell the
    # visitor whether their guess matched; never to invent a claim.
    supported_prediction: str = ""

    @property
    def title(self) -> str:
        return tr(self.title_key)

    @property
    def observe_prompt(self) -> str:
        return tr(self.observe_prompt_key)

    @property
    def question(self) -> str:
        return tr(self.question_key)

    def choice(self, key: str) -> Optional[Choice]:
        return next((c for c in self.choices if c.key == key), None)

    def next_untried_choice(self, tried) -> Optional[str]:
        """The choice to run next, given what this visitor has already
        seen. Declaration order decides, so an experiment puts its most
        striking option first and a fresh visitor gets that one; a replay
        then shows them the side they have not seen. Once everything has
        been tried it cycles back to the first rather than refusing to
        run."""
        if not self.choices:
            return None
        seen = set(tried or ())
        return next((c.key for c in self.choices if c.key not in seen),
                    self.choices[0].key)

    def all_choices_tried(self, tried) -> bool:
        """True once the visitor has run every side of this experiment."""
        if not self.choices:
            return False
        return {c.key for c in self.choices}.issubset(set(tried or ()))

    def contrast_choice(self) -> Optional[str]:
        """The non-baseline choice -- the one the experiment is *about*.
        Used to phrase the comparison in a fixed direction regardless of
        which side the visitor happens to be watching."""
        return next((c.key for c in self.choices if c.key != self.baseline_choice), None)

    def prediction(self, key: str) -> Optional[Prediction]:
        return next((p for p in self.predictions if p.key == key), None)


# The MVP experiment. Wording is deliberately about *airflow*, which is
# what the fan unambiguously changes; the temperature effect is real but
# small (~1-2 C), so the reveal text is generated from measured numbers
# (see reveal_sentences) rather than asserting a fixed claim here.
FAN_EXPERIMENT = Experiment(
    key="fan",
    title_key="experiment_fan_title",
    # Deliberately does not say where the smoke ends up -- the story beat
    # points that out at the moment it actually happens, and announcing it
    # up front spoils the one thing the observe step exists to reveal.
    observe_prompt_key="experiment_fan_observe_prompt",
    question_key="experiment_fan_question",
    # Short labels on purpose: three choice buttons have to sit side by
    # side and stay readable from a couple of metres away.
    predictions=(
        Prediction("bigger", "prediction_bigger_fire", "📈"),
        Prediction("cooler", "prediction_cooler_room", "❄️", metric_key="room_temp",
                   confirmation_key="prediction_cooler_confirmation"),
        Prediction("nothing", "prediction_nothing_changes", "😐"),
    ),
    choices=(
        Choice("fan_on", "choice_fan_on", "🌬️", {"vod": 2}, short_label_key="choice_fan_on_short"),
        Choice("fan_off", "choice_fan_off", "🚫", {"vod": 0}, short_label_key="choice_fan_off_short"),
    ),
    held={"candles": 0, "door": 1, "voc": 0},
    baseline_choice="fan_off",
    supported_prediction="cooler",
)

EXPERIMENTS = (FAN_EXPERIMENT,)


@dataclass(frozen=True)
class ExploreOption:
    """One state of a free-play control -- e.g. the "OFF" or "ON" side
    of the fan. `value` is the raw factor value stored on a manifest
    entry (an int, matching ScenarioEntry's own fields)."""
    value: object
    label_key: str
    icon: str

    @property
    def label(self) -> str:
        return tr(self.label_key)


@dataclass(frozen=True)
class ExploreControl:
    """A real, data-driven control the visitor can flip during free
    exploration -- before the guided prediction begins, not instead of
    it. Resolution goes through resolve_case_index() exactly like a
    Choice, so a control can never point at a scenario this study
    doesn't have.

    Changes compose: flipping the fan and then the vent lands on the
    real fan-on + vent-shut scenario, not a reset back to a fixed
    reference (this study's own factorial has every combination these
    controls can reach -- see EXPLORE_CONTROLS' own comment). The
    composition itself lives in PublicExperience._on_explore_changed/
    _current_factors, not here: this dataclass only describes one
    control's own key/label/factor/options.

    `held` is this control's baseline -- every *other* exposed factor's
    default value, used only to check the control is actually usable at
    all (is_available, below) and as case_for's fixed-baseline
    resolution, which is still what _current_factors falls back to
    before any scenario has been loaded yet. It is not consulted by the
    live compose-on-top-of-what's-already-loaded path once one is.
    """
    key: str
    label_key: str
    icon: str
    factor: str             # manifest field name, e.g. "vod"
    options: tuple           # ExploreOption entries, in display order
    held: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return tr(self.label_key)

    @property
    def default_value(self):
        return self.options[0].value if self.options else None

    def case_for(self, manifest: list, value) -> Optional[int]:
        """This control's own factor at `value`, every other exposed
        factor at its baseline (`held`) -- used for availability checks
        and as a before-anything-is-loaded fallback, not the live
        compose-with-whatever-else-is-set path (see the class
        docstring)."""
        return resolve_case_index(manifest, {**self.held, self.factor: value})

    def is_available(self, manifest: list) -> bool:
        """True only when *every* option resolves to a real scenario --
        a control half-missing from the study is not shown at all."""
        return bool(self.options) and all(
            self.case_for(manifest, opt.value) is not None for opt in self.options)


# Four explore controls, all real factors in this study's own factorial
# (candles x door x vod x voc, all 24 combinations present -- see
# manifest.factor_counts) -- one control per factor the original FDS
# input files actually vary, named and stated to match them exactly
# (fds/generate_sim.py's candle/door/vertical_opening_1/vertical_opening_2,
# manifest.py's candles/door/vod/voc). No control is a stand-in name for a
# factor ("fan" used to be) or an invented state -- every ExploreOption
# below is one of the real levels generate_sim.py actually produced.
#
# Vent 1 (vod) has three real states: open, closed, and HVAC -- the HVAC
# state is a real powered fan (see fds/template_hvac.fds's &HVAC blocks),
# which is why it gets its own distinct "FAN ON" option/color rather than
# being folded into "open" as the old binary fan control did (that control
# skipped the closed state entirely -- see git history).
#
# Vent 2 (voc) has two real states, open/closed, and no fan -- there is no
# HVAC block on this opening. Its measured effect is close to null (open ->
# closed, vod=0 baseline: mean air speed 0.085 -> 0.076 m/s, below the
# 0.02 m/s noticeable_delta; room temp 26.5 -> 26.3 C, below the 0.3 C
# one) -- measured the same way the module docstring's door numbers were,
# not assumed identical to them. Exposing it anyway teaches a real, honest
# lesson ("not everything you can touch changes much") via the existing
# noticeable_delta machinery, which already reports "about the same"
# rather than dressing up a null result.
#
# Door: also close to null (module docstring: +-0.03 m/s, 0.0-1.7 C) --
# previously left out of free play for exactly that reason. Exposed here
# per supervisor feedback: the original setup has a real, visible door and
# hiding it was itself teaching an incomplete picture of the setup. Same
# honesty machinery as Vent 2 covers it.
EXPLORE_CONTROLS = (
    ExploreControl(
        "vent1", "control_vent1_label", "🌬️", "vod",
        (ExploreOption(0, "option_vent_open", "🔓"),
         ExploreOption(1, "option_vent_closed", "🔒"),
         ExploreOption(2, "option_fan_on", "🌀")),
        held={"candles": 0, "door": 1, "voc": 0}),
    ExploreControl(
        "candles", "control_candles_label", "🕯️", "candles",
        (ExploreOption(0, "option_one_candle", "🕯️"),
         ExploreOption(1, "option_two_candles", "🕯️🕯️")),
        held={"door": 1, "vod": 0, "voc": 0}),
    ExploreControl(
        "vent2", "control_vent2_label", "🪟", "voc",
        (ExploreOption(0, "option_vent_open", "🔓"),
         ExploreOption(1, "option_vent_closed", "🔒")),
        held={"candles": 0, "door": 1, "vod": 0}),
    ExploreControl(
        "door", "control_door_label", "🚪", "door",
        (ExploreOption(0, "option_door_narrow", "🚪"),
         ExploreOption(1, "option_door_wide", "🚪")),
        held={"candles": 0, "vod": 0, "voc": 0}),
)


def available_explore_controls(manifest: list) -> list:
    """Explore controls this study can actually back, in declaration
    order. Empty (not a placeholder) when the manifest has none -- the
    caller simply shows no explore row rather than a broken one."""
    return [c for c in EXPLORE_CONTROLS if c.is_available(manifest)]


def resolve_case_index(manifest: list, factors: dict) -> Optional[int]:
    """The case_index of the scenario matching every factor in `factors`,
    or None if this study has no such scenario."""
    for entry in manifest or []:
        if all(getattr(entry, name, None) == value for name, value in factors.items()):
            return entry.case_index
    return None


def resolve_choice(manifest: list, experiment: Experiment, choice_key: str) -> Optional[int]:
    """case_index for one choice of an experiment (held factors plus that
    choice's overrides)."""
    choice = experiment.choice(choice_key)
    if choice is None:
        return None
    return resolve_case_index(manifest, {**experiment.held, **choice.factors})


def is_available(manifest: list, experiment: Experiment) -> bool:
    """True when every choice resolves to a real scenario."""
    return all(resolve_choice(manifest, experiment, c.key) is not None
               for c in experiment.choices)


@dataclass(frozen=True)
class Measurement:
    """Real, computed values for one choice's scenario. Every field is a
    direct reduction over stored FDS data."""
    case_index: int
    mean_airspeed: float
    max_airspeed: float
    mean_temperature_end: float
    peak_temperature: float


def measure_case(store, case_index: Optional[int]) -> Optional[Measurement]:
    """The same reduction measure() uses, keyed directly by case_index --
    for comparing two scenarios a visitor actually explored during free
    play, which need not be either side of a declared Experiment/Choice.
    """
    if case_index is None:
        return None
    temperature = np.asarray(store.get(case_index, DEFAULT_SLICE_KEY))
    try:
        velocity = np.asarray(store.get(case_index, VELOCITY_KEY))
        mean_v = float(velocity.mean())
        max_v = float(velocity.max())
    except Exception:  # noqa: BLE001 - a study without velocity still gets temperatures
        mean_v = max_v = 0.0
    return Measurement(
        case_index=case_index,
        mean_airspeed=mean_v,
        max_airspeed=max_v,
        mean_temperature_end=float(temperature[-1].mean()),
        peak_temperature=float(temperature.max()),
    )


def measure(store, manifest: list, experiment: Experiment, choice_key: str) -> Optional[Measurement]:
    """Compute the summary numbers the reveal and science views show.

    Cheap for this dataset (all fields are in the .npy disk cache), and
    the caller caches the result per choice anyway.
    """
    return measure_case(store, resolve_choice(manifest, experiment, choice_key))


@dataclass(frozen=True)
class Metric:
    """One measurable a public science card can report.

    Declarative and experiment-agnostic: every field a Measurement
    carries can be described once here and then rendered, ranked and
    phrased generically, so a future experiment needs no new presentation
    code. `noticeable_delta` is the honesty guard -- a change smaller
    than this is reported as "about the same" rather than dressed up.
    """
    key: str
    label_key: str      # short: the rows are aligned and must not wrap
    icon: str
    unit: str
    decimals: int
    field: str          # attribute name on Measurement
    higher_word_key: str
    lower_word_key: str
    noticeable_delta: float
    meaning_key: str    # one line of "what does this actually mean?"
    # Spoken form, for the guide's one-sentence interpretation. `subject`
    # is a clause opener ("the air moved") and the two words complete it
    # in whichever direction was measured, so a sentence can be composed
    # generically rather than templated per experiment. `higher_word` /
    # `lower_word` above stay as they are -- they read correctly beside a
    # number ("6.4x stronger"), which is a different job.
    kid_subject_key: str = ""
    kid_more_key: str = ""
    kid_less_key: str = ""
    # Why the finding matters physically. Deliberately a separate field
    # from the spoken finding: finding_sentence() states what the
    # simulation *measured*, this states what that measurement *means*.
    # The two must not be generated from one another -- a measured
    # direction can be composed mechanically, a physical claim cannot.
    # Every explanation below is either textbook advection or something
    # this study's own numbers demonstrate; none asserts a mechanism the
    # data does not show. Optional: a metric with none simply contributes
    # no explanation line.
    explanation_key: str = ""

    @property
    def label(self) -> str:
        return tr(self.label_key)

    @property
    def higher_word(self) -> str:
        return tr(self.higher_word_key)

    @property
    def lower_word(self) -> str:
        return tr(self.lower_word_key)

    @property
    def meaning(self) -> str:
        return tr(self.meaning_key)

    @property
    def kid_subject(self) -> str:
        return tr(self.kid_subject_key) if self.kid_subject_key else ""

    @property
    def kid_more(self) -> str:
        return tr(self.kid_more_key) if self.kid_more_key else ""

    @property
    def kid_less(self) -> str:
        return tr(self.kid_less_key) if self.kid_less_key else ""

    @property
    def explanation(self) -> str:
        return tr(self.explanation_key) if self.explanation_key else ""


# The metrics the public science card reports, in display order. Ranking
# for the hero metric is computed, not implied by this order.
PUBLIC_METRICS = (
    Metric("airspeed", "metric_airspeed_label", "💨", "m/s", 3, "mean_airspeed",
           "metric_stronger", "metric_weaker", 0.02,
           "metric_airspeed_meaning",
           kid_subject_key="metric_airspeed_kid_subject",
           kid_more_key="metric_airspeed_kid_more", kid_less_key="metric_airspeed_kid_less",
           explanation_key="metric_airspeed_explanation"),
    # "Air temp.", not "Room temp.": mean_temperature_end averages the
    # whole simulated area, of which the enclosed room is only about a
    # third (schematic.ROOM_X/ROOM_Z vs the domain). Calling it the room
    # average overstated what the number covers; restricting the
    # reduction to the room instead would be a geometry-aware
    # computation, which does not belong in the public layer.
    Metric("room_temp", "metric_room_temp_label", "🌡️", "°C", 1, "mean_temperature_end",
           "metric_warmer", "metric_cooler", 0.3,
           "metric_room_temp_meaning",
           kid_subject_key="metric_room_temp_kid_subject",
           kid_more_key="metric_room_temp_kid_more", kid_less_key="metric_room_temp_kid_less",
           explanation_key="metric_room_temp_explanation"),
    # 40 °C keeps the flame row consistent with what the reveal already
    # tells visitors -- that the flame burns about as hot either way --
    # while still showing both measured values.
    Metric("flame_temp", "metric_flame_temp_label", "🕯️", "°C", 0, "peak_temperature",
           "metric_hotter", "metric_cooler", 40.0,
           "metric_flame_temp_meaning",
           kid_subject_key="metric_flame_temp_kid_subject",
           kid_more_key="metric_flame_temp_kid_more", kid_less_key="metric_flame_temp_kid_less",
           explanation_key="metric_flame_temp_explanation"),
)


@dataclass(frozen=True)
class MetricComparison:
    """One metric measured on both sides of an experiment, always in the
    baseline -> contrast direction so neither the numbers nor the wording
    depend on which side the visitor happened to watch last."""
    metric: Metric
    baseline: float
    contrast: float

    @property
    def delta(self) -> float:
        return self.contrast - self.baseline

    @property
    def ratio(self) -> float:
        return self.contrast / self.baseline if self.baseline else 0.0

    @property
    def relative_change(self) -> float:
        """Fractional change, used to rank which metric moved most."""
        return abs(self.delta) / abs(self.baseline) if self.baseline else 0.0

    @property
    def is_noticeable(self) -> bool:
        return abs(self.delta) >= self.metric.noticeable_delta

    def change_text(self) -> str:
        """"6.4x stronger" / "1.5 °C cooler" / "about the same"."""
        if not self.is_noticeable:
            return tr("metric_about_the_same")
        word = self.metric.higher_word if self.delta > 0 else self.metric.lower_word
        if self.ratio >= 2.0:
            return f"{self.ratio:.1f}× {word}"
        return f"{abs(self.delta):.{self.metric.decimals}f} {self.metric.unit} {word}"

    def value_text(self, value: float) -> str:
        return f"{value:.{self.metric.decimals}f}"


def compare_metrics(baseline: Measurement, contrast: Measurement,
                    metrics=PUBLIC_METRICS) -> list:
    """Every metric measured on both sides, in declaration order."""
    if baseline is None or contrast is None:
        return []
    return [MetricComparison(m, getattr(baseline, m.field), getattr(contrast, m.field))
            for m in metrics]


def strongest_metric(comparisons: list):
    """The metric that changed most, by fractional change -- the finding
    the science card leads with. None if nothing moved noticeably, so a
    null result is never dressed up as a headline."""
    noticeable = [c for c in comparisons if c.is_noticeable]
    if not noticeable:
        return None
    return max(noticeable, key=lambda c: c.relative_change)


def secondary_metric(comparisons: list, hero=None):
    """The next real finding after the hero, or None.

    Only metrics that clear their own noticeable_delta qualify, so a
    negligible difference is never promoted into a second chart -- the
    same guard that makes change_text() say "about the same" decides
    whether a metric gets a visual at all.
    """
    if hero is None:
        hero = strongest_metric(comparisons)
    if hero is None:
        return None
    rest = [c for c in comparisons
            if c.is_noticeable and c.metric.key != hero.metric.key]
    if not rest:
        return None
    return max(rest, key=lambda c: c.relative_change)


def unchanged_metrics(comparisons: list, shown: list) -> list:
    """Metrics not given their own visual -- reported as plain rows so
    the evidence stays complete even when a value barely moved."""
    shown_keys = {c.metric.key for c in shown if c is not None}
    return [c for c in comparisons if c.metric.key not in shown_keys]


def effect_size(store, manifest: list, experiment: Experiment) -> float:
    """Ratio of mean air speed between the strongest and the baseline
    choice -- the number that justifies shipping this experiment at all.
    Returns 1.0 (no effect) when it cannot be computed."""
    baseline = measure(store, manifest, experiment, experiment.baseline_choice)
    if baseline is None or baseline.mean_airspeed <= 0:
        return 1.0
    ratios = []
    for choice in experiment.choices:
        if choice.key == experiment.baseline_choice:
            continue
        other = measure(store, manifest, experiment, choice.key)
        if other is not None:
            ratios.append(other.mean_airspeed / baseline.mean_airspeed)
    return max(ratios) if ratios else 1.0
