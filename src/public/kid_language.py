"""Plain-language interpretation of real simulation values.

Pure functions, no Qt, no state -- so the thresholds are unit-testable,
which matters more here than anywhere else in the public layer: these are
the sentences a child walks away believing.

Calibration note (this is the load-bearing part). The bands below are
tuned to *this* dataset, which is a tabletop candle experiment in a
0.73 x 0.22 m box, not a building fire. Measured across all 24 scenarios:

  - the flame core reaches ~330-470 C
  - the air away from the flame stays roughly 20-35 C for the whole run
  - mean air speed is ~0.08-0.12 m/s with the vents open or closed, and
    ~0.55-0.74 m/s with the fan (HVAC) running; 99th-percentile speed
    goes from ~0.5 m/s to ~3.0 m/s

A generic five-band "cool -> DANGEROUS" temperature scale would therefore
label almost the entire room "cool" and teach the visitor nothing, or --
worse, if the bands were set for a house fire -- imply the room is deadly
when it is 26 C. The bands here describe what is actually on screen.
"""

from __future__ import annotations

from registry import AMBIENT_C
from public.i18n import tr

# Temperature bands, C. Chosen against the measured distribution above:
# everything up to ~28 C is "room temperature" in this box, the plume and
# ceiling layer live in the 30-120 C range, and only the flame itself goes
# beyond that.
#
# The middle element of each tuple is an i18n.py translation key, not
# display text -- band membership (e.g. STRONG_AIRFLOW_KEYS below) is
# compared by key, which must stay stable across a language switch. Call
# temperature_phrase()/airflow_phrase() to get the translated text.
_TEMPERATURE_BANDS = (
    (28.0, "temp_band_room", "🧊"),
    (45.0, "temp_band_little_warm", "🙂"),
    (90.0, "temp_band_warm_air", "😅"),
    (200.0, "temp_band_hot", "🥵"),
    (400.0, "temp_band_very_hot", "🌋"),
    (float("inf"), "temp_band_flame", "🕯️"),
)

# Air-speed bands, m/s. Boundaries are placed in the *gap* between the
# two regimes this dataset actually contains, so the two never share a
# label: buoyancy-only runs (vents open or closed) average 0.04-0.12 m/s,
# and fan runs average 0.54-0.74 m/s. The 0.45 boundary sits in the empty
# space between them rather than inside either cluster.
_AIRFLOW_BANDS = (
    (0.05, "airflow_band_still", "·"),
    (0.20, "airflow_band_gentle", "💨"),
    (0.45, "airflow_band_moving", "💨"),
    (1.50, "airflow_band_strong", "🌬️"),
    (float("inf"), "airflow_band_very_strong", "🌬️"),
)


# Band keys that count as "the air is really moving". Named here so
# callers can react to a measured speed without duplicating thresholds --
# compared against airflow_band_key()'s stable key, never against the
# translated airflow_phrase() text.
STRONG_AIRFLOW_KEYS = ("airflow_band_strong", "airflow_band_very_strong")

# One colour per _TEMPERATURE_BANDS entry, same order -- so the public
# thermometer widget's fill colour never disagrees with the phrase beside
# it (both come from the same band boundaries, defined once here).
_TEMPERATURE_BAND_COLORS = ("#5AA9E6", "#5FD68A", "#F5D547", "#FF9440", "#FF5A36", "#FF2E00")


# The thermometer's tube is not a linear scale: 0-SCALE_BREAK_C gets
# SCALE_BREAK_FRAC of the tube's own height, and SCALE_BREAK_C..display
# ceiling gets the rest. Almost everything a visitor actually watches
# change during free exploration (the "whole room average" reading)
# sits well under 100 C -- the flame itself pins near the display
# ceiling regardless of what's happening in the room, so giving that
# range the same linear share as 0-100 C would waste most of the tube
# on a number that barely moves. Shared by the fill height, the tick
# marks, and the gradient stops (temperature_gradient_stops, below) so
# all three always agree on where a given degree actually sits.
SCALE_BREAK_C = 100.0
SCALE_BREAK_FRAC = 0.62


def temperature_scale_fraction(value_c: float, display_max_c: float) -> float:
    """0..1 position along the thermometer's tube for `value_c`, using
    the two-segment scale SCALE_BREAK_C/SCALE_BREAK_FRAC describe.
    Clamped to [0, display_max_c] first -- same behaviour the old
    linear mapping had for a value past the display ceiling."""
    value_c = max(0.0, min(value_c, display_max_c))
    if value_c <= SCALE_BREAK_C:
        return SCALE_BREAK_FRAC * (value_c / SCALE_BREAK_C)
    span = display_max_c - SCALE_BREAK_C
    if span <= 0:
        return 1.0
    return SCALE_BREAK_FRAC + (1.0 - SCALE_BREAK_FRAC) * ((value_c - SCALE_BREAK_C) / span)


def temperature_gradient_stops(display_max_c: float) -> list:
    """(position, hex_color) pairs for a full-height blue-to-red gradient
    brush, position 1.0 = coolest (bottom) to 0.0 = hottest (top) -- the
    exact same band boundaries/colours temperature_color()'s flat fill
    uses, just expressed as a continuous gradient so a partial fill
    reveals a smooth blue->red sweep instead of jumping between flat
    colours as the reading crosses a band threshold. Positions come from
    temperature_scale_fraction, the same non-linear scale the tube's
    fill height and tick marks use."""
    stops = []
    for (threshold, _label, _icon), color in zip(_TEMPERATURE_BANDS, _TEMPERATURE_BAND_COLORS):
        frac = temperature_scale_fraction(min(threshold, display_max_c), display_max_c)
        stops.append((1.0 - frac, color))
    return stops


def _band(value: float, bands) -> tuple:
    for threshold, key, icon in bands:
        if value < threshold:
            return key, icon
    return bands[-1][1], bands[-1][2]


def temperature_phrase(temp_c: float) -> str:
    """"Warm air" etc. for a temperature in C."""
    return tr(_band(temp_c, _TEMPERATURE_BANDS)[0])


def temperature_icon(temp_c: float) -> str:
    return _band(temp_c, _TEMPERATURE_BANDS)[1]


def mood_emoji(temp_c: float) -> str:
    """A simple 3-tier mood (🥶 cold / 🙂 comfortable / 🥵 hot) for the
    thermometer's quick-glance indicator -- distinct from temperature_
    icon()'s existing 6-tier band icon, which is more precise but too
    granular for "how does this feel?" at a glance. Reuses this dataset's
    own real thresholds rather than inventing a fourth scale: below
    AMBIENT_C is genuinely cooler than the room started at, 45 C is the
    same "a little warm" boundary _TEMPERATURE_BANDS already uses."""
    if temp_c < AMBIENT_C - 2.0:
        return "🥶"
    if temp_c < 45.0:
        return "🙂"
    return "🥵"


def temperature_color(temp_c: float) -> str:
    """Hex fill colour for the band `temp_c` falls in -- the same bands
    temperature_phrase()/temperature_icon() use, for the thermometer
    widget's coloured tube."""
    for (threshold, _label, _icon), color in zip(_TEMPERATURE_BANDS, _TEMPERATURE_BAND_COLORS):
        if temp_c < threshold:
            return color
    return _TEMPERATURE_BAND_COLORS[-1]


# Playful reactions for the "find the hottest place" game -- the exact
# same band thresholds/icons _TEMPERATURE_BANDS already uses (never a
# separate arbitrary scale), just game-voiced text instead of a plain
# description. Keys, not literal text -- see _TEMPERATURE_BANDS' own note.
_HEAT_GUESS_REACTION_KEYS = (
    "heat_guess_cool", "heat_guess_warmer", "heat_guess_hot", "heat_guess_thats_hot",
    "heat_guess_wow", "heat_guess_wow_flame",
)


def heat_guess_reaction(temp_c: float) -> str:
    """"🧊 Cool!" up to "🕯️ WOW!" for a guessed location's real measured
    temperature -- reuses _TEMPERATURE_BANDS' own boundaries so this can
    never drift from the thermometer's own bands."""
    for (threshold, _key, icon), text_key in zip(_TEMPERATURE_BANDS, _HEAT_GUESS_REACTION_KEYS):
        if temp_c < threshold:
            return f"{icon} {tr(text_key)}"
    return f"{_TEMPERATURE_BANDS[-1][2]} {tr(_HEAT_GUESS_REACTION_KEYS[-1])}"


def airflow_phrase(speed_ms: float) -> str:
    return tr(_band(speed_ms, _AIRFLOW_BANDS)[0])


def airflow_band_key(speed_ms: float) -> str:
    """The stable, language-independent band id for a measured speed --
    compare against STRONG_AIRFLOW_KEYS with this, never against
    airflow_phrase()'s translated text (which changes with the language)."""
    return _band(speed_ms, _AIRFLOW_BANDS)[0]


def airflow_icon(speed_ms: float) -> str:
    return _band(speed_ms, _AIRFLOW_BANDS)[1]


def warmth_above_ambient(temp_c: float) -> str:
    """How much warmer than the room started, phrased for a child. The
    absolute number is small here (a few degrees), so saying "6 degrees
    warmer than when we started" is both true and more meaningful than
    "26 C"."""
    delta = temp_c - AMBIENT_C
    if delta < 0.5:
        return tr("warmth_same_as_start")
    if delta < 1.5:
        return tr("warmth_one_degree")
    return tr("warmth_degrees_warmer", delta=delta)


# A change this many times over reads as "much", below it as "a little".
# Same boundary compare_phrase() already uses for "times stronger".
MUCH_CHANGE_RATIO = 2.0


def finding_sentence(comparison) -> str:
    """A one-clause, child-readable statement of what a measured metric
    did: "the air moved much faster", "the room got a little cooler".

    Composed from the metric's declared spoken form plus the *measured*
    direction and size, so it works for any metric without a template per
    experiment. Returns "" when the change is below that metric's own
    noticeable_delta -- the honesty guard stays authoritative here exactly
    as it does for change_text(), so a negligible difference is never
    narrated as a discovery.

    `comparison` is duck-typed (an experiments.MetricComparison); this
    module deliberately imports nothing from experiments, so the language
    layer stays the leaf it has always been. `metric.kid_subject`/
    `kid_more`/`kid_less` are themselves already-translated text (see
    experiments.Metric's own tr()-backed properties), so only the
    intensifier here needs its own per-language form -- English and
    German both happen to take it as a prefix on the comparative word
    ("much faster" / "viel schneller"), so one template covers both.
    """
    metric = comparison.metric
    if not comparison.is_noticeable or not metric.kid_subject:
        return ""
    word = metric.kid_more if comparison.delta > 0 else metric.kid_less
    if not word:
        return ""
    ratio = comparison.ratio if comparison.ratio >= 1 else (
        1.0 / comparison.ratio if comparison.ratio else 1.0)
    intensifier = tr("intensifier_much") if ratio >= MUCH_CHANGE_RATIO else tr("intensifier_a_little")
    return f"{metric.kid_subject} {intensifier}{word}"


def mascot_finding(comparison, both_tried: bool = False) -> str:
    """The guide's line about the headline result, or "" if there isn't
    one worth stating. Kept to a single short sentence -- the exact
    values stay on the card; the guide only interprets."""
    sentence = finding_sentence(comparison)
    if not sentence:
        return ""
    if both_tried:
        return tr("mascot_finding_both_tried", sentence=sentence)
    return tr("mascot_finding_default", icon=comparison.metric.icon, sentence=sentence)


def compare_phrase(value_a: float, value_b: float, noun: str) -> str:
    """An honest comparison between two measured values -- returns a
    "barely changed" phrasing when the difference is small, so a weak
    effect never gets narrated as a strong one. `value_a` is the
    reference (what happened without the change)."""
    if value_a <= 0:
        return tr("compare_phrase_changed", noun=noun)
    ratio = value_b / value_a
    if ratio >= 2.0:
        return tr("compare_phrase_times_stronger", noun=noun, ratio=ratio)
    if ratio >= 1.25:
        return tr("compare_phrase_stronger", noun=noun)
    if ratio <= 0.5:
        return tr("compare_phrase_half", noun=noun)
    if ratio <= 0.8:
        return tr("compare_phrase_weaker", noun=noun)
    return tr("compare_phrase_barely", noun=noun)
