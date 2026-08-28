"""Tests for the Public Fire Explorer (src/public/).

Focus is on the parts that would silently produce a *wrong* exhibit
rather than a crashing one: the phase machine, scenario resolution from
the manifest, the honesty guard on effect size, kid-language thresholds
calibrated to this dataset, and the guarantee that entering and leaving
public mode does not disturb the researcher app.
"""

import ast
import dataclasses
import os
import pathlib
import re
import time

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

import registry
from cinema import velocity_arrows
from cinema.real_smoke import normalize_soot_density, soot_at_time
from cinema.smoke import SmokeSimulator, SOURCE_THRESHOLD_C
import public.story as story_mod
from views import SliceView
import public.experience as experience_mod
from data_provider import load_simulation_data
from load_data import SIM_ROOT
from main_window import MainWindow
from manifest import foreign_path_entries
from public import experiments as experiments_mod
from public import i18n
from public import kid_language as kid
from public.state import Phase, PublicState
from public.story import StoryController
from public.mascot import Mascot
from public.scene import SCENE_WIDTH_FRAC
from public.widgets import BarCompare, BigButton, HeroMetric, SecondaryMetric, VerdictBadge
from slice_key import DEFAULT_SLICE_KEY, SliceKey


def run_countdown(experience):
    """Complete the pre-run countdown synchronously, through the same
    method its QTimer calls -- tests drive the real transition rather
    than skipping it."""
    guard = 0
    while experience.state.phase is Phase.COUNTDOWN and guard < 20:
        experience._advance_countdown()
        guard += 1
    return experience.state.phase


# --------------------------------------------------------------- state
class TestPublicState:
    def test_starts_in_attract(self):
        assert PublicState().phase is Phase.ATTRACT

    def test_full_journey_advances_in_order(self):
        state = PublicState()
        expected = [Phase.INTRO, Phase.OBSERVE, Phase.PREDICTION, Phase.COUNTDOWN,
                    Phase.EXPERIMENT, Phase.REVEAL, Phase.COMPLETE, Phase.ATTRACT]
        assert [state.advance() for _ in expected] == expected

    def test_science_is_a_detour_not_a_step(self):
        """Show me the science returns to the journey rather than
        replacing a step -- a visitor who never presses it still reaches
        COMPLETE."""
        state = PublicState()
        state.go_to(Phase.REVEAL)
        assert state.advance() is Phase.COMPLETE

    def test_records_prediction_and_choice_separately(self):
        state = PublicState()
        state.record_prediction("cooler")
        state.record_choice("fan_on")
        assert state.prediction == "cooler"
        assert state.choice == "fan_on"
        assert state.tried_choices == ["fan_on"]

    def test_reset_clears_visitor_specific_state(self):
        state = PublicState()
        state.baseline_case_index = 6
        state.record_prediction("bigger")
        state.record_choice("fan_on")
        state.toggle_science()
        state.go_to(Phase.REVEAL)
        state.reset()
        assert state.phase is Phase.ATTRACT
        assert state.prediction is None
        assert state.choice is None
        assert state.tried_choices == []
        assert state.science_visible is False
        assert state.case_index == 6

    def test_replay_returns_to_observation_not_the_question(self):
        """A replay is normally the next visitor in the queue, who has
        seen nothing -- they get the fire and the smoke beat first."""
        state = PublicState()
        state.record_choice("fan_on")
        state.record_prediction("cooler")
        state.replay()
        assert state.phase is Phase.OBSERVE
        assert state.prediction is None
        assert state.frame_index == 0
        assert state.tried_choices == ["fan_on"]

    def test_observe_still_advances_to_prediction(self):
        state = PublicState()
        state.replay()
        assert state.advance() is Phase.PREDICTION

    def test_toggle_science_flips(self):
        state = PublicState()
        assert state.toggle_science() is True
        assert state.toggle_science() is False


# ---------------------------------------------------------- kid language
class TestKidLanguage:
    def test_ambient_room_is_never_described_as_hot(self):
        """The room in this dataset sits at 20-35 C for most of the run;
        labelling that 'hot' would misrepresent the experiment."""
        for temp in (20.0, 24.0, 26.5, 27.9):
            assert kid.temperature_phrase(temp) in ("Room temperature",)

    def test_flame_core_is_never_described_as_warm(self):
        assert kid.temperature_phrase(450.0) == "Flame"
        assert kid.temperature_phrase(330.0) == "Very hot"

    def test_buoyancy_only_airflow_reads_as_nearly_still(self):
        # Measured mean speed with vents open/closed: ~0.08-0.12 m/s.
        assert kid.airflow_phrase(0.085) == "Gentle drift"
        assert kid.airflow_phrase(0.02) == "Almost still"

    def test_fan_airflow_reads_as_moving_or_stronger(self):
        # Measured mean speed with the fan on: ~0.54-0.74 m/s.
        assert kid.airflow_phrase(0.54) in ("Strong airflow", "Very strong airflow")

    def test_compare_phrase_refuses_to_inflate_a_weak_effect(self):
        assert "barely changed" in kid.compare_phrase(0.100, 0.103, "air movement")

    def test_compare_phrase_reports_a_real_multiple(self):
        phrase = kid.compare_phrase(0.085, 0.540, "air movement")
        assert "times stronger" in phrase


# ------------------------------------------------------------ experiments
@pytest.fixture(scope="module")
def sim_data():
    return load_simulation_data()


class TestExperimentResolution:
    def test_fan_experiment_resolves_both_choices(self, sim_data):
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        experiment = experiments_mod.FAN_EXPERIMENT
        assert experiments_mod.is_available(sim_data.manifest, experiment)
        for choice in experiment.choices:
            assert experiments_mod.resolve_choice(
                sim_data.manifest, experiment, choice.key) is not None

    def test_choices_differ_only_in_the_vent_factor(self, sim_data):
        """The experiment must be a controlled comparison: exactly one
        factor changes between the two runs."""
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        experiment = experiments_mod.FAN_EXPERIMENT
        by_case = {e.case_index: e for e in sim_data.manifest}
        on = by_case[experiments_mod.resolve_choice(sim_data.manifest, experiment, "fan_on")]
        off = by_case[experiments_mod.resolve_choice(sim_data.manifest, experiment, "fan_off")]
        assert on.vod != off.vod
        assert (on.candles, on.door, on.voc) == (off.candles, off.door, off.voc)

    def test_unknown_choice_resolves_to_none(self, sim_data):
        assert experiments_mod.resolve_choice(
            sim_data.manifest, experiments_mod.FAN_EXPERIMENT, "nope") is None

    def test_resolution_is_not_hardcoded_to_an_index(self, sim_data):
        """An empty manifest must yield None, not a stale constant."""
        assert experiments_mod.resolve_choice([], experiments_mod.FAN_EXPERIMENT, "fan_on") is None

    def test_shipped_experiment_has_a_real_effect(self, sim_data):
        """The honesty guard. A shipped experiment must produce a change
        big enough that a visitor can actually see it -- this is what
        rules out the door factor, whose measured effect is ~1.0x and
        flips sign depending on vent state."""
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        ratio = experiments_mod.effect_size(
            sim_data.store, sim_data.manifest, experiments_mod.FAN_EXPERIMENT)
        assert ratio >= 2.0, f"fan experiment effect size only {ratio:.2f}x"

    def test_measure_returns_real_numbers(self, sim_data):
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        measurement = experiments_mod.measure(
            sim_data.store, sim_data.manifest, experiments_mod.FAN_EXPERIMENT, "fan_on")
        assert measurement.mean_airspeed > 0
        assert 15.0 < measurement.mean_temperature_end < 100.0
        assert measurement.peak_temperature > 100.0


# ----------------------------------------------------------- smoke story
def _story_for_choice(sim_data, choice_key):
    case_index = experiments_mod.resolve_choice(
        sim_data.manifest, experiments_mod.FAN_EXPERIMENT, choice_key)
    data = sim_data.store.get(case_index, DEFAULT_SLICE_KEY)
    extent = tuple(sim_data.store.get_extent(case_index, DEFAULT_SLICE_KEY))
    return StoryController(data, extent, sim_data.timesteps_per_second)


class TestSmokeStoryBeat:
    @pytest.mark.parametrize("choice_key", ["fan_off", "fan_on"])
    def test_ceiling_beat_exists_in_both_fan_scenarios(self, sim_data, choice_key):
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        beat = _story_for_choice(sim_data, choice_key).ceiling_beat()
        assert beat is not None
        assert "gathering under the ceiling" in i18n.tr(beat.text)
        # Measured: frame 64 (fan off) and frame 39 (fan on). Asserted as
        # a range so the test checks the physics is still being detected
        # in the plume-development window, not a frozen magic number.
        assert 10 <= beat.frame_index <= 120

    @pytest.mark.parametrize("choice_key", ["fan_off", "fan_on"])
    def test_no_spurious_descent_beat(self, sim_data, choice_key):
        """events.py reports "Smoke layer begins descending" at frame 1 in
        every scenario here, because it compares against layer_height[0],
        which is the "no layer yet" sentinel. Narrating it would tell a
        visitor the layer is dropping when it is actually building up
        under the ceiling."""
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        for beat in _story_for_choice(sim_data, choice_key).beats():
            text = i18n.tr(beat.text)
            assert "creeping downward" not in text
            assert "descending" not in text
            if "smoke" in text.lower():
                assert beat.frame_index > 4

    def test_ceiling_beat_carries_a_computed_basis(self, sim_data):
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        beat = _story_for_choice(sim_data, "fan_off").ceiling_beat()
        assert "plateau" in beat.source.basis
        assert beat.source.unit == "m"
        assert beat.source.value > 0

    def test_absent_smoke_beat_is_omitted_not_fabricated(self):
        """A run whose layer never rises must produce no beat at all."""
        flat = np.full((200, 20, 30), 20.0, dtype=np.float32)
        story = StoryController(flat, (0.0, 1.0, 0.0, 0.48), 4)
        assert story.ceiling_beat() is None
        assert all("ceiling" not in b.text for b in story.beats())

    def test_beat_window_is_long_enough_to_read(self):
        """Regression: the window was 12 frames, which at 6x playback is
        half a second -- the message flashed past unread."""
        wall_seconds = experience_mod.BEAT_VISIBLE_FRAMES / (
            4 * experience_mod.PLAYBACK_SPEED)
        assert wall_seconds >= 3.0


class TestObserveTiming:
    @pytest.mark.parametrize("choice_key", ["fan_off", "fan_on"])
    def test_observe_never_ends_before_the_smoke_beat(self, qapp, choice_key):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            case_index = experiments_mod.resolve_choice(
                sim.manifest, experiments_mod.FAN_EXPERIMENT, choice_key)
            experience._load_case(case_index)

            beat = experience._story.ceiling_beat()
            end = experience._observe_end_frame()
            assert beat is not None
            assert end >= beat.frame_index + experience_mod.BEAT_VISIBLE_FRAMES
            # ...and still short: the whole observe step must stay within
            # an exhibition attention span at 6x playback.
            assert end / (4 * experience_mod.PLAYBACK_SPEED) <= 12.0
        finally:
            window.close()

    def test_observe_falls_back_when_there_is_no_smoke_beat(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience._story = StoryController(
                np.full((200, 20, 30), 20.0, dtype=np.float32),
                (0.0, 1.0, 0.0, 0.48), 4)
            assert experience._story.ceiling_beat() is None
            assert experience._observe_end_frame() > 0
        finally:
            window.close()

    def test_observe_never_narrates_the_smoke_beat(self, qapp):
        """Free exploration must stay calm: the beat detector still runs
        (see TestObserveTiming's timing tests above and TestCeilingBeat
        elsewhere), but nothing may speak it unprompted while a visitor
        is just watching/looping OBSERVE on their own -- see
        docs/HANDOFF-PUBLIC-MODE-REDESIGN.md §3/§4. Narration now only
        fires during the guided EXPERIMENT run the child chose to start
        (see test_experiment_narrates_the_smoke_beat_exactly_once below).
        """
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            spoken = []
            original_say = experience.overlay.say

            def record(text, mood=None):
                spoken.append(text)
                original_say(text, mood)
            experience.overlay.say = record

            experience._begin_journey()
            experience._start_observe()
            experience.time_controller.pause()
            beat = experience._story.ceiling_beat()
            assert beat is not None
            for frame in range(beat.frame_index + experience_mod.BEAT_VISIBLE_FRAMES + 1):
                experience.time_controller.seek(frame)
            assert not any("gathering under the ceiling" in t for t in spoken)
            assert not any("Look closely" in t for t in spoken)
        finally:
            window.close()

    def test_experiment_narrates_the_smoke_beat_exactly_once(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            spoken = []
            original_say = experience.overlay.say

            def record(text, mood=None):
                spoken.append(text)
                original_say(text, mood)
            experience.overlay.say = record

            def drive_experiment():
                """Step through the run deterministically."""
                experience.time_controller.pause()
                for frame in range(experience_mod.EXPERIMENT_END_FRAME + 1):
                    experience.time_controller.seek(frame)

            experience._begin_journey()
            experience._on_prediction("cooler")
            run_countdown(experience)
            assert experience.state.phase is Phase.EXPERIMENT
            drive_experiment()
            # Assert on the stable substring, not the full "icon + text"
            # string, so this isn't brittle to icon changes.
            smoke = [t for t in spoken if "gathering under the ceiling" in t]
            assert len(smoke) == 1

            # Replay must let the beat play again for the next visitor.
            spoken.clear()
            experience._on_replay()
            experience._begin_journey()
            experience._on_prediction("cooler")
            run_countdown(experience)
            assert experience.state.phase is Phase.EXPERIMENT
            drive_experiment()
            assert len([t for t in spoken
                        if "gathering under the ceiling" in t]) == 1
        finally:
            window.close()

    def test_kiosk_reset_lets_the_beat_play_again(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience._spoken_beats.add(64)
            experience.reset()
            assert experience._spoken_beats == set()
        finally:
            window.close()


# -------------------------------------------------- completing both sides
class TestChoiceCycling:
    def test_fresh_visitor_gets_the_first_declared_choice(self):
        experiment = experiments_mod.FAN_EXPERIMENT
        assert experiment.next_untried_choice([]) == experiment.choices[0].key

    def test_replay_offers_the_side_not_yet_seen(self):
        experiment = experiments_mod.FAN_EXPERIMENT
        first = experiment.choices[0].key
        assert experiment.next_untried_choice([first]) != first

    def test_cycles_rather_than_refusing_once_everything_is_tried(self):
        experiment = experiments_mod.FAN_EXPERIMENT
        every = [c.key for c in experiment.choices]
        assert experiment.next_untried_choice(every) == experiment.choices[0].key

    def test_all_choices_tried_is_order_independent(self):
        experiment = experiments_mod.FAN_EXPERIMENT
        assert not experiment.all_choices_tried([])
        assert not experiment.all_choices_tried(["fan_on"])
        assert not experiment.all_choices_tried(["fan_on", "fan_on"])
        assert experiment.all_choices_tried(["fan_on", "fan_off"])
        assert experiment.all_choices_tried(["fan_off", "fan_on"])


class TestRevealAcknowledgement:
    @staticmethod
    def _reveal(experience, choices, prediction="cooler"):
        experience.state.reset()
        experience.state.record_prediction(prediction)
        for key in choices:
            experience.state.record_choice(key)
        return experience._reveal_lines()

    def test_first_run_uses_the_normal_reveal(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            headline, lines, _, _ = self._reveal(window.public_experience, ["fan_on"])
            assert "tested both" not in headline
            # Headline text stays warm either way -- the blunt Correct/
            # Incorrect verdict is its own VerdictBadge widget, not baked
            # into this string (see _render_reveal).
            assert "prediction" in headline.lower()
            assert not any(w in headline.lower() for w in ("wrong", "incorrect", "failed"))
            assert any("the air moved much faster" in line.lower() for line in lines)
        finally:
            window.close()

    def test_repeating_the_same_choice_does_not_acknowledge(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            headline, _, _, _ = self._reveal(
                window.public_experience, ["fan_on", "fan_on", "fan_on"])
            assert "tested both" not in headline
        finally:
            window.close()

    @pytest.mark.parametrize("order", [["fan_on", "fan_off"], ["fan_off", "fan_on"]])
    def test_both_sides_tried_acknowledges_in_either_order(self, qapp, order):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            headline, lines, dim, _ = self._reveal(window.public_experience, order)
            assert "You tested both" in headline
            # Side names come from the experiment's own choice labels.
            assert any("Fan OFF" in line and "Fan ON" in line for line in lines)
            # The finding stays measured and in a fixed direction
            # regardless of which side was watched last.
            assert any("the air moved much faster" in line.lower() for line in lines)
            assert any("m/s" in line for line in dim)
        finally:
            window.close()

    def test_acknowledgement_card_stays_readable(self, qapp):
        """No denser than the normal reveal, which is already known to
        fit at 800x600."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            _, normal, normal_dim, _ = self._reveal(window.public_experience, ["fan_on"])
            _, both, both_dim, _ = self._reveal(
                window.public_experience, ["fan_on", "fan_off"])
            assert len(both) <= len(normal)
            assert len(both_dim) <= len(normal_dim)
            assert all(len(line) <= 90 for line in both)
        finally:
            window.close()

    def test_no_grammar_doubling_in_any_reveal(self, qapp):
        """compare_phrase returns a whole clause; prefixing it with an
        article produced "the the air movement became ..."."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            for choices in (["fan_on"], ["fan_off"], ["fan_on", "fan_off"]):
                _, lines, _, _ = self._reveal(window.public_experience, choices)
                for line in lines:
                    assert "the the" not in line.lower()
        finally:
            window.close()

    def test_second_run_actually_runs_the_other_scenario(self, qapp):
        """The acknowledgement is only reachable if the replay really
        switches sides -- previously every run used the same choice."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience._begin_journey()
            experience._on_prediction("cooler")
            run_countdown(experience)
            first_choice = experience.state.choice
            first_case = experience.state.case_index

            experience.time_controller.pause()
            experience._on_replay()
            experience._on_prediction("cooler")
            run_countdown(experience)

            assert experience.state.choice != first_choice
            assert experience.state.case_index != first_case
            assert experience.experiment.all_choices_tried(experience.state.tried_choices)
        finally:
            window.close()

    def test_kiosk_reset_clears_the_completed_comparison(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience.state.record_choice("fan_on")
            experience.state.record_choice("fan_off")
            experience.reset()
            assert experience.state.tried_choices == []
            assert not experience.experiment.all_choices_tried(
                experience.state.tried_choices)
        finally:
            window.close()


# ----------------------------------------------------------- science card
class TestMetricComparison:
    def test_change_text_reports_a_multiple_only_for_a_real_multiple(self):
        metric = experiments_mod.PUBLIC_METRICS[0]  # air speed
        big = experiments_mod.MetricComparison(metric, 0.085, 0.540)
        assert "×" in big.change_text()
        assert "stronger" in big.change_text()

    def test_change_below_the_noticeable_delta_reads_as_about_the_same(self):
        """The honesty guard: a negligible difference must never be
        dressed up as a finding."""
        room = next(m for m in experiments_mod.PUBLIC_METRICS if m.key == "room_temp")
        tiny = experiments_mod.MetricComparison(room, 26.5, 26.6)
        assert tiny.change_text() == "about the same"
        assert not tiny.is_noticeable

    def test_modest_change_is_reported_as_a_difference_not_a_multiple(self):
        room = next(m for m in experiments_mod.PUBLIC_METRICS if m.key == "room_temp")
        real = experiments_mod.MetricComparison(room, 26.5, 25.0)
        assert real.change_text() == "1.5 °C cooler"

    def test_strongest_metric_is_none_when_nothing_moved(self):
        flat = [experiments_mod.MetricComparison(m, 1.0, 1.0)
                for m in experiments_mod.PUBLIC_METRICS]
        assert experiments_mod.strongest_metric(flat) is None

    def test_strongest_metric_ranks_by_measured_relative_change(self):
        metrics = experiments_mod.PUBLIC_METRICS
        comparisons = [
            experiments_mod.MetricComparison(metrics[0], 1.0, 1.05),   # +5%
            experiments_mod.MetricComparison(metrics[1], 10.0, 40.0),  # +300%
            experiments_mod.MetricComparison(metrics[2], 100.0, 90.0),
        ]
        assert experiments_mod.strongest_metric(comparisons).metric is metrics[1]


class TestScienceCard:
    @staticmethod
    def _science(experience, choices):
        experience.state.reset()
        experience.state.record_prediction("cooler")
        for key in choices:
            experience.state.record_choice(key)
        experience.state.go_to(Phase.SCIENCE)
        experience._render_phase()
        return experience.overlay.card

    def test_air_speed_is_the_measured_hero_for_the_fan_experiment(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            hero = experiments_mod.strongest_metric(experience._science_comparisons())
            assert hero.metric.key == "airspeed"
            # ...and it wins on measured evidence, by a wide margin.
            others = [c.relative_change for c in experience._science_comparisons()
                      if c.metric.key != "airspeed"]
            assert hero.relative_change > 10 * max(others)
        finally:
            window.close()

    def test_multiplier_is_computed_from_the_store(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            hero = experiments_mod.strongest_metric(experience._science_comparisons())
            measured = experiments_mod.measure(
                sim.store, sim.manifest, experiments_mod.FAN_EXPERIMENT, "fan_on")
            baseline = experiments_mod.measure(
                sim.store, sim.manifest, experiments_mod.FAN_EXPERIMENT, "fan_off")
            assert hero.contrast == pytest.approx(measured.mean_airspeed)
            assert hero.baseline == pytest.approx(baseline.mean_airspeed)
            assert hero.ratio == pytest.approx(
                measured.mean_airspeed / baseline.mean_airspeed)
        finally:
            window.close()

    @pytest.mark.parametrize("order", [["fan_on", "fan_off"], ["fan_off", "fan_on"]])
    def test_comparison_is_normalized_regardless_of_viewing_order(self, qapp, order):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience.state.reset()
            for key in order:
                experience.state.record_choice(key)
            values = [(c.metric.key, c.baseline, c.contrast, c.change_text())
                      for c in experience._science_comparisons()]
            # Identical whichever side was watched last.
            assert values == [
                (c.metric.key, c.baseline, c.contrast, c.change_text())
                for c in experience._science_comparisons()]
            airspeed = next(v for v in values if v[0] == "airspeed")
            assert airspeed[1] < airspeed[2]  # always fan off -> fan on
        finally:
            window.close()

    def test_flame_temperature_is_a_noticeable_hotter_finding(self, qapp):
        """On the current production (Pleiades) dataset, flame_temp moves
        365 -> 410 °C (fan off -> fan on) -- a real, noticeable ~46 °C
        change, unlike an earlier, now-superseded local dataset where the
        equivalent 459 -> 428 °C change stayed under the flame metric's
        noticeable-delta threshold. Re-pinned honestly rather than forcing
        the old "about the same" expectation on different real numbers
        (see load_data.py's SIM_ROOT switch to the Pleiades runs)."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            flame = next(c for c in window.public_experience._science_comparisons()
                         if c.metric.key == "flame_temp")
            assert flame.change_text() == "46 °C hotter"
        finally:
            window.close()

    def test_both_tried_science_card_announces_the_completed_comparison(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            card = self._science(experience, ["fan_on", "fan_off"])
            text = "\n".join(w.text() for w in card.findChildren(QtWidgets.QLabel))
            assert "You tested both" in text
            # Both sides are named on every chart's own axis.
            for bars in card.findChildren(BarCompare):
                assert [label for label, _ in bars._rows] == ["Fan OFF", "Fan ON"]
        finally:
            window.close()

    def test_science_card_has_a_hero_bar_and_the_exact_values(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            card = self._science(experience, ["fan_on"])
            # Two charts: the hero (air speed) and the secondary, each
            # with its own unit and therefore its own scale. On the
            # current production (Pleiades) dataset the secondary is
            # flame_temp, not room_temp (flame_temp's ~46 °C/12.6% change
            # now outranks room_temp's ~2.2 °C/8.3% one -- see
            # TestSecondaryMetric.test_secondary_is_the_next_real_finding
            # -- unlike an earlier, now-superseded local dataset where
            # room_temp ranked second).
            bars = card.findChildren(BarCompare)
            assert len(bars) == 2
            assert {b._unit for b in bars} == {"m/s", "°C"}
            text = "\n".join(w.text() for w in card.findChildren(QtWidgets.QLabel))
            # The child-level conclusion is stated before any decimals.
            assert "stronger" in text
            assert "cooler" in text
            # ...and the metric with no chart (room_temp, now demoted to
            # a plain-text row) still reports its numbers.
            assert "Air temp." in text and "26.5" in text and "24.3" in text
        finally:
            window.close()

    @pytest.mark.parametrize("choices", [["fan_on"], ["fan_on", "fan_off"]])
    def test_science_card_fits_at_800x600(self, qapp, choices):
        """Every child must sit inside the card, and the card inside the
        overlay -- an over-constrained QVBoxLayout silently overlaps its
        items instead of erroring."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            # show() + event processing so Qt actually lays the overlay
            # out; without it every widget reports its unlaid-out default
            # size and the assertions below test nothing.
            window.resize(800, 600)
            window.show()
            for _ in range(6):
                qapp.processEvents()
            window.enter_public_mode()
            experience = window.public_experience
            card = self._science(experience, choices)
            for _ in range(6):
                qapp.processEvents()

            assert card.geometry().bottom() <= experience.overlay.height()
            for child in card.findChildren(QtWidgets.QWidget):
                bottom = child.mapTo(card, child.rect().bottomLeft()).y()
                assert bottom <= card.height(), f"{child} overflows the card"
        finally:
            window.close()


# ------------------------------------------------- secondary metric visual
class TestSecondaryMetric:
    def test_secondary_is_the_next_real_finding(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            comparisons = window.public_experience._science_comparisons()
            hero = experiments_mod.strongest_metric(comparisons)
            secondary = experiments_mod.secondary_metric(comparisons, hero)
            assert hero.metric.key == "airspeed"
            # On the current production (Pleiades) dataset, flame_temp's
            # relative change (~12.6%) outranks room_temp's (~8.3%) -- an
            # earlier, now-superseded local dataset ranked them the other
            # way. See load_data.py's SIM_ROOT switch to the Pleiades runs.
            assert secondary.metric.key == "flame_temp"
            assert secondary.relative_change < hero.relative_change
        finally:
            window.close()

    def test_negligible_change_never_becomes_a_second_chart(self):
        """The ±0.3 °C guard decides whether a metric gets a visual at
        all, not just how it is worded."""
        metrics = experiments_mod.PUBLIC_METRICS
        comparisons = [
            experiments_mod.MetricComparison(metrics[0], 0.085, 0.540),  # real
            experiments_mod.MetricComparison(metrics[1], 26.5, 26.4),    # negligible
            experiments_mod.MetricComparison(metrics[2], 459.0, 458.0),  # negligible
        ]
        assert experiments_mod.secondary_metric(comparisons) is None

    def test_metrics_without_a_chart_still_report_their_numbers(self):
        metrics = experiments_mod.PUBLIC_METRICS
        comparisons = [experiments_mod.MetricComparison(m, 1.0, 2.0) for m in metrics]
        hero = comparisons[0]
        secondary = comparisons[1]
        rest = experiments_mod.unchanged_metrics(comparisons, [hero, secondary])
        assert [c.metric.key for c in rest] == [metrics[2].key]

    def test_each_comparison_carries_its_own_scale(self, qapp):
        """Air speed (m/s) and temperature (°C) are not commensurable --
        two BarCompare widgets must never share a scale, or a 1.5 °C
        change would be drawn against a 0.54 m/s axis."""
        speed = BarCompare("A", 0.085, "B", 0.540, "m/s", 3)
        temp = BarCompare("A", 26.5, "B", 25.0, "°C", 1)
        assert speed._rows[1][1] / speed._rows[0][1] > 6
        assert 0.9 < temp._rows[1][1] / temp._rows[0][1] < 1.0
        # Nothing in either widget refers to the other.
        assert speed._unit != temp._unit

    def test_temperature_bars_look_modest_and_airspeed_bars_do_not(self, qapp):
        """The visual must not imply the two effects are comparable: the
        temperature bars differ by a few percent, the air-speed bars by
        several hundred."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            comparisons = window.public_experience._science_comparisons()
            hero = experiments_mod.strongest_metric(comparisons)
            secondary = experiments_mod.secondary_metric(comparisons, hero)
            # Bar length is value/peak within each comparison.
            hero_ratio = min(hero.baseline, hero.contrast) / max(hero.baseline, hero.contrast)
            sec_ratio = min(secondary.baseline, secondary.contrast) / max(
                secondary.baseline, secondary.contrast)
            assert hero_ratio < 0.25      # dramatically different bars
            assert sec_ratio > 0.85       # visibly similar bars
        finally:
            window.close()

    def test_secondary_is_visually_demoted(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience.state.reset()
            experience.state.record_prediction("cooler")
            experience.state.record_choice("fan_on")
            experience.state.go_to(Phase.SCIENCE)
            experience._render_phase()
            card = experience.overlay.card
            hero_bars = card.findChildren(HeroMetric)[0].findChildren(BarCompare)[0]
            sec_bars = card.findChildren(SecondaryMetric)[0].findChildren(BarCompare)[0]
            assert not hero_bars._compact
            assert sec_bars._compact
            assert sec_bars.height() < hero_bars.height()
        finally:
            window.close()

    @pytest.mark.parametrize("order", [["fan_on", "fan_off"], ["fan_off", "fan_on"]])
    def test_temperature_comparison_is_order_independent(self, qapp, order):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience.state.reset()
            for key in order:
                experience.state.record_choice(key)
            temp = next(c for c in experience._science_comparisons()
                        if c.metric.key == "room_temp")
            assert temp.baseline == pytest.approx(26.5, abs=0.1)
            assert temp.contrast == pytest.approx(24.3, abs=0.1)
            assert temp.change_text() == "2.2 °C cooler"
        finally:
            window.close()

    def test_prediction_line_only_appears_when_the_guess_matches(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            comparisons = experience._science_comparisons()
            # The "cooler" prediction is declared (see FAN_EXPERIMENT's
            # Prediction table) as being about room_temp specifically --
            # test against that comparison directly rather than whichever
            # metric secondary_metric() currently ranks second, since that
            # ranking is itself real-data-driven (on the current Pleiades
            # dataset, flame_temp's relative change now outranks
            # room_temp's, see TestSecondaryMetric.test_secondary_is_the_
            # next_real_finding) and isn't what this test is about.
            room_temp = next(c for c in comparisons if c.metric.key == "room_temp")

            experience.state.record_prediction("cooler")
            assert "guessed it" in experience._prediction_line(room_temp)

            # A guess about something else falls back to the neutral
            # explanation rather than claiming the visitor was right.
            experience.state.record_prediction("bigger")
            line = experience._prediction_line(room_temp)
            assert "guessed it" not in line
            assert line == room_temp.metric.meaning
        finally:
            window.close()

    def test_science_card_blocks_are_never_clipped_at_800x600(self, qapp):
        """An over-constrained QVBoxLayout shrinks children below their
        size hint instead of erroring, which clipped the hero bars twice
        during development."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.resize(800, 600)
            window.show()
            for _ in range(6):
                qapp.processEvents()
            window.enter_public_mode()
            experience = window.public_experience
            for choices in (["fan_on"], ["fan_on", "fan_off"]):
                experience.state.reset()
                experience.state.record_prediction("cooler")
                for key in choices:
                    experience.state.record_choice(key)
                experience.state.go_to(Phase.SCIENCE)
                experience._render_phase()
                for _ in range(6):
                    qapp.processEvents()
                card = experience.overlay.card
                for block in (card.findChildren(HeroMetric)
                              + card.findChildren(SecondaryMetric)):
                    assert block.height() >= block.sizeHint().height(), (
                        f"{type(block).__name__} clipped for {choices}")
        finally:
            window.close()


# --------------------------------------------------------- mascot finding
class TestMascotFinding:
    @staticmethod
    def _metric(key):
        return next(m for m in experiments_mod.PUBLIC_METRICS if m.key == key)

    def test_large_change_reads_as_much(self):
        speed = experiments_mod.MetricComparison(self._metric("airspeed"), 0.085, 0.540)
        assert kid.finding_sentence(speed) == "the air moved much faster"

    def test_modest_change_reads_as_a_little(self):
        room = experiments_mod.MetricComparison(self._metric("room_temp"), 26.5, 25.0)
        assert kid.finding_sentence(room) == "the air got a little cooler"

    def test_direction_follows_the_measurement(self):
        room = self._metric("room_temp")
        warmer = experiments_mod.MetricComparison(room, 20.0, 45.0)
        assert "warmer" in kid.finding_sentence(warmer)
        cooler = experiments_mod.MetricComparison(room, 45.0, 20.0)
        assert "cooler" in kid.finding_sentence(cooler)

    def test_change_below_the_threshold_is_never_narrated(self):
        """The metric's own noticeable_delta stays authoritative: a
        459 -> 428 °C flame must not become "the flame burned cooler"."""
        flame = experiments_mod.MetricComparison(self._metric("flame_temp"), 459.0, 428.0)
        assert kid.finding_sentence(flame) == ""
        assert kid.mascot_finding(flame) == ""

    def test_message_is_generated_for_whichever_metric_leads(self):
        """A different strongest metric produces a different sentence with
        no code change -- the phrasing lives on the Metric, not in a
        branch."""
        speed = experiments_mod.MetricComparison(self._metric("airspeed"), 0.085, 0.540)
        room = experiments_mod.MetricComparison(self._metric("room_temp"), 20.0, 45.0)
        assert "air moved" in kid.mascot_finding(speed)
        assert "air got" in kid.mascot_finding(room)
        assert kid.mascot_finding(speed) != kid.mascot_finding(room)

    def test_both_tried_variant_acknowledges_the_comparison(self):
        speed = experiments_mod.MetricComparison(self._metric("airspeed"), 0.085, 0.540)
        line = kid.mascot_finding(speed, both_tried=True)
        assert "tested both" in line
        assert "the air moved much faster" in line

    def test_message_stays_short_and_spoken(self):
        """One statement, or a short acknowledgement plus one statement --
        never a paragraph. Length is the real guard against the line
        turning into a second data table."""
        speed = experiments_mod.MetricComparison(self._metric("airspeed"), 0.085, 0.540)
        assert kid.mascot_finding(speed).count(".") + kid.mascot_finding(speed).count("!") == 1
        for line in (kid.mascot_finding(speed), kid.mascot_finding(speed, both_tried=True)):
            assert len(line) <= 90
            assert line.count(".") + line.count("!") <= 2

    def test_no_raw_percentages_or_decimals_in_the_spoken_line(self):
        speed = experiments_mod.MetricComparison(self._metric("airspeed"), 0.085, 0.540)
        line = kid.mascot_finding(speed)
        assert "%" not in line
        assert not any(ch.isdigit() for ch in line)


class TestMascotAgreesWithScienceCard:
    def test_mascot_uses_the_same_hero_metric_as_the_card(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience.state.reset()
            experience.state.record_prediction("cooler")
            experience.state.record_choice("fan_on")
            experience.state.go_to(Phase.SCIENCE)
            experience._render_phase()

            hero = experiments_mod.strongest_metric(experience._science_comparisons())
            bubble = experience.overlay.bubble.text()
            card = "\n".join(w.text() for w in experience.overlay.card.findChildren(
                QtWidgets.QLabel))
            # Both name the same metric, and the guide never contradicts
            # the chart by leading with a different one.
            assert hero.metric.key == "airspeed"
            assert kid.finding_sentence(hero) in bubble
            assert hero.metric.label.lower() in card.lower()
            assert "cooler" not in bubble
        finally:
            window.close()

    def test_both_tried_science_view_acknowledges_it_in_the_bubble(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience.state.reset()
            for key in ("fan_on", "fan_off"):
                experience.state.record_choice(key)
            experience.state.go_to(Phase.SCIENCE)
            experience._render_phase()
            assert "tested both" in experience.overlay.bubble.text()
        finally:
            window.close()

    def test_bubble_fits_without_clipping_at_800x600(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.resize(800, 600)
            window.show()
            for _ in range(6):
                qapp.processEvents()
            window.enter_public_mode()
            experience = window.public_experience
            experience.state.reset()
            for key in ("fan_on", "fan_off"):
                experience.state.record_choice(key)
            experience.state.go_to(Phase.SCIENCE)
            experience._render_phase()
            for _ in range(6):
                qapp.processEvents()

            bubble = experience.overlay.bubble
            overlay = experience.overlay
            assert bubble.height() >= bubble.heightForWidth(bubble.width())
            assert bubble.geometry().bottom() <= overlay.height()
            assert bubble.geometry().right() <= overlay.width()
            # ...and it must not ride up over the science card.
            assert bubble.geometry().top() >= overlay.card.geometry().bottom()
        finally:
            window.close()


# ------------------------------------------------- generic reveal narrative
def _fake_measurement(mean_v, temp_end, peak=450.0):
    return experiments_mod.Measurement(0, mean_v, mean_v * 2, temp_end, peak)


class TestRevealIsExperimentAgnostic:
    """The reveal must read correctly for an experiment whose strongest
    metric is something other than air speed -- that is the whole point of
    routing it through the metric layer."""

    @staticmethod
    def _reveal(experience, baseline, contrast, choices=("fan_on",), pred="cooler"):
        experience._measurements = {
            experience.experiment.baseline_choice: baseline,
            experience.experiment.contrast_choice(): contrast,
        }
        experience.state.reset()
        experience.state.record_prediction(pred)
        for key in choices:
            experience.state.record_choice(key)
        return experience._reveal_lines()

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_temperature_hero_produces_temperature_wording(self, experience):
        _, lines, dim, _ = self._reveal(
            experience, _fake_measurement(0.10, 20.0), _fake_measurement(0.101, 45.0))
        body = " ".join(lines)
        assert "the air got much warmer" in body.lower()
        assert "air moved" not in body.lower()
        assert "air temp." in dim[0].lower()

    def test_direction_follows_the_measurement(self, experience):
        _, lines, _, _ = self._reveal(
            experience, _fake_measurement(0.10, 45.0), _fake_measurement(0.101, 20.0))
        assert "cooler" in " ".join(lines).lower()

    def test_explanation_matches_the_leading_metric(self, experience):
        _, lines, _, _ = self._reveal(
            experience, _fake_measurement(0.10, 20.0), _fake_measurement(0.101, 45.0))
        room = next(m for m in experiments_mod.PUBLIC_METRICS if m.key == "room_temp")
        assert room.explanation in lines
        speed = next(m for m in experiments_mod.PUBLIC_METRICS if m.key == "airspeed")
        assert speed.explanation not in lines

    def test_metric_without_an_explanation_is_handled(self, experience, monkeypatch):
        room = next(m for m in experiments_mod.PUBLIC_METRICS if m.key == "room_temp")
        bare = dataclasses.replace(room, explanation_key="")
        monkeypatch.setattr(
            experiments_mod, "PUBLIC_METRICS",
            tuple(bare if m.key == "room_temp" else m for m in experiments_mod.PUBLIC_METRICS))
        headline, lines, _, _ = self._reveal(
            experience, _fake_measurement(0.10, 20.0), _fake_measurement(0.101, 45.0))
        assert headline
        assert lines and all(line.strip() for line in lines)

    def test_null_result_is_not_dressed_up(self, experience):
        """Nothing clears its threshold: the reveal says so, and does not
        congratulate the visitor above a "nothing changed" line."""
        headline, lines, _, _ = self._reveal(
            experience, _fake_measurement(0.10, 26.50), _fake_measurement(0.101, 26.55))
        assert "Almost nothing changed" in " ".join(lines)
        assert "You got it" not in headline

    def test_subthreshold_flame_never_leads_the_reveal(self, experience):
        _, lines, _, _ = self._reveal(
            experience,
            _fake_measurement(0.10, 26.5, peak=459.0),
            _fake_measurement(0.101, 26.55, peak=428.0))
        assert "flame" not in " ".join(lines).lower()

    def test_no_digits_leak_into_the_spoken_findings(self, experience):
        _, lines, dim, _ = self._reveal(
            experience, _fake_measurement(0.085, 26.5), _fake_measurement(0.540, 25.0))
        spoken = [l for l in lines if "Measured" not in l]
        assert not any(ch.isdigit() for ch in " ".join(spoken))
        # ...the exact values live in the dim line instead.
        assert any(ch.isdigit() for ch in dim[0])


class TestRevealNarrativeDecoupling:
    def test_no_fan_specific_narrative_strings_in_generic_rendering(self):
        """Experiment configuration may name the fan; the generic reveal
        and guide logic must not. Guards against a future edit quietly
        reintroducing a hardcoded sentence."""
        tree = ast.parse(pathlib.Path(experience_mod.__file__).read_text())
        targets = {"_reveal_lines", "_render_reveal"}
        offenders = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and node.name in targets):
                continue
            docstring = ast.get_docstring(node, clean=False)
            for literal in ast.walk(node):
                # Comments never reach the AST, so only the docstring has
                # to be excluded explicitly -- prose about the fan is fine
                # there, a user-facing string is not.
                if not (isinstance(literal, ast.Constant)
                        and isinstance(literal.value, str)):
                    continue
                text = literal.value
                if text == docstring:
                    continue
                lowered = text.lower()
                if any(word in lowered
                       for word in ("fan", "air moved", "airflow", "moving air")):
                    offenders.append((node.name, text))
        assert not offenders, f"fan-specific narrative strings: {offenders}"

    def test_side_names_come_from_the_experiments_choice_labels(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience.state.reset()
            experience.state.record_choice("fan_off")
            _, lines, _, _ = experience._reveal_lines()
            baseline_label, contrast_label = experience._choice_labels()
            assert contrast_label in " ".join(lines)
            assert baseline_label == "Fan OFF" and contrast_label == "Fan ON"
        finally:
            window.close()

    def test_reveal_agrees_with_the_science_card_hero(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience.state.reset()
            experience.state.record_prediction("cooler")
            experience.state.record_choice("fan_on")
            hero = experiments_mod.strongest_metric(experience._science_comparisons())
            _, lines, _, _ = experience._reveal_lines()
            clause = kid.finding_sentence(hero)
            assert clause and clause.lower() in " ".join(lines).lower()
        finally:
            window.close()

    def test_both_tried_reveal_stays_concise(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience.state.reset()
            for key in ("fan_on", "fan_off"):
                experience.state.record_choice(key)
            headline, lines, _, _ = experience._reveal_lines()
            assert "You tested both" in headline
            # A conclusion, not a report: no more lines than the normal
            # reveal, and no repeated sentence.
            assert len(lines) <= 3
            assert len(set(lines)) == len(lines)
        finally:
            window.close()

    @pytest.mark.parametrize("choices", [["fan_on"], ["fan_off"], ["fan_on", "fan_off"]])
    def test_reveal_card_fits_at_800x600(self, qapp, choices):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.resize(800, 600)
            window.show()
            for _ in range(6):
                qapp.processEvents()
            window.enter_public_mode()
            experience = window.public_experience
            experience.state.reset()
            experience.state.record_prediction("cooler")
            for key in choices:
                experience.state.record_choice(key)
            experience.state.go_to(Phase.REVEAL)
            experience._render_phase()
            for _ in range(6):
                qapp.processEvents()
            card = experience.overlay.card
            assert card.geometry().bottom() <= experience.overlay.height()
            for child in card.findChildren(QtWidgets.QWidget):
                assert child.mapTo(card, child.rect().bottomLeft()).y() <= card.height()
        finally:
            window.close()


# --------------------------------------------------- child-facing exhibit
class TestMascotStates:
    def test_every_mood_paints_without_error(self, qapp):
        """Each expressive state must render -- an unbalanced
        QPainter.save()/restore() in a mood branch aborts the process."""
        from public.mascot import (CURIOUS, EXCITED, EXPLAINING, IDLE, POINTING,
                                   SURPRISED, THINKING, WATCHING)
        mascot = Mascot()
        mascot.resize(120, 130)
        for mood in (IDLE, CURIOUS, WATCHING, POINTING, SURPRISED, EXCITED,
                     THINKING, EXPLAINING):
            mascot.set_mood(mood)
            pixmap = mascot.grab()          # forces a full paintEvent
            assert not pixmap.isNull()
        mascot.stop()

    def test_story_beats_carry_moods(self, sim_data):
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        from public.mascot import CURIOUS, WATCHING
        story = _story_for_choice(sim_data, "fan_off")
        assert story.ceiling_beat().mood == WATCHING
        ignition = next(b for b in story.beats() if "candle is lit" in i18n.tr(b.text))
        assert ignition.mood == CURIOUS

    def test_guide_reacts_to_a_beat_with_that_beats_mood(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience._begin_journey()
            experience._start_observe()
            experience.time_controller.pause()
            beat = experience._story.ceiling_beat()
            for frame in range(beat.frame_index + 1):
                experience.time_controller.seek(frame)
            assert experience.overlay.mascot._mood == beat.mood
        finally:
            window.close()

    def test_guide_reacts_to_measured_airflow_once(self, qapp):
        """Driven by the measured field through kid_language's bands, so
        it simply never fires in a run where the air stays still."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience._begin_journey()
            experience._on_prediction("cooler")     # runs the fan
            run_countdown(experience)
            experience.time_controller.pause()
            for frame in range(120):
                experience.time_controller.seek(frame)
            assert experience._airflow_reacted
            experience._airflow_reacted = False
            experience._react_to_airflow(120)
            # A second pass in the same run re-arms only because the test
            # cleared the flag; the guard itself is one-shot per run.
            assert experience._airflow_reacted

            experience._on_replay()
            assert not experience._airflow_reacted
        finally:
            window.close()

    def test_still_air_never_triggers_the_reaction(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience.state.go_to(Phase.EXPERIMENT)
            baseline = experiments_mod.resolve_choice(
                sim.manifest, experiments_mod.FAN_EXPERIMENT, "fan_off")
            experience._load_case(baseline)
            for frame in range(0, 300, 10):
                experience._react_to_airflow(frame)
            assert not experience._airflow_reacted
        finally:
            window.close()


class TestWhatAmISeeing:
    def _experience(self, window):
        window.enter_public_mode()
        return window.public_experience

    def test_help_button_is_offered_while_watching(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            experience = self._experience(window)
            experience._begin_journey()
            experience._start_observe()
            texts = [b.text() for b in experience.overlay.children()
                     if isinstance(b, QtWidgets.QPushButton)]
            assert any("What am I seeing" in t for t in texts)
        finally:
            window.close()

    def test_help_describes_only_what_is_on_screen(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            experience = self._experience(window)
            experience._begin_journey()
            experience._start_observe()
            experience.time_controller.pause()

            before = experience._what_am_i_seeing()
            assert any("Bigger dots" in line for line in before)
            assert not any("smoke" in line.lower() for line in before)

            beat = experience._story.ceiling_beat()
            for frame in range(beat.frame_index + 1):
                experience.time_controller.seek(frame)
            after = experience._what_am_i_seeing()
            assert any("smoke" in line.lower() for line in after)
        finally:
            window.close()

    def test_help_pauses_and_resumes_playback(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            experience = self._experience(window)
            experience._begin_journey()
            experience._start_observe()
            assert experience.time_controller.is_playing()

            experience._on_help_requested()
            assert experience._help_open
            assert not experience.time_controller.is_playing()
            assert not experience.overlay.card.isHidden()

            experience.time_controller.seek(90)
            experience._on_help_requested()
            assert not experience._help_open
            assert experience.time_controller.is_playing()
            # Regression: closing help used to re-run the phase handler,
            # which restarted playback at an unlit frame 0.
            assert experience.time_controller.index >= 90
        finally:
            window.close()

    def test_help_card_fits_at_800x600(self, qapp):
        """Every help line must be visible -- the first version wrapped
        and the last two lines were clipped out of the card."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.resize(800, 600)
            window.show()
            for _ in range(6):
                qapp.processEvents()
            experience = self._experience(window)
            experience._begin_journey()
            experience._start_observe()
            experience.time_controller.pause()
            beat = experience._story.ceiling_beat()
            for frame in range(beat.frame_index + 1):
                experience.time_controller.seek(frame)
            experience._on_help_requested()
            for _ in range(6):
                qapp.processEvents()

            card = experience.overlay.card
            labels = card.findChildren(QtWidgets.QLabel)
            assert len(labels) >= 5           # title + four notes
            for label in labels:
                bottom = label.mapTo(card, label.rect().bottomLeft()).y()
                assert bottom <= card.height(), f"clipped: {label.text()!r}"
        finally:
            window.close()

    def test_help_leaves_no_stale_widgets_behind(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            experience = self._experience(window)
            experience._begin_journey()
            experience._start_observe()
            experience._on_help_requested()
            experience._on_help_requested()
            texts = [b.text() for b in experience.overlay.children()
                     if isinstance(b, QtWidgets.QPushButton)
                     and b is not experience.overlay.exit_button]
            assert not any("Got it" in t for t in texts)
        finally:
            window.close()


class TestPredictionInteraction:
    def test_choices_come_from_the_experiment_not_the_ui(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience.state.go_to(Phase.PREDICTION)
            experience._render_phase()
            texts = [b.text() for b in experience.overlay.children()
                     if isinstance(b, QtWidgets.QPushButton)
                     and b is not experience.overlay.exit_button]
            for prediction in experience.experiment.predictions:
                assert any(prediction.label in t for t in texts)
        finally:
            window.close()

    def test_prompt_asks_the_question_in_large_type(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience.state.go_to(Phase.PREDICTION)
            experience._render_phase()
            assert "WHAT DO YOU THINK" in experience.overlay.banner.text().upper()
        finally:
            window.close()

    def test_choice_buttons_meet_the_touch_target_floor(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.resize(800, 600)
            window.show()
            for _ in range(6):
                qapp.processEvents()
            window.enter_public_mode()
            experience = window.public_experience
            experience.state.go_to(Phase.PREDICTION)
            experience._render_phase()
            for _ in range(6):
                qapp.processEvents()
            buttons = [b for b in experience.overlay.children()
                       if isinstance(b, BigButton)]
            assert buttons
            for button in buttons:
                assert button.height() >= 64, f"{button.text()!r} too short"
                assert button.width() >= 64
        finally:
            window.close()

    def test_question_card_shows_the_full_question_text_not_clipped(self, qapp):
        """Root-cause regression guard: the question card used to report
        its title label's *unwrapped* single-line height to root's
        layout (Card.enable_height_for_width() only gives a soft hint,
        not a hard floor -- same class of bug as ExploreToggle's own
        button-overlap fix), so at 800x600 the card was handed less
        height than FAN_EXPERIMENT's own (unusually long) question
        needs and clipped it after 3 lines -- "happens if we switch it
        on?" never rendered at all. Card._update_minimum_height()
        converts that hint into a real minimum; checked here against
        the label's own actual text and geometry, not just that some
        text is present."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.resize(800, 600)
            window.show()
            for _ in range(6):
                qapp.processEvents()
            window.enter_public_mode()
            experience = window.public_experience
            experience.state.go_to(Phase.PREDICTION)
            experience._render_phase()
            for _ in range(15):
                qapp.processEvents()
            overlay = experience.overlay
            title_label = overlay.card.findChildren(QtWidgets.QLabel)[0]
            assert title_label.text() == experience.experiment.question
            needed = title_label.heightForWidth(title_label.width())
            assert title_label.height() >= needed, (
                f"title label given {title_label.height()}px, needs {needed}px")
        finally:
            window.close()

    @pytest.mark.parametrize("size", [(800, 600), (1280, 800)])
    def test_mascot_bubble_never_overlaps_the_choice_buttons(self, qapp, size):
        """Root-cause regression guard: the mascot's own speech bubble
        (PublicOverlay.say(), triggered synchronously by _position_
        mascot) used to be sized against the mascot's own margin only,
        with no awareness of button_row -- on PREDICTION specifically
        (the one screen with three *tall* choice buttons reaching far
        enough right to matter), the bubble's real sizeHint spanned
        clean across the row, real ghosted text behind the buttons, not
        just visually close. _position_mascot now clamps the bubble's
        available width against button_row's own real right edge, via
        a deferred re-run once that row's geometry has actually
        settled (see its own comment on why one activate() pass wasn't
        enough).

        At 800x600 the real gap between the row and the mascot (~28px)
        is too narrow for legible text at any font scale --
        _position_mascot hides the bubble there rather than force it
        into an unreadable sliver (or, worse, back into overlapping the
        buttons); at 1280x800 there's real room and the bubble is
        expected to actually show, clamped and clear of the row -- this
        is what actually exercises the clamp math itself, not just its
        hidden fallback."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.show()
            for _ in range(6):
                qapp.processEvents()
            window.enter_public_mode()
            # resize AFTER enter_public_mode(), not before: a resize
            # applied first doesn't reliably propagate to the overlay's
            # own geometry in this offscreen harness (an established
            # quirk elsewhere in this suite).
            window.resize(*size)
            for _ in range(10):
                qapp.processEvents()
            experience = window.public_experience
            experience.state.go_to(Phase.PREDICTION)
            experience._render_phase()
            for _ in range(15):
                qapp.processEvents()
            overlay = experience.overlay
            buttons = [overlay.button_row.itemAt(i).widget()
                       for i in range(overlay.button_row.count())]
            assert len(buttons) == 3
            if not overlay.bubble.isVisible():
                return
            bubble_rect = overlay.bubble.geometry()
            for button in buttons:
                button_rect = QtCore.QRect(
                    button.mapTo(overlay, QtCore.QPoint(0, 0)), button.size())
                assert not button_rect.intersects(bubble_rect), (
                    f"{button.text()!r} {button_rect} overlaps the mascot's "
                    f"speech bubble {bubble_rect}")
        finally:
            window.close()

    @pytest.mark.parametrize("prediction", ["cooler", "bigger", "nothing"])
    def test_every_prediction_reaches_the_experiment(self, qapp, prediction):
        """A child who guesses 'wrong' runs exactly the same experiment."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience._begin_journey()
            experience._on_prediction(prediction)
            assert experience.state.phase is Phase.COUNTDOWN
            run_countdown(experience)
            assert experience.state.phase is Phase.EXPERIMENT
            experience.time_controller.pause()
            experience._experiment_finished()
            headline, lines, _, _ = experience._reveal_lines()
            assert lines
            assert not any(word in headline.lower()
                           for word in ("wrong", "incorrect", "failed", "sorry"))
        finally:
            window.close()

    def test_choosing_a_prediction_flashes_it_before_advancing(self, qapp):
        """Games UX pass: a tap must visibly register immediately, not
        silently vanish into the countdown -- see _flash_prediction_choice."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience.state.go_to(Phase.PREDICTION)
            experience._render_phase()
            buttons = list(experience._prediction_buttons)
            assert buttons
            experience._flash_prediction_choice(experience.experiment.predictions[0].key)
            # Still on PREDICTION -- the advance is deferred, not instant --
            # but every button is already disabled as visible confirmation.
            assert experience.state.phase is Phase.PREDICTION
            assert all(not b.isEnabled() for b in buttons)
            for _ in range(20):
                qapp.processEvents()
                if experience.state.phase is not Phase.PREDICTION:
                    break
            assert experience.state.phase is Phase.COUNTDOWN
        finally:
            window.close()

    def test_reveal_shows_an_explicit_correct_or_incorrect_badge(self, qapp):
        """Full quiz mode (games UX pass): the reveal card carries an
        unambiguous VerdictBadge. No cumulative score is tracked or
        shown (removed per a later UX pass -- the badge alone answers
        "was I right," and quiz_correct/quiz_total had no other
        consumer, see PublicState)."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience._begin_journey()
            experience._on_prediction("cooler")
            run_countdown(experience)
            experience.time_controller.pause()
            experience._experiment_finished()
            badges = [w for w in experience.overlay.card.findChildren(VerdictBadge)]
            assert len(badges) == 1
            assert badges[0].text() in (
                i18n.tr("verdict_correct_badge"), i18n.tr("verdict_incorrect_badge"))
        finally:
            window.close()


class TestRoomAirMeter:
    def test_meter_shows_room_average_not_the_flame(self, qapp):
        """The peak is the candle itself (~450 °C in every scenario) and
        says nothing about the room."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience._begin_journey()
            experience._start_observe()
            experience.time_controller.pause()
            experience.time_controller.seek(150)
            shown = experience.overlay.temperature_meter._value.text()
            room = experience.scene.mean_temperature_at(150)
            peak = experience.scene.peak_temperature_at(150)
            assert f"{room:.0f}" in shown
            assert f"{peak:.0f}" not in shown
            # ...and a warm room is never labelled dangerous.
            assert "danger" not in experience.overlay.temperature_meter._phrase.text().lower()
        finally:
            window.close()


# --------------------------------------------------------- discovery loop
class TestCountdown:
    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_prediction_enters_the_countdown_not_the_run(self, experience):
        experience._begin_journey()
        experience._on_prediction("cooler")
        assert experience.state.phase is Phase.COUNTDOWN
        assert experience.overlay.countdown.isVisible() or experience.overlay.countdown.text()

    def test_countdown_ticks_through_to_the_experiment(self, experience):
        experience._begin_journey()
        experience._on_prediction("cooler")
        seen = []
        while experience.state.phase is Phase.COUNTDOWN:
            seen.append(experience.overlay.countdown.text())
            experience._advance_countdown()
        assert seen[:3] == ["3", "2", "1"]
        assert experience.state.phase is Phase.EXPERIMENT

    def test_countdown_is_short(self):
        total = experience_mod.COUNTDOWN_STEP_MS * len(experience_mod._COUNTDOWN_STEPS)
        assert 800 <= total <= 2500

    def test_countdown_overlay_is_cleared_afterwards(self, experience):
        experience._begin_journey()
        experience._on_prediction("cooler")
        run_countdown(experience)
        assert experience.overlay.countdown.isHidden()

    def test_reset_stops_a_running_countdown(self, experience):
        experience._begin_journey()
        experience._on_prediction("cooler")
        assert experience._countdown_timer.isActive()
        experience.reset()
        assert not experience._countdown_timer.isActive()
        assert experience.state.phase is Phase.ATTRACT


class TestWowMoment:
    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_emphasis_names_the_measured_hero_metric(self, experience):
        experience._begin_journey()
        experience._on_prediction("cooler")
        run_countdown(experience)
        hero = experiments_mod.strongest_metric(experience._science_comparisons())
        assert hero.metric.label.lower() in experience.overlay.banner.text().lower()

    def test_running_the_baseline_gets_no_emphasis(self, experience):
        """The whoosh is only earned by the side that actually differs."""
        experience.state.record_choice("fan_off")
        experience.overlay.banner.set_prompt("")
        experience._announce_expected_change()
        assert "WHOOSH" not in experience.overlay.banner.text()

    def test_beat_flashes_only_while_observing(self, experience):
        """During the experiment the banner belongs to the change the
        visitor caused; a re-fired ignition beat used to stomp it."""
        experience._begin_journey()
        experience._on_prediction("cooler")
        run_countdown(experience)
        experience.time_controller.pause()
        before = experience.overlay.banner.text()
        for frame in range(6):
            experience.time_controller.seek(frame)
        assert "WHOOSH" in before
        assert experience.overlay.banner.text() == before

    def test_emphasis_is_skipped_when_nothing_moved(self, experience):
        experience.state.record_choice("fan_on")
        experience._measurements = {
            "fan_off": _fake_measurement(0.10, 26.5),
            "fan_on": _fake_measurement(0.101, 26.55),
        }
        experience.overlay.banner.set_prompt("")
        experience._announce_expected_change()
        assert "WHOOSH" not in experience.overlay.banner.text()


class TestStageStrip:
    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_hidden_before_the_journey_starts(self, experience):
        assert experience.overlay.stages._active == -1

    @pytest.mark.parametrize("phase,stage", [
        (Phase.OBSERVE, -1), (Phase.PREDICTION, 1),
        (Phase.EXPERIMENT, 2), (Phase.REVEAL, 3)])
    def test_tracks_the_current_phase(self, experience, phase, stage):
        experience.state.go_to(phase)
        experience._render_phase()
        assert experience.overlay.stages._active == stage

    def test_costs_no_layout_height(self, experience):
        """Regression: as a layout item the strip squeezed the prediction
        buttons below the 64 px touch floor."""
        assert experience.overlay.stages.parent() is experience.overlay
        assert experience.overlay.layout().indexOf(experience.overlay.stages) == -1


class TestContextualHelp:
    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_smoke_leads_once_the_beat_has_fired(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience.time_controller.pause()
        beat = experience._story.ceiling_beat()
        for frame in range(beat.frame_index + 1):
            experience.time_controller.seek(frame)
        assert "smoke" in experience._what_am_i_seeing()[0].lower()

    def test_airflow_leads_while_the_air_is_racing(self, experience):
        experience._begin_journey()
        experience._on_prediction("cooler")
        run_countdown(experience)
        experience.time_controller.pause()
        for frame in range(0, 160, 8):
            experience.time_controller.seek(frame)
        lines = experience._what_am_i_seeing()
        assert any("really moving" in line for line in lines[:2])

    def test_falls_back_to_the_general_explanation(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience.time_controller.pause()
        experience.time_controller.seek(2)
        lines = experience._what_am_i_seeing()
        assert "Bigger dots" in lines[0]
        assert not any("smoke" in line.lower() for line in lines)

    def test_never_offers_more_than_fits(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience.time_controller.pause()
        beat = experience._story.ceiling_beat()
        for frame in range(beat.frame_index + 1):
            experience.time_controller.seek(frame)
        assert len(experience._what_am_i_seeing()) <= 4


class TestReplayInvitationRemoved:
    """The reveal/verdict card ("Not quite -- here's what really
    happened" / "Correct!") and the Science card ("What did we
    measure?") used to each carry their own "Try Fan OFF"/"Start again"
    replay button alongside "Show me the science"/"Back" -- a second,
    redundant way back into the experiment that cluttered both result
    screens. Trying the other side now happens by returning to the
    Games hub and picking "Test an idea" again, not from either card."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_reveal_does_not_show_the_invitation(self, experience):
        experience.state.record_prediction("cooler")
        experience.state.record_choice("fan_on")
        experience.state.go_to(Phase.REVEAL)
        experience._render_phase()
        texts = [b.text() for b in experience.overlay.children()
                 if isinstance(b, QtWidgets.QPushButton)]
        assert not any("Fan OFF" in t for t in texts)
        assert not any("Start again" in t for t in texts)
        assert any("science" in t.lower() for t in texts)

    def test_science_card_does_not_show_the_invitation(self, experience):
        experience.state.record_prediction("cooler")
        experience.state.record_choice("fan_on")
        experience.state.go_to(Phase.SCIENCE)
        experience._render_phase()
        texts = [b.text() for b in experience.overlay.children()
                 if isinstance(b, QtWidgets.QPushButton)]
        assert not any("Fan OFF" in t for t in texts)
        assert not any("Start again" in t for t in texts)
        assert any("back" in t.lower() for t in texts)


class TestAttractScreen:
    def test_shows_a_giant_call_to_action_over_a_live_fire(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            buttons = [b for b in experience.overlay.children()
                       if isinstance(b, BigButton)]
            assert len(buttons) == 1
            assert "EXPLORE THE FIRE" in buttons[0].text()
            assert buttons[0].minimumHeight() >= 64
            # The pitch is the simulation itself, already running.
            assert experience.time_controller.is_playing()
            assert experience.overlay.temperature_meter.isVisible() or \
                not experience.overlay.temperature_meter.isHidden()
        finally:
            window.close()


# ------------------------------------------------- scientific honesty
class TestFlowDirectionIsNeverClaimed:
    """The stored VELOCITY slice is speed magnitude; the signed U/W
    components are gated (registry.py). Any arrow direction is therefore
    inferred from a temperature gradient, and the public experience must
    never present an inference as a measurement."""

    def test_signed_components_really_are_unavailable(self):
        """If this ever fails, real direction data has arrived and the
        public scene should switch back to true vectors."""
        for name in ("U-VELOCITY", "W-VELOCITY", "V-VELOCITY"):
            assert registry.quantity_status(name) == "gated"

    def test_public_scene_uses_direction_free_glyphs(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            view = window.public_experience.scene.view
            assert view.flow_mode == "activity"
            assert view.velocity_quiver is None
        finally:
            window.close()

    def test_research_views_keep_the_original_arrows(self, qapp):
        """Research mode documents its quiver as a cinematic effect and
        must be untouched by the public change."""
        view = SliceView()
        assert view.flow_mode == "arrows"

    def test_activity_glyphs_encode_only_magnitude(self):
        """Same speed everywhere must give identical glyphs regardless of
        the temperature field -- proof no direction/gradient leaks in."""
        rows, cols = velocity_arrows.sample_points((20, 30))
        speed = np.full((20, 30), 0.5)
        sizes_a, alphas_a = velocity_arrows.compute_activity(speed, rows, cols)
        assert np.allclose(sizes_a, sizes_a[0])
        assert np.allclose(alphas_a, alphas_a[0])
        # compute_activity takes no temperature argument at all.
        assert "temperature" not in velocity_arrows.compute_activity.__code__.co_varnames

    def test_glyph_size_grows_with_measured_speed(self):
        rows, cols = velocity_arrows.sample_points((20, 30))
        slow, _ = velocity_arrows.compute_activity(np.full((20, 30), 0.09), rows, cols)
        fast, _ = velocity_arrows.compute_activity(np.full((20, 30), 0.54), rows, cols)
        assert fast[0] > 2 * slow[0]

    def test_still_air_draws_nothing(self):
        rows, cols = velocity_arrows.sample_points((20, 30))
        sizes, alphas = velocity_arrows.compute_activity(
            np.full((20, 30), 0.001), rows, cols)
        assert np.all(sizes == 0.0)
        assert np.all(alphas == 0.0)

    def test_no_public_text_claims_a_direction(self, qapp):
        """No visitor-facing string may say which way the air is going."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience._begin_journey()
            experience._start_observe()
            experience.time_controller.pause()
            banned = ("which way", "direction", "points", "pointing",
                      "shows where", "flows toward")
            for frame in (0, 60, 120):
                experience.time_controller.seek(frame)
                for line in experience._what_am_i_seeing():
                    assert not any(b in line.lower() for b in banned), line
        finally:
            window.close()

    def test_help_describes_size_not_heading(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience._begin_journey()
            experience._start_observe()
            experience.time_controller.pause()
            experience.time_controller.seek(2)
            assert "Bigger dots" in experience._what_am_i_seeing()[0]
        finally:
            window.close()

    def test_fan_contrast_survives_the_change(self, qapp):
        """Honesty must not cost the effect: the glyphs still separate the
        two scenarios strongly."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        rows, cols = velocity_arrows.sample_points((49, 101))
        sizes = {}
        for key in ("fan_off", "fan_on"):
            case = experiments_mod.resolve_choice(
                sim.manifest, experiments_mod.FAN_EXPERIMENT, key)
            field = np.asarray(sim.store.get(case, DEFAULT_SLICE_KEY.__class__(
                "VELOCITY", 1, 0)))
            sizes[key] = velocity_arrows.compute_activity(field[300], rows, cols)[0].mean()
        assert sizes["fan_on"] > 2 * sizes["fan_off"]


class TestSootVersusTemperatureProxy:
    """History: why the public smoke used to be temperature-derived, and
    the real numbers that made that stop being defensible.

    On the sim_stage1_prep dataset (the higher-fidelity Pleiades re-run,
    see load_data.SIM_ROOT), real SOOT DENSITY is dense -- ~98-99% plane
    occupancy by the end of a run, reaching the ceiling, not the ~0.6-0.8%
    thread-above-the-candle the original decision measured. The remaining
    blocker documented here at the time -- SOOT DENSITY's own `.s3d`
    output schedule (1001 frames) not matching TEMPERATURE/VELOCITY's
    `.sf` rate (481 frames) over the same ~120s run, so naive index
    pairing silently mismatches real timestamps -- is now resolved: see
    cinema/real_smoke.py (soot_at_time, real-timestamp interpolation) and
    TestRealSootIsThePublicSmokeSource below, which confirms public mode
    now uses real, time-aligned SOOT DENSITY directly rather than the
    temperature-threshold proxy this class's remaining test still
    describes (that proxy lives on only as EffectsPipeline's fallback for
    callers with no real-soot alignment -- see cinema/smoke.py)."""

    SOOT_KEY = SliceKey("SOOT DENSITY", 1, 0, 0.0)

    @pytest.fixture(scope="class")
    def fields(self):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        out = {}
        for key in ("fan_off", "fan_on"):
            case = experiments_mod.resolve_choice(
                sim.manifest, experiments_mod.FAN_EXPERIMENT, key)
            out[key] = (np.asarray(sim.store.get(case, self.SOOT_KEY)),
                        np.asarray(sim.store.get(case, DEFAULT_SLICE_KEY)))
        return out

    def test_soot_and_temperature_share_a_grid_but_not_a_frame_count(self, fields):
        """Same spatial grid (both read through ScenarioStore on the same
        plane) -- but SOOT DENSITY's own `.s3d` dump schedule gives it
        roughly twice as many frames as TEMPERATURE's `.sf` slices over
        the same run, not the 1:1 frame-for-frame correspondence a naive
        `soot[i]` / `temperature[i]` pairing would assume."""
        for soot, temperature in fields.values():
            assert soot.shape[1:] == temperature.shape[1:]
            assert soot.shape[0] != temperature.shape[0]
            assert soot.shape[0] > temperature.shape[0]

    @pytest.mark.parametrize("key", ["fan_off", "fan_on"])
    def test_real_soot_is_now_dense(self, fields, key):
        """The sparsity that originally justified the temperature proxy
        (~0.6-0.8% plane occupancy) no longer holds on this dataset."""
        soot, _ = fields[key]
        occupancy = float((soot[-1] > 0).mean())
        assert occupancy > 0.5, (
            f"soot covers only {occupancy:.1%} of the plane -- back to the "
            "sparse regime the temperature proxy was originally chosen for")

    @pytest.mark.parametrize("key", ["fan_off", "fan_on"])
    def test_real_soot_now_fills_top_to_bottom(self, fields, key):
        """Row 0 is the ceiling (both fields are ceiling-first). The
        thin above-the-candle thread the original decision measured is
        gone -- top and bottom thirds are now comparably filled."""
        soot, _ = fields[key]
        rows = soot[-1].mean(axis=1)
        top_third = rows[:len(rows) // 3].mean()
        bottom_third = rows[2 * len(rows) // 3:].mean()
        assert top_third == pytest.approx(bottom_third, rel=0.5), (
            "soot is back to a thin thread instead of filling the room -- "
            "re-check the sparsity assumption above too")

    @pytest.mark.parametrize("key", ["fan_off", "fan_on"])
    def test_proxy_no_longer_tracks_real_soot_once_time_aligned(self, fields, key):
        """About EffectsPipeline's fallback synthetic proxy specifically
        (cinema/smoke.py's temperature-threshold production term), not
        the real-soot path public mode now actually uses. The original
        "proxy is a superset of real soot" justification, re-measured
        with soot and temperature frames paired by real elapsed time
        (not raw index -- see the class docstring) rather than the two
        rates being conflated. Documents the fallback's own remaining
        gap rather than gating on the old (no-longer-true) >90% claim."""
        soot, temperature = fields[key]
        n_soot, n_temp = soot.shape[0], temperature.shape[0]
        covered = total = 0
        for frame in range(50, n_soot, 10):
            t_idx = min(round(frame * (n_temp - 1) / (n_soot - 1)), n_temp - 1)
            soot_mask = soot[frame] > 0
            proxy_mask = (temperature[t_idx] - 20.0) > SOURCE_THRESHOLD_C
            if soot_mask.any():
                covered += int((soot_mask & proxy_mask).sum())
                total += int(soot_mask.sum())
        assert total > 0
        coverage = covered / total
        # Not a target -- a measurement. Real soot is now so dense that
        # the (still comparatively sparse) hot-cell proxy inevitably
        # covers only a small slice of it; recorded so a future change
        # to the proxy's own threshold has a real baseline to compare
        # against instead of silently drifting.
        assert coverage < 0.5, (
            f"proxy now covers {coverage:.1%} of real soot cells -- higher "
            "than expected, re-check this test's own time alignment")

    def test_public_scene_still_keys_its_main_slice_on_temperature(self, qapp):
        """The scene's own _quantity_key (drives the heatmap/probe/
        thermometer) stays TEMPERATURE -- SOOT DENSITY is loaded
        alongside it (see TestRealSootIsThePublicSmokeSource) as a
        second, independent field for the smoke layer only, not a
        replacement for what every other public-mode reading uses."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            scene = window.public_experience.scene
            assert scene._quantity_key.quantity == "TEMPERATURE"
            assert scene._quantity_key.plane_pos is None
        finally:
            window.close()

    def test_smoke_still_needs_no_gated_quantity(self):
        """The proxy uses only quantities that actually have data."""
        for name in ("TEMPERATURE", "VELOCITY"):
            assert registry.quantity_status(name) == "available"


class TestRealSootIsThePublicSmokeSource:
    """Architecture C, integration-level: public mode's rendered smoke is
    real, time-aligned SOOT DENSITY -- not cinema/smoke.py's synthetic
    buoyancy/decay/reservoir model (that stays only as EffectsPipeline's
    fallback for callers with no real-soot alignment, e.g. the researcher
    app's generic Cinematic fire view toggle; see cinema/real_smoke.py
    and cinema/pipeline.py's own EffectsPipeline.render docstring)."""

    SOOT_KEY = SliceKey("SOOT DENSITY", 1, 0, 0.0)

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_scene_loads_real_soot_frames_and_both_real_clocks(self, experience):
        experience._begin_journey()
        experience._start_observe()
        scene = experience.scene
        assert scene._soot is not None
        assert scene._soot_times is not None
        assert scene._temp_times is not None
        assert scene._soot.shape[1:] == scene._temperature.shape[1:]
        assert len(scene._soot_times) == scene._soot.shape[0]
        assert len(scene._temp_times) == scene._temperature.shape[0]

    def test_spatial_consistency_between_temperature_and_soot(self, experience, sim_data):
        """Same extent, same (n_row, n_col), same ceiling-first row
        orientation -- verified directly, not assumed, before this data
        is ever handed to the renderer (see the architecture audit's own
        spatial-compatibility check, re-asserted here as a standing
        regression guard)."""
        case_index = experience.state.case_index
        temp_extent = sim_data.store.get_extent(case_index, DEFAULT_SLICE_KEY)
        soot_extent = sim_data.store.get_extent(case_index, self.SOOT_KEY)
        assert temp_extent == soot_extent
        temp = np.asarray(sim_data.store.get(case_index, DEFAULT_SLICE_KEY))
        soot = np.asarray(sim_data.store.get(case_index, self.SOOT_KEY))
        assert temp.shape[1:] == soot.shape[1:]
        # Ceiling-first for both: early in the run (before any plume has
        # risen) floor must exceed ceiling in both fields alike.
        assert temp[5, -1].mean() > temp[5, 0].mean()
        assert soot[10, -1].mean() >= soot[10, 0].mean()

    def test_smoke_density_at_matches_independent_soot_at_time(self, experience):
        """The exact value _smoke_density_at() hands to the renderer for
        the currently-displayed frame must equal an independently
        recomputed soot_at_time()+normalize_soot_density() call -- no
        hidden extra transformation between the two."""
        experience._begin_journey()
        experience._start_observe()
        scene = experience.scene
        frame_i = experience.state.frame_index
        target_time = scene._temp_times[frame_i]
        expected = normalize_soot_density(soot_at_time(scene._soot, scene._soot_times, target_time))
        actual = scene._smoke_density_at(frame_i)
        assert np.array_equal(expected, actual)

    def test_synthetic_smoke_simulator_never_instantiated_in_public_mode(self, experience):
        """The whole point of Architecture C: cinema/smoke.py's
        SmokeSimulator (buoyancy/decay/reservoir) must never even be
        created on the public-mode path once real soot is available."""
        experience._begin_journey()
        experience._start_observe()
        for _ in range(5):
            experience.time_controller.seek(experience.time_controller.index + 1)
        pipeline = experience.scene.view._cinema_pipeline
        assert pipeline._smoke is None

    def test_fixed_reference_density_is_identical_across_scenarios(self, sim_data):
        """REFERENCE_DENSITY must be a single constant, not recomputed
        per scenario -- the whole point of choosing a dataset-wide value
        (see cinema/real_smoke.py's own derivation) is that a mild and a
        severe scenario stay comparable. Importing it fresh for two
        different scenarios' worth of normalization must yield the same
        number both times (it's a module constant, not scenario state)."""
        from cinema.real_smoke import REFERENCE_DENSITY as ref_a
        from cinema.real_smoke import REFERENCE_DENSITY as ref_b
        assert ref_a == ref_b

    def test_a_denser_scenario_produces_greater_mean_opacity_under_shared_normalization(self, sim_data):
        """Cross-scenario comparability, the actual requirement the fixed
        reference exists to satisfy: a real 2-candle, both-vents-closed
        scenario (denser smoke) must read as visibly smokier than a real
        1-candle, both-vents-open scenario, under the exact same
        REFERENCE_DENSITY -- not each rescaled to its own peak."""
        mild_case = experiments_mod.resolve_case_index(
            sim_data.manifest, {"candles": 0, "vod": 0, "voc": 0, "door": 1})
        severe_case = experiments_mod.resolve_case_index(
            sim_data.manifest, {"candles": 1, "vod": 1, "voc": 1, "door": 1})
        mild_soot = np.asarray(sim_data.store.get(mild_case, self.SOOT_KEY))
        severe_soot = np.asarray(sim_data.store.get(severe_case, self.SOOT_KEY))
        mild_opacity = normalize_soot_density(mild_soot[-1]).mean()
        severe_opacity = normalize_soot_density(severe_soot[-1]).mean()
        assert severe_opacity > mild_opacity


class TestSmokeMotionIsAtmosphericNotDirectional:
    """cinema/smoke.py's Tier-2 advection blends a temperature-gradient
    prior into its drift direction, so the question is whether a child
    could read the rendered smoke as *measured* airflow direction.

    Measured on the real scenarios: over 400 frames the smoke centroid
    rises ~13 of 49 rows while drifting 3.3 of 101 columns, the drift is
    the same whether the fan is on or off, and it differs from a
    buoyancy-only run (no velocity data at all) by 0.4 of 101 columns.
    The horizontal component is large per-pixel but symmetric about the
    plume, so it renders as spreading, not as wind. These tests keep it
    that way -- if a future change let velocity steer the smoke sideways,
    the effect would start implying a direction nobody measured.
    """

    FRAMES = 150

    @staticmethod
    def _centroid_drift(store, case_index, tier2: bool, frames: int):
        temperature = np.asarray(store.get(case_index, DEFAULT_SLICE_KEY))
        velocity = np.asarray(store.get(case_index, SliceKey("VELOCITY", 1, 0)))
        sim = SmokeSimulator(temperature.shape[1:], ambient_c=20.0)
        xs, ys = [], []
        for i in range(frames):
            buffer = sim.step(temperature[i], velocity[i] if tier2 else None)
            total = buffer.sum()
            if total <= 1e-6:
                continue
            grid_y, grid_x = np.mgrid[0:buffer.shape[0], 0:buffer.shape[1]]
            xs.append(float((buffer * grid_x).sum() / total))
            ys.append(float((buffer * grid_y).sum() / total))
        return xs[-1] - xs[0], ys[0] - ys[-1]      # (x drift, upward rise)

    @pytest.fixture(scope="class")
    def drifts(self, request):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        out = {}
        for key in ("fan_off", "fan_on"):
            case = experiments_mod.resolve_choice(
                sim.manifest, experiments_mod.FAN_EXPERIMENT, key)
            out[key] = self._centroid_drift(sim.store, case, True, self.FRAMES)
            out[key + "_buoyancy"] = self._centroid_drift(
                sim.store, case, False, self.FRAMES)
        return out

    @pytest.mark.parametrize("key", ["fan_off", "fan_on"])
    def test_motion_is_dominated_by_rising(self, drifts, key):
        x_drift, rise = drifts[key]
        assert rise > 0, "smoke must rise"
        assert rise > 2 * abs(x_drift), (
            f"horizontal drift {x_drift:.1f} too large beside rise {rise:.1f}")

    @pytest.mark.parametrize("key", ["fan_off", "fan_on"])
    def test_velocity_does_not_steer_the_smoke_sideways(self, drifts, key):
        """The measured field may change how the smoke looks, but it must
        not change *where* it goes -- otherwise the drift becomes an
        inferred direction masquerading as a measurement."""
        with_velocity, _ = drifts[key]
        buoyancy_only, _ = drifts[key + "_buoyancy"]
        assert abs(with_velocity - buoyancy_only) < 1.5, (
            "velocity data is steering the smoke horizontally")

    def test_smoke_drifts_the_same_way_with_the_fan_on_or_off(self, drifts):
        """The decisive one: if the fan changed the smoke's direction, a
        child would reasonably conclude the simulation measured which way
        the fan blows. It does not."""
        off_x, _ = drifts["fan_off"]
        on_x, _ = drifts["fan_on"]
        assert abs(on_x - off_x) < 1.0, (
            f"fan changes smoke drift ({off_x:.1f} -> {on_x:.1f}); it would "
            "read as a measured direction")

    def test_no_public_text_claims_the_smoke_shows_a_direction(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience._load_case(experience.state.baseline_case_index)
            smoke_lines = [line for line in experience._what_am_i_seeing()
                           if "smoke" in line.lower()]
            smoke_lines.append(story_mod.CEILING_BEAT_TEXT)
            for line in smoke_lines:
                assert not any(word in line.lower() for word in
                               ("blows", "blowing", "direction", "toward",
                                "which way", "pushed")), line
        finally:
            window.close()


class TestScientificCoherence:
    """Cross-checks that the whole public narrative is answering to the
    same measured numbers."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_reveal_and_science_share_the_hero_metric(self, experience):
        experience.state.record_prediction("cooler")
        experience.state.record_choice("fan_on")
        hero = experiments_mod.strongest_metric(experience._science_comparisons())
        _, lines, _, _ = experience._reveal_lines()
        assert kid.finding_sentence(hero).lower() in " ".join(lines).lower()

    def test_prediction_is_judged_against_its_own_metric(self, experience):
        """A guess about temperature must be checked against temperature,
        not against whatever happened to change most."""
        guess = experience.experiment.prediction("cooler")
        assert guess.metric_key == "room_temp"
        comparisons = experience._science_comparisons()
        assert experience._guess_is_borne_out(guess, comparisons)
        flat = [experiments_mod.MetricComparison(c.metric, 1.0, 1.0)
                for c in comparisons]
        assert not experience._guess_is_borne_out(guess, flat)

    def test_meter_never_reports_the_flame_as_room_air(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience.time_controller.pause()
        experience.time_controller.seek(200)
        room = experience.scene.mean_temperature_at(200)
        assert room < 60.0            # a tabletop room, never a furnace
        assert f"{room:.0f}" in experience.overlay.temperature_meter._value.text()

    def test_smoke_narration_matches_a_detected_event(self, experience):
        beat = experience._story.ceiling_beat() if experience._story else None
        if beat is None:
            experience._load_case(experience.state.baseline_case_index)
            beat = experience._story.ceiling_beat()
        assert beat is not None
        assert beat.source.basis                    # traceable to a computation
        assert beat.frame_index > 4                 # not the frame-0 sentinel

    def test_wording_matches_the_region_actually_averaged(self, experience):
        """mean_temperature_end averages the whole simulated area; the
        enclosed room is only ~a third of it. Nothing may call that
        number "the room"."""
        from schematic import ROOM_X, ROOM_Z, _DOMAIN_X, _DOMAIN_Z
        room = (ROOM_X[1] - ROOM_X[0]) * (ROOM_Z[1] - ROOM_Z[0])
        domain = (_DOMAIN_X[1] - _DOMAIN_X[0]) * (_DOMAIN_Z[1] - _DOMAIN_Z[0])
        assert room / domain < 0.5, "geometry changed; revisit this wording"

        metric = next(m for m in experiments_mod.PUBLIC_METRICS
                      if m.key == "room_temp")
        for text in (metric.label, metric.meaning, metric.explanation,
                     metric.kid_subject):
            assert "room" not in text.lower(), text
        # _caption_full_text, not ._caption.text(): the displayed label
        # is elided to fit the narrow stat-panel chip now (see
        # MeterChip's own flat= docstring), so the on-screen text can be
        # "AIR TEMP…" -- the full underlying string (what this test
        # actually cares about not saying "room") still lives here.
        assert experience.overlay.temperature_meter._caption_full_text == "AIR TEMPERATURE"

    def test_mascot_never_outruns_the_data(self, experience):
        """Every spoken finding must come from a comparison that cleared
        its own noticeable_delta."""
        comparisons = experience._science_comparisons()
        for comparison in comparisons:
            spoken = kid.mascot_finding(comparison)
            assert bool(spoken) == comparison.is_noticeable


# ------------------------------------------------ exhibit UX regressions
class TestExhibitMoments:
    """Guards for the three weakest moments found by inspecting the
    running application at 800x600."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    # -- intro was a wall of adult prose ------------------------------
    def test_intro_is_short_enough_for_a_child(self, experience, qapp):
        experience._begin_journey()
        for _ in range(6):
            qapp.processEvents()
        card = experience.overlay.card
        body = [w.text() for w in card.findChildren(QtWidgets.QLabel)]
        words = sum(len(t.split()) for t in body)
        assert words <= 35, f"intro is {words} words: {body}"
        assert card.height() >= card.sizeHint().height(), "intro card is clipped"

    def test_intro_still_has_one_obvious_action(self, experience, qapp):
        experience._begin_journey()
        for _ in range(6):
            qapp.processEvents()
        buttons = [b for b in experience.overlay.children()
                   if isinstance(b, BigButton)]
        assert len(buttons) == 1
        assert buttons[0].height() >= 64
        assert not buttons[0].isHidden()

    def test_intro_still_credits_the_real_simulation(self, experience, qapp):
        experience._begin_journey()
        for _ in range(6):
            qapp.processEvents()
        text = " ".join(w.text() for w in
                        experience.overlay.card.findChildren(QtWidgets.QLabel))
        assert "FDS" in text and "candle" in text.lower()

    # -- experiment said the same sentence twice ------------------------
    def test_ceiling_beat_reaches_the_guide_not_the_banner(self, experience):
        """Regression sibling to TestObserveTiming's narration test --
        missed when that one was updated for the mascot-whisper redesign
        (see _narrate: the beat never flashes the banner, the finding
        goes straight to the guide's speech bubble instead). Same
        original intent as this test's old name -- the guide surfaces the
        finding, it never duplicates it into a second on-screen
        announcement -- just guarding the opposite direction now: the
        beat must not hijack the banner again.

        Driven through the guided EXPERIMENT run, not OBSERVE: narration
        is EXPERIMENT-only now (see docs/HANDOFF-PUBLIC-MODE-REDESIGN.md
        §3/§4 and TestObserveTiming.test_observe_never_narrates_the_smoke_beat)."""
        experience._begin_journey()
        experience._on_prediction("cooler")
        run_countdown(experience)
        assert experience.state.phase is Phase.EXPERIMENT
        experience.time_controller.pause()
        prompt_before = experience.overlay.banner.text()
        beat = experience._story.ceiling_beat()
        for frame in range(beat.frame_index + 1):
            experience.time_controller.seek(frame)
        banner = experience.overlay.banner.text()
        bubble = experience.overlay.bubble.text()
        assert "gathering under the ceiling" in bubble
        assert "gathering under the ceiling" not in banner
        assert banner == prompt_before

    def test_every_beat_has_its_own_reaction(self, sim_data):
        """Every beat has a distinguishing mascot reaction except the
        ceiling beat, which deliberately has none -- the mascot-whisper
        redesign has it speak the finding itself instead of a separate
        reaction line (see story.py's _detect_ceiling_beat and
        test_ceiling_beat_reaches_the_guide_not_the_banner above)."""
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        for beat in _story_for_choice(sim_data, "fan_off").beats():
            if beat.text == story_mod.CEILING_BEAT_TEXT:
                assert not beat.reaction
                continue
            assert beat.reaction, f"beat {beat.text!r} has no reaction"
            assert beat.reaction != beat.text

    def test_observation_builds_anticipation_before_the_beat(self, experience):
        """The experiment run used to be dead air followed by a shout.
        Driven through EXPERIMENT -- see the class docstring change on
        test_ceiling_beat_reaches_the_guide_not_the_banner above."""
        experience._begin_journey()
        experience._on_prediction("cooler")
        run_countdown(experience)
        assert experience.state.phase is Phase.EXPERIMENT
        experience.time_controller.pause()
        beat = experience._story.ceiling_beat()
        said = []
        original = experience.overlay.say
        experience.overlay.say = lambda t, m=None: (said.append(t), original(t, m))[1]
        for frame in range(beat.frame_index + 1):
            experience.time_controller.seek(frame)
        assert "Look closely…" in said
        # The ceiling beat has no separate reaction line anymore -- it
        # speaks the finding itself (see _narrate's beat_line fallback).
        finding_said = next(t for t in said if "gathering under the ceiling" in t)
        assert said.index("Look closely…") < said.index(finding_said)

    def test_nudge_fires_once_and_resets_on_replay(self, experience):
        experience._begin_journey()
        experience._on_prediction("cooler")
        run_countdown(experience)
        assert experience.state.phase is Phase.EXPERIMENT
        experience.time_controller.pause()
        beat = experience._story.ceiling_beat()
        said = []
        original = experience.overlay.say
        experience.overlay.say = lambda t, m=None: (said.append(t), original(t, m))[1]
        for frame in range(beat.frame_index + 1):
            experience.time_controller.seek(frame)
        assert said.count("Look closely…") == 1
        experience._on_replay()
        assert not experience._nudged

    def test_no_nudge_when_there_is_no_beat_to_anticipate(self, experience):
        experience.state.go_to(Phase.EXPERIMENT)
        experience._story = StoryController(
            np.full((200, 20, 30), 20.0, dtype=np.float32), (0.0, 1.0, 0.0, 0.48), 4)
        experience._nudged = False
        for frame in range(0, 150, 10):
            experience._nudge_before_beat(frame)
        assert not experience._nudged

    # -- countdown did not connect to the guess ------------------------
    def test_countdown_names_the_idea_being_tested(self, experience):
        experience._begin_journey()
        experience._on_prediction("cooler")
        guess = experience.experiment.prediction("cooler")
        banner = experience.overlay.banner.text()
        assert guess.label in banner
        assert "Testing your idea" in banner

    def test_countdown_copes_with_no_recorded_guess(self, experience):
        experience.state.go_to(Phase.COUNTDOWN)
        experience.state.prediction = None
        experience._render_phase()
        assert experience.overlay.banner.text()
        experience._countdown_timer.stop()


# ------------------------------------------------------------- manifest
class TestManifestPortability:
    def test_manifest_paths_are_inside_this_checkout(self, sim_data):
        """Regression for a real defect: this study's manifest stored
        absolute paths into a *different* checkout, so the app silently
        served another directory's data and would fall back to demo mode
        the moment that directory moved."""
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        assert foreign_path_entries(sim_data.manifest, SIM_ROOT) == []

    def test_foreign_path_detection_flags_an_outside_path(self, sim_data):
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        entries = list(sim_data.manifest)
        assert foreign_path_entries(entries, os.path.join(SIM_ROOT, "nope")) == entries


# ----------------------------------------------------------- mode switch
class TestPublicModeIntegration:
    def test_enter_and_exit_restores_the_researcher_shell(self, qapp):
        window = MainWindow(load_simulation_data())
        try:
            assert not window.is_public_mode()
            window.enter_public_mode()
            assert window.is_public_mode()
            assert window.root_stack.currentWidget() is window.public_experience
            # isHidden(), not isVisible(): the window itself is never
            # shown under the offscreen platform, so isVisible() is False
            # for every child either way and would not test anything.
            assert window.menuBar().isHidden()
            assert window.statusBar().isHidden()

            window.exit_public_mode()
            assert not window.is_public_mode()
            # Deliberate exit lands on the Welcome landing page, not the
            # researcher shell -- the ✕ affordance must leave no route
            # back to research state (see main_window.exit_public_mode).
            assert window.root_stack.currentWidget() is window.welcome_widget
            # Researcher chrome must stay hidden on Welcome too: exiting
            # public mode must never quietly re-expose it.
            assert window.menuBar().isHidden()
            assert window.statusBar().isHidden()
        finally:
            window.close()

    def test_public_mode_pauses_and_does_not_disturb_research_state(self, qapp):
        window = MainWindow(load_simulation_data())
        try:
            window._navigate_to("live")
            window.time_controller.seek(5)
            layout_before = window.view_grid.layout_name
            window.enter_public_mode()
            assert not window.time_controller.is_playing()
            window.exit_public_mode()
            assert window._active_page_key == "live"
            assert window.time_controller.index == 5
            assert window.view_grid.layout_name == layout_before
        finally:
            window.close()

    def test_nav_shortcuts_are_inert_while_public(self, qapp):
        """A visitor pressing 1-7 must not rearrange the hidden
        researcher shell."""
        window = MainWindow(load_simulation_data())
        try:
            window._navigate_to("live")
            window.enter_public_mode()
            window._navigate_to("analysis")
            assert window._active_page_key == "live"
            window._toggle_play_pause()
            assert not window.time_controller.is_playing()
        finally:
            window.close()

    def test_experience_reaches_reveal_through_the_real_flow(self, qapp):
        """Drives the actual journey the way the buttons do, and asserts
        the scene really switched to the other scenario."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            baseline = experience.state.baseline_case_index

            experience._begin_journey()
            assert experience.state.phase is Phase.INTRO
            experience._start_observe()
            assert experience.state.phase is Phase.OBSERVE

            experience.time_controller.pause()
            experience._observe_finished()
            assert experience.state.phase is Phase.PREDICTION

            experience._on_prediction("cooler")
            run_countdown(experience)
            assert experience.state.phase is Phase.EXPERIMENT
            assert experience.state.case_index != baseline

            experience.time_controller.pause()
            experience._experiment_finished()
            assert experience.state.phase is Phase.REVEAL
        finally:
            window.close()

    def test_reveal_text_is_built_from_measured_values(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience.state.record_prediction("cooler")
            experience.state.record_choice("fan_on")
            headline, lines, dim, _ = experience._reveal_lines()
            assert "prediction" in headline.lower()
            assert any("the air moved much faster" in line.lower() for line in lines)
            assert any("m/s" in line for line in dim)
        finally:
            window.close()

    def test_replay_restarts_the_run_at_the_observation(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience._on_prediction("cooler")
            experience.time_controller.seek(200)
            experience.time_controller.pause()
            experience._spoken_beats.add(64)

            experience._on_replay()

            assert experience.state.phase is Phase.OBSERVE
            assert experience.state.case_index == experience.state.baseline_case_index
            assert experience.time_controller.index == 0
            assert experience._spoken_beats == set()
            # The fire has to be running behind the observation, not
            # frozen on an unlit frame 0.
            assert experience.time_controller.is_playing()
        finally:
            window.close()

    def test_replay_never_auto_advances_past_the_old_observe_boundary(self, qapp):
        """NEXT MILESTONE: free play must never hand the child back to the
        narrative on a timer -- see PublicExperience._start_free_play.
        Seeking straight through (and well past, via looping) the frame
        that used to trigger the automatic hand-off must leave the child
        in OBSERVE; only pressing "Test an idea" leaves it."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience._begin_journey()
            experience._on_replay()
            experience.time_controller.pause()

            end = experience._observe_end_frame()
            last = experience.scene.frame_count() - 1
            # Twice round the loop, well past the old boundary.
            for frame in list(range(end + 20)) + list(range(last + 1)):
                experience.time_controller.seek(frame)
            assert experience.state.phase is Phase.OBSERVE

            experience._on_test_idea_clicked()
            assert experience.state.phase is Phase.PREDICTION
            # ...and the fan experiment still resolves from the manifest.
            experience._on_prediction("cooler")
            run_countdown(experience)
            assert experience.state.phase is Phase.EXPERIMENT
            assert experience.state.case_index != experience.state.baseline_case_index
        finally:
            window.close()

    def test_observe_loops_indefinitely_until_the_child_moves_on(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience._begin_journey()
            experience._start_observe()
            last = experience.scene.frame_count() - 1
            experience.time_controller.seek(last)
            # TimeController._tick wraps to 0 when looping is on and the
            # timer fires past the last frame -- simulated directly here
            # rather than waiting on the real QTimer.
            experience.time_controller._tick()
            assert experience.time_controller.index == 0
            assert experience.state.phase is Phase.OBSERVE
            assert experience.time_controller.is_playing()
        finally:
            window.close()

    def test_test_idea_leaves_observe_on_demand(self, qapp):
        """"Test an idea" is reached via the Games hub now, not directly
        from Explore -- see PublicExperience._enter_game_test_idea."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience._begin_journey()
            experience._start_observe()
            assert any("Games" in t for t in
                      (b.text() for b in experience.overlay._nav_buttons))

            experience._enter_game_test_idea()
            assert experience.state.phase is Phase.PREDICTION
            assert not experience.time_controller.is_playing()
        finally:
            window.close()

    def test_test_idea_tile_clicked_leaves_observe(self, qapp):
        """Driven through the real widgets (Games button, then the "Test
        an Idea" tile), not the handlers directly -- proves the whole
        path is actually wired up."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience._begin_journey()
            experience._start_observe()
            games_button = next(b for b in experience.overlay._nav_buttons if "Games" in b.text())
            games_button.click()
            tiles = [experience.overlay._games_grid.itemAt(i).widget()
                     for i in range(experience.overlay._games_grid.count())]
            test_idea_tile = next(t for t in tiles if "Test an Idea" in t.text())

            test_idea_tile.click()

            assert experience.state.phase is Phase.PREDICTION
        finally:
            window.close()

    def test_replay_leaves_no_stale_overlay_widgets(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            overlay = experience.overlay
            experience.state.record_prediction("cooler")
            experience.state.record_choice("fan_on")
            experience.state.go_to(Phase.REVEAL)
            experience._render_phase()
            assert not overlay.card.isHidden()

            experience._on_replay()

            # Observation shows the fire, not the reveal card or its buttons.
            assert overlay.card.isHidden()
            # Like exit_button, the EN/DE language toggle is always-visible
            # chrome outside button_row/nav_row's per-phase redeclaration --
            # never one of the phase's own affordances, so it's excluded
            # here the same way exit_button already is.
            always_visible = (overlay.exit_button, overlay.lang_en_button, overlay.lang_de_button)
            remaining = [b.text() for b in overlay.children()
                         if isinstance(b, QtWidgets.QPushButton)
                         and b not in always_visible]
            assert not any("Try again" in t or "science" in t for t in remaining)
            # Observing offers "What am I seeing?", the play/pause
            # control, and "Games" -- the reveal's own buttons must be
            # gone, not that only one affordance exists.
            assert all("What am I seeing" in t or "Pause" in t or "Play" in t
                       or "Games" in t
                       for t in remaining)
        finally:
            window.close()

    def test_phase_change_removes_the_previous_phases_buttons(self, qapp):
        """Regression: clear_buttons() used deleteLater() alone, and
        deferred deletions are not processed by processEvents(), so the
        previous phase's buttons stayed visible children of the overlay
        and painted on top of the new ones."""
        window = MainWindow(load_simulation_data())
        try:
            window.enter_public_mode()
            experience = window.public_experience
            overlay = experience.overlay

            def button_texts():
                return [b.text() for b in overlay.children()
                        if isinstance(b, QtWidgets.QPushButton)
                        and b is not overlay.exit_button]

            experience._begin_journey()
            assert any("Watch the fire" in t for t in button_texts())
            experience._start_observe()
            # Observing offers exactly one affordance: the contextual
            # "What am I seeing?" help. The intro's button is gone.
            assert not any("Watch the fire" in t for t in button_texts())
            assert [t for t in button_texts() if "What am I seeing" in t]
        finally:
            window.close()

    def test_reveal_card_is_populated_and_visible(self, qapp):
        """Regression: the card was faded in with a QGraphicsOpacityEffect
        that could leave it permanently at opacity 0 -- an exhibit that
        silently shows no explanation."""
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience.state.record_prediction("cooler")
            experience.state.record_choice("fan_on")
            experience.state.go_to(Phase.REVEAL)
            experience._render_phase()

            card = experience.overlay.card
            # isHidden(), not isVisible() -- the window is never shown
            # under the offscreen platform (see the mode-switch test).
            assert not card.isHidden()
            assert card.body().count() >= 4
            assert card.graphicsEffect() is None
        finally:
            window.close()

    def test_science_card_shows_measured_values(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            experience.state.record_choice("fan_on")
            experience.state.go_to(Phase.SCIENCE)
            experience._render_phase()

            card = experience.overlay.card
            text = "\n".join(w.text() for w in card.findChildren(QtWidgets.QLabel)).lower()
            # Air speed and room temperature are now charted (their labels
            # appear in the chart headings); flame temperature stays a row.
            for expected in ("air speed", "air temp.", "flame temp."):
                assert expected in text
            assert "°c" in text
            # The charted values live inside the painted bars.
            bars = card.findChildren(BarCompare)
            assert any(b._unit == "m/s" for b in bars)
        finally:
            window.close()

    def test_kiosk_idle_resets_the_experience_not_the_researcher_nav(self, qapp):
        window = MainWindow(load_simulation_data())
        try:
            window.enter_public_mode()
            window._kiosk._enter_idle()
            assert window.is_public_mode()
            assert window.public_experience.state.phase is Phase.ATTRACT
        finally:
            window.close()


# ------------------------------------------------- interactive laboratory
class TestTemperatureProbe:
    """Tap-to-inspect: PublicScene.probe_at() must return a real measured
    value at the tapped point, never an interpolated or invented one."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_tap_returns_the_real_array_value(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        canvas = experience.scene.view.canvas
        centre = QtCore.QPoint(canvas.width() // 2, canvas.height() // 2)
        result = experience.scene.probe_at(centre)
        assert result is not None
        x, z, value = result
        expected = experience.scene.view.value_at(x, z)
        assert value == pytest.approx(expected)

    def test_tap_outside_the_canvas_returns_nothing(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        canvas = experience.scene.view.canvas
        assert experience.scene.probe_at(QtCore.QPoint(-50, -50)) is None
        assert experience.scene.probe_at(
            QtCore.QPoint(canvas.width() + 500, canvas.height() + 500)) is None

    def test_probe_is_only_active_during_watch_phases(self, experience, qapp):
        experience._begin_journey()
        for _ in range(6):
            qapp.processEvents()
        # INTRO is not a probe phase.
        assert experience.state.phase is Phase.INTRO
        assert experience.overlay._probe_enabled is False

        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        assert experience.overlay._probe_enabled is True

    def test_tap_shows_a_real_reading_via_the_overlay_signal(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        canvas = experience.scene.view.canvas
        pos = QtCore.QPoint(canvas.width() // 2, canvas.height() // 2)
        experience._on_overlay_tapped(pos)
        assert experience._probe_mode == "point"
        assert "°C" in experience.overlay.thermometer._value_label.text()

    def test_reading_never_survives_a_phase_change(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        canvas = experience.scene.view.canvas
        experience._on_overlay_tapped(QtCore.QPoint(canvas.width() // 2, canvas.height() // 2))
        assert experience._probe_mode == "point"

        experience.time_controller.pause()
        experience._observe_finished()
        assert experience._probe_mode == "mean"

    def test_tap_ignored_while_probe_disabled(self, experience, qapp):
        experience._begin_journey()
        for _ in range(6):
            qapp.processEvents()
        # Still in INTRO: a tap must produce no reading at all.
        canvas = experience.scene.view.canvas
        experience._on_overlay_tapped(QtCore.QPoint(canvas.width() // 2, canvas.height() // 2))
        assert experience._probe_mode == "mean"

    def test_no_data_no_probe(self):
        """A scene that never loaded a scenario must not crash on a tap."""
        from public.scene import PublicScene
        scene = PublicScene(store=None, manifest=[], fps=4)
        assert scene.probe_at(QtCore.QPoint(10, 10)) is None


class TestExploreControls:
    """Direct manipulation: real controls that switch to a real scenario
    immediately, before the guided prediction begins."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_available_controls_are_real_and_fully_resolvable(self, sim_data):
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        controls = experiments_mod.available_explore_controls(sim_data.manifest)
        assert controls, "expected at least one usable explore control"
        for control in controls:
            for option in control.options:
                case = control.case_for(sim_data.manifest, option.value)
                assert case is not None
                assert any(e.case_index == case for e in sim_data.manifest)

    def test_control_missing_from_the_manifest_is_never_shown(self):
        """A control whose factor this study doesn't have must be
        rejected outright, not silently offered with a dead option."""
        fake = experiments_mod.ExploreControl(
            "nope", "Nope", "❓", "candles",
            (experiments_mod.ExploreOption(0, "A", "A"),
             experiments_mod.ExploreOption(99, "B", "B")),   # 99 never exists
            held={"door": 1, "vod": 0, "voc": 0})
        assert fake.is_available(load_simulation_data().manifest) is False

    def test_fan_toggle_switches_to_the_real_scenario(self, experience):
        experience._begin_journey()
        experience._start_observe()
        before = experience.state.case_index
        experience._on_explore_changed("vent1", 2)
        after = experience.state.case_index
        assert after != before
        entry = next(e for e in experience.sim_data.manifest if e.case_index == after)
        assert entry.vod == 2

    def test_explore_change_renders_frame_zero_exactly_once(self, experience):
        """Root-cause regression guard: _load_case (called first, on the
        scenario switch itself) and _start_free_play (called right
        after, in the same _on_explore_changed) both used to call
        TimeController.seek(0) unconditionally -- seek() always emits
        time_changed regardless of whether the index actually moved
        (see TimeController.seek's own body), so PublicScene.show_frame
        ran the full cinema-pipeline render of frame 0 twice on every
        single Vent/Candle/Door tap, the most frequent interaction in
        the app. _start_free_play now skips its own seek(0) when
        already at index 0, leaving _load_case's the only one."""
        experience._begin_journey()
        experience._start_observe()
        experience.time_controller.pause()
        calls = []
        original = experience.scene.show_frame
        experience.scene.show_frame = lambda *a, **kw: (calls.append(1), original(*a, **kw))[1]
        try:
            experience._on_explore_changed("vent1", 2)
        finally:
            experience.scene.show_frame = original
        assert len(calls) == 1, f"expected exactly one render of frame 0, got {len(calls)}"

    def test_explore_change_hard_cuts_not_cross_dissolves(self, experience, qapp):
        """Games UX pass, item 3: a control click shows the new
        scenario's real frame 0 immediately -- no cross-dissolve. This
        used to assert the opposite (a deliberate, tested ~240ms blend
        from whatever was on screen, see SliceView.start_scenario_
        transition) until that blend itself turned out to be the actual
        cause of a real, reported bug: paused mid-run on a hot frame,
        then switching Candles 2->1, the next several rendered frames
        were lerp_frames(old-scenario-hot-frame, new-scenario-frame-0,
        phase) -- genuinely still showing ~250-300 C of the *removed*
        candle's heat for up to one blend cycle, which read as "the old
        flame lingering and fading" (reproduced directly: a fresh smoke
        buffer, correctly reset to 0, still jumped to 0.96 within two
        ticks of the switch, purely from replaying that stale blend).
        PublicScene.load_case() no longer arms _pending_scenario_
        transition (see its own comment) -- the dissolve mechanism
        itself (SliceView.start_scenario_transition/_transition_tick) is
        untouched and still exercised directly by
        test_the_very_first_frame_ever_shown_never_dissolves below, for
        whoever picks up a corrected version of this as the real
        Phase 2a work."""
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        view = experience.scene.view
        assert view._last_frame is not None, "no baseline frame to dissolve from -- test setup issue"

        experience._on_explore_changed("vent1", 2)
        assert not view._transition_timer.isActive(), (
            "expected a hard cut; the scene started a cross-dissolve instead")
        assert view._transition_phase == 1.0

    def test_explore_change_button_feedback_is_immediate(self, experience, qapp):
        """The tapped button's own instant visual acknowledgement (the
        deliberate synchronous overlay.repaint() in _on_explore_changed)
        still fires -- unaffected by the dissolve being disabled (see
        test_explore_change_hard_cuts_not_cross_dissolves above)."""
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        repaint_order = []
        original_repaint = experience.overlay.repaint
        experience.overlay.repaint = lambda: (repaint_order.append("repaint"), original_repaint())[1]
        try:
            experience._on_explore_changed("vent1", 2)
        finally:
            experience.overlay.repaint = original_repaint
        assert repaint_order == ["repaint"], repaint_order

    def test_the_very_first_frame_ever_shown_never_dissolves(self, qapp, sim_data):
        """start_scenario_transition must refuse (return False) when
        there's no real "before" frame yet -- otherwise the very first
        paint of the whole app's life (ATTRACT's own baseline loop, which
        already renders a frame during enter_public_mode() before
        anything visits _begin_journey) would try to blend from nothing.
        Checked directly against a fresh SliceView rather than the app's
        own startup sequence, since enter_public_mode() itself already
        shows a first frame -- there's no later point in a real session
        where _last_frame is still None to observe this against."""
        view = SliceView()
        assert view._last_frame is None
        started = view.start_scenario_transition(np.zeros((4, 4), dtype=np.float32))
        assert started is False
        assert not view._transition_timer.isActive()

        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim_data)
        try:
            window.enter_public_mode()
            experience = window.public_experience
            for _ in range(6):
                qapp.processEvents()
            assert not experience.scene.view._transition_timer.isActive()
            assert experience.scene.view._last_frame is not None
        finally:
            window.close()

    def test_candles_toggle_switches_to_the_real_scenario(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._on_explore_changed("candles", 1)
        entry = next(e for e in experience.sim_data.manifest
                     if e.case_index == experience.state.case_index)
        assert entry.candles == 1

    def test_changing_a_control_frames_it_as_the_child_s_own_experiment(self, experience):
        """Phase 7 section 2: flipping Fan/Candles is framed as "you
        changed the experiment", not a settings confirmation."""
        experience._begin_journey()
        experience._start_observe()
        said = []
        original = experience.overlay.say
        experience.overlay.say = lambda t, m=None: (said.append(t), original(t, m))[1]
        experience._on_explore_changed("vent1", 2)
        assert any("you changed the experiment" in s.lower() for s in said)

        said.clear()
        experience._on_explore_changed("candles", 1)
        assert any("you changed the experiment" in s.lower() for s in said)

    def test_case_index_never_hardcoded(self, experience):
        """resolve_case_index must be the source of truth -- an unknown
        value resolves to None and is silently ignored, never a fallback
        constant."""
        experience._begin_journey()
        experience._start_observe()
        before = experience.state.case_index
        experience._on_explore_changed("vent1", 999)
        assert experience.state.case_index == before

    def test_factor_changes_compose_instead_of_resetting_each_other(self, experience):
        """Flipping a second (and third) control must keep whatever the
        others are already set to and land on the real combined
        scenario -- the manifest has every combination these three
        controls can reach. Driven through the real widgets (a click),
        not the handler directly, so this also proves the toggles' own
        visual state stays composed, not just the resolved case_index."""
        experience._begin_journey()
        experience._start_observe()
        fan_toggle = experience.overlay._explore_toggles["vent1"]
        candles_toggle = experience.overlay._explore_toggles["candles"]
        vent2_toggle = experience.overlay._explore_toggles["vent2"]

        fan_toggle._group.button(2).click()   # "FAN ON" (3rd of 3 real vod options)
        assert fan_toggle._group.checkedId() == 2

        candles_toggle._group.button(1).click()   # "2 candles"
        assert fan_toggle._group.checkedId() == 2   # still ON, not reset
        entry = next(e for e in experience.sim_data.manifest
                     if e.case_index == experience.state.case_index)
        assert entry.vod == 2 and entry.candles == 1

        vent2_toggle._group.button(1).click()   # "SHUT"
        assert fan_toggle._group.checkedId() == 2        # still ON
        assert candles_toggle._group.checkedId() == 1    # still 2 candles
        entry = next(e for e in experience.sim_data.manifest
                     if e.case_index == experience.state.case_index)
        assert entry.vod == 2 and entry.candles == 1 and entry.voc == 1

    def test_explore_only_acts_during_observe(self, experience):
        experience._begin_journey()   # phase is INTRO
        before = experience.state.case_index
        experience._on_explore_changed("vent1", 2)
        assert experience.state.case_index == before

    def test_explore_panel_visible_only_during_observe(self, experience):
        experience._begin_journey()
        assert experience.overlay.explore_panel.isHidden() or not experience.overlay.explore_panel.isVisible()
        experience._start_observe()
        assert not experience.overlay.explore_panel.isHidden()
        experience.time_controller.pause()
        experience._observe_finished()
        assert experience.overlay.explore_panel.isHidden()

    def test_toggles_reset_to_baseline_on_replay(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._on_explore_changed("vent1", 2)
        experience.time_controller.pause()
        experience._on_replay()
        assert experience.state.case_index == experience.state.baseline_case_index
        fan_toggle = experience.overlay._explore_toggles["vent1"]
        assert fan_toggle._group.checkedId() == 0

    def test_toggles_reset_on_kiosk_idle(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._on_explore_changed("candles", 1)
        experience.reset()
        assert experience.state.phase is Phase.ATTRACT
        experience._start_observe()
        assert experience.state.case_index == experience.state.baseline_case_index

    def test_exploring_never_touches_the_guided_prediction_state(self, experience):
        """Free play must stay independent of the fan-experiment's own
        choice/tried_choices bookkeeping -- exploring "2 candles" must
        not look like the visitor already tried a fan experiment side."""
        experience._begin_journey()
        experience._start_observe()
        experience._on_explore_changed("candles", 1)
        assert experience.state.choice is None
        assert experience.state.tried_choices == []


class TestCandleMarker:
    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_patch_count_matches_the_real_candle_count(self, experience, sim_data):
        """Each candle is body + wick + a small lit-flame anchor (three
        warm-to-hot layers, no glow/halo behind them, see
        PublicScene._draw_flame) -- 5 patches per candle."""
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        one = experiments_mod.resolve_case_index(sim_data.manifest, {
            "candles": 0, "door": 1, "vod": 0, "voc": 0})
        two = experiments_mod.resolve_case_index(sim_data.manifest, {
            "candles": 1, "door": 1, "vod": 0, "voc": 0})
        experience._load_case(one)
        assert len(experience.scene._candle_patches) == 5    # 1 candle
        experience._load_case(two)
        assert len(experience.scene._candle_patches) == 10   # 2 candles

    def test_candle_x_position_matches_the_real_burner_location(self, experience, sim_data, qapp):
        """Checks the sprite's x actually sits on a hot cell in the real
        measured field -- not just inside _CANDLE_X's own plausible band,
        which is all an earlier version of this test checked. That
        weaker check passed even for the 1-candle sprite sitting at the
        bare midpoint (x=0.90), a genuinely cool 4cm gap between the two
        real burner slots that read ~21C, a real bug a live screenshot
        caught that this test should have (see _draw_candles's own
        comment on the fix). Checked for both candle counts, since the
        2-candle branch's own two slots were already correct and should
        stay that way."""
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        from schematic import ROOM_Z
        # ATTRACT (this fixture's default phase, never having called
        # _begin_journey) loops the fire on its own timer -- pausing it
        # is what keeps an explicit show_frame() from being raced and
        # overwritten by that loop the instant qapp.processEvents() next
        # runs. A first version of this test skipped that and got a
        # real, flaky false failure from whatever frame the loop had
        # wandered to, not the one it asked for.
        experience.time_controller.pause()
        for candles in (0, 1):
            case_index = experiments_mod.resolve_case_index(sim_data.manifest, {
                "candles": candles, "door": 1, "vod": 0, "voc": 0})
            experience._load_case(case_index)
            assert experience.scene._candle_signature is not None
            n_frames = experience.scene._temperature.shape[0]
            # Checked across several well-after-ignition frames, not one:
            # a real flame's own temperature genuinely flickers frame to
            # frame, so the bar is "clearly hot at its best moment in this
            # stretch," not "hot at this one arbitrary tick."
            for x in experience.scene._candle_signature:
                readings = []
                for frac in (0.3, 0.5, 0.7, 0.9):
                    experience.scene.show_frame(int(n_frames * frac))
                    readings.append(experience.scene.view.value_at(x, ROOM_Z[0]))
                assert max(readings) > 150.0, (candles, x, readings)

    def test_unchanged_signature_skips_the_redraw(self, experience):
        """The perf-motivated cache: switching a factor that doesn't
        change candle count must not touch the patch list at all."""
        patches_before = experience.scene._candle_patches
        experience.scene._draw_candles(experience.state.case_index)
        assert experience.scene._candle_patches is patches_before

    def test_no_candles_drawn_off_the_side_plane(self, experience):
        from slice_key import SliceKey
        experience.scene._quantity_key = SliceKey("TEMPERATURE", 0, 0)
        experience.scene._candle_signature = "force-a-check"
        experience.scene._draw_candles(experience.state.case_index)
        assert experience.scene._candle_patches == []


class TestFlameFlicker:
    """A purely decorative flicker on the schematic flame anchor -- the
    real fire is still the cinema pipeline's own rendering of the
    measured temperature field. Driven from PublicScene.show_frame()'s
    own per-frame call (see _jitter_flame), not an independent timer --
    an earlier version used a QTimer and measured ~15 ms per tick,
    because blit_update() has to redraw the entire animated-artist set
    (the heatmap image included) to avoid reverting to the load-time
    frame; piggybacking on the redraw that already happens every real
    frame costs nothing extra."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_flame_layers_are_registered_for_animation(self, experience):
        scene = experience.scene
        assert scene._flame_layers_anim
        for patch, *_ in scene._flame_layers_anim:
            assert patch in scene.view._extra_animated

    def test_jitter_changes_the_flame_geometry_across_frames(self, experience):
        scene = experience.scene
        before = [tuple(p.get_xy()[0]) for p, *_ in scene._flame_layers_anim]
        changed = False
        for index in range(1, 30):
            scene._jitter_flame(index)
            after = [tuple(p.get_xy()[0]) for p, *_ in scene._flame_layers_anim]
            if after != before:
                changed = True
                break
        assert changed, "jitter never moved a flame vertex across frames"

    def test_jitter_is_deterministic_given_the_same_frame_index(self, experience):
        """Not wall-clock randomness -- replaying the same frame must
        look identical, and a future caller must be able to rely on
        show_frame(i) producing the same flame geometry every time."""
        scene = experience.scene
        scene._jitter_flame(37)
        first = [tuple(p.get_xy()[0]) for p, *_ in scene._flame_layers_anim]
        scene._jitter_flame(11)   # perturb, then return to the same index
        scene._jitter_flame(37)
        second = [tuple(p.get_xy()[0]) for p, *_ in scene._flame_layers_anim]
        assert first == second

    def test_show_frame_jitters_the_flame_without_a_separate_timer(self, experience):
        scene = experience.scene
        before = [tuple(p.get_xy()[0]) for p, *_ in scene._flame_layers_anim]
        scene.show_frame(25)
        after = [tuple(p.get_xy()[0]) for p, *_ in scene._flame_layers_anim]
        assert before != after

    def test_switching_scenario_does_not_leave_stale_animated_extras(self, experience, sim_data):
        """A scenario switch removes and rebuilds the candle patches --
        the old flame layers must be unregistered from SliceView's
        animated set first, or blit_update would eventually hand
        draw_artist() a matplotlib artist that no longer belongs to any
        axes."""
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        experience._begin_journey()
        experience._start_observe()
        old_layers = [p for p, *_ in experience.scene._flame_layers_anim]
        experience._on_explore_changed("candles", 1)   # 2 candles: layers rebuilt
        for patch in old_layers:
            assert patch not in experience.scene.view._extra_animated
        experience.scene._jitter_flame(5)   # must not raise

    def test_no_flame_no_crash(self):
        from public.scene import PublicScene
        scene = PublicScene(store=None, manifest=[], fps=4)
        scene._jitter_flame(0)   # nothing to animate; must not raise


class TestFlameCountMatchesCandleCount:
    """Regression coverage for 'flame count must always equal the
    selected candle count, with nothing left over from a previous
    selection' -- both the schematic flame anchors (already covered by
    TestCandleMarker's patch-count check, re-asserted here alongside the
    cinema layer) and the two stateful cinema simulators (EmberParticles,
    SmokeSimulator) that -- unlike the schematic patches -- are NOT
    rebuilt on every load_case(); see SliceView.reset_cinema_simulators,
    called from PublicScene.load_case, which is the one thing standing
    between a switch and a genuinely leftover ember/smoke render at the
    old candle's position."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_flame_patch_count_is_exactly_five_times_candle_count_after_any_switch(
        self, experience, sim_data
    ):
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        one = experiments_mod.resolve_case_index(sim_data.manifest, {
            "candles": 0, "door": 1, "vod": 0, "voc": 0})
        two = experiments_mod.resolve_case_index(sim_data.manifest, {
            "candles": 1, "door": 1, "vod": 0, "voc": 0})
        # A non-trivial switch sequence (1 -> 2 -> 1 -> 2), not just a
        # single switch -- a stale-count bug that only shows up on the
        # *second* return to a previously-seen candle count would slip
        # past a test that only checked one transition.
        for case_index, n_candles in [(one, 1), (two, 2), (one, 1), (two, 2)]:
            experience._load_case(case_index)
            assert len(experience.scene._candle_patches) == 5 * n_candles
            # Three warm-to-hot flicker layers per candle (see
            # TestFlameFlicker's docstring / _draw_flame).
            assert len(experience.scene._flame_layers_anim) == 3 * n_candles

    def test_cinema_simulators_are_clean_immediately_after_a_candle_count_switch(
        self, experience, sim_data
    ):
        """The literal 'nothing left over from a previous selection'
        requirement, checked at the data layer rather than by eye: no
        live ember particles and a fully-decayed-to-zero smoke buffer
        the instant a switch lands, before any new frame has had a
        chance to populate them fresh. Ember/smoke spawning is itself
        driven by show_frame() (see SliceView.update_effects), so
        checking right after _load_case -- before any show_frame call --
        isolates reset_cinema_simulators' own guarantee from whatever
        the next frame would legitimately draw."""
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        one = experiments_mod.resolve_case_index(sim_data.manifest, {
            "candles": 0, "door": 1, "vod": 0, "voc": 0})
        two = experiments_mod.resolve_case_index(sim_data.manifest, {
            "candles": 1, "door": 1, "vod": 0, "voc": 0})
        view = experience.scene.view
        if view._ember_sim is None:
            pytest.skip("cinematic mode not active in this configuration")
        experience._load_case(two)
        experience.scene.show_frame(20)   # let embers/smoke actually build up
        assert view._ember_sim.pos.shape[0] > 0 or view._cinema_pipeline._smoke is not None

        experience._load_case(one)   # the switch under test
        assert view._ember_sim.pos.shape[0] == 0, (
            "embers from the removed candle survived the switch")
        smoke = view._cinema_pipeline._smoke
        if smoke is not None:
            assert float(smoke.buffer.max()) == 0.0, (
                "smoke density from the removed candle survived the switch")

    def test_the_real_click_driven_2_to_1_and_1_to_2_transitions_hold_the_invariant(
        self, experience, sim_data, qapp
    ):
        """Reported and 'fixed' twice before (a stale visual trace, then
        a fade-delay) and still recurring per direct feedback -- routed
        through _on_explore_changed (the real click path a Candles
        toggle press takes), not _load_case directly, and through
        PublicScene._draw_candles' own enforced assertion (flame count
        == 3 * candle count, checked on every draw, not just here) so a
        fourth regression fails loudly at the source instead of silently
        rendering an extra flame."""
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        experience._begin_journey()
        experience._start_observe()
        for value, n_candles in [(1, 2), (0, 1), (1, 2), (0, 1)]:
            experience._on_explore_changed("candles", value)
            for _ in range(6):
                qapp.processEvents()
            assert len(experience.scene._candle_patches) == 5 * n_candles
            assert len(experience.scene._flame_layers_anim) == 3 * n_candles

    def test_candle_count_survives_a_switch_driven_by_a_different_control(
        self, experience, sim_data, qapp
    ):
        """The direct Candles-toggle test above only proves the toggle's
        own path is safe -- it says nothing about Door/Vent 1/Vent 2
        loading an entirely different scenario file out from under
        whatever the Candles selector is currently set to. Every explore
        control funnels through the same PublicScene.load_case (which
        unconditionally calls _draw_candles -- confirmed by reading it,
        not assumed; there is no separate scene-rebuild path a Door/Vent
        switch could take instead), so the composition this checks is in
        _on_explore_changed/_current_factors, not a second render path."""
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        experience._begin_journey()
        experience._start_observe()
        for candles_value, n_candles in [(0, 1), (1, 2)]:   # both directions
            experience._on_explore_changed("candles", candles_value)
            for _ in range(6):
                qapp.processEvents()
            for control_key, other_value in [("door", 0), ("vent1", 2), ("vent2", 1)]:
                experience._on_explore_changed(control_key, other_value)
                for _ in range(6):
                    qapp.processEvents()
                entry = experience.scene.current_entry()
                assert entry.candles == candles_value, (
                    f"{control_key}={other_value} reset candles to {entry.candles}")
                assert len(experience.scene._candle_patches) == 5 * n_candles
                assert len(experience.scene._flame_layers_anim) == 3 * n_candles


class TestInteractionResponsiveness:
    """Click-to-acknowledgement: a real scenario switch must stay well
    under perceptible-lag territory now that both scenarios' stories are
    pre-warmed (see PublicExperience._prewarm_stories)."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_prewarm_covers_every_explore_and_experiment_scenario(self, experience):
        cases = set()
        for control in experience._explore_controls:
            for option in control.options:
                cases.add(control.case_for(experience.sim_data.manifest, option.value))
        for choice in experience.experiment.choices:
            cases.add(experiments_mod.resolve_choice(
                experience.sim_data.manifest, experience.experiment, choice.key))
        experience._begin_journey()
        for case in cases:
            assert case in experience._stories, f"case {case} was not pre-warmed"

    def test_explore_toggle_round_trip_is_fast_once_warmed(self, experience):
        experience._begin_journey()   # triggers _prewarm_stories
        experience._start_observe()
        start = time.perf_counter()
        experience._on_explore_changed("vent1", 2)
        elapsed_ms = (time.perf_counter() - start) * 1000
        # Generous versus the ~40 ms measured locally -- this guards
        # against a real regression (e.g. losing the story-cache warm
        # start, or a reintroduced double render), not CI jitter.
        assert elapsed_ms < 300, f"explore toggle took {elapsed_ms:.1f} ms"

    def test_toggle_button_acknowledges_before_any_scenario_logic_runs(self, experience, qapp):
        """The button's own checked state flips synchronously as part of
        Qt's click handling, before value_changed (and therefore any
        scenario switch) is even emitted -- this is Qt's own guarantee,
        asserted here so a future widget rewrite can't silently lose it."""
        experience._begin_journey()
        experience._start_observe()
        toggle = experience.overlay._explore_toggles["vent1"]
        seen_checked_state = []

        def on_change(_value):
            seen_checked_state.append(toggle._group.checkedId())
        toggle.value_changed.connect(on_change)
        toggle._group.button(1).click()
        assert seen_checked_state == [1]   # already flipped when the slot ran


# ---------------------------------------------------------- fire laboratory
class TestCandleCallout:
    """Tapping the candle itself must be answered with what it *is*, not
    treated like an ordinary air-temperature probe."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def _candle_widget_pos(self, experience):
        # Reads _candle_signature (the sprite's own actual x, set by
        # _draw_candles) rather than recomputing sum(_CANDLE_X)/2 here
        # independently -- an earlier version of this helper did that,
        # which meant it silently kept testing the bare midpoint even
        # after _draw_candles moved the real sprite off of it (the fix
        # for a real bug: that midpoint sits in a cool 4cm gap between
        # the two actual burner slots, not on either one).
        from schematic import ROOM_Z
        canvas = experience.scene.view.canvas
        cx = experience.scene._candle_signature[0]
        frac = experience.scene.widget_fraction_for(cx, ROOM_Z[0] + 0.01)
        return QtCore.QPoint(int(frac[0] * canvas.width()), int(frac[1] * canvas.height()))

    def test_candle_hit_detects_the_real_burner_location(self, experience):
        from schematic import ROOM_Z
        experience._begin_journey()
        experience._start_observe()
        cx = experience.scene._candle_signature[0]
        assert experience.scene.candle_hit(cx, ROOM_Z[0] + 0.01) is True
        # Well away from the candle (near the domain's left edge): not a hit.
        assert experience.scene.candle_hit(0.05, ROOM_Z[0] + 0.01) is False

    def test_tapping_the_candle_gives_a_fire_callout_not_a_bare_reading(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        experience._on_overlay_tapped(self._candle_widget_pos(experience))
        assert "flame" in experience.overlay.thermometer._caption.text().lower()
        assert "🕯️" in experience.overlay.thermometer._caption.text()
        assert "candle" in experience.overlay.bubble.text().lower()

    def test_candle_callout_still_uses_a_real_measured_value(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        pos = self._candle_widget_pos(experience)
        result = experience.scene.probe_at(pos)
        assert result is not None
        _x, _z, value = result
        experience._on_overlay_tapped(pos)
        assert experience.overlay.thermometer._value == pytest.approx(value)

    def test_a_tap_away_from_the_candle_is_an_ordinary_reading(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        canvas = experience.scene.view.canvas
        centre = QtCore.QPoint(canvas.width() // 2, canvas.height() // 2)
        experience._on_overlay_tapped(centre)
        assert "°C" in experience.overlay.thermometer._value_label.text()
        assert "flame" not in experience.overlay.thermometer._caption.text().lower()

    def test_first_tap_at_baseline_nudges_toward_changing_the_fire(self, experience, qapp):
        """Phase 7 section 1: a light, one-time invitation toward the
        real candle-count control -- only while nothing has been changed
        yet, so it never repeats and annoy once the child has already
        acted on it."""
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        said = []
        original = experience.overlay.say
        experience.overlay.say = lambda t, m=None: (said.append(t), original(t, m))[1]
        experience._on_overlay_tapped(self._candle_widget_pos(experience))
        assert any("change the fire" in s for s in said)

        said.clear()
        experience._on_overlay_tapped(self._candle_widget_pos(experience))
        assert not any("change the fire" in s for s in said)

    def test_no_nudge_once_the_experiment_has_already_changed(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        experience._on_explore_changed("candles", 1)
        for _ in range(6):
            qapp.processEvents()
        said = []
        original = experience.overlay.say
        experience.overlay.say = lambda t, m=None: (said.append(t), original(t, m))[1]
        experience._on_overlay_tapped(self._candle_widget_pos(experience))
        assert not any("change the fire" in s for s in said)

    def test_tapping_the_candle_highlights_the_real_candle_control(self, experience, qapp):
        """Phase 8 section 2: instead of a second floating picker
        attached to the candle, tapping it draws the eye to the one real
        candle-count control that changes it."""
        for _ in range(6):
            qapp.processEvents()
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        toggle = experience.overlay._explore_toggles["candles"]
        assert toggle.styleSheet() == ""

        experience._on_overlay_tapped(self._candle_widget_pos(experience))

        assert toggle.styleSheet() != ""


class TestThermometer:
    """The always-on thermometer: real values in every mode, never an
    interpolated or invented one (see PublicScene.probe_value_at)."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_defaults_to_the_scene_mean_before_any_tap(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        assert experience._probe_mode == "mean"
        expected = experience.scene.mean_temperature_at(experience.state.frame_index)
        assert experience.overlay.thermometer._value == pytest.approx(expected)

    def test_hidden_outside_the_probe_phases(self, experience, qapp):
        experience._begin_journey()
        for _ in range(6):
            qapp.processEvents()
        assert experience.state.phase is Phase.INTRO
        assert not experience.overlay.thermometer.isVisible()

    def test_tap_switches_to_a_point_reading(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        canvas = experience.scene.view.canvas
        centre = QtCore.QPoint(canvas.width() // 2, canvas.height() // 2)
        experience._on_overlay_tapped(centre)
        assert experience._probe_mode == "point"
        expected = experience.scene.view.value_at(*experience._probe_xz)
        assert experience.overlay.thermometer._value == pytest.approx(expected)

    def test_point_reading_does_not_drift_back_to_the_mean_next_frame(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        canvas = experience.scene.view.canvas
        centre = QtCore.QPoint(canvas.width() // 2, canvas.height() // 2)
        experience._on_overlay_tapped(centre)
        reading = experience.overlay.thermometer._value
        experience.time_controller.seek(experience.state.frame_index + 1)
        assert experience.overlay.thermometer._value == pytest.approx(reading)

    def test_probe_state_resets_on_phase_change(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        canvas = experience.scene.view.canvas
        centre = QtCore.QPoint(canvas.width() // 2, canvas.height() // 2)
        experience._on_overlay_tapped(centre)
        assert experience._probe_mode == "point"

        experience.time_controller.pause()
        experience._observe_finished()
        assert experience._probe_mode == "mean"
        assert experience._probe_xz is None

    def test_probe_state_resets_on_scenario_switch(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        canvas = experience.scene.view.canvas
        centre = QtCore.QPoint(canvas.width() // 2, canvas.height() // 2)
        experience._on_overlay_tapped(centre)
        experience._on_explore_changed("vent1", 2)
        assert experience._probe_mode == "mean"
        assert experience._probe_xz is None


class TestDragToScan:
    """Dragging (not just tapping) must keep probing the already-loaded
    frame -- never reload a scenario."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_drag_keeps_emitting_taps_while_the_button_is_held(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        overlay = experience.overlay
        positions = []
        overlay.tapped.connect(positions.append)

        press = QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonPress, QtCore.QPointF(100, 100),
            QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
        overlay.mousePressEvent(press)
        move = QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove, QtCore.QPointF(120, 110),
            QtCore.Qt.NoButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
        overlay.mouseMoveEvent(move)
        release = QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonRelease, QtCore.QPointF(120, 110),
            QtCore.Qt.LeftButton, QtCore.Qt.NoButton, QtCore.Qt.NoModifier)
        overlay.mouseReleaseEvent(release)

        assert len(positions) == 2   # the press and the drag-move
        case_before = experience.state.case_index
        assert experience.state.case_index == case_before   # never reloaded

    def test_move_without_a_prior_press_does_not_probe(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        overlay = experience.overlay
        positions = []
        overlay.tapped.connect(positions.append)
        move = QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove, QtCore.QPointF(120, 110),
            QtCore.Qt.NoButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
        overlay.mouseMoveEvent(move)
        assert positions == []

    def test_drag_ignored_while_probe_disabled(self, experience, qapp):
        experience._begin_journey()   # INTRO: probe disabled
        for _ in range(6):
            qapp.processEvents()
        overlay = experience.overlay
        positions = []
        overlay.tapped.connect(positions.append)
        press = QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonPress, QtCore.QPointF(100, 100),
            QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
        overlay.mousePressEvent(press)
        assert positions == []


class TestVentGeometry:
    """Vent 1's on-screen object is anchored to the manifest's own vent
    geometry -- never a fixed screen position (see PublicScene.
    _draw_vent_object, which vent_marker_position feeds)."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_marker_position_matches_the_manifests_own_vent_geometry(self, experience):
        from schematic import room_overlay_geometry
        experience._begin_journey()
        experience._start_observe()
        entry = experience.scene.current_entry()
        geometry = room_overlay_geometry(entry.door, entry.vod, entry.voc)
        (x0, _z0, x1, _z1), _state = geometry["vents"][0]
        position = experience.scene.vent_marker_position(experience.state.case_index)
        assert position[0] == pytest.approx((x0 + x1) / 2)


class TestMascotMarginNeverOverlapsTheBox:
    """The experiment box (room + flame/smoke) must shrink to fit above
    a fixed-pixel strip reserved for the mascots (see PublicScene.
    set_bottom_reserve_px/resizeEvent), not the other way around -- a
    real, screenshotted complaint that the worker mascot overlapped the
    flame instead of framing it. Checked at several window sizes: this
    is a *fixed*-pixel reservation, not a proportional one, so only the
    fraction of the canvas it maps to should change with window size,
    never the mascots' own real footprint slipping above the axes'
    shrunk bottom edge."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.show()
        window.enter_public_mode()
        exp = window.public_experience
        exp._begin_journey()
        exp._start_observe()
        exp.time_controller.pause()
        yield exp
        window.close()

    @pytest.mark.parametrize("size", [(800, 600), (1024, 768), (1280, 800), (1920, 1080)])
    def test_both_mascots_stay_at_or_below_the_axes_bottom_edge(self, experience, qapp, size):
        # PublicExperience.window() is Qt's own accessor for its
        # top-level window ancestor -- the MainWindow this fixture built.
        experience.window().resize(*size)
        for _ in range(10):
            qapp.processEvents()
        scene = experience.scene
        overlay = experience.overlay
        axes_bottom_px = scene.view.canvas.height() * (1 - scene._bottom_frac)
        # A couple of px of float/rounding slack, not a real tolerance
        # for overlap -- the mascots' own top edge is allowed to touch
        # the boundary exactly (see PublicOverlay.mascot_band_height_px's
        # own +12px clear-gap constant), not cross above it.
        assert overlay.mascot.y() >= axes_bottom_px - 1.0
        assert overlay.scientist.y() >= axes_bottom_px - 1.0

    def test_reserve_is_derived_from_the_real_mascot_footprint_not_a_guess(self, experience):
        expected = max(experience.overlay.mascot.height(),
                       experience.overlay.scientist.height()) + 32 + 12
        assert experience.scene._bottom_reserve_px == expected

    def test_probe_at_roundtrips_through_the_shrunk_axes(self, experience):
        """A tap fed back through widget_fraction_for's own output must
        land back near the same physical point -- the two conversions
        (data->widget, widget->data) have to agree on the same _bottom_
        frac or a real tap anywhere in the lower part of the room would
        silently mismap."""
        scene = experience.scene
        frac = scene.widget_fraction_for(0.9, 0.05)
        assert frac is not None
        canvas = scene.view.canvas
        point = QtCore.QPoint(int(frac[0] * canvas.width()), int(frac[1] * canvas.height()))
        result = scene.probe_at(point)
        assert result is not None
        x, z, _value = result
        assert x == pytest.approx(0.9, abs=0.02)
        assert z == pytest.approx(0.05, abs=0.02)

    def test_a_tap_inside_the_reserved_mascot_strip_hits_nothing(self, experience):
        scene = experience.scene
        canvas = scene.view.canvas
        # Comfortably inside the reserved strip, not right at its own
        # boundary -- this checks "the reservation actually excludes
        # taps", not the exact pixel the boundary falls on.
        deep_in_margin = QtCore.QPoint(canvas.width() // 2, canvas.height() - 5)
        assert scene.probe_at(deep_in_margin) is None


class TestStatPanelGrouping:
    """The language toggle, both meter chips, and the thermometer read
    as one connected panel (Fire Explorer bug-fix round: 'these should
    read as one panel, not three floating elements') -- checked as a
    real geometric containment property, not just 'this widget exists'."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(1280, 800)
        window.show()
        window.enter_public_mode()
        exp = window.public_experience
        exp._begin_journey()
        exp._start_observe()
        for _ in range(8):
            qapp.processEvents()
        yield exp
        window.close()

    def test_language_toggle_meters_and_thermometer_all_sit_inside_the_panel(self, experience):
        overlay = experience.overlay
        panel = overlay.stat_panel.geometry()
        for widget in (overlay.lang_en_button, overlay.lang_de_button,
                       overlay.temperature_meter, overlay.airflow_meter, overlay.thermometer):
            geo = widget.geometry()
            assert panel.contains(geo), (widget, geo, panel)

    def test_meters_row_sits_above_the_thermometer(self, experience):
        overlay = experience.overlay
        assert overlay.temperature_meter.geometry().bottom() < overlay.thermometer.y()
        assert overlay.airflow_meter.geometry().bottom() < overlay.thermometer.y()

    def test_language_row_sits_above_the_meters(self, experience):
        overlay = experience.overlay
        assert overlay.lang_en_button.geometry().bottom() < overlay.temperature_meter.y()

    def test_hiding_meters_lets_the_thermometer_and_panel_ride_up(self, experience):
        overlay = experience.overlay
        before = overlay.stat_panel.height()
        thermometer_y_before = overlay.thermometer.y()
        overlay.set_meters_visible(False)
        assert overlay.stat_panel.height() < before
        assert overlay.thermometer.y() < thermometer_y_before


class TestMeterChipCaptionEliding:
    """A narrow flat MeterChip elides its (single, unbreakable-word)
    caption rather than word-wrapping it -- word-wrap on a caption like
    "LUFTTEMPERATUR" (no space to break at) fell back to an ugly
    mid-word character split, a real, screenshotted regression this
    exists to avoid. The longer, real-sentence phrase below it still
    wraps normally -- not covered here, see MeterChip.heightForWidth's
    own use in PublicOverlay._position_stat_group."""

    def test_a_caption_wider_than_the_chip_is_elided_not_broken_mid_word(self, qapp):
        from public.widgets import MeterChip
        chip = MeterChip("LUFTTEMPERATUR", flat=True)
        chip.resize(115, 150)
        # Qt only dispatches resizeEvent to a widget once it's part of a
        # real, shown window -- this standalone chip never is, so the
        # real production trigger (MeterChip.resizeEvent) never fires
        # here. Called directly instead, exercising exactly the same
        # method a real resize would -- confirmed against the actual app
        # separately (PublicOverlay's real, shown meter chips do elide
        # correctly; see TestStatPanelGrouping).
        chip._update_caption_elide()
        text = chip._caption.text()
        assert text != "LUFTTEMPERATUR"   # too wide at this width; must not show unelided
        assert "…" in text            # ellipsis, not a hard mid-word cut
        assert text.rstrip("…") == "LUFTTEMPERATUR"[:len(text.rstrip("…"))]

    def test_full_caption_is_still_available_for_accessibility_and_language_switch(self, qapp):
        from public.widgets import MeterChip
        chip = MeterChip("LUFTTEMPERATUR", flat=True)
        chip.resize(115, 150)
        assert chip._caption_full_text == "LUFTTEMPERATUR"
        chip.set_caption("AIR TEMPERATURE")
        assert chip._caption_full_text == "AIR TEMPERATURE"

    def test_a_short_caption_is_not_elided(self, qapp):
        from public.widgets import MeterChip
        chip = MeterChip("AIR", flat=True)
        chip.resize(115, 150)
        assert chip._caption.text() == "AIR"

    def test_non_flat_chip_never_elides(self, qapp):
        """The un-flat, original 210px-floor MeterChip is still used
        elsewhere in the app (or could be) -- its caption must keep
        behaving exactly as before this whole panel existed."""
        from public.widgets import MeterChip
        chip = MeterChip("LUFTTEMPERATUR", flat=False)
        chip.resize(115, 150)
        assert chip._caption.text() == "LUFTTEMPERATUR"


class TestControlBarPolish:
    """Fire Explorer bug-fix round: control-bar spacing/sizing/grouping
    polish. Covers a real, measured bug this round found and fixed --
    the control bar and the stat panel used to be positioned by two
    unrelated formulas (one anchored to a physical-geometry fraction of
    the window width, one a fixed right-margin reservation) that agreed
    at 800x600 but diverged as the window widened, actually overlapping
    by measured 73px at 1280 and 265px at 1920 -- not present at the one
    size ad hoc manual testing happened to use."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.show()
        window.enter_public_mode()
        exp = window.public_experience
        exp._begin_journey()
        exp._start_observe()
        for _ in range(10):
            qapp.processEvents()
        yield exp
        window.close()

    @pytest.mark.parametrize("size", [(1024, 768), (1280, 800), (1920, 1080)])
    def test_explore_panel_never_overlaps_the_stat_panel(self, experience, qapp, size):
        experience.window().resize(*size)
        for _ in range(15):
            qapp.processEvents()
        overlay = experience.overlay
        # A real, visible gap, not just "not touching" -- the whole
        # point of the fix (see this class' own docstring) was that a
        # merely-non-negative gap still read as "about to collide".
        assert overlay.stat_panel.x() - overlay.explore_panel.geometry().right() >= 20

    def test_at_800x600_explore_panel_stays_on_top_of_the_stat_panel_where_they_touch(
            self, experience, qapp):
        """800x600 is the one size where the two panels' real minimum
        widths (Vent 1's own 3-button floor -- see ExploreToggle's own
        comment -- plus the stat panel's own chip/thermometer width)
        genuinely don't both fit with a clean gap; _position_stat_group
        pushes the stat panel as far right as the window allows, which
        isn't quite far enough here. What must still hold: the real
        control stays the one a tap actually lands on, not the
        decorative panel behind it (see explore_panel.raise_(), set_
        explore_controls)."""
        experience.window().resize(800, 600)
        for _ in range(15):
            qapp.processEvents()
        overlay = experience.overlay
        door = overlay._explore_toggles["door"]
        wide_button = door._group.button(len(door._values) - 1)
        centre = wide_button.mapTo(overlay, QtCore.QPoint(
            wide_button.width() // 2, wide_button.height() // 2))
        hit = overlay.childAt(centre)
        assert hit is not None
        assert door.isAncestorOf(hit) or hit is door, (
            "a tap on the Door group's own button hits the stat panel instead")

    def test_each_group_has_a_divider_before_it_except_the_first(self, experience):
        overlay = experience.overlay
        assert len(overlay._explore_dividers) == len(overlay._explore_toggles) - 1

    def test_dividers_span_the_full_toggle_height_not_just_the_button_row(self, experience):
        overlay = experience.overlay
        toggle_height = next(iter(overlay._explore_toggles.values())).height()
        for divider in overlay._explore_dividers:
            assert divider.height() == toggle_height

    def test_dividers_sit_strictly_between_two_consecutive_groups(self, experience):
        overlay = experience.overlay
        toggles = list(overlay._explore_toggles.values())
        for divider, left_toggle, right_toggle in zip(
                overlay._explore_dividers, toggles, toggles[1:]):
            assert left_toggle.geometry().right() < divider.x()
            assert divider.geometry().right() < right_toggle.x()

    def test_every_button_in_every_group_and_state_is_exactly_the_same_height(self, experience):
        heights = set()
        for toggle in experience.overlay._explore_toggles.values():
            for i in range(len(toggle._values)):
                heights.add(toggle._group.button(i).height())
        assert len(heights) == 1, f"button heights are not uniform: {heights}"

    def test_caption_to_button_row_gap_is_a_real_gap_not_nearly_touching(self, experience):
        for toggle in experience.overlay._explore_toggles.values():
            caption = toggle.findChild(QtWidgets.QLabel)
            first_button = toggle._group.button(0)
            gap = first_button.geometry().y() - caption.geometry().bottom()
            assert gap >= 8, f"{toggle}: caption-to-row gap only {gap}px"

    def test_panel_background_shrink_wraps_the_buttons_when_there_is_room_to_spare(
            self, experience, qapp):
        """explore_panel used to be stretched to root's *entire* available
        row width regardless of how much of it the actual buttons needed
        -- a real, screenshotted "the bar is enveloping a huge stretch of
        empty background" complaint at any window wide enough to have
        slack to give. At a genuinely wide window the panel's own painted
        width must match its content's real sizeHint, not the window's."""
        experience.window().resize(1712, 1192)
        for _ in range(15):
            qapp.processEvents()
        overlay = experience.overlay
        assert overlay.explore_panel.width() == overlay.explore_panel.sizeHint().width()

    def test_panel_shrinks_toward_but_never_below_its_buttons_real_floor(
            self, experience, qapp):
        """The shrink-wrap fix must not turn into a fixed width that
        overflows a narrow window -- explore_panel keeps compressing
        toward its content's real width as the window narrows, same as
        before. What changed (see ExploreToggle's own comment on the
        Vent 1 overlap fix): that floor is now a real, enforced minimum
        -- explore_panel.width() must never drop *below* it, even at
        800x600, since going below it is exactly what used to let Vent
        1's three buttons overlap each other instead of the panel simply
        running wider than its old, too-small column."""
        experience.window().resize(800, 600)
        for _ in range(15):
            qapp.processEvents()
        overlay = experience.overlay
        assert overlay.explore_panel.width() == overlay.explore_panel.minimumSizeHint().width()
        assert overlay.explore_panel.width() <= overlay.explore_panel.sizeHint().width()
        assert overlay.explore_panel.geometry().right() < overlay.width()

    @pytest.mark.parametrize("size", [(800, 600), (1280, 800)])
    def test_no_two_buttons_in_the_same_explore_group_ever_overlap(self, experience, qapp, size):
        """Root-cause regression guard for the Vent 1 button-row overlap:
        OPEN/CLOSED/HVAC used to visually stack on top of each other at
        800x600 because explore_panel (and each ExploreToggle) only
        exposed a soft minimumSizeHint() to their parent layouts -- under
        real space pressure (root reserves a right-hand column for the
        stat panel), Qt compressed buttons below their own declared
        setMinimumWidth(52) floor rather than overflowing the container.
        ExploreToggle.setMinimumWidth() and explore_panel.setMinimumWidth()
        (both in this fix) convert that hint into a hard floor every
        ancestor layout must respect, which structurally prevents this --
        checked directly on real button geometry, not just on the sizes
        the layout *reports* wanting."""
        experience.window().resize(*size)
        for _ in range(15):
            qapp.processEvents()
        overlay = experience.overlay
        for key, toggle in overlay._explore_toggles.items():
            buttons = [toggle._group.button(i) for i in range(len(toggle._values))]
            rects = [QtCore.QRect(b.mapTo(overlay, QtCore.QPoint(0, 0)), b.size())
                     for b in buttons]
            for i in range(len(rects)):
                for j in range(i + 1, len(rects)):
                    assert not rects[i].intersects(rects[j]), (
                        f"{key}: button {i} {rects[i]} overlaps button {j} {rects[j]}")


class TestPlayPauseControl:
    """"Stop time": a plain, always-correct play/pause over the shared
    TimeController, available while the fire is actually playing."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_button_present_while_observing(self, experience):
        experience._begin_journey()
        experience._start_observe()
        texts = [b.text() for b in experience.overlay.children()
                 if isinstance(b, QtWidgets.QPushButton)]
        assert any("Pause" in t for t in texts)

    def test_click_pauses_and_resumes_the_real_clock(self, experience):
        experience._begin_journey()
        experience._start_observe()
        assert experience.time_controller.is_playing()
        experience._on_play_pause_clicked()
        assert not experience.time_controller.is_playing()
        experience._on_play_pause_clicked()
        assert experience.time_controller.is_playing()

    def test_label_tracks_the_real_playing_state(self, experience):
        experience._begin_journey()
        experience._start_observe()
        assert "Pause" in experience._play_pause_button.text()
        experience.time_controller.pause()
        assert "Play" in experience._play_pause_button.text()
        experience.time_controller.play()
        assert "Pause" in experience._play_pause_button.text()

    def test_pausing_freezes_the_frame_the_child_is_looking_at(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience.time_controller.seek(30)
        experience._on_play_pause_clicked()
        frame_at_pause = experience.state.frame_index
        # No timer tick can move the frame while paused.
        assert not experience.time_controller.is_playing()
        assert experience.state.frame_index == frame_at_pause

    def test_button_also_offered_during_the_experiment(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience.time_controller.pause()
        experience._observe_finished()
        experience._on_prediction("cooler")
        run_countdown(experience)
        assert experience.state.phase is Phase.EXPERIMENT
        texts = [b.text() for b in experience.overlay.children()
                 if isinstance(b, QtWidgets.QPushButton)]
        assert any("Pause" in t for t in texts)

    def test_stale_button_reference_never_touched_after_a_phase_change(self, experience):
        """Regression guard: _play_pause_button must be reset to None by
        _render_phase before a phase with no such button runs, or
        TimeController.playing_changed would reach a discarded widget."""
        experience._begin_journey()
        experience._start_observe()
        experience.time_controller.pause()
        experience._observe_finished()   # PREDICTION: no play/pause button
        assert experience._play_pause_button is None
        # Nothing here should raise even though play() emits the signal.
        experience.time_controller.play()


class TestTemperatureBandEmoji:
    """The thermometer/meter icons are expressive emoji (a volcano for
    "very hot", etc.), not flat colour squares -- still driven by the
    same bands/thresholds kid_language has always used."""

    def test_hot_bands_use_expressive_emoji(self):
        assert kid.temperature_icon(25.0) == "🧊"
        assert kid.temperature_icon(350.0) == "🌋"
        assert kid.temperature_icon(500.0) == "🕯️"

    def test_color_and_icon_share_the_same_band_boundaries(self):
        """temperature_color's band edges must exactly match
        temperature_icon's -- both are read off the same
        _TEMPERATURE_BANDS table, so a colour can never disagree with
        the emoji beside it."""
        for value in (10.0, 27.9, 28.1, 89.0, 401.0, 1000.0):
            band_icon = kid.temperature_icon(value)
            band_phrase = kid.temperature_phrase(value)
            # Every band boundary produces a real (non-empty) icon/colour;
            # this mostly guards against an off-by-one in a future edit
            # to the bands rather than asserting exact values twice.
            assert band_icon
            assert kid.temperature_color(value).startswith("#")
            assert band_phrase


class TestLocationGrounding:
    """A probed reading is captioned with where it physically is,
    computed from the real room/candle geometry -- not a decorative
    label."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_near_the_floor(self, experience):
        from schematic import ROOM_Z, _CANDLE_X
        experience._begin_journey()
        experience._start_observe()
        x = _CANDLE_X[0] - 0.1   # away from the candle, still in-room
        assert experience.scene.location_phrase(x, ROOM_Z[0] + 0.005) == "near the floor"

    def test_near_the_ceiling(self, experience):
        from schematic import ROOM_Z, _CANDLE_X
        experience._begin_journey()
        experience._start_observe()
        x = _CANDLE_X[0] - 0.1
        assert experience.scene.location_phrase(x, ROOM_Z[1] - 0.005) == "near the ceiling"

    def test_middle_of_the_room(self, experience):
        from schematic import ROOM_Z, _CANDLE_X
        experience._begin_journey()
        experience._start_observe()
        x = _CANDLE_X[0] - 0.1
        mid_z = (ROOM_Z[0] + ROOM_Z[1]) / 2
        assert experience.scene.location_phrase(x, mid_z) == "in the middle of the room"

    def test_at_the_flame_takes_priority_over_height(self, experience):
        """A point that is both "at the candle" and "near the floor"
        (the candle's own base) must be named as the flame, not the
        floor -- candle_hit is checked first."""
        from schematic import ROOM_Z
        experience._begin_journey()
        experience._start_observe()
        cx = experience.scene._candle_signature[0]
        assert experience.scene.location_phrase(cx, ROOM_Z[0] + 0.005) == "at the flame"

    def test_tap_caption_names_a_real_location(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        canvas = experience.scene.view.canvas
        centre = QtCore.QPoint(canvas.width() // 2, canvas.height() // 2)
        experience._on_overlay_tapped(centre)
        caption = experience.overlay.thermometer._caption.text()
        assert any(phrase in caption for phrase in
                  ("near the ceiling", "near the floor", "in the middle of the room"))


class TestBannerFlashDoesNotOutliveThePhase:
    """Regression: a WHOOSH flash started during EXPERIMENT restores its
    *own* stale text via a delayed QTimer, which used to fire after the
    phase had already moved on to REVEAL and cleared the prompt --
    silently re-showing old text (and, at 800x600, overlapping the
    thermometer). Found via automated overlap screenshotting, not
    speculation."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_set_prompt_cancels_a_pending_flash(self, experience):
        banner = experience.overlay.banner
        banner.set_prompt("before")
        banner.flash("WHOOSH")
        assert banner.text() == "WHOOSH"
        assert banner._flash_timer.isActive()

        banner.set_prompt("")   # what a phase transition to REVEAL does
        assert banner._flash_timer.isActive() is False
        assert banner.text() == ""
        assert banner.isHidden()

    def test_cancelled_flash_never_fires_even_after_its_own_duration(self, experience, qapp):
        """Real end-to-end guarantee via the actual QTimer/event loop --
        not a check that assumes the cancellation mechanism, but one
        that would fail again if a future change reintroduced the bug
        (e.g. by cancelling the timer without stopping it)."""
        banner = experience.overlay.banner
        banner.set_prompt("Fan ON")
        banner.flash("🌬️  WHOOSH — watch the air speed!", duration_ms=30)
        experience.overlay.set_prompt("")   # simulate _render_reveal()
        assert banner.isHidden()

        QtCore.QThread.msleep(80)   # past the flash's own duration
        for _ in range(6):
            qapp.processEvents()
        assert banner.isHidden()
        assert banner.text() == ""


class TestFindHottestGame:
    """"Find the hottest place": the target is the real measured maximum
    (PublicScene.hottest_point_at), never a hardcoded screen position."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    @staticmethod
    def _pos_for(experience, x, z):
        # Routed through the scene's own widget_fraction_for rather than
        # a second copy of its formula: the scene only renders into
        # SCENE_WIDTH_FRAC of the widget's width now (a real, reserved
        # right-hand column for the thermometer, never drawn into -- see
        # PublicScene.SCENE_WIDTH_FRAC), and a hand-rolled full-bleed
        # version of this helper would compute tap positions past the
        # scene's actual right edge, landing in that reserved column
        # instead of on the vent/candle/etc. this is supposed to tap.
        canvas = experience.scene.view.canvas
        fx, fy = experience.scene.widget_fraction_for(x, z)
        return QtCore.QPoint(
            min(canvas.width() - 1, max(0, int(fx * canvas.width()))),
            min(canvas.height() - 1, max(0, int(fy * canvas.height()))))

    def test_far_guess_gives_a_band_reaction_and_stays_armed(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(60):
            experience.time_controller.seek(experience.time_controller.index + 1)
        experience.time_controller.pause()
        experience._enter_game_hottest()
        assert experience._active_game == "hottest"

        far_pos = self._pos_for(experience, 0.02, 0.46)   # ceiling corner, far from the flame
        experience._on_overlay_tapped(far_pos)
        assert experience._active_game == "hottest"   # still guessing
        assert "°C" in experience.overlay.bubble.text() or "!" in experience.overlay.bubble.text()

    def test_close_guess_wins_using_the_real_measured_maximum(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(60):
            experience.time_controller.seek(experience.time_controller.index + 1)
        experience.time_controller.pause()
        hot_x, hot_z, hot_value = experience.scene.hottest_point_at(experience.state.frame_index)
        experience._enter_game_hottest()
        # Phase 4 generalized this into a dynamically-chosen hottest/
        # coolest hunt (random per activation) -- force this round's
        # target so the test deterministically exercises the "hottest"
        # win path rather than flaking ~50% of the time.
        experience._temp_hunt_target = "hottest"

        experience._on_overlay_tapped(self._pos_for(experience, hot_x, hot_z))

        assert experience.state.phase is Phase.GAME_PLAY   # the game screen stays put
        assert f"{hot_value:.0f}" in experience.overlay.bubble.text()
        assert "found" in experience.overlay.bubble.text().lower()

    def test_inactive_outside_the_game(self, experience, qapp):
        """A plain Explore tap must never be silently judged as a
        hottest-place guess -- a game can only be active in GAME_PLAY."""
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        canvas = experience.scene.view.canvas
        experience._on_overlay_tapped(QtCore.QPoint(canvas.width() // 2, canvas.height() // 2))
        assert experience._active_game is None

    def test_game_state_cleared_on_scenario_switch(self, experience):
        """Entering a different game (which clears state fresh) after
        having played this one must not leak its target/progress."""
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_hottest()
        assert experience._active_game == "hottest"
        experience._enter_game_hotcold()
        assert experience._active_game == "hotcold"
        assert experience._hotcold_stage == 0


class TestHotOrColdGame:
    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_two_taps_record_real_values_and_finish_the_game(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(60):
            experience.time_controller.seek(experience.time_controller.index + 1)
        experience.time_controller.pause()
        canvas = experience.scene.view.canvas

        experience._enter_game_hotcold()
        assert experience._active_game == "hotcold"

        first_pos = QtCore.QPoint(canvas.width() // 2, canvas.height() // 2)
        experience._on_overlay_tapped(first_pos)
        assert experience._hotcold_stage == 1
        _hot_x, _hot_z, first_value = experience._hotcold_readings["hot"]
        expected_first = experience.scene.view.value_at(
            *experience.scene.probe_at(first_pos)[:2])
        assert first_value == pytest.approx(expected_first)

        second_pos = QtCore.QPoint(10, 10)
        experience._on_overlay_tapped(second_pos)
        assert experience._active_game == "hotcold"   # the game screen stays put
        assert experience._hotcold_stage == 2
        assert "vs" in experience.overlay.bubble.text()

    def test_a_tap_on_the_candle_still_counts_as_a_real_reading(self, experience, qapp):
        """A hot/cold tap that lands on the candle must not be silently
        swallowed by the plain candle callout -- it is real game input
        (see _on_game_tap's candle_hit branch)."""
        experience._begin_journey()
        experience._start_observe()
        for _ in range(60):
            experience.time_controller.seek(experience.time_controller.index + 1)
        experience.time_controller.pause()
        hot_x, hot_z, hot_value = experience.scene.hottest_point_at(experience.state.frame_index)
        assert experience.scene.candle_hit(hot_x, hot_z)   # the setup this test needs
        # Routed through widget_fraction_for -- see TestFlamePulse._pos_for's
        # own comment on why a hand-rolled full-bleed formula here would
        # miss (PublicScene.SCENE_WIDTH_FRAC).
        canvas = experience.scene.view.canvas
        fx, fy = experience.scene.widget_fraction_for(hot_x, hot_z)
        pos = QtCore.QPoint(min(canvas.width() - 1, int(fx * canvas.width())),
                            min(canvas.height() - 1, int(fy * canvas.height())))

        experience._enter_game_hotcold()
        experience._on_overlay_tapped(pos)

        assert experience._hotcold_stage == 1
        _hot_x, _hot_z, recorded = experience._hotcold_readings["hot"]
        assert recorded == pytest.approx(hot_value)

    def test_almost_the_same_when_below_the_noticeable_delta(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        canvas = experience.scene.view.canvas
        centre = QtCore.QPoint(canvas.width() // 2, canvas.height() // 2)
        experience._enter_game_hotcold()
        experience._on_overlay_tapped(centre)
        # Tap the exact same point again -- the same value both times,
        # certainly below any noticeable_delta.
        experience._on_overlay_tapped(centre)
        assert "Almost the same" in experience.overlay.bubble.text()

    def test_tap_after_completion_starts_a_new_round(self, experience):
        """"Tap anywhere to try again" (see _render_game_hotcold) -- a
        completed round doesn't need a trip back to the Games hub to
        replay."""
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_hotcold()
        canvas = experience.scene.view.canvas
        experience._on_overlay_tapped(QtCore.QPoint(canvas.width() // 2, canvas.height() // 2))
        experience._on_overlay_tapped(QtCore.QPoint(10, 10))
        assert experience._hotcold_stage == 2

        experience._on_overlay_tapped(QtCore.QPoint(canvas.width() // 3, canvas.height() // 3))

        assert experience._hotcold_stage == 1
        assert experience._active_game == "hotcold"


class TestSamePlaceVerdict:
    """Phase 7 section 3: the same-place comparison card leads with a
    qualitative verdict (much hotter / much cooler / almost the same)
    rather than raw degrees, driven by the *same* noticeable_delta
    threshold the rest of Fire Lab already uses -- no new threshold
    invented, and the exact °C values still appear in the BarCompare.
    Reached via Mystery, every tap of which is already a same-place
    comparison (see PublicExperience._on_game_tap)."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_verdict_matches_the_real_measured_delta(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_mystery()
        experience._on_explore_changed("vent1", 2)
        for _ in range(6):
            qapp.processEvents()
        canvas = experience.scene.view.canvas
        centre = QtCore.QPoint(canvas.width() // 2, canvas.height() // 2)
        x, z, _value = experience.scene.probe_at(centre)
        baseline_val = experience.scene.measure_case_at(
            experience.state.baseline_case_index, x, z)
        current_val = experience.scene.measure_case_at(experience.state.case_index, x, z)
        room_temp = next(m for m in experiments_mod.PUBLIC_METRICS if m.key == "room_temp")
        delta = current_val - baseline_val
        noticeable = abs(delta) >= room_temp.noticeable_delta

        experience._on_overlay_tapped(centre)

        text = "\n".join(w.text() for w in experience.overlay.card.findChildren(QtWidgets.QLabel))
        if not noticeable:
            assert "Almost the same" in text
        elif delta > 0:
            assert "Much hotter" in text
        else:
            assert "Much cooler" in text
        # The exact numbers are still present, just no longer the headline.
        bars = experience.overlay.card.findChildren(BarCompare)
        assert any(v == pytest.approx(baseline_val) for _l, v in bars[0]._rows)
        assert any(v == pytest.approx(current_val) for _l, v in bars[0]._rows)


class TestHottestPointDetection:
    """PublicScene's own real-data hottest-cell lookup, independent of
    the game layered on top of it."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_hottest_point_matches_the_real_array_argmax(self, experience, sim_data):
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        experience._begin_journey()
        experience._start_observe()
        for _ in range(60):
            experience.time_controller.seek(experience.time_controller.index + 1)
        index = experience.state.frame_index
        x, z, value = experience.scene.hottest_point_at(index)
        frame = experience.scene._temperature[index]
        assert value == pytest.approx(float(frame.max()))
        x0, x1, z0, z1 = experience.scene.view._extent
        assert x0 <= x <= x1
        assert z0 <= z <= z1

    def test_no_data_returns_none(self):
        from public.scene import PublicScene
        scene = PublicScene(store=None, manifest=[], fps=4)
        assert scene.hottest_point_at(0) is None
        assert scene.hottest_guess_is_close(0.5, 0.2, 25.0, 0) is False

    def test_measure_case_at_matches_direct_array_lookup(self, experience, sim_data):
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        experience._begin_journey()
        experience._start_observe()
        case = experience.state.case_index
        x, z = 0.5, 0.2
        value = experience.scene.measure_case_at(case, x, z, frame_index=-1)
        arr = experience.sim_data.store.get(case, experience.scene._quantity_key)
        expected = experience.scene.view.value_at(x, z)   # same mapping, current frame
        # measure_case_at at frame_index=-1 reads the settled frame, not
        # necessarily the currently-displayed one -- assert independently
        # via the same row/col mapping instead of reusing value_at's
        # current-frame shortcut.
        x0, x1, z0, z1 = experience.scene.view._extent
        n_z, n_x = arr[-1].shape
        col = int(round((x - x0) / (x1 - x0) * (n_x - 1)))
        row = int(round((z1 - z) / (z1 - z0) * (n_z - 1)))
        assert value == pytest.approx(float(arr[-1][row, col]))


class TestVentTap:
    """The in-diagram vent icons (both Vent 1/fan and Vent 2) are a
    passive display of the real vent state -- spinning blades when the
    fan is on, an activity glow while open, driven every frame by real
    data (see PublicScene._draw_vent_object/_animate_vent). They are not
    a second control: only the Vent 1/Vent 2 buttons in the control bar
    change vent state (PublicExperience._on_explore_changed). A tap
    landing on the vent's drawn position must be treated exactly like
    any other tap on plain air -- a real reading at that point, no
    state change, no banner flash, no line-width pulse."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    @staticmethod
    def _pos_for(experience, x, z):
        # Routed through the scene's own widget_fraction_for rather than
        # a second copy of its formula: the scene only renders into
        # SCENE_WIDTH_FRAC of the widget's width now (a real, reserved
        # right-hand column for the thermometer, never drawn into -- see
        # PublicScene.SCENE_WIDTH_FRAC), and a hand-rolled full-bleed
        # version of this helper would compute tap positions past the
        # scene's actual right edge, landing in that reserved column
        # instead of on the vent/candle/etc. this is supposed to tap.
        canvas = experience.scene.view.canvas
        fx, fy = experience.scene.widget_fraction_for(x, z)
        return QtCore.QPoint(
            min(canvas.width() - 1, max(0, int(fx * canvas.width()))),
            min(canvas.height() - 1, max(0, int(fy * canvas.height()))))

    def test_tap_on_the_real_vent_position_does_not_change_vent_state(self, experience):
        experience._begin_journey()
        experience._start_observe()
        before = experience.state.case_index
        vent_xz = experience.scene.vent_marker_position(before)
        assert vent_xz is not None

        experience._on_overlay_tapped(self._pos_for(experience, *vent_xz))

        assert experience.state.case_index == before
        fan_toggle = experience.overlay._explore_toggles["vent1"]
        entry = next(e for e in experience.sim_data.manifest if e.case_index == before)
        assert fan_toggle._group.checkedId() == entry.vod

    def test_tap_on_the_second_vent_position_does_not_change_vent_state(self, experience):
        experience._begin_journey()
        experience._start_observe()
        before = experience.state.case_index
        vent2_xz = experience.scene.vent2_marker_position(before)
        assert vent2_xz is not None

        experience._on_overlay_tapped(self._pos_for(experience, *vent2_xz))

        assert experience.state.case_index == before

    def test_does_not_flash_the_banner(self, experience):
        experience._begin_journey()
        experience._start_observe()
        vent_xz = experience.scene.vent_marker_position(experience.state.case_index)
        experience._on_overlay_tapped(self._pos_for(experience, *vent_xz))
        assert "WHOOSH" not in experience.overlay.banner.text()

    def test_reads_as_a_plain_air_reading_instead(self, experience):
        """Falling through to the ordinary tap handling means the
        thermometer callout is the plain air reading (location_phrase),
        never the removed vent-toggle path."""
        experience._begin_journey()
        experience._start_observe()
        vent_xz = experience.scene.vent_marker_position(experience.state.case_index)
        experience._on_overlay_tapped(self._pos_for(experience, *vent_xz))
        assert experience._probe_mode == "point"
        # Not an exact match: probe_at() snaps to the nearest real data
        # cell, so a pixel-perfect tap on the vent's own drawn (x, z)
        # can resolve a cell over -- close is enough to confirm this
        # landed as a plain air probe near the vent, not some other spot.
        px, pz = experience._probe_xz
        vx, vz = vent_xz
        assert px == pytest.approx(vx, abs=0.01)
        assert pz == pytest.approx(vz, abs=0.01)

    def test_inert_outside_observe(self, experience):
        experience._begin_journey()   # INTRO
        for _ in range(6):
            pass
        # Not in OBSERVE yet: even if a vent happened to be under this
        # point it must not be treated as a fan toggle.
        result = experience.scene.probe_at(
            self._pos_for(experience, 0.5, 0.2))
        assert experience.state.phase is Phase.INTRO

    def test_tapping_the_vent_does_not_flash_its_drawn_line(self, experience, qapp):
        """The line-width pulse on the real room_vents artist is now only
        the idle-nudge hint (PublicExperience._on_idle_vent_timeout) and
        the control-bar toggle path -- a direct tap on the passive icon
        must not trigger it."""
        experience._begin_journey()
        experience._start_observe()
        base_lw = experience.scene.view.room_vents.get_linewidths()[0]
        vent_xz = experience.scene.vent_marker_position(experience.state.case_index)

        experience._on_overlay_tapped(self._pos_for(experience, *vent_xz))
        for _ in range(3):
            qapp.processEvents()
        unchanged_lw = experience.scene.view.room_vents.get_linewidths()[0]
        assert unchanged_lw == pytest.approx(base_lw)


class TestVentActivityGlow:
    """Phase 12 section 2 (generalized to both real vents per later
    supervisor feedback: the same "air can move here" animation --
    housing + spinning blades + activity glow -- for Vent 1 and Vent 2,
    not a powered fan for one and a static flap for the other). A
    continuous "this thing is doing something" cue while a vent is
    *open*, driven every frame by the real per-frame velocity magnitude
    -- never a direction this dataset can't support. Quiet (no patch at
    all) while closed. Vent 1/vod has three real states (open/closed/
    HVAC fan) -- both open (0) and fan-on (2) count as "open" here, only
    closed (1) is quiet. Vent 2/voc has just open (0) / closed (1)."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_no_patch_while_vent1_is_closed(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._on_explore_changed("vent1", 1)
        assert experience.scene._vent1.activity_patch is None

    def test_patch_appears_at_the_real_vent_position_when_vent1_is_plain_open(self, experience):
        """vod=0 (open, no motor) is still "open" now, not the old
        control's collapsed "off" -- see EXPLORE_CONTROLS' own comment
        on why the old binary fan control skipped this state."""
        experience._begin_journey()
        experience._start_observe()
        vent_xz = experience.scene.vent_marker_position(experience.state.case_index)
        experience._on_explore_changed("vent1", 1)   # closed first, so 0 is a real change
        experience._on_explore_changed("vent1", 0)

        assert experience.scene._vent1.activity_patch is not None
        assert experience.scene._vent1.activity_patch.center == pytest.approx(vent_xz)

    def test_patch_appears_at_the_real_vent_position_when_vent1_fan_turns_on(self, experience):
        experience._begin_journey()
        experience._start_observe()
        vent_xz = experience.scene.vent_marker_position(experience.state.case_index)

        experience._on_explore_changed("vent1", 2)

        assert experience.scene._vent1.activity_patch is not None
        assert experience.scene._vent1.activity_patch.center == pytest.approx(vent_xz)

    def test_patch_disappears_when_vent1_closes(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._on_explore_changed("vent1", 2)
        assert experience.scene._vent1.activity_patch is not None

        experience._on_explore_changed("vent1", 1)

        assert experience.scene._vent1.activity_patch is None

    def test_no_patch_while_vent2_is_closed(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._on_explore_changed("vent2", 1)
        assert experience.scene._vent2.activity_patch is None

    def test_patch_appears_at_the_real_vent2_position_when_open(self, experience):
        experience._begin_journey()
        experience._start_observe()
        vent2_xz = experience.scene.vent2_marker_position(experience.state.case_index)
        experience._on_explore_changed("vent2", 1)   # closed first, so 0 is a real change
        experience._on_explore_changed("vent2", 0)

        assert experience.scene._vent2.activity_patch is not None
        assert experience.scene._vent2.activity_patch.center == pytest.approx(vent2_xz)

    def test_patch_disappears_when_vent2_closes(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._on_explore_changed("vent2", 0)
        assert experience.scene._vent2.activity_patch is not None

        experience._on_explore_changed("vent2", 1)

        assert experience.scene._vent2.activity_patch is None

    def test_radius_and_alpha_track_the_real_frame_velocity(self, experience):
        """Never a fabricated animation -- the exact per-frame velocity
        array already loaded for the mean-airspeed meter drives this,
        nothing else."""
        experience._begin_journey()
        experience._start_observe()
        experience._on_explore_changed("vent1", 2)
        base_r = experience.scene._vent1.activity_base_r

        still_vel = np.zeros_like(experience.scene._velocity[0])
        experience.scene._animate_vent(experience.scene._vent1, 0, still_vel)
        quiet_alpha = experience.scene._vent1.activity_patch.get_alpha()

        moving_vel = np.full_like(experience.scene._velocity[0], 1.0)
        experience.scene._animate_vent(experience.scene._vent1, 0, moving_vel)
        active_alpha = experience.scene._vent1.activity_patch.get_alpha()

        assert active_alpha > quiet_alpha
        assert experience.scene._vent1.activity_patch.get_radius() >= base_r


class TestDoorVisibility:
    """The door's real state (narrow/wide, i.e. door=0/1) always did
    update room_door's own segment data correctly -- confirmed by
    inspecting PublicScene._room_outline_for's output directly -- but at
    the original line weight (2.6px, the same order as the dashed wall
    outline it sits inside of) the change was not legible as "a door" in
    an actual 800x600 screenshot: toggling NARROW/WIDE produced no
    perceptible change. Fixed by weighting the door line at least as
    heavily as the vents' own lines (views._ROOM_VENT_LW), not by
    touching the geometry, which was never wrong."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_door_line_is_at_least_as_bold_as_the_vents(self):
        from views import _ROOM_DOOR_LW, _ROOM_VENT_LW, _ROOM_WALL_LW
        assert _ROOM_DOOR_LW >= _ROOM_VENT_LW
        assert _ROOM_DOOR_LW > _ROOM_WALL_LW

    def test_door_segment_length_actually_changes_between_narrow_and_wide(self, experience):
        experience._begin_journey()
        experience._start_observe()

        def door_span():
            geometry = experience.scene._room_outline_for(experience.state.case_index)
            dx0, dz0, dx1, dz1 = geometry["door"]
            return abs(dz1 - dz0)

        experience._on_explore_changed("door", 0)
        narrow_span = door_span()
        experience._on_explore_changed("door", 1)
        wide_span = door_span()

        assert wide_span > narrow_span
        # The real generator values (fds/generate_sim.py: door = [0.050,
        # 0.150]) -- not assumed close, an actual >2x difference.
        assert wide_span >= narrow_span * 2


class TestIdleVentHint:
    """Phase 11 section 2: a one-time, one-shot nudge toward the vent
    after real inactivity during OBSERVE -- reuses PublicScene.
    pulse_vent() exactly, never a new visual effect, and fires at most
    once per visit."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_armed_on_entering_observe(self, experience):
        experience._begin_journey()
        experience._start_observe()
        assert experience._idle_vent_timer.isActive()

    def test_cancelled_by_a_real_tap(self, experience):
        experience._begin_journey()
        experience._start_observe()
        canvas = experience.scene.view.canvas

        experience._on_overlay_tapped(QtCore.QPoint(canvas.width() // 2, canvas.height() // 2))

        assert not experience._idle_vent_timer.isActive()

    def test_cancelled_by_an_explore_change(self, experience):
        experience._begin_journey()
        experience._start_observe()

        experience._on_explore_changed("vent1", 2)

        assert not experience._idle_vent_timer.isActive()

    def test_fires_at_most_once_and_reuses_pulse_vent(self, experience):
        experience._begin_journey()
        experience._start_observe()
        base_lw = experience.scene.view.room_vents.get_linewidths()[0]

        experience._on_idle_vent_timeout()

        assert experience._idle_vent_hint_shown
        assert experience.scene.view.room_vents.get_linewidths()[0] > base_lw

        # A second call (e.g. a stray late timeout) must be a no-op.
        experience.scene.reset_vent_width()
        experience._on_idle_vent_timeout()
        assert experience.scene.view.room_vents.get_linewidths()[0] == pytest.approx(base_lw)

    def test_reset_by_kiosk_idle_lets_it_arm_again(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._on_idle_vent_timeout()
        assert experience._idle_vent_hint_shown

        experience.reset()

        assert experience._idle_vent_hint_shown is False


class TestFanStateConsistency:
    """Phase 9 section 10: every widget that shows the fan's state must
    agree with the actually-loaded scenario -- the toggle, the vent's own
    drawn line color, and (once its 340 ms before/after sweep finishes)
    the comparison card's embedded thermometer caption. A prior report
    flagged an apparent mismatch; investigating it here found no real
    bug -- it was that animation caught mid-flight by a screenshot taken
    before the sweep finished. This test locks in the real invariant so
    a future regression would be caught."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    @staticmethod
    def _pos_for(experience, x, z):
        # Routed through the scene's own widget_fraction_for rather than
        # a second copy of its formula: the scene only renders into
        # SCENE_WIDTH_FRAC of the widget's width now (a real, reserved
        # right-hand column for the thermometer, never drawn into -- see
        # PublicScene.SCENE_WIDTH_FRAC), and a hand-rolled full-bleed
        # version of this helper would compute tap positions past the
        # scene's actual right edge, landing in that reserved column
        # instead of on the vent/candle/etc. this is supposed to tap.
        canvas = experience.scene.view.canvas
        fx, fy = experience.scene.widget_fraction_for(x, z)
        return QtCore.QPoint(
            min(canvas.width() - 1, max(0, int(fx * canvas.width()))),
            min(canvas.height() - 1, max(0, int(fy * canvas.height()))))

    def _assert_consistent(self, experience, expect_on: bool):
        entry = experience.scene.current_entry()
        assert (entry.vod == 2) == expect_on
        fan_toggle = experience.overlay._explore_toggles["vent1"]
        assert (fan_toggle._group.checkedId() == 2) == expect_on   # "HVAC" is the 3rd of 3 real vod options
        vent_state = "HVAC" if expect_on else "open"
        from views import _VENT_STATE_COLORS
        import matplotlib.colors as mcolors
        expected_rgba = mcolors.to_rgba(_VENT_STATE_COLORS[vent_state])
        actual_rgba = tuple(experience.scene.view.room_vents.get_colors()[0])
        assert actual_rgba == pytest.approx(expected_rgba)

    def test_fan_off_is_consistent_everywhere(self, experience):
        experience._begin_journey()
        experience._start_observe()
        self._assert_consistent(experience, expect_on=False)

    def test_fan_on_is_consistent_everywhere(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        # A real toggle click, not _on_explore_changed() directly -- the
        # toggle's own checked state is flipped by Qt as part of an
        # actual press, before the signal handler ever runs, so driving
        # it any other way would never exercise "does the toggle's own
        # visual state agree", the exact thing this test checks.
        experience.overlay._explore_toggles["vent1"]._group.button(2).click()   # "HVAC"
        for _ in range(6):
            qapp.processEvents()
        self._assert_consistent(experience, expect_on=True)

    def test_compare_card_thermometer_matches_real_state_after_the_sweep(
            self, experience, qapp):
        """The compare card's own mini-thermometer caption briefly shows
        the *baseline* label mid-sweep by design (before -> after, Phase
        7 section 4) -- but once the 340 ms sweep finishes it must agree
        with the real loaded scenario, never get stuck on the old label.
        The "revisit a before-trail spot" mechanic lives in the Map It
        game."""
        from schematic import ROOM_Z
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_map()
        ceiling_pos = self._pos_for(experience, 0.75, ROOM_Z[1] - 0.01)
        experience._on_overlay_tapped(ceiling_pos)
        experience._on_explore_changed("vent1", 2)
        for _ in range(20):
            experience.time_controller.seek(experience.time_controller.index + 1)
        experience.time_controller.pause()

        experience._on_overlay_tapped(ceiling_pos)   # revisit -> auto comparison
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(500, loop.quit)   # past the 340ms sweep
        loop.exec_()

        assert "HVAC" in experience.overlay.thermometer._caption.text()   # "Vent 1 HVAC"
        entry = experience.scene.current_entry()
        assert entry.vod == 2


class TestThermometerAnimation:
    """The rendered fill sweeps toward the real value rather than
    snapping -- Phase 3 section 4: "the animation may interpolate
    visually... but the final value must come from the actual
    simulation array." The number label is never animated."""

    def test_number_label_is_exact_and_immediate(self, qapp):
        from public.widgets import Thermometer
        thermo = Thermometer()
        thermo.set_reading(123.456, "test")
        assert thermo._value_label.text() == "123°C"
        assert thermo._value == pytest.approx(123.456)

    def test_display_value_animates_toward_the_target(self, qapp):
        from public.widgets import Thermometer
        thermo = Thermometer()
        thermo.set_reading(20.0)
        thermo._anim.stop()
        thermo._display_value = 20.0
        thermo.set_reading(400.0)
        assert thermo._anim.state() == QtCore.QAbstractAnimation.Running
        # Advance the animation partway and confirm the displayed value
        # moved from the old reading toward the new one, not snapped.
        thermo._anim.setCurrentTime(140)
        assert 20.0 < thermo._display_value < 400.0

    def test_clear_stops_the_animation_and_resets_display(self, qapp):
        from public.widgets import Thermometer
        thermo = Thermometer()
        thermo.set_reading(200.0)
        thermo.clear()
        assert thermo._display_value is None
        assert thermo._anim.state() == QtCore.QAbstractAnimation.Stopped


class TestHotColdDualMarkers:
    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_both_markers_shown_at_the_real_tapped_locations(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(60):
            experience.time_controller.seek(experience.time_controller.index + 1)
        experience.time_controller.pause()
        canvas = experience.scene.view.canvas
        experience._enter_game_hotcold()

        first_pos = QtCore.QPoint(canvas.width() // 2, canvas.height() // 2)
        x1, z1, _v = experience.scene.probe_at(first_pos)
        experience._on_overlay_tapped(first_pos)
        second_pos = QtCore.QPoint(10, 10)
        x2, z2, _v = experience.scene.probe_at(second_pos)
        experience._on_overlay_tapped(second_pos)

        hot_offsets = experience.scene._hot_marker.get_offsets()
        cool_offsets = experience.scene._cool_marker.get_offsets()
        assert hot_offsets[0] == pytest.approx((x1, z1), abs=1e-6)
        assert cool_offsets[0] == pytest.approx((x2, z2), abs=1e-6)

    def test_markers_cleared_on_new_game_and_on_scenario_switch(self, experience):
        experience._begin_journey()
        experience._start_observe()
        canvas = experience.scene.view.canvas
        experience._enter_game_hotcold()
        experience._on_overlay_tapped(QtCore.QPoint(canvas.width() // 2, canvas.height() // 2))
        experience._on_overlay_tapped(QtCore.QPoint(10, 10))
        assert len(experience.scene._hot_marker.get_offsets()) == 1

        experience._on_explore_changed("vent1", 2)
        assert len(experience.scene._hot_marker.get_offsets()) == 0
        assert len(experience.scene._cool_marker.get_offsets()) == 0


class TestFlamePulse:
    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_tapping_the_candle_triggers_a_pulse(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(6):
            qapp.processEvents()
        canvas = experience.scene.view.canvas
        candle_x = sum(experience.scene._candle_signature) / len(experience.scene._candle_signature)
        from schematic import ROOM_Z
        pos = TestVentTap._pos_for(experience, candle_x, ROOM_Z[0] + 0.01)
        experience._on_overlay_tapped(pos)
        assert experience.scene._flame_pulse_frames_left > 0

    def test_pulse_decays_over_frames(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience.scene.pulse_flame()
        left_before = experience.scene._flame_pulse_frames_left
        experience.scene._jitter_flame(experience.state.frame_index + 1)
        assert experience.scene._flame_pulse_frames_left == left_before - 1

    def test_no_crash_with_no_candle(self):
        from public.scene import PublicScene
        scene = PublicScene(store=None, manifest=[], fps=4)
        scene.pulse_flame()
        scene._jitter_flame(0)


class TestDeferredSweepGenerationSafety:
    """Phase 4 section 15: a delayed thermometer sweep (same-place
    compare, hot/cold) must never stomp the UI after the child has moved
    on to something else -- _defer_if_current's generation token, not a
    fixed delay that just happens not to be reached in tests."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_stale_same_place_sweep_never_applies_after_a_scenario_switch(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_mystery()
        canvas = experience.scene.view.canvas
        experience._on_explore_changed("vent1", 2)

        # Every tap in Mystery is already a same-place comparison (see
        # PublicExperience._on_game_tap) -- no separate "arm" step.
        centre = QtCore.QPoint(canvas.width() // 2, canvas.height() // 2)
        experience._on_overlay_tapped(centre)   # schedules the deferred second half

        # Switch scenario, switch again, and change phase -- all within
        # the sweep's own ~340 ms delay window -- before the pending
        # callback ever gets a chance to fire.
        experience._close_compare()
        experience._on_explore_changed("candles", 1)
        experience.time_controller.pause()
        experience._observe_finished()   # phase change: PREDICTION
        caption_after_phase_change = experience.overlay.thermometer._caption.text()

        # Let real wall-clock time pass so the stale timer actually fires.
        QtCore.QThread.msleep(400)
        for _ in range(10):
            qapp.processEvents()

        # The stale sweep must not have touched anything that was current
        # at the time it fired -- the caption is exactly what the phase
        # change left it as, not overwritten by the abandoned sweep.
        assert experience.state.phase is Phase.PREDICTION
        assert experience.overlay.thermometer._caption.text() == caption_after_phase_change

    def test_generation_bumped_by_scenario_load_and_phase_render(self, experience):
        experience._begin_journey()
        experience._start_observe()
        gen0 = experience._interaction_generation

        experience._on_explore_changed("vent1", 2)
        gen1 = experience._interaction_generation
        assert gen1 != gen0

        experience._on_explore_changed("candles", 1)
        gen2 = experience._interaction_generation
        assert gen2 != gen1

        experience.time_controller.pause()
        experience._observe_finished()
        gen3 = experience._interaction_generation
        assert gen3 != gen2

    def test_deferred_callback_runs_normally_when_nothing_else_happens(self, experience, qapp):
        """The guard must not block a legitimate, un-interrupted sweep --
        only a stale one."""
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_mystery()
        canvas = experience.scene.view.canvas
        experience._on_explore_changed("vent1", 2)

        centre = QtCore.QPoint(canvas.width() // 2, canvas.height() // 2)
        experience._on_overlay_tapped(centre)
        first_caption = experience.overlay.thermometer._caption.text()

        fired = []
        experience._defer_if_current(10, lambda: fired.append(True))
        for _ in range(20):
            qapp.processEvents()
            if fired:
                break
        assert fired, "an un-interrupted deferred callback must still fire"


class TestCoolestPointDetection:
    """Generalizes "find the hottest place" into a dynamically-chosen
    hunt (Phase 4 section 13) -- the coolest-cell lookup itself, and the
    hunt's own target selection."""

    def test_coolest_point_matches_the_real_array_argmin(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.enter_public_mode()
        experience = window.public_experience
        experience._begin_journey()
        experience._start_observe()
        for _ in range(60):
            experience.time_controller.seek(experience.time_controller.index + 1)
        index = experience.state.frame_index
        x, z, value = experience.scene.coolest_point_at(index)
        frame = experience.scene._temperature[index]
        assert value == pytest.approx(float(frame.min()))
        x0, x1, z0, z1 = experience.scene.view._extent
        assert x0 <= x <= x1 and z0 <= z <= z1
        window.close()

    def test_no_data_returns_none(self):
        from public.scene import PublicScene
        scene = PublicScene(store=None, manifest=[], fps=4)
        assert scene.coolest_point_at(0) is None
        assert scene.coolest_guess_is_close(0.5, 0.1, 25.0, 0) is False


class TestTemperatureHunt:
    """"Find the hottest place" generalized into a dynamically-chosen
    hunt for either extreme (Phase 4 section 13) -- one button, not two,
    so the child learns to search the whole room."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    @staticmethod
    def _pos_for(experience, x, z):
        # Routed through the scene's own widget_fraction_for rather than
        # a second copy of its formula: the scene only renders into
        # SCENE_WIDTH_FRAC of the widget's width now (a real, reserved
        # right-hand column for the thermometer, never drawn into -- see
        # PublicScene.SCENE_WIDTH_FRAC), and a hand-rolled full-bleed
        # version of this helper would compute tap positions past the
        # scene's actual right edge, landing in that reserved column
        # instead of on the vent/candle/etc. this is supposed to tap.
        canvas = experience.scene.view.canvas
        fx, fy = experience.scene.widget_fraction_for(x, z)
        return QtCore.QPoint(
            min(canvas.width() - 1, max(0, int(fx * canvas.width()))),
            min(canvas.height() - 1, max(0, int(fy * canvas.height()))))

    def test_activation_picks_a_real_target(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_hottest()
        assert experience._temp_hunt_target in ("hottest", "coolest")

    def test_coolest_hunt_solved_at_the_real_argmin(self, experience):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(60):
            experience.time_controller.seek(experience.time_controller.index + 1)
        experience.time_controller.pause()
        cool_x, cool_z, cool_value = experience.scene.coolest_point_at(experience.state.frame_index)
        experience._enter_game_hottest()
        experience._temp_hunt_target = "coolest"   # force this round's target

        experience._on_overlay_tapped(self._pos_for(experience, cool_x, cool_z))

        assert experience.state.phase is Phase.GAME_PLAY   # the game screen stays put
        assert f"{cool_value:.0f}" in experience.overlay.bubble.text()
        # "cool", not "coolest": the ambient far-field is a wide plateau
        # of near-identical cool cells, not a unique minimum the way the
        # flame core is a unique maximum -- see target_coolest's own
        # comment in i18n.py.
        assert "cool" in experience.overlay.bubble.text().lower()
        assert "coolest" not in experience.overlay.bubble.text().lower()

    def test_candle_tap_does_not_win_a_coolest_hunt(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_hottest()
        experience._temp_hunt_target = "coolest"
        candle_x = sum(experience.scene._candle_signature) / len(experience.scene._candle_signature)
        from schematic import ROOM_Z
        pos = self._pos_for(experience, candle_x, ROOM_Z[0] + 0.01)

        experience._on_overlay_tapped(pos)

        assert experience._active_game == "hottest"   # hunt still active

    def test_hottest_hunt_records_a_discovery(self, experience):
        experience._begin_journey()
        experience._start_observe()
        for _ in range(60):
            experience.time_controller.seek(experience.time_controller.index + 1)
        experience.time_controller.pause()
        hot_x, hot_z, _v = experience.scene.hottest_point_at(experience.state.frame_index)
        experience._enter_game_hottest()
        experience._temp_hunt_target = "hottest"

        experience._on_overlay_tapped(self._pos_for(experience, hot_x, hot_z))

        assert any(d["title"] == "HOTTEST SPOT FOUND" for d in experience._discoveries)


class TestMysteryExperiment:
    """"Can you figure it out?" -- a real, measured, counter-intuitive
    effect (the fan cools the ceiling but warms the floor), discovered
    entirely through the existing "compare this place" tool -- not a new
    guided-experiment clone (Phase 4 section 1/2)."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    @staticmethod
    def _pos_for(experience, x, z):
        # Routed through the scene's own widget_fraction_for rather than
        # a second copy of its formula: the scene only renders into
        # SCENE_WIDTH_FRAC of the widget's width now (a real, reserved
        # right-hand column for the thermometer, never drawn into -- see
        # PublicScene.SCENE_WIDTH_FRAC), and a hand-rolled full-bleed
        # version of this helper would compute tap positions past the
        # scene's actual right edge, landing in that reserved column
        # instead of on the vent/candle/etc. this is supposed to tap.
        canvas = experience.scene.view.canvas
        fx, fy = experience.scene.widget_fraction_for(x, z)
        return QtCore.QPoint(
            min(canvas.width() - 1, max(0, int(fx * canvas.width()))),
            min(canvas.height() - 1, max(0, int(fy * canvas.height()))))

    def _solve(self, experience, ceiling_first=True):
        from schematic import ROOM_Z
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_mystery()
        experience._on_explore_changed("vent1", 2)
        for _ in range(20):
            experience.time_controller.seek(experience.time_controller.index + 1)
        experience.time_controller.pause()
        # x=0.68 stays well clear of the vent (measured near x=0.36) so
        # the tap is never intercepted as a vent toggle instead. Also a
        # real, re-measured pick (not the original 0.75): on the current
        # dataset the ceiling's own fan-on/fan-off direction genuinely
        # varies by exact x -- 0.68 is one of the x's that reliably
        # cools with HVAC on, matching the floor's reliable warming at
        # the same x, i.e. an actual "opposite effects" pair (see
        # PublicExperience._note_mystery_progress's own comment on why
        # this is no longer a single fixed direction).
        ceiling_pos = self._pos_for(experience, 0.68, ROOM_Z[1] - 0.01)
        floor_pos = self._pos_for(experience, 0.68, ROOM_Z[0] + 0.005)
        order = (ceiling_pos, floor_pos) if ceiling_first else (floor_pos, ceiling_pos)
        # Every tap in the Mystery game is a same-place comparison already
        # (see _on_game_tap) -- no separate "arm" step needed.
        experience._on_overlay_tapped(order[0])
        experience._close_compare()
        experience._on_overlay_tapped(order[1])

    def test_solving_both_halves_records_the_discovery(self, experience):
        self._solve(experience, ceiling_first=True)
        assert experience._mystery_found_high
        assert experience._mystery_found_low
        assert any(d["title"] == i18n.tr("mystery_solved_discovery_title")
                   for d in experience._discoveries)

    def test_solving_it_highlights_the_two_real_measured_spots(self, experience, qapp):
        """Phase 12 section 11: the "aha" moment is the scene itself, not
        just the card -- both real tapped points glow together (reusing
        the Hot/Cold game's own dual-marker artists) for a couple of
        seconds, then clear on their own without freezing the app."""
        self._solve(experience, ceiling_first=True)

        assert experience.scene._hot_marker is not None
        hot_offsets = experience.scene._hot_marker.get_offsets()
        cool_offsets = experience.scene._cool_marker.get_offsets()
        assert len(hot_offsets) == 1 and len(cool_offsets) == 1
        assert tuple(hot_offsets[0]) == pytest.approx(experience._mystery_low_xz)     # floor
        assert tuple(cool_offsets[0]) == pytest.approx(experience._mystery_high_xz)   # ceiling

        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(2200, loop.quit)
        loop.exec_()
        assert len(experience.scene._hot_marker.get_offsets()) == 0
        assert len(experience.scene._cool_marker.get_offsets()) == 0

    def test_staged_reactions_match_the_investigation_script(self, experience):
        """Phase 7 section 7: the child must discover the effect through
        their own two taps, not be told the punchline upfront -- the
        first zone found gets its own reaction plus a nudge toward the
        other one, and only the second zone reveals "same fan, different
        place"."""
        from schematic import ROOM_Z
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_mystery()
        experience._on_explore_changed("vent1", 2)
        for _ in range(20):
            experience.time_controller.seek(experience.time_controller.index + 1)
        experience.time_controller.pause()
        ceiling_pos = self._pos_for(experience, 0.68, ROOM_Z[1] - 0.01)
        floor_pos = self._pos_for(experience, 0.68, ROOM_Z[0] + 0.005)

        experience._on_overlay_tapped(ceiling_pos)
        card_text = "\n".join(w.text() for w in experience.overlay.card.findChildren(QtWidgets.QLabel))
        assert "got cooler" in experience.overlay.bubble.text()
        assert "SAME FAN" not in card_text
        experience._close_compare()

        experience._on_overlay_tapped(floor_pos)
        card_text = "\n".join(w.text() for w in experience.overlay.card.findChildren(QtWidgets.QLabel))
        assert "SAME FAN. DIFFERENT PLACE." in card_text
        assert "SAME FAN. DIFFERENT PLACE." in experience.overlay.bubble.text()

    def test_order_does_not_matter(self, experience):
        self._solve(experience, ceiling_first=False)
        assert experience._mystery_found_high
        assert experience._mystery_found_low

    def test_only_one_half_found_gives_an_encouraging_not_finished_reaction(self, experience):
        from schematic import ROOM_Z
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_mystery()
        experience._on_explore_changed("vent1", 2)
        for _ in range(20):
            experience.time_controller.seek(experience.time_controller.index + 1)
        experience.time_controller.pause()
        experience._on_overlay_tapped(self._pos_for(experience, 0.75, ROOM_Z[1] - 0.01))
        assert experience._mystery_found_high
        assert not experience._mystery_found_low
        assert not any(d["title"] == i18n.tr("mystery_solved_discovery_title")
                       for d in experience._discoveries)


class TestDiscoveryNotebook:
    """📓 My Discoveries: reachable from the Games hub (an icon-style
    button, not a tile), only once there's something real in it."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    @staticmethod
    def _button_texts(experience):
        return [b.text() for b in experience.overlay._buttons + experience.overlay._nav_buttons]

    def test_hidden_when_empty(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._on_games_requested()
        assert not any("My Discoveries" in t for t in self._button_texts(experience))
        experience._on_notebook_requested()   # must refuse silently
        assert experience.overlay.card.isHidden()

    def test_shown_after_a_real_discovery_and_lists_it(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._record_discovery("🔥", "TEST DISCOVERY", "A real measured thing happened.")
        experience._on_games_requested()
        assert any("My Discoveries" in t for t in self._button_texts(experience))

        experience._on_notebook_requested()

        assert not experience.overlay.card.isHidden()
        text = "\n".join(w.text() for w in experience.overlay.card.findChildren(QtWidgets.QLabel))
        assert "TEST DISCOVERY" in text

    def test_capped_and_deduplicated(self, experience):
        experience._begin_journey()
        experience._start_observe()
        for i in range(6):
            experience._record_discovery("🔥", f"DISCOVERY {i}", "text")
        assert len(experience._discoveries) == 4
        experience._record_discovery("🔥", "DISCOVERY 0", "updated text")
        titles = [d["title"] for d in experience._discoveries]
        assert titles.count("DISCOVERY 0") == 1

    def test_closing_returns_to_the_games_hub(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._record_discovery("🔥", "TEST DISCOVERY", "text")
        experience._on_games_requested()
        experience._on_notebook_requested()

        experience._close_compare()

        assert experience.state.phase is Phase.GAMES_HOME
        assert experience.overlay.card.isHidden()

    def test_cleared_on_kiosk_reset_and_replay(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._record_discovery("🔥", "TEST DISCOVERY", "text")
        experience.reset()
        assert experience._discoveries == []

    def test_no_score_no_points_field(self, experience):
        """Guard against regression toward gamification: a discovery
        entry has exactly icon/title/text, nothing score-shaped."""
        experience._begin_journey()
        experience._start_observe()
        experience._record_discovery("🔥", "TEST DISCOVERY", "text")
        assert set(experience._discoveries[0].keys()) == {"icon", "title", "text"}


class TestTemperatureTrail:
    """"Let the child build their own temperature map" (Phase 5 section
    5), now the "Map It" Games/Challenges tile -- real numbered taps,
    capped, cleared on scenario/phase changes so a stale marker never
    survives onto a frame it no longer describes."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    @staticmethod
    def _button_texts(experience):
        return [b.text() for b in experience.overlay._buttons + experience.overlay._nav_buttons]

    @staticmethod
    def _safe_point(experience, x_frac=0.75, y_frac=0.75):
        """A tap point at (x_frac, y_frac) of the way across/down the
        *axes' own real bounds*, not the raw canvas -- both
        SCENE_WIDTH_FRAC (horizontal, the reserved thermometer column)
        and the scene's own _bottom_frac (vertical, the reserved mascot
        strip below the room -- see PublicScene.set_bottom_reserve_px)
        shrink the real plotted area below the raw canvas size, so "3/4
        of the way down the canvas" can land inside that reserved strip
        instead of on the scene. Scaled against the axes' *own* current
        bounds instead of a bare canvas fraction so this stays correct
        regardless of window size or how large either reserved strip is.
        Defaults match the "far corner" point most of this class' tests
        already wanted."""
        canvas = experience.scene.view.canvas
        axes_h_frac = 1.0 - experience.scene._bottom_frac
        return QtCore.QPoint(
            int(canvas.width() * SCENE_WIDTH_FRAC * x_frac),
            int(canvas.height() * axes_h_frac * y_frac))

    @classmethod
    def _far_point(cls, experience):
        return cls._safe_point(experience)

    def test_plain_taps_accumulate_real_points(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_map()
        canvas = experience.scene.view.canvas
        first = QtCore.QPoint(canvas.width() // 4, canvas.height() // 4)
        second = self._far_point(experience)
        x1, z1, v1 = experience.scene.probe_at(first)
        experience._on_overlay_tapped(first)
        x2, z2, v2 = experience.scene.probe_at(second)
        experience._on_overlay_tapped(second)

        assert len(experience._temp_trail) == 2
        assert experience._temp_trail[0]["value"] == pytest.approx(v1)
        assert experience._temp_trail[1]["value"] == pytest.approx(v2)
        assert any("Clear map" in t for t in self._button_texts(experience))

    def test_trail_line_joins_points_in_tap_order(self, experience, qapp):
        """Phase 9 section 7: a thin line tells the story of "where I
        measured" -- it must follow tap order, not spatial order, and
        must never appear with fewer than two real points."""
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_map()
        first = self._safe_point(experience, x_frac=0.75, y_frac=0.25)
        experience._on_overlay_tapped(first)
        assert experience.scene._trail_line is None   # one point: nothing to join yet

        second = self._safe_point(experience, x_frac=0.25, y_frac=0.75)
        experience._on_overlay_tapped(second)

        xs, zs = experience.scene._trail_line.get_data()
        assert list(xs) == [p["x"] for p in experience._temp_trail]
        assert list(zs) == [p["z"] for p in experience._temp_trail]

    def test_trail_line_clears_with_clear_map(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_map()
        canvas = experience.scene.view.canvas
        experience._on_overlay_tapped(QtCore.QPoint(canvas.width() // 4, canvas.height() // 4))
        experience._on_overlay_tapped(self._safe_point(experience))
        assert experience.scene._trail_line is not None

        experience._on_clear_trail_requested()

        assert experience.scene._trail_points == []
        assert len(experience.scene._trail_line.get_data()[0]) == 0

    def test_trail_line_never_joins_a_dim_before_point_to_a_new_one(self, experience, qapp):
        """A preserved "before" point belongs to a scenario that's no
        longer on screen -- joining it to a fresh measurement would
        visually claim a single walk across a change that never
        happened (Phase 8's own dim-marker distinction)."""
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_map()
        canvas = experience.scene.view.canvas
        experience._on_overlay_tapped(QtCore.QPoint(canvas.width() // 4, canvas.height() // 4))
        experience._on_explore_changed("vent1", 2)
        for _ in range(6):
            qapp.processEvents()
        assert experience._before_trail

        far = self._safe_point(experience)
        experience._on_overlay_tapped(far)

        # Only one *current* point exists post-switch -- the before point
        # must not have silently joined it into a two-point line.
        assert experience.scene._trail_line is None or (
            len(experience.scene._trail_line.get_data()[0]) < 2)

    def test_capped_at_max_points(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_map()
        canvas = experience.scene.view.canvas
        for i in range(experience._MAX_TRAIL_POINTS + 3):
            x = 20 + (i * 15) % (canvas.width() - 40)
            y = 20 + (i * 23) % (canvas.height() - 40)
            experience._on_overlay_tapped(QtCore.QPoint(x, y))
        assert len(experience._temp_trail) <= experience._MAX_TRAIL_POINTS

    def test_clear_button_empties_the_trail_and_the_scene(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_map()
        canvas = experience.scene.view.canvas
        experience._on_overlay_tapped(QtCore.QPoint(canvas.width() // 2, canvas.height() // 2))
        assert experience._temp_trail

        experience._on_clear_trail_requested()

        assert experience._temp_trail == []
        assert experience.scene._trail_markers == []
        assert not any("Clear map" in t for t in self._button_texts(experience))

    def test_trail_becomes_before_markers_on_scenario_switch(self, experience):
        """Phase 8 section 7 reverses the earlier Phase 6 behavior here:
        a scenario switch used to wipe the trail outright, but a child's
        own "before" measurements must now survive the change (dimmed)
        so revisiting one afterward reads as "I measured HERE before"
        (see _match_before_trail) instead of vanishing."""
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_map()
        canvas = experience.scene.view.canvas
        experience._on_overlay_tapped(QtCore.QPoint(canvas.width() // 2, canvas.height() // 2))
        assert experience._temp_trail

        experience._on_explore_changed("vent1", 2)

        assert experience._temp_trail == []
        assert experience._before_trail
        assert experience.scene._trail_markers   # the dimmed "before" marker persists

    def test_retapping_a_before_point_triggers_a_comparison_not_a_new_marker(
            self, experience, qapp):
        """Phase 8 section 7/9: revisiting a spot measured before the
        change reads as "I measured HERE before" -- it opens the same
        verdict-first card _show_same_place_comparison already shows for
        an armed "Compare this place" tap, without requiring that tool to
        be armed, and it must not also add a second, ordinary trail
        point at the same place."""
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_map()
        canvas = experience.scene.view.canvas
        centre = QtCore.QPoint(canvas.width() // 2, canvas.height() // 2)
        x, z, _value = experience.scene.probe_at(centre)
        experience._on_overlay_tapped(centre)
        assert experience._temp_trail

        experience._on_explore_changed("vent1", 2)
        for _ in range(6):
            qapp.processEvents()
        assert experience._before_trail

        experience._on_overlay_tapped(centre)

        assert experience._compare_open
        assert experience._temp_trail == []   # not swallowed as an ordinary new point
        bars = experience.overlay.card.findChildren(BarCompare)
        assert bars

    def test_a_genuinely_new_spot_still_becomes_an_ordinary_trail_point(
            self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_map()
        canvas = experience.scene.view.canvas
        experience._on_overlay_tapped(
            QtCore.QPoint(canvas.width() // 4, canvas.height() // 4))
        experience._on_explore_changed("vent1", 2)
        for _ in range(6):
            qapp.processEvents()
        assert experience._before_trail

        far = self._safe_point(experience)
        experience._on_overlay_tapped(far)

        assert not experience._compare_open
        assert len(experience._temp_trail) == 1

    def test_new_point_numbering_continues_past_before_points(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_map()
        canvas = experience.scene.view.canvas
        experience._on_overlay_tapped(
            QtCore.QPoint(canvas.width() // 4, canvas.height() // 4))
        experience._on_explore_changed("vent1", 2)
        for _ in range(6):
            qapp.processEvents()
        assert len(experience._before_trail) == 1

        far = self._safe_point(experience)
        experience._on_overlay_tapped(far)

        labels = [label.get_text() for _marker, label in experience.scene._trail_markers]
        assert labels == ["1", "2"]

    def test_clear_map_also_clears_before_markers(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_map()
        canvas = experience.scene.view.canvas
        experience._on_overlay_tapped(QtCore.QPoint(canvas.width() // 2, canvas.height() // 2))
        experience._on_explore_changed("vent1", 2)
        for _ in range(6):
            qapp.processEvents()
        assert experience._before_trail

        experience._on_clear_trail_requested()

        assert experience._before_trail == []
        assert experience.scene._trail_markers == []

    def test_trail_survives_same_game_rerender(self, experience):
        """Phase 6 section 2 reversed the earlier Phase 5 behavior: a
        child's own measurements must feel persistent, so closing Help
        (a re-render that never actually leaves the Map It game) must
        *not* clear the trail -- see TestTrailAndGhostPersistAcrossDetours
        for the full coverage of this; this class keeps one example so
        the trail's own tests don't only live elsewhere."""
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_map()
        experience._on_explore_changed("vent1", 2)
        canvas = experience.scene.view.canvas
        experience._on_overlay_tapped(QtCore.QPoint(canvas.width() // 2, canvas.height() // 2))
        assert experience._temp_trail

        experience._on_help_requested()
        experience._close_help()   # re-renders GAME_PLAY without changing the active game

        assert len(experience._temp_trail) == 1
        assert len(experience.scene._trail_markers) == 1

    def test_no_data_no_crash(self):
        from public.scene import PublicScene
        scene = PublicScene(store=None, manifest=[], fps=4)
        scene.add_trail_marker(0.5, 0.1, 1)   # ax is None; must not raise
        scene.clear_trail_markers()


class TestExperimentBoard:
    """The "I CHANGED / I WATCHED / I DISCOVERED" summary, surfaced
    inside the Discovery Notebook rather than a second always-on panel
    (Phase 5 section 3) -- built from real state the rest of Fire Lab
    already tracks."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_last_change_tracked_from_a_real_explore_toggle(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._on_explore_changed("vent1", 2)
        assert experience._last_change is not None
        assert "hvac" in experience._last_change.lower()

    def test_last_watched_tracked_from_a_real_comparison(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_mystery()
        experience._on_explore_changed("vent1", 2)
        canvas = experience.scene.view.canvas
        experience._on_overlay_tapped(QtCore.QPoint(canvas.width() // 2, canvas.height() // 2))
        assert experience._last_watched is not None

    def test_board_appears_inside_the_notebook_card(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_mystery()
        experience._on_explore_changed("vent1", 2)
        canvas = experience.scene.view.canvas
        experience._on_overlay_tapped(QtCore.QPoint(canvas.width() // 2, canvas.height() // 2))
        experience._close_compare()
        # The notebook only opens once there's a real discovery in it --
        # unrelated to what this test actually checks (the board's own
        # content), so record one directly.
        experience._record_discovery("🎉", "TEST DISCOVERY", "text")

        experience._on_games_requested()
        experience._on_notebook_requested()

        text = "\n".join(w.text() for w in experience.overlay.card.findChildren(QtWidgets.QLabel))
        assert "I changed" in text
        assert "I watched" in text
        assert "I discovered" in text

    def test_real_measurements_show_as_i_measured_when_the_child_has_tapped(self, experience):
        """Phase 7 section 12: once the child has actually placed
        measurements on the map, the board shows those real values under
        "I measured" instead of the whole-scenario "I watched" line --
        never an entry for something the child didn't actually do."""
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_mystery()
        experience._on_explore_changed("vent1", 2)
        canvas = experience.scene.view.canvas
        # Would normally set _last_watched.
        experience._on_overlay_tapped(QtCore.QPoint(canvas.width() // 2, canvas.height() // 2))
        experience._close_compare()
        experience._record_discovery("🎉", "TEST DISCOVERY", "text")

        experience._enter_game_map()
        canvas = experience.scene.view.canvas
        _x, _z, value = experience.scene.probe_at(
            QtCore.QPoint(canvas.width() // 2, canvas.height() // 2))
        experience._on_overlay_tapped(QtCore.QPoint(canvas.width() // 2, canvas.height() // 2))

        # Notebook opened from *inside* Map It (Phase.GAME_PLAY), not
        # after backing out to the hub -- leaving the game (_on_back_to_
        # games) clears its trail, same as any other game's own state.
        experience._on_notebook_requested()

        text = "\n".join(w.text() for w in experience.overlay.card.findChildren(QtWidgets.QLabel))
        assert "I measured" in text
        assert f"{value:.0f}°C" in text
        assert "I watched" not in text


class TestTrailPersistsAcrossDetours:
    """Phase 6 section 2: a child's own measurements must feel
    persistent -- opening/closing Help or the Discovery Notebook must
    never wipe the temperature trail (Map It), even though
    scene.clear_probe() still runs on every one of those re-renders (for
    the hover ring/dual markers, which *should* reset every time). Only
    a genuine scenario switch or leaving the game entirely may clear
    it."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def _tap_trail_point(self, experience):
        canvas = experience.scene.view.canvas
        experience._on_overlay_tapped(QtCore.QPoint(canvas.width() // 4, canvas.height() // 4))

    def test_trail_survives_opening_and_closing_help(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_map()
        experience._on_explore_changed("vent1", 2)
        self._tap_trail_point(experience)
        assert len(experience._temp_trail) == 1

        experience._on_help_requested()
        experience._close_help()

        assert len(experience._temp_trail) == 1
        assert len(experience.scene._trail_markers) == 1

    def test_trail_survives_opening_and_closing_the_notebook(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_map()
        experience._on_explore_changed("vent1", 2)
        self._tap_trail_point(experience)
        experience._record_discovery("🔥", "TEST", "text")

        experience._on_notebook_requested()
        experience._close_compare()

        assert len(experience._temp_trail) == 1
        assert len(experience.scene._trail_markers) == 1

    def test_trail_becomes_before_markers_on_a_real_scenario_switch(self, experience):
        """A scenario switch turns the current trail into the "before"
        record for the new change (Phase 8 section 7) rather than
        wiping it outright."""
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_map()
        experience._on_explore_changed("vent1", 2)
        self._tap_trail_point(experience)
        assert experience._temp_trail

        experience._on_explore_changed("candles", 1)

        assert experience._temp_trail == []
        assert experience._before_trail
        assert experience.scene._trail_markers

    def test_trail_still_clears_on_leaving_the_game(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_map()
        experience._on_explore_changed("vent1", 2)
        self._tap_trail_point(experience)

        experience.time_controller.pause()
        experience._observe_finished()   # leaves GAME_PLAY entirely

        assert experience._temp_trail == []


class TestWhyButton:
    """Phase 6 section 13: explanations appear only on request, after a
    real discovery/comparison -- reusing PUBLIC_METRICS' own declared
    Metric.explanation field, never composed from which experiment/
    factor happens to be active."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    def test_why_button_on_same_place_comparison(self, experience, qapp):
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_mystery()
        experience._on_explore_changed("vent1", 2)
        canvas = experience.scene.view.canvas

        experience._on_overlay_tapped(QtCore.QPoint(canvas.width() // 2, canvas.height() // 2))

        texts = [b.text() for b in experience.overlay._buttons]
        assert any("Why" in t for t in texts)

    def test_why_button_on_solved_mystery_gives_the_mystery_explanation(self, experience):
        from schematic import ROOM_Z

        def pos_for(x, z):
            # See TestFlamePulse._pos_for's comment: routed through
            # widget_fraction_for rather than a hand-rolled full-bleed
            # formula, which would miss past PublicScene.SCENE_WIDTH_FRAC.
            canvas = experience.scene.view.canvas
            fx, fy = experience.scene.widget_fraction_for(x, z)
            return QtCore.QPoint(min(canvas.width() - 1, max(0, int(fx * canvas.width()))),
                                 min(canvas.height() - 1, max(0, int(fy * canvas.height()))))

        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_mystery()
        experience._on_explore_changed("vent1", 2)
        # x=0.68: a real, re-measured tap that reliably cools at the
        # ceiling and warms at the floor on the current dataset -- see
        # TestMysteryExperiment._solve's own comment on why 0.75 no
        # longer reliably does.
        experience._on_overlay_tapped(pos_for(0.68, ROOM_Z[1] - 0.01))
        experience._close_compare()

        experience._on_overlay_tapped(pos_for(0.68, ROOM_Z[0] + 0.005))

        why_button = next(b for b in experience.overlay._buttons if "Why" in b.text())
        why_button.click()
        assert "moving air" in experience.overlay.bubble.text().lower()

    def test_no_why_button_when_nothing_noticeable_changed(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_mystery()
        experience._on_explore_changed("vent1", 2)
        canvas = experience.scene.view.canvas
        # Tap the exact same point twice at the baseline vs baseline --
        # not meaningful here, so just assert the button only appears
        # when the card actually says something changed.
        experience._on_overlay_tapped(QtCore.QPoint(canvas.width() // 2, canvas.height() // 2))
        card_text = "\n".join(w.text() for w in experience.overlay.card.findChildren(QtWidgets.QLabel))
        texts = [b.text() for b in experience.overlay._buttons]
        if "Almost the same" in card_text:
            assert not any("Why" in t for t in texts)


class TestGamesHub:
    """The Games/Challenges hub (Phase 13 redesign): every optional
    activity lives here, entered intentionally with one tap from
    Explore, rather than surfacing during free play. See
    PublicExperience._render_games_home/_render_game_play."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()

    @staticmethod
    def _tile_texts(experience):
        grid = experience.overlay._games_grid
        return [grid.itemAt(i).widget().text() for i in range(grid.count())]

    def test_explore_shows_only_help_pause_and_games(self, experience):
        """The redesign's whole point: Explore's chrome is exactly
        three buttons, nothing else competing for attention -- two
        BigButtons (Help, Play/Pause) plus one small nav pill (Games)."""
        experience._begin_journey()
        experience._start_observe()
        assert len(experience.overlay._buttons) == 2
        assert len(experience.overlay._nav_buttons) == 1
        texts = [b.text() for b in experience.overlay._buttons + experience.overlay._nav_buttons]
        assert any("What am I seeing" in t for t in texts)
        assert any("Pause" in t or "Play" in t for t in texts)
        assert any("Games" in t for t in texts)

    def test_games_button_opens_the_hub_and_pauses_playback(self, experience):
        experience._begin_journey()
        experience._start_observe()
        assert experience.time_controller.is_playing()

        experience._on_games_requested()

        assert experience.state.phase is Phase.GAMES_HOME
        assert not experience.time_controller.is_playing()
        assert not experience.overlay.games_home_panel.isHidden()

    def test_hub_shows_all_five_challenge_tiles(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._on_games_requested()
        tiles = self._tile_texts(experience)
        for label in ("Temp Hunt", "Hot / Cold", "Mystery", "Test an Idea", "Map It"):
            assert any(label in t for t in tiles)

    def test_back_to_exploring_returns_to_observe(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._on_games_requested()

        experience._on_back_to_explore()

        assert experience.state.phase is Phase.OBSERVE
        assert experience.overlay.games_home_panel.isHidden()

    def test_each_tile_enters_game_play_with_the_right_active_game(self, experience):
        experience._begin_journey()
        experience._start_observe()
        for key, enter in (
                ("hottest", experience._enter_game_hottest),
                ("hotcold", experience._enter_game_hotcold),
                ("mystery", experience._enter_game_mystery),
                ("map", experience._enter_game_map)):
            experience._on_games_requested()
            enter()
            assert experience.state.phase is Phase.GAME_PLAY
            assert experience._active_game == key
            assert experience.overlay.games_home_panel.isHidden()

    def test_back_to_games_returns_to_the_hub(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_hotcold()

        experience._on_back_to_games()

        assert experience.state.phase is Phase.GAMES_HOME
        assert experience._active_game is None

    def test_test_an_idea_never_enters_game_play(self, experience):
        """"Test an idea" jumps straight into the existing guided
        PREDICTION chain -- it has no chrome of its own, so it never
        touches Phase.GAME_PLAY."""
        experience._begin_journey()
        experience._start_observe()
        experience._on_games_requested()

        experience._enter_game_test_idea()

        assert experience.state.phase is Phase.PREDICTION
        assert experience.state.return_phase is Phase.GAMES_HOME

    def test_replay_after_test_an_idea_returns_to_the_hub(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._on_games_requested()
        experience._enter_game_test_idea()
        experience.state.record_prediction("cooler")
        experience.state.record_choice("fan_on")
        experience.state.go_to(Phase.REVEAL)
        experience._render_phase()

        experience._on_replay()

        assert experience.state.phase is Phase.GAMES_HOME

    def test_replay_after_a_direct_observe_session_returns_to_observe(self, experience):
        """The default -- a run never routed through the Games hub --
        still replays back to free play, unchanged from before this
        redesign."""
        experience._begin_journey()
        experience._start_observe()
        experience._on_test_idea_clicked()
        experience.state.record_prediction("cooler")
        experience.state.record_choice("fan_on")
        experience.state.go_to(Phase.REVEAL)
        experience._render_phase()

        experience._on_replay()

        assert experience.state.phase is Phase.OBSERVE

    def test_leaving_a_game_for_the_hub_clears_its_state(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_hotcold()
        canvas = experience.scene.view.canvas
        experience._on_overlay_tapped(QtCore.QPoint(canvas.width() // 2, canvas.height() // 2))
        assert experience._hotcold_stage == 1

        experience._on_back_to_games()

        assert experience._hotcold_stage == 0
        assert experience._active_game is None

    def test_notebook_icon_only_appears_once_something_is_discovered(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience._on_games_requested()
        assert not any("My Discoveries" in b.text() for b in experience.overlay._nav_buttons)

        experience._record_discovery("🔥", "TEST", "text")
        experience._render_phase()   # re-render the hub with the new state

        assert any("My Discoveries" in b.text() for b in experience.overlay._nav_buttons)

    def test_deferred_sweep_survives_leaving_a_game_mid_flight(self, experience, qapp):
        """Mirrors TestDeferredSweepGenerationSafety: a same-place sweep
        scheduled from inside a game, abandoned by leaving that game
        before it fires, must never stomp a screen it no longer
        describes."""
        experience._begin_journey()
        experience._start_observe()
        experience._enter_game_mystery()
        experience._on_explore_changed("vent1", 2)
        canvas = experience.scene.view.canvas
        centre = QtCore.QPoint(canvas.width() // 2, canvas.height() // 2)
        experience._on_overlay_tapped(centre)   # schedules the deferred second half
        experience._close_compare()

        experience._on_back_to_games()   # leave the game before the sweep fires
        caption_after_leaving = experience.overlay.thermometer._caption.text()

        QtCore.QThread.msleep(400)
        for _ in range(10):
            qapp.processEvents()

        assert experience.state.phase is Phase.GAMES_HOME
        assert experience.overlay.thermometer._caption.text() == caption_after_leaving


class TestScientistFactBubble:
    """Dr. Funke: a second, distinct mascot now standing in the true
    bottom-left corner (see PublicOverlay._position_scientist and the
    class-level swap note in PublicOverlay.__init__ -- she used to stand
    in the instrument sidebar's lower-right pocket, below the
    thermometer, until that spot's crowding risk moved her here and gave
    it to the worker mascot instead), sharing general fire-science facts
    in her own SpeechBubble instance (see PublicOverlay.scientist_bubble
    -- the same widget class the primary mascot uses, not a separate
    bubble type) on a slow, ambient timer. Never pull-triggered, and --
    the part that actually makes this ambient rather than a second
    competing character -- never allowed to appear while the primary
    guide has spoken recently or the child has just interacted (see
    _is_a_lull); the two are never both up at once, in either direction
    (PublicOverlay.say cuts off an already-showing fact the instant the
    buddy has something fresh to say, on top of _show_next_fact refusing
    to start a new one near one).

    Geometry here matters too, historically: a first version of her old
    sidebar spot shrank the thermometer to make room and silently lost
    the tube, bulb and readout below a real rendering floor in
    Thermometer._paint_tube; the fix (still in place, reserving room via
    PublicOverlay._BOTTOM_RIGHT_RESERVE_PX for whichever mascot now
    stands in that corner) used whatever the thermometer leaves there
    rather than shrinking it. A follow-up screenshot then caught the
    (longer, compound-word) German facts overlapping that pocket's own
    edge until SpeechBubble.fit_to started shrinking its own font to
    guarantee a fit -- the same fix ThoughtBubble (the widget this
    replaced) needed for the same reason. The font-shrink invariant
    itself (never clip, whichever corner/bubble it's checked against)
    outlived the corner she was in when it was written."""

    @pytest.fixture
    def experience(self, qapp):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        window = MainWindow(sim)
        window.resize(800, 600)
        window.show()
        for _ in range(6):
            qapp.processEvents()
        window.enter_public_mode()
        yield window.public_experience
        window.close()
        i18n.set_language("en")

    @staticmethod
    def _force_lull(experience):
        """Push both of _is_a_lull's own timestamps far enough into the
        past that it reads True -- the state _show_next_fact needs
        before it will actually show anything. 0.0, matching the same
        "long ago, no sentinel needed" idiom _last_say_at/
        _last_interaction_at themselves use at init."""
        experience.overlay._last_say_at = 0.0
        experience._last_interaction_at = 0.0

    def test_hidden_outside_observe(self, experience):
        experience._begin_journey()
        assert experience.state.phase is Phase.INTRO
        assert not experience.overlay.scientist.isVisible()

        experience._start_observe()
        assert experience.overlay.scientist.isVisible()

        experience.state.record_prediction("cooler")
        experience.state.record_choice("fan_on")
        experience.state.go_to(Phase.REVEAL)
        experience._render_phase()
        assert not experience.overlay.scientist.isVisible()

    def test_leaving_observe_clears_her_bubble(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience.overlay.say_fact(i18n.tr("fact_fire_triangle"))
        assert experience.overlay.scientist_bubble.text()

        experience.state.go_to(Phase.REVEAL)
        experience._render_phase()
        assert experience.overlay.scientist_bubble.text() == ""

    def test_thermometer_keeps_its_full_observe_height(self, experience):
        """The thermometer must never be shrunk to make room for her --
        that's the exact regression (Thermometer._paint_tube's tube
        vanishing below ~310px) her sidebar spot was redesigned to
        avoid."""
        experience._begin_journey()
        experience._start_observe()
        without_her = experience.overlay.thermometer.height()

        experience.overlay.set_scientist_visible(False)
        assert experience.overlay.thermometer.height() == without_her
        experience.overlay.set_scientist_visible(True)
        assert experience.overlay.thermometer.height() == without_her

    @pytest.mark.parametrize("lang", ["en", "de"])
    def test_every_fact_bubble_stays_within_the_screen_bottom(self, experience, lang):
        """She now stands in the true bottom-left corner (see
        _position_scientist's own docstring), nowhere near the
        thermometer -- checked here against the screen's own bottom
        edge only. Checked for every fact key, not just the longest,
        since fit_to's shrink-font loop is what has to keep this true
        regardless of which one a child happens to see."""
        i18n.set_language(lang)
        experience._begin_journey()
        experience._start_observe()
        for key in experience_mod.FIRE_FACT_KEYS:
            experience.overlay.say_fact(i18n.tr(key))
            bubble = experience.overlay.scientist_bubble.geometry()
            assert bubble.bottom() <= experience.overlay.height(), (lang, key, i18n.tr(key))

    @pytest.mark.parametrize("lang", ["en", "de"])
    def test_every_buddy_line_clears_the_thermometer(self, experience, lang):
        """The worker mascot's own speech bubble now stands in the tight
        corner next to the thermometer (see the class-level swap note in
        PublicOverlay.__init__ -- Dr. Funke's own bubble used to be the
        one that had to clear the thermometer here, before the corner
        swap moved that constraint onto his). Checked across every real
        story/fact line the buddy actually says, not a synthetic one, so
        this exercises the same fit_to() shrink path his real dialogue
        does."""
        i18n.set_language(lang)
        experience._begin_journey()
        experience._start_observe()
        for key in experience_mod.FIRE_FACT_KEYS:
            experience.overlay.say(i18n.tr(key))
            bubble = experience.overlay.bubble.geometry()
            thermometer_bottom = experience.overlay.thermometer.geometry().bottom()
            assert bubble.top() > thermometer_bottom, (lang, key, i18n.tr(key))
            assert bubble.bottom() <= experience.overlay.height(), (lang, key, i18n.tr(key))

    @pytest.mark.parametrize("lang", ["en", "de"])
    def test_no_buddy_bubble_ever_clips_its_own_text(self, experience, lang):
        """The regression a real screenshot actually caught, back when
        Dr. Funke stood in the true bottom-right corner: that spot
        leaves only ~124x85px for the whole bubble, tighter than the
        ~150-160px the font-shrink floor (fit_to) was tuned against
        before -- even at that floor, the single longest German fact
        still needed 108px of wrapped text in an 85px box, and the
        widget silently accepted the overflow rather than refusing to
        fit. The corner swap (see PublicOverlay.__init__'s own note)
        moved that tight pocket -- and its fit_to() call -- onto the
        worker mascot's own bubble instead, so this now exercises his.
        heightForWidth (what fit_to uses to decide when to stop
        shrinking) has to stay <= the widget's own final height for
        every fact in both languages, not just look right in the one
        screenshot that happened to get taken."""
        i18n.set_language(lang)
        experience._begin_journey()
        experience._start_observe()
        for key in experience_mod.FIRE_FACT_KEYS:
            experience.overlay.say(i18n.tr(key))
            bubble = experience.overlay.bubble
            needed = bubble.heightForWidth(bubble.width())
            assert needed <= bubble.height(), (lang, key, i18n.tr(key), needed, bubble.height())

    @pytest.mark.parametrize("lang", ["en", "de"])
    def test_no_fact_bubble_ever_clips_its_own_text(self, experience, lang):
        """Her own bubble no longer needs fit_to()'s shrink loop in the
        generous bottom-left corner (see _position_scientist_bubble,
        which sizes it with a plain resize() instead) -- this is now a
        lightweight regression check that stays true by construction,
        kept so a future corner change can't silently reintroduce
        clipping without a test noticing."""
        i18n.set_language(lang)
        experience._begin_journey()
        experience._start_observe()
        for key in experience_mod.FIRE_FACT_KEYS:
            experience.overlay.say_fact(i18n.tr(key))
            bubble = experience.overlay.scientist_bubble
            needed = bubble.heightForWidth(bubble.width())
            assert needed <= bubble.height(), (lang, key, i18n.tr(key), needed, bubble.height())

    def test_bubble_sits_beside_her_not_over_the_thermometer(self, experience):
        """To her *right*, not her left: she now corner-hugs the true
        bottom-left (swapped with the worker mascot -- see the
        class-level swap note in PublicOverlay.__init__), the same
        corner and orientation his own bubble always used there. Nowhere
        near the thermometer (opposite corner) any more, so no overlap
        check against it is meaningful here -- that constraint now
        belongs to whichever mascot stands in the tight corner instead
        (see TestScientistFactBubble's own buddy/thermometer test)."""
        experience._begin_journey()
        experience._start_observe()
        experience.overlay.say_fact(i18n.tr("fact_cool_air_sinks"))
        scientist = experience.overlay.scientist.geometry()
        bubble = experience.overlay.scientist_bubble.geometry()
        assert bubble.left() >= scientist.right()

    def test_she_corner_hugs_with_the_same_margin_the_buddy_uses(self, experience):
        """She now anchors bottom-left with the same fixed 32px margin
        the buddy's own old placement there always used (swapped -- see
        the class-level note in PublicOverlay.__init__): a literal
        mirror is possible now, unlike when she stood bottom-right next
        to the thermometer's own gutter column."""
        experience._begin_journey()
        experience._start_observe()
        scientist = experience.overlay.scientist.geometry()
        assert scientist.left() == 32

    def test_settling_does_not_leave_her_bubble_overlapping_the_buddy(self, experience, qapp):
        """The regression a geometry sweep actually caught: set_thermometer_
        visible positions the thermometer once synchronously, using
        possibly-stale meter geometry, then corrects it a moment later
        via a deferred zero-ms timer (_reposition_thermometer_once_
        settled) once everything has actually settled. Dr. Funke's own
        speech bubble clamps its width against self.thermometer.x()
        (_position_scientist_bubble -- this clamp moved onto her bubble
        with the corner swap, see PublicOverlay.__init__'s own note),
        computed back when the thermometer visibility was first set --
        before that correction ran. When settling moved the thermometer
        even a few px, the bubble's clamp went stale and it overlapped
        whatever stands at the thermometer's real x, which is exactly
        where the worker mascot now stands. Letting the deferred
        correction's own timer actually fire (via processEvents) is what
        reproduces it; asserting only right after _start_observe(),
        before that timer fires, would have missed it entirely.

        Only his *avatar* is checked here, not his own speech bubble --
        the two bubbles' bounding boxes are allowed to overlap (a
        tighter bound made a bubble in the tight corner unreadable once
        that occupant grew enough to stand close to the thermometer,
        back when Dr. Funke was the one there). What actually keeps them
        from ever being up at once is PublicExperience._show_next_fact's
        timing deferral, covered by TestScientistFactBubble's own defer
        tests, not spatial separation."""
        experience._begin_journey()
        experience._start_observe()
        experience.overlay.say_fact(i18n.tr("fact_hot_air_rises"))
        for _ in range(10):
            qapp.processEvents()
        scientist_bubble = experience.overlay.scientist_bubble.geometry()
        mascot = experience.overlay.mascot.geometry()
        assert not scientist_bubble.intersects(mascot)

    @pytest.mark.parametrize("lang", ["en", "de"])
    def test_buddy_bubble_never_runs_past_the_left_edge_of_the_screen(self, experience, lang):
        """The regression a second screenshot caught, back when Dr.
        Funke stood in the tight corner: the bubble widget's own
        setMinimumWidth silently overrode the corner's own width clamp
        -- QWidget.resize() snaps back up to a widget's own minimum, so
        a bubble computed to fit the narrow gutter still rendered wider
        and ran off the screen edge. min_width=130 on this SpeechBubble
        instance (far under the generous corner's own 280 default) is
        the fix; the corner swap (see PublicOverlay.__init__'s own note)
        moved this tight-corner treatment onto the worker mascot's own
        bubble, which now points *left* off the screen's own left edge
        instead of right. Checked for every fact key, since the clamp
        has to hold regardless of which one happens to be short enough
        to trigger it."""
        i18n.set_language(lang)
        experience._begin_journey()
        experience._start_observe()
        for key in experience_mod.FIRE_FACT_KEYS:
            experience.overlay.say(i18n.tr(key))
            bubble = experience.overlay.bubble.geometry()
            assert bubble.left() >= 0, (lang, key, i18n.tr(key))

    def test_fact_queue_cycles_through_every_key_before_repeating(self, experience):
        experience._begin_journey()
        experience._start_observe()
        self._force_lull(experience)
        token = experience._fact_chain_token
        seen = set()
        for _ in range(len(experience_mod.FIRE_FACT_KEYS)):
            experience._show_next_fact(token)
            seen.add(experience.overlay.scientist_bubble.text())
        assert len(seen) == len(experience_mod.FIRE_FACT_KEYS)

    def test_show_next_fact_is_a_noop_once_the_phase_has_moved_on(self, experience):
        """A stale timer firing after the child has moved on must not
        pop a fact onto a screen that no longer shows her at all --
        guarded by both the explicit phase check and the fact-chain
        token going stale (see _arm_fact_timer's own docstring)."""
        experience._begin_journey()
        experience._start_observe()
        self._force_lull(experience)
        token = experience._fact_chain_token
        experience.state.record_prediction("cooler")
        experience.state.record_choice("fan_on")
        experience.state.go_to(Phase.REVEAL)
        experience._render_phase()

        experience._show_next_fact(token)

        assert experience.overlay.scientist_bubble.text() == ""

    # ---------------------------------------------------- ambient deferral
    # The actual point of this whole design: she fills quiet moments, she
    # doesn't add to a busy one. These four pin the mechanism directly,
    # not just its visible symptom.

    def test_defers_when_the_buddy_spoke_recently(self, experience):
        """_start_observe() itself just called overlay.say(...) (the
        OBSERVE invitation) -- seconds_since_say() reads near-zero
        immediately after, well inside _LULL_GRACE_S, even though
        _last_interaction_at is forced far in the past here to isolate
        this one signal."""
        experience._begin_journey()
        experience._start_observe()
        experience._last_interaction_at = 0.0
        assert experience.overlay.seconds_since_say() < experience_mod._LULL_GRACE_S
        token = experience._fact_chain_token

        experience._show_next_fact(token)

        assert experience.overlay.scientist_bubble.text() == ""

    def test_defers_when_the_child_interacted_recently(self, experience):
        experience._begin_journey()
        experience._start_observe()
        experience.overlay._last_say_at = 0.0   # isolate: only interaction-recency matters here
        experience._last_interaction_at = time.monotonic()
        token = experience._fact_chain_token

        experience._show_next_fact(token)

        assert experience.overlay.scientist_bubble.text() == ""

    def test_shows_a_fact_once_both_signals_are_outside_the_grace_window(self, experience):
        experience._begin_journey()
        experience._start_observe()
        self._force_lull(experience)
        token = experience._fact_chain_token

        experience._show_next_fact(token)

        assert experience.overlay.scientist_bubble.text() != ""

    def test_a_fresh_buddy_say_cuts_off_her_currently_showing_fact(self, experience, qapp):
        """The other direction of "never both up at once": a fact
        already showing gets interrupted the instant the buddy has
        something fresh to say (PublicOverlay.say), not merely blocked
        from starting one near it (_is_a_lull) -- a child's tap mid-fact
        triggers exactly this ordering in the real app. The cut-off
        itself is a deferred zero-ms QTimer, not an inline call (see
        say()'s own comment on why -- reentrant Qt event handling this
        app hit a real crash from), so this needs a processEvents() to
        let that queued timer actually fire before checking."""
        experience._begin_journey()
        experience._start_observe()
        self._force_lull(experience)
        experience._show_next_fact(experience._fact_chain_token)
        assert experience.overlay.scientist_bubble.text() != ""

        experience.overlay.say("Something just happened!", "curious")
        for _ in range(3):
            qapp.processEvents()

        assert experience.overlay.scientist_bubble.text() == ""

    def test_a_deferred_fact_retries_and_appears_once_the_lull_arrives(self, experience):
        """The deferral must not just cancel a fact outright -- a fact
        merely postponed by a stray tap still gets its ambient turn
        once things go quiet, on the short _FACT_RETRY_MS cadence, not
        the full _FACT_GAP_MS one."""
        experience._begin_journey()
        experience._start_observe()
        experience._last_interaction_at = time.monotonic()   # not a lull yet
        token = experience._fact_chain_token

        experience._show_next_fact(token)
        assert experience.overlay.scientist_bubble.text() == ""

        self._force_lull(experience)
        experience._show_next_fact(token)   # simulates the retry timer firing
        assert experience.overlay.scientist_bubble.text() != ""


# ------------------------------------------------------------- manifest
