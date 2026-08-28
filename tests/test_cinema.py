"""Unit tests for cinema/luts.py + cinema/pipeline.py (FireLab roadmap
Phase 2, tasks 1-4: FireLUT + alpha + filmic tone map + auto-exposure,
bloom + HRR-driven flicker + sub-frame interpolation, smoke tiers 1-2,
heat shimmer + ember particles). No Qt/matplotlib canvas involved -- pure
array math, see test_views.py's TestSliceViewCinematicMode for the
SliceView/scatter-artist integration."""

import numpy as np
import pytest

from cinema.bloom import apply_bloom
from cinema.interp import lerp_frames
from cinema.luts import FIRE_RGBA_LUT
from cinema.particles import EmberParticles
from cinema.pipeline import AutoExposure, EffectsPipeline, _suppress_haze_over_smoke, filmic_tonemap
from cinema.real_smoke import REFERENCE_DENSITY, normalize_soot_density, soot_at_time
from cinema.shimmer import HeatShimmer
from cinema.smoke import SmokeSimulator, composite_over, smoke_rgba
from data_provider import load_simulation_data
from slice_key import DEFAULT_SLICE_KEY, SliceKey


class TestFireLUT:
    def test_shape_and_dtype(self):
        assert FIRE_RGBA_LUT.shape == (256, 4)
        assert FIRE_RGBA_LUT.dtype == np.uint8

    def test_ambient_end_is_transparent(self):
        assert FIRE_RGBA_LUT[0, 3] == 0

    def test_hot_end_is_opaque(self):
        assert FIRE_RGBA_LUT[-1, 3] == 255

    def test_alpha_is_monotonically_non_decreasing(self):
        alpha = FIRE_RGBA_LUT[:, 3].astype(np.int64)
        assert (np.diff(alpha) >= 0).all()


class TestFilmicTonemap:
    def test_zero_maps_to_zero(self):
        assert filmic_tonemap(np.array([0.0]))[0] == 0.0

    def test_never_exceeds_one_and_approaches_it(self):
        # A Reinhard-style shoulder rolls off toward 1.0 without ever
        # hard-clipping to it -- that's what avoids the "flat white blob"
        # look a plain Normalize gives flashover frames.
        t = np.linspace(0.0, 1.0, 50)
        out = filmic_tonemap(t)
        assert out[-1] < 1.0
        assert out[-1] > 0.75

    def test_compresses_relative_to_linear_in_upper_range(self):
        # Filmic curve should sit at-or-below the identity line for a
        # Reinhard-style shoulder to actually roll off highlights.
        t = np.linspace(0.0, 1.0, 50)
        assert (filmic_tonemap(t) <= t + 1e-9).all()

    def test_monotonically_increasing(self):
        t = np.linspace(0.0, 1.0, 100)
        assert (np.diff(filmic_tonemap(t)) >= 0).all()


class TestAutoExposure:
    def test_locked_never_moves(self):
        exp = AutoExposure(vmax_init=300.0)
        exp.locked = True
        exp.update(np.full((10, 10), 900.0))
        assert exp.vmax == 300.0

    def test_unlocked_tracks_toward_hotter_frames(self):
        exp = AutoExposure(vmax_init=300.0, tau_frames=4.0)
        vmax_before = exp.vmax
        for _ in range(50):
            exp.update(np.full((10, 10), 900.0))
        assert exp.vmax > vmax_before
        assert abs(exp.vmax - 900.0) < 1.0, "should converge close to the sustained percentile"


class TestBloom:
    def test_shape_and_dtype_preserved(self):
        rgba = FIRE_RGBA_LUT[np.full((20, 20), 255, dtype=np.uint8)]
        intensity = np.ones((20, 20), dtype=np.float32)
        out = apply_bloom(rgba, intensity)
        assert out.shape == rgba.shape
        assert out.dtype == np.uint8

    def test_hot_spot_raises_alpha_in_neighboring_ambient_pixels(self):
        """The glow's whole point: light spills into pixels that were
        fully transparent (ambient), not just brightens already-hot ones."""
        intensity = np.zeros((21, 21), dtype=np.float32)
        intensity[10, 10] = 1.0
        rgba = np.zeros((21, 21, 4), dtype=np.uint8)
        rgba[10, 10] = FIRE_RGBA_LUT[-1]
        out = apply_bloom(rgba, intensity, strength=1.0)
        assert out[10, 8, 3] > 0, "a nearby ambient pixel should pick up some halo alpha"

    def test_zero_intensity_no_glow(self):
        rgba = FIRE_RGBA_LUT[np.zeros((10, 10), dtype=np.uint8)]
        intensity = np.zeros((10, 10), dtype=np.float32)
        out = apply_bloom(rgba, intensity)
        assert (out == rgba).all()

class TestSuppressHazeOverSmoke:
    """cinema.pipeline._suppress_haze_over_smoke: the actual mechanism
    behind the 'smoke plume reads as orange' report, applied to the
    fully-composed (LUT + bloom) fire layer rather than to bloom alone
    -- see its own docstring for the real, measured counter-example
    (a strongly-ventilated flame's own lower peak temperature) that
    ruled out an opacity-based version of this check."""

    def test_haze_pixel_under_dense_smoke_loses_most_of_its_alpha(self):
        frame = np.full((21, 21), 20.0, dtype=np.float32)
        frame[10, 8] = 100.0   # warm, but nowhere near flame-core temperature
        fire_rgba = np.zeros((21, 21, 4), dtype=np.uint8)
        fire_rgba[10, 8] = [255, 187, 7, 255]   # a fully-opaque, warm pixel (the real regression)
        density = np.zeros((21, 21), dtype=np.float32)
        density[10, 8] = 1.0

        out = _suppress_haze_over_smoke(fire_rgba, frame, density)
        assert out[10, 8, 3] < fire_rgba[10, 8, 3]

    def test_true_flame_core_is_untouched_regardless_of_smoke_density(self):
        """Opacity alone can't tell a hot haze pixel from a true flame
        pixel (a first version of this check used alpha and broke on
        exactly this), but raw temperature always can: >= the ceiling
        means "still burning," and that alpha survives even saturated
        smoke density on the same cell."""
        frame = np.full((21, 21), 20.0, dtype=np.float32)
        frame[10, 8] = 400.0   # real flame-core territory
        fire_rgba = np.zeros((21, 21, 4), dtype=np.uint8)
        fire_rgba[10, 8] = [255, 120, 20, 255]
        density = np.zeros((21, 21), dtype=np.float32)
        density[10, 8] = 1.0

        out = _suppress_haze_over_smoke(fire_rgba, frame, density)
        assert out[10, 8, 3] == fire_rgba[10, 8, 3] == 255

    def test_no_smoke_no_suppression_even_for_a_haze_pixel(self):
        frame = np.full((21, 21), 20.0, dtype=np.float32)
        frame[10, 8] = 100.0
        fire_rgba = np.zeros((21, 21, 4), dtype=np.uint8)
        fire_rgba[10, 8] = [255, 187, 7, 200]
        density = np.zeros((21, 21), dtype=np.float32)   # no smoke anywhere

        out = _suppress_haze_over_smoke(fire_rgba, frame, density)
        assert out[10, 8, 3] == fire_rgba[10, 8, 3]

    def test_shape_and_dtype_preserved(self):
        frame = np.full((12, 12), 20.0, dtype=np.float32)
        fire_rgba = np.zeros((12, 12, 4), dtype=np.uint8)
        density = np.zeros((12, 12), dtype=np.float32)
        out = _suppress_haze_over_smoke(fire_rgba, frame, density)
        assert out.shape == fire_rgba.shape
        assert out.dtype == np.uint8


class TestLerpFrames:
    def test_endpoints_and_midpoint(self):
        a = np.zeros((3, 3), dtype=np.float32)
        b = np.full((3, 3), 10.0, dtype=np.float32)
        assert (lerp_frames(a, b, 0.0) == a).all()
        assert (lerp_frames(a, b, 1.0) == b).all()
        assert (lerp_frames(a, b, 0.5) == 5.0).all()

    def test_clamps_out_of_range_t(self):
        a = np.zeros((2, 2), dtype=np.float32)
        b = np.full((2, 2), 10.0, dtype=np.float32)
        assert (lerp_frames(a, b, -1.0) == a).all()
        assert (lerp_frames(a, b, 2.0) == b).all()


class TestSmokeSimulator:
    def test_no_source_below_threshold_stays_empty(self):
        sim = SmokeSimulator((10, 10), ambient_c=20.0)
        ambient_frame = np.full((10, 10), 20.0, dtype=np.float32)
        for _ in range(5):
            density = sim.step(ambient_frame)
        assert (density == 0.0).all()

    def test_hot_frame_accumulates_then_decays(self):
        sim = SmokeSimulator((10, 10), ambient_c=20.0)
        hot_frame = np.full((10, 10), 300.0, dtype=np.float32)
        d1 = sim.step(hot_frame).copy()
        d2 = sim.step(hot_frame).copy()
        assert d2.sum() > d1.sum(), "sustained heat should keep building smoke density"
        ambient_frame = np.full((10, 10), 20.0, dtype=np.float32)
        for _ in range(50):
            after_decay = sim.step(ambient_frame)
        assert after_decay.sum() < d2.sum(), "removing the source should let density decay away"

    def test_tier2_moves_mass_toward_velocity_direction(self):
        """A point source with a strong velocity field should advect
        differently than Tier 1's fixed drift -- exercises the tier-2
        (velocity_frame given) code path without asserting exact physics."""
        shape = (21, 21)
        sim = SmokeSimulator(shape, ambient_c=20.0)
        frame = np.full(shape, 20.0, dtype=np.float32)
        frame[10, 10] = 400.0  # a single hot cell as the plume source
        velocity = np.full(shape, 3.0, dtype=np.float32)
        for _ in range(10):
            density = sim.step(frame, velocity_frame=velocity)
        assert density.sum() > 0.0
        assert np.isfinite(density).all()


class TestSootAtTime:
    """cinema/real_smoke.py: the temporal-alignment function Architecture
    C depends on -- TEMPERATURE and SOOT DENSITY are recorded on
    different real output schedules (~481 vs ~1001 frames over the same
    interval on the real dataset), so pairing them by frame *index*
    would silently pair mismatched instants. These use small synthetic
    arrays (no real dataset needed) to pin down soot_at_time()'s exact
    boundary behavior in isolation."""

    TIMES = np.array([0.0, 1.0, 2.5, 4.0, 6.0])
    FRAMES = np.array([np.full((2, 2), v) for v in (0.0, 10.0, 20.0, 40.0, 60.0)])

    def test_exact_first_timestamp_returns_that_frame_unmodified(self):
        out = soot_at_time(self.FRAMES, self.TIMES, 0.0)
        assert (out == 0.0).all()

    def test_exact_last_timestamp_returns_that_frame_unmodified(self):
        out = soot_at_time(self.FRAMES, self.TIMES, 6.0)
        assert (out == 60.0).all()

    def test_exact_interior_timestamp_returns_that_frame_unmodified(self):
        out = soot_at_time(self.FRAMES, self.TIMES, 2.5)
        assert (out == 20.0).all()

    def test_between_two_timestamps_matches_hand_computed_linear_interpolation(self):
        # Halfway between t=0 (density 0) and t=1 (density 10).
        out = soot_at_time(self.FRAMES, self.TIMES, 0.5)
        assert out[0, 0] == pytest.approx(5.0)
        # A non-half fraction, between t=2.5 (20) and t=4.0 (40).
        expected = 20.0 + (3.0 - 2.5) / (4.0 - 2.5) * (40.0 - 20.0)
        out2 = soot_at_time(self.FRAMES, self.TIMES, 3.0)
        assert out2[0, 0] == pytest.approx(expected)

    def test_no_extrapolation_before_first_timestamp(self):
        """A target time before the recorded interval clamps to the
        first frame -- it must never linearly project past real data."""
        out = soot_at_time(self.FRAMES, self.TIMES, -5.0)
        assert (out == 0.0).all()

    def test_no_extrapolation_after_last_timestamp(self):
        out = soot_at_time(self.FRAMES, self.TIMES, 100.0)
        assert (out == 60.0).all()

    def test_single_frame_series_returns_that_frame(self):
        out = soot_at_time(self.FRAMES[:1], self.TIMES[:1], 5.0)
        assert (out == 0.0).all()


class TestNormalizeSootDensity:
    def test_zero_density_is_zero_opacity(self):
        assert normalize_soot_density(np.array([0.0]))[0] == 0.0

    def test_reference_density_maps_to_full_opacity(self):
        assert normalize_soot_density(np.array([REFERENCE_DENSITY]))[0] == pytest.approx(1.0)

    def test_density_above_reference_clips_to_one_not_beyond(self):
        assert normalize_soot_density(np.array([REFERENCE_DENSITY * 5]))[0] == 1.0

    def test_never_negative(self):
        assert normalize_soot_density(np.array([-10.0]))[0] == 0.0


class TestEffectsPipelineRealSmokeIntegration:
    """Confirms EffectsPipeline.render() actually uses a passed-in real
    density directly, and -- the point of Architecture C -- never falls
    back to instantiating the synthetic SmokeSimulator (with its
    buoyancy/decay/reservoir constants) when real density is supplied.
    The synthetic path is exercised separately by TestSmokeSimulator/
    TestSmokeDoesNotHomogenizeAcrossHeight below, unchanged, since it's
    still the fallback for callers with no real-soot alignment (the
    researcher app's generic Cinematic fire view toggle)."""

    def test_real_density_frame_is_used_directly(self):
        shape = (10, 10)
        pipe = EffectsPipeline(vmin=20.0, vmax_init=300.0)
        frame = np.full(shape, 20.0, dtype=np.float32)
        density = np.zeros(shape, dtype=np.float32)
        density[5, 5] = 1.0
        out = pipe.render(frame, smoke_density_frame=density)
        assert out.shape == shape + (4,)
        assert out[5, 5, 3] > out[0, 0, 3], "the real-density hot cell must be more opaque than empty cells"

    def test_synthetic_smoke_simulator_never_instantiated_when_real_density_given(self):
        shape = (10, 10)
        pipe = EffectsPipeline(vmin=20.0, vmax_init=300.0)
        frame = np.full(shape, 20.0, dtype=np.float32)
        density = np.full(shape, 0.5, dtype=np.float32)
        for _ in range(5):
            pipe.render(frame, smoke_density_frame=density)
        assert pipe._smoke is None, "the synthetic SmokeSimulator must stay uninstantiated on the real-soot path"

    def test_omitting_real_density_still_falls_back_to_synthetic_smoke(self):
        """The researcher app's generic cinematic toggle has no per-
        scenario real-soot alignment plumbed to it -- render() without
        smoke_density_frame must keep working exactly as before."""
        shape = (10, 10)
        pipe = EffectsPipeline(vmin=20.0, vmax_init=300.0)
        frame = np.full(shape, 400.0, dtype=np.float32)
        for _ in range(3):
            pipe.render(frame)
        assert pipe._smoke is not None
        assert isinstance(pipe._smoke, SmokeSimulator)


class TestSmokeDoesNotHomogenizeAcrossHeight:
    """A real, reproduced regression: pushing BUOYANCY_SPEED /
    TIER2_BASE_BUOYANCY_FRAC high enough to chase ceiling/floor density
    ratio (~7.6 rows/frame constant vertical speed, against a 49-row
    grid -- CFL number far above 1) made map_coordinates' semi-Lagrangian
    step homogenize the buffer across height within ~144 frames: density
    stopped varying with row at all and became a flat function of column
    only -- a hard-edged vertical band with no plume shape, not real
    stratification. The ceiling/floor ratio that version reported
    "improved" was this homogenization, not genuine physics.

    This is a distinct failure mode from "not stratified enough" (which
    TestSmokeSimulator and the wider stratification benchmark cover) --
    a field can have a perfectly plausible ceiling/floor ratio while
    still being flat/broken internally, so it needs its own check.
    Plain row-to-row correlation (row 0 vs row -1) does NOT reliably
    catch this: healthy stratified data can *also* have a highly
    correlated top/bottom shape (both peak near the same column, near
    the same source) even with very different absolute magnitudes --
    measured on this exact regression, corr ranged 0.76-0.99, which
    overlaps the corrected field's own 0.67-0.99. What actually
    discriminates is per-column variation across height: at least one
    meaningfully-dense column in the broken field had essentially zero
    density variation from ceiling to floor (coefficient of variation
    0.01-0.04 across all 6 benchmark scenarios); the corrected field's
    worst column is 0.08-0.14, over 2x higher than the broken field's
    best. This test would have failed on the pre-fix constants and
    passes on the corrected ones -- verified directly by temporarily
    reintroducing BUOYANCY_SPEED=2.3/TIER2_BASE_BUOYANCY_FRAC=3.3 during
    development."""

    # Cases 0/4/5/9/18/21 -- the same 6-scenario benchmark the
    # stratification fix itself is measured against (varying fan/door/
    # candle-count configuration), so this guard exercises the same
    # real conditions any future buoyancy-parameter change would be
    # tuned against.
    CASES = [0, 4, 5, 9, 18, 21]
    MIN_COLUMN_CV = 0.05  # broken field's own worst case was 0.036; fixed field's own worst was 0.080

    @pytest.fixture(scope="class")
    def sim_data(self):
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        return sim

    @staticmethod
    def _worst_column_cv(density: np.ndarray) -> float:
        """Lowest per-column coefficient of variation (std/mean across
        rows) among columns with meaningfully non-trivial density --
        near 0 means that column looks identical at every height."""
        col_mean = density.mean(axis=0)
        peak = density.max()
        if peak <= 0:
            return float("inf")   # nothing produced at all; not this test's concern
        active = col_mean > 0.1 * peak
        if not active.any():
            return float("inf")
        col_std = density.std(axis=0)
        cv = col_std[active] / np.maximum(col_mean[active], 1e-6)
        return float(cv.min())

    @pytest.mark.parametrize("case_index", CASES)
    def test_density_varies_with_row_not_just_column(self, sim_data, case_index):
        temperature = np.asarray(sim_data.store.get(case_index, DEFAULT_SLICE_KEY))
        velocity = np.asarray(sim_data.store.get(case_index, SliceKey("VELOCITY", 1, 0)))
        sim = SmokeSimulator(temperature.shape[1:], ambient_c=20.0)
        density = None
        for i in range(temperature.shape[0]):
            density = sim.step(temperature[i], velocity[i])
        worst_cv = self._worst_column_cv(density)
        assert worst_cv > self.MIN_COLUMN_CV, (
            f"case {case_index}: at least one meaningfully-dense column has "
            f"coefficient of variation {worst_cv:.4f} across height (<= "
            f"{self.MIN_COLUMN_CV}) -- density isn't varying with row, the "
            f"buffer has homogenized (see this class's own docstring)")


class TestSmokeCompositing:
    def test_smoke_rgba_alpha_tracks_density(self):
        density = np.array([[0.0, 1.0]], dtype=np.float32)
        rgba = smoke_rgba(density)
        assert rgba[0, 0, 3] == 0
        assert rgba[0, 1, 3] > 0

    def test_smoke_tint_reads_as_neutral_grey_not_a_warm_flame_color(self):
        """A real report: the plume was mistakable for fire. FIRE_RGBA_LUT's
        own warm end (the thing it must stay visually distinct from) has a
        wide R > G > B spread (e.g. its hottest entries approach pure warm
        white/yellow from an orange base) -- smoke's tint should instead be
        close to R == G == B, so it reads as grey/grey-white regardless of
        how much of the warm LUT range sits nearby on screen."""
        density = np.array([[1.0]], dtype=np.float32)
        r, g, b, _ = smoke_rgba(density)[0, 0]
        channel_spread = int(max(r, g, b)) - int(min(r, g, b))
        assert channel_spread <= 12, (
            f"smoke tint ({r}, {g}, {b}) has too wide a channel spread to read as neutral grey")

    def test_opaque_top_fully_occludes_bottom(self):
        top = np.zeros((2, 2, 4), dtype=np.uint8)
        top[..., 0] = 200
        top[..., 3] = 255
        bottom = np.zeros((2, 2, 4), dtype=np.uint8)
        bottom[..., 1] = 200
        bottom[..., 3] = 255
        out = composite_over(top, bottom)
        assert (out == top).all()

    def test_transparent_top_shows_bottom(self):
        top = np.zeros((2, 2, 4), dtype=np.uint8)
        bottom = np.zeros((2, 2, 4), dtype=np.uint8)
        bottom[..., 1] = 200
        bottom[..., 3] = 255
        out = composite_over(top, bottom)
        assert (out == bottom).all()


class TestEffectsPipelineFlicker:
    def test_zero_hrr_intensity_gives_deterministic_repeated_output(self):
        """hrr_intensity=0 should mean no flicker modulation at all --
        with exposure locked (isolating flicker from auto-exposure's own
        frame-to-frame adaptation), a constant input frame should render
        identically every call."""
        pipeline = EffectsPipeline(vmin=20.0, vmax_init=250.0)
        pipeline.exposure.locked = True
        frame = np.full((30, 30), 250.0, dtype=np.float32)
        first = pipeline.render(frame, hrr_intensity=0.0)
        second = pipeline.render(frame, hrr_intensity=0.0)
        assert (first == second).all()

    def test_nonzero_hrr_intensity_varies_frame_to_frame(self):
        pipeline = EffectsPipeline(vmin=20.0, vmax_init=250.0)
        pipeline.exposure.locked = True
        frame = np.full((30, 30), 250.0, dtype=np.float32)
        renders = [pipeline.render(frame, hrr_intensity=1.0) for _ in range(5)]
        assert any(not (renders[0] == r).all() for r in renders[1:])


class TestEffectsPipeline:
    def test_render_output_shape_and_dtype(self):
        pipeline = EffectsPipeline(vmin=20.0, vmax_init=300.0)
        frame = np.full((49, 101), 250.0, dtype=np.float32)
        rgba = pipeline.render(frame)
        assert rgba.shape == (49, 101, 4)
        assert rgba.dtype == np.uint8

    def test_ambient_frame_nearly_transparent(self):
        """Fire-realism pass: a subtle ambient backdrop glow (~5% max
        strength) means ambient pixels are no longer *exactly* alpha=0,
        just very close to it -- "doesn't float in pure black" is the
        point, not full opacity."""
        pipeline = EffectsPipeline(vmin=20.0, vmax_init=300.0)
        frame = np.full((49, 101), 20.0, dtype=np.float32)
        rgba = pipeline.render(frame)
        assert rgba[..., 3].max() <= 15
        assert rgba[..., 3].min() == 0

    def test_render_cost_within_budget(self):
        """DoD (ROADMAP-FIRELAB.md Phase 2 task 1): full chain should be
        well under the per-frame budget even before upsampling/bloom/smoke
        are added on top in later tasks."""
        pipeline = EffectsPipeline(vmin=20.0, vmax_init=300.0)
        frame = np.random.default_rng(0).uniform(20.0, 400.0, size=(49, 101)).astype(np.float32)
        for _ in range(20):
            pipeline.render(frame)
        assert pipeline.last_cost_ms < 4.0


class TestHeatShimmer:
    def test_ambient_frame_is_unwarped(self):
        shimmer = HeatShimmer()
        image = np.zeros((30, 30, 4), dtype=np.uint8)
        image[10, 10] = [255, 0, 0, 255]
        ambient = np.full((30, 30), 20.0, dtype=np.float32)
        out = shimmer.warp(image, ambient, ambient_c=20.0)
        assert (out == image).all(), "no heat above ambient should mean no displacement at all"

    def test_hot_frame_shape_and_dtype_preserved(self):
        shimmer = HeatShimmer()
        image = np.full((30, 30, 4), 128, dtype=np.uint8)
        hot = np.full((30, 30), 300.0, dtype=np.float32)
        out = shimmer.warp(image, hot, ambient_c=20.0)
        assert out.shape == image.shape
        assert out.dtype == np.uint8

    def test_advances_over_time(self):
        """Successive calls should scroll the noise field, not repeat the
        exact same warp every frame."""
        shimmer = HeatShimmer()
        image = np.zeros((40, 40, 4), dtype=np.uint8)
        image[20, 20] = [255, 255, 0, 255]
        hot = np.full((40, 40), 300.0, dtype=np.float32)
        first = shimmer.warp(image.copy(), hot, ambient_c=20.0)
        second = shimmer.warp(image.copy(), hot, ambient_c=20.0)
        assert not (first == second).all()


class TestEmberParticles:
    def test_no_spawn_below_threshold(self):
        sim = EmberParticles((20, 20))
        cool_frame = np.full((20, 20), 100.0, dtype=np.float32)  # well under the 150C-above-ambient knee
        for _ in range(10):
            sim.step(cool_frame, ambient_c=20.0)
        assert len(sim.pos) == 0

    def test_hot_frame_spawns_and_caps_at_max(self):
        sim = EmberParticles((20, 20), max_particles=15)
        hot_frame = np.full((20, 20), 400.0, dtype=np.float32)
        for _ in range(60):
            sim.step(hot_frame, ambient_c=20.0)
        assert 0 < len(sim.pos) <= 15

    def test_particles_die_of_old_age(self):
        sim = EmberParticles((20, 20), max_particles=10)
        hot_frame = np.full((20, 20), 400.0, dtype=np.float32)
        sim.step(hot_frame, ambient_c=20.0)
        assert len(sim.pos) > 0
        cool_frame = np.full((20, 20), 20.0, dtype=np.float32)
        for _ in range(200):  # far past any particle's max lifetime, no new spawns
            sim.step(cool_frame, ambient_c=20.0)
        assert len(sim.pos) == 0

    def test_render_arrays_shapes_match_particle_count(self):
        sim = EmberParticles((20, 20), max_particles=10)
        hot_frame = np.full((20, 20), 400.0, dtype=np.float32)
        for _ in range(20):
            sim.step(hot_frame, ambient_c=20.0)
        offsets, sizes, colors = sim.render_arrays()
        n = len(sim.pos)
        assert offsets.shape == (n, 2)
        assert sizes.shape == (n,)
        assert colors.shape == (n, 4)
