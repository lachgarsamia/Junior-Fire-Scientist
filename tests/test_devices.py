"""Tests for the Virtual Device Network's compute functions (src/devices.py).

Focused on the smoke_detector model (photoelectric/optical, from SOOT
DENSITY) since no prior test file exercised devices.py's compute logic;
light regression coverage for the pre-existing heat_detector/sprinkler
models is included so a future change can't silently break them.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import devices as dv  # noqa: E402
from slice_key import SliceKey  # noqa: E402


class FakeProvider:
    """Minimal QuantityProvider stand-in: one (n_times,1,1) field per
    quantity name, read back through the same provider.get/get_extent
    interface probe_series() uses -- no plumbing changes needed to test
    against it."""

    def __init__(self, fields: dict):
        self._fields = {k: np.asarray(v, dtype=float).reshape(-1, 1, 1) for k, v in fields.items()}

    def get(self, scenario, key: SliceKey):
        return self._fields[key.quantity]

    def get_extent(self, scenario, key: SliceKey):
        return None


FPS = 1


# ------------------------------------------------------- unit conversion (1)
def test_soot_density_unit_conversion_1mg_per_m3():
    # 1 mg/m^3 -> 1e-6 kg/m^3 -> Cs = 8700 * 1e-6 = 0.0087 1/m (worked example)
    assert dv.soot_density_kg_per_m3(1.0) == pytest.approx(1.0e-6)
    cs = dv.extinction_coefficient_per_m(1.0)
    assert cs == pytest.approx(0.0087)


def test_extinction_coefficient_uses_provided_km():
    cs = dv.extinction_coefficient_per_m(2.0, km_m2_per_kg=1000.0)
    assert cs == pytest.approx(1000.0 * 2.0 * 1.0e-6)


# ------------------------------------------------------------- zero soot (2)
def test_zero_soot_gives_zero_obscuration_and_no_activation():
    provider = FakeProvider({"SOOT DENSITY": [0.0, 0.0, 0.0, 0.0]})
    r = dv.compute_smoke_detector(provider, 0, 0.0, 0.0, FPS)
    assert all(c == 0.0 for c in r["extinction_coefficient_per_m"])
    assert all(o == 0.0 for o in r["obscuration_pct_per_m"])
    assert r["activated"] is False
    assert r["activation_time_s"] is None
    assert r["activation_frame"] is None


# --------------------------------------------------------- monotonicity (3)
def test_obscuration_increases_monotonically_with_soot():
    soot = [0.0, 500.0, 1500.0, 3000.0, 6000.0]
    provider = FakeProvider({"SOOT DENSITY": soot})
    r = dv.compute_smoke_detector(provider, 0, 0.0, 0.0, FPS)
    obscuration = r["obscuration_pct_per_m"]
    assert all(b > a for a, b in zip(obscuration, obscuration[1:]))
    cs = r["extinction_coefficient_per_m"]
    assert all(b > a for a, b in zip(cs, cs[1:]))


# ----------------------------------------------------- threshold crossing (4)
def test_threshold_crossing_activates_at_the_right_frame():
    # threshold 2.5 %/ft -> Cs_threshold ~= 0.2409 1/m (via _cs_threshold_per_m_from_pct_per_ft)
    cs_threshold = dv._cs_threshold_per_m_from_pct_per_ft(2.5)
    # soot (mg/m^3) whose Cs = Km*rho*1e-6 straddles cs_threshold at frame 2
    below = (cs_threshold * 0.5) / dv.MASS_EXTINCTION_COEFFICIENT_M2_PER_KG / dv.MG_M3_TO_KG_M3
    above = (cs_threshold * 2.0) / dv.MASS_EXTINCTION_COEFFICIENT_M2_PER_KG / dv.MG_M3_TO_KG_M3
    provider = FakeProvider({"SOOT DENSITY": [0.0, below, above, above]})
    r = dv.compute_smoke_detector(provider, 0, 0.0, 0.0, FPS,
                                  smoke_threshold_obscuration_per_ft=2.5)
    assert r["activated"] is True
    assert r["activation_frame"] == 2
    assert r["activation_time_s"] == pytest.approx(2.0 / FPS)
    assert r["threshold_extinction_coefficient_per_m"] == pytest.approx(cs_threshold)


# ------------------------------------------------------------ no crossing (5)
def test_never_reaches_threshold_reports_none_not_a_false_negative_claim():
    provider = FakeProvider({"SOOT DENSITY": [0.0, 1.0, 2.0, 3.0]})
    r = dv.compute_smoke_detector(provider, 0, 0.0, 0.0, FPS,
                                  smoke_threshold_obscuration_per_ft=99.0)
    assert r["activated"] is False
    assert r["activation_time_s"] is None
    assert r["activation_frame"] is None
    # never claims the detector "would never activate" -- basis stays scoped
    # to the simulated interval, no such absolute claim appears in it.
    assert "never" not in r["basis"].lower()


def test_basis_states_photoelectric_scope_and_excludes_ionization():
    provider = FakeProvider({"SOOT DENSITY": [0.0]})
    r = dv.compute_smoke_detector(provider, 0, 0.0, 0.0, FPS)
    basis = r["basis"].lower()
    assert "photoelectric" in basis or "optical" in basis
    assert "ionization" in basis
    assert "estimate" in basis or "estimated" in basis


def test_smoke_detector_device_end_to_end_via_device_compute():
    provider = FakeProvider({"SOOT DENSITY": [0.0, 6000.0, 6000.0]})
    d = dv.Device(id="d1", name="SD-01", type="smoke_detector", scenario=0, position=(0.0, 0.0))
    d.compute(provider, FPS)
    assert d.results["activated"] is True
    state = d.state_at(d.results["activation_frame"])
    assert state["active"] is True
    insight = d.summary_insight()
    assert insight is not None and "SD-01" in insight.statement
    assert insight.quantity == "SOOT DENSITY"


# ---------------------------------------------- existing-behavior regression (6)
def test_heat_detector_regression_activates_on_crossing():
    provider = FakeProvider({"TEMPERATURE": [20.0, 50.0, 80.0, 90.0]})
    r = dv.compute_heat_detector(provider, 0, 0.0, 0.0, FPS, activation_temp=74.0)
    assert r["activated"] is True
    assert r["activation_frame"] == 2
    assert r["basis"]


def test_heat_detector_regression_no_crossing():
    provider = FakeProvider({"TEMPERATURE": [20.0, 30.0, 40.0]})
    r = dv.compute_heat_detector(provider, 0, 0.0, 0.0, FPS, activation_temp=74.0)
    assert r["activated"] is False
    assert r["activation_time_s"] is None


def test_sprinkler_regression_runs_and_reports_reduced_model_without_velocity():
    provider = FakeProvider({"TEMPERATURE": [20.0] + [200.0] * 20})
    r = dv.compute_sprinkler(provider, 0, 0.0, 0.0, FPS, rti=100.0, activation_temp=68.0)
    assert isinstance(r["activated"], bool)
    assert r["reduced_model"] is True
    assert "basis" in r


def test_thermocouple_regression_reports_max_and_basis():
    provider = FakeProvider({"TEMPERATURE": [20.0, 40.0, 30.0]})
    r = dv.compute_thermocouple(provider, 0, 0.0, 0.0, FPS)
    assert r["max_temperature_C"] == pytest.approx(40.0)
    assert r["basis"]
