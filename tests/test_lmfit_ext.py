"""Unit tests for lmfit extensions in threadcount.lmfit_ext (Phase 1.8D).

Covers: aic_real, bic_real, stderrsdict, valerrsdict,
        set_param_hint_endswith, order_gauss, summary_array.
"""

import math
from types import SimpleNamespace

import lmfit
import numpy as np
import pytest

import threadcount as tc
import threadcount.lmfit_ext as tle

# ---------------------------------------------------------------------------
# Module-level real ModelResult (Const_1GaussModel on clean synthetic data)
# ---------------------------------------------------------------------------
_X = np.linspace(4990.0, 5025.0, 120)
_Y = 2.0 + 50.0 * np.exp(-((_X - 5006.843) ** 2) / (2 * 1.0**2))

_MODEL_1G = tc.models.Const_1GaussModel()
_RESULT = _MODEL_1G.fit(_Y, _MODEL_1G.guess(_Y, x=_X), x=_X, method="least_squares")


# ---------------------------------------------------------------------------
# Helpers for order_gauss tests
# ---------------------------------------------------------------------------


def _make_2g_params(c1, h1, c2, h2, s1=1.0, s2=1.0):
    """Build a minimal 2-gaussian Parameters object for order_gauss tests."""
    params = lmfit.Parameters()
    params.add("g1_center", value=c1)
    params.add("g1_height", value=h1)
    params.add("g1_sigma", value=s1)
    params.add("g2_center", value=c2)
    params.add("g2_height", value=h2)
    params.add("g2_sigma", value=s2)
    return params


# ---------------------------------------------------------------------------
# aic_real / bic_real
# ---------------------------------------------------------------------------


class TestAicRealBicReal:
    """1.8D — aic_real / bic_real: information-criterion properties."""

    def test_aic_real_is_finite(self):
        assert _RESULT.aic_real is not None
        assert np.isfinite(_RESULT.aic_real)

    def test_aic_real_formula(self):
        """aic_real = chisqr + 2 * nvarys (exact)."""
        expected = _RESULT.chisqr + 2 * _RESULT.nvarys
        assert _RESULT.aic_real == pytest.approx(expected)

    def test_bic_real_formula(self):
        """bic_real = chisqr + log(ndata) * nvarys (exact)."""
        expected = _RESULT.chisqr + math.log(_RESULT.ndata) * _RESULT.nvarys
        assert _RESULT.bic_real == pytest.approx(expected)

    def test_bic_real_greater_than_aic_for_large_ndata(self):
        """bic > aic when ndata > e^2 (~7.4); with 120 data points this holds."""
        assert _RESULT.bic_real > _RESULT.aic_real

    def test_aic_real_missing_chisqr_returns_none(self):
        """AttributeError on self.chisqr must be caught; function returns None."""
        assert tle.aic_real(SimpleNamespace()) is None

    def test_bic_real_missing_chisqr_returns_none(self):
        assert tle.bic_real(SimpleNamespace()) is None

    def test_aic_real_none_chisqr_returns_none(self):
        """TypeError from None + int must be caught; function returns None."""
        # init_vals=[] prevents AttributeError when Python eagerly evaluates
        # len(self.init_vals) as the default argument in getattr(self, "nvarys", ...)
        obj = SimpleNamespace(chisqr=None, nvarys=2, init_vals=[])
        assert tle.aic_real(obj) is None

    def test_bic_real_none_chisqr_returns_none(self):
        obj = SimpleNamespace(
            chisqr=None, nvarys=2, ndata=10, init_vals=[], residual=[]
        )
        assert tle.bic_real(obj) is None


# ---------------------------------------------------------------------------
# stderrsdict
# ---------------------------------------------------------------------------


class TestStderrsdict:
    """1.8D — stderrsdict: {param_name: stderr} mapping from Parameters."""

    def test_returns_dict(self):
        assert isinstance(_RESULT.params.stderrsdict(), dict)

    def test_keys_equal_params_keys(self):
        d = _RESULT.params.stderrsdict()
        assert set(d.keys()) == set(_RESULT.params.keys())

    def test_values_equal_per_param_stderr(self):
        d = _RESULT.params.stderrsdict()
        for name, stderr in d.items():
            assert stderr == _RESULT.params[name].stderr

    def test_length_equals_params_length(self):
        d = _RESULT.params.stderrsdict()
        assert len(d) == len(_RESULT.params)


# ---------------------------------------------------------------------------
# valerrsdict
# ---------------------------------------------------------------------------


class TestValerrsdict:
    """1.8D — valerrsdict: {k: value, k_err: stderr, ...} mapping."""

    def test_returns_dict(self):
        assert isinstance(_RESULT.params.valerrsdict(), dict)

    def test_both_key_and_key_err_present_for_each_param(self):
        d = _RESULT.params.valerrsdict()
        for name in _RESULT.params.keys():
            assert name in d
            assert name + "_err" in d

    def test_value_matches_param_value(self):
        d = _RESULT.params.valerrsdict()
        for name in _RESULT.params.keys():
            assert d[name] == _RESULT.params[name].value

    def test_err_matches_param_stderr(self):
        d = _RESULT.params.valerrsdict()
        for name in _RESULT.params.keys():
            assert d[name + "_err"] == _RESULT.params[name].stderr

    def test_total_length_is_twice_params(self):
        """Length = 2 * len(params): one value + one err entry per parameter."""
        d = _RESULT.params.valerrsdict()
        assert len(d) == 2 * len(_RESULT.params)


# ---------------------------------------------------------------------------
# set_param_hint_endswith
# ---------------------------------------------------------------------------


class TestSetParamHintEndswith:
    """1.8D — set_param_hint_endswith: apply hints to all matching param names.

    Registered on lmfit.model.Model as set_param_hint_endswith (singular).
    """

    def test_matching_params_receive_hint(self):
        """All params ending in 'sigma' must have min=0.5 after the call."""
        model = tc.models.Const_2GaussModel()
        model.set_param_hint_endswith("sigma", min=0.5)
        sigma_params = [n for n in model.param_names if n.endswith("sigma")]
        assert len(sigma_params) >= 2  # sanity: expect at least g1_sigma, g2_sigma
        for name in sigma_params:
            assert model.param_hints.get(name, {}).get("min") == 0.5

    def test_nonmatching_params_not_affected(self):
        """Non-sigma params must not receive a min=0.5 hint."""
        model = tc.models.Const_2GaussModel()
        model.set_param_hint_endswith("sigma", min=0.5)
        for name in model.param_names:
            if not name.endswith("sigma"):
                assert model.param_hints.get(name, {}).get("min") != 0.5

    def test_no_match_is_silent_noop(self):
        """A suffix matching no parameter must not raise."""
        model = tc.models.Const_2GaussModel()
        model.set_param_hint_endswith("zzz_no_match_at_all", min=0.5)

    def test_multiple_kwargs_applied_together(self):
        """Both min and max are applied in a single call."""
        model = tc.models.Const_2GaussModel()
        model.set_param_hint_endswith("sigma", min=0.3, max=5.0)
        for name in model.param_names:
            if name.endswith("sigma"):
                hints = model.param_hints.get(name, {})
                assert hints.get("min") == 0.3
                assert hints.get("max") == 5.0

    def test_preexisting_stricter_min_not_overwritten(self):
        """A pre-set tighter min must not be loosened by a batch hint call."""
        model = tc.models.Const_2GaussModel()
        model.set_param_hint("g2_sigma", min=2.0)
        model.set_param_hint_endswith("sigma", min=0.5)
        assert model.param_hints["g2_sigma"]["min"] == 2.0


# ---------------------------------------------------------------------------
# order_gauss
# ---------------------------------------------------------------------------


class TestOrderGauss:
    """1.8D — order_gauss: sort gaussian Parameters by center (taller second when close)."""

    def test_g2_center_less_reorders_centers_ascending(self):
        """g2 center < g1 center → after order_gauss, g1_center < g2_center."""
        params = _make_2g_params(c1=5010.0, h1=20.0, c2=5000.0, h2=30.0)
        params.order_gauss()
        assert params["g1_center"].value < params["g2_center"].value

    def test_heights_follow_center_reorder(self):
        """After reorder, the values associated with each center move together."""
        params = _make_2g_params(c1=5010.0, h1=20.0, c2=5000.0, h2=30.0)
        params.order_gauss()
        # original g2 (center=5000, height=30) → now g1
        assert params["g1_height"].value == pytest.approx(30.0)
        assert params["g2_height"].value == pytest.approx(20.0)

    def test_close_centers_taller_placed_second(self):
        """Centers within delta_x: taller component placed at g2 (second) position."""
        # gap=0.3 < delta_x=0.5; g1 is taller (h=50 vs h=20) → must swap heights
        params = _make_2g_params(c1=5006.0, h1=50.0, c2=5006.3, h2=20.0)
        params.order_gauss(delta_x=0.5)
        assert params["g2_height"].value > params["g1_height"].value

    def test_close_centers_already_correct_no_swap(self):
        """Centers within delta_x but g2 already taller: no swap needed."""
        params = _make_2g_params(c1=5006.0, h1=20.0, c2=5006.3, h2=50.0)
        params.order_gauss(delta_x=0.5)
        assert params["g2_height"].value > params["g1_height"].value

    def test_single_gaussian_returns_unchanged(self):
        """ngauss=1: order_gauss returns immediately without modifying values."""
        params = lmfit.Parameters()
        params.add("g1_center", value=5006.843)
        params.add("g1_height", value=30.0)
        params.add("g1_sigma", value=1.0)
        params.order_gauss()
        assert params["g1_center"].value == pytest.approx(5006.843)

    def test_expressions_cleared_on_all_params(self):
        """All parameter expressions must be cleared (None or '') before reordering."""
        params = _make_2g_params(c1=5010.0, h1=20.0, c2=5000.0, h2=30.0)
        params["g1_center"].set(expr="g2_center + 10.0")
        params["g2_sigma"].set(expr="g1_sigma")
        params.order_gauss()
        for p in params.values():
            assert not p.expr  # None or ""

    def test_already_sorted_not_modified(self):
        """Centers already in ascending order with gap > delta_x: no change."""
        params = _make_2g_params(c1=5000.0, h1=30.0, c2=5010.0, h2=20.0)
        params.order_gauss()
        assert params["g1_center"].value == pytest.approx(5000.0)
        assert params["g2_center"].value == pytest.approx(5010.0)

    def test_three_gaussians_fully_reordered_to_ascending(self):
        """3 shuffled gaussians (5020, 5000, 5010) must sort to (5000, 5010, 5020)."""
        params = lmfit.Parameters()
        for prefix, c, h in [
            ("g1_", 5020.0, 10.0),
            ("g2_", 5000.0, 20.0),
            ("g3_", 5010.0, 15.0),
        ]:
            params.add(prefix + "center", value=c)
            params.add(prefix + "height", value=h)
            params.add(prefix + "sigma", value=1.0)
        params.order_gauss()
        assert params["g1_center"].value == pytest.approx(5000.0)
        assert params["g2_center"].value == pytest.approx(5010.0)
        assert params["g3_center"].value == pytest.approx(5020.0)

    def test_no_gaussian_params_returns_unchanged(self):
        """ngauss=0 (no g*_sigma params): order_gauss returns immediately."""
        params = lmfit.Parameters()
        params.add("c", value=2.0)
        params.order_gauss()
        assert params["c"].value == pytest.approx(2.0)

    @pytest.mark.xfail(
        reason=(
            "Bug §2.24: order_gauss assumes g{n}_height params exist and calls "
            ".value on self.get('g{n}_height'), crashing with "
            "AttributeError: 'NoneType' object has no attribute 'value' when "
            "height parameters are absent (e.g. standard lmfit GaussianModel "
            "which uses amplitude instead of height)."
        ),
        strict=True,
    )
    def test_order_gauss_missing_height_param_does_not_crash(self):
        """order_gauss must not crash when g{n}_height is absent."""
        params = lmfit.Parameters()
        params.add("g1_center", value=5010.0)
        params.add("g1_sigma", value=1.0)
        params.add("g2_center", value=5000.0)
        params.add("g2_sigma", value=1.0)
        params.order_gauss()  # currently crashes: AttributeError on None.value


# ---------------------------------------------------------------------------
# summary_array
# ---------------------------------------------------------------------------


class TestSummaryArray:
    """1.8D — summary_array: numpy float array of fit attributes + param values."""

    def test_fit_info_and_param_info_values_correct(self):
        """fit_info attributes come first; param_info values follow via valerrsdict."""
        arr = _RESULT.summary_array(
            fit_info=["redchi"],
            param_info=["g1_center", "g1_center_err"],
        )
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == float
        assert len(arr) == 3
        assert arr[0] == pytest.approx(_RESULT.redchi)
        assert arr[1] == pytest.approx(_RESULT.params["g1_center"].value)
        assert arr[2] == pytest.approx(_RESULT.params["g1_center"].stderr)

    def test_empty_lists_return_zero_length_array(self):
        arr = _RESULT.summary_array(fit_info=[], param_info=[])
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 0

    def test_none_defaults_to_empty(self):
        """Default fit_info=None and param_info=None both map to []."""
        arr = _RESULT.summary_array()
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 0

    def test_multiple_fit_info_attrs_in_order(self):
        """Multiple fit_info entries are placed in order before param values."""
        arr = _RESULT.summary_array(
            fit_info=["redchi", "aic_real"],
            param_info=["g1_center"],
        )
        assert len(arr) == 3
        assert arr[0] == pytest.approx(_RESULT.redchi)
        assert arr[1] == pytest.approx(_RESULT.aic_real)
        assert arr[2] == pytest.approx(_RESULT.params["g1_center"].value)

    def test_missing_param_info_key_produces_nan(self):
        """Key absent from valerrsdict → d.get returns None → np.array dtype=float → NaN."""
        arr = _RESULT.summary_array(fit_info=[], param_info=["nonexistent_key"])
        assert len(arr) == 1
        assert np.isnan(arr[0])

    def test_returns_float_array(self):
        arr = _RESULT.summary_array(fit_info=["redchi"], param_info=["g1_center"])
        assert arr.dtype == float

    def test_bad_fit_info_key_raises_attributeerror(self):
        """A misspelt fit_info attribute raises AttributeError (no silent NaN)."""
        with pytest.raises(AttributeError):
            _RESULT.summary_array(fit_info=["nonexistent_fit_info_key"])


# ---------------------------------------------------------------------------
# plot2 / plot_components — matplotlib smoke tests (Phase 1.10)
# ---------------------------------------------------------------------------


class TestPlot2Smoke:
    """1.10 — plot2: smoke test using the Agg backend."""

    def setup_method(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_returns_fig_and_two_axes(self):
        """plot2() returns (fig, ax_res, ax_fit) without raising."""
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure

        result = _RESULT.plot2()
        assert isinstance(result, tuple) and len(result) == 3
        fig, ax_res, ax_fit = result
        assert isinstance(fig, Figure)
        assert isinstance(ax_res, Axes)
        assert isinstance(ax_fit, Axes)
        plt.close("all")

    def test_accepts_existing_figure(self):
        """Passing an existing Figure reuses it instead of creating a new one."""
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        existing = plt.figure()
        fig, _, _ = _RESULT.plot2(fig=existing)
        assert fig is existing
        plt.close("all")


class TestPlotComponentsSmoke:
    """1.10 — plot_components: smoke test using the Agg backend."""

    def setup_method(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_returns_axes(self):
        """plot_components() returns an Axes object without raising."""
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes

        ax = _RESULT.plot_components()
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_accepts_existing_axes(self):
        """Passing an existing Axes reuses it and returns the same object."""
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes

        fig, existing_ax = plt.subplots()
        returned = _RESULT.plot_components(ax=existing_ax)
        assert returned is existing_ax
        plt.close("all")


# ---------------------------------------------------------------------------
# Numba setup_bounds patch (Phase 0b.1)
# ---------------------------------------------------------------------------


class TestNumbaKernels:
    """0b.1 — JIT kernels match the analytic upstream formulas exactly.

    Each kernel is a direct translation of the corresponding lambda in the
    upstream lmfit Parameter.setup_bounds.  Testing against the analytic
    formula (which is the same math, computed in plain Python) confirms the
    njit compilation did not introduce any numerical difference.
    """

    def test_lower_bound_kernel_matches_formula(self):
        """_from_internal_lower_bound(min, val) == min - 1 + sqrt(val^2 + 1)."""
        from numpy import sqrt

        from threadcount.lmfit_ext import _from_internal_lower_bound

        for min_val, val in [(-5.0, 0.0), (2.0, 1.5), (0.0, -3.0), (1.0, 10.0)]:
            expected = min_val - 1.0 + sqrt(val * val + 1)
            assert _from_internal_lower_bound(min_val, val) == pytest.approx(expected)

    def test_upper_bound_kernel_matches_formula(self):
        """_from_internal_upper_bound(max, val) == max + 1 - sqrt(val^2 + 1)."""
        from numpy import sqrt

        from threadcount.lmfit_ext import _from_internal_upper_bound

        for max_val, val in [(5.0, 0.0), (-2.0, 1.5), (10.0, -3.0), (0.0, 2.0)]:
            expected = max_val + 1 - sqrt(val * val + 1)
            assert _from_internal_upper_bound(max_val, val) == pytest.approx(expected)

    def test_two_sided_kernel_matches_formula(self):
        """_from_internal_bound2(min, max, val) == min + (sin(val)+1)*(max-min)/2."""
        from numpy import sin

        from threadcount.lmfit_ext import _from_internal_bound2

        for min_val, max_val, val in [
            (0.0, 1.0, 0.0),
            (-1.0, 1.0, 1.5),
            (2.0, 8.0, -0.7),
        ]:
            expected = min_val + (sin(val) + 1) * (max_val - min_val) / 2.0
            assert _from_internal_bound2(min_val, max_val, val) == pytest.approx(
                expected
            )


class TestNumbaMatchesLmfit:
    """0b.1 — our patch produces from_internal identical to the real lmfit.

    For each bound case we run two branches on the same Parameter value:
      - branch A: _original_setup_bounds  (lmfit's own code, saved before
                  extend_lmfit replaced it)
      - branch B: _numba_setup_bounds     (our JIT patch)

    We compare from_internal at a grid of internal values, so any formula
    divergence is caught regardless of self-consistency within either branch.
    """

    # internal values to probe for each bound case
    _PROBE_VALS = [-10.0, -1.0, -0.001, 0.0, 0.001, 1.0, 10.0]

    def _run_both(self, value, min=-np.inf, max=np.inf):
        """Return (from_internal_orig, from_internal_numba) callables for a param."""
        from threadcount.lmfit_ext import _numba_setup_bounds, _original_setup_bounds

        p_orig = lmfit.Parameter(name="x", value=value, min=min, max=max)
        _original_setup_bounds(p_orig)

        p_numba = lmfit.Parameter(name="x", value=value, min=min, max=max)
        _numba_setup_bounds(p_numba)

        return p_orig.from_internal, p_numba.from_internal

    def test_unbounded_from_internal_matches(self):
        """Unbounded: our from_internal == lmfit's at all probe values."""
        fi_orig, fi_numba = self._run_both(3.7)
        for v in self._PROBE_VALS:
            assert fi_numba(v) == pytest.approx(fi_orig(v), rel=1e-12), (
                f"unbounded mismatch at v={v}: numba={fi_numba(v)} orig={fi_orig(v)}"
            )

    def test_lower_bound_only_from_internal_matches(self):
        """Only min set: our from_internal == lmfit's at all probe values."""
        fi_orig, fi_numba = self._run_both(5.0, min=2.0)
        for v in self._PROBE_VALS:
            assert fi_numba(v) == pytest.approx(fi_orig(v), rel=1e-12), (
                f"lower-bound mismatch at v={v}: numba={fi_numba(v)} orig={fi_orig(v)}"
            )

    def test_upper_bound_only_from_internal_matches(self):
        """Only max set: our from_internal == lmfit's at all probe values."""
        fi_orig, fi_numba = self._run_both(-1.0, max=3.0)
        for v in self._PROBE_VALS:
            assert fi_numba(v) == pytest.approx(fi_orig(v), rel=1e-12), (
                f"upper-bound mismatch at v={v}: numba={fi_numba(v)} orig={fi_orig(v)}"
            )

    def test_two_sided_from_internal_matches(self):
        """Both bounds set: our from_internal == lmfit's at all probe values."""
        fi_orig, fi_numba = self._run_both(0.3, min=0.0, max=1.0)
        for v in self._PROBE_VALS:
            assert fi_numba(v) == pytest.approx(fi_orig(v), rel=1e-12), (
                f"two-sided mismatch at v={v}: numba={fi_numba(v)} orig={fi_orig(v)}"
            )

    def test_setup_bounds_return_value_matches(self):
        """The internal value returned by setup_bounds itself must also match."""
        from threadcount.lmfit_ext import _numba_setup_bounds, _original_setup_bounds

        cases = [
            dict(value=3.7),
            dict(value=5.0, min=2.0),
            dict(value=-1.0, max=3.0),
            dict(value=0.3, min=0.0, max=1.0),
        ]
        for kwargs in cases:
            p_orig = lmfit.Parameter(name="x", **kwargs)
            p_numba = lmfit.Parameter(name="x", **kwargs)
            iv_orig = _original_setup_bounds(p_orig)
            iv_numba = _numba_setup_bounds(p_numba)
            assert iv_numba == pytest.approx(iv_orig, rel=1e-12, abs=1e-15), (
                f"setup_bounds return mismatch for {kwargs}: "
                f"numba={iv_numba} orig={iv_orig}"
            )


class TestNumbaSetupBounds:
    """0b.1 — _numba_setup_bounds: self-consistent round-trips for all four bound cases.

    Verifies that from_internal is a right-inverse of the internal value
    returned by setup_bounds (i.e. the transform is invertible).  These tests
    do NOT verify correctness against lmfit — that is done by TestNumbaMatchesLmfit.
    """

    def _make_param(self, value, min=-np.inf, max=np.inf):
        p = lmfit.Parameter(name="x", value=value, min=min, max=max)
        return p

    def test_unbounded_from_internal_is_identity(self):
        """No bounds: from_internal(v) == float(v)."""
        p = self._make_param(3.7)
        p.setup_bounds()
        assert p.from_internal(3.7) == pytest.approx(3.7)
        assert p.from_internal(-100.0) == pytest.approx(-100.0)

    def test_lower_bound_only_round_trip(self):
        """Only min set: round-trip from_internal(setup_bounds()) recovers value."""
        value = 5.0
        p = self._make_param(value, min=2.0)
        internal = p.setup_bounds()
        recovered = p.from_internal(internal)
        assert recovered == pytest.approx(value, rel=1e-10)

    def test_upper_bound_only_round_trip(self):
        """Only max set: round-trip recovers value."""
        value = -1.0
        p = self._make_param(value, max=3.0)
        internal = p.setup_bounds()
        recovered = p.from_internal(internal)
        assert recovered == pytest.approx(value, rel=1e-10)

    def test_two_sided_bound_round_trip(self):
        """Both min and max set: round-trip recovers value."""
        value = 0.3
        p = self._make_param(value, min=0.0, max=1.0)
        internal = p.setup_bounds()
        recovered = p.from_internal(internal)
        assert recovered == pytest.approx(value, rel=1e-10)

    def test_value_at_lower_bound_edge(self):
        """Value exactly at min: round-trip still works."""
        value = 2.0
        p = self._make_param(value, min=2.0)
        internal = p.setup_bounds()
        recovered = p.from_internal(internal)
        assert recovered == pytest.approx(value, rel=1e-9)

    def test_value_at_midpoint_two_sided(self):
        """Value at midpoint of two-sided range recovers correctly."""
        value = 0.5
        p = self._make_param(value, min=0.0, max=1.0)
        internal = p.setup_bounds()
        recovered = p.from_internal(internal)
        assert recovered == pytest.approx(value, rel=1e-10)


class TestNumbaSetupBoundsIntegration:
    """0b.1 — bounded fit still converges with the monkey-patched setup_bounds.

    Uses a 1-gaussian model with tight bounds on sigma to confirm that the
    optimizer successfully navigates the Minuit-style transformed space under
    the numba JIT patch.
    """

    def test_bounded_fit_converges_and_recovers_parameters(self):
        """Fit with bounded sigma converges; recovered params within 1% of truth."""
        model = tc.models.Const_1GaussModel()
        params = model.guess(_Y, x=_X)
        # apply tight but valid bounds around the known sigma=1.0
        params["g1_sigma"].min = 0.5
        params["g1_sigma"].max = 3.0
        params["g1_height"].min = 0.0
        result = model.fit(_Y, params, x=_X, method="least_squares")
        assert result.success
        assert result.params["g1_center"].value == pytest.approx(5006.843, rel=0.01)
        assert result.params["g1_sigma"].value == pytest.approx(1.0, rel=0.05)

    def test_bounded_fit_respects_bounds(self):
        """Fitted parameter values stay within specified bounds."""
        model = tc.models.Const_1GaussModel()
        params = model.guess(_Y, x=_X)
        params["g1_sigma"].min = 0.5
        params["g1_sigma"].max = 3.0
        result = model.fit(_Y, params, x=_X, method="least_squares")
        assert result.params["g1_sigma"].value >= 0.5
        assert result.params["g1_sigma"].value <= 3.0
