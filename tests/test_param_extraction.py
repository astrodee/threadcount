"""Tests for parameter extraction utilities (roadmap item 1.5).

Covers:
  - threadcount.fit.get_param_values / vget_param_values:
      - Extract from a lmfit Parameters object (branch 2).
      - Extract from a ModelResult's embedded params (branch 1).
      - Extract a ModelResult attribute that is not a parameter (branch 3).
      - Returns default_value when param_name is absent (branch 2 falls to branch 3).
      - Returns default_value when input is None.
      - Returns default_value when input has no .get() method (branch 3 → AttributeError).
      - Vectorized vget_param_values applied to an object array.
  - threadcount.lmfit_ext.summary_array (monkey-patched onto ModelResult):
      - With explicit fit_info and param_info, returns correct values in correct order.
      - With empty (default) lists, returns a zero-length float array.
      - Unknown param_info keys become NaN (None → float → NaN).
      - fit_info and param_info may be mixed arbitrarily.
      - Returned array is always dtype float.
"""

import numpy as np
import pytest
from lmfit.models import GaussianModel

import threadcount  # must import to trigger extend_lmfit
import threadcount.fit as tc_fit

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gaussian_params():
    """Return a lmfit Parameters object from a simple GaussianModel."""
    model = GaussianModel()
    return model.make_params(center=5.0, sigma=2.0, amplitude=10.0)


@pytest.fixture(scope="module")
def gaussian_model_result():
    """Return a fitted ModelResult on a Gaussian spectrum with small injected noise.

    A small amount of noise (σ=1e-3) is added so the covariance matrix is
    non-zero — without it lmfit produces stderr=0 and the uncertainties
    library emits a warning.  The noise level is ~1/2000 of the peak so the
    fit still recovers the injected parameters to better than 1e-3 absolute.
    Injected: center=5.0, sigma=2.0, amplitude=10.0
    """
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 200)
    true_center = 5.0
    true_sigma = 2.0
    true_amplitude = 10.0
    y = (
        true_amplitude
        / (true_sigma * np.sqrt(2 * np.pi))
        * np.exp(-((x - true_center) ** 2) / (2 * true_sigma**2))
    )
    # Add tiny noise so the covariance matrix is non-zero; without this lmfit
    # produces stderr=0 on all parameters and the uncertainties library warns
    # "Using UFloat objects with std_dev==0 may give unexpected results."
    y += rng.normal(0, 1e-3, y.shape)
    model = GaussianModel()
    params = model.make_params(center=5.0, sigma=2.0, amplitude=10.0)
    result = model.fit(y, params, x=x, method="least_squares")
    assert result.success or result.redchi < 1e-6
    return result


# ===========================================================================
# get_param_values — branch 1: ModelResult.params.get(name).value
# ===========================================================================


class TestGetParamValuesFromModelResult:
    """Branch 1: params is a ModelResult; extract from result.params."""

    def test_extracts_center(self, gaussian_model_result):
        val = tc_fit.get_param_values(gaussian_model_result, "center")
        assert val == pytest.approx(5.0, abs=1e-3)

    def test_extracts_sigma(self, gaussian_model_result):
        val = tc_fit.get_param_values(gaussian_model_result, "sigma")
        assert val == pytest.approx(2.0, abs=1e-3)

    def test_extracts_amplitude(self, gaussian_model_result):
        val = tc_fit.get_param_values(gaussian_model_result, "amplitude")
        assert val == pytest.approx(10.0, abs=1e-3)

    def test_missing_param_returns_default(self, gaussian_model_result):
        """A parameter name that does not exist falls through all branches."""
        val = tc_fit.get_param_values(gaussian_model_result, "nonexistent_param")
        assert np.isnan(val)

    def test_missing_param_custom_default(self, gaussian_model_result):
        val = tc_fit.get_param_values(
            gaussian_model_result, "nonexistent_param", default_value=-999.0
        )
        assert val == -999.0


# ===========================================================================
# get_param_values — branch 2: params is a Parameters object
# ===========================================================================


class TestGetParamValuesFromParameters:
    """Branch 2: params is a lmfit Parameters object."""

    def test_extracts_sigma(self, gaussian_params):
        val = tc_fit.get_param_values(gaussian_params, "sigma")
        assert val == pytest.approx(2.0)

    def test_extracts_center(self, gaussian_params):
        val = tc_fit.get_param_values(gaussian_params, "center")
        assert val == pytest.approx(5.0)

    def test_missing_key_returns_nan(self, gaussian_params):
        val = tc_fit.get_param_values(gaussian_params, "fwhm_bogus")
        assert np.isnan(val)

    def test_missing_key_custom_default(self, gaussian_params):
        val = tc_fit.get_param_values(gaussian_params, "fwhm_bogus", default_value=0.0)
        assert val == 0.0


# ===========================================================================
# get_param_values — branch 3: ModelResult attribute (not a parameter)
# ===========================================================================


# BUG (roadmap §2.18): the docstring says branch 3 can extract a ModelResult
# attribute such as 'redchi' via params.get(param_name, default_value).  In
# practice the ModelResult class in this lmfit fork does NOT implement .get(),
# so the except-AttributeError guard fires and default_value is returned
# instead.  The tests below assert what SHOULD work per the docstring; they are
# marked xfail(strict=True) so they fail loudly now and turn green automatically
# once §2.18 is fixed.
@pytest.mark.xfail(
    strict=True,
    reason="Bug §2.18: ModelResult.get() absent in lmfit fork so branch 3 is unreachable",
)
class TestGetParamValuesModelResultAttribute:
    """Branch 3: param_name is a ModelResult attribute (e.g. chisqr, redchi).

    Per the docstring, ``get_param_values(result, 'redchi')`` should return
    ``result.redchi`` via ``result.get('redchi', default_value)``.  This does
    not work in the current lmfit fork because ``ModelResult`` has no ``.get()``
    method.  See roadmap §2.18 for the fix.
    """

    def test_extracts_redchi(self, gaussian_model_result):
        val = tc_fit.get_param_values(gaussian_model_result, "redchi")
        assert val < 1e-6

    def test_extracts_chisqr(self, gaussian_model_result):
        val = tc_fit.get_param_values(gaussian_model_result, "chisqr")
        assert val >= 0.0

    def test_extracts_nvarys(self, gaussian_model_result):
        val = tc_fit.get_param_values(gaussian_model_result, "nvarys")
        assert val == 3


# ===========================================================================
# get_param_values — None and non-dict inputs
# ===========================================================================


class TestGetParamValuesEdgeCases:
    """Edge cases: None input, plain int, plain string."""

    def test_none_input_returns_nan(self):
        val = tc_fit.get_param_values(None, "sigma")
        assert np.isnan(val)

    def test_none_input_custom_default(self):
        val = tc_fit.get_param_values(None, "sigma", default_value=-1.0)
        assert val == -1.0

    def test_int_input_returns_default(self):
        """An int has no .get() method — branch 3 raises AttributeError → default."""
        val = tc_fit.get_param_values(42, "sigma")
        assert np.isnan(val)

    def test_plain_dict_returns_value_via_get(self):
        """A plain dict's .get() returns the value directly, not via the .value
        accessor, so it can return any type stored in the dict."""
        d = {"sigma": 3.14}
        val = tc_fit.get_param_values(d, "sigma")
        assert val == pytest.approx(3.14)

    def test_plain_dict_missing_key_returns_default(self):
        d = {"sigma": 3.14}
        val = tc_fit.get_param_values(d, "center", default_value=99.0)
        assert val == 99.0


# ===========================================================================
# vget_param_values — vectorised version
# ===========================================================================


class TestVGetParamValues:
    """vget_param_values broadcasts get_param_values over a numpy object array."""

    def test_array_of_params_objects(self, gaussian_params):
        model = GaussianModel()
        p2 = model.make_params(center=10.0, sigma=4.0, amplitude=20.0)
        # Parameters.__array__ only accepts `self` in this lmfit fork, so we
        # must build the object array without passing dtype to __array__.
        arr = np.empty(2, dtype=object)
        arr[0] = gaussian_params
        arr[1] = p2
        result = tc_fit.vget_param_values(arr, "sigma")
        np.testing.assert_allclose(result, [2.0, 4.0])

    def test_array_of_model_results(self, gaussian_model_result):
        arr = np.array([gaussian_model_result, gaussian_model_result], dtype=object)
        result = tc_fit.vget_param_values(arr, "center")
        assert result.shape == (2,)
        np.testing.assert_allclose(result, 5.0, atol=1e-3)

    def test_array_custom_default_value(self, gaussian_params):
        """Custom default_value is forwarded through vget_param_values."""
        arr = np.empty(2, dtype=object)
        arr[0] = gaussian_params
        arr[1] = None
        result = tc_fit.vget_param_values(arr, "sigma", default_value=-1.0)
        assert result[0] == pytest.approx(2.0)
        assert result[1] == -1.0

    def test_array_with_none_entries(self, gaussian_params):
        arr = np.empty(2, dtype=object)
        arr[0] = gaussian_params
        arr[1] = None
        result = tc_fit.vget_param_values(arr, "sigma")
        assert result[0] == pytest.approx(2.0)
        assert np.isnan(result[1])

    def test_2d_array(self, gaussian_params):
        arr = np.empty((2, 2), dtype=object)
        for idx in np.ndindex(2, 2):
            arr[idx] = gaussian_params
        result = tc_fit.vget_param_values(arr, "center")
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, 5.0)


# ===========================================================================
# summary_array — monkey-patched onto ModelResult by lmfit_ext.extend_lmfit
# ===========================================================================


class TestSummaryArray:
    """Tests for ModelResult.summary_array (added by lmfit_ext)."""

    def test_returns_float_array(self, gaussian_model_result):
        arr = gaussian_model_result.summary_array(
            fit_info=["redchi"], param_info=["center"]
        )
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == float

    def test_fit_info_values(self, gaussian_model_result):
        arr = gaussian_model_result.summary_array(fit_info=["redchi", "chisqr"])
        assert arr.shape == (2,)
        # redchi ≈ noise_variance × ndata/ndof ≈ 1e-6 for σ=1e-3 noise;
        # threshold 1e-4 gives ample headroom without being sensitive to the
        # exact RNG draw.
        assert arr[0] < 1e-4
        assert arr[1] >= 0.0  # chisqr

    def test_param_info_value(self, gaussian_model_result):
        """param_info entries resolve to value and stderr via valerrsdict."""
        arr = gaussian_model_result.summary_array(param_info=["center", "center_err"])
        assert arr.shape == (2,)
        assert arr[0] == pytest.approx(5.0, abs=1e-3)  # center value
        # center_err should be a small positive finite number for a noisy fit
        assert np.isfinite(arr[1])
        assert arr[1] > 0

    def test_param_info_none_stderr_produces_nan(self, gaussian_model_result):
        """Parameters with stderr=None (e.g. expression-constrained ones) produce
        NaN in the output, not an error."""
        # GaussianModel's 'height' is expression-constrained; its stderr may be
        # None if uncertainty propagation through max() fails (lmfit fork limitation).
        arr = gaussian_model_result.summary_array(param_info=["height_err"])
        assert arr.shape == (1,)
        # Either NaN (stderr=None) or a finite positive number are both acceptable;
        # the important thing is that no exception is raised.
        assert np.isnan(arr[0]) or (np.isfinite(arr[0]) and arr[0] >= 0)

    def test_fit_info_invalid_attribute_raises(self, gaussian_model_result):
        """Passing an invalid fit_info key raises AttributeError because
        summary_array uses getattr() with no fallback.
        This is an unhandled edge case — see roadmap for a robustness fix."""
        with pytest.raises(AttributeError):
            gaussian_model_result.summary_array(fit_info=["no_such_attribute"])

    def test_param_info_unknown_key_is_nan(self, gaussian_model_result):
        """valerrsdict returns None for unknown keys; float(None) raises, so
        summary_array should produce NaN for them."""
        arr = gaussian_model_result.summary_array(param_info=["no_such_param"])
        assert arr.shape == (1,)
        assert np.isnan(arr[0])

    def test_empty_defaults_return_zero_length_array(self, gaussian_model_result):
        arr = gaussian_model_result.summary_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (0,)

    def test_mixed_fit_info_and_param_info(self, gaussian_model_result):
        """fit_info entries come first, then param_info entries."""
        arr = gaussian_model_result.summary_array(
            fit_info=["redchi"], param_info=["center", "sigma"]
        )
        assert arr.shape == (3,)
        # Index 0: redchi — small but not zero due to noise; see test_fit_info_values
        assert arr[0] < 1e-4
        # Index 1: center value
        assert arr[1] == pytest.approx(5.0, abs=1e-3)
        # Index 2: sigma value
        assert arr[2] == pytest.approx(2.0, abs=1e-3)

    def test_ordering_is_fit_info_then_param_info(self, gaussian_model_result):
        """Explicitly verify that fit_info comes before param_info in the output."""
        arr_mixed = gaussian_model_result.summary_array(
            fit_info=["chisqr"], param_info=["amplitude"]
        )
        arr_chisqr = gaussian_model_result.summary_array(fit_info=["chisqr"])
        arr_amplitude = gaussian_model_result.summary_array(param_info=["amplitude"])
        assert arr_mixed[0] == pytest.approx(arr_chisqr[0])
        assert arr_mixed[1] == pytest.approx(arr_amplitude[0])
