"""Unit tests for AIC model-selection logic in threadcount.fit (Phase 1.8B).

Covers: get_aic, choose_model_aic_single, choose_model_aic,
        get_ngaussians, get_gcomponent_comparison, marginal_fits.
"""

from types import SimpleNamespace

import lmfit
import numpy as np
import pytest

import threadcount as tc
import threadcount.fit as tf

# ---------------------------------------------------------------------------
# Lightweight mock helpers
# ---------------------------------------------------------------------------


def _mock_model(success, aic_real):
    """Return a minimal object with .success (Python bool) and .aic_real."""
    return SimpleNamespace(success=success, aic_real=aic_real)


class _MockComponent:
    def __init__(self, name, prefix):
        self._name = name
        self.prefix = prefix


class _MockParam:
    def __init__(self, value):
        self.value = value


class _MockParams:
    def __init__(self, values):
        self._values = values  # dict: param_name -> float

    def get(self, key, default=None):
        if key in self._values:
            return _MockParam(self._values[key])
        return default

    def __getitem__(self, key):
        return _MockParam(self._values[key])


class _MockFit:
    def __init__(self, components, params_values, success=True):
        self.components = components
        self.params = _MockParams(params_values)
        self.success = success


def _gauss_comp(prefix):
    """Return a mock gaussian Model component with the given prefix."""
    return _MockComponent("gaussianmodel", prefix)


def _single_spaxel_fit_list(fit_result):
    """Wrap one fit result in a (1-model, 1, 1) structure for marginal_fits."""
    return [np.array([[fit_result]], dtype=object)]


# Module-level mock fit objects shared across test classes.
_G1_FLUX = 100.0
_G1_CENTER = 5006.843
_G2_FLUX = 20.0  # ratio = 0.20 < default flux threshold 0.25
_G2_CENTER = 5007.0  # |delta| = 0.157 < default dmu threshold 0.5

_FIT_1G = _MockFit(
    [_gauss_comp("g1_")],
    {"g1_flux": _G1_FLUX, "g1_center": _G1_CENTER},
)
_FIT_2G = _MockFit(
    [_gauss_comp("g1_"), _gauss_comp("g2_")],
    {
        "g1_flux": _G1_FLUX,
        "g1_center": _G1_CENTER,
        "g2_flux": _G2_FLUX,
        "g2_center": _G2_CENTER,
    },
)


# ---------------------------------------------------------------------------
# get_aic
# ---------------------------------------------------------------------------


class TestGetAic:
    """1.8B — get_aic: extract aic_real from a successful ModelResult."""

    def test_success_true_returns_aic_real(self):
        m = _mock_model(True, -1234.5)
        assert tf.get_aic(m) == pytest.approx(-1234.5)

    def test_success_false_returns_error(self):
        m = _mock_model(False, -1234.5)
        assert np.isnan(tf.get_aic(m))

    def test_no_aic_real_attribute_returns_error(self):
        """AttributeError on model.aic_real access must be caught and return error."""
        m = SimpleNamespace(success=True)  # no aic_real attribute
        assert np.isnan(tf.get_aic(m))

    def test_none_model_returns_error(self):
        """None has no .success; AttributeError must be caught and return error."""
        assert np.isnan(tf.get_aic(None))

    def test_custom_error_value_returned(self):
        m = _mock_model(False, -100.0)
        assert tf.get_aic(m, error=-999.0) == pytest.approx(-999.0)


# ---------------------------------------------------------------------------
# choose_model_aic_single
# ---------------------------------------------------------------------------


class TestChooseModelAicSingle:
    """1.8B — choose_model_aic_single: choose best model by delta-AIC."""

    def test_none_returns_minus_one(self):
        assert tf.choose_model_aic_single(None) == -1

    def test_single_model_returns_one(self):
        assert tf.choose_model_aic_single([_mock_model(True, -500.0)]) == 1

    def test_two_models_complex_preferred(self):
        # aic[1] - aic[0] = -1200 - (-1000) = -200 < d_aic=-150 → model 2
        m = [_mock_model(True, -1000.0), _mock_model(True, -1200.0)]
        assert tf.choose_model_aic_single(m) == 2

    def test_two_models_simple_preferred(self):
        # aic[1] - aic[0] = -1100 - (-1000) = -100 > d_aic=-150 → model 1
        m = [_mock_model(True, -1000.0), _mock_model(True, -1100.0)]
        assert tf.choose_model_aic_single(m) == 1

    def test_two_models_all_nan_returns_minus_one(self):
        m = [_mock_model(False, np.nan), _mock_model(False, np.nan)]
        assert tf.choose_model_aic_single(m) == -1

    def test_three_models_branch_2_better_than_1_and_3_better_than_2(self):
        # 2>1 (-200<-150) and 3>2 (-200<-150) → 3
        m = [
            _mock_model(True, -1000.0),
            _mock_model(True, -1200.0),
            _mock_model(True, -1400.0),
        ]
        assert tf.choose_model_aic_single(m) == 3

    def test_three_models_branch_2_better_than_1_but_3_not(self):
        # 2>1 (-200<-150) but 3 not better than 2 (-100>-150) → 2
        m = [
            _mock_model(True, -1000.0),
            _mock_model(True, -1200.0),
            _mock_model(True, -1100.0),
        ]
        assert tf.choose_model_aic_single(m) == 2

    def test_three_models_branch_2_not_better_3_better_than_1(self):
        # 2 not better than 1 (-100>-150), but 3>1 (-200<-150) → 3
        m = [
            _mock_model(True, -1000.0),
            _mock_model(True, -900.0),
            _mock_model(True, -1200.0),
        ]
        assert tf.choose_model_aic_single(m) == 3

    def test_three_models_branch_none_better(self):
        # 2 not better than 1, 3 not better than 1 → 1
        m = [
            _mock_model(True, -1000.0),
            _mock_model(True, -900.0),
            _mock_model(True, -950.0),
        ]
        assert tf.choose_model_aic_single(m) == 1

    def test_custom_d_aic_threshold_respected(self):
        # Diff = -120; with d_aic=-100 this crosses the threshold → 2;
        # with default -150 it does not → 1.
        m = [_mock_model(True, -1000.0), _mock_model(True, -1120.0)]
        assert tf.choose_model_aic_single(m, d_aic=-100) == 2
        assert tf.choose_model_aic_single(m) == 1

    def test_difference_exactly_at_threshold_returns_simpler(self):
        """When aic[1]-aic[0] == d_aic exactly, < is False → simpler model wins."""
        m = [_mock_model(True, -1000.0), _mock_model(True, -1150.0)]  # diff = -150
        assert tf.choose_model_aic_single(m, d_aic=-150) == 1

    @pytest.mark.xfail(
        reason=(
            "Bug: len>3 silently falls through to 'return 0+1' (model 1) instead "
            "of raising ValueError. The docstring has a TODO to generalise beyond 3."
        ),
        strict=True,
    )
    def test_more_than_three_models_raises(self):
        """Should raise ValueError for unsupported list lengths > 3."""
        m = [
            _mock_model(True, -1000.0),
            _mock_model(True, -1200.0),
            _mock_model(True, -1400.0),
            _mock_model(True, -1600.0),
        ]
        with pytest.raises(ValueError):
            tf.choose_model_aic_single(m)

    @pytest.mark.xfail(
        reason=(
            "Bug: when only the simpler model (index 0) fails, aic[0]=NaN so "
            "aic[1]-aic[0]=NaN. NaN < d_aic is False in numpy, so the function "
            "returns model 1 (the failed model) instead of model 2 (the only "
            "successful one)."
        ),
        strict=True,
    )
    def test_only_simpler_model_fails_returns_complex(self):
        """If model 1 has NaN AIC (failed) and model 2 is valid, must return 2."""
        m = [_mock_model(False, np.nan), _mock_model(True, -1200.0)]
        assert tf.choose_model_aic_single(m) == 2


# ---------------------------------------------------------------------------
# choose_model_aic
# ---------------------------------------------------------------------------


class TestChooseModelAic:
    """1.8B — choose_model_aic: spatially-broadcast wrapper."""

    def test_single_1d_list_returns_scalar(self):
        """1D input (single spaxel) delegates to choose_model_aic_single."""
        m = [_mock_model(True, -1000.0), _mock_model(True, -1200.0)]
        result = tf.choose_model_aic(m)
        assert isinstance(result, (int, np.integer))
        assert result == 2

    def test_2d_spatial_output_shape_and_values(self):
        """3D input (ny, nx, n_models) produces output of shape (ny, nx)."""
        # shape (2, 1, 2): 2 rows, 1 column, 2 models
        grid = np.empty((2, 1, 2), dtype=object)
        grid[0, 0, 0] = _mock_model(True, -1000.0)  # spaxel (0,0): diff=-200 → model 2
        grid[0, 0, 1] = _mock_model(True, -1200.0)
        grid[1, 0, 0] = _mock_model(True, -1000.0)  # spaxel (1,0): diff=-100 → model 1
        grid[1, 0, 1] = _mock_model(True, -1100.0)
        result = tf.choose_model_aic(grid)
        assert result.shape == (2, 1)
        assert result[0, 0] == 2
        assert result[1, 0] == 1

    def test_all_nan_spaxel_assigned_minus_one(self):
        """Spaxels where every model has NaN AIC must receive -1."""
        grid = np.empty((1, 1, 2), dtype=object)
        grid[0, 0, 0] = _mock_model(False, np.nan)
        grid[0, 0, 1] = _mock_model(False, np.nan)
        result = tf.choose_model_aic(grid)
        assert result[0, 0] == -1

    def test_d_aic_propagated_to_spatial_broadcast(self):
        """Custom d_aic must reach choose_model_aic_single in each spaxel."""
        grid = np.empty((1, 1, 2), dtype=object)
        # diff = -120; with d_aic=-100 → model 2; with default -150 → model 1
        grid[0, 0, 0] = _mock_model(True, -1000.0)
        grid[0, 0, 1] = _mock_model(True, -1120.0)
        assert tf.choose_model_aic(grid, d_aic=-100)[0, 0] == 2
        assert tf.choose_model_aic(grid)[0, 0] == 1


# ---------------------------------------------------------------------------
# get_ngaussians
# ---------------------------------------------------------------------------


class TestGetNGaussians:
    """1.8B — get_ngaussians: count gaussian components in a Model or ModelResult."""

    def test_zero_gaussians(self):
        assert tf.get_ngaussians(lmfit.models.ConstantModel()) == 0

    def test_one_gaussian(self):
        assert tf.get_ngaussians(tc.models.Const_1GaussModel()) == 1

    def test_two_gaussians(self):
        assert tf.get_ngaussians(tc.models.Const_2GaussModel()) == 2

    def test_three_gaussians(self):
        assert tf.get_ngaussians(tc.models.Const_3GaussModel()) == 3

    def test_constant_component_not_counted(self):
        """Const_1GaussModel = 1 gaussian + 1 constant; count must be 1, not 2."""
        model = tc.models.Const_1GaussModel()
        assert len(model.components) > 1  # confirm there are non-gaussian components
        assert tf.get_ngaussians(model) == 1


# ---------------------------------------------------------------------------
# get_gcomponent_comparison
# ---------------------------------------------------------------------------


class TestGetGComponentComparison:
    """1.8B — get_gcomponent_comparison: relative flux/centre of each secondary gaussian."""

    def test_single_gaussian_returns_empty_list(self):
        assert tf.get_gcomponent_comparison(_FIT_1G) == []

    def test_two_gaussian_result_has_one_row(self):
        """Only one secondary component (g2); main gaussian (g1) is excluded."""
        result = tf.get_gcomponent_comparison(_FIT_2G)
        assert len(result) == 1

    def test_two_gaussian_flux_ratio(self):
        """result[0, 0] must be g2_flux / g1_flux (main = highest flux)."""
        result = tf.get_gcomponent_comparison(_FIT_2G)
        assert result[0, 0] == pytest.approx(_G2_FLUX / _G1_FLUX)

    def test_two_gaussian_delta_center(self):
        """result[0, 1] must be g2_center - g1_center."""
        result = tf.get_gcomponent_comparison(_FIT_2G)
        assert result[0, 1] == pytest.approx(_G2_CENTER - _G1_CENTER)

    def test_secondary_component_is_main_when_higher_flux(self):
        """Main = highest flux; if g2 has more flux, g1 becomes the secondary."""
        fit = _MockFit(
            [_gauss_comp("g1_"), _gauss_comp("g2_")],
            {
                "g1_flux": 20.0,  # lower flux → secondary
                "g1_center": 5006.843,
                "g2_flux": 100.0,  # higher flux → main
                "g2_center": 5007.0,
            },
        )
        result = tf.get_gcomponent_comparison(fit)
        # g1 is the secondary; ratio = 20/100 = 0.2; delta = g1_center - g2_center
        assert result[0, 0] == pytest.approx(20.0 / 100.0)
        assert result[0, 1] == pytest.approx(5006.843 - 5007.0)


# ---------------------------------------------------------------------------
# marginal_fits
# ---------------------------------------------------------------------------


class TestMarginalFits:
    """1.8B — marginal_fits: flag spaxels for manual inspection."""

    def test_none_model_not_flagged(self):
        """`None` means the spaxel was not fit (below SNR); user does not need to check."""
        fit_list = _single_spaxel_fit_list(None)
        choices = np.array([[1]])
        assert tf.marginal_fits(fit_list, choices)[0, 0] == False  # noqa: E712

    def test_failed_fit_flagged(self):
        """A failed fit (.success=False) should always be flagged for inspection."""
        failed = _MockFit(
            [_gauss_comp("g1_")],
            {"g1_flux": 50.0, "g1_center": _G1_CENTER},
            success=False,
        )
        fit_list = _single_spaxel_fit_list(failed)
        choices = np.array([[1]])
        assert tf.marginal_fits(fit_list, choices)[0, 0] == True  # noqa: E712

    def test_single_gaussian_not_flagged(self):
        """A single-gaussian fit needs no inspection."""
        fit_list = _single_spaxel_fit_list(_FIT_1G)
        choices = np.array([[1]])
        assert tf.marginal_fits(fit_list, choices)[0, 0] == False  # noqa: E712

    def test_two_gaussian_both_thresholds_met_flagged(self):
        """ratio=0.20 < 0.25 AND |delta|=0.157 < 0.5 → embedded gaussian → flag."""
        fit_list = _single_spaxel_fit_list(_FIT_2G)
        choices = np.array([[1]])
        assert tf.marginal_fits(fit_list, choices)[0, 0] == True  # noqa: E712

    def test_two_gaussian_ratio_above_threshold_not_flagged(self):
        """ratio=0.40 ≥ 0.25 → second component significant enough; no flag."""
        fit = _MockFit(
            [_gauss_comp("g1_"), _gauss_comp("g2_")],
            {
                "g1_flux": 100.0,
                "g1_center": _G1_CENTER,
                "g2_flux": 40.0,
                "g2_center": 5007.0,
            },
        )
        fit_list = _single_spaxel_fit_list(fit)
        choices = np.array([[1]])
        assert tf.marginal_fits(fit_list, choices)[0, 0] == False  # noqa: E712

    def test_two_gaussian_delta_above_threshold_not_flagged(self):
        """ratio=0.20 < 0.25 but |delta|=0.657 ≥ 0.5 → components well separated; no flag."""
        fit = _MockFit(
            [_gauss_comp("g1_"), _gauss_comp("g2_")],
            {
                "g1_flux": 100.0,
                "g1_center": _G1_CENTER,
                "g2_flux": 20.0,
                "g2_center": 5007.5,
            },
        )
        fit_list = _single_spaxel_fit_list(fit)
        choices = np.array([[1]])
        assert tf.marginal_fits(fit_list, choices)[0, 0] == False  # noqa: E712

    def test_choices_minus_one_not_flagged(self):
        """choices=-1 means invalid spaxel; np.choose clips -2 to 0.

        In production fit_list[0] is always None when choices=-1 (all models
        failed).  The None-path sets output=False for the right reason.  This
        test documents the expected end-to-end behaviour and the clipping
        dependency so a future change to mode='wrap' would be caught;
        """
        fit_list = [np.array([[None]], dtype=object)]
        choices = np.array([[-1]])
        assert tf.marginal_fits(fit_list, choices)[0, 0] == False  # noqa: E712

    def test_custom_flux_threshold(self):
        """flux parameter: ratio=0.30, which is above 0.25 (default) but below 0.35."""
        fit = _MockFit(
            [_gauss_comp("g1_"), _gauss_comp("g2_")],
            {
                "g1_flux": 100.0,
                "g1_center": _G1_CENTER,
                "g2_flux": 30.0,
                "g2_center": 5007.0,
            },  # |delta|=0.157 < 0.5
        )
        fit_list = _single_spaxel_fit_list(fit)
        choices = np.array([[1]])
        # ratio=0.30 ≥ 0.25 default → not flagged
        assert tf.marginal_fits(fit_list, choices)[0, 0] == False  # noqa: E712
        # ratio=0.30 < 0.35 custom → flagged
        assert tf.marginal_fits(fit_list, choices, flux=0.35)[0, 0] == True  # noqa: E712

    def test_custom_dmu_threshold(self):
        """dmu parameter: |delta|=0.40, which is below 0.5 (default) but above 0.35."""
        fit = _MockFit(
            [_gauss_comp("g1_"), _gauss_comp("g2_")],
            {
                "g1_flux": 100.0,
                "g1_center": _G1_CENTER,
                "g2_flux": 20.0,
                "g2_center": _G1_CENTER + 0.40,
            },  # ratio=0.20 < 0.25
        )
        fit_list = _single_spaxel_fit_list(fit)
        choices = np.array([[1]])
        # |delta|=0.40 < 0.5 default → flagged
        assert tf.marginal_fits(fit_list, choices)[0, 0] == True  # noqa: E712
        # |delta|=0.40 ≥ 0.35 custom (stricter) — wait, dmu is upper bound for flagging
        # smaller dmu threshold means we only flag tighter separations → not flagged
        assert tf.marginal_fits(fit_list, choices, dmu=0.35)[0, 0] == False  # noqa: E712
