"""Unit tests for process_single_spectrum and choose_best_fits (Phase 1.9A).

Source: threadcount.procedures.fit_line
"""

import copy
from types import SimpleNamespace

import lmfit
import mpdaf.obj
import numpy as np
import pytest

import threadcount as tc
from threadcount.procedures import fit_line

# ---------------------------------------------------------------------------
# Module-level constants (matching conftest.py — not imported directly)
# ---------------------------------------------------------------------------
_LINE_CENTER = 5006.843  # [O III] 5007 Å
_WAVE_START = 4990.0
_SNR_THRESHOLD = 3.0

# ---------------------------------------------------------------------------
# Mock helpers for choose_best_fits tests
# ---------------------------------------------------------------------------


class _Comp:
    """Minimal model component stub for get_ngaussians (needs ._name)."""

    def __init__(self, prefix, is_gaussian=True):
        self._name = "gaussianmodel" if is_gaussian else "constantmodel"
        self.prefix = prefix


def _mock_1g(aic=-1000.0, success=True):
    """Mock 1-gaussian ModelResult for AIC and marginal_fits."""
    return SimpleNamespace(success=success, aic_real=aic, components=[_Comp("g1_")])


def _mock_2g(aic=-900.0, success=True):
    """Mock 2-gaussian ModelResult for AIC and marginal_fits.

    With aic1=-1000, aic2=-900: diff = +100 > d_aic=-150, so model 1 is chosen.
    With aic1=-1000, aic2=-1200: diff = -200 < d_aic=-150, so model 2 is chosen.
    """
    return SimpleNamespace(
        success=success, aic_real=aic, components=[_Comp("g1_"), _Comp("g2_")]
    )


def _make_fit_results(shape, aic1=-1000.0, aic2=-900.0):
    """Build (n_models=2, ny, nx) object array and its (ny, nx, n_models) transpose."""
    ny, nx = shape
    arr1 = np.empty((ny, nx), dtype=object)
    arr2 = np.empty((ny, nx), dtype=object)
    for idx in np.ndindex(ny, nx):
        arr1[idx] = _mock_1g(aic=aic1)
        arr2[idx] = _mock_2g(aic=aic2)
    fit_results = np.array([arr1, arr2])  # (2, ny, nx)
    fit_results_T = fit_results.transpose((1, 2, 0))  # (ny, nx, 2)
    return fit_results, fit_results_T


# ---------------------------------------------------------------------------
# process_single_spectrum
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _subcube(synthetic_cube):
    """Return a subcube of the synthetic cube around the [O III] line.

    No spatial averaging — just a wavelength slice.  The raw per-spaxel
    spectra still contain a clean Gaussian, which is enough for fitting.
    """
    return synthetic_cube.select_lambda(4990.0, 5025.0)


@pytest.fixture()
def _s():
    """Minimal settings namespace for process_single_spectrum."""
    return SimpleNamespace(
        lmfit_kwargs={"method": "least_squares"},
        chop_bandwidth=False,
    )


class TestProcessSingleSpectrum:
    """1.9A — process_single_spectrum: SNR gating, fitting, and chop_bandwidth."""

    # ------------------------------------------------------------------
    # SNR gate
    # ------------------------------------------------------------------

    def test_snr_below_threshold_returns_none_list(self, _subcube, _s):
        """SNR < threshold → skip fitting, return [None]."""
        snr = np.full(_subcube.shape[1:], 1.0)  # well below threshold of 3
        models = [tc.models.Const_1GaussModel()]
        result = fit_line.process_single_spectrum(
            _subcube, snr, _SNR_THRESHOLD, models, _s, (0, 0)
        )
        assert result == [None]

    def test_snr_equal_to_threshold_proceeds_to_fitting(self, _subcube, _s):
        """SNR exactly equal to threshold is NOT gated out (gate is strict '<')."""
        snr = np.full(_subcube.shape[1:], _SNR_THRESHOLD)
        models = [tc.models.Const_1GaussModel()]
        result = fit_line.process_single_spectrum(
            _subcube, snr, _SNR_THRESHOLD, models, _s, (0, 0)
        )
        # The gate is: snr < threshold, so snr==threshold passes through.
        assert result != [None]
        assert result[0] is not None

    def test_snr_nan_returns_none_list(self, _subcube, _s):
        """NaN SNR value must return [None]."""
        snr = np.full(_subcube.shape[1:], np.nan)
        models = [tc.models.Const_1GaussModel()]
        result = fit_line.process_single_spectrum(
            _subcube, snr, _SNR_THRESHOLD, models, _s, (0, 0)
        )
        assert result == [None]

    # ------------------------------------------------------------------
    # Successful fits
    # ------------------------------------------------------------------

    def test_single_model_success_returns_one_element_list(self, _subcube, _s):
        """Single model, high-SNR spaxel → returns a one-element list."""
        snr = np.full(_subcube.shape[1:], 20.0)
        models = [tc.models.Const_1GaussModel()]
        result = fit_line.process_single_spectrum(
            _subcube, snr, _SNR_THRESHOLD, models, _s, (0, 0)
        )
        assert len(result) == 1

    def test_single_model_result_is_not_none(self, _subcube, _s):
        """Single successful fit returns a non-None result."""
        snr = np.full(_subcube.shape[1:], 20.0)
        models = [tc.models.Const_1GaussModel()]
        result = fit_line.process_single_spectrum(
            _subcube, snr, _SNR_THRESHOLD, models, _s, (0, 0)
        )
        assert result[0] is not None

    def test_single_model_result_successful(self, _subcube, _s):
        """The returned ModelResult must have success=True."""
        snr = np.full(_subcube.shape[1:], 20.0)
        models = [tc.models.Const_1GaussModel()]
        result = fit_line.process_single_spectrum(
            _subcube, snr, _SNR_THRESHOLD, models, _s, (0, 0)
        )
        assert result[0].success

    def test_two_models_returns_two_element_list(self, _subcube, _s):
        """Two models, high SNR → returns a list with two entries."""
        snr = np.full(_subcube.shape[1:], 20.0)
        models = [tc.models.Const_1GaussModel(), tc.models.Const_2GaussModel()]
        result = fit_line.process_single_spectrum(
            _subcube, snr, _SNR_THRESHOLD, models, _s, (0, 0)
        )
        assert len(result) == 2

    def test_two_models_both_not_none(self, _subcube, _s):
        """Both model fits must be non-None when SNR is well above threshold."""
        snr = np.full(_subcube.shape[1:], 20.0)
        models = [tc.models.Const_1GaussModel(), tc.models.Const_2GaussModel()]
        result = fit_line.process_single_spectrum(
            _subcube, snr, _SNR_THRESHOLD, models, _s, (0, 0)
        )
        assert result[0] is not None
        assert result[1] is not None

    def test_result_spaxel_index_applies_correctly(self, _subcube, _s):
        """Fitting different spaxels returns independent results."""
        snr = np.full(_subcube.shape[1:], 20.0)
        models = [tc.models.Const_1GaussModel()]
        r0 = fit_line.process_single_spectrum(
            _subcube, snr, _SNR_THRESHOLD, models, _s, (0, 0)
        )
        r1 = fit_line.process_single_spectrum(
            _subcube, snr, _SNR_THRESHOLD, models, _s, (3, 5)
        )
        # Both should succeed; the fitted center should be close to _LINE_CENTER
        for r in (r0, r1):
            assert len(r) == 1
            assert r[0] is not None

    # ------------------------------------------------------------------
    # Fit failure paths (via monkeypatch)
    # ------------------------------------------------------------------

    def test_lmfit_returns_none_returns_none_list(self, _subcube, _s, monkeypatch):
        """lmfit returning None (all-masked spectrum) → [None]."""
        monkeypatch.setattr(
            mpdaf.obj.spectrum.Spectrum, "lmfit", lambda self, *a, **kw: None
        )
        snr = np.full(_subcube.shape[1:], 20.0)
        models = [tc.models.Const_1GaussModel()]
        result = fit_line.process_single_spectrum(
            _subcube, snr, _SNR_THRESHOLD, models, _s, (0, 0)
        )
        assert result == [None]

    def test_first_model_fail_chop_false_returns_none_list(
        self, _subcube, _s, monkeypatch
    ):
        """First fit fails, chop_bandwidth=False → return [None] immediately."""
        failed = SimpleNamespace(success=False)
        monkeypatch.setattr(
            mpdaf.obj.spectrum.Spectrum, "lmfit", lambda self, *a, **kw: failed
        )
        models = [tc.models.Const_1GaussModel()]
        snr = np.full(_subcube.shape[1:], 20.0)
        result = fit_line.process_single_spectrum(
            _subcube, snr, _SNR_THRESHOLD, models, _s, (0, 0)
        )
        assert result == [None]

    def test_first_model_fail_chop_true_retry_succeeds(self, _subcube, monkeypatch):
        """chop_bandwidth=True: first fit fails, chopped retry succeeds → [result]."""
        s_chop = SimpleNamespace(
            lmfit_kwargs={"method": "least_squares"}, chop_bandwidth=True
        )
        call_count = {"n": 0}

        def _fake_lmfit(self, *args, **kwargs):
            if call_count["n"] == 0:
                call_count["n"] += 1
                return SimpleNamespace(success=False)
            call_count["n"] += 1
            return SimpleNamespace(success=True)

        monkeypatch.setattr(mpdaf.obj.spectrum.Spectrum, "lmfit", _fake_lmfit)
        snr = np.full(_subcube.shape[1:], 20.0)
        models = [tc.models.Const_1GaussModel()]
        result = fit_line.process_single_spectrum(
            _subcube, snr, _SNR_THRESHOLD, models, s_chop, (0, 0)
        )
        assert len(result) == 1
        assert result[0].success is True
        assert call_count["n"] == 2  # first call failed, second succeeded

    def test_first_model_fail_chop_true_retry_also_fails_returns_none(
        self, _subcube, monkeypatch
    ):
        """chop_bandwidth=True: both full and chopped fits fail → [None]."""
        s_chop = SimpleNamespace(
            lmfit_kwargs={"method": "least_squares"}, chop_bandwidth=True
        )
        failed = SimpleNamespace(success=False)
        monkeypatch.setattr(
            mpdaf.obj.spectrum.Spectrum, "lmfit", lambda self, *a, **kw: failed
        )
        snr = np.full(_subcube.shape[1:], 20.0)
        models = [tc.models.Const_1GaussModel()]
        result = fit_line.process_single_spectrum(
            _subcube, snr, _SNR_THRESHOLD, models, s_chop, (0, 0)
        )
        assert result == [None]

    def test_first_model_fail_chop_true_retry_returns_none_on_masked_spectrum(
        self, _subcube, monkeypatch
    ):
        """chop_bandwidth=True: first fails (success=False), retry returns None → [None].

        Currently crashes with AttributeError because the retry has no None guard.
        """
        s_chop = SimpleNamespace(
            lmfit_kwargs={"method": "least_squares"}, chop_bandwidth=True
        )
        call_count = {"n": 0}

        def _fake_lmfit(self, *args, **kwargs):
            if call_count["n"] == 0:
                call_count["n"] += 1
                return SimpleNamespace(success=False)
            call_count["n"] += 1
            return None  # chopped spectrum all-masked

        monkeypatch.setattr(mpdaf.obj.spectrum.Spectrum, "lmfit", _fake_lmfit)
        snr = np.full(_subcube.shape[1:], 20.0)
        models = [tc.models.Const_1GaussModel()]
        # Should return [None], but currently raises AttributeError
        result = fit_line.process_single_spectrum(
            _subcube, snr, _SNR_THRESHOLD, models, s_chop, (0, 0)
        )
        assert result == [None]


# ---------------------------------------------------------------------------
# choose_best_fits
# ---------------------------------------------------------------------------


class TestChooseBestFits:
    """1.9A — choose_best_fits: AIC selection, marginal_fits, always_manually_choose."""

    @pytest.fixture()
    def s_noninteractive(self):
        return SimpleNamespace(
            d_aic=-150.0,
            interactively_choose_fits=False,
            always_manually_choose=[],
        )

    # ------------------------------------------------------------------
    # Single model: everything is None
    # ------------------------------------------------------------------

    def test_single_model_auto_aic_choices_is_none(self, s_noninteractive):
        """Single model → auto_aic_choices is None (no AIC selection needed)."""
        models = [tc.models.Const_1GaussModel()]
        shape = (2, 2)
        fit_results = np.full((1, *shape), None, dtype=object)
        fit_results_T = fit_results.transpose((1, 2, 0))
        _, _, auto, _ = fit_line.choose_best_fits(
            models, fit_results_T, s_noninteractive, fit_results
        )
        assert auto is None

    def test_single_model_user_check_is_none(self, s_noninteractive):
        """Single model → user_check is None."""
        models = [tc.models.Const_1GaussModel()]
        shape = (2, 2)
        fit_results = np.full((1, *shape), None, dtype=object)
        fit_results_T = fit_results.transpose((1, 2, 0))
        _, _, _, user = fit_line.choose_best_fits(
            models, fit_results_T, s_noninteractive, fit_results
        )
        assert user is None

    def test_single_model_final_choices_is_none(self, s_noninteractive):
        """Single model → final_choices is None."""
        models = [tc.models.Const_1GaussModel()]
        shape = (2, 2)
        fit_results = np.full((1, *shape), None, dtype=object)
        fit_results_T = fit_results.transpose((1, 2, 0))
        _, final, _, _ = fit_line.choose_best_fits(
            models, fit_results_T, s_noninteractive, fit_results
        )
        assert final is None

    def test_single_model_chosen_models_equals_fit_results_0(self, s_noninteractive):
        """Single model → chosen_models is fit_results[0] (the only model grid)."""
        models = [tc.models.Const_1GaussModel()]
        shape = (2, 2)
        fit_results = np.full((1, *shape), None, dtype=object)
        fit_results_T = fit_results.transpose((1, 2, 0))
        chosen, _, _, _ = fit_line.choose_best_fits(
            models, fit_results_T, s_noninteractive, fit_results
        )
        np.testing.assert_array_equal(chosen, fit_results[0])

    # ------------------------------------------------------------------
    # Multiple models, non-interactive
    # ------------------------------------------------------------------

    def test_multiple_models_final_choices_populated(self, s_noninteractive):
        """Two models, interactively_choose_fits=False → final_choices is not None."""
        models = [tc.models.Const_1GaussModel(), tc.models.Const_2GaussModel()]
        fit_results, fit_results_T = _make_fit_results((2, 2))
        _, final, _, _ = fit_line.choose_best_fits(
            models, fit_results_T, s_noninteractive, fit_results
        )
        assert final is not None

    def test_multiple_models_final_choices_shape_matches_spatial(
        self, s_noninteractive
    ):
        """final_choices must have the same shape as the spatial grid."""
        models = [tc.models.Const_1GaussModel(), tc.models.Const_2GaussModel()]
        shape = (3, 4)
        fit_results, fit_results_T = _make_fit_results(shape)
        _, final, _, _ = fit_line.choose_best_fits(
            models, fit_results_T, s_noninteractive, fit_results
        )
        assert final.shape == shape

    def test_multiple_models_final_choices_equals_auto_aic_choices(
        self, s_noninteractive
    ):
        """Non-interactive: final_choices must equal auto_aic_choices exactly."""
        models = [tc.models.Const_1GaussModel(), tc.models.Const_2GaussModel()]
        fit_results, fit_results_T = _make_fit_results((2, 2))
        _, final, auto, _ = fit_line.choose_best_fits(
            models, fit_results_T, s_noninteractive, fit_results
        )
        np.testing.assert_array_equal(final, auto)

    def test_multiple_models_user_check_not_none(self, s_noninteractive):
        """Two models → user_check is not None (comes from marginal_fits)."""
        models = [tc.models.Const_1GaussModel(), tc.models.Const_2GaussModel()]
        fit_results, fit_results_T = _make_fit_results((2, 2))
        _, _, _, user = fit_line.choose_best_fits(
            models, fit_results_T, s_noninteractive, fit_results
        )
        assert user is not None

    def test_multiple_models_user_check_shape_matches_spatial(self, s_noninteractive):
        """user_check must have the same shape as the spatial grid."""
        models = [tc.models.Const_1GaussModel(), tc.models.Const_2GaussModel()]
        shape = (3, 4)
        fit_results, fit_results_T = _make_fit_results(shape)
        _, _, _, user = fit_line.choose_best_fits(
            models, fit_results_T, s_noninteractive, fit_results
        )
        assert user.shape == shape

    # ------------------------------------------------------------------
    # always_manually_choose
    # ------------------------------------------------------------------

    def test_always_manually_choose_sets_user_check_true(self):
        """Pixels in always_manually_choose must have user_check=True regardless."""
        models = [tc.models.Const_1GaussModel(), tc.models.Const_2GaussModel()]
        shape = (2, 2)
        fit_results, fit_results_T = _make_fit_results(shape)
        s = SimpleNamespace(
            d_aic=-150.0,
            interactively_choose_fits=False,
            always_manually_choose=[(0, 0), (1, 1)],
        )
        _, _, _, user = fit_line.choose_best_fits(models, fit_results_T, s, fit_results)
        assert user[(0, 0)]
        assert user[(1, 1)]

    def test_empty_always_manually_choose_user_check_from_marginal_only(
        self, s_noninteractive
    ):
        """always_manually_choose=[] → user_check determined purely by marginal_fits.

        With AIC choosing model 1 (1G) everywhere, no embedded components
        are possible, so marginal_fits returns False for every spaxel.
        """
        models = [tc.models.Const_1GaussModel(), tc.models.Const_2GaussModel()]
        # aic1=-1000, aic2=-900: diff=+100 > d_aic=-150, model 1 chosen everywhere
        fit_results, fit_results_T = _make_fit_results(
            (2, 2), aic1=-1000.0, aic2=-900.0
        )
        _, _, _, user = fit_line.choose_best_fits(
            models, fit_results_T, s_noninteractive, fit_results
        )
        # marginal_fits: 1G chosen, ngauss=1 → no need to check → all False
        assert not user.any()

    # ------------------------------------------------------------------
    # auto_aic_choices values and chosen_models content
    # ------------------------------------------------------------------

    def test_auto_aic_choices_all_one_when_model_one_wins(self, s_noninteractive):
        """AIC gap +100 > d_aic=-150 → model 1 chosen everywhere (value==1)."""
        models = [tc.models.Const_1GaussModel(), tc.models.Const_2GaussModel()]
        # aic2 - aic1 = -900 - (-1000) = +100 > -150: model 1 wins
        fit_results, fit_results_T = _make_fit_results(
            (3, 3), aic1=-1000.0, aic2=-900.0
        )
        _, _, auto, _ = fit_line.choose_best_fits(
            models, fit_results_T, s_noninteractive, fit_results
        )
        assert (auto == 1).all()

    def test_multiple_models_chosen_models_shape_matches_spatial(
        self, s_noninteractive
    ):
        """chosen_models must have the same spatial shape as the input grid."""
        models = [tc.models.Const_1GaussModel(), tc.models.Const_2GaussModel()]
        shape = (3, 4)
        fit_results, fit_results_T = _make_fit_results(shape)
        chosen, _, _, _ = fit_line.choose_best_fits(
            models, fit_results_T, s_noninteractive, fit_results
        )
        assert chosen.shape == shape

    def test_auto_aic_chooses_model_two_when_significantly_better(
        self, s_noninteractive, monkeypatch
    ):
        """AIC gap -300 < d_aic=-150 → model 2 chosen everywhere.

        marginal_fits is patched out because _mock_2g() has no .params, which
        is only needed by marginal_fits — not by the AIC-selection path itself.
        chosen_models must then contain the 2-gaussian mock objects from
        fit_results[1], not the 1-gaussian objects from fit_results[0].
        """
        models = [tc.models.Const_1GaussModel(), tc.models.Const_2GaussModel()]
        shape = (2, 2)
        # aic2 - aic1 = -1200 - (-900) = -300 < -150: model 2 wins
        fit_results, fit_results_T = _make_fit_results(shape, aic1=-900.0, aic2=-1200.0)
        monkeypatch.setattr(
            tc.fit, "marginal_fits", lambda *a, **kw: np.zeros(shape, dtype=bool)
        )
        chosen, _, auto, _ = fit_line.choose_best_fits(
            models, fit_results_T, s_noninteractive, fit_results
        )
        assert (auto == 2).all()
        # chosen_models[0, 0] should be the 2G mock (from fit_results[1])
        assert len(chosen[0, 0].components) == 2
