"""Smoke tests for threadcount.procedures.fit_lines (Phase 1.7)."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

import threadcount as tc
from threadcount.procedures import fit_line, fit_lines

# Module-level constants matching conftest.py (conftest is not meant to be imported
# directly; repeat only the values needed here).
_LINE_CENTER = 5006.843  # [O III] 5007 rest wavelength in Å
_CUBE_NY = 10
_CUBE_NX = 10


class TestUpdateSettings:
    """1.7a — unit tests for fit_lines.update_settings."""

    @pytest.fixture()
    def settings(self, default_settings):
        """Return a deep copy of the session fixture so mutations are isolated."""
        return copy.deepcopy(default_settings)

    # ------------------------------------------------------------------
    # kernel
    # ------------------------------------------------------------------

    def test_kernel_is_2d_ndarray(self, settings):
        fit_lines.update_settings(settings)
        assert isinstance(settings.kernel, np.ndarray)
        assert settings.kernel.ndim == 2

    def test_kernel_sum_positive(self, settings):
        fit_lines.update_settings(settings)
        assert settings.kernel.sum() > 0

    def test_kernel_odd_side_lengths(self, settings):
        fit_lines.update_settings(settings)
        assert settings.kernel.shape[0] % 2 == 1
        assert settings.kernel.shape[1] % 2 == 1

    def test_kernel_center_pixel_is_one(self, settings):
        """The kernel must be centred on the origin — the middle pixel must be 1."""
        fit_lines.update_settings(settings)
        cy = settings.kernel.shape[0] // 2
        cx = settings.kernel.shape[1] // 2
        assert settings.kernel[cy, cx] == 1

    # ------------------------------------------------------------------
    # instrument_dispersion_rest
    # ------------------------------------------------------------------

    def test_instrument_dispersion_rest_equals_dispersion_when_z_zero(self, settings):
        assert settings.z_set == 0
        fit_lines.update_settings(settings)
        assert settings.instrument_dispersion_rest == pytest.approx(
            settings.instrument_dispersion
        )

    def test_instrument_dispersion_rest_scales_correctly_for_nonzero_z(self, settings):
        """dispersion_rest = dispersion / (1 + z) — verify the non-trivial case."""
        settings.z_set = 0.5
        fit_lines.update_settings(settings)
        expected = settings.instrument_dispersion / (1 + 0.5)
        assert settings.instrument_dispersion_rest == pytest.approx(expected)

    # ------------------------------------------------------------------
    # always_manually_choose normalisation
    # ------------------------------------------------------------------

    def test_always_manually_choose_none_becomes_empty_list(self, settings):
        settings.always_manually_choose = None
        fit_lines.update_settings(settings)
        assert settings.always_manually_choose == []

    def test_always_manually_choose_list_is_preserved(self, settings):
        settings.always_manually_choose = [(1, 2), (3, 4)]
        fit_lines.update_settings(settings)
        assert settings.always_manually_choose == [(1, 2), (3, 4)]

    # ------------------------------------------------------------------
    # comment
    # ------------------------------------------------------------------

    def test_comment_is_nonempty_string(self, settings):
        fit_lines.update_settings(settings)
        assert isinstance(settings.comment, str)
        assert len(settings.comment) > 0

    def test_comment_contains_instrument_dispersion(self, settings):
        fit_lines.update_settings(settings)
        assert "instrument_dispersion" in settings.comment

    def test_comment_contains_snr_lower_limit(self, settings):
        fit_lines.update_settings(settings)
        assert "snr_lower_limit" in settings.comment

    def test_comment_contains_units(self, settings):
        fit_lines.update_settings(settings)
        assert "units" in settings.comment

    def test_prior_comment_is_preserved_and_separated(self, settings):
        """A pre-existing comment should appear before the new block, separated by a newline."""
        settings.comment = "initial note"
        fit_lines.update_settings(settings)
        assert settings.comment.startswith("initial note\n")

    def test_prior_comment_already_ending_in_newline_not_doubled(self, settings):
        """A comment that already ends with \\n must not get an extra blank line."""
        settings.comment = "initial note\n"
        fit_lines.update_settings(settings)
        assert not settings.comment.startswith("initial note\n\n")


# ---------------------------------------------------------------------------
# 1.7b — Single-spaxel fit test
# ---------------------------------------------------------------------------


class TestSingleSpaxelFit:
    """1.7b — unit test for fit_line.process_single_spectrum on a single spaxel.

    Manually assembles the inputs that fit_line.run() would produce for one
    spaxel and calls process_single_spectrum() directly, without going through
    the full pipeline.
    """

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @pytest.fixture(scope="class")
    def subcube_av(self, synthetic_cube):
        """Spatially-averaged subcube cut to the [O III] 5007 line window.

        Mirrors the first two steps of fit_line.run():
          1. select_lambda to the line window
          2. spatial_average with a radius-1.5 kernel
        """
        kernel = tc.fit.get_reg_image(tc.fit.get_region(1.5))
        subcube = synthetic_cube.select_lambda(
            tc.lines.L_OIII5007.low, tc.lines.L_OIII5007.high
        )
        return tc.fit.spatial_average(subcube, kernel)

    @pytest.fixture()
    def minimal_settings(self):
        """Minimal SimpleNamespace containing only what process_single_spectrum uses.

        process_single_spectrum only reads three attributes from ``s``:
          - lmfit_kwargs  (passed directly to mpdaf Spectrum.lmfit)
          - chop_bandwidth  (controls retry logic on fit failure)
          - instrument_dispersion_rest  (not accessed inside the function itself,
            but included here for completeness and because callers like fit_line.run
            set the sigma hint using it before calling this function)
        """
        return SimpleNamespace(
            lmfit_kwargs={"method": "least_squares"},
            chop_bandwidth=False,
            instrument_dispersion_rest=0.8,
        )

    @pytest.fixture()
    def result(self, subcube_av, minimal_settings):
        """Return value of process_single_spectrum at the centre spaxel (5, 5)."""
        snr_image = np.full((_CUBE_NY, _CUBE_NX), 999.0)
        return fit_line.process_single_spectrum(
            subcube_av,
            snr_image,
            snr_threshold=3,
            models=[tc.models.Const_1GaussModel()],
            s=minimal_settings,
            idx=(5, 5),
        )

    # ------------------------------------------------------------------
    # Happy-path assertions (ROADMAP spec)
    # ------------------------------------------------------------------

    def test_returns_list(self, result):
        assert isinstance(result, list)

    def test_returns_list_of_length_one(self, result):
        assert len(result) == 1

    def test_result_element_is_not_none(self, result):
        assert result[0] is not None

    def test_success_is_true(self, result):
        assert result[0].success

    def test_center_within_tolerance(self, result):
        """Recovered g1_center must be within 0.5 Å of the injected line center."""
        center = result[0].params["g1_center"].value
        assert abs(center - _LINE_CENTER) < 0.5

    # ------------------------------------------------------------------
    # SNR filtering — gaps identified during the 1.7b code review
    # ------------------------------------------------------------------

    def test_snr_below_threshold_returns_none_list(self, subcube_av, minimal_settings):
        """Spaxels with SNR < threshold must be skipped without fitting."""
        low_snr_image = np.full((_CUBE_NY, _CUBE_NX), 1.0)  # well below threshold=3
        result = fit_line.process_single_spectrum(
            subcube_av,
            low_snr_image,
            snr_threshold=3,
            models=[tc.models.Const_1GaussModel()],
            s=minimal_settings,
            idx=(5, 5),
        )
        assert result == [None]

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Bug in process_single_spectrum: the NaN-SNR guard uses "
            "`np.isnan(snr_image[idx]) is True`, which is an identity comparison "
            "against the Python True singleton. np.isnan() returns numpy.bool_ "
            "(confirmed numpy 1.26.4), which is NOT the same object, so the "
            "comparison always evaluates to False. Spaxels whose SNR is NaN are "
            "therefore NOT skipped — they are passed straight to the fitter. "
            "Fix: replace `is True` with a plain truthiness test: "
            "`np.isnan(snr_image[idx])`. Fix target: Phase 2."
        ),
    )
    def test_snr_nan_skips_spaxel(self, subcube_av, minimal_settings):
        """A spaxel with SNR = NaN must be treated as failing the threshold check."""
        nan_snr_image = np.full((_CUBE_NY, _CUBE_NX), 999.0)
        nan_snr_image[5, 5] = np.nan
        result = fit_line.process_single_spectrum(
            subcube_av,
            nan_snr_image,
            snr_threshold=3,
            models=[tc.models.Const_1GaussModel()],
            s=minimal_settings,
            idx=(5, 5),
        )
        assert result == [None]
