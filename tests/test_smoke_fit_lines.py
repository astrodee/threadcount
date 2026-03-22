"""Smoke tests for threadcount.procedures.fit_lines (Phase 1.7)."""

import copy
import sys
from types import SimpleNamespace

import lmfit
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


# ---------------------------------------------------------------------------
# 1.7c — fit_line.run() for a single line, no MC, no file I/O assertions
# ---------------------------------------------------------------------------

_SUBREGION_SLICE = (slice(None), slice(4, 7), slice(4, 7))  # 3×3 centre region


class TestFitLineRun3x3:
    """1.7c — call fit_line.run() on a 3×3 subregion with mc_n_iterations=0.

    Does NOT assert file content — that is covered by 1.7d.
    """

    @pytest.fixture(scope="class")
    def run_state(self, synthetic_cube, tmp_path_factory):
        """Build settings, run fit_line.run(), and return the mutated settings object.

        The fixture is class-scoped so the (slow) fitting step runs only once
        for all assertions in this class.
        """
        tmp_path = tmp_path_factory.mktemp("fit_line_3x3")
        s = SimpleNamespace(
            setup_parameters=False,
            monitor_pixels=[],
            baseline_subtract=None,
            baseline_fit_range=None,
            output_filename=str(tmp_path / "output"),
            save_plots=False,
            region_averaging_radius=1.5,
            instrument_dispersion=0.8,
            lmfit_kwargs={"method": "least_squares"},
            snr_lower_limit=3,
            lines=[tc.lines.L_OIII5007],
            models=[[tc.models.Const_1GaussModel()]],
            d_aic=-150,
            interactively_choose_fits=False,
            always_manually_choose=[],
            mc_snr=25,
            mc_n_iterations=0,
            parallel=False,
            n_process=4,
            chop_bandwidth=False,
            SNR_HalfBW=9,
            SNR_Baseline_q=0.15,
            # Spatial subregion: centre 3×3 of the synthetic cube
            cube=synthetic_cube[_SUBREGION_SLICE],
            continuum_cube=None,
            z_set=0,
            comment="",
            _i=0,
        )
        fit_lines.update_settings(s)
        fit_line.run(s)
        return s

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    def test_run_does_not_raise(self, run_state):
        """fit_line.run() must set model_results — proves it ran to completion."""
        assert hasattr(run_state, "model_results")

    def test_model_results_shape(self, run_state):
        """model_results must be shaped (n_models, ny, nx) = (1, 3, 3)."""
        assert run_state.model_results.shape == (1, 3, 3)

    def test_all_results_not_none(self, run_state):
        """Every spaxel must have been fitted — the synthetic SNR is >> threshold."""
        assert all(
            run_state.model_results[0, y, x] is not None
            for y in range(3)
            for x in range(3)
        )

    def test_all_results_are_model_result_instances(self, run_state):
        """Every fitted entry must be an lmfit ModelResult, not some other object."""
        assert all(
            isinstance(run_state.model_results[0, y, x], lmfit.model.ModelResult)
            for y in range(3)
            for x in range(3)
        )

    def test_all_results_success(self, run_state):
        """Every fitted spaxel must report success=True — clean synthetic data."""
        assert all(
            run_state.model_results[0, y, x].success for y in range(3) for x in range(3)
        )


# ---------------------------------------------------------------------------
# 1.7d — Output files are created and non-empty
# ---------------------------------------------------------------------------

_LINE_SAVE_STR = "5007"  # str(round(L_OIII5007.center)) — used in output filenames

# Sanity-check: if L_OIII5007's save_str ever changes, the file-name assertions
# below will give a confusing "FileNotFoundError" rather than a clear failure.
assert tc.lines.L_OIII5007.save_str == _LINE_SAVE_STR, (
    f"L_OIII5007.save_str changed to {tc.lines.L_OIII5007.save_str!r}; "
    f"update _LINE_SAVE_STR to match"
)


def _assert_has_data_rows(path):
    """Assert *path* contains at least one non-comment, non-blank line.

    ``ResultDict.savetxt`` writes comment lines starting with ``#`` before any
    data.  A file that is non-zero bytes but consists *only* of comment lines
    (e.g. when every spaxel returned None) would pass a simple size check yet
    contain no usable data.
    """
    data_lines = [
        line
        for line in path.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert len(data_lines) > 0, f"{path.name} contains no data rows (only comments)"


def _make_run_settings(
    synthetic_cube, tmp_path, models_list, _i=0, baseline_subtract=None
):
    """Build and return a settings SimpleNamespace ready for fit_line.run().

    Parameters
    ----------
    models_list : list of lmfit.Model
        The list of models to fit (one entry → no best_fit file written).
    _i : int, optional
        Index into ``s.lines`` / ``s.models`` that ``fit_line.run`` will
        process.  Defaults to 0.  Must be 0 for a single-line setup.
    baseline_subtract : None
        Must remain None for any test relying on the session-scoped
        ``synthetic_cube``.  If non-None, ``fit_line.run`` would call
        ``tc.fit.remove_baseline``, which subtracts in-place from a subcube
        that is a *view* into ``synthetic_cube``, corrupting shared test state.
        Any test that needs baseline subtraction must pass a private
        ``synthetic_cube.copy()`` directly rather than going through this helper.
    """
    assert baseline_subtract is None, (
        "_make_run_settings shares the session-scoped synthetic_cube. "
        "baseline_subtract != None would mutate it via a select_lambda view. "
        "Create a dedicated test that passes synthetic_cube.copy() directly."
    )
    s = SimpleNamespace(
        setup_parameters=False,
        monitor_pixels=[],
        baseline_subtract=None,
        baseline_fit_range=None,
        output_filename=str(tmp_path / "output"),
        save_plots=False,
        region_averaging_radius=1.5,
        instrument_dispersion=0.8,
        lmfit_kwargs={"method": "least_squares"},
        snr_lower_limit=3,
        lines=[tc.lines.L_OIII5007],
        models=[models_list],
        d_aic=-150,
        interactively_choose_fits=False,
        always_manually_choose=[],
        mc_snr=25,
        mc_n_iterations=0,
        parallel=False,
        n_process=4,
        chop_bandwidth=False,
        SNR_HalfBW=9,
        SNR_Baseline_q=0.15,
        cube=synthetic_cube,
        continuum_cube=None,
        z_set=0,
        comment="",
        _i=_i,
    )
    assert s._i < len(s.lines) and s._i < len(s.models), (
        f"_i={s._i} is out of range for lines (len={len(s.lines)}) "
        f"and models (len={len(s.models)}). "
        "Add additional lines/models to the helper call, or use a dedicated fixture."
    )
    fit_lines.update_settings(s)
    return s


class TestFitLineRunOutputFiles:
    """1.7d — assert the expected output .txt files are created and non-empty.

    Two sub-cases:
    - Single model: only simple_model.txt and mc_best_fit.txt are written.
    - Two models: all three files (simple_model, best_fit, mc_best_fit) are written.
    """

    # ------------------------------------------------------------------
    # Single-model run
    # ------------------------------------------------------------------

    @pytest.fixture(scope="class")
    def single_model_run(self, synthetic_cube, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("output_single")
        s = _make_run_settings(
            synthetic_cube, tmp_path, [tc.models.Const_1GaussModel()]
        )
        fit_line.run(s)
        return s, tmp_path

    def test_simple_model_file_exists_single(self, single_model_run):
        s, tmp_path = single_model_run
        expected = tmp_path / f"output_{_LINE_SAVE_STR}_simple_model.txt"
        assert expected.exists()

    def test_simple_model_file_has_data_rows_single(self, single_model_run):
        s, tmp_path = single_model_run
        expected = tmp_path / f"output_{_LINE_SAVE_STR}_simple_model.txt"
        _assert_has_data_rows(expected)

    def test_mc_best_fit_file_exists_single(self, single_model_run):
        s, tmp_path = single_model_run
        expected = tmp_path / f"output_{_LINE_SAVE_STR}_mc_best_fit.txt"
        assert expected.exists()

    def test_mc_best_fit_file_has_data_rows_single(self, single_model_run):
        s, tmp_path = single_model_run
        expected = tmp_path / f"output_{_LINE_SAVE_STR}_mc_best_fit.txt"
        _assert_has_data_rows(expected)

    def test_best_fit_file_absent_single(self, single_model_run):
        """best_fit.txt must NOT be written when there is only one model."""
        s, tmp_path = single_model_run
        absent = tmp_path / f"output_{_LINE_SAVE_STR}_best_fit.txt"
        assert not absent.exists()

    # ------------------------------------------------------------------
    # Two-model run
    # ------------------------------------------------------------------

    @pytest.fixture(scope="class")
    def two_model_run(self, synthetic_cube, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("output_two")
        s = _make_run_settings(
            synthetic_cube,
            tmp_path,
            [tc.models.Const_1GaussModel(), tc.models.Const_2GaussModel()],
        )
        fit_line.run(s)
        return s, tmp_path

    def test_simple_model_file_exists_two(self, two_model_run):
        s, tmp_path = two_model_run
        expected = tmp_path / f"output_{_LINE_SAVE_STR}_simple_model.txt"
        assert expected.exists()

    def test_simple_model_file_has_data_rows_two(self, two_model_run):
        s, tmp_path = two_model_run
        expected = tmp_path / f"output_{_LINE_SAVE_STR}_simple_model.txt"
        _assert_has_data_rows(expected)

    def test_best_fit_file_exists_two(self, two_model_run):
        """best_fit.txt must be written when there are multiple models."""
        s, tmp_path = two_model_run
        expected = tmp_path / f"output_{_LINE_SAVE_STR}_best_fit.txt"
        assert expected.exists()

    def test_best_fit_file_has_data_rows_two(self, two_model_run):
        s, tmp_path = two_model_run
        expected = tmp_path / f"output_{_LINE_SAVE_STR}_best_fit.txt"
        _assert_has_data_rows(expected)

    def test_mc_best_fit_file_exists_two(self, two_model_run):
        s, tmp_path = two_model_run
        expected = tmp_path / f"output_{_LINE_SAVE_STR}_mc_best_fit.txt"
        assert expected.exists()

    def test_mc_best_fit_file_has_data_rows_two(self, two_model_run):
        s, tmp_path = two_model_run
        expected = tmp_path / f"output_{_LINE_SAVE_STR}_mc_best_fit.txt"
        _assert_has_data_rows(expected)


# ---------------------------------------------------------------------------
# 1.7e — ResultDict.loadtxt round-trip
# ---------------------------------------------------------------------------


class TestResultDictRoundTrip:
    """1.7e — load the files written by 1.7d and assert the round-trip is intact.

    The ``single_model_run`` fixture is duplicated from TestFitLineRunOutputFiles
    because pytest does not share class-scoped fixtures across classes; the fit
    therefore runs a second time here.  The two files checked (simple_model and
    mc_best_fit) cover the single-model code path; the full-pipeline path is
    exercised in 1.7d.
    """

    # ------------------------------------------------------------------
    # Fixture — borrow the already-run single-model output directory
    # ------------------------------------------------------------------

    @pytest.fixture(scope="class")
    def single_model_run(self, synthetic_cube, tmp_path_factory):
        """Duplicate of TestFitLineRunOutputFiles.single_model_run.

        pytest class-scoped fixtures are not shared across classes, so this
        fixture is repeated here.  The class-scope ensures fit_line.run()
        executes only once for all assertions in this class.
        """
        tmp_path = tmp_path_factory.mktemp("roundtrip_single")
        s = _make_run_settings(
            synthetic_cube, tmp_path, [tc.models.Const_1GaussModel()]
        )
        fit_line.run(s)
        return s, tmp_path

    @pytest.fixture(scope="class")
    def simple_model_result(self, single_model_run):
        """Load simple_model.txt with ResultDict.loadtxt."""
        s, tmp_path = single_model_run
        fname = str(tmp_path / f"output_{_LINE_SAVE_STR}_simple_model.txt")
        return tc.fit.ResultDict.loadtxt(fname)

    @pytest.fixture(scope="class")
    def mc_result(self, single_model_run):
        """Load mc_best_fit.txt with ResultDict.loadtxt."""
        s, tmp_path = single_model_run
        fname = str(tmp_path / f"output_{_LINE_SAVE_STR}_mc_best_fit.txt")
        return tc.fit.ResultDict.loadtxt(fname)

    # ------------------------------------------------------------------
    # simple_model.txt assertions
    # ------------------------------------------------------------------

    def test_simple_model_is_result_dict(self, simple_model_result):
        """Loaded object must be a ResultDict (OrderedDict subclass)."""
        from collections import OrderedDict

        assert isinstance(simple_model_result, tc.fit.ResultDict)
        assert isinstance(simple_model_result, OrderedDict)

    def test_simple_model_has_row_key(self, simple_model_result):
        assert "row" in simple_model_result

    def test_simple_model_has_col_key(self, simple_model_result):
        assert "col" in simple_model_result

    def test_simple_model_spatial_shape(self, simple_model_result):
        """Inferred spatial shape must match the full 10×10 synthetic cube."""
        ny = int(simple_model_result["row"].max()) + 1
        nx = int(simple_model_result["col"].max()) + 1
        assert (ny, nx) == (_CUBE_NY, _CUBE_NX)

    def test_simple_model_has_g1_center(self, simple_model_result):
        assert "g1_center" in simple_model_result

    def test_simple_model_g1_center_has_finite_values(self, simple_model_result):
        """All spaxels on a clean SNR=25 cube must produce a finite g1_center."""
        assert np.all(np.isfinite(simple_model_result["g1_center"]))

    def test_simple_model_g1_center_near_line(self, simple_model_result):
        """All finite g1_center values must lie within 0.5 Å of the injected center."""
        centers = simple_model_result["g1_center"]
        finite = centers[np.isfinite(centers)]
        assert finite.size > 0, "no finite g1_center values — all fits failed"
        assert np.all(np.abs(finite - _LINE_CENTER) < 0.5)

    def test_simple_model_comment_is_str(self, simple_model_result):
        """loadtxt must reconstruct the .comment attribute as a string."""
        assert isinstance(simple_model_result.comment, str)

    # ------------------------------------------------------------------
    # mc_best_fit.txt assertions
    # ------------------------------------------------------------------

    def test_mc_result_is_result_dict(self, mc_result):
        from collections import OrderedDict

        assert isinstance(mc_result, tc.fit.ResultDict)
        assert isinstance(mc_result, OrderedDict)

    def test_mc_result_has_row_key(self, mc_result):
        assert "row" in mc_result

    def test_mc_result_has_col_key(self, mc_result):
        assert "col" in mc_result

    def test_mc_result_spatial_shape(self, mc_result):
        ny = int(mc_result["row"].max()) + 1
        nx = int(mc_result["col"].max()) + 1
        assert (ny, nx) == (_CUBE_NY, _CUBE_NX)

    def test_mc_result_has_avg_g1_center(self, mc_result):
        """mc_best_fit uses 'avg_' prefix; the key must be 'avg_g1_center'."""
        assert "avg_g1_center" in mc_result

    def test_mc_result_avg_g1_center_finite(self, mc_result):
        """mc_iter(0) returns [original_fit], so every spaxel should be finite."""
        assert np.all(np.isfinite(mc_result["avg_g1_center"]))

    def test_mc_result_avg_g1_center_near_line(self, mc_result):
        """All avg_g1_center values must lie within 0.5 Å of the injected center."""
        centers = mc_result["avg_g1_center"]
        finite = centers[np.isfinite(centers)]
        assert finite.size > 0, "no finite avg_g1_center values — all mc fits failed"
        assert np.all(np.abs(finite - _LINE_CENTER) < 0.5)

    def test_mc_result_comment_is_str(self, mc_result):
        """loadtxt must reconstruct the .comment attribute as a string."""
        assert isinstance(mc_result.comment, str)


# ---------------------------------------------------------------------------
# 1.7f — parallel=True behaviour
# ---------------------------------------------------------------------------


class TestParallelRunNotSupported:
    """1.7f-i — parallel=True raises ValueError when the fork context is absent.

    This is the code path exercised on Windows today (fork is unavailable),
    reproduced here on all platforms via monkeypatching so the error path is
    covered in CI regardless of OS.
    """

    def test_parallel_raises_when_ctx_none(self, synthetic_cube, tmp_path, monkeypatch):
        """fit_line.run() must raise ValueError if parallel=True and ctx is None."""
        monkeypatch.setattr(fit_line, "ctx", None)

        s = SimpleNamespace(
            setup_parameters=False,
            monitor_pixels=[],
            baseline_subtract=None,
            baseline_fit_range=None,
            output_filename=str(tmp_path / "output"),
            save_plots=False,
            region_averaging_radius=1.5,
            instrument_dispersion=0.8,
            lmfit_kwargs={"method": "least_squares"},
            snr_lower_limit=3,
            lines=[tc.lines.L_OIII5007],
            models=[[tc.models.Const_1GaussModel()]],
            d_aic=-150,
            interactively_choose_fits=False,
            always_manually_choose=[],
            mc_snr=25,
            mc_n_iterations=0,
            parallel=True,
            n_process=2,
            chop_bandwidth=False,
            SNR_HalfBW=9,
            SNR_Baseline_q=0.15,
            cube=synthetic_cube[_SUBREGION_SLICE],
            continuum_cube=None,
            z_set=0,
            comment="",
            _i=0,
        )
        fit_lines.update_settings(s)

        with pytest.raises(ValueError, match="parallel"):
            fit_line.run(s)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="parallel=True requires the 'fork' start method, not available on Windows "
    "(§6.3 — replace multiprocessing with joblib — will fix this)",
)
class TestParallelRun:
    """1.7f-ii — parallel=True actually produces the same outputs as parallel=False.

    Skipped on Windows: fork context is not available there.  Once §6.3 is
    implemented (joblib backend), this skip can be removed.

    Uses ``Const_1GaussModel_fast`` (not ``Const_1GaussModel``) because the
    fast variant wraps a module-level numba function that is picklable, whereas
    ``Const_1GaussModel`` embeds a local lambda inside ``ConstantModel.__init__``
    that cannot cross the process boundary even with the ``fork`` start method
    (results must be pickled on return from the worker).
    """

    @pytest.fixture(scope="class")
    def parallel_run_state(self, synthetic_cube, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("fit_line_parallel")
        s = SimpleNamespace(
            setup_parameters=False,
            monitor_pixels=[],
            baseline_subtract=None,
            baseline_fit_range=None,
            output_filename=str(tmp_path / "output"),
            save_plots=False,
            region_averaging_radius=1.5,
            instrument_dispersion=0.8,
            lmfit_kwargs={"method": "least_squares"},
            snr_lower_limit=3,
            lines=[tc.lines.L_OIII5007],
            models=[[tc.models.Const_1GaussModel_fast()]],
            d_aic=-150,
            interactively_choose_fits=False,
            always_manually_choose=[],
            mc_snr=25,
            mc_n_iterations=0,
            parallel=True,
            n_process=2,
            chop_bandwidth=False,
            SNR_HalfBW=9,
            SNR_Baseline_q=0.15,
            cube=synthetic_cube[_SUBREGION_SLICE],
            continuum_cube=None,
            z_set=0,
            comment="",
            _i=0,
        )
        fit_lines.update_settings(s)
        fit_line.run(s)
        return s

    def test_parallel_run_completes(self, parallel_run_state):
        """fit_line.run() must return without error when parallel=True."""
        assert parallel_run_state is not None

    def test_parallel_model_results_shape(self, parallel_run_state):
        """model_results must have the expected spatial shape after a parallel run."""
        results = parallel_run_state.model_results
        assert results.shape == (1, 3, 3)

    def test_parallel_model_results_not_all_none(self, parallel_run_state):
        """At least one spaxel must have a successful fit."""
        flat = parallel_run_state.model_results.flat
        assert any(r is not None for r in flat)

    def test_parallel_g1_center_survives_pickling(self, parallel_run_state):
        """g1_center must be close to the injected line after the pool round-trip.

        ModelResult objects are pickled in the worker process and unpickled in
        the main process.  A corrupt or zeroed-out result would still satisfy the
        shape/not-None tests above but would fail here.  This is the primary
        content-correctness guard for the parallel path.
        """
        non_none = [r for r in parallel_run_state.model_results.flat if r is not None]
        assert len(non_none) > 0, "no successful fits in parallel run"
        centers = [r.params["g1_center"].value for r in non_none]
        assert all(abs(c - _LINE_CENTER) < 0.5 for c in centers), (
            f"parallel g1_center values out of range: {centers}"
        )
