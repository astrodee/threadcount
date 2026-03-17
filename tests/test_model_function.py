"""Tests for model functions and model classes (roadmap item 1.4).

Covers:
  - test_numba_accuracy: njit gaussian3CH_d matches pure-Python (fixed seeded RNG)
  - TestGaussianModelH
  - TestConst1GaussModel
  - TestConst1GaussModelFast
  - TestConst2GaussModel
  - TestConst2GaussModelFast
  - TestQuadratic1GaussModel
  - TestQuadratic2GaussModel
  - TestConst3GaussModel
  - TestQuadratic3GaussModel
  - TestConst3GaussModelFast
  - TestConst4GaussModelFast
  - TestConst4GaussModelConstrainedSIIFast
  - TestConst6GaussModelFast
  - TestConst6GaussModelConstrainedHaNIIFast
  - TestLog10DoubleExponentialModel
  - TestGuessMultiline2
  - TestGuessMultiline3
  - TestGuessMultiline2D
  - TestGuessMultiline3D
  - TestGuessMultiline4D
  - TestGuessMultiline6D
"""

import lmfit
import numpy as np
import pytest

import threadcount.models as tc_models
from threadcount.models.basic import (
    guess_from_peak,
    mean_edges,
    reapply_certain_model_hints,
)
from threadcount.models.fast_models import (
    _guess_multiline2_d,
    _guess_multiline3_d,
    _guess_multiline4_d,
    _guess_multiline6_d,
    gaussian1CH_d,
    gaussian2CH_d,
    gaussian3CH_d,
    gaussian4CH_constrained_SII_d_DELTAX24,
    gaussian6CH_constrained_HaNII_d_DELTAX24,
    gaussian6CH_constrained_HaNII_d_DELTAX64,
)
from threadcount.models.models import (
    _guess_multiline2,
    _guess_multiline3,
    gaussian2CH,
    gaussian3CH,
    gaussianH,
    log10_sum,
    set_common_limits,
)

tiny = lmfit.models.tiny

# Seed used for all randomised sub-tests.
RNG_SEED = 42

# ---------------------------------------------------------------------------
# Pure-Python reference implementation of gaussian3CH_d.
# Intentionally NOT imported from fast_models so the reference stays
# independent of any numba compilation state.
# ---------------------------------------------------------------------------


def _gaussian3CH_d_pure(
    x,
    g1_height=1.0,
    deltax=0.0,
    g1_sigma=1.0,
    g2_height=1.0,
    g2_center=0.0,
    g2_sigma=1.0,
    g3_height=1.0,
    deltaxhi=0.0,
    g3_sigma=1.0,
    c=0.0,
):
    """Pure-Python (no numba) reference matching fast_models.gaussian3CH_d."""
    f = (
        g1_height
        * np.exp(-((1.0 * x - g2_center - deltax) ** 2) / max(tiny, (2 * g1_sigma**2)))
        + g2_height
        * np.exp(-((1.0 * x - g2_center) ** 2) / max(tiny, (2 * g2_sigma**2)))
        + g3_height
        * np.exp(
            -((1.0 * x - g2_center - deltaxhi) ** 2) / max(tiny, (2 * g3_sigma**2))
        )
        + c
    )
    return f


def test_numba_accuracy():
    """fast_models.gaussian3CH_d (@njit) matches the pure-Python reference.

    Uses a seeded RNG so the test is fully reproducible (the original used
    an unseeded random.uniform which produced different values on every run).
    np.allclose is used instead of np.array_equal to allow for any sub-ULP
    floating-point differences that a JIT compiler might introduce.
    """
    rng = np.random.default_rng(RNG_SEED)
    x = np.linspace(6517, 6618, 80)
    kw = {
        "g1_height": rng.uniform(1.0, 6.0),
        "deltax": rng.uniform(-17.0, -13.0),
        "g1_sigma": rng.uniform(1.0, 3.0),
        "g2_height": rng.uniform(4.0, 7.0),
        "g2_center": rng.uniform(6558.0, 6570.0),
        "g2_sigma": rng.uniform(1.0, 3.0),
        "g3_height": rng.uniform(0.0, 4.0),
        "deltaxhi": rng.uniform(19.0, 23.0),
        "g3_sigma": rng.uniform(1.0, 3.0),
    }
    kw["c"] = rng.uniform(-1.0, 5.0)  # also exercise nonzero constant offset
    res_python = _gaussian3CH_d_pure(x, **kw)
    res_numba = gaussian3CH_d(x, **kw)
    assert np.allclose(res_python, res_numba, rtol=1e-12), (
        "numba @njit version disagrees with pure-Python reference"
    )


# ---------------------------------------------------------------------------
# Spectrum builders and assertion helper
# ---------------------------------------------------------------------------


def _gauss(x, height, center, sigma):
    return height * np.exp(-((x - center) ** 2) / (2 * sigma**2))


def _make_1g(x, height, center, sigma, c=0.0):
    return c + _gauss(x, height, center, sigma)


def _make_2g(x, h1, c1, s1, h2, c2, s2, c=0.0):
    return c + _gauss(x, h1, c1, s1) + _gauss(x, h2, c2, s2)


def _make_3g(x, h1, c1, s1, h2, c2, s2, h3, c3, s3, c=0.0):
    return c + _gauss(x, h1, c1, s1) + _gauss(x, h2, c2, s2) + _gauss(x, h3, c3, s3)


REL_TOL = 0.05  # 5 % default tolerance on recovered fit parameters


def _assert_close(fitted, expected, name, rel_tol=REL_TOL):
    rel_err = abs(fitted - expected) / max(abs(expected), 1e-10)
    assert rel_err < rel_tol, (
        f"{name}: fitted={fitted:.5g}, expected={expected:.5g}, rel_err={rel_err:.2%}"
    )


# ===========================================================================
# 1. GaussianModelH
# ===========================================================================


class TestGaussianModelH:
    """Single-Gaussian model parameterised by height (not amplitude)."""

    H, CEN, SIG = 10.0, 6563.0, 1.5

    @pytest.fixture
    def xy(self):
        x = np.linspace(6545.0, 6585.0, 120)
        return x, _make_1g(x, self.H, self.CEN, self.SIG)

    def test_param_names(self):
        model = tc_models.GaussianModelH()
        assert {"height", "center", "sigma"}.issubset(model.param_names)

    def test_fwhm_and_flux_are_constrained(self):
        """fwhm and flux must be derived expressions, not free parameters."""
        model = tc_models.GaussianModelH()
        pars = model.make_params(height=1.0, center=0.0, sigma=1.0)
        assert pars["fwhm"].expr is not None
        assert pars["flux"].expr is not None

    def test_height_and_sigma_min_zero(self):
        """Default param hints enforce height >= 0 and sigma >= 0."""
        model = tc_models.GaussianModelH()
        pars = model.make_params()
        assert pars["height"].min >= 0.0
        assert pars["sigma"].min >= 0.0

    def test_eval_correct(self):
        """Model.eval() with known params reproduces the injected spectrum exactly."""
        x = np.linspace(6545.0, 6585.0, 120)
        model = tc_models.GaussianModelH()
        pars = model.make_params(height=self.H, center=self.CEN, sigma=self.SIG)
        assert np.allclose(
            model.eval(pars, x=x), _make_1g(x, self.H, self.CEN, self.SIG), rtol=1e-12
        )

    def test_fit_recovers_parameters(self, xy):
        """Fit from auto-guess should recover injected parameters to 5 %."""
        x, y = xy
        model = tc_models.GaussianModelH()
        result = model.fit(y, model.guess(y, x=x), x=x)
        assert result.redchi < 1e-4, (
            f"redchi={result.redchi:.3g}: fit did not converge on noiseless data"
        )
        _assert_close(result.params["height"].value, self.H, "height")
        _assert_close(result.params["center"].value, self.CEN, "center")
        _assert_close(result.params["sigma"].value, self.SIG, "sigma")

    def test_fit_fwhm_consistent(self, xy):
        """fwhm expression must be proportional to sigma at the fit optimum.

        NOTE — known precision limitation: lmfit's fwhm_expr formats the
        factor as ``:.7f`` → ``2.3548200``, truncating the exact value
        2.3548200450309493.  The resulting systematic error in fwhm is
        ~4.5e-8 * sigma per Å, so for typical sigma ~1-4 Å the fwhm
        value is wrong by up to ~2e-7 Å.  The tolerance below documents
        this behaviour rather than asserting exact equality (see ROADMAP §2.12).
        """
        x, y = xy
        model = tc_models.GaussianModelH()
        result = model.fit(y, model.guess(y, x=x), x=x)
        fwhm_expected = 2 * np.sqrt(2 * np.log(2)) * result.params["sigma"].value
        assert abs(result.params["fwhm"].value - fwhm_expected) < 1e-6

    def test_flux_numerical_value(self, xy):
        """Fitted flux must equal sqrt(2*pi)*height*sigma to within the known
        truncation error in flux_expr's :.7f format (see ROADMAP \u00a72.12)."""
        x, y = xy
        model = tc_models.GaussianModelH()
        result = model.fit(y, model.guess(y, x=x), x=x)
        h = result.params["height"].value
        s = result.params["sigma"].value
        flux_exact = np.sqrt(2 * np.pi) * h * s
        # Allow 1e-5 relative tolerance to cover the :.7f truncation issue
        assert abs(result.params["flux"].value - flux_exact) < 1e-5 * flux_exact

    def test_guess_negative_profile(self):
        """GaussianModelH.guess(negative=True) is overridden by the min=0 param hint.

        guess_from_peak() correctly returns a negative height, but
        reapply_certain_model_hints() then enforces the min=0 constraint, clamping
        the value to 0.  This is a known limitation: GaussianModelH does not support
        absorption-line fitting.  The test documents the actual (clamped) behaviour.
        """
        x = np.linspace(6545.0, 6585.0, 120)
        y = -_make_1g(x, self.H, self.CEN, self.SIG)  # inverted peak
        model = tc_models.GaussianModelH()
        pars = model.guess(y, x=x, negative=True)
        # The min=0 hint overrides negative=True — height is clamped to 0
        assert pars["height"].min == 0.0, "height should have min=0 from param hint"
        assert pars["height"].value == 0.0, (
            f"Expected height clamped to 0 (min=0 hint overrides negative=True), "
            f"got {pars['height'].value}"
        )

    def test_prefix_propagation(self):
        """GaussianModelH(prefix='ha_') should prefix all parameter names.

        Unlike the composite model classes (Const_1GaussModel, etc.) that silently
        drop the prefix argument (see ROADMAP §2.13), GaussianModelH inherits
        directly from lmfit.Model and must propagate the prefix correctly.

        Notes
        -----
        ``model.param_names`` only contains the function-signature parameters
        (height, center, sigma).  The constrained expressions fwhm and flux are
        added via ``set_param_hint`` and therefore appear only in the
        ``make_params()`` output, not in ``param_names`` directly.
        """
        prefix = "ha_"
        model_prefixed = tc_models.GaussianModelH(prefix=prefix)
        model_plain = tc_models.GaussianModelH()

        # Function-signature params in param_names must all carry the prefix
        core_names = {"height", "center", "sigma"}
        for name in core_names:
            assert f"{prefix}{name}" in model_prefixed.param_names, (
                f"'{prefix}{name}' not found in param_names: {model_prefixed.param_names}"
            )
            assert name not in model_prefixed.param_names, (
                f"Un-prefixed '{name}' found in prefixed model's param_names"
            )

        # make_params() must include the constrained fwhm and flux under prefixed names
        x = np.linspace(6545.0, 6585.0, 120)
        pars_prefixed = model_prefixed.make_params(
            **{
                f"{prefix}height": self.H,
                f"{prefix}center": self.CEN,
                f"{prefix}sigma": self.SIG,
            }
        )
        for name in ("height", "center", "sigma", "fwhm", "flux"):
            assert f"{prefix}{name}" in pars_prefixed, (
                f"'{prefix}{name}' missing from make_params() output"
            )
            assert name not in pars_prefixed, (
                f"Un-prefixed '{name}' found in make_params() output of prefixed model"
            )

        # fwhm and flux expressions must reference the prefixed sigma/height
        assert f"{prefix}sigma" in pars_prefixed[f"{prefix}fwhm"].expr, (
            f"fwhm expr '{pars_prefixed[f'{prefix}fwhm'].expr}' does not reference {prefix}sigma"
        )
        assert f"{prefix}height" in pars_prefixed[f"{prefix}flux"].expr, (
            f"flux expr '{pars_prefixed[f'{prefix}flux'].expr}' does not reference {prefix}height"
        )

        # eval() must produce the same values as the un-prefixed model for
        # identical physical parameters
        pars_plain = model_plain.make_params(
            height=self.H, center=self.CEN, sigma=self.SIG
        )
        np.testing.assert_allclose(
            model_prefixed.eval(pars_prefixed, x=x),
            model_plain.eval(pars_plain, x=x),
            rtol=1e-12,
            err_msg="Prefixed and un-prefixed GaussianModelH must evaluate identically",
        )


# ===========================================================================
# 2. Const_1GaussModel
# ===========================================================================


class TestConst1GaussModel:
    """Constant + 1 Gaussian composite model."""

    H, CEN, SIG, C = 15.0, 5007.0, 1.2, 3.0

    @pytest.fixture
    def xy(self):
        x = np.linspace(4990.0, 5025.0, 140)
        return x, _make_1g(x, self.H, self.CEN, self.SIG, c=self.C)

    def test_param_names(self):
        model = tc_models.Const_1GaussModel()
        for p in ("g1_height", "g1_center", "g1_sigma", "c"):
            assert p in model.param_names

    def test_fwhm_flux_constrained(self):
        """g1_fwhm and g1_flux must be constrained expressions (inherited from
        GaussianModelH via composite model prefix propagation)."""
        model = tc_models.Const_1GaussModel()
        pars = model.make_params()
        assert pars["g1_fwhm"].expr is not None, (
            "g1_fwhm should be a constrained expression"
        )
        assert pars["g1_flux"].expr is not None, (
            "g1_flux should be a constrained expression"
        )

    def test_eval_correct(self, xy):
        x, y = xy
        model = tc_models.Const_1GaussModel()
        pars = model.make_params(
            g1_height=self.H, g1_center=self.CEN, g1_sigma=self.SIG, c=self.C
        )
        assert np.allclose(model.eval(pars, x=x), y, rtol=1e-12)

    def test_fit_recovers_parameters(self, xy):
        """Fit from auto-guess should recover all four parameters to 5 %."""
        x, y = xy
        model = tc_models.Const_1GaussModel()
        result = model.fit(y, model.guess(y, x=x), x=x)
        assert result.redchi < 1e-4, (
            f"redchi={result.redchi:.3g}: fit did not converge on noiseless data"
        )
        _assert_close(result.params["g1_height"].value, self.H, "g1_height")
        _assert_close(result.params["g1_center"].value, self.CEN, "g1_center")
        _assert_close(result.params["g1_sigma"].value, self.SIG, "g1_sigma")
        _assert_close(result.params["c"].value, self.C, "c")

    def test_fit_noisy_converges(self, xy):
        """Fit should recover parameters to 10 % with mild Gaussian noise."""
        x, y = xy
        rng = np.random.default_rng(RNG_SEED)
        y_noisy = y + rng.normal(0, 0.5, y.shape)
        model = tc_models.Const_1GaussModel()
        result = model.fit(y_noisy, model.guess(y_noisy, x=x), x=x)
        _assert_close(
            result.params["g1_height"].value, self.H, "g1_height", rel_tol=0.10
        )
        _assert_close(
            result.params["g1_center"].value, self.CEN, "g1_center", rel_tol=0.10
        )


# ===========================================================================
# 3. Const_1GaussModel_fast
# ===========================================================================


class TestConst1GaussModelFast:
    """Fast (numba @njit) version of Const_1GaussModel."""

    H, CEN, SIG, C = 15.0, 5007.0, 1.2, 3.0

    @pytest.fixture
    def xy(self):
        x = np.linspace(4990.0, 5025.0, 140)
        return x, _make_1g(x, self.H, self.CEN, self.SIG, c=self.C)

    def test_param_names(self):
        model = tc_models.Const_1GaussModel_fast()
        for p in ("g1_height", "g1_center", "g1_sigma", "c"):
            assert p in model.param_names

    def test_fwhm_flux_constrained(self):
        """Fast model must expose g1_fwhm and g1_flux as constrained derived params."""
        model = tc_models.Const_1GaussModel_fast()
        pars = model.make_params()
        assert pars["g1_fwhm"].expr is not None
        assert pars["g1_flux"].expr is not None

    def test_eval_correct(self, xy):
        x, y = xy
        model = tc_models.Const_1GaussModel_fast()
        pars = model.make_params(
            g1_height=self.H, g1_center=self.CEN, g1_sigma=self.SIG, c=self.C
        )
        assert np.allclose(model.eval(pars, x=x), y, rtol=1e-12)

    def test_fit_recovers_parameters(self, xy):
        """Fit from auto-guess should recover all four parameters to 5 %."""
        x, y = xy
        model = tc_models.Const_1GaussModel_fast()
        result = model.fit(y, model.guess(y, x=x), x=x)
        assert result.redchi < 1e-4, (
            f"redchi={result.redchi:.3g}: fit did not converge on noiseless data"
        )
        _assert_close(result.params["g1_height"].value, self.H, "g1_height")
        _assert_close(result.params["g1_center"].value, self.CEN, "g1_center")
        _assert_close(result.params["g1_sigma"].value, self.SIG, "g1_sigma")
        _assert_close(result.params["c"].value, self.C, "c")

    def test_guess_produces_finite_params(self, xy):
        """model.guess() must not produce NaN or infinite free-parameter values."""
        x, y = xy
        model = tc_models.Const_1GaussModel_fast()
        pars = model.guess(y, x=x)
        for name, par in pars.items():
            if par.expr is None and par.vary:
                assert np.isfinite(par.value), f"Non-finite guess for '{name}'"

    def test_parity_with_standard(self, xy):
        """Fast and standard Const_1GaussModel must agree to 0.1 % on all params."""
        x, y = xy
        m_std = tc_models.Const_1GaussModel()
        m_fast = tc_models.Const_1GaussModel_fast()
        r_std = m_std.fit(y, m_std.guess(y, x=x), x=x)
        r_fast = m_fast.fit(y, m_fast.guess(y, x=x), x=x)
        for par in ("g1_height", "g1_center", "g1_sigma", "c"):
            v_std = r_std.params[par].value
            v_fast = r_fast.params[par].value
            diff = abs(v_std - v_fast) / max(abs(v_std), 1e-10)
            assert diff < 0.001, (
                f"{par}: standard={v_std:.6g}, fast={v_fast:.6g}, rel_diff={diff:.2%}"
            )


# ===========================================================================
# 4. Const_2GaussModel
# ===========================================================================


class TestConst2GaussModel:
    """Constant + 2 Gaussians composite model.

    Test scenario: narrow + broad component sharing the same centre, which is
    the typical galaxy-outflow use-case this model was designed for.
    The default _guess_2gauss places g1 at center - 2*sigma0 and g2 at center,
    so it is most accurate when both components overlap strongly.
    """

    # Dominant narrow component + fainter broad one, same centre.
    H_NARROW, SIG_NARROW = 20.0, 1.2
    H_BROAD, SIG_BROAD = 6.0, 4.0
    CEN, C = 6563.0, 1.5

    @pytest.fixture
    def xy(self):
        x = np.linspace(6535.0, 6595.0, 200)
        y = (
            self.C
            + _gauss(x, self.H_NARROW, self.CEN, self.SIG_NARROW)
            + _gauss(x, self.H_BROAD, self.CEN, self.SIG_BROAD)
        )
        return x, y

    def test_param_names(self):
        model = tc_models.Const_2GaussModel()
        for p in (
            "g1_height",
            "g1_center",
            "g1_sigma",
            "g2_height",
            "g2_center",
            "g2_sigma",
            "c",
        ):
            assert p in model.param_names

    def test_fwhm_flux_constrained_both_components(self):
        """Both g1 and g2 must expose fwhm and flux as constrained expressions
        (inherited from GaussianModelH via prefix propagation)."""
        model = tc_models.Const_2GaussModel()
        pars = model.make_params()
        for comp in ("g1_", "g2_"):
            assert pars[comp + "fwhm"].expr is not None, (
                f"{comp}fwhm should be constrained"
            )
            assert pars[comp + "flux"].expr is not None, (
                f"{comp}flux should be constrained"
            )

    def test_eval_correct(self, xy):
        x, y = xy
        model = tc_models.Const_2GaussModel()
        # Assign broad to g1, narrow to g2 (arbitrary; model has no ordering)
        pars = model.make_params(
            g1_height=self.H_BROAD,
            g1_center=self.CEN,
            g1_sigma=self.SIG_BROAD,
            g2_height=self.H_NARROW,
            g2_center=self.CEN,
            g2_sigma=self.SIG_NARROW,
            c=self.C,
        )
        assert np.allclose(model.eval(pars, x=x), y, rtol=1e-12)

    def test_fit_recovers_parameters(self, xy):
        """Fit from near-truth initial params should recover components to 5 %.

        Components are identified by sorting fitted sigmas — whichever of g1/g2
        has the smaller sigma is the narrow component, and vice-versa.  This is
        more robust than relying on a fixed label assignment.
        """
        x, y = xy
        model = tc_models.Const_2GaussModel()
        # Start with g1=broad, g2=narrow — a deliberate assignment close to truth
        pars = model.make_params(
            g1_height=self.H_BROAD * 0.9,
            g1_center=self.CEN,
            g1_sigma=self.SIG_BROAD * 1.1,
            g2_height=self.H_NARROW * 1.1,
            g2_center=self.CEN,
            g2_sigma=self.SIG_NARROW * 0.9,
            c=self.C * 1.1,
        )
        result = model.fit(y, pars, x=x)
        assert result.redchi < 1e-4, (
            f"redchi={result.redchi:.3g}: fit did not converge on noiseless data"
        )
        # Identify components by sigma magnitude — avoids fragile label-order assumptions
        s1 = result.params["g1_sigma"].value
        s2 = result.params["g2_sigma"].value
        narrow, broad = ("g1", "g2") if s1 < s2 else ("g2", "g1")
        # The two sigmas must be near the two injected values (order-agnostic)
        fitted_sigmas = sorted([s1, s2])
        injected_sigmas = sorted([self.SIG_NARROW, self.SIG_BROAD])
        for fs, es in zip(fitted_sigmas, injected_sigmas):
            _assert_close(fs, es, f"sigma match ({fs:.3f} vs {es:.3f})")
        _assert_close(
            result.params[f"{narrow}_height"].value, self.H_NARROW, f"{narrow}_height"
        )
        _assert_close(
            result.params[f"{broad}_height"].value, self.H_BROAD, f"{broad}_height"
        )
        _assert_close(result.params["c"].value, self.C, "c")

    def test_guess_produces_finite_params(self, xy):
        """model.guess() must not produce NaN or infinite free-parameter values."""
        x, y = xy
        model = tc_models.Const_2GaussModel()
        pars = model.guess(y, x=x)
        for name, par in pars.items():
            if par.expr is None and par.vary:
                assert np.isfinite(par.value), f"Non-finite guess for '{name}'"

    def test_fit_from_auto_guess(self):
        """Auto-guess convergence test using data that matches _guess_2gauss defaults.

        _guess_2gauss with default args places g1 at center - 2*sigma0, g2 at
        center, with height ratio 1:4.  We construct noiseless data that satisfies
        these assumptions so the starting point is close to truth and the optimizer
        can be expected to converge.

        Components are identified by sorting fitted centers from low to high
        wavelength, making the test robust to label-order ambiguity.
        """
        SIG = 1.5
        CEN = 6563.0
        H1 = 5.0  # component at CEN - 2*SIG
        H2 = 20.0  # dominant component at CEN
        C = 1.0
        x = np.linspace(6535.0, 6595.0, 200)
        y = C + _gauss(x, H1, CEN - 2 * SIG, SIG) + _gauss(x, H2, CEN, SIG)
        model = tc_models.Const_2GaussModel()
        result = model.fit(y, model.guess(y, x=x), x=x)
        assert result.redchi < 1e-3, (
            f"Auto-guess fit did not converge: redchi={result.redchi:.3g}"
        )
        # Sort components by fitted center (low → high) for label-agnostic comparison
        comps = sorted(
            [
                (
                    result.params[f"g{i}_center"].value,
                    result.params[f"g{i}_height"].value,
                    result.params[f"g{i}_sigma"].value,
                )
                for i in (1, 2)
            ]
        )
        expected = sorted(
            [
                (CEN - 2 * SIG, H1, SIG),
                (CEN, H2, SIG),
            ]
        )
        for (cf, hf, sf), (ce, he, se), label in zip(comps, expected, ("low", "high")):
            _assert_close(cf, ce, f"{label} center")
            _assert_close(hf, he, f"{label} height")
            _assert_close(sf, se, f"{label} sigma")
        _assert_close(result.params["c"].value, C, "c")

    def test_guess_absolute_centers(self, xy):
        """_guess_2gauss with absolute_centers=True places components at
        center + offsets, not center + sigma0 * offsets."""
        x, y = xy
        model = tc_models.Const_2GaussModel()
        OFFSET1, OFFSET2 = -5.0, 3.0
        pars = model.guess(
            y,
            x=x,
            absolute_centers=True,
            centers=(OFFSET1, OFFSET2),
        )
        # guess_from_peak returns a center near the peak
        _, guessed_center, _ = guess_from_peak(y, x)
        assert abs(pars["g1_center"].value - (guessed_center + OFFSET1)) < 0.2, (
            f"g1_center={pars['g1_center'].value:.3f} expected near "
            f"{guessed_center + OFFSET1:.3f} (absolute_centers=True)"
        )
        assert abs(pars["g2_center"].value - (guessed_center + OFFSET2)) < 0.2, (
            f"g2_center={pars['g2_center'].value:.3f} expected near "
            f"{guessed_center + OFFSET2:.3f} (absolute_centers=True)"
        )


# ===========================================================================
# 5. Const_2GaussModel_fast
# ===========================================================================


class TestConst2GaussModelFast:
    """Fast (numba @njit) version of Const_2GaussModel.

    Key parameterisation difference from the standard model:
      ``deltax`` (free)  = g1_center − g2_center
      ``g1_center``      = constrained expression ``g2_center + deltax``
    """

    H_NARROW, SIG_NARROW = 20.0, 1.2
    H_BROAD, SIG_BROAD = 6.0, 4.0
    CEN, C = 6563.0, 1.5

    @pytest.fixture
    def xy(self):
        x = np.linspace(6535.0, 6595.0, 200)
        y = (
            self.C
            + _gauss(x, self.H_NARROW, self.CEN, self.SIG_NARROW)
            + _gauss(x, self.H_BROAD, self.CEN, self.SIG_BROAD)
        )
        return x, y

    def test_param_names(self):
        """deltax must be a free parameter; g1_center should NOT be free."""
        model = tc_models.Const_2GaussModel_fast()
        for p in (
            "g1_height",
            "deltax",
            "g1_sigma",
            "g2_height",
            "g2_center",
            "g2_sigma",
            "c",
        ):
            assert p in model.param_names

    def test_g1_center_is_constrained(self):
        """g1_center must be a derived expression involving g2_center and deltax."""
        model = tc_models.Const_2GaussModel_fast()
        pars = model.make_params()
        assert pars["g1_center"].expr is not None
        assert "g2_center" in pars["g1_center"].expr
        assert "deltax" in pars["g1_center"].expr

    def test_fwhm_flux_constrained_both_components(self):
        """Both g1 and g2 must expose fwhm and flux as constrained derived params."""
        model = tc_models.Const_2GaussModel_fast()
        pars = model.make_params()
        for comp in ("g1_", "g2_"):
            assert pars[comp + "fwhm"].expr is not None
            assert pars[comp + "flux"].expr is not None

    def test_eval_correct(self, xy):
        """eval() with deltax=0 places both components at g2_center."""
        x, y = xy
        model = tc_models.Const_2GaussModel_fast()
        # broad -> g1, narrow -> g2, both at CEN via deltax=0
        pars = model.make_params(
            g1_height=self.H_BROAD,
            deltax=0.0,
            g1_sigma=self.SIG_BROAD,
            g2_height=self.H_NARROW,
            g2_center=self.CEN,
            g2_sigma=self.SIG_NARROW,
            c=self.C,
        )
        assert np.allclose(model.eval(pars, x=x), y, rtol=1e-12)

    def test_fit_recovers_parameters(self, xy):
        """Fit from near-truth initial params should recover components to 5 %."""
        x, y = xy
        model = tc_models.Const_2GaussModel_fast()
        pars = model.make_params(
            g1_height=self.H_BROAD * 0.9,
            deltax=0.0,
            g1_sigma=self.SIG_BROAD * 1.1,
            g2_height=self.H_NARROW * 1.1,
            g2_center=self.CEN,
            g2_sigma=self.SIG_NARROW * 0.9,
            c=self.C * 1.1,
        )
        result = model.fit(y, pars, x=x)
        assert result.redchi < 1e-4, (
            f"redchi={result.redchi:.3g}: fit did not converge on noiseless data"
        )
        # g2 is the dominant (narrow) component; g1 is the broad wing
        _assert_close(result.params["g2_height"].value, self.H_NARROW, "g2_height")
        _assert_close(result.params["g2_center"].value, self.CEN, "g2_center")
        _assert_close(result.params["g1_sigma"].value, self.SIG_BROAD, "g1_sigma")
        _assert_close(result.params["c"].value, self.C, "c")
        # With both components at the same centre deltax should remain near 0
        assert abs(result.params["deltax"].value) < 0.5, (
            f"deltax={result.params['deltax'].value:.4f} should be near 0"
        )

    def test_guess_produces_finite_params(self, xy):
        """model.guess() must not produce NaN or infinite free-parameter values."""
        x, y = xy
        model = tc_models.Const_2GaussModel_fast()
        pars = model.guess(y, x=x)
        for name, par in pars.items():
            if par.expr is None and par.vary:
                assert np.isfinite(par.value), f"Non-finite guess for '{name}'"

    def test_eval_nonzero_deltax(self):
        """eval() with nonzero deltax places g1 at g2_center + deltax."""
        # Two components with genuinely different centres
        DELTAX = -10.0  # g1 is 10 Å blueward of g2
        x = np.linspace(6530.0, 6600.0, 200)
        model = tc_models.Const_2GaussModel_fast()
        pars = model.make_params(
            g1_height=8.0,
            deltax=DELTAX,
            g1_sigma=1.5,
            g2_height=15.0,
            g2_center=6563.0,
            g2_sigma=1.5,
            c=0.0,
        )
        y = model.eval(pars, x=x)
        # Manually verify: peak near g2_center+deltax and near g2_center
        # The combined peak maximum should be between the two centres
        assert y.max() > 0
        # Evaluate again with deltax=0; the two spectra must differ
        pars_zero = pars.copy()
        pars_zero["deltax"].set(value=0.0)
        y_zero = model.eval(pars_zero, x=x)
        assert not np.allclose(y, y_zero), "Changing deltax had no effect on eval"

    def test_fit_from_auto_guess(self):
        """Auto-guess convergence for Const_2GaussModel_fast using data that
        matches _guess_2gauss_d defaults.

        _guess_2gauss_d with defaults places g1 at center - 2*sigma0 (i.e.
        deltax = -2*sigma0) and g2 at center, heights ratio 1:4.  We construct
        noiseless data matching those assumptions to verify convergence.

        The fast model has unambiguous labels: g2_center is the free centre
        parameter and deltax = g1_center - g2_center, so no sorting is needed.
        """
        SIG = 1.5
        CEN = 6563.0
        DELTAX = -2.0 * SIG  # = -3.0 Angstrom
        H1 = 5.0  # g1 component at CEN + DELTAX
        H2 = 20.0  # g2 dominant component at CEN
        C = 1.0
        x = np.linspace(6535.0, 6595.0, 200)
        y = C + _gauss(x, H1, CEN + DELTAX, SIG) + _gauss(x, H2, CEN, SIG)
        model = tc_models.Const_2GaussModel_fast()
        result = model.fit(y, model.guess(y, x=x), x=x)
        assert result.redchi < 1e-3, (
            f"Auto-guess fit did not converge: redchi={result.redchi:.3g}"
        )
        _assert_close(result.params["g2_center"].value, CEN, "g2_center", rel_tol=0.05)
        _assert_close(result.params["deltax"].value, DELTAX, "deltax", rel_tol=0.10)
        _assert_close(result.params["g1_height"].value, H1, "g1_height", rel_tol=0.10)
        _assert_close(result.params["g2_height"].value, H2, "g2_height", rel_tol=0.10)
        _assert_close(result.params["g1_sigma"].value, SIG, "g1_sigma", rel_tol=0.10)
        _assert_close(result.params["g2_sigma"].value, SIG, "g2_sigma", rel_tol=0.10)
        _assert_close(result.params["c"].value, C, "c", rel_tol=0.10)

    def test_fit_with_offset_components(self):
        """Fit two well-separated Gaussians with nonzero deltax injection.
        This exercises the core use-case where g1 and g2 have different centres."""
        DELTAX = -10.0
        CEN2 = 6563.0
        x = np.linspace(6530.0, 6600.0, 300)
        y = _make_2g(x, 8.0, CEN2 + DELTAX, 1.5, 15.0, CEN2, 1.5)
        model = tc_models.Const_2GaussModel_fast()
        pars = model.make_params(
            g1_height=8.0 * 0.9,
            deltax=DELTAX * 1.1,
            g1_sigma=1.5,
            g2_height=15.0 * 1.1,
            g2_center=CEN2,
            g2_sigma=1.5,
            c=0.0,
        )
        result = model.fit(y, pars, x=x)
        assert result.redchi < 1e-4, f"redchi={result.redchi:.3g}"
        _assert_close(result.params["deltax"].value, DELTAX, "deltax")
        _assert_close(result.params["g2_center"].value, CEN2, "g2_center")

    def test_parity_with_standard(self, xy):
        """Fast and standard Const_2GaussModel must agree to 0.1 % on physical params.

        The two models use different parameterisations (free centers vs deltax),
        so we compare the physical quantities --- heights, centers, sigmas, c ---
        by sorting both results by fitted sigma to avoid label-order ambiguity.
        """
        x, y = xy
        m_std = tc_models.Const_2GaussModel()
        m_fast = tc_models.Const_2GaussModel_fast()
        # Standard model: start from near-truth initial params
        pars_std = m_std.make_params(
            g1_height=self.H_BROAD * 0.9,
            g1_center=self.CEN,
            g1_sigma=self.SIG_BROAD * 1.1,
            g2_height=self.H_NARROW * 1.1,
            g2_center=self.CEN,
            g2_sigma=self.SIG_NARROW * 0.9,
            c=self.C * 1.1,
        )
        r_std = m_std.fit(y, pars_std, x=x)
        # Fast model: equivalent near-truth start
        pars_fast = m_fast.make_params(
            g1_height=self.H_BROAD * 0.9,
            deltax=0.0,
            g1_sigma=self.SIG_BROAD * 1.1,
            g2_height=self.H_NARROW * 1.1,
            g2_center=self.CEN,
            g2_sigma=self.SIG_NARROW * 0.9,
            c=self.C * 1.1,
        )
        r_fast = m_fast.fit(y, pars_fast, x=x)
        assert r_std.redchi < 1e-4, (
            f"Standard model did not converge: redchi={r_std.redchi:.3g}"
        )
        assert r_fast.redchi < 1e-4, (
            f"Fast model did not converge: redchi={r_fast.redchi:.3g}"
        )

        # Sort both sets of components by sigma (narrow first) to align labels
        def _components(r):
            comps = sorted(
                [
                    (
                        r.params[f"g{i}_sigma"].value,
                        r.params[f"g{i}_height"].value,
                        r.params[f"g{i}_center"].value,
                    )
                    for i in (1, 2)
                ]
            )
            return comps

        std_comps = _components(r_std)
        fast_comps = _components(r_fast)
        for (sig_s, h_s, cen_s), (sig_f, h_f, cen_f), label in zip(
            std_comps, fast_comps, ("narrow", "broad")
        ):
            diff_sig = abs(sig_s - sig_f) / max(abs(sig_s), 1e-10)
            diff_h = abs(h_s - h_f) / max(abs(h_s), 1e-10)
            diff_cen = abs(cen_s - cen_f) / max(abs(cen_s), 1e-10)
            assert diff_sig < 0.001, (
                f"{label} sigma: std={sig_s:.6g}, fast={sig_f:.6g}, rel_diff={diff_sig:.2%}"
            )
            assert diff_h < 0.001, (
                f"{label} height: std={h_s:.6g}, fast={h_f:.6g}, rel_diff={diff_h:.2%}"
            )
            assert diff_cen < 0.001, (
                f"{label} center: std={cen_s:.6g}, fast={cen_f:.6g}, rel_diff={diff_cen:.2%}"
            )
        diff_c = abs(r_std.params["c"].value - r_fast.params["c"].value) / max(
            abs(r_std.params["c"].value), 1e-10
        )
        assert diff_c < 0.001, (
            f"c: std={r_std.params['c'].value:.6g}, fast={r_fast.params['c'].value:.6g}, "
            f"rel_diff={diff_c:.2%}"
        )


# ===========================================================================
# 6. Quadratic_1GaussModel
# ===========================================================================


class TestQuadratic1GaussModel:
    """Quadratic (a·x²+b·x+c) + 1 Gaussian composite model.

    _guess_1gauss estimates g1 params and c.  The quadratic coefficients a and
    b are not touched by the guess function and remain at the model default (0).
    """

    H, CEN, SIG, C = 12.0, 5007.0, 1.3, 3.0

    @pytest.fixture
    def xy(self):
        x = np.linspace(4990.0, 5025.0, 140)
        return x, _make_1g(x, self.H, self.CEN, self.SIG, c=self.C)

    def test_param_names(self):
        model = tc_models.Quadratic_1GaussModel()
        for p in ("g1_height", "g1_center", "g1_sigma", "a", "b", "c"):
            assert p in model.param_names

    def test_fwhm_flux_constrained(self):
        """g1_fwhm and g1_flux must be derived expressions."""
        model = tc_models.Quadratic_1GaussModel()
        pars = model.make_params()
        assert pars["g1_fwhm"].expr is not None
        assert pars["g1_flux"].expr is not None

    def test_eval_correct(self, xy):
        """eval() with a=b=0 and known params reproduces the injected spectrum."""
        x, y = xy
        model = tc_models.Quadratic_1GaussModel()
        pars = model.make_params(
            g1_height=self.H,
            g1_center=self.CEN,
            g1_sigma=self.SIG,
            a=0.0,
            b=0.0,
            c=self.C,
        )
        assert np.allclose(model.eval(pars, x=x), y, rtol=1e-12)

    def test_eval_quadratic_baseline_additive(self):
        """Non-zero a and b must shift the output by the expected quadratic amount."""
        x = np.linspace(4990.0, 5025.0, 140)
        A, B = 1e-4, -1.0
        model = tc_models.Quadratic_1GaussModel()
        pars_flat = model.make_params(
            g1_height=self.H,
            g1_center=self.CEN,
            g1_sigma=self.SIG,
            a=0.0,
            b=0.0,
            c=self.C,
        )
        pars_quad = pars_flat.copy()
        pars_quad["a"].set(value=A)
        pars_quad["b"].set(value=B)
        diff = model.eval(pars_quad, x=x) - model.eval(pars_flat, x=x)
        assert np.allclose(diff, A * x**2 + B * x, rtol=1e-12)

    def test_fit_recovers_parameters(self, xy):
        """Fit from near-truth initial params should recover Gaussian params to 5 %."""
        x, y = xy
        model = tc_models.Quadratic_1GaussModel()
        pars = model.make_params(
            g1_height=self.H * 0.9,
            g1_center=self.CEN,
            g1_sigma=self.SIG * 1.1,
            a=0.0,
            b=0.0,
            c=self.C * 1.1,
        )
        result = model.fit(y, pars, x=x)
        assert result.redchi < 1e-4, f"redchi={result.redchi:.3g}"
        _assert_close(result.params["g1_height"].value, self.H, "g1_height")
        _assert_close(result.params["g1_center"].value, self.CEN, "g1_center")
        _assert_close(result.params["g1_sigma"].value, self.SIG, "g1_sigma")

    def test_guess_produces_finite_params(self, xy):
        """model.guess() must not produce NaN or infinite free-parameter values."""
        x, y = xy
        model = tc_models.Quadratic_1GaussModel()
        pars = model.guess(y, x=x)
        for name, par in pars.items():
            if par.expr is None and par.vary:
                assert np.isfinite(par.value), f"Non-finite guess for '{name}'"

    def test_guess_leaves_a_b_at_default(self, xy):
        """_guess_1gauss does not update a or b; they stay at the model default value."""
        x, y = xy
        model = tc_models.Quadratic_1GaussModel()
        default_pars = model.make_params()
        guessed_pars = model.guess(y, x=x)
        assert guessed_pars["a"].value == pytest.approx(default_pars["a"].value)
        assert guessed_pars["b"].value == pytest.approx(default_pars["b"].value)

    def test_fit_recovers_quadratic_baseline(self):
        """Fit must recover non-zero a and b from a combined arch+tilt baseline.

        Truth: a_arch*(x-x_mid)^2 + slope*(x-x_mid)  (concave-down arch, ~25%
        of peak height, plus a gentle positive tilt ~12.5% of peak edge-to-edge).
        Fit starts with a=b=0 and c=C — the optimizer must discover the curvature
        and tilt from the residuals alone.  a and b must recover to within 10%.
        """
        x = np.linspace(4990.0, 5025.0, 140)
        x_mid = 0.5 * (x[0] + x[-1])
        half_w = 0.5 * (x[-1] - x[0])
        arch_amp = self.H / 4  # 3.0  — 25% of peak
        tilt_amp = self.H / 8  # 1.5  — 12.5% of peak, edge-to-edge
        a_truth = -arch_amp / half_w**2
        slope = tilt_amp / (x[-1] - x[0])
        b_truth = -2 * a_truth * x_mid + slope
        quad = a_truth * (x - x_mid) ** 2 + slope * (x - x_mid)
        y = _make_1g(x, self.H, self.CEN, self.SIG, c=self.C) + quad
        model = tc_models.Quadratic_1GaussModel()
        pars = model.make_params(
            g1_height=self.H,
            g1_center=self.CEN,
            g1_sigma=self.SIG,
            a=0.0,
            b=0.0,
            c=self.C,
        )
        result = model.fit(y, pars, x=x, method="least_squares")
        assert result.redchi < 1e-4, f"redchi={result.redchi:.3g}"
        _assert_close(result.params["a"].value, a_truth, "a", rel_tol=0.10)
        _assert_close(result.params["b"].value, b_truth, "b", rel_tol=0.10)

    def test_fit_recovers_quadratic_baseline_from_auto_guess(self):
        """Same arch+tilt baseline as test_fit_recovers_quadratic_baseline but
        starting from model.guess() rather than near-truth Gaussian params.

        guess() estimates Gaussian params from the data and leaves a=b=0, so
        the optimizer must discover the curvature from a fully auto-guessed
        starting point.
        """
        x = np.linspace(4990.0, 5025.0, 140)
        x_mid = 0.5 * (x[0] + x[-1])
        half_w = 0.5 * (x[-1] - x[0])
        arch_amp = self.H / 4
        tilt_amp = self.H / 8
        a_truth = -arch_amp / half_w**2
        slope = tilt_amp / (x[-1] - x[0])
        b_truth = -2 * a_truth * x_mid + slope
        quad = a_truth * (x - x_mid) ** 2 + slope * (x - x_mid)
        y = _make_1g(x, self.H, self.CEN, self.SIG, c=self.C) + quad
        model = tc_models.Quadratic_1GaussModel()
        result = model.fit(y, model.guess(y, x=x), x=x, method="least_squares")
        assert result.redchi < 1e-4, f"redchi={result.redchi:.3g}"
        _assert_close(result.params["a"].value, a_truth, "a", rel_tol=0.10)
        _assert_close(result.params["b"].value, b_truth, "b", rel_tol=0.10)


# ===========================================================================
# 7. Quadratic_2GaussModel
# ===========================================================================


class TestQuadratic2GaussModel:
    """Quadratic (a·x²+b·x+c) + 2 Gaussians composite model.

    Uses _guess_2gauss; a and b are not guessed and remain at the model default.
    """

    H_NARROW, SIG_NARROW = 20.0, 1.2
    H_BROAD, SIG_BROAD = 6.0, 4.0
    CEN, C = 6563.0, 1.5

    @pytest.fixture
    def xy(self):
        x = np.linspace(6535.0, 6595.0, 200)
        y = (
            self.C
            + _gauss(x, self.H_NARROW, self.CEN, self.SIG_NARROW)
            + _gauss(x, self.H_BROAD, self.CEN, self.SIG_BROAD)
        )
        return x, y

    def test_param_names(self):
        model = tc_models.Quadratic_2GaussModel()
        for p in (
            "g1_height",
            "g1_center",
            "g1_sigma",
            "g2_height",
            "g2_center",
            "g2_sigma",
            "a",
            "b",
            "c",
        ):
            assert p in model.param_names

    def test_fwhm_flux_constrained_both_components(self):
        """Both g1 and g2 must expose fwhm and flux as constrained expressions."""
        model = tc_models.Quadratic_2GaussModel()
        pars = model.make_params()
        for comp in ("g1_", "g2_"):
            assert pars[comp + "fwhm"].expr is not None
            assert pars[comp + "flux"].expr is not None

    def test_eval_correct(self, xy):
        x, y = xy
        model = tc_models.Quadratic_2GaussModel()
        pars = model.make_params(
            g1_height=self.H_BROAD,
            g1_center=self.CEN,
            g1_sigma=self.SIG_BROAD,
            g2_height=self.H_NARROW,
            g2_center=self.CEN,
            g2_sigma=self.SIG_NARROW,
            a=0.0,
            b=0.0,
            c=self.C,
        )
        assert np.allclose(model.eval(pars, x=x), y, rtol=1e-12)

    def test_fit_recovers_parameters(self, xy):
        """Fit from near-truth initial params should recover components to 5 %."""
        x, y = xy
        model = tc_models.Quadratic_2GaussModel()
        pars = model.make_params(
            g1_height=self.H_BROAD * 0.9,
            g1_center=self.CEN,
            g1_sigma=self.SIG_BROAD * 1.1,
            g2_height=self.H_NARROW * 1.1,
            g2_center=self.CEN,
            g2_sigma=self.SIG_NARROW * 0.9,
            a=0.0,
            b=0.0,
            c=self.C * 1.1,
        )
        result = model.fit(y, pars, x=x)
        assert result.redchi < 1e-4, f"redchi={result.redchi:.3g}"
        s1 = result.params["g1_sigma"].value
        s2 = result.params["g2_sigma"].value
        narrow, broad = ("g1", "g2") if s1 < s2 else ("g2", "g1")
        _assert_close(
            result.params[f"{narrow}_height"].value, self.H_NARROW, f"{narrow}_height"
        )
        _assert_close(
            result.params[f"{broad}_height"].value, self.H_BROAD, f"{broad}_height"
        )
        _assert_close(result.params["c"].value, self.C, "c")

    def test_guess_produces_finite_params(self, xy):
        x, y = xy
        model = tc_models.Quadratic_2GaussModel()
        pars = model.guess(y, x=x)
        for name, par in pars.items():
            if par.expr is None and par.vary:
                assert np.isfinite(par.value), f"Non-finite guess for '{name}'"

    def test_guess_leaves_a_b_at_default(self, xy):
        """_guess_2gauss does not update a or b; they stay at the model default."""
        x, y = xy
        model = tc_models.Quadratic_2GaussModel()
        default_pars = model.make_params()
        guessed_pars = model.guess(y, x=x)
        assert guessed_pars["a"].value == pytest.approx(default_pars["a"].value)
        assert guessed_pars["b"].value == pytest.approx(default_pars["b"].value)

    def test_fit_recovers_quadratic_baseline(self):
        """Fit must recover non-zero a and b from a combined arch+tilt baseline.

        Truth: a_arch*(x-x_mid)^2 + slope*(x-x_mid)  (concave-down arch, ~25%
        of the narrow-component peak height, plus a gentle positive tilt ~12.5%
        of peak edge-to-edge).  Fit starts with a=b=0 and c=C.
        """
        x = np.linspace(6535.0, 6595.0, 200)
        x_mid = 0.5 * (x[0] + x[-1])
        half_w = 0.5 * (x[-1] - x[0])
        arch_amp = self.H_NARROW / 4  # 5.0  — 25% of narrow peak
        tilt_amp = self.H_NARROW / 8  # 2.5  — 12.5% of narrow peak
        a_truth = -arch_amp / half_w**2
        slope = tilt_amp / (x[-1] - x[0])
        b_truth = -2 * a_truth * x_mid + slope
        quad = a_truth * (x - x_mid) ** 2 + slope * (x - x_mid)
        y = (
            self.C
            + _gauss(x, self.H_NARROW, self.CEN, self.SIG_NARROW)
            + _gauss(x, self.H_BROAD, self.CEN, self.SIG_BROAD)
            + quad
        )
        model = tc_models.Quadratic_2GaussModel()
        pars = model.make_params(
            g1_height=self.H_BROAD,
            g1_center=self.CEN,
            g1_sigma=self.SIG_BROAD,
            g2_height=self.H_NARROW,
            g2_center=self.CEN,
            g2_sigma=self.SIG_NARROW,
            a=0.0,
            b=0.0,
            c=self.C,
        )
        result = model.fit(y, pars, x=x, method="least_squares")
        assert result.redchi < 1e-4, f"redchi={result.redchi:.3g}"
        _assert_close(result.params["a"].value, a_truth, "a", rel_tol=0.10)
        _assert_close(result.params["b"].value, b_truth, "b", rel_tol=0.10)

    def test_fit_recovers_quadratic_baseline_from_auto_guess(self):
        """Same arch+tilt baseline as test_fit_recovers_quadratic_baseline but
        starting from model.guess().

        _guess_2gauss is designed for the narrow+broad co-centred layout used
        in the class fixture, so it produces a valid Gaussian starting point.
        a and b remain at 0 from the guess; the optimizer discovers them.
        """
        x = np.linspace(6535.0, 6595.0, 200)
        x_mid = 0.5 * (x[0] + x[-1])
        half_w = 0.5 * (x[-1] - x[0])
        arch_amp = self.H_NARROW / 4
        tilt_amp = self.H_NARROW / 8
        a_truth = -arch_amp / half_w**2
        slope = tilt_amp / (x[-1] - x[0])
        b_truth = -2 * a_truth * x_mid + slope
        quad = a_truth * (x - x_mid) ** 2 + slope * (x - x_mid)
        y = (
            self.C
            + _gauss(x, self.H_NARROW, self.CEN, self.SIG_NARROW)
            + _gauss(x, self.H_BROAD, self.CEN, self.SIG_BROAD)
            + quad
        )
        model = tc_models.Quadratic_2GaussModel()
        result = model.fit(y, model.guess(y, x=x), x=x, method="least_squares")
        assert result.redchi < 1e-4, f"redchi={result.redchi:.3g}"
        _assert_close(result.params["a"].value, a_truth, "a", rel_tol=0.10)
        _assert_close(result.params["b"].value, b_truth, "b", rel_tol=0.10)


# ===========================================================================
# 8. Const_3GaussModel
# ===========================================================================


class TestConst3GaussModel:
    """Constant + 3 Gaussians composite model.

    _guess_3gauss defaults place components at (center−σ₀, center, center+σ₀)
    with heights in ratio 1:4:1 and all sigmas equal.  The fixture constructs
    data matching these assumptions so auto-guess tests are meaningful.
    """

    SIG = 1.5
    CEN = 6563.0
    H1, H2, H3 = 3.0, 12.0, 3.0  # heights ratio 1:4:1
    C = 1.5

    @pytest.fixture
    def xy(self):
        x = np.linspace(6535.0, 6595.0, 200)
        y = _make_3g(
            x,
            self.H1,
            self.CEN - self.SIG,
            self.SIG,
            self.H2,
            self.CEN,
            self.SIG,
            self.H3,
            self.CEN + self.SIG,
            self.SIG,
            c=self.C,
        )
        return x, y

    def test_param_names(self):
        model = tc_models.Const_3GaussModel()
        for p in (
            "g1_height",
            "g1_center",
            "g1_sigma",
            "g2_height",
            "g2_center",
            "g2_sigma",
            "g3_height",
            "g3_center",
            "g3_sigma",
            "c",
        ):
            assert p in model.param_names

    def test_fwhm_flux_constrained_all_components(self):
        """All three components must expose fwhm and flux as constrained expressions."""
        model = tc_models.Const_3GaussModel()
        pars = model.make_params()
        for comp in ("g1_", "g2_", "g3_"):
            assert pars[comp + "fwhm"].expr is not None, (
                f"{comp}fwhm should be constrained"
            )
            assert pars[comp + "flux"].expr is not None, (
                f"{comp}flux should be constrained"
            )

    def test_eval_correct(self, xy):
        x, y = xy
        model = tc_models.Const_3GaussModel()
        pars = model.make_params(
            g1_height=self.H1,
            g1_center=self.CEN - self.SIG,
            g1_sigma=self.SIG,
            g2_height=self.H2,
            g2_center=self.CEN,
            g2_sigma=self.SIG,
            g3_height=self.H3,
            g3_center=self.CEN + self.SIG,
            g3_sigma=self.SIG,
            c=self.C,
        )
        assert np.allclose(model.eval(pars, x=x), y, rtol=1e-12)

    def test_fit_recovers_parameters(self):
        """Fit from near-truth initial params on asymmetric, well-separated components.

        The symmetric H1=H3 fixture has height-redistribution degeneracy (equal-sigma
        Gaussians at ±σ₀ can exchange heights without raising residuals).  This method
        uses bespoke data with three distinct, well-separated (>4σ) components and
        sorts by center for label-agnostic comparison.
        """
        H1, H2, H3 = 4.0, 15.0, 6.0
        c1, c2, c3 = self.CEN - 6.0, self.CEN, self.CEN + 5.0
        SIG = 1.3
        C = 1.5
        x = np.linspace(6530.0, 6600.0, 200)
        y = _make_3g(x, H1, c1, SIG, H2, c2, SIG, H3, c3, SIG, c=C)
        model = tc_models.Const_3GaussModel()
        pars = model.make_params(
            g1_height=H1 * 0.9,
            g1_center=c1,
            g1_sigma=SIG * 1.1,
            g2_height=H2 * 1.1,
            g2_center=c2,
            g2_sigma=SIG * 0.9,
            g3_height=H3 * 0.9,
            g3_center=c3,
            g3_sigma=SIG * 1.1,
            c=C * 1.1,
        )
        result = model.fit(y, pars, x=x)
        assert result.redchi < 1e-4, f"redchi={result.redchi:.3g}"
        # Sort components by fitted center to assign labels unambiguously
        fitted = sorted(
            [
                (
                    result.params[f"g{i}_center"].value,
                    result.params[f"g{i}_height"].value,
                )
                for i in (1, 2, 3)
            ]
        )
        expected = sorted([(c1, H1), (c2, H2), (c3, H3)])
        for (cf, hf), (ce, he) in zip(fitted, expected):
            _assert_close(cf, ce, f"center~{ce:.0f}")
            _assert_close(hf, he, f"height~{he:.0f}")
        _assert_close(result.params["c"].value, C, "c")

    def test_guess_produces_finite_params(self, xy):
        x, y = xy
        model = tc_models.Const_3GaussModel()
        pars = model.guess(y, x=x)
        for name, par in pars.items():
            if par.expr is None and par.vary:
                assert np.isfinite(par.value), f"Non-finite guess for '{name}'"

    def test_fit_from_auto_guess(self):
        """Auto-guess should converge on a 3-component model at 1 σ peak spacing.

        Distinct sigmas per component make the problem uniquely identifiable
        (equal sigmas create label-order degeneracy) and break the symmetric
        local minima that trap Levenberg–Marquardt.

        method='least_squares' (scipy TRF) is the production default in
        threadcount and the only optimizer that reliably recovers parameters at
        ≤1 σ spacing with 0 % error.  leastsq (LM) fails here (~260 % height
        error); see ROADMAP §1.4 for the full benchmark table.

        Components are identified by sorting fitted centers low → mid → high.
        """
        H1, H2, H3 = 3.0, 12.0, 3.0
        SIG1, SIG2, SIG3 = 1.0, 1.5, 1.1  # distinct — breaks label degeneracy
        SPACING = self.SIG  # 1 σ spacing; least_squares handles it
        c1 = self.CEN - SPACING
        c2 = self.CEN
        c3 = self.CEN + SPACING
        x = np.linspace(6535.0, 6595.0, 200)
        y = _make_3g(x, H1, c1, SIG1, H2, c2, SIG2, H3, c3, SIG3, c=self.C)
        model = tc_models.Const_3GaussModel()
        result = model.fit(y, model.guess(y, x=x), x=x, method="least_squares")
        assert result.redchi < 1e-4, (
            f"Auto-guess did not converge: redchi={result.redchi:.3g}"
        )
        # Sort components by fitted center (low → mid → high)
        comps = sorted(
            [
                (
                    result.params[f"g{i}_center"].value,
                    result.params[f"g{i}_height"].value,
                    result.params[f"g{i}_sigma"].value,
                )
                for i in (1, 2, 3)
            ]
        )
        expected = [(c1, H1, SIG1), (c2, H2, SIG2), (c3, H3, SIG3)]
        for (cf, hf, sf), (ce, he, se), label in zip(
            comps, expected, ("low", "mid", "high")
        ):
            _assert_close(cf, ce, f"{label} center")
            _assert_close(hf, he, f"{label} height")
            _assert_close(sf, se, f"{label} sigma")
        _assert_close(result.params["c"].value, self.C, "c")


# ===========================================================================
# 9. Quadratic_3GaussModel
# ===========================================================================


class TestQuadratic3GaussModel:
    """Quadratic (a·x²+b·x+c) + 3 Gaussians composite model.

    Shares _guess_3gauss with Const_3GaussModel; a and b are not guessed.
    """

    SIG = 1.5
    CEN = 6563.0
    H1, H2, H3 = 3.0, 12.0, 3.0
    C = 1.5

    @pytest.fixture
    def xy(self):
        x = np.linspace(6535.0, 6595.0, 200)
        y = _make_3g(
            x,
            self.H1,
            self.CEN - self.SIG,
            self.SIG,
            self.H2,
            self.CEN,
            self.SIG,
            self.H3,
            self.CEN + self.SIG,
            self.SIG,
            c=self.C,
        )
        return x, y

    def test_param_names(self):
        model = tc_models.Quadratic_3GaussModel()
        for p in (
            "g1_height",
            "g1_center",
            "g1_sigma",
            "g2_height",
            "g2_center",
            "g2_sigma",
            "g3_height",
            "g3_center",
            "g3_sigma",
            "a",
            "b",
            "c",
        ):
            assert p in model.param_names

    def test_fwhm_flux_constrained_all_components(self):
        model = tc_models.Quadratic_3GaussModel()
        pars = model.make_params()
        for comp in ("g1_", "g2_", "g3_"):
            assert pars[comp + "fwhm"].expr is not None
            assert pars[comp + "flux"].expr is not None

    def test_eval_correct(self, xy):
        x, y = xy
        model = tc_models.Quadratic_3GaussModel()
        pars = model.make_params(
            g1_height=self.H1,
            g1_center=self.CEN - self.SIG,
            g1_sigma=self.SIG,
            g2_height=self.H2,
            g2_center=self.CEN,
            g2_sigma=self.SIG,
            g3_height=self.H3,
            g3_center=self.CEN + self.SIG,
            g3_sigma=self.SIG,
            a=0.0,
            b=0.0,
            c=self.C,
        )
        assert np.allclose(model.eval(pars, x=x), y, rtol=1e-12)

    def test_fit_recovers_parameters(self):
        """Fit from near-truth initial params on asymmetric, well-separated components.

        Same degeneracy applies as for TestConst3GaussModel: the symmetric fixture
        data allows height redistribution.  Uses bespoke well-separated components.
        """
        H1, H2, H3 = 4.0, 15.0, 6.0
        c1, c2, c3 = self.CEN - 6.0, self.CEN, self.CEN + 5.0
        SIG = 1.3
        C = 1.5
        x = np.linspace(6530.0, 6600.0, 200)
        y = _make_3g(x, H1, c1, SIG, H2, c2, SIG, H3, c3, SIG, c=C)
        model = tc_models.Quadratic_3GaussModel()
        pars = model.make_params(
            g1_height=H1 * 0.9,
            g1_center=c1,
            g1_sigma=SIG * 1.1,
            g2_height=H2 * 1.1,
            g2_center=c2,
            g2_sigma=SIG * 0.9,
            g3_height=H3 * 0.9,
            g3_center=c3,
            g3_sigma=SIG * 1.1,
            a=0.0,
            b=0.0,
            c=C * 1.1,
        )
        result = model.fit(y, pars, x=x)
        assert result.redchi < 1e-4, f"redchi={result.redchi:.3g}"
        # Sort components by fitted center for label-agnostic comparison
        fitted = sorted(
            [
                (
                    result.params[f"g{i}_center"].value,
                    result.params[f"g{i}_height"].value,
                )
                for i in (1, 2, 3)
            ]
        )
        expected = sorted([(c1, H1), (c2, H2), (c3, H3)])
        for (cf, hf), (ce, he) in zip(fitted, expected):
            _assert_close(cf, ce, f"center~{ce:.0f}")
            _assert_close(hf, he, f"height~{he:.0f}")
        _assert_close(result.params["c"].value, C, "c")

    def test_guess_produces_finite_params(self, xy):
        x, y = xy
        model = tc_models.Quadratic_3GaussModel()
        pars = model.guess(y, x=x)
        for name, par in pars.items():
            if par.expr is None and par.vary:
                assert np.isfinite(par.value), f"Non-finite guess for '{name}'"

    def test_guess_leaves_a_b_at_default(self, xy):
        """_guess_3gauss does not update a or b; they stay at the model default."""
        x, y = xy
        model = tc_models.Quadratic_3GaussModel()
        default_pars = model.make_params()
        guessed_pars = model.guess(y, x=x)
        assert guessed_pars["a"].value == pytest.approx(default_pars["a"].value)
        assert guessed_pars["b"].value == pytest.approx(default_pars["b"].value)

    def test_fit_recovers_quadratic_baseline(self):
        """Fit must recover non-zero a and b from a combined arch+tilt baseline.

        Uses the same well-separated 3-component setup as test_fit_recovers_parameters
        (H=[4,15,6], three distinct centres) to avoid the degenerate symmetric
        fixture.  Truth: arch ~25% of tallest peak, tilt ~12.5% of tallest peak.
        Fit starts with a=b=0 and c=C.
        """
        H1, H2, H3 = 4.0, 15.0, 6.0
        c1, c2, c3 = self.CEN - 6.0, self.CEN, self.CEN + 5.0
        SIG = 1.3
        C = 1.5
        x = np.linspace(6530.0, 6600.0, 200)
        x_mid = 0.5 * (x[0] + x[-1])
        half_w = 0.5 * (x[-1] - x[0])
        arch_amp = H2 / 4  # 3.75  — 25% of tallest peak
        tilt_amp = H2 / 8  # 1.875 — 12.5% of tallest peak
        a_truth = -arch_amp / half_w**2
        slope = tilt_amp / (x[-1] - x[0])
        b_truth = -2 * a_truth * x_mid + slope
        quad = a_truth * (x - x_mid) ** 2 + slope * (x - x_mid)
        y = _make_3g(x, H1, c1, SIG, H2, c2, SIG, H3, c3, SIG, c=C) + quad
        model = tc_models.Quadratic_3GaussModel()
        pars = model.make_params(
            g1_height=H1,
            g1_center=c1,
            g1_sigma=SIG,
            g2_height=H2,
            g2_center=c2,
            g2_sigma=SIG,
            g3_height=H3,
            g3_center=c3,
            g3_sigma=SIG,
            a=0.0,
            b=0.0,
            c=C,
        )
        result = model.fit(y, pars, x=x, method="least_squares")
        assert result.redchi < 1e-4, f"redchi={result.redchi:.3g}"
        _assert_close(result.params["a"].value, a_truth, "a", rel_tol=0.10)
        _assert_close(result.params["b"].value, b_truth, "b", rel_tol=0.10)


# ===========================================================================
# 10. Const_3GaussModel_fast
# ===========================================================================


class TestConst3GaussModelFast:
    """Fast (numba @njit) version of Const_3GaussModel.

    Key parameterisation:
      ``deltax``   (free)  = g1_center − g2_center
      ``deltaxhi`` (free)  = g3_center − g2_center
      ``g1_center`` = constrained expression ``g2_center + deltax``
      ``g3_center`` = constrained expression ``g2_center + deltaxhi``
    """

    SIG = 1.5
    CEN = 6563.0
    H1, H2, H3 = 3.0, 12.0, 3.0
    C = 1.5

    @pytest.fixture
    def xy(self):
        x = np.linspace(6535.0, 6595.0, 200)
        y = _make_3g(
            x,
            self.H1,
            self.CEN - self.SIG,
            self.SIG,
            self.H2,
            self.CEN,
            self.SIG,
            self.H3,
            self.CEN + self.SIG,
            self.SIG,
            c=self.C,
        )
        return x, y

    def test_param_names(self):
        """deltax and deltaxhi must be free; g1_center and g3_center must not be."""
        model = tc_models.Const_3GaussModel_fast()
        for p in (
            "g1_height",
            "deltax",
            "g1_sigma",
            "g2_height",
            "g2_center",
            "g2_sigma",
            "g3_height",
            "deltaxhi",
            "g3_sigma",
            "c",
        ):
            assert p in model.param_names

    def test_g1_g3_centers_constrained(self):
        """g1_center and g3_center must be derived expressions."""
        model = tc_models.Const_3GaussModel_fast()
        pars = model.make_params()
        assert pars["g1_center"].expr is not None
        assert "g2_center" in pars["g1_center"].expr
        assert "deltax" in pars["g1_center"].expr
        assert pars["g3_center"].expr is not None
        assert "g2_center" in pars["g3_center"].expr
        assert "deltaxhi" in pars["g3_center"].expr

    def test_fwhm_flux_constrained_all_components(self):
        model = tc_models.Const_3GaussModel_fast()
        pars = model.make_params()
        for comp in ("g1_", "g2_", "g3_"):
            assert pars[comp + "fwhm"].expr is not None
            assert pars[comp + "flux"].expr is not None

    def test_eval_correct_with_offsets(self, xy):
        """eval() with deltax=−SIG, deltaxhi=+SIG reproduces the three-component data."""
        x, y = xy
        DELTAX = -self.SIG
        DELTAXHI = self.SIG
        model = tc_models.Const_3GaussModel_fast()
        pars = model.make_params(
            g1_height=self.H1,
            deltax=DELTAX,
            g1_sigma=self.SIG,
            g2_height=self.H2,
            g2_center=self.CEN,
            g2_sigma=self.SIG,
            g3_height=self.H3,
            deltaxhi=DELTAXHI,
            g3_sigma=self.SIG,
            c=self.C,
        )
        assert np.allclose(model.eval(pars, x=x), y, rtol=1e-12)

    def test_eval_changes_with_deltax(self):
        """Changing deltax/deltaxhi must alter eval() output."""
        x = np.linspace(6535.0, 6595.0, 200)
        model = tc_models.Const_3GaussModel_fast()
        pars = model.make_params(
            g1_height=self.H1,
            deltax=0.0,
            g1_sigma=self.SIG,
            g2_height=self.H2,
            g2_center=self.CEN,
            g2_sigma=self.SIG,
            g3_height=self.H3,
            deltaxhi=0.0,
            g3_sigma=self.SIG,
            c=self.C,
        )
        y_zero = model.eval(pars, x=x)
        pars_offset = pars.copy()
        pars_offset["deltax"].set(value=-self.SIG)
        pars_offset["deltaxhi"].set(value=self.SIG)
        y_offset = model.eval(pars_offset, x=x)
        assert not np.allclose(y_zero, y_offset), (
            "Changing deltax/deltaxhi had no effect on eval output"
        )

    def test_fit_recovers_parameters(self, xy):
        """Fit from near-truth initial params should recover all components to 5 %."""
        x, y = xy
        DELTAX = -self.SIG
        DELTAXHI = self.SIG
        model = tc_models.Const_3GaussModel_fast()
        pars = model.make_params(
            g1_height=self.H1 * 0.9,
            deltax=DELTAX * 1.1,
            g1_sigma=self.SIG * 1.1,
            g2_height=self.H2 * 1.1,
            g2_center=self.CEN,
            g2_sigma=self.SIG * 0.9,
            g3_height=self.H3 * 0.9,
            deltaxhi=DELTAXHI * 1.1,
            g3_sigma=self.SIG * 1.1,
            c=self.C * 1.1,
        )
        result = model.fit(y, pars, x=x)
        assert result.redchi < 1e-4, f"redchi={result.redchi:.3g}"
        _assert_close(result.params["g2_height"].value, self.H2, "g2_height")
        _assert_close(result.params["g2_center"].value, self.CEN, "g2_center")
        _assert_close(result.params["deltax"].value, DELTAX, "deltax")
        _assert_close(result.params["deltaxhi"].value, DELTAXHI, "deltaxhi")
        _assert_close(result.params["c"].value, self.C, "c")

    def test_guess_produces_finite_params(self, xy):
        x, y = xy
        model = tc_models.Const_3GaussModel_fast()
        pars = model.guess(y, x=x)
        for name, par in pars.items():
            if par.expr is None and par.vary:
                assert np.isfinite(par.value), f"Non-finite guess for '{name}'"

    def test_fit_from_auto_guess(self):
        """Auto-guess convergence for data near the _guess_3gauss_d defaults.

        Distinct sigmas per component break the label-order degeneracy that
        arises when all sigmas are equal (the shared xy fixture), so all
        parameters can be asserted with the standard 5 % tolerance.
        g1_center and g3_center are constrained expressions; lmfit evaluates
        them so .value gives the computed position for sorting.

        Unlike Const_3GaussModel, the deltax/deltaxhi parameterisation is well
        enough conditioned that the default leastsq (LM) optimizer converges at
        1 σ component spacing — no need to specify method='least_squares' here.
        """
        H1, H2, H3 = 3.0, 12.0, 3.0
        SIG1, SIG2, SIG3 = 1.0, 1.5, 1.1  # distinct — breaks label degeneracy
        DELTAX = -self.SIG  # g1_center = g2_center + deltax
        DELTAXHI = self.SIG  # g3_center = g2_center + deltaxhi
        c1, c2, c3 = self.CEN + DELTAX, self.CEN, self.CEN + DELTAXHI
        C = self.C
        x = np.linspace(6535.0, 6595.0, 200)
        y = _make_3g(x, H1, c1, SIG1, H2, c2, SIG2, H3, c3, SIG3, c=C)
        model = tc_models.Const_3GaussModel_fast()
        result = model.fit(y, model.guess(y, x=x), x=x)
        assert result.redchi < 1e-4, f"redchi={result.redchi:.3g}"
        # Sort components by fitted center (low → mid → high)
        comps = sorted(
            [
                (
                    result.params[f"g{i}_center"].value,
                    result.params[f"g{i}_height"].value,
                    result.params[f"g{i}_sigma"].value,
                )
                for i in (1, 2, 3)
            ]
        )
        expected = [(c1, H1, SIG1), (c2, H2, SIG2), (c3, H3, SIG3)]
        for (cf, hf, sf), (ce, he, se), label in zip(
            comps, expected, ("low", "mid", "high")
        ):
            _assert_close(cf, ce, f"{label} center")
            _assert_close(hf, he, f"{label} height")
            _assert_close(sf, se, f"{label} sigma")
        _assert_close(result.params["c"].value, C, "c")

    def test_parity_with_standard(self, xy):
        """Fast and standard Const_3GaussModel must agree to 0.1 % on physical params.

        The two models use different parameterisations (free centers vs deltax/deltaxhi),
        so we compare the physical quantities --- heights, centers, sigmas, c ---
        by sorting both results by fitted center to avoid label-order ambiguity.
        """
        x, y = xy
        m_std = tc_models.Const_3GaussModel()
        m_fast = tc_models.Const_3GaussModel_fast()
        DELTAX = -self.SIG
        DELTAXHI = self.SIG
        # Standard model: start from near-truth initial params
        pars_std = m_std.make_params(
            g1_height=self.H1 * 0.9,
            g1_center=self.CEN + DELTAX,
            g1_sigma=self.SIG * 1.1,
            g2_height=self.H2 * 1.1,
            g2_center=self.CEN,
            g2_sigma=self.SIG * 0.9,
            g3_height=self.H3 * 0.9,
            g3_center=self.CEN + DELTAXHI,
            g3_sigma=self.SIG * 1.1,
            c=self.C * 1.1,
        )
        r_std = m_std.fit(y, pars_std, x=x, method="least_squares")
        # Fast model: equivalent near-truth start
        pars_fast = m_fast.make_params(
            g1_height=self.H1 * 0.9,
            deltax=DELTAX * 1.1,
            g1_sigma=self.SIG * 1.1,
            g2_height=self.H2 * 1.1,
            g2_center=self.CEN,
            g2_sigma=self.SIG * 0.9,
            g3_height=self.H3 * 0.9,
            deltaxhi=DELTAXHI * 1.1,
            g3_sigma=self.SIG * 1.1,
            c=self.C * 1.1,
        )
        r_fast = m_fast.fit(y, pars_fast, x=x)
        assert r_std.redchi < 1e-4, (
            f"Standard model did not converge: redchi={r_std.redchi:.3g}"
        )
        assert r_fast.redchi < 1e-4, (
            f"Fast model did not converge: redchi={r_fast.redchi:.3g}"
        )

        # Sort both sets of components by center (low → mid → high) for label-agnostic comparison
        def _components(r):
            return sorted(
                [
                    (
                        r.params[f"g{i}_center"].value,
                        r.params[f"g{i}_height"].value,
                        r.params[f"g{i}_sigma"].value,
                    )
                    for i in (1, 2, 3)
                ]
            )

        std_comps = _components(r_std)
        fast_comps = _components(r_fast)
        for (cen_s, h_s, sig_s), (cen_f, h_f, sig_f), label in zip(
            std_comps, fast_comps, ("low", "mid", "high")
        ):
            for qty, v_s, v_f in (
                ("center", cen_s, cen_f),
                ("height", h_s, h_f),
                ("sigma", sig_s, sig_f),
            ):
                diff = abs(v_s - v_f) / max(abs(v_s), 1e-10)
                assert diff < 0.001, (
                    f"{label} {qty}: std={v_s:.6g}, fast={v_f:.6g}, rel_diff={diff:.2%}"
                )
        diff_c = abs(r_std.params["c"].value - r_fast.params["c"].value) / max(
            abs(r_std.params["c"].value), 1e-10
        )
        assert diff_c < 0.001, (
            f"c: std={r_std.params['c'].value:.6g}, fast={r_fast.params['c'].value:.6g}, "
            f"rel_diff={diff_c:.2%}"
        )


# ===========================================================================
# 11. Const_4GaussModel_fast
# ===========================================================================


class TestConst4GaussModelFast:
    """Fast 4-Gaussian model (g4 as reference, g1–g3 centers constrained).

    Constraint: g{i}_center = g4_center + deltax{i}  for i in (1, 2, 3).
    """

    G4_CEN = 6566.0
    DELTAX1, DELTAX2, DELTAX3 = -9.0, -6.0, -3.0
    H1, H2, H3, H4 = 3.0, 5.0, 9.0, 12.0
    SIG = 1.5
    C = 1.0

    @pytest.fixture
    def xy(self):
        x = np.linspace(6545.0, 6585.0, 200)
        y = (
            self.C
            + _gauss(x, self.H1, self.G4_CEN + self.DELTAX1, self.SIG)
            + _gauss(x, self.H2, self.G4_CEN + self.DELTAX2, self.SIG)
            + _gauss(x, self.H3, self.G4_CEN + self.DELTAX3, self.SIG)
            + _gauss(x, self.H4, self.G4_CEN, self.SIG)
        )
        return x, y

    def test_param_names(self):
        model = tc_models.Const_4GaussModel_fast()
        for p in (
            "g1_height",
            "deltax1",
            "g1_sigma",
            "g2_height",
            "deltax2",
            "g2_sigma",
            "g3_height",
            "deltax3",
            "g3_sigma",
            "g4_height",
            "g4_center",
            "g4_sigma",
            "c",
        ):
            assert p in model.param_names

    def test_constrained_centers(self):
        """g1-, g2-, g3_center must have expressions referencing g4_center + deltax{i}."""
        model = tc_models.Const_4GaussModel_fast()
        pars = model.make_params()
        for i, dx in ((1, "deltax1"), (2, "deltax2"), (3, "deltax3")):
            expr = pars[f"g{i}_center"].expr
            assert expr is not None, f"g{i}_center should be constrained"
            assert "g4_center" in expr
            assert dx in expr

    def test_fwhm_flux_constrained_all_components(self):
        model = tc_models.Const_4GaussModel_fast()
        pars = model.make_params()
        for comp in ("g1_", "g2_", "g3_", "g4_"):
            assert pars[comp + "fwhm"].expr is not None
            assert pars[comp + "flux"].expr is not None

    def test_eval_correct(self, xy):
        x, y = xy
        model = tc_models.Const_4GaussModel_fast()
        pars = model.make_params(
            g1_height=self.H1,
            deltax1=self.DELTAX1,
            g1_sigma=self.SIG,
            g2_height=self.H2,
            deltax2=self.DELTAX2,
            g2_sigma=self.SIG,
            g3_height=self.H3,
            deltax3=self.DELTAX3,
            g3_sigma=self.SIG,
            g4_height=self.H4,
            g4_center=self.G4_CEN,
            g4_sigma=self.SIG,
            c=self.C,
        )
        assert np.allclose(model.eval(pars, x=x), y, rtol=1e-12)

    def test_fit_recovers_parameters(self, xy):
        """Fit from near-truth initial params recovers all parameters to 5 %."""
        x, y = xy
        model = tc_models.Const_4GaussModel_fast()
        pars = model.make_params(
            g1_height=self.H1 * 0.9,
            deltax1=self.DELTAX1 * 1.05,
            g1_sigma=self.SIG * 1.1,
            g2_height=self.H2 * 1.1,
            deltax2=self.DELTAX2 * 0.95,
            g2_sigma=self.SIG * 0.9,
            g3_height=self.H3 * 0.9,
            deltax3=self.DELTAX3 * 1.05,
            g3_sigma=self.SIG * 1.1,
            g4_height=self.H4 * 1.1,
            g4_center=self.G4_CEN,
            g4_sigma=self.SIG * 0.9,
            c=self.C * 1.1,
        )
        result = model.fit(y, pars, x=x, method="least_squares")
        assert result.redchi < 1e-4, f"redchi={result.redchi:.3g}"
        _assert_close(result.params["g4_center"].value, self.G4_CEN, "g4_center")
        _assert_close(result.params["g1_height"].value, self.H1, "g1_height")
        _assert_close(result.params["g2_height"].value, self.H2, "g2_height")
        _assert_close(result.params["g3_height"].value, self.H3, "g3_height")
        _assert_close(result.params["g4_height"].value, self.H4, "g4_height")
        _assert_close(result.params["g1_sigma"].value, self.SIG, "g1_sigma")
        _assert_close(result.params["g2_sigma"].value, self.SIG, "g2_sigma")
        _assert_close(result.params["g3_sigma"].value, self.SIG, "g3_sigma")
        _assert_close(result.params["g4_sigma"].value, self.SIG, "g4_sigma")
        _assert_close(result.params["deltax1"].value, self.DELTAX1, "deltax1")
        _assert_close(result.params["deltax2"].value, self.DELTAX2, "deltax2")
        _assert_close(result.params["deltax3"].value, self.DELTAX3, "deltax3")
        _assert_close(result.params["c"].value, self.C, "c")

    def test_guess_produces_finite_params(self, xy):
        x, y = xy
        model = tc_models.Const_4GaussModel_fast()
        pars = model.guess(y, x=x)
        for name, par in pars.items():
            if par.expr is None and par.vary:
                assert np.isfinite(par.value), f"Non-finite guess for '{name}'"

    def test_fit_from_auto_guess(self):
        """Auto-guess convergence on data matching _guess_4gauss_d defaults.

        Default centers=(-2, -1, +1, +2)*σ₀.  g4 is the rightmost reference
        component.  Recovered centers and heights are sorted by wavelength before
        comparison, so the test is insensitive to which component the optimizer
        labels as g4.
        """
        SIG = 1.5
        CEN = 6563.0
        # _guess_4gauss_d places g4 at CEN + 2*SIG (the +2 default center)
        G4_CEN = CEN + 2 * SIG  # 6566
        c1 = CEN - 2 * SIG  # 6560
        c2 = CEN - SIG  # 6561.5
        c3 = CEN + SIG  # 6564.5
        c4 = G4_CEN  # 6566
        H1, H2, H3, H4 = 2.0, 3.0, 8.0, 10.0
        C = 1.0
        x = np.linspace(6545.0, 6585.0, 200)
        y = (
            C
            + _gauss(x, H1, c1, SIG)
            + _gauss(x, H2, c2, SIG)
            + _gauss(x, H3, c3, SIG)
            + _gauss(x, H4, c4, SIG)
        )
        model = tc_models.Const_4GaussModel_fast()
        result = model.fit(y, model.guess(y, x=x), x=x, method="least_squares")
        assert result.redchi < 1e-3, f"redchi={result.redchi:.3g}"
        # Sort components by fitted absolute wavelength, then compare to truth.
        p = result.params
        g4c = p["g4_center"].value
        fitted = sorted(
            [
                (g4c + p["deltax1"].value, p["g1_height"].value),
                (g4c + p["deltax2"].value, p["g2_height"].value),
                (g4c + p["deltax3"].value, p["g3_height"].value),
                (g4c, p["g4_height"].value),
            ]
        )
        truth = sorted([(c1, H1), (c2, H2), (c3, H3), (c4, H4)])
        for (fc, fh), (tc, th) in zip(fitted, truth):
            _assert_close(fc, tc, f"center@{tc:.1f}")
            _assert_close(fh, th, f"height@{tc:.1f}")
        for i in (1, 2, 3, 4):
            _assert_close(result.params[f"g{i}_sigma"].value, SIG, f"g{i}_sigma")
        _assert_close(result.params["c"].value, C, "c")


# ===========================================================================
# 12. Const_4GaussModel_constrained_SII_fast
# ===========================================================================


class TestConst4GaussModelConstrainedSIIFast:
    """Fast 4-Gaussian model for the [SII]λλ6716,6731 doublet.

    The g4–g2 wavelength separation is hardcoded (DELTAX24 = −14.37 Å).
    Free parameters deltax12, deltax34 parameterise outflow offsets within
    each doublet member.  g4 is the SII 6731 reference; g2 is SII 6716.
    """

    G4_CEN = 6731.0  # SII 6731 (reference / systemic)
    DELTAX12 = -5.0  # outflow offset within SII 6716 pair
    DELTAX34 = -5.0  # outflow offset within SII 6731 pair
    H1, H2, H3, H4 = 2.0, 8.0, 2.0, 8.0
    SIG = 1.2
    C = 1.0

    @pytest.fixture
    def xy(self):
        c1 = self.G4_CEN + gaussian4CH_constrained_SII_d_DELTAX24 + self.DELTAX12
        c2 = self.G4_CEN + gaussian4CH_constrained_SII_d_DELTAX24
        c3 = self.G4_CEN + self.DELTAX34
        c4 = self.G4_CEN
        x = np.linspace(6700.0, 6750.0, 200)
        y = (
            self.C
            + _gauss(x, self.H1, c1, self.SIG)
            + _gauss(x, self.H2, c2, self.SIG)
            + _gauss(x, self.H3, c3, self.SIG)
            + _gauss(x, self.H4, c4, self.SIG)
        )
        return x, y

    def test_param_names(self):
        model = tc_models.Const_4GaussModel_constrained_SII_fast()
        for p in (
            "g1_height",
            "deltax12",
            "g1_sigma",
            "g2_height",
            "g2_sigma",
            "g3_height",
            "deltax34",
            "g3_sigma",
            "g4_height",
            "g4_center",
            "g4_sigma",
            "c",
        ):
            assert p in model.param_names

    def test_constrained_centers(self):
        """g2_center contains hardcoded DELTAX24; g1 also has deltax12; g3 has deltax34."""
        model = tc_models.Const_4GaussModel_constrained_SII_fast()
        pars = model.make_params()
        assert pars["g2_center"].expr is not None
        assert "g4_center" in pars["g2_center"].expr
        assert str(gaussian4CH_constrained_SII_d_DELTAX24) in pars["g2_center"].expr
        assert pars["g1_center"].expr is not None
        assert "deltax12" in pars["g1_center"].expr
        assert pars["g3_center"].expr is not None
        assert "deltax34" in pars["g3_center"].expr

    def test_fwhm_flux_constrained_all_components(self):
        model = tc_models.Const_4GaussModel_constrained_SII_fast()
        pars = model.make_params()
        for comp in ("g1_", "g2_", "g3_", "g4_"):
            assert pars[comp + "fwhm"].expr is not None
            assert pars[comp + "flux"].expr is not None

    def test_eval_correct(self, xy):
        """eval() with known params reproduces the injected SII doublet spectrum."""
        x, y = xy
        model = tc_models.Const_4GaussModel_constrained_SII_fast()
        pars = model.make_params(
            g1_height=self.H1,
            deltax12=self.DELTAX12,
            g1_sigma=self.SIG,
            g2_height=self.H2,
            g2_sigma=self.SIG,
            g3_height=self.H3,
            deltax34=self.DELTAX34,
            g3_sigma=self.SIG,
            g4_height=self.H4,
            g4_center=self.G4_CEN,
            g4_sigma=self.SIG,
            c=self.C,
        )
        assert np.allclose(model.eval(pars, x=x), y, rtol=1e-12)

    def test_fit_recovers_parameters(self, xy):
        """Fit from near-truth initial params recovers key parameters to 5 %."""
        x, y = xy
        model = tc_models.Const_4GaussModel_constrained_SII_fast()
        pars = model.make_params(
            g1_height=self.H1 * 0.9,
            deltax12=self.DELTAX12 * 1.1,
            g1_sigma=self.SIG * 1.1,
            g2_height=self.H2 * 1.1,
            g2_sigma=self.SIG * 0.9,
            g3_height=self.H3 * 0.9,
            deltax34=self.DELTAX34 * 1.1,
            g3_sigma=self.SIG * 1.1,
            g4_height=self.H4 * 1.1,
            g4_center=self.G4_CEN,
            g4_sigma=self.SIG * 0.9,
            c=self.C * 1.1,
        )
        result = model.fit(y, pars, x=x, method="least_squares")
        assert result.redchi < 1e-4, f"redchi={result.redchi:.3g}"
        _assert_close(result.params["g4_center"].value, self.G4_CEN, "g4_center")
        _assert_close(result.params["g4_height"].value, self.H4, "g4_height")
        _assert_close(result.params["g2_height"].value, self.H2, "g2_height")
        _assert_close(result.params["deltax12"].value, self.DELTAX12, "deltax12")
        _assert_close(result.params["deltax34"].value, self.DELTAX34, "deltax34")
        _assert_close(result.params["c"].value, self.C, "c")

    def test_guess_produces_finite_params(self, xy):
        x, y = xy
        model = tc_models.Const_4GaussModel_constrained_SII_fast()
        pars = model.guess(y, x=x)
        for name, par in pars.items():
            if par.expr is None and par.vary:
                assert np.isfinite(par.value), f"Non-finite guess for '{name}'"

    def test_fit_from_auto_guess(self, xy):
        """Auto-guess on SII doublet data should converge with TRF optimizer.

        _guess_multiline4_constrained_d places g4_center at peak + deltax4
        (default deltax4=10).  With H2=H4=8, the spectrum maximum sits near
        SII 6716, so the initial g4_center lands ~4 Å below the truth; the LM
        optimizer traps in this case.  method='least_squares' (TRF) recovers the
        correct solution, consistent with production usage in analyze_outflow_extent.
        """
        x, y = xy
        model = tc_models.Const_4GaussModel_constrained_SII_fast()
        result = model.fit(y, model.guess(y, x=x), x=x, method="least_squares")
        assert result.redchi < 1e-3, (
            f"Auto-guess fit did not converge: redchi={result.redchi:.3g}"
        )
        _assert_close(
            result.params["g4_center"].value, self.G4_CEN, "g4_center", rel_tol=0.10
        )
        _assert_close(result.params["g2_height"].value, self.H2, "g2_height")
        _assert_close(result.params["deltax12"].value, self.DELTAX12, "deltax12")
        _assert_close(result.params["c"].value, self.C, "c", rel_tol=0.10)


# ===========================================================================
# 13. Const_6GaussModel_fast
# ===========================================================================


class TestConst6GaussModelFast:
    """Fast 6-Gaussian model (g4 as reference, g1–g3 and g5–g6 centers constrained).

    Constraint: g{i}_center = g4_center + deltax{i}  for i in (1, 2, 3, 5, 6).
    """

    G4_CEN = 6566.0
    DELTAX1, DELTAX2, DELTAX3 = -8.0, -5.0, -2.0
    DELTAX5, DELTAX6 = 5.0, 9.0
    H1, H2, H3, H4, H5, H6 = 2.0, 3.0, 7.0, 10.0, 4.0, 2.0
    SIG = 1.5
    C = 1.0

    @pytest.fixture
    def xy(self):
        x = np.linspace(6540.0, 6595.0, 200)
        y = (
            self.C
            + _gauss(x, self.H1, self.G4_CEN + self.DELTAX1, self.SIG)
            + _gauss(x, self.H2, self.G4_CEN + self.DELTAX2, self.SIG)
            + _gauss(x, self.H3, self.G4_CEN + self.DELTAX3, self.SIG)
            + _gauss(x, self.H4, self.G4_CEN, self.SIG)
            + _gauss(x, self.H5, self.G4_CEN + self.DELTAX5, self.SIG)
            + _gauss(x, self.H6, self.G4_CEN + self.DELTAX6, self.SIG)
        )
        return x, y

    def test_param_names(self):
        model = tc_models.Const_6GaussModel_fast()
        for p in (
            "g1_height",
            "deltax1",
            "g1_sigma",
            "g2_height",
            "deltax2",
            "g2_sigma",
            "g3_height",
            "deltax3",
            "g3_sigma",
            "g4_height",
            "g4_center",
            "g4_sigma",
            "g5_height",
            "deltax5",
            "g5_sigma",
            "g6_height",
            "deltax6",
            "g6_sigma",
            "c",
        ):
            assert p in model.param_names

    def test_constrained_centers(self):
        """g1–g3, g5–g6 centers must be constrained expressions referencing g4_center."""
        model = tc_models.Const_6GaussModel_fast()
        pars = model.make_params()
        for i, dx in (
            (1, "deltax1"),
            (2, "deltax2"),
            (3, "deltax3"),
            (5, "deltax5"),
            (6, "deltax6"),
        ):
            expr = pars[f"g{i}_center"].expr
            assert expr is not None, f"g{i}_center should be constrained"
            assert "g4_center" in expr
            assert dx in expr

    def test_fwhm_flux_constrained_all_components(self):
        model = tc_models.Const_6GaussModel_fast()
        pars = model.make_params()
        for comp in ("g1_", "g2_", "g3_", "g4_", "g5_", "g6_"):
            assert pars[comp + "fwhm"].expr is not None
            assert pars[comp + "flux"].expr is not None

    def test_eval_correct(self, xy):
        x, y = xy
        model = tc_models.Const_6GaussModel_fast()
        pars = model.make_params(
            g1_height=self.H1,
            deltax1=self.DELTAX1,
            g1_sigma=self.SIG,
            g2_height=self.H2,
            deltax2=self.DELTAX2,
            g2_sigma=self.SIG,
            g3_height=self.H3,
            deltax3=self.DELTAX3,
            g3_sigma=self.SIG,
            g4_height=self.H4,
            g4_center=self.G4_CEN,
            g4_sigma=self.SIG,
            g5_height=self.H5,
            deltax5=self.DELTAX5,
            g5_sigma=self.SIG,
            g6_height=self.H6,
            deltax6=self.DELTAX6,
            g6_sigma=self.SIG,
            c=self.C,
        )
        assert np.allclose(model.eval(pars, x=x), y, rtol=1e-12)

    def test_fit_recovers_parameters(self, xy):
        """Fit from near-truth initial params recovers key parameters to 5 %."""
        x, y = xy
        model = tc_models.Const_6GaussModel_fast()
        pars = model.make_params(
            g1_height=self.H1 * 0.9,
            deltax1=self.DELTAX1 * 1.05,
            g1_sigma=self.SIG * 1.1,
            g2_height=self.H2 * 1.1,
            deltax2=self.DELTAX2 * 0.95,
            g2_sigma=self.SIG * 0.9,
            g3_height=self.H3 * 0.9,
            deltax3=self.DELTAX3 * 1.05,
            g3_sigma=self.SIG * 1.1,
            g4_height=self.H4 * 1.1,
            g4_center=self.G4_CEN,
            g4_sigma=self.SIG * 0.9,
            g5_height=self.H5 * 0.9,
            deltax5=self.DELTAX5 * 1.05,
            g5_sigma=self.SIG * 1.1,
            g6_height=self.H6 * 1.1,
            deltax6=self.DELTAX6 * 0.95,
            g6_sigma=self.SIG * 0.9,
            c=self.C * 1.1,
        )
        result = model.fit(y, pars, x=x, method="least_squares")
        assert result.redchi < 1e-4, f"redchi={result.redchi:.3g}"
        _assert_close(result.params["g4_center"].value, self.G4_CEN, "g4_center")
        _assert_close(result.params["g1_height"].value, self.H1, "g1_height")
        _assert_close(result.params["g2_height"].value, self.H2, "g2_height")
        _assert_close(result.params["g3_height"].value, self.H3, "g3_height")
        _assert_close(result.params["g4_height"].value, self.H4, "g4_height")
        _assert_close(result.params["g5_height"].value, self.H5, "g5_height")
        _assert_close(result.params["g6_height"].value, self.H6, "g6_height")
        _assert_close(result.params["g1_sigma"].value, self.SIG, "g1_sigma")
        _assert_close(result.params["g2_sigma"].value, self.SIG, "g2_sigma")
        _assert_close(result.params["g3_sigma"].value, self.SIG, "g3_sigma")
        _assert_close(result.params["g4_sigma"].value, self.SIG, "g4_sigma")
        _assert_close(result.params["g5_sigma"].value, self.SIG, "g5_sigma")
        _assert_close(result.params["g6_sigma"].value, self.SIG, "g6_sigma")
        _assert_close(result.params["deltax1"].value, self.DELTAX1, "deltax1")
        _assert_close(result.params["deltax2"].value, self.DELTAX2, "deltax2")
        _assert_close(result.params["deltax3"].value, self.DELTAX3, "deltax3")
        _assert_close(result.params["deltax5"].value, self.DELTAX5, "deltax5")
        _assert_close(result.params["deltax6"].value, self.DELTAX6, "deltax6")
        _assert_close(result.params["c"].value, self.C, "c")

    def test_guess_produces_finite_params(self, xy):
        x, y = xy
        model = tc_models.Const_6GaussModel_fast()
        pars = model.guess(y, x=x)
        for name, par in pars.items():
            if par.expr is None and par.vary:
                assert np.isfinite(par.value), f"Non-finite guess for '{name}'"

    def test_fit_from_auto_guess(self):
        """Auto-guess convergence for all six component centers and heights.

        Bounds on g4_center and all deltax params prevent the optimizer from
        sending components outside the data range, which would otherwise cause
        label-permutation failures.  Recovered components are sorted by fitted
        absolute wavelength before comparing to truth (insensitive to which
        model label ends up at which physical component).
        """
        CEN = 6563.0
        SIG = 1.5
        G4_CEN = CEN + 2 * SIG  # 6566.0
        c1 = CEN - 2 * SIG  # 6560.0
        c2 = CEN - SIG  # 6561.5
        c3 = CEN + SIG  # 6564.5
        c5 = CEN + 5 * SIG  # 6570.5
        c6 = CEN + 6 * SIG  # 6572.0
        H1, H2, H3, H4, H5, H6 = 2.0, 3.0, 8.0, 10.0, 3.0, 2.0
        C = 1.0
        x = np.linspace(6540.0, 6595.0, 200)
        y = (
            C
            + _gauss(x, H1, c1, SIG)
            + _gauss(x, H2, c2, SIG)
            + _gauss(x, H3, c3, SIG)
            + _gauss(x, H4, G4_CEN, SIG)
            + _gauss(x, H5, c5, SIG)
            + _gauss(x, H6, c6, SIG)
        )
        model = tc_models.Const_6GaussModel_fast()
        pars = model.guess(y, x=x)
        xspan = x[-1] - x[0]
        pars["g4_center"].set(min=x[0], max=x[-1])
        for _dx in ("deltax1", "deltax2", "deltax3", "deltax5", "deltax6"):
            pars[_dx].set(min=-xspan, max=xspan)
        result = model.fit(y, pars, x=x, method="least_squares")
        assert result.redchi < 1e-3, f"redchi={result.redchi:.3g}"
        p = result.params
        g4c = p["g4_center"].value
        fitted = sorted(
            [
                (g4c + p["deltax1"].value, p["g1_height"].value),
                (g4c + p["deltax2"].value, p["g2_height"].value),
                (g4c + p["deltax3"].value, p["g3_height"].value),
                (g4c, p["g4_height"].value),
                (g4c + p["deltax5"].value, p["g5_height"].value),
                (g4c + p["deltax6"].value, p["g6_height"].value),
            ]
        )
        truth = sorted([(c1, H1), (c2, H2), (c3, H3), (G4_CEN, H4), (c5, H5), (c6, H6)])
        for (fc, fh), (tc, th) in zip(fitted, truth):
            _assert_close(fc, tc, f"center@{tc:.1f}")
            _assert_close(fh, th, f"height@{tc:.1f}")
        for i in (1, 2, 3, 4, 5, 6):
            _assert_close(p[f"g{i}_sigma"].value, SIG, f"g{i}_sigma")
        _assert_close(p["c"].value, C, "c")


# ===========================================================================
# 14. Const_6GaussModel_constrained_HaNII_fast
# ===========================================================================


class TestConst6GaussModelConstrainedHaNIIFast:
    """Fast 6-Gaussian model for the Hα + [NII]λλ6548,6583 complex.

    Physical offsets DELTAX24 (NII 6548 − Hα) and DELTAX64 (NII 6583 − Hα)
    are hardcoded.  Height factors g{1,3,5}_h_factor parameterise the outflow
    component amplitudes relative to their systemic pair.
    """

    G4_CEN = 6563.0  # Hα (reference / systemic)
    DELTAX12, DELTAX34, DELTAX56 = -5.0, -5.0, -5.0
    G1_H_FACTOR = 0.30  # outflow / systemic for NII 6548 pair
    G3_H_FACTOR = 0.40  # outflow / systemic for Hα pair
    G5_H_FACTOR = 0.35  # outflow / systemic for NII 6583 pair (distinct from G1)
    G2_HEIGHT, G4_HEIGHT, G6_HEIGHT = 5.0, 15.0, 6.0
    SIG_SYS = 1.2  # systemic components (g2, g4, g6)
    SIG_OUT = 1.8  # outflow components (g1, g3, g5) — broader, breaks degeneracy
    C = 1.0

    @pytest.fixture
    def xy(self):
        d24 = gaussian6CH_constrained_HaNII_d_DELTAX24
        d64 = gaussian6CH_constrained_HaNII_d_DELTAX64
        c1 = self.G4_CEN + self.DELTAX12 + d24
        c2 = self.G4_CEN + d24
        c3 = self.G4_CEN + self.DELTAX34
        c4 = self.G4_CEN
        c5 = self.G4_CEN + self.DELTAX56 + d64
        c6 = self.G4_CEN + d64
        h1 = self.G2_HEIGHT * self.G1_H_FACTOR
        h2 = self.G2_HEIGHT
        h3 = self.G4_HEIGHT * self.G3_H_FACTOR
        h4 = self.G4_HEIGHT
        h5 = self.G6_HEIGHT * self.G5_H_FACTOR
        h6 = self.G6_HEIGHT
        x = np.linspace(6530.0, 6610.0, 300)
        y = (
            self.C
            + _gauss(x, h1, c1, self.SIG_OUT)
            + _gauss(x, h2, c2, self.SIG_SYS)
            + _gauss(x, h3, c3, self.SIG_OUT)
            + _gauss(x, h4, c4, self.SIG_SYS)
            + _gauss(x, h5, c5, self.SIG_OUT)
            + _gauss(x, h6, c6, self.SIG_SYS)
        )
        return x, y

    def test_param_names(self):
        model = tc_models.Const_6GaussModel_constrained_HaNII_fast()
        for p in (
            "g1_h_factor",
            "deltax12",
            "g1_sigma",
            "g2_height",
            "g2_sigma",
            "g3_h_factor",
            "deltax34",
            "g3_sigma",
            "g4_height",
            "g4_center",
            "g4_sigma",
            "g5_h_factor",
            "deltax56",
            "g5_sigma",
            "g6_height",
            "g6_sigma",
            "c",
        ):
            assert p in model.param_names

    def test_constrained_centers_and_heights(self):
        """Centers g1–g3, g5–g6 and heights g1, g3, g5 must be constrained expressions."""
        model = tc_models.Const_6GaussModel_constrained_HaNII_fast()
        pars = model.make_params()
        # centers
        assert pars["g2_center"].expr is not None
        assert str(gaussian6CH_constrained_HaNII_d_DELTAX24) in pars["g2_center"].expr
        assert pars["g6_center"].expr is not None
        assert str(gaussian6CH_constrained_HaNII_d_DELTAX64) in pars["g6_center"].expr
        assert pars["g3_center"].expr is not None
        assert "deltax34" in pars["g3_center"].expr
        # heights
        assert pars["g1_height"].expr is not None
        assert "g1_h_factor" in pars["g1_height"].expr
        assert pars["g3_height"].expr is not None
        assert "g3_h_factor" in pars["g3_height"].expr
        assert pars["g5_height"].expr is not None
        assert "g5_h_factor" in pars["g5_height"].expr

    def test_fwhm_flux_constrained_all_components(self):
        model = tc_models.Const_6GaussModel_constrained_HaNII_fast()
        pars = model.make_params()
        for comp in ("g1_", "g2_", "g3_", "g4_", "g5_", "g6_"):
            assert pars[comp + "fwhm"].expr is not None
            assert pars[comp + "flux"].expr is not None

    def test_eval_correct(self, xy):
        """eval() with known params reproduces the injected Hα+[NII] spectrum."""
        x, y = xy
        model = tc_models.Const_6GaussModel_constrained_HaNII_fast()
        pars = model.make_params(
            g1_h_factor=self.G1_H_FACTOR,
            deltax12=self.DELTAX12,
            g1_sigma=self.SIG_OUT,
            g2_height=self.G2_HEIGHT,
            g2_sigma=self.SIG_SYS,
            g3_h_factor=self.G3_H_FACTOR,
            deltax34=self.DELTAX34,
            g3_sigma=self.SIG_OUT,
            g4_height=self.G4_HEIGHT,
            g4_center=self.G4_CEN,
            g4_sigma=self.SIG_SYS,
            g5_h_factor=self.G5_H_FACTOR,
            deltax56=self.DELTAX56,
            g5_sigma=self.SIG_OUT,
            g6_height=self.G6_HEIGHT,
            g6_sigma=self.SIG_SYS,
            c=self.C,
        )
        assert np.allclose(model.eval(pars, x=x), y, rtol=1e-12)

    def test_fit_recovers_parameters(self, xy):
        """Fit from near-truth initial params recovers key parameters to 5 %.

        Outflow components (g1, g3, g5) use SIG_OUT and systemic (g2, g4, g6)
        use SIG_SYS so that the optimizer can distinguish each component.
        h_factors and sigmas are checked directly by parameter name — no
        sorting needed because the model's constraint expressions tie each
        h_factor to its specific pair.
        """
        x, y = xy
        model = tc_models.Const_6GaussModel_constrained_HaNII_fast()
        pars = model.make_params(
            g1_h_factor=self.G1_H_FACTOR * 0.9,
            deltax12=self.DELTAX12 * 1.1,
            g1_sigma=self.SIG_OUT * 1.1,
            g2_height=self.G2_HEIGHT * 1.1,
            g2_sigma=self.SIG_SYS * 0.9,
            g3_h_factor=self.G3_H_FACTOR * 0.9,
            deltax34=self.DELTAX34 * 1.1,
            g3_sigma=self.SIG_OUT * 0.9,
            g4_height=self.G4_HEIGHT * 1.1,
            g4_center=self.G4_CEN,
            g4_sigma=self.SIG_SYS * 1.1,
            g5_h_factor=self.G5_H_FACTOR * 0.9,
            deltax56=self.DELTAX56 * 1.1,
            g5_sigma=self.SIG_OUT * 1.1,
            g6_height=self.G6_HEIGHT * 1.1,
            g6_sigma=self.SIG_SYS * 0.9,
            c=self.C * 1.1,
        )
        result = model.fit(y, pars, x=x, method="least_squares")
        assert result.redchi < 1e-4, f"redchi={result.redchi:.3g}"
        _assert_close(result.params["g4_center"].value, self.G4_CEN, "g4_center")
        _assert_close(result.params["g4_height"].value, self.G4_HEIGHT, "g4_height")
        _assert_close(result.params["g2_height"].value, self.G2_HEIGHT, "g2_height")
        _assert_close(result.params["g6_height"].value, self.G6_HEIGHT, "g6_height")
        _assert_close(result.params["c"].value, self.C, "c")
        # g{1,3,5}_h_factor are explicit free parameters; the model's constraint
        # expressions tie each ratio to its specific pair, so no sorting is needed.
        _assert_close(
            result.params["g1_h_factor"].value, self.G1_H_FACTOR, "g1_h_factor"
        )
        _assert_close(
            result.params["g3_h_factor"].value, self.G3_H_FACTOR, "g3_h_factor"
        )
        _assert_close(
            result.params["g5_h_factor"].value, self.G5_H_FACTOR, "g5_h_factor"
        )
        _assert_close(result.params["g1_sigma"].value, self.SIG_OUT, "g1_sigma")
        _assert_close(result.params["g3_sigma"].value, self.SIG_OUT, "g3_sigma")
        _assert_close(result.params["g5_sigma"].value, self.SIG_OUT, "g5_sigma")
        _assert_close(result.params["g2_sigma"].value, self.SIG_SYS, "g2_sigma")
        _assert_close(result.params["g4_sigma"].value, self.SIG_SYS, "g4_sigma")
        _assert_close(result.params["g6_sigma"].value, self.SIG_SYS, "g6_sigma")

    def test_guess_produces_finite_params(self, xy):
        x, y = xy
        model = tc_models.Const_6GaussModel_constrained_HaNII_fast()
        pars = model.guess(y, x=x)
        for name, par in pars.items():
            if par.expr is None and par.vary:
                assert np.isfinite(par.value), f"Non-finite guess for '{name}'"

    def test_fit_from_auto_guess(self, xy):
        """Auto-guess on Hα+[NII] data should converge.

        _guess_multiline6_constrained_d uses deltax4=0 (g4 placed at spectrum
        peak) and deltax12=deltax34=deltax56=−5 for outflow offsets.  The fit is
        tightly constrained by the hardcoded DELTAX24/DELTAX64 separations, so
        even a rough starting point typically converges.
        """
        x, y = xy
        model = tc_models.Const_6GaussModel_constrained_HaNII_fast()
        result = model.fit(y, model.guess(y, x=x), x=x, method="least_squares")
        assert result.redchi < 1e-3, (
            f"Auto-guess fit did not converge: redchi={result.redchi:.3g}"
        )
        _assert_close(
            result.params["g4_center"].value, self.G4_CEN, "g4_center", rel_tol=0.10
        )
        _assert_close(result.params["g2_height"].value, self.G2_HEIGHT, "g2_height")
        _assert_close(result.params["g6_height"].value, self.G6_HEIGHT, "g6_height")
        _assert_close(
            result.params["g1_h_factor"].value, self.G1_H_FACTOR, "g1_h_factor"
        )
        _assert_close(
            result.params["g3_h_factor"].value, self.G3_H_FACTOR, "g3_h_factor"
        )
        _assert_close(
            result.params["g5_h_factor"].value, self.G5_H_FACTOR, "g5_h_factor"
        )
        _assert_close(result.params["g1_sigma"].value, self.SIG_OUT, "g1_sigma")
        _assert_close(result.params["g3_sigma"].value, self.SIG_OUT, "g3_sigma")
        _assert_close(result.params["g5_sigma"].value, self.SIG_OUT, "g5_sigma")
        _assert_close(result.params["g2_sigma"].value, self.SIG_SYS, "g2_sigma")
        _assert_close(result.params["g4_sigma"].value, self.SIG_SYS, "g4_sigma")
        _assert_close(result.params["g6_sigma"].value, self.SIG_SYS, "g6_sigma")


# ===========================================================================
# 15. Log10_DoubleExponentialModel
# ===========================================================================


class TestLog10DoubleExponentialModel:
    r"""Model: log10(A1·exp(-x/τ1) + A2·exp(-x/τ2)) for two-component outflow profiles.

    All four free parameters have min=0.  The ExponentialModel convention
    (A·exp(-x/decay)) means `decay` here is a positive scale length τ.
    Tests use a physically decaying profile (flux decreasing with radius).
    """

    A1, D1, A2, D2 = 100.0, 3.0, 50.0, 8.0  # (amplitude, scale_length) pairs

    @pytest.fixture
    def xy(self):
        x = np.linspace(0.5, 20.0, 100)
        y = np.log10(self.A1 * np.exp(-x / self.D1) + self.A2 * np.exp(-x / self.D2))
        return x, y

    def test_param_names(self):
        model = tc_models.Log10_DoubleExponentialModel()
        for p in ("e1_amplitude", "e1_decay", "e2_amplitude", "e2_decay"):
            assert p in model.param_names

    def test_amplitude_decay_min_zero(self):
        """Default param hints enforce min ≥ 0 on all four free parameters."""
        model = tc_models.Log10_DoubleExponentialModel()
        pars = model.make_params()
        for p in ("e1_amplitude", "e1_decay", "e2_amplitude", "e2_decay"):
            assert pars[p].min >= 0.0

    def test_eval_correct(self, xy):
        """eval() with known params reproduces the injected log10-exponential data."""
        x, y = xy
        model = tc_models.Log10_DoubleExponentialModel()
        pars = model.make_params(
            e1_amplitude=self.A1,
            e1_decay=self.D1,
            e2_amplitude=self.A2,
            e2_decay=self.D2,
        )
        assert np.allclose(model.eval(pars, x=x), y, rtol=1e-10)

    def test_fit_recovers_parameters(self, xy):
        """Fit from near-truth initial params recovers all four parameters to 5 %."""
        x, y = xy
        model = tc_models.Log10_DoubleExponentialModel()
        pars = model.make_params(
            e1_amplitude=self.A1 * 0.9,
            e1_decay=self.D1 * 1.1,
            e2_amplitude=self.A2 * 1.1,
            e2_decay=self.D2 * 0.9,
        )
        result = model.fit(y, pars, x=x, method="least_squares")
        assert result.redchi < 1e-4, f"redchi={result.redchi:.3g}"
        _assert_close(result.params["e1_amplitude"].value, self.A1, "e1_amplitude")
        _assert_close(result.params["e1_decay"].value, self.D1, "e1_decay")
        _assert_close(result.params["e2_amplitude"].value, self.A2, "e2_amplitude")
        _assert_close(result.params["e2_decay"].value, self.D2, "e2_decay")

    def test_guess_produces_finite_params(self, xy):
        x, y = xy
        model = tc_models.Log10_DoubleExponentialModel()
        pars = model.guess(y, x=x)
        for name, par in pars.items():
            if par.expr is None and par.vary:
                assert np.isfinite(par.value), f"Non-finite guess for '{name}'"

    def test_fit_from_auto_guess(self, xy):
        """Fit from auto-guess with broken degeneracy converges.

        The default a2_factor=d2_factor=1 starts both exponentials identically,
        creating a symmetric flat direction in the objective function.  A ×2
        perturbation to e2_decay breaks this symmetry.  The double-exponential
        landscape is non-convex so convergence to the true solution is not
        guaranteed; we assert only that the fit exits with a small residual,
        with parameter recovery covered by test_fit_recovers_parameters.
        """
        x, y = xy
        model = tc_models.Log10_DoubleExponentialModel()
        pars = model.guess(y, x=x)
        pars["e2_decay"].value *= 2.0  # break the e1 == e2 degeneracy
        result = model.fit(y, pars, x=x)
        assert result.redchi < 1e-2, f"redchi={result.redchi:.3g}"


# ===========================================================================
# 16. Standalone numpy model functions (gaussianH, gaussian2CH, gaussian3CH)
# ===========================================================================


class TestStandaloneModelFunctions:
    """Tests for the bare numpy callables exported from models.py.

    These functions are the underlying callables for the composite models, so
    if they silently misbehave (e.g. wrong sign, wrong parameter mapping) every
    composite-model test could be wrong in the same way.
    """

    X = np.linspace(-5.0, 5.0, 200)

    # ---- gaussianH --------------------------------------------------------

    def test_gaussianH_peak_equals_height(self):
        """At x=center, gaussianH returns exactly height."""
        # No `max(tiny,...)` guard needed when sigma is normal, so rtol=1e-15
        assert np.isclose(
            tc_models.gaussianH(0.0, height=7.0, center=0.0, sigma=1.0), 7.0
        )

    def test_gaussianH_zero_far_from_center(self):
        """Far from center the function decays to effectively zero."""
        val = tc_models.gaussianH(1000.0, height=1.0, center=0.0, sigma=1.0)
        assert val < 1e-100

    def test_gaussianH_sigma_zero_guard(self):
        """sigma=0 must not raise (max(tiny,...) guard); result is height at center."""
        # When sigma=0 the denominator is max(tiny, 0) = tiny, so exp(-big) ≈ 0
        # at any x != center, but at x==center it is height*exp(0) = height.
        val = tc_models.gaussianH(0.0, height=5.0, center=0.0, sigma=0.0)
        assert val == pytest.approx(5.0)

    def test_gaussianH_matches_formula(self):
        """gaussianH(x) == height * exp(-(x-center)^2 / (2*sigma^2)) elementwise."""
        h, cen, sig = 3.0, 1.5, 0.8
        expected = h * np.exp(-((self.X - cen) ** 2) / (2 * sig**2))
        result = tc_models.gaussianH(self.X, height=h, center=cen, sigma=sig)
        assert np.allclose(result, expected, rtol=1e-12)

    # ---- gaussian2CH -------------------------------------------------------

    def test_gaussian2CH_is_sum_of_two_gaussianH_plus_c(self):
        """gaussian2CH == gaussianH(g1) + gaussianH(g2) + c."""
        h1, c1, s1 = 4.0, -1.0, 0.7
        h2, c2, s2 = 2.0, 1.5, 1.2
        c = 0.5
        expected = (
            tc_models.gaussianH(self.X, h1, c1, s1)
            + tc_models.gaussianH(self.X, h2, c2, s2)
            + c
        )
        result = tc_models.gaussian2CH(
            self.X,
            g1_height=h1,
            g1_center=c1,
            g1_sigma=s1,
            g2_height=h2,
            g2_center=c2,
            g2_sigma=s2,
            c=c,
        )
        assert np.allclose(result, expected, rtol=1e-12)

    def test_gaussian2CH_constant_offset_respected(self):
        """Setting c shifts the entire profile uniformly."""
        y0 = tc_models.gaussian2CH(self.X, c=0.0)
        y1 = tc_models.gaussian2CH(self.X, c=3.7)
        assert np.allclose(y1 - y0, 3.7, rtol=1e-12)

    # ---- gaussian3CH -------------------------------------------------------

    def test_gaussian3CH_is_sum_of_three_gaussianH_plus_c(self):
        """gaussian3CH == gaussianH(g1) + gaussianH(g2) + gaussianH(g3) + c."""
        params = dict(
            g1_height=3.0,
            g1_center=-2.0,
            g1_sigma=0.6,
            g2_height=5.0,
            g2_center=0.0,
            g2_sigma=1.0,
            g3_height=2.0,
            g3_center=2.0,
            g3_sigma=0.8,
            c=1.0,
        )
        expected = (
            tc_models.gaussianH(
                self.X, params["g1_height"], params["g1_center"], params["g1_sigma"]
            )
            + tc_models.gaussianH(
                self.X, params["g2_height"], params["g2_center"], params["g2_sigma"]
            )
            + tc_models.gaussianH(
                self.X, params["g3_height"], params["g3_center"], params["g3_sigma"]
            )
            + params["c"]
        )
        result = tc_models.gaussian3CH(self.X, **params)
        assert np.allclose(result, expected, rtol=1e-12)


# ===========================================================================
# 17. Numba @njit parity — gaussian1CH_d and gaussian2CH_d
#    (gaussian3CH_d parity is already tested in test_numba_accuracy above)
# ===========================================================================


def _gaussian1CH_d_pure(x, g1_height=1.0, g1_center=0.0, g1_sigma=1.0, c=0.0):
    """Pure-Python reference for gaussian1CH_d."""
    return (
        g1_height * np.exp(-((1.0 * x - g1_center) ** 2) / max(tiny, (2 * g1_sigma**2)))
        + c
    )


def _gaussian2CH_d_pure(
    x,
    g1_height=1.0,
    deltax=0.0,
    g1_sigma=1.0,
    g2_height=1.0,
    g2_center=0.0,
    g2_sigma=1.0,
    c=0.0,
):
    """Pure-Python reference for gaussian2CH_d."""
    return (
        g1_height
        * np.exp(-((1.0 * x - g2_center - deltax) ** 2) / max(tiny, (2 * g1_sigma**2)))
        + g2_height
        * np.exp(-((1.0 * x - g2_center) ** 2) / max(tiny, (2 * g2_sigma**2)))
        + c
    )


def test_numba_accuracy_1CH_d():
    """gaussian1CH_d (@njit) matches pure-Python reference including nonzero c."""
    rng = np.random.default_rng(RNG_SEED)
    x = np.linspace(4990.0, 5025.0, 100)
    kw = {
        "g1_height": rng.uniform(1.0, 20.0),
        "g1_center": rng.uniform(5000.0, 5015.0),
        "g1_sigma": rng.uniform(0.5, 3.0),
        "c": rng.uniform(-1.0, 5.0),
    }
    assert np.allclose(_gaussian1CH_d_pure(x, **kw), gaussian1CH_d(x, **kw), rtol=1e-12)


def test_numba_accuracy_2CH_d():
    """gaussian2CH_d (@njit) matches pure-Python reference including nonzero deltax and c."""
    rng = np.random.default_rng(RNG_SEED + 1)
    x = np.linspace(6530.0, 6600.0, 120)
    kw = {
        "g1_height": rng.uniform(1.0, 10.0),
        "deltax": rng.uniform(-15.0, 15.0),
        "g1_sigma": rng.uniform(0.5, 3.0),
        "g2_height": rng.uniform(5.0, 20.0),
        "g2_center": rng.uniform(6555.0, 6575.0),
        "g2_sigma": rng.uniform(0.5, 3.0),
        "c": rng.uniform(-1.0, 5.0),
    }
    assert np.allclose(_gaussian2CH_d_pure(x, **kw), gaussian2CH_d(x, **kw), rtol=1e-12)


# ===========================================================================
# 18. basic.py — guess_from_peak
# ===========================================================================


class TestGuessFromPeak:
    """Unit tests for basic.guess_from_peak."""

    def test_returns_positive_height_for_emission(self):
        x = np.linspace(0.0, 10.0, 200)
        y = 1.0 + 8.0 * np.exp(-((x - 5.0) ** 2) / (2 * 0.5**2))
        h, cen, sig = guess_from_peak(y, x)
        assert h > 0
        assert abs(cen - 5.0) < 1.0
        assert sig > 0

    def test_center_near_peak(self):
        """Estimated centre should be within 1 Å of the true peak."""
        x = np.linspace(6540.0, 6590.0, 200)
        y = _make_1g(x, 10.0, 6563.0, 1.5)
        _, cen, _ = guess_from_peak(y, x)
        assert abs(cen - 6563.0) < 1.5

    def test_sigma_reasonable(self):
        """Estimated sigma should be within 50 % of the injected sigma."""
        x = np.linspace(6540.0, 6590.0, 200)
        sig_true = 1.5
        y = _make_1g(x, 10.0, 6563.0, sig_true)
        _, _, sig = guess_from_peak(y, x)
        assert abs(sig - sig_true) / sig_true < 0.5

    def test_negative_returns_negative_height(self):
        """negative=True should return a negative height for an absorption line."""
        x = np.linspace(0.0, 10.0, 200)
        y = -5.0 * np.exp(-((x - 5.0) ** 2) / (2 * 0.8**2))
        h, cen, _ = guess_from_peak(y, x, negative=True)
        assert h < 0
        assert abs(cen - 5.0) < 1.0

    def test_unsorted_x_gives_same_result(self):
        """Reversing x (descending) must give the same estimates as ascending."""
        x = np.linspace(6540.0, 6590.0, 200)
        y = _make_1g(x, 10.0, 6563.0, 1.5)
        h1, c1, s1 = guess_from_peak(y, x)
        h2, c2, s2 = guess_from_peak(y[::-1], x[::-1])
        assert np.isclose(h1, h2)
        assert np.isclose(c1, c2)
        assert np.isclose(s1, s2)

    def test_fallback_sigma_when_few_halfmax_points(self):
        """When the peak is so narrow that <=2 points lie above half-max, the
        fallback sigma = (xmax - xmin) / 6 is used instead of crashing."""
        x = np.linspace(0.0, 6.0, 10)  # coarse grid — narrow peak has few points
        y = np.zeros(10)
        y[5] = 10.0  # single-pixel spike: only 1 point above half-max
        h, cen, sig = guess_from_peak(y, x)
        assert sig == pytest.approx((x[-1] - x[0]) / 6.0)


# ===========================================================================
# 19. basic.py — mean_edges
# ===========================================================================


class TestMeanEdges:
    """Unit tests for basic.mean_edges."""

    Y_FLAT = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    def test_basic_mean_of_edges(self):
        """10 % edges of a 10-element array = first and last 1 element."""
        val = mean_edges(self.Y_FLAT, edge_fraction=0.1)
        # first element=1, last element=10 -> mean of [1, 10] = 5.5
        assert val == pytest.approx(5.5)

    def test_edge_fraction_half_returns_all_mean(self):
        """edge_fraction=0.5 must return the mean of all elements."""
        val = mean_edges(self.Y_FLAT, edge_fraction=0.5)
        assert val == pytest.approx(np.mean(self.Y_FLAT))

    def test_edge_fraction_zero_uses_single_endpoints(self):
        """edge_fraction=0 is allowed; limit is clamped to 1 so we still
        get the mean of y[0] and y[-1]."""
        val = mean_edges(self.Y_FLAT, edge_fraction=0.0)
        assert val == pytest.approx((1.0 + 10.0) / 2.0)

    def test_raises_for_fraction_above_half(self):
        with pytest.raises(ValueError, match="[Ee]dge fraction"):
            mean_edges(self.Y_FLAT, edge_fraction=0.6)

    def test_x_sorting_applied(self):
        """When x is given in reverse order the result must still use the
        correct edge elements (first and last in sorted-x order)."""
        x_desc = np.arange(10, 0, -1, dtype=float)
        y = np.arange(10, dtype=float)  # y[0]=0 corresponds to x=10 (last in sorted)
        # After sorting by x ascending: x=[1..10], y=[9,8,7,...,0]
        # edge elements (10%): y_sorted[0]=9 and y_sorted[-1]=0 -> mean=4.5
        val = mean_edges(y, x=x_desc, edge_fraction=0.1)
        assert val == pytest.approx(4.5)


# ===========================================================================
# 20. basic.py — reapply_certain_model_hints
# ===========================================================================


class TestReapplyCertainModelHints:
    """Unit tests for basic.reapply_certain_model_hints."""

    def test_vary_false_value_is_enforced(self):
        """A hint with vary=False and value=X should override the parameter value."""
        model = tc_models.GaussianModelH()
        model.set_param_hint("sigma", vary=False, value=2.5)
        pars = model.make_params(height=1.0, center=0.0, sigma=99.0)
        pars = reapply_certain_model_hints(model, pars)
        assert pars["sigma"].value == pytest.approx(2.5)
        assert pars["sigma"].vary is False

    def test_expr_is_enforced(self):
        """A hint with expr should turn the parameter into a constrained expression."""
        model = tc_models.GaussianModelH()
        model.set_param_hint("sigma", expr="center/1000.0")
        pars = model.make_params(height=1.0, center=5000.0, sigma=1.0)
        pars = reapply_certain_model_hints(model, pars)
        assert pars["sigma"].expr is not None
        assert "center" in pars["sigma"].expr

    def test_no_matching_hint_is_noop(self):
        """Calling with no relevant hints must not mutate any parameter."""
        model = tc_models.GaussianModelH()
        # Clear all hints so reapply is a noop
        model.param_hints.clear()
        pars = model.make_params(height=5.0, center=3.0, sigma=1.5)
        original_values = {k: p.value for k, p in pars.items() if p.expr is None}
        pars = reapply_certain_model_hints(model, pars)
        for k, v in original_values.items():
            assert pars[k].value == pytest.approx(v)

    def test_vary_false_without_value_is_skipped(self):
        """vary=False without a value key in the hint must not raise and must
        leave the guessed value unchanged (known design constraint).

        set_param_hint("sigma", vary=False) without a value kwarg naturally
        produces a hint dict with no 'value' key, so this is the real scenario.
        """
        model = tc_models.GaussianModelH()
        model.param_hints.clear()  # remove all existing hints including min=0
        model.set_param_hint(
            "sigma", vary=False
        )  # no value kwarg — no 'value' key stored
        assert "value" not in model.param_hints["sigma"], (
            "Precondition: hint must not contain a 'value' key for this test"
        )
        pars = model.make_params(height=1.0, center=0.0, sigma=2.0)
        guessed_sigma = pars["sigma"].value
        pars = reapply_certain_model_hints(model, pars)
        # The function should not crash, and should not change the value
        assert pars["sigma"].value == pytest.approx(guessed_sigma)


# ===========================================================================
# 21. models.py — log10_sum and set_common_limits
# ===========================================================================


class TestLog10Sum:
    """Unit tests for the log10_sum operator used in Log10_DoubleExponentialModel."""

    def test_basic_values(self):
        a = np.array([10.0, 100.0])
        b = np.array([90.0, 0.0])
        result = log10_sum(a, b)
        assert np.allclose(result, np.log10(a + b), rtol=1e-12)

    def test_scalar_inputs(self):
        assert log10_sum(np.float64(10.0), np.float64(90.0)) == pytest.approx(
            np.log10(100.0)
        )

    def test_matches_numpy_log10_of_sum(self):
        rng = np.random.default_rng(RNG_SEED)
        a = rng.uniform(0.1, 10.0, 50)
        b = rng.uniform(0.1, 10.0, 50)
        assert np.allclose(log10_sum(a, b), np.log10(a + b), rtol=1e-12)


class TestSetCommonLimits:
    """Unit tests for models.set_common_limits."""

    @pytest.fixture
    def model_and_pars(self):
        x = np.linspace(6540.0, 6590.0, 100)
        y = _make_1g(x, 10.0, 6563.0, 1.5, c=1.0)
        model = tc_models.Const_1GaussModel()
        pars = model.guess(y, x=x)
        return model, pars, x, y

    def test_height_params_get_min_zero(self, model_and_pars):
        _, pars, x, y = model_and_pars
        pars = set_common_limits(pars, x, y)
        assert pars["g1_height"].min >= 0.0

    def test_sigma_params_get_nonneg_min(self, model_and_pars):
        _, pars, x, y = model_and_pars
        pars = set_common_limits(pars, x, y)
        assert pars["g1_sigma"].min >= 0.0

    def test_center_params_get_bounded_away_from_edges(self, model_and_pars):
        """g1_center.min and g1_center.max should be strictly inside [x[0], x[-1]]."""
        _, pars, x, y = model_and_pars
        pars = set_common_limits(pars, x, y)
        assert pars["g1_center"].min > x[0]
        assert pars["g1_center"].max < x[-1]

    def test_c_param_gets_finite_bounds(self, model_and_pars):
        """The constant c should get finite min and max bounds."""
        _, pars, x, y = model_and_pars
        pars = set_common_limits(pars, x, y)
        assert np.isfinite(pars["c"].min)
        assert np.isfinite(pars["c"].max)

    def test_min_sigma_global_clamps_value(self, model_and_pars):
        """When models.min_sigma > 0, any sigma whose guessed value falls below
        min_sigma must be clamped up to min_sigma."""
        import threadcount.models.models as tc_models_module

        original_min_sigma = tc_models_module.min_sigma
        try:
            tc_models_module.min_sigma = 2.0  # larger than guessed sigma ~1.5
            _, pars, x, y = model_and_pars
            pars = set_common_limits(pars, x, y)
            assert pars["g1_sigma"].min == pytest.approx(2.0)
            assert pars["g1_sigma"].value >= 2.0, (
                f"sigma.value={pars['g1_sigma'].value:.3f} should be clamped to min_sigma=2.0"
            )
        finally:
            tc_models_module.min_sigma = original_min_sigma  # always restore

    @pytest.mark.parametrize(
        "model_cls, n_gauss, heights, center_offsets, sigmas",
        [
            (
                tc_models.Const_2GaussModel,
                2,
                [20.0, 6.0],
                [0.0, 0.0],  # narrow + broad, same centre (designed use-case)
                [1.2, 4.0],
            ),
            (
                tc_models.Const_3GaussModel,
                3,
                [3.0, 12.0, 3.0],
                [-1.5, 0.0, 1.5],  # (CEN-SIG, CEN, CEN+SIG) — _guess_3gauss scenario
                [1.5, 1.5, 1.5],
            ),
        ],
        ids=["Const_2GaussModel", "Const_3GaussModel"],
    )
    def test_all_gaussian_params_bounded_multi_component(
        self, model_cls, n_gauss, heights, center_offsets, sigmas
    ):
        """All g{n}_height, g{n}_sigma, and g{n}_center params on 2G and 3G models
        receive appropriate bounds from set_common_limits.

        Verifies the property that set_common_limits is not limited to 1-component
        models — it iterates over all parameters by suffix, so every component in a
        multi-Gaussian model should be bounded consistently.
        """
        CEN, C = 6563.0, 1.5
        x = np.linspace(6535.0, 6595.0, 200)
        y = C + sum(
            _gauss(x, h, CEN + dc, s)
            for h, dc, s in zip(heights, center_offsets, sigmas)
        )
        model = model_cls()
        pars = model.guess(y, x=x)
        pars = set_common_limits(pars, x, y)

        for n in range(1, n_gauss + 1):
            assert pars[f"g{n}_height"].min >= 0.0, (
                f"g{n}_height.min should be >= 0, got {pars[f'g{n}_height'].min}"
            )
            assert pars[f"g{n}_sigma"].min >= 0.0, (
                f"g{n}_sigma.min should be >= 0, got {pars[f'g{n}_sigma'].min}"
            )
            assert pars[f"g{n}_center"].min > x[0], (
                f"g{n}_center.min={pars[f'g{n}_center'].min:.3f} should be > x[0]={x[0]:.3f}"
            )
            assert pars[f"g{n}_center"].max < x[-1], (
                f"g{n}_center.max={pars[f'g{n}_center'].max:.3f} should be < x[-1]={x[-1]:.3f}"
            )

        assert np.isfinite(pars["c"].min), "c.min should be finite"
        assert np.isfinite(pars["c"].max), "c.max should be finite"


# ===========================================================================
# 22. Verify @njit decoration is active for all fast_models functions
# ===========================================================================


def test_njit_functions_are_compiled():
    """All @njit-decorated functions in fast_models must be numba CPUDispatcher
    instances, not plain Python callables.

    If numba is unavailable or the decorator silently fell back to pure Python,
    the CPUDispatcher isinstance check will fail.
    """
    from numba.core.registry import CPUDispatcher

    from threadcount.models.fast_models import (
        gaussian1CH_d,
        gaussian2CH_d,
        gaussian3CH_d,
        gaussian4CH_constrained_SII_d,
        gaussian4CH_d,
        gaussian6CH_constrained_HaNII_d,
        gaussian6CH_d,
    )

    njit_fns = [
        gaussian1CH_d,
        gaussian2CH_d,
        gaussian3CH_d,
        gaussian4CH_d,
        gaussian4CH_constrained_SII_d,
        gaussian6CH_d,
        gaussian6CH_constrained_HaNII_d,
    ]
    for fn in njit_fns:
        assert isinstance(fn, CPUDispatcher), (
            f"{fn.__name__} is not a numba CPUDispatcher — @njit is not active"
        )


# ===========================================================================
# 23. Direct tests for _guess_multiline2 / _guess_multiline3  (standard models)
#     and _guess_multiline{2,3,4,6}_d (fast models) — roadmap §1.4a-F
#
# Pattern (from eso120_10x10_thread_runner_fast_nelder_sii.py):
#   model.guess = lambda data, x: _guess_multiline2_d(
#       self=model, data=data, x=x, sigma0=1.2,
#       centers=(-14.36, 0), absolute_centers=True, ...)
# Each class patches the bound method, runs model.guess(), then checks:
#   1. all free parameters are finite
#   2. center offsets match the documented formula
# ===========================================================================

# Single-peak spectrum shared by all six guess-function test classes.
_GUESS_X = np.linspace(6540.0, 6590.0, 200)
_GUESS_Y = _make_1g(_GUESS_X, 10.0, 6563.0, 1.5, c=1.0)


class TestGuessMultiline2:
    """Direct tests for _guess_multiline2 (standard 2-component guess helper).

    g2 is the reference component; g1 is placed relative to g2 by ``centers[0]``.
    Usage pattern: ``model.guess = lambda data, x: _guess_multiline2(self=model, ...)``.
    """

    def test_params_are_finite(self):
        """Patched model.guess() returns finite values for all free parameters."""
        model = tc_models.Const_2GaussModel()
        model.guess = lambda data, x: _guess_multiline2(
            self=model, data=data, x=x, sigma0=1.2, heights=(1, 4), centers=(-2, 0)
        )
        pars = model.guess(_GUESS_Y, _GUESS_X)
        for name, par in pars.items():
            if par.expr is None and par.vary:
                assert np.isfinite(par.value), f"Non-finite guess for '{name}'"

    def test_center_offsets_relative(self):
        """g1_center - g2_center = sigma0*(centers[0]-centers[1]) with absolute_centers=False."""
        model = tc_models.Const_2GaussModel()
        S0, C0, C1 = 1.2, -3.0, 1.0
        model.guess = lambda data, x: _guess_multiline2(
            self=model,
            data=data,
            x=x,
            sigma0=S0,
            centers=(C0, C1),
            absolute_centers=False,
        )
        pars = model.guess(_GUESS_Y, _GUESS_X)
        expected = S0 * (C0 - C1)
        actual = pars["g1_center"].value - pars["g2_center"].value
        assert abs(actual - expected) < 1e-10, (
            f"g1-g2 center offset={actual:.8g}, expected={expected:.8g}"
        )

    def test_center_offsets_absolute(self):
        """g1_center - g2_center = centers[0]-centers[1] with absolute_centers=True.

        This is the SII usage pattern: centers=(-14.36, 0), absolute_centers=True
        places g1 exactly 14.36 Å blueward of g2 regardless of sigma0.
        """
        model = tc_models.Const_2GaussModel()
        OFF0, OFF1 = -14.36, 0.0
        model.guess = lambda data, x: _guess_multiline2(
            self=model,
            data=data,
            x=x,
            centers=(OFF0, OFF1),
            absolute_centers=True,
        )
        pars = model.guess(_GUESS_Y, _GUESS_X)
        expected = OFF0 - OFF1
        actual = pars["g1_center"].value - pars["g2_center"].value
        assert abs(actual - expected) < 1e-10, (
            f"g1-g2 center offset={actual:.8g}, expected={expected:.8g}"
        )

    def test_height_ratio_matches_input(self):
        """g1_height / g2_height = heights[0] / heights[1]."""
        model = tc_models.Const_2GaussModel()
        H0, H1 = 1.0, 4.0
        model.guess = lambda data, x: _guess_multiline2(
            self=model, data=data, x=x, heights=(H0, H1)
        )
        pars = model.guess(_GUESS_Y, _GUESS_X)
        actual_ratio = pars["g1_height"].value / pars["g2_height"].value
        assert abs(actual_ratio - H0 / H1) < 1e-10, (
            f"height ratio={actual_ratio:.6g}, expected={H0 / H1:.6g}"
        )


class TestGuessMultiline3:
    """Direct tests for _guess_multiline3 (standard 3-component guess helper).

    g2 is the reference; g1 and g3 are placed relative to g2 by centers[0] and
    centers[2] respectively.  Usage pattern mirrors TestGuessMultiline2.
    """

    def test_params_are_finite(self):
        """Patched model.guess() returns finite values for all free parameters."""
        model = tc_models.Const_3GaussModel()
        model.guess = lambda data, x: _guess_multiline3(
            self=model, data=data, x=x, sigma0=1.2
        )
        pars = model.guess(_GUESS_Y, _GUESS_X)
        for name, par in pars.items():
            if par.expr is None and par.vary:
                assert np.isfinite(par.value), f"Non-finite guess for '{name}'"

    def test_center_offsets_relative(self):
        """g{1,2,3}_center - g2_center = sigma0*(centers[i]-centers[1]) with absolute_centers=False."""
        model = tc_models.Const_3GaussModel()
        S0, CENTS = 1.2, (-3.0, 0.0, 2.5)
        model.guess = lambda data, x: _guess_multiline3(
            self=model,
            data=data,
            x=x,
            sigma0=S0,
            centers=CENTS,
            absolute_centers=False,
        )
        pars = model.guess(_GUESS_Y, _GUESS_X)
        for i, key in enumerate(("g1_center", "g2_center", "g3_center")):
            expected_rel = S0 * (CENTS[i] - CENTS[1])
            actual_rel = pars[key].value - pars["g2_center"].value
            assert abs(actual_rel - expected_rel) < 1e-10, (
                f"{key} relative offset={actual_rel:.8g}, expected={expected_rel:.8g}"
            )

    def test_center_offsets_absolute(self):
        """g{1,2,3}_center - g2_center = centers[i]-centers[1] with absolute_centers=True."""
        model = tc_models.Const_3GaussModel()
        OFFS = (-14.0, 0.0, 21.0)
        model.guess = lambda data, x: _guess_multiline3(
            self=model,
            data=data,
            x=x,
            centers=OFFS,
            absolute_centers=True,
        )
        pars = model.guess(_GUESS_Y, _GUESS_X)
        for i, key in enumerate(("g1_center", "g2_center", "g3_center")):
            expected_rel = OFFS[i] - OFFS[1]
            actual_rel = pars[key].value - pars["g2_center"].value
            assert abs(actual_rel - expected_rel) < 1e-10, (
                f"{key} relative offset={actual_rel:.8g}, expected={expected_rel:.8g}"
            )

    def test_height_ratios_match_input(self):
        """g{1,2,3}_height / g2_height = heights[i] / heights[1]."""
        model = tc_models.Const_3GaussModel()
        H_RATIO = (1.0, 4.0, 2.0)
        model.guess = lambda data, x: _guess_multiline3(
            self=model, data=data, x=x, heights=H_RATIO
        )
        pars = model.guess(_GUESS_Y, _GUESS_X)
        for i, key in enumerate(("g1_height", "g2_height", "g3_height")):
            actual_ratio = pars[key].value / pars["g2_height"].value
            expected_ratio = H_RATIO[i] / H_RATIO[1]
            assert abs(actual_ratio - expected_ratio) < 1e-10, (
                f"{key} / g2_height={actual_ratio:.6g}, expected={expected_ratio:.6g}"
            )


class TestGuessMultiline2D:
    """Direct tests for _guess_multiline2_d (fast 2-component guess helper).

    The key output parameter is ``deltax = g1_center - g2_center``.
    Usage pattern (from eso120_10x10_thread_runner_fast_nelder_sii.py)::

        model.guess = lambda data, x: _guess_multiline2_d(
            self=model, data=data, x=x,
            sigma0=1.2, centers=(-14.36, 0), absolute_centers=True)
    """

    def test_params_are_finite(self):
        """Patched model.guess() returns finite values for all free parameters."""
        model = tc_models.Const_2GaussModel_fast()
        model.guess = lambda data, x: _guess_multiline2_d(
            self=model, data=data, x=x, sigma0=1.2, heights=(1, 4), centers=(-2, 0)
        )
        pars = model.guess(_GUESS_Y, _GUESS_X)
        for name, par in pars.items():
            if par.expr is None and par.vary:
                assert np.isfinite(par.value), f"Non-finite guess for '{name}'"

    def test_deltax_relative(self):
        """deltax = sigma0*(centers[0]-centers[1]) with absolute_centers=False."""
        model = tc_models.Const_2GaussModel_fast()
        S0, C0, C1 = 1.2, -3.0, 0.0
        model.guess = lambda data, x: _guess_multiline2_d(
            self=model,
            data=data,
            x=x,
            sigma0=S0,
            centers=(C0, C1),
            absolute_centers=False,
        )
        pars = model.guess(_GUESS_Y, _GUESS_X)
        expected = S0 * (C0 - C1)
        assert abs(pars["deltax"].value - expected) < 1e-10, (
            f"deltax={pars['deltax'].value:.8g}, expected={expected:.8g}"
        )

    def test_deltax_absolute(self):
        """deltax = centers[0]-centers[1] with absolute_centers=True.

        Reproduces the SII script pattern: centers=(-14.36, 0), absolute_centers=True
        gives deltax = -14.36 regardless of the guessed peak position.
        """
        model = tc_models.Const_2GaussModel_fast()
        OFF0, OFF1 = -14.36, 0.0
        model.guess = lambda data, x: _guess_multiline2_d(
            self=model,
            data=data,
            x=x,
            centers=(OFF0, OFF1),
            absolute_centers=True,
        )
        pars = model.guess(_GUESS_Y, _GUESS_X)
        expected = OFF0 - OFF1
        assert abs(pars["deltax"].value - expected) < 1e-10, (
            f"deltax={pars['deltax'].value:.8g}, expected={expected:.8g}"
        )

    def test_height_ratio_matches_input(self):
        """g1_height / g2_height = heights[0] / heights[1]."""
        model = tc_models.Const_2GaussModel_fast()
        H0, H1 = 1.0, 4.0
        model.guess = lambda data, x: _guess_multiline2_d(
            self=model, data=data, x=x, heights=(H0, H1)
        )
        pars = model.guess(_GUESS_Y, _GUESS_X)
        actual_ratio = pars["g1_height"].value / pars["g2_height"].value
        assert abs(actual_ratio - H0 / H1) < 1e-10, (
            f"height ratio={actual_ratio:.6g}, expected={H0 / H1:.6g}"
        )


class TestGuessMultiline3D:
    """Direct tests for _guess_multiline3_d (fast 3-component guess helper).

    Outputs: ``deltax = g1_center - g2_center``,
             ``deltaxhi = g3_center - g2_center``.
    """

    def test_params_are_finite(self):
        """Patched model.guess() returns finite values for all free parameters."""
        model = tc_models.Const_3GaussModel_fast()
        model.guess = lambda data, x: _guess_multiline3_d(
            self=model, data=data, x=x, sigma0=1.2
        )
        pars = model.guess(_GUESS_Y, _GUESS_X)
        for name, par in pars.items():
            if par.expr is None and par.vary:
                assert np.isfinite(par.value), f"Non-finite guess for '{name}'"

    def test_deltax_relative(self):
        """deltax and deltaxhi encode offsets from g2 with absolute_centers=False.

        deltax   = sigma0*(centers[0]-centers[1])
        deltaxhi = sigma0*(centers[2]-centers[1])
        """
        model = tc_models.Const_3GaussModel_fast()
        S0, CENTS = 1.2, (-3.0, 0.0, 2.5)
        model.guess = lambda data, x: _guess_multiline3_d(
            self=model,
            data=data,
            x=x,
            sigma0=S0,
            centers=CENTS,
            absolute_centers=False,
        )
        pars = model.guess(_GUESS_Y, _GUESS_X)
        assert abs(pars["deltax"].value - S0 * (CENTS[0] - CENTS[1])) < 1e-10, (
            f"deltax={pars['deltax'].value:.8g}, expected={S0 * (CENTS[0] - CENTS[1]):.8g}"
        )
        assert abs(pars["deltaxhi"].value - S0 * (CENTS[2] - CENTS[1])) < 1e-10, (
            f"deltaxhi={pars['deltaxhi'].value:.8g}, expected={S0 * (CENTS[2] - CENTS[1]):.8g}"
        )

    def test_deltax_absolute(self):
        """deltax = centers[0]-centers[1] and deltaxhi = centers[2]-centers[1] with absolute_centers=True."""
        model = tc_models.Const_3GaussModel_fast()
        OFFS = (-14.0, 0.0, 21.0)
        model.guess = lambda data, x: _guess_multiline3_d(
            self=model,
            data=data,
            x=x,
            centers=OFFS,
            absolute_centers=True,
        )
        pars = model.guess(_GUESS_Y, _GUESS_X)
        assert abs(pars["deltax"].value - (OFFS[0] - OFFS[1])) < 1e-10, (
            f"deltax={pars['deltax'].value:.8g}, expected={OFFS[0] - OFFS[1]:.8g}"
        )
        assert abs(pars["deltaxhi"].value - (OFFS[2] - OFFS[1])) < 1e-10, (
            f"deltaxhi={pars['deltaxhi'].value:.8g}, expected={OFFS[2] - OFFS[1]:.8g}"
        )


class TestGuessMultiline4D:
    """Direct tests for _guess_multiline4_d (fast 4-component guess helper).

    g4 is the reference; deltax{1,2,3} = g{1,2,3}_center - g4_center.
    """

    def test_params_are_finite(self):
        """Patched model.guess() returns finite values for all free parameters."""
        model = tc_models.Const_4GaussModel_fast()
        model.guess = lambda data, x: _guess_multiline4_d(
            self=model, data=data, x=x, sigma0=1.5
        )
        pars = model.guess(_GUESS_Y, _GUESS_X)
        for name, par in pars.items():
            if par.expr is None and par.vary:
                assert np.isfinite(par.value), f"Non-finite guess for '{name}'"

    def test_deltax_relative(self):
        """deltax{1,2,3} = sigma0*(centers[{0,1,2}]-centers[3]) with absolute_centers=False.

        The guessed center cancels out of all three deltax values, so the
        assertion is exact regardless of what guess_from_peak returns.
        """
        model = tc_models.Const_4GaussModel_fast()
        S0 = 1.5
        CENTS = (-2.0, -1.0, 1.0, 2.0)  # g4 is at centers[3]
        model.guess = lambda data, x: _guess_multiline4_d(
            self=model,
            data=data,
            x=x,
            sigma0=S0,
            centers=CENTS,
            absolute_centers=False,
        )
        pars = model.guess(_GUESS_Y, _GUESS_X)
        for i, key in enumerate(("deltax1", "deltax2", "deltax3")):
            expected = S0 * (CENTS[i] - CENTS[3])
            assert abs(pars[key].value - expected) < 1e-10, (
                f"{key}={pars[key].value:.8g}, expected={expected:.8g}"
            )

    def test_deltax_absolute(self):
        """deltax{1,2,3} = centers[{0,1,2}]-centers[3] with absolute_centers=True."""
        model = tc_models.Const_4GaussModel_fast()
        OFFS = (-9.0, -6.0, -3.0, 0.0)  # g4 at centers[3]
        model.guess = lambda data, x: _guess_multiline4_d(
            self=model,
            data=data,
            x=x,
            centers=OFFS,
            absolute_centers=True,
        )
        pars = model.guess(_GUESS_Y, _GUESS_X)
        for i, key in enumerate(("deltax1", "deltax2", "deltax3")):
            expected = OFFS[i] - OFFS[3]
            assert abs(pars[key].value - expected) < 1e-10, (
                f"{key}={pars[key].value:.8g}, expected={expected:.8g}"
            )


class TestGuessMultiline6D:
    """Direct tests for _guess_multiline6_d (fast 6-component guess helper).

    g4 is the reference; deltax{1,2,3,5,6} = g{1,2,3,5,6}_center - g4_center.
    (There is no deltax4 — g4_center is the free reference parameter.)
    """

    def test_params_are_finite(self):
        """Patched model.guess() returns finite values for all free parameters."""
        model = tc_models.Const_6GaussModel_fast()
        model.guess = lambda data, x: _guess_multiline6_d(
            self=model, data=data, x=x, sigma0=1.5
        )
        pars = model.guess(_GUESS_Y, _GUESS_X)
        for name, par in pars.items():
            if par.expr is None and par.vary:
                assert np.isfinite(par.value), f"Non-finite guess for '{name}'"

    def test_deltax_relative(self):
        """deltax{1,2,3,5,6} = sigma0*(centers[{0,1,2,4,5}]-centers[3]) with absolute_centers=False."""
        model = tc_models.Const_6GaussModel_fast()
        S0 = 1.5
        CENTS = (-2.0, -1.0, 1.0, 2.0, 5.0, 6.0)  # g4 is at centers[3]
        model.guess = lambda data, x: _guess_multiline6_d(
            self=model,
            data=data,
            x=x,
            sigma0=S0,
            centers=CENTS,
            absolute_centers=False,
        )
        pars = model.guess(_GUESS_Y, _GUESS_X)
        for c_idx, key in (
            (0, "deltax1"),
            (1, "deltax2"),
            (2, "deltax3"),
            (4, "deltax5"),
            (5, "deltax6"),
        ):
            expected = S0 * (CENTS[c_idx] - CENTS[3])
            assert abs(pars[key].value - expected) < 1e-10, (
                f"{key}={pars[key].value:.8g}, expected={expected:.8g}"
            )

    def test_deltax_absolute(self):
        """deltax{1,2,3,5,6} = centers[{0,1,2,4,5}]-centers[3] with absolute_centers=True."""
        model = tc_models.Const_6GaussModel_fast()
        OFFS = (-8.0, -5.0, -2.0, 0.0, 3.0, 7.0)  # g4 at centers[3]
        model.guess = lambda data, x: _guess_multiline6_d(
            self=model,
            data=data,
            x=x,
            centers=OFFS,
            absolute_centers=True,
        )
        pars = model.guess(_GUESS_Y, _GUESS_X)
        for c_idx, key in (
            (0, "deltax1"),
            (1, "deltax2"),
            (2, "deltax3"),
            (4, "deltax5"),
            (5, "deltax6"),
        ):
            expected = OFFS[c_idx] - OFFS[3]
            assert abs(pars[key].value - expected) < 1e-10, (
                f"{key}={pars[key].value:.8g}, expected={expected:.8g}"
            )
