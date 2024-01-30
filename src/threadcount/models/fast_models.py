"""Custom Models that should function just like lmfit's models."""
import numpy as np

from numba import njit
import lmfit

from .basic import guess_from_peak, mean_edges

tiny = lmfit.models.tiny  # 1.0e-15

__all__ = [
    "Const_2GaussModel_fast",
    "Const_3GaussModel_fast",
    "gaussian2CH_d",
    "gaussian3CH_d",
    "_guess_2gauss_d",
    "_guess_3gauss_d",
    "_guess_multiline2_d",
]


def fwhm_expr_fast(model, comp_pre):
    """Return constraint expression for fwhm of one component."""
    fmt = "{factor:.7f}*{prefix:s}sigma"
    return fmt.format(factor=model.fwhm_factor, prefix=model.prefix + comp_pre)


def flux_expr_fast(model, comp_pre):
    """Return constraint expression for line flux of one component."""
    fmt = "{factor:.7f}*{prefix:s}height*{prefix:s}sigma"
    return fmt.format(factor=model.flux_factor, prefix=model.prefix + comp_pre)


@njit
def gaussian2CH_d(
    x,
    g1_height=1.0,
    deltax=0.0,
    g1_sigma=1.0,
    g2_height=1.0,
    g2_center=0.0,
    g2_sigma=1.0,
    c=0.0,
):
    """Return a 2-Gaussian function in 1-dimension."""
    f = (
        g1_height * np.exp(-((1.0 * x - g2_center - deltax) ** 2) / max(tiny, (2 * g1_sigma**2)))
        + g2_height * np.exp(-((1.0 * x - g2_center) ** 2) / max(tiny, (2 * g2_sigma**2)))
        + c
    )
    return f


@njit
def gaussian3CH_d(
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
    """Return a 3-Gaussian function in 1-dimension."""
    f = (
        g1_height * np.exp(-((1.0 * x - g2_center - deltax) ** 2) / max(tiny, (2 * g1_sigma**2)))
        + g2_height * np.exp(-((1.0 * x - g2_center) ** 2) / max(tiny, (2 * g2_sigma**2)))
        + g3_height
        * np.exp(-((1.0 * x - g2_center - deltaxhi) ** 2) / max(tiny, (2 * g3_sigma**2)))
        + c
    )
    return f


def _guess_2gauss_d(
    self,
    data,
    x,
    sigma0=None,
    heights=(1, 4),
    sigma_factors=(1, 1),
    centers=(-2, 0),
    absolute_centers=False,
    **kwargs
):
    """Estimate initial model parameter values from data.
    Used for fast model.
    """
    height, center, sigma = guess_from_peak(data, x)
    constant = mean_edges(data, edge_fraction=0.1)

    # fill in any missing function parameters based on 1gauss guess:
    if sigma0 is None:
        sigma0 = sigma

    # calculate g2 guesses based off 1gauss guess and function parameters
    g1_sigma, g2_sigma = sigma0 * np.array(sigma_factors)
    if absolute_centers:
        g1_center, g2_center = center + np.array(centers)
    else:
        g1_center, g2_center = center + sigma0 * np.array(centers)

    deltax = g1_center - g2_center

    a = sigma / (heights[0] * g1_sigma + heights[1] * g2_sigma)
    g1_height, g2_height = a * height * np.array(heights)

    pars = self.make_params(
        deltax=deltax,
        g1_height=g1_height,
        g1_sigma=g1_sigma,
        g2_height=g2_height,
        g2_center=g2_center,
        g2_sigma=g2_sigma,
        c=constant,
    )

    return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


def _guess_3gauss_d(
    self,
    data,
    x,
    sigma0=None,
    heights=(1, 4, 1),
    sigma_factors=(1, 1, 1),
    centers=(-1, 0, 1),
    absolute_centers=False,
    **kwargs
):
    """Estimate initial model parameter values from data.
    Used for fast model.
    """
    height, center, sigma = guess_from_peak(data, x)
    constant = mean_edges(data, edge_fraction=0.1)

    # fill in any missing function parameters based on 1gauss guess:
    if sigma0 is None:
        sigma0 = sigma

    # calculate component guesses based off 1gauss guess and function parameters
    g1_sigma, g2_sigma, g3_sigma = sigma0 * np.array(sigma_factors)
    if absolute_centers:
        g1_center, g2_center, g3_center = center + np.array(centers)
    else:
        g1_center, g2_center, g3_center = center + sigma0 * np.array(centers)

    deltax = g1_center - g2_center
    deltaxhi = g3_center - g2_center

    a = sigma / (heights[0] * g1_sigma + heights[1] * g2_sigma + heights[2] * g3_sigma)
    g1_height, g2_height, g3_height = a * height * np.array(heights)

    pars = self.make_params(
        deltax=deltax,
        deltaxhi=deltaxhi,
        g1_height=g1_height,
        g1_sigma=g1_sigma,
        g2_height=g2_height,
        g2_center=g2_center,
        g2_sigma=g2_sigma,
        g3_height=g3_height,
        g3_sigma=g3_sigma,
        c=constant,
    )

    return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


class Const_2GaussModel_fast(lmfit.Model):
    """The fast evaluation version of Const_2GaussModel.
    It is created using lmfit.Model instead of CompositeModel.
    """

    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    """float: Factor used to create :func:`lmfit.models.fwhm_expr`."""
    flux_factor = np.sqrt(2 * np.pi)
    """float: Factor used to create :func:`flux_expr`."""

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):  # noqa
        kwargs.update(
            {"prefix": prefix, "nan_policy": nan_policy, "independent_vars": independent_vars}
        )
        super().__init__(gaussian2CH_d, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        comp_pre = ["g1_", "g2_"]
        for comp in comp_pre:
            self.set_param_hint(comp + "sigma", min=0)
            self.set_param_hint(comp + "height", min=0)
            self.set_param_hint(comp + "fwhm", expr=fwhm_expr_fast(self, comp))
            self.set_param_hint(comp + "flux", expr=flux_expr_fast(self, comp))
        self.set_param_hint("g1_center", expr="g2_center+deltax")

    def _reprstring(self, long=False):
        return "constant + 2 gaussians (fast)"

    guess = _guess_2gauss_d

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC


class Const_3GaussModel_fast(lmfit.Model):
    """The fast evaluation version of Const_3GaussModel.
    It is created using lmfit.Model instead of CompositeModel.
    """

    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    """float: Factor used to create :func:`lmfit.models.fwhm_expr`."""
    flux_factor = np.sqrt(2 * np.pi)
    """float: Factor used to create :func:`flux_expr`."""

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):  # noqa
        kwargs.update(
            {
                "prefix": prefix,
                "nan_policy": nan_policy,
                "independent_vars": independent_vars,
            }
        )
        super().__init__(gaussian3CH_d, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        comp_pre = ["g1_", "g2_", "g3_"]
        for comp in comp_pre:
            self.set_param_hint(comp + "sigma", min=0)
            self.set_param_hint(comp + "height", min=0)
            self.set_param_hint(comp + "fwhm", expr=fwhm_expr_fast(self, comp))
            self.set_param_hint(comp + "flux", expr=flux_expr_fast(self, comp))
        self.set_param_hint("g1_center", expr="g2_center+deltax")
        self.set_param_hint("g3_center", expr="g2_center+deltaxhi")

    guess = _guess_3gauss_d

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC


def _guess_multiline2_d(
    self,
    data,
    x,
    sigma0=None,
    heights=(1, 4),
    sigma_factors=(1, 1),
    centers=(-1, 0),
    absolute_centers=False,
    focus_lam=None,
    **kwargs
):
    """Estimate initial model parameter values from data.
    Used for fast models.
    """
    if focus_lam is None:
        focus_lam = [np.min(x), np.max(x)]
        focus_lam[0] = self.param_hints["g2_center"].get("min", focus_lam[0])
        focus_lam[1] = self.param_hints["g2_center"].get("max", focus_lam[1])

    focus_index = (x > focus_lam[0]) & (x < focus_lam[1])

    height, center, sigma = guess_from_peak(data[focus_index], x[focus_index])
    constant = mean_edges(data, edge_fraction=0.1)

    # fill in any missing function parameters based on 1gauss guess:
    if sigma0 is None:
        sigma0 = sigma

    # calculate component guesses based off 1gauss guess and function parameters
    g1_sigma, g2_sigma = sigma0 * np.array(sigma_factors)
    if absolute_centers:
        g1_center, g2_center = center + np.array(centers)
    else:
        g1_center, g2_center = center + sigma0 * np.array(centers)

    deltax = g1_center - g2_center

    # The factor "a" is not for a guess based on a single line
    # a = sigma / (heights[0] * g1_sigma + heights[1] * g2_sigma + heights[2] * g3_sigma)
    g1_height, g2_height = height * np.array(heights)

    pars = self.make_params(
        deltax=deltax,
        g1_height=g1_height,
        g1_sigma=g1_sigma,
        g2_height=g2_height,
        g2_center=g2_center,
        g2_sigma=g2_sigma,
        c=constant,
    )

    return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)
