"""Custom Models that should function just like lmfit's models."""

import numpy as np

from numba import njit
import lmfit

from .basic import guess_from_peak, mean_edges
from .models import _guess_1gauss

tiny = lmfit.models.tiny  # 1.0e-15

__all__ = [
    "Const_1GaussModel_fast",
    "Const_2GaussModel_fast",
    "Const_3GaussModel_fast",
    "Const_4GaussModel_fast",
    "Const_4GaussModel_constrained_SII_fast",
    "Const_6GaussModel_fast",
    "Const_6GaussModel_constrained_HaNII_fast",
    "gaussian1CH_d",
    "gaussian2CH_d",
    "gaussian3CH_d",
    "gaussian4CH_d",
    "gaussian4CH_constrained_SII_d",
    "gaussian6CH_d",
    "gaussian6CH_constrained_HaNII_d",
    "_guess_1gauss_d",
    "_guess_2gauss_d",
    "_guess_3gauss_d",
    "_guess_4gauss_d",
    "_guess_6gauss_d",
    "_guess_multiline2_d",
    "_guess_multiline3_d",
    "_guess_multiline4_d",
    "_guess_multiline4_constrained_d",
    "_guess_multiline6_d",
    "_guess_multiline6_constrained_d",
]

gaussian4CH_constrained_SII_d_DELTAX24 = -14.37
gaussian6CH_constrained_HaNII_d_DELTAX24 = -14.769
gaussian6CH_constrained_HaNII_d_DELTAX64 = 20.64


def fwhm_expr_fast(model, comp_pre):
    """Return constraint expression for fwhm of one component."""
    fmt = "{factor:.7f}*{prefix:s}sigma"
    return fmt.format(factor=model.fwhm_factor, prefix=model.prefix + comp_pre)


def flux_expr_fast(model, comp_pre):
    """Return constraint expression for line flux of one component."""
    fmt = "{factor:.7f}*{prefix:s}height*{prefix:s}sigma"
    return fmt.format(factor=model.flux_factor, prefix=model.prefix + comp_pre)


@njit
def gaussian1CH_d(
    x,
    g1_height=1.0,
    g1_center=0.0,
    g1_sigma=1.0,
    c=0.0,
):
    """Return a 1-Gaussian function in 1-dimension."""
    f = (
        g1_height * np.exp(-((1.0 * x - g1_center) ** 2) / max(tiny, (2 * g1_sigma**2)))
        + c
    )
    return f


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
        g1_height
        * np.exp(-((1.0 * x - g2_center - deltax) ** 2) / max(tiny, (2 * g1_sigma**2)))
        + g2_height
        * np.exp(-((1.0 * x - g2_center) ** 2) / max(tiny, (2 * g2_sigma**2)))
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


@njit
def gaussian4CH_d(
    x,
    g1_height=1.0,
    deltax1=0.0,
    g1_sigma=1.0,
    g2_height=1.0,
    deltax2=0.0,
    g2_sigma=1.0,
    g3_height=1.0,
    deltax3=0.0,
    g3_sigma=1.0,
    g4_height=1.0,
    g4_center=0.0,
    g4_sigma=1.0,
    c=0.0,
):
    """Return a 4-Gaussian function in 1-dimension."""
    f = (
        g1_height
        * np.exp(
            -((1.0 * x - (g4_center + deltax1)) ** 2) / max(tiny, (2 * g1_sigma**2))
        )
        + g2_height
        * np.exp(
            -((1.0 * x - (g4_center + deltax2)) ** 2) / max(tiny, (2 * g2_sigma**2))
        )
        + g3_height
        * np.exp(
            -((1.0 * x - (g4_center + deltax3)) ** 2) / max(tiny, (2 * g3_sigma**2))
        )
        + g4_height
        * np.exp(-((1.0 * x - g4_center) ** 2) / max(tiny, (2 * g4_sigma**2)))
        + c
    )
    return f


@njit
def gaussian4CH_constrained_SII_d(
    x,
    g1_height=1.0,
    deltax12=0.0,
    g1_sigma=1.0,
    g2_height=1.0,
    g2_sigma=1.0,
    g3_height=1.0,
    deltax34=0.0,
    g3_sigma=1.0,
    g4_height=1.0,
    g4_center=0.0,
    g4_sigma=1.0,
    c=0.0,
):
    """Return a 4-Gaussian function in 1-dimension.
    motivating idea: difference between center 2 and 4 is constrained by physics
    limits can be put to constrain delta 1 and 2, and delta 3 and 4"""
    deltax24 = gaussian4CH_constrained_SII_d_DELTAX24
    f = (
        g1_height
        * np.exp(
            -((1.0 * x - (g4_center + deltax24 + deltax12)) ** 2)
            / max(tiny, (2 * g1_sigma**2))
        )
        + g2_height
        * np.exp(
            -((1.0 * x - (g4_center + deltax24)) ** 2) / max(tiny, (2 * g2_sigma**2))
        )
        + g3_height
        * np.exp(
            -((1.0 * x - (g4_center + deltax34)) ** 2) / max(tiny, (2 * g3_sigma**2))
        )
        + g4_height
        * np.exp(-((1.0 * x - g4_center) ** 2) / max(tiny, (2 * g4_sigma**2)))
        + c
    )
    return f


@njit
def gaussian6CH_d(
    x,
    g1_height=1.0,
    deltax1=0.0,
    g1_sigma=1.0,
    g2_height=1.0,
    deltax2=0.0,
    g2_sigma=1.0,
    g3_height=1.0,
    deltax3=0.0,
    g3_sigma=1.0,
    g4_height=1.0,
    g4_center=0.0,
    g4_sigma=1.0,
    g5_height=1.0,
    deltax5=0.0,
    g5_sigma=1.0,
    g6_height=1.0,
    deltax6=0.0,
    g6_sigma=1.0,
    c=0.0,
):
    """Return a 6-Gaussian function in 1-dimension."""
    f = (
        g1_height
        * np.exp(
            -((1.0 * x - (g4_center + deltax1)) ** 2) / max(tiny, (2 * g1_sigma**2))
        )
        + g2_height
        * np.exp(
            -((1.0 * x - (g4_center + deltax2)) ** 2) / max(tiny, (2 * g2_sigma**2))
        )
        + g3_height
        * np.exp(
            -((1.0 * x - (g4_center + deltax3)) ** 2) / max(tiny, (2 * g3_sigma**2))
        )
        + g4_height
        * np.exp(-((1.0 * x - g4_center) ** 2) / max(tiny, (2 * g4_sigma**2)))
        + g5_height
        * np.exp(
            -((1.0 * x - (g4_center + deltax5)) ** 2) / max(tiny, (2 * g5_sigma**2))
        )
        + g6_height
        * np.exp(
            -((1.0 * x - (g4_center + deltax6)) ** 2) / max(tiny, (2 * g6_sigma**2))
        )
        + c
    )
    return f


@njit
def gaussian6CH_constrained_HaNII_d(
    x,
    g1_h_factor=1.0,
    deltax12=0.0,
    g1_sigma=1.0,
    g2_height=1.0,
    g2_sigma=1.0,
    g3_h_factor=1.0,
    deltax34=0.0,
    g3_sigma=1.0,
    g4_height=1.0,
    g4_center=0.0,
    g4_sigma=1.0,
    g5_h_factor=1.0,
    deltax56=0.0,
    g5_sigma=1.0,
    g6_height=1.0,
    g6_sigma=1.0,
    c=0.0,
):
    """Return a 6-Gaussian function in 1-dimension."""
    deltax24 = gaussian6CH_constrained_HaNII_d_DELTAX24
    deltax64 = gaussian6CH_constrained_HaNII_d_DELTAX64
    f = (
        g2_height
        * g1_h_factor
        * np.exp(
            -((1.0 * x - (g4_center + deltax12 + deltax24)) ** 2)
            / max(tiny, (2 * g1_sigma**2))
        )
        + g2_height
        * np.exp(
            -((1.0 * x - (g4_center + deltax24)) ** 2) / max(tiny, (2 * g2_sigma**2))
        )
        + g4_height
        * g3_h_factor
        * np.exp(
            -((1.0 * x - (g4_center + deltax34)) ** 2) / max(tiny, (2 * g3_sigma**2))
        )
        + g4_height
        * np.exp(-((1.0 * x - g4_center) ** 2) / max(tiny, (2 * g4_sigma**2)))
        + g6_height
        * g5_h_factor
        * np.exp(
            -((1.0 * x - (g4_center + deltax56 + deltax64)) ** 2)
            / max(tiny, (2 * g5_sigma**2))
        )
        + g6_height
        * np.exp(
            -((1.0 * x - (g4_center + deltax64)) ** 2) / max(tiny, (2 * g6_sigma**2))
        )
        + c
    )
    return f


_guess_1gauss_d = _guess_1gauss


def _guess_2gauss_d(
    self,
    data,
    x,
    sigma0=None,
    heights=(1, 4),
    sigma_factors=(1, 1),
    centers=(-2, 0),
    absolute_centers=False,
    **kwargs,
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
    **kwargs,
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


def _guess_4gauss_d(
    self,
    data,
    x,
    sigma0=None,
    heights=(1, 1, 4, 4),
    sigma_factors=(1, 1, 1, 1),
    centers=(-2, -1, 1, 2),
    absolute_centers=False,
    **kwargs,
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
    g1_sigma, g2_sigma, g3_sigma, g4_sigma = sigma0 * np.array(sigma_factors)
    if absolute_centers:
        g1_center, g2_center, g3_center, g4_center = center + np.array(centers)
    else:
        g1_center, g2_center, g3_center, g4_center = center + sigma0 * np.array(centers)

    deltax1 = g1_center - g4_center
    deltax2 = g2_center - g4_center
    deltax3 = g3_center - g4_center

    a = sigma / (
        heights[0] * g1_sigma
        + heights[1] * g2_sigma
        + heights[2] * g3_sigma
        + heights[3] * g4_sigma
    )
    g1_height, g2_height, g3_height, g4_height = a * height * np.array(heights)

    pars = self.make_params(
        deltax1=deltax1,
        deltax2=deltax2,
        deltax3=deltax3,
        g4_center=g4_center,
        g1_height=g1_height,
        g1_sigma=g1_sigma,
        g2_height=g2_height,
        g2_sigma=g2_sigma,
        g3_height=g3_height,
        g3_sigma=g3_sigma,
        g4_height=g4_height,
        g4_sigma=g4_sigma,
        c=constant,
    )

    return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


def _guess_6gauss_d(
    self,
    data,
    x,
    sigma0=None,
    heights=(1, 1, 4, 4, 1, 1),
    sigma_factors=(1, 1, 1, 1, 1, 1),
    centers=(-2, -1, 1, 2, 5, 6),
    absolute_centers=False,
    **kwargs,
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
    g1_sigma, g2_sigma, g3_sigma, g4_sigma, g5_sigma, g6_sigma = sigma0 * np.array(
        sigma_factors
    )
    if absolute_centers:
        g1_center, g2_center, g3_center, g4_center, g5_center, g6_center = (
            center + np.array(centers)
        )
    else:
        g1_center, g2_center, g3_center, g4_center, g5_center, g6_center = (
            center + sigma0 * np.array(centers)
        )

    deltax1 = g1_center - g4_center
    deltax2 = g2_center - g4_center
    deltax3 = g3_center - g4_center
    deltax5 = g5_center - g4_center
    deltax6 = g6_center - g4_center

    a = sigma / (
        heights[0] * g1_sigma
        + heights[1] * g2_sigma
        + heights[2] * g3_sigma
        + heights[3] * g4_sigma
        + heights[4] * g5_sigma
        + heights[5] * g6_sigma
    )
    g1_height, g2_height, g3_height, g4_height, g5_height, g6_height = (
        a * height * np.array(heights)
    )

    pars = self.make_params(
        deltax1=deltax1,
        deltax2=deltax2,
        deltax3=deltax3,
        g4_center=g4_center,
        deltax5=deltax5,
        deltax6=deltax6,
        g1_height=g1_height,
        g1_sigma=g1_sigma,
        g2_height=g2_height,
        g2_sigma=g2_sigma,
        g3_height=g3_height,
        g3_sigma=g3_sigma,
        g4_height=g4_height,
        g4_sigma=g4_sigma,
        g5_height=g5_height,
        g5_sigma=g5_sigma,
        g6_height=g6_height,
        g6_sigma=g6_sigma,
        c=constant,
    )

    return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


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
    **kwargs,
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


def _guess_multiline3_d(
    self,
    data,
    x,
    sigma0=None,
    heights=(1, 4, 1),
    sigma_factors=(1, 1, 1),
    centers=(-1, 0, 1),
    absolute_centers=False,
    focus_lam=None,
    **kwargs,
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
    g1_sigma, g2_sigma, g3_sigma = sigma0 * np.array(sigma_factors)
    if absolute_centers:
        g1_center, g2_center, g3_center = center + np.array(centers)
    else:
        g1_center, g2_center, g3_center = center + sigma0 * np.array(centers)

    deltax = g1_center - g2_center
    deltaxhi = g3_center - g2_center

    # The factor "a" is not for a guess based on a single line
    # a = sigma / (heights[0] * g1_sigma + heights[1] * g2_sigma + heights[2] * g3_sigma)
    g1_height, g2_height, g3_height = height * np.array(heights)

    pars = self.make_params(
        g1_height=g1_height,
        deltax=deltax,
        g1_sigma=g1_sigma,
        g2_height=g2_height,
        g2_center=g2_center,
        g2_sigma=g2_sigma,
        g3_height=g3_height,
        deltaxhi=deltaxhi,
        g3_sigma=g3_sigma,
        c=constant,
    )

    return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


def _guess_multiline4_d(
    self,
    data,
    x,
    sigma0=None,
    heights=(1, 1, 4, 4),
    sigma_factors=(1, 1, 1, 1),
    centers=(-2, -1, 1, 2),
    focus_lam=None,
    absolute_centers=False,
    **kwargs,
):
    """Estimate initial model parameter values from data.
    Used for fast models.
    """
    if focus_lam is None:
        focus_lam = [np.min(x), np.max(x)]
        focus_lam[0] = self.param_hints["g4_center"].get("min", focus_lam[0])
        focus_lam[1] = self.param_hints["g4_center"].get("max", focus_lam[1])

    focus_index = (x > focus_lam[0]) & (x < focus_lam[1])

    height, center, sigma = guess_from_peak(data[focus_index], x[focus_index])
    constant = mean_edges(data, edge_fraction=0.1)

    # fill in any missing function parameters based on 1gauss guess:
    if sigma0 is None:
        sigma0 = sigma

    # calculate component guesses based off 1gauss guess and function parameters
    g1_sigma, g2_sigma, g3_sigma, g4_sigma = sigma0 * np.array(sigma_factors)
    if absolute_centers:
        g1_center, g2_center, g3_center, g4_center = center + np.array(centers)
    else:
        g1_center, g2_center, g3_center, g4_center = center + sigma0 * np.array(centers)

    deltax1 = g1_center - g4_center
    deltax2 = g2_center - g4_center
    deltax3 = g3_center - g4_center

    g1_height, g2_height, g3_height, g4_height = height * np.array(heights)

    pars = self.make_params(
        deltax1=deltax1,
        deltax2=deltax2,
        deltax3=deltax3,
        g4_center=g4_center,
        g1_height=g1_height,
        g1_sigma=g1_sigma,
        g2_height=g2_height,
        g2_sigma=g2_sigma,
        g3_height=g3_height,
        g3_sigma=g3_sigma,
        g4_height=g4_height,
        g4_sigma=g4_sigma,
        c=constant,
    )

    return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


def _guess_multiline6_d(
    self,
    data,
    x,
    sigma0=None,
    heights=(1, 1, 4, 4, 1, 1),
    sigma_factors=(1, 1, 1, 1, 1, 1),
    centers=(-2, -1, 1, 2, 5, 6),
    focus_lam=None,
    absolute_centers=False,
    **kwargs,
):
    """Estimate initial model parameter values from data.
    Used for fast models.
    """
    if focus_lam is None:
        focus_lam = [np.min(x), np.max(x)]
        focus_lam[0] = self.param_hints["g4_center"].get("min", focus_lam[0])
        focus_lam[1] = self.param_hints["g4_center"].get("max", focus_lam[1])

    focus_index = (x > focus_lam[0]) & (x < focus_lam[1])

    height, center, sigma = guess_from_peak(data[focus_index], x[focus_index])
    constant = mean_edges(data, edge_fraction=0.1)

    # fill in any missing function parameters based on 1gauss guess:
    if sigma0 is None:
        sigma0 = sigma

    # calculate component guesses based off 1gauss guess and function parameters
    g1_sigma, g2_sigma, g3_sigma, g4_sigma, g5_sigma, g6_sigma = sigma0 * np.array(
        sigma_factors
    )
    if absolute_centers:
        g1_center, g2_center, g3_center, g4_center, g5_center, g6_center = (
            center + np.array(centers)
        )
    else:
        g1_center, g2_center, g3_center, g4_center, g5_center, g6_center = (
            center + sigma0 * np.array(centers)
        )

    deltax1 = g1_center - g4_center
    deltax2 = g2_center - g4_center
    deltax3 = g3_center - g4_center
    deltax5 = g5_center - g4_center
    deltax6 = g6_center - g4_center

    g1_height, g2_height, g3_height, g4_height, g5_height, g6_height = (
        height * np.array(heights)
    )

    pars = self.make_params(
        deltax1=deltax1,
        deltax2=deltax2,
        deltax3=deltax3,
        g4_center=g4_center,
        deltax5=deltax5,
        deltax6=deltax6,
        g1_height=g1_height,
        g1_sigma=g1_sigma,
        g2_height=g2_height,
        g2_sigma=g2_sigma,
        g3_height=g3_height,
        g3_sigma=g3_sigma,
        g4_height=g4_height,
        g4_sigma=g4_sigma,
        g5_height=g5_height,
        g5_sigma=g5_sigma,
        g6_height=g6_height,
        g6_sigma=g6_sigma,
        c=constant,
    )

    return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


def _guess_multiline4_constrained_d(
    self,
    data,
    x,
    heights=(0.5, 1, 0.5, 1),
    sigma0=1.2,
    sigma_factors=(1, 1, 1, 1),
    deltax12=-5,
    deltax34=-5,
    deltax4=10,
    focus_lam=None,
    **kwargs,
):
    """Estimate initial model parameter values from data.
    Used for fast models.
    guess c from edges of x window
    guesses g4 center and height from focus_lam range or from g4_center min/max
    If sigma0 is none then sigma0 is set for g4 estimated sigma
    hard coded in deltax guesses
    deltax4 is difference between guess from peak and desired position.
    """
    if focus_lam is None:
        focus_lam = [np.min(x), np.max(x)]
        focus_lam[0] = self.param_hints["g4_center"].get("min", focus_lam[0])
        focus_lam[1] = self.param_hints["g4_center"].get("max", focus_lam[1])

    focus_index = (x > focus_lam[0]) & (x < focus_lam[1])

    height, center, sigma = guess_from_peak(data[focus_index], x[focus_index])
    constant = mean_edges(data, edge_fraction=0.1)

    # fill in any missing function parameters based on 1gauss guess:
    if sigma0 is None:
        sigma0 = sigma

    # calculate component guesses based off 1gauss guess and function parameters
    g1_sigma, g2_sigma, g3_sigma, g4_sigma = sigma0 * np.array(sigma_factors)

    g1_height, g2_height, g3_height, g4_height = height * np.array(heights)

    g4_center = center + deltax4

    pars = self.make_params(
        deltax12=deltax12,
        deltax34=deltax34,
        g4_center=g4_center,
        g1_height=g1_height,
        g1_sigma=g1_sigma,
        g2_height=g2_height,
        g2_sigma=g2_sigma,
        g3_height=g3_height,
        g3_sigma=g3_sigma,
        g4_height=g4_height,
        g4_sigma=g4_sigma,
        c=constant,
    )

    return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


def _guess_multiline6_constrained_d(
    self,
    data,
    x,
    heights=(0.5, 1, 0.5, 1, 0.5, 1),
    sigma0=1.2,
    sigma_factors=(1, 1, 1, 1, 1, 1),
    deltax12=-5,
    deltax34=-5,
    deltax56=-5,
    deltax4=0,
    focus_lam=None,
    **kwargs,
):
    """Estimate initial model parameter values from data.
    Used for fast models.
    guess c from edges of x window
    guesses g4 center and height from focus_lam range or from g4_center min/max
    If sigma0 is none then sigma0 is set for g4 estimated sigma
    hard coded in deltax guesses
    deltax4 is difference between guess from peak and desired position.
    """
    if focus_lam is None:
        focus_lam = [np.min(x), np.max(x)]
        focus_lam[0] = self.param_hints["g4_center"].get("min", focus_lam[0])
        focus_lam[1] = self.param_hints["g4_center"].get("max", focus_lam[1])

    focus_index = (x > focus_lam[0]) & (x < focus_lam[1])

    height, center, sigma = guess_from_peak(data[focus_index], x[focus_index])
    constant = mean_edges(data, edge_fraction=0.1)

    # fill in any missing function parameters based on 1gauss guess:
    if sigma0 is None:
        sigma0 = sigma

    # calculate component guesses based off 1gauss guess and function parameters
    g1_sigma, g2_sigma, g3_sigma, g4_sigma, g5_sigma, g6_sigma = sigma0 * np.array(
        sigma_factors
    )

    g1_height, g2_height, g3_height, g4_height, g5_height, g6_height = (
        height * np.array(heights)
    )

    g4_center = center + deltax4

    pars = self.make_params(
        deltax12=deltax12,
        deltax34=deltax34,
        deltax56=deltax56,
        g4_center=g4_center,
        g1_h_factor=g1_height / g2_height,
        g1_sigma=g1_sigma,
        g2_height=g2_height,
        g2_sigma=g2_sigma,
        g3_h_factor=g3_height / g4_height,
        g3_sigma=g3_sigma,
        g4_height=g4_height,
        g4_sigma=g4_sigma,
        g5_h_factor=g5_height / g6_height,
        g5_sigma=g5_sigma,
        g6_height=g6_height,
        g6_sigma=g6_sigma,
        c=constant,
    )

    return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


class Const_1GaussModel_fast(lmfit.Model):
    """The fast evaluation version of Const_1GaussModel.
    It is created using lmfit.Model instead of CompositeModel.
    """

    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    """float: Factor used to create :func:`lmfit.models.fwhm_expr`."""
    flux_factor = np.sqrt(2 * np.pi)
    """float: Factor used to create :func:`flux_expr`."""

    def __init__(
        self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs
    ):  # noqa
        kwargs.update(
            {
                "prefix": prefix,
                "nan_policy": nan_policy,
                "independent_vars": independent_vars,
            }
        )
        super().__init__(gaussian1CH_d, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        comp_pre = ["g1_"]
        for comp in comp_pre:
            self.set_param_hint(comp + "sigma", min=0)
            self.set_param_hint(comp + "height", min=0)
            self.set_param_hint(comp + "fwhm", expr=fwhm_expr_fast(self, comp))
            self.set_param_hint(comp + "flux", expr=flux_expr_fast(self, comp))

    def _reprstring(self, long=False):
        return "constant + 1 gaussian (fast)"

    guess = _guess_1gauss_d

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC


class Const_2GaussModel_fast(lmfit.Model):
    """The fast evaluation version of Const_2GaussModel.
    It is created using lmfit.Model instead of CompositeModel.
    """

    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    """float: Factor used to create :func:`lmfit.models.fwhm_expr`."""
    flux_factor = np.sqrt(2 * np.pi)
    """float: Factor used to create :func:`flux_expr`."""

    def __init__(
        self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs
    ):  # noqa
        kwargs.update(
            {
                "prefix": prefix,
                "nan_policy": nan_policy,
                "independent_vars": independent_vars,
            }
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

    def __init__(
        self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs
    ):  # noqa
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


class Const_4GaussModel_fast(lmfit.Model):
    """A fast evaluation version of Const_4GaussModel.
    It is created using lmfit.Model instead of CompositeModel.
    """

    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    """float: Factor used to create :func:`lmfit.models.fwhm_expr`."""
    flux_factor = np.sqrt(2 * np.pi)
    """float: Factor used to create :func:`flux_expr`."""

    def __init__(
        self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs
    ):  # noqa
        kwargs.update(
            {
                "prefix": prefix,
                "nan_policy": nan_policy,
                "independent_vars": independent_vars,
            }
        )
        super().__init__(gaussian4CH_d, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        comp_pre = ["g1_", "g2_", "g3_", "g4_"]
        for comp in comp_pre:
            self.set_param_hint(comp + "sigma", min=0)
            self.set_param_hint(comp + "height", min=0)
            self.set_param_hint(comp + "fwhm", expr=fwhm_expr_fast(self, comp))
            self.set_param_hint(comp + "flux", expr=flux_expr_fast(self, comp))
        self.set_param_hint("g1_center", expr="g4_center+deltax1")
        self.set_param_hint("g2_center", expr="g4_center+deltax2")
        self.set_param_hint("g3_center", expr="g4_center+deltax3")

    guess = _guess_4gauss_d

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC


class Const_4GaussModel_constrained_SII_fast(lmfit.Model):
    """A fast evaluation version of Const_4GaussModel.
    It is created using lmfit.Model instead of CompositeModel.
    """

    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    """float: Factor used to create :func:`lmfit.models.fwhm_expr`."""
    flux_factor = np.sqrt(2 * np.pi)
    """float: Factor used to create :func:`flux_expr`."""

    def __init__(
        self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs
    ):  # noqa
        kwargs.update(
            {
                "prefix": prefix,
                "nan_policy": nan_policy,
                "independent_vars": independent_vars,
            }
        )
        super().__init__(gaussian4CH_constrained_SII_d, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        d24 = gaussian4CH_constrained_SII_d_DELTAX24
        comp_pre = ["g1_", "g2_", "g3_", "g4_"]
        for comp in comp_pre:
            self.set_param_hint(comp + "sigma", min=0)
            self.set_param_hint(comp + "height", min=0)
            self.set_param_hint(comp + "fwhm", expr=fwhm_expr_fast(self, comp))
            self.set_param_hint(comp + "flux", expr=flux_expr_fast(self, comp))
        self.set_param_hint("g1_center", expr=f"g4_center+deltax12+{d24}")
        self.set_param_hint("g2_center", expr=f"g4_center+{d24}")
        self.set_param_hint("g3_center", expr=f"g4_center+deltax34")

    guess = _guess_multiline4_constrained_d

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC


class Const_6GaussModel_fast(lmfit.Model):
    """A fast evaluation version of Const_6GaussModel.
    It is created using lmfit.Model instead of CompositeModel.
    """

    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    """float: Factor used to create :func:`lmfit.models.fwhm_expr`."""
    flux_factor = np.sqrt(2 * np.pi)
    """float: Factor used to create :func:`flux_expr`."""

    def __init__(
        self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs
    ):  # noqa
        kwargs.update(
            {
                "prefix": prefix,
                "nan_policy": nan_policy,
                "independent_vars": independent_vars,
            }
        )
        super().__init__(gaussian6CH_d, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        comp_pre = ["g1_", "g2_", "g3_", "g4_", "g5_", "g6_"]
        for comp in comp_pre:
            self.set_param_hint(comp + "sigma", min=0)
            self.set_param_hint(comp + "height", min=0)
            self.set_param_hint(comp + "fwhm", expr=fwhm_expr_fast(self, comp))
            self.set_param_hint(comp + "flux", expr=flux_expr_fast(self, comp))
        self.set_param_hint("g1_center", expr="g4_center+deltax1")
        self.set_param_hint("g2_center", expr="g4_center+deltax2")
        self.set_param_hint("g3_center", expr="g4_center+deltax3")
        self.set_param_hint("g5_center", expr="g4_center+deltax5")
        self.set_param_hint("g6_center", expr="g4_center+deltax6")

    guess = _guess_6gauss_d

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC


class Const_6GaussModel_constrained_HaNII_fast(lmfit.Model):
    """A fast evaluation version of Const_6GaussModel.
    It is created using lmfit.Model instead of CompositeModel.
    """

    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    """float: Factor used to create :func:`lmfit.models.fwhm_expr`."""
    flux_factor = np.sqrt(2 * np.pi)
    """float: Factor used to create :func:`flux_expr`."""

    def __init__(
        self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs
    ):  # noqa
        kwargs.update(
            {
                "prefix": prefix,
                "nan_policy": nan_policy,
                "independent_vars": independent_vars,
            }
        )
        super().__init__(gaussian6CH_constrained_HaNII_d, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):

        d24 = gaussian6CH_constrained_HaNII_d_DELTAX24
        d64 = gaussian6CH_constrained_HaNII_d_DELTAX64
        comp_pre = ["g1_", "g2_", "g3_", "g4_", "g5_", "g6_"]
        for comp in comp_pre:
            self.set_param_hint(comp + "sigma", min=0)
            self.set_param_hint(comp + "height", min=0)
            self.set_param_hint(comp + "fwhm", expr=fwhm_expr_fast(self, comp))
            self.set_param_hint(comp + "flux", expr=flux_expr_fast(self, comp))
        self.set_param_hint("g1_center", expr=f"g4_center+deltax12+{d24}")
        self.set_param_hint("g2_center", expr=f"g4_center+{d24}")
        self.set_param_hint("g3_center", expr=f"g4_center+deltax34")
        self.set_param_hint("g5_center", expr=f"g4_center+deltax56+{d64}")
        self.set_param_hint("g6_center", expr=f"g4_center+{d64}")
        self.set_param_hint("g1_height", expr="g2_height*g1_h_factor")
        self.set_param_hint("g3_height", expr="g4_height*g3_h_factor")
        self.set_param_hint("g5_height", expr="g6_height*g5_h_factor")

    guess = _guess_multiline6_constrained_d

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC
