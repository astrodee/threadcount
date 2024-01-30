"""Custom Models that should function just like lmfit's models."""
import numpy as np
import operator

import lmfit

from .basic import guess_from_peak, mean_edges

tiny = lmfit.models.tiny  # 1.0e-15

__all__ = [
    "Const_1GaussModel",
    "Const_2GaussModel",
    "Const_3GaussModel",
    "GaussianModelH",
    "Log10_DoubleExponentialModel",
    "Quadratic_1GaussModel",
    "Quadratic_2GaussModel",
    "Quadratic_3GaussModel",
    "gaussianH",
    "gaussian2CH",
    "gaussian3CH",
    "_guess_1gauss",
    "_guess_2gauss",
    "_guess_2gauss_old",
    "_guess_3gauss",
    "_guess_3gauss_old",
    "_guess_multiline2",
    "_guess_multiline3",
]


def gaussianH(x, height=1.0, center=0.0, sigma=1.0):
    """Return a 1-dimensional Gaussian function.

    gaussian(x, height, center, sigma) =
        height * exp(-(1.0*x-center)**2 / (2*sigma**2))
    """
    return height * np.exp(-((1.0 * x - center) ** 2) / max(tiny, (2 * sigma**2)))


def gaussian2CH(
    x,
    g1_height=1.0,
    g1_center=0.0,
    g1_sigma=1.0,
    g2_height=1.0,
    g2_center=0.0,
    g2_sigma=1.0,
    c=0.0,
):
    """Return a 2-Gaussian function in 1-dimension."""
    f = (
        g1_height * np.exp(-((1.0 * x - g1_center) ** 2) / max(tiny, (2 * g1_sigma**2)))
        + g2_height * np.exp(-((1.0 * x - g2_center) ** 2) / max(tiny, (2 * g2_sigma**2)))
        + c
    )
    return f


def gaussian3CH(
    x,
    g1_height=1.0,
    g1_center=0.0,
    g1_sigma=1.0,
    g2_height=1.0,
    g2_center=0.0,
    g2_sigma=1.0,
    g3_height=1.0,
    g3_center=0.0,
    g3_sigma=1.0,
    c=0.0,
):
    """Return a 3-Gaussian function in 1-dimension."""
    f = (
        g1_height * np.exp(-((1.0 * x - g1_center) ** 2) / max(tiny, (2 * g1_sigma**2)))
        + g2_height * np.exp(-((1.0 * x - g2_center) ** 2) / max(tiny, (2 * g2_sigma**2)))
        + g3_height * np.exp(-((1.0 * x - g3_center) ** 2) / max(tiny, (2 * g3_sigma**2)))
        + c
    )
    return f


def flux_expr(model):
    """Return constraint expression for line flux."""
    fmt = "{factor:.7f}*{prefix:s}height*{prefix:s}sigma"
    return fmt.format(factor=model.flux_factor, prefix=model.prefix)


def _guess_1gauss(self, data, x, **kwargs):
    """Estimate initial model parameter values from data.

    The data for gaussian g1 will be guessed by 1 gaussian plus constant.

    a and b model parameters are initialized by any model parameter hint and not
    affected by the guess function.

    Parameters
    ----------
    data : array_like
        Array of data (i.e., y-values) to use to guess parameter values.
    x : array_like
        Array of values for the independent variable (i.e., x-values).
    **kws : optional
        Additional keyword arguments, passed to model function.

    Returns
    -------
    params : :class:`~lmfit.parameter.Parameters`
        Initial, guessed values for the parameters of a Model.
    """
    g1_height, g1_center, g1_sigma = guess_from_peak(data, x)
    constant = mean_edges(data, edge_fraction=0.1)

    pars = self.make_params(
        g1_height=g1_height,
        g1_center=g1_center,
        g1_sigma=g1_sigma,
        c=constant,
    )

    return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


def _guess_2gauss(
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

    The parameters will control the computation of initial guesses for the
    two gaussians in relation to guesses for if this had only 1 gaussian component.
    I beleive these default parameters do an okay job at
    many of the test-cases I have given it.

    The height, center, and sigma are estimated assuming this is 1 gaussian
    component. Using these parameters and the function inputs, the height, center,
    and sigma guesses for each gaussian component will be calculated.

    I will conserve ~flux, meaning in this case sum(g#_height * g#_sigma) = height * sigma.
    If `sigma0` is None, then sigma0 is set automatically to this sigma from the
    1 gaussian guess. Then the rest of the factors are calculated as so::

        [g1_sigma, g2_sigma] = sigma_factors * sigma0
        if absolute_centers is False:
            [g1_center, g2_center] = center + sigma0 * centers
        if absolute_centers is True:
            [g1_center, g2_center] = center + centers
        [g1_height, g2_height] = A * height * heights

    and the overall amplitude scaling factor A is computed by::

        g1_height * g1_sigma + g2_height * g2_sigma = height * sigma
        A * height * heights[0] * g1_sigma + A * height * heights[1] * g2_sigma = height * sigma
        A = sigma / (heights[0] * g1_sigma + heights[1] * g2_sigma)

    Parameters
    ----------
    data : array_like
        Array of data (i.e., y-values) to use to guess parameter values.
    x : array_like
        Array of values for the independent variable (i.e., x-values).
    sigma0 : float, optional
        Sets the reference value for computing sigmas and centers, by default None.
        If None, this will be set to the sigma returned from :func:`guess_from_peak`,
        which is related to FWHM.
    heights : array_like of floats of length 2, optional
        A list containing relative heights of [g1_height, g2_height], by default [1,4].
        These will have an overall scale factor computed, so no need to normalize.
    sigma_factors : array_like of floats of length 2, optional
        [g1_sigma, g2_sigma] = `sigma0` * sigma_factors, by default [1,1]
    centers : array_like, optional
        Change from the 1 gaussian guessed center, in units of `sigma0` unless
        `absolute_centers` is True, by default [-0.5,0]
        [g1_center, g2_center] = center + `sigma0` * `centers`
    absolute_centers : bool, optional
        If True, modifies the centers equation to:
        [g1_center, g2_center] = center + `centers`, by default False
    **kws : optional
        Additional keyword arguments, passed to model function.

    Returns
    -------
    params : :class:`~lmfit.parameter.Parameters`
        Initial, guessed values for the parameters of a Model.
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

    a = sigma / (heights[0] * g1_sigma + heights[1] * g2_sigma)
    g1_height, g2_height = a * height * np.array(heights)

    pars = self.make_params(
        g1_height=g1_height,
        g1_center=g1_center,
        g1_sigma=g1_sigma,
        g2_height=g2_height,
        g2_center=g2_center,
        g2_sigma=g2_sigma,
        c=constant,
    )

    return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


def _guess_2gauss_old(
    self, data, x, g1_sigma=None, h2_factor=0.25, s2_factor=1, cen2_offset=None, **kwargs
):
    """Estimate initial model parameter values from data.

    The data for gaussian g1 will be guessed by 1 gaussian plus constant.
    The parameters will control the computation of initial guesses for the
    second gaussian. I beleive these default parameters do an okay job at
    many of the test-cases I have given it.

    Parameters
    ----------
    data : array_like
        Array of data (i.e., y-values) to use to guess parameter values.
    x : array_like
        Array of values for the independent variable (i.e., x-values).
    g1_sigma : float, optional
        Sets the guess value of parameter g1_sigma. Overrides the sigma returned
        from :func:`guess_from_peak` (since it can be artifically wide.)
    h2_factor : float, default 0.25
        g2_height = g1_height * h2_factor
    s2_factor : float, default 1
        g2_sigma = g1_sigma * s2_factor
    cen2_offset : float, default None
        g2_center = g1_center + cen2_offset. If None, the default will be
        calculated: cen2_offset = -0.5 * g1_sigma
    **kws : optional
        Additional keyword arguments, passed to model function.

    Returns
    -------
    params : :class:`~lmfit.parameter.Parameters`
        Initial, guessed values for the parameters of a Model.
    """
    g1_height, g1_center, g1_sigma_temp = guess_from_peak(data, x)
    constant = mean_edges(data, edge_fraction=0.1)

    # fill in any missing function parameters based on 1gauss guess:
    if g1_sigma is None:
        g1_sigma = g1_sigma_temp
    if cen2_offset is None:
        cen2_offset = -0.5 * g1_sigma

    # calculate g2 guesses based off 1gauss guess and function parameters
    g2_sigma = g1_sigma * s2_factor
    g2_height = g1_height * h2_factor
    g2_center = g1_center + cen2_offset

    pars = self.make_params(
        g1_height=g1_height,
        g1_center=g1_center,
        g1_sigma=g1_sigma,
        g2_height=g2_height,
        g2_center=g2_center,
        g2_sigma=g2_sigma,
        c=constant,
    )

    return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


def _guess_3gauss(
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

    The data for gaussian g1 will be guessed by 1 gaussian plus constant.
    The parameters will control the computation of initial guesses for the
    second gaussian. I beleive these default parameters do an okay job at
    many of the test-cases I have given it.

    Parameters
    ----------
    data : array_like
        Array of data (i.e., y-values) to use to guess parameter values.
    x : array_like
        Array of values for the independent variable (i.e., x-values).
    sigma0 : float, optional
        Sets the reference value for computing sigmas and centers, by default None.
        If None, this will be set to the sigma returned from :func:`guess_from_peak`,
        which is related to FWHM.
    heights : array_like of floats of length 3, optional
        A list containing relative heights of [g1_height, g2_height, g3_height],
        by default (1,4,1).
        These will have an overall scale factor computed, so no need to normalize.
    sigma_factors : array_like of floats of length 3, optional
        [g1_sigma, g2_sigma, g3_sigma] = `sigma0` * sigma_factors, by default (1,1,1)
    centers : array_like of floats of length 3, optional
        Change from the 1 gaussian guessed center, in units of `sigma0` unless
        `absolute_centers` is True, by default (-1,0,1)
        [g1_center, g2_center, g3_center] = center + `sigma0` * `centers`
    absolute_centers : bool, optional
        If True, modifies the centers equation to:
        [g1_center, g2_center, g3_center] = center + `centers`, by default False
    **kws : optional
        Additional keyword arguments, passed to model function.

    Returns
    -------
    params : :class:`~lmfit.parameter.Parameters`
        Initial, guessed values for the parameters of a Model.
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

    a = sigma / (heights[0] * g1_sigma + heights[1] * g2_sigma + heights[2] * g3_sigma)
    g1_height, g2_height, g3_height = a * height * np.array(heights)

    pars = self.make_params(
        g1_height=g1_height,
        g1_center=g1_center,
        g1_sigma=g1_sigma,
        g2_height=g2_height,
        g2_center=g2_center,
        g2_sigma=g2_sigma,
        g3_height=g3_height,
        g3_center=g3_center,
        g3_sigma=g3_sigma,
        c=constant,
    )

    return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


def _guess_multiline3(
    self,
    data,
    x,
    sigma0=None,
    heights=(1, 4, 1),
    sigma_factors=(1, 1, 1),
    centers=(-1, 0, 1),
    absolute_centers=False,
    focus_lam=None,
    **kwargs
):
    """Estimate initial model parameter values from data.

    This version is for fitting multiple emission lines that are near each other
    (e.g. Halpha + NII). It is currently hard coded to use the middle Gaussian (g2) as the line
    that is focused on, and then shifts the other Gaussians relative to that. By default
    it uses the bounds on g2 set in model.set_param_hint. This can be overwritten by specifying
    focus_lam = [start wavelength, end wavelength] of your region that you want to use to guide
    your initial guesses.

     Parameters
     ----------
     data : array_like
         Array of data (i.e., y-values) to use to guess parameter values.
     x : array_like
         Array of values for the independent variable (i.e., x-values).
     sigma0 : float, optional
         Sets the reference value for computing sigmas and centers, by default None.
         If None, this will be set to the sigma returned from :func:`guess_from_peak`,
         which is related to FWHM.
     heights : array_like of floats of length 3, optional
         A list containing relative heights of [g1_height, g2_height, g3_height],
         by default (1,4,1).
         These will have an overall scale factor computed, so no need to normalize.
     sigma_factors : array_like of floats of length 3, optional
         [g1_sigma, g2_sigma, g3_sigma] = `sigma0` * sigma_factors, by default (1,1,1)
     centers : array_like of floats of length 3, optional
         Change from the 1 gaussian guessed center, in units of `sigma0` unless
         `absolute_centers` is True, by default (-1,0,1)
         [g1_center, g2_center, g3_center] = center + `sigma0` * `centers`
     absolute_centers : bool, optional
         If True, modifies the centers equation to:
         [g1_center, g2_center, g3_center] = center + `centers`, by default False
     focus_lam : array_like of floats of length 2, optional
         Sets the wavelength range of the guess region for g2, by default None.
         If None, this will be set to the model's param_hint for 'g2_center' [min, max].
         If the param_hint min/max is not set, then the min/max of x is used.
     **kws : optional
         Additional keyword arguments, passed to model function.

     Returns
     -------
     params : :class:`~lmfit.parameter.Parameters`
         Initial, guessed values for the parameters of a Model.
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

    # The factor "a" is not for a guess based on a single line
    # a = sigma / (heights[0] * g1_sigma + heights[1] * g2_sigma + heights[2] * g3_sigma)
    g1_height, g2_height, g3_height = height * np.array(heights)

    pars = self.make_params(
        g1_height=g1_height,
        g1_center=g1_center,
        g1_sigma=g1_sigma,
        g2_height=g2_height,
        g2_center=g2_center,
        g2_sigma=g2_sigma,
        g3_height=g3_height,
        g3_center=g3_center,
        g3_sigma=g3_sigma,
        c=constant,
    )

    return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


def _guess_multiline2(
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

    This version is for fitting 2  emission lines that are near each other
    (e.g. [OII] doublet). It is currently hard coded to use the middle Gaussian (g2) as the line
    that is focused on, and then shifts the other Gaussian relative to that. By default
    it uses the bounds on g2 set in model.set_param_hint. This can be overwritten by specifying
    focus_lam = [start wavelength, end wavelength] of your region that you want to use to guide
    your initial guesses.

     Parameters
     ----------
     data : array_like
         Array of data (i.e., y-values) to use to guess parameter values.
     x : array_like
         Array of values for the independent variable (i.e., x-values).
     sigma0 : float, optional
         Sets the reference value for computing sigmas and centers, by default None.
         If None, this will be set to the sigma returned from :func:`guess_from_peak`,
         which is related to FWHM.
     heights : array_like of floats of length 2, optional
         A list containing relative heights of [g1_height, g2_height],
         by default (1,4).
         These will have an overall scale factor computed, so no need to normalize.
     sigma_factors : array_like of floats of length 2, optional
         [g1_sigma, g2_sigma] = `sigma0` * sigma_factors, by default (1,1)
     centers : array_like of floats of length 2, optional
         Change from the 1 gaussian guessed center, in units of `sigma0` unless
         `absolute_centers` is True, by default (-1,0)
         [g1_center, g2_center] = center + `sigma0` * `centers`
     absolute_centers : bool, optional
         If True, modifies the centers equation to:
         [g1_center, g2_center] = center + `centers`, by default False
     focus_lam : array_like of floats of length 2, optional
         Sets the wavelength range of the guess region for g2, by default None.
         If None, this will be set to the model's param_hint for 'g2_center' [min, max].
         If the param_hint min/max is not set, then the min/max of x is used.
     **kws : optional
         Additional keyword arguments, passed to model function.

     Returns
     -------
     params : :class:`~lmfit.parameter.Parameters`
         Initial, guessed values for the parameters of a Model.
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

    # The factor "a" is not for a guess based on a single line
    # a = sigma / (heights[0] * g1_sigma + heights[1] * g2_sigma + heights[2] * g3_sigma)
    g1_height, g2_height = height * np.array(heights)

    pars = self.make_params(
        g1_height=g1_height,
        g1_center=g1_center,
        g1_sigma=g1_sigma,
        g2_height=g2_height,
        g2_center=g2_center,
        g2_sigma=g2_sigma,
        c=constant,
    )

    return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


def _guess_3gauss_old(
    self,
    data,
    x,
    g1_sigma=None,
    h2_factor=0.25,
    s2_factor=1,
    cen2_offset=None,
    h3_factor=0.25,
    s3_factor=1,
    cen3_offset=None,
    **kwargs
):
    """Estimate initial model parameter values from data.

    The data for gaussian g1 will be guessed by 1 gaussian plus constant.
    The parameters will control the computation of initial guesses for the
    second gaussian. I beleive these default parameters do an okay job at
    many of the test-cases I have given it.

    Parameters
    ----------
    data : array_like
        Array of data (i.e., y-values) to use to guess parameter values.
    x : array_like
        Array of values for the independent variable (i.e., x-values).
    g1_sigma : float, optional
        Sets the guess value of parameter g1_sigma. Overrides the sigma returned
        from :func:`guess_from_peak` (since it can be artifically wide.)
    h2_factor : float, default 0.25
        g2_height = g1_height * h2_factor
    s2_factor : float, default 1
        g2_sigma = g1_sigma * s2_factor
    cen2_offset : float, default None
        g2_center = g1_center + cen2_offset. If None, the default will be
        calculated: cen2_offset = -0.5 * g1_sigma
    h3_factor : float, default 0.25
        g3_height = g1_height * h3_factor
    s3_factor : float, default 1
        g3_sigma = g1_sigma * s3_factor
    cen3_offset : float, default None
        g3_center = g1_center + cen3_offset. If None, the default will be
        calculated: cen3_offset = -0.5 * g1_sigma
    **kws : optional
        Additional keyword arguments, passed to model function.

    Returns
    -------
    params : :class:`~lmfit.parameter.Parameters`
        Initial, guessed values for the parameters of a Model.
    """
    g1_height, g1_center, g1_sigma_temp = guess_from_peak(data, x)
    constant = mean_edges(data, edge_fraction=0.1)

    # fill in any missing function parameters based on 1gauss guess:
    if g1_sigma is None:
        g1_sigma = g1_sigma_temp
    if cen2_offset is None:
        cen2_offset = -0.5 * g1_sigma
    if cen3_offset is None:
        cen3_offset = 0.5 * g1_sigma

    # calculate g2, g3 guesses based off 1gauss guess and function parameters
    g2_sigma = g1_sigma * s2_factor
    g3_sigma = g1_sigma * s3_factor

    g2_height = g1_height * h2_factor
    g3_height = g1_height * h3_factor

    g2_center = g1_center + cen2_offset
    g3_center = g1_center + cen3_offset

    pars = self.make_params(
        g1_height=g1_height,
        g1_center=g1_center,
        g1_sigma=g1_sigma,
        g2_height=g2_height,
        g2_center=g2_center,
        g2_sigma=g2_sigma,
        g3_height=g3_height,
        g3_center=g3_center,
        g3_sigma=g3_sigma,
        c=constant,
    )

    return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


class GaussianModelH(lmfit.Model):
    r"""A model heavily based on lmfit's :class:`~lmfit.models.GaussianModel`, fitting height instead of amplitude.

    A model based on a Gaussian or normal distribution lineshape.
    The model has three Parameters: `height`, `center`, and `sigma`.
    In addition, parameters `fwhm` and `flux` are included as
    constraints to report full width at half maximum and integrated flux, respectively.

    .. math::

       f(x; A, \mu, \sigma) = A e^{[{-{(x-\mu)^2}/{{2\sigma}^2}}]}

    where the parameter `height` corresponds to :math:`A`, `center` to
    :math:`\mu`, and `sigma` to :math:`\sigma`. The full width at half
    maximum is :math:`2\sigma\sqrt{2\ln{2}}`, approximately
    :math:`2.3548\sigma`.

    For more information, see: https://en.wikipedia.org/wiki/Normal_distribution

    The default model is constrained by default param hints so that height > 0.
    You may adjust this as you would in any lmfit model, either directly adjusting
    the parameters after they have been made ( params['height'].set(min=-np.inf) ),
    or by changing the model param hints ( model.set_param_hint('height',min=-np.inf) ).

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
        super().__init__(gaussianH, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint("sigma", min=0)
        self.set_param_hint("height", min=0)
        self.set_param_hint("fwhm", expr=lmfit.models.fwhm_expr(self))
        self.set_param_hint("flux", expr=flux_expr(self))

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data, :func:`guess_from_peak`.

        Parameters
        ----------
        data : array_like
            Array of data (i.e., y-values) to use to guess parameter values.
        x : array_like
            Array of values for the independent variable (i.e., x-values).
        negative : bool, default False
            If True, guess height value assuming height < 0.
        **kws : optional
            Additional keyword arguments, passed to model function.

        Returns
        -------
        params : :class:`~lmfit.parameter.Parameters`
            Initial, guessed values for the parameters of a :class:`lmfit.model.Model`.

        """
        height, center, sigma = guess_from_peak(data, x, negative=negative)
        pars = self.make_params(height=height, center=center, sigma=sigma)

        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC


class Const_1GaussModel(lmfit.model.CompositeModel):
    """Constant + 1 Gaussian Model.

    Essentially created by:

    ``lmfit.models.ConstantModel() + GaussianModelH(prefix="g1_")``

    The param names are ['g1_height',
    'g1_center',
    'g1_sigma',
    'c']
    """

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):  # noqa
        kwargs.update({"nan_policy": nan_policy, "independent_vars": independent_vars})
        if prefix != "":
            print(
                "{}: I don't know how to get prefixes working on composite models yet. "
                "Prefix is ignored.".format(self.__class__.__name__)
            )

        g1 = GaussianModelH(prefix="g1_", **kwargs)
        c = lmfit.models.ConstantModel(prefix="", **kwargs)

        # the below lines gives g1 + c
        super().__init__(g1, c, operator.add)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        # GaussianModelH paramhints already sets sigma min=0 and height min=0
        pass

    def _reprstring(self, long=False):
        return "constant + 1 gaussian"

    guess = _guess_1gauss

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC


class Quadratic_1GaussModel(lmfit.model.CompositeModel):
    r"""Quadratic + 1 Gaussian Model.

    Essentially created by:

    ``lmfit.models.QuadraticModel() + GaussianModelH(prefix="g1_")``

    .. math::

        f(x) = a x^2 + b x + c + A e^{[{-{(x-\mu)^2}/{{2\sigma}^2}}]}

    where the parameter `g1_height` corresponds to :math:`A`, `g1_center` to
    :math:`\mu`, and `g1_sigma` to :math:`\sigma`.

    The param names are ['g1_height',
    'g1_center',
    'g1_sigma',
    'a',
    'b',
    'c']
    """

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):  # noqa
        kwargs.update({"nan_policy": nan_policy, "independent_vars": independent_vars})
        if prefix != "":
            print(
                "{}: I don't know how to get prefixes working on composite models yet. "
                "Prefix is ignored.".format(self.__class__.__name__)
            )

        g1 = GaussianModelH(prefix="g1_", **kwargs)
        quad = lmfit.models.QuadraticModel(prefix="", **kwargs)

        # the below lines gives g1 + quad
        super().__init__(g1, quad, operator.add)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        # GaussianModelH paramhints already sets sigma min=0 and height min=0
        pass

    def _reprstring(self, long=False):
        return "quadratic + 1 gaussian"

    guess = _guess_1gauss

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC


class Const_2GaussModel(lmfit.model.CompositeModel):
    """Constant + 2 Gaussians Model.

    Essentially created by:

    ``lmfit.models.ConstantModel() + GaussianModelH(prefix="g1_")
    + GaussianModelH(prefix="g2_")``

    The param names are ['g1_height',
    'g1_center',
    'g1_sigma',
    'g2_height',
    'g2_center',
    'g2_sigma',
    'c']
    """

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):  # noqa
        kwargs.update({"nan_policy": nan_policy, "independent_vars": independent_vars})
        if prefix != "":
            print(
                "{}: I don't know how to get prefixes working on composite models yet. "
                "Prefix is ignored.".format(self.__class__.__name__)
            )

        g1 = GaussianModelH(prefix="g1_", **kwargs)
        g2 = GaussianModelH(prefix="g2_", **kwargs)
        c = lmfit.models.ConstantModel(prefix="", **kwargs)

        # the below lines gives g1 + g2 + c
        super().__init__(g1 + g2, c, operator.add)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        # GaussianModelH paramhints already sets sigma min=0 and height min=0
        pass
        # TODO: add constraints to allow for wavelength ordering
        # # g1_center < g2_center
        # # g2_center - g1_center = delta1 where delta1 > 0
        # self.set_param_hint("delta1", value=1, min=0, vary=True)
        # self.set_param_hint("g1_center", expr="g2_center-delta1")

    def _reprstring(self, long=False):
        return "constant + 2 gaussians"

    guess = _guess_2gauss

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC


class Quadratic_2GaussModel(lmfit.model.CompositeModel):
    """Quadratic + 2 Gaussians Model.

    :math:`f(x) = a x^2 + b x + c + 2gaussians`. Essentially created by:

    ``lmfit.models.QuadraticModel() + GaussianModelH(prefix="g1_")
    + GaussianModelH(prefix="g2_")``

    The param names are ['g1_height',
    'g1_center',
    'g1_sigma',
    'g2_height',
    'g2_center',
    'g2_sigma',
    'a','b','c']
    """

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):  # noqa
        kwargs.update({"nan_policy": nan_policy, "independent_vars": independent_vars})
        if prefix != "":
            print(
                "{}: I don't know how to get prefixes working on composite models yet. "
                "Prefix is ignored.".format(self.__class__.__name__)
            )

        g1 = GaussianModelH(prefix="g1_", **kwargs)
        g2 = GaussianModelH(prefix="g2_", **kwargs)
        quad = lmfit.models.QuadraticModel(prefix="", **kwargs)

        # the below lines gives g1 + g2 + c
        super().__init__(g1 + g2, quad, operator.add)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        # GaussianModelH paramhints already sets sigma min=0 and height min=0
        pass

    def _reprstring(self, long=False):
        return "quadratic + 2 gaussians"

    guess = _guess_2gauss

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC


class Const_3GaussModel(lmfit.model.CompositeModel):
    """Constant + 3 Gaussians Model.

    Essentially created by:

    ``lmfit.models.ConstantModel() + GaussianModelH(prefix="g1_")
    + GaussianModelH(prefix="g2_") + GaussianModelH(prefix="g3_")``

    The param names are ['g1_height',
    'g1_center',
    'g1_sigma',
    'g2_height',
    'g2_center',
    'g2_sigma',
    'g3_height',
    'g3_center',
    'g3_sigma',
    'c']
    """

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):  # noqa
        kwargs.update({"nan_policy": nan_policy, "independent_vars": independent_vars})
        if prefix != "":
            print(
                "{}: I don't know how to get prefixes working on composite models yet. "
                "Prefix is ignored.".format(self.__class__.__name__)
            )

        g1 = GaussianModelH(prefix="g1_", **kwargs)
        g2 = GaussianModelH(prefix="g2_", **kwargs)
        g3 = GaussianModelH(prefix="g3_", **kwargs)
        c = lmfit.models.ConstantModel(prefix="", **kwargs)

        # the below lines gives g1 + g2 + g3 + c
        super().__init__(g1 + g2 + g3, c, operator.add)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        # GaussianModelH paramhints already sets sigma min=0 and height min=0
        pass

    def _reprstring(self, long=False):
        return "constant + 3 gaussians"

    guess = _guess_3gauss

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC


class Quadratic_3GaussModel(lmfit.model.CompositeModel):
    """Quadratic + 3 Gaussians Model.

    Essentially created by:

    ``lmfit.models.QuadraticModel() + GaussianModelH(prefix="g1_")
    + GaussianModelH(prefix="g2_") + GaussianModelH(prefix="g3_")``

    The param names are ['g1_height',
    'g1_center',
    'g1_sigma',
    'g2_height',
    'g2_center',
    'g2_sigma',
    'g3_height',
    'g3_center',
    'g3_sigma',
    'a','b','c']
    """

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):  # noqa
        kwargs.update({"nan_policy": nan_policy, "independent_vars": independent_vars})
        if prefix != "":
            print(
                "{}: I don't know how to get prefixes working on composite models yet. "
                "Prefix is ignored.".format(self.__class__.__name__)
            )

        g1 = GaussianModelH(prefix="g1_", **kwargs)
        g2 = GaussianModelH(prefix="g2_", **kwargs)
        g3 = GaussianModelH(prefix="g3_", **kwargs)
        quad = lmfit.models.QuadraticModel(prefix="", **kwargs)

        # the below lines gives g1 + g2 + g3 + quad
        super().__init__(g1 + g2 + g3, quad, operator.add)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        # GaussianModelH paramhints already sets sigma min=0 and height min=0
        pass

    def _reprstring(self, long=False):
        return "quadratic + 3 gaussians"

    guess = _guess_3gauss

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC


min_sigma = 0


# def set_component_param_hints(
#     model,param_hint_list = None

# ):
#     if param_hint_list is None:
#         param_hint_list = [
#             {'sigma': {'min': 0}},
#         ]

#     for comp in model.components:
#         if comp.prefix+'sigma' in comp.param_names:
#             temp.set_param_hint(comp.prefix+'sigma',min=min_sigma, max=max_sigma)


def set_common_limits(params, x, data):
    # set all g?_height.min = 0
    # currently deactivated: set all g?_height.max = 1.25*max(data)
    # set all g?_sigma.min = min_sigma (module global variable)
    #   also if g?_sigma.value is < min_sigma, sets g?_sigma.value=min_sigma
    # currently deactivated: set all g?_sigma.max = delta_x/4
    #       This was because I noticed without this and c.max/min,
    #       The code was fitting a wide gaussian for any baseline curvature
    # set all g?_center.max/min to be not edge 10%.
    # set c.max/min related to the baseline standard deviation
    global min_sigma
    sort_increasing = np.argsort(x)
    x = x[sort_increasing]
    limit = int(len(x) * 0.1)  # index of 10% of x data.
    # sig_max = (x[-1] - x[0]) / 4.0
    # h_max = 1.25 * np.max(data)
    # standard deviation of baseline area (1st, last 10% data)
    baseline_std = np.std(data[np.r_[:limit, -limit:]])
    for k, param in params.items():
        if k.endswith("height"):
            param.set(min=0)
            # param.set(min=0, max=h_max)
        elif k.endswith("sigma"):
            param.set(min=min_sigma)
            # param.set(min=min_sigma, max=sig_max)
            if param.value < min_sigma:
                param.set(value=min_sigma)
        elif k.endswith("center"):
            param.set(min=x[limit], max=x[-limit])
        elif k == "c":
            param.set(min=param.value - 2 * baseline_std, max=param.value + 2 * baseline_std)
    return params


def log10_sum(arr1, arr2):
    """Add the 2 arguments and return the log10 of result."""
    return np.log10(arr1 + arr2)


class Log10_DoubleExponentialModel(lmfit.model.CompositeModel):
    r"""Log10 of double exponential decay Model.

    .. math::

        f(x) = log_{10}(A_1 e^{-x/\tau_1} + A_2 e^{-x/\tau_2})

    where the parameter `e1_amplitude` corresponds to :math:`A_1`, `e1_decay` to
    :math:`\tau_1`, `e2_amplitude` corresponds to :math:`A_2`, `e2_decay` to
    :math:`\tau_2`.

    Utilizes :class:`lmfit.models.ExponentialModel` and a custom operation function.

    The param names are ['e1_amplitude','e1_decay','e2_amplitude','e2_decay']
    """

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):  # noqa
        kwargs.update({"nan_policy": nan_policy, "independent_vars": independent_vars})
        if prefix != "":
            print(
                "{}: I don't know how to get prefixes working on composite models yet. "
                "Prefix is ignored.".format(self.__class__.__name__)
            )

        e1 = lmfit.models.ExponentialModel(prefix="e1_")
        e2 = lmfit.models.ExponentialModel(prefix="e2_")

        # the below lines gives np.log10(e1 + e2)
        super().__init__(e1, e2, log10_sum)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint("e1_amplitude", min=0)
        self.set_param_hint("e2_amplitude", min=0)
        self.set_param_hint("e1_decay", min=0)
        self.set_param_hint("e2_decay", min=0)

    def _reprstring(self, long=False):
        return "log10(exp1 + exp2)"

    def guess(self, data, x, a2_factor=1, d2_factor=1, **kwargs):
        """Estimate initial model parameter values from data.

        Uses the :class:`lmfit.models.ExponentialModel` guess function for the
        first exponential. Uses the function parameters to guess the second.

        Parameters
        ----------
        data : array_like
            Array of data (i.e., y-values) to use to guess parameter values.
        x : array_like
            Array of values for the independent variable (i.e., x-values).
        a2_factor : float, default 1
            e2_amplitude = e1_amplitude * a2_factor
        d2_factor : float, default 1
            e2_decay = e1_decay * d2_factor
        **kws : optional
            Additional keyword arguments, passed to model function.

        Returns
        -------
        params : :class:`~lmfit.parameter.Parameters`
            Initial, guessed values for the parameters of a Model.
        """
        # guess e1_ using ExpontialModel guess function.
        init_params = lmfit.models.ExponentialModel().guess(10**data, x, **kwargs)

        e1_amplitude = init_params["amplitude"].value
        e1_decay = init_params["decay"].value

        e2_amplitude = e1_amplitude * a2_factor
        e2_decay = e1_decay * d2_factor

        pars = self.make_params(
            e1_amplitude=e1_amplitude,
            e1_decay=e1_decay,
            e2_amplitude=e2_amplitude,
            e2_decay=e2_decay,
        )

        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC
