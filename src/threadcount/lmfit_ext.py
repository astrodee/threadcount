"""Functions to extend classes Model, ModelResult, Parameters from package lmfit."""
from copy import copy
import lmfit
import numpy as np


def plot2(
    self,
    datafmt="o",
    fitfmt="-",
    initfmt="--",
    xlabel=None,
    ylabel=None,
    yerr=None,
    numpoints=None,
    fig=None,
    data_kws=None,
    fit_kws=None,
    init_kws=None,
    ax_res_kws=None,
    ax_fit_kws=None,
    fig_kws=None,
    show_init=False,
    parse_complex="abs",
    title=None,
):
    """Plot the fit results and residuals using matplotlib.

    The method will produce a matplotlib figure (if package available)
    with both results of the fit and the residuals plotted. If the fit
    model included weights, errorbars will also be plotted. To show
    the initial conditions for the fit, pass the argument
    ``show_init=True``.

    Parameters
    ----------
    datafmt : str, optional
        Matplotlib format string for data points.
    fitfmt : str, optional
        Matplotlib format string for fitted curve.
    initfmt : str, optional
        Matplotlib format string for initial conditions for the fit.
    xlabel : str, optional
        Matplotlib format string for labeling the x-axis.
    ylabel : str, optional
        Matplotlib format string for labeling the y-axis.
    yerr : numpy.ndarray, optional
        Array of uncertainties for data array.
    numpoints : int, optional
        If provided, the final and initial fit curves are evaluated
        not only at data points, but refined to contain `numpoints`
        points in total.
    fig : matplotlib.figure.Figure, optional
        The figure to plot on. The default is None, which means use
        the current pyplot figure or create one if there is none.
    data_kws : dict, optional
        Keyword arguments passed to the plot function for data points.
    fit_kws : dict, optional
        Keyword arguments passed to the plot function for fitted curve.
    init_kws : dict, optional
        Keyword arguments passed to the plot function for the initial
        conditions of the fit.
    ax_res_kws : dict, optional
        Keyword arguments for the axes for the residuals plot.
    ax_fit_kws : dict, optional
        Keyword arguments for the axes for the fit plot.
    fig_kws : dict, optional
        Keyword arguments for a new figure, if a new one is created.
    show_init : bool, optional
        Whether to show the initial conditions for the fit (default is
        False).
    parse_complex : {'abs', 'real', 'imag', 'angle'}, optional
        How to reduce complex data for plotting. Options are one of:
        `'abs'` (default), `'real'`, `'imag'`, or `'angle'`, which
        correspond to the NumPy functions with the same name.
    title : str, optional
        Matplotlib format string for figure title.

    Returns
    -------
    matplotlib.figure.Figure

    See Also
    --------
    ModelResult.plot_fit : Plot the fit results using matplotlib.
    ModelResult.plot_residuals : Plot the fit residuals using matplotlib.

    Notes
    -----
    The method combines `ModelResult.plot_fit` and
    `ModelResult.plot_residuals`.
    If `yerr` is specified or if the fit model included weights, then
    `matplotlib.axes.Axes.errorbar` is used to plot the data. If
    `yerr` is not specified and the fit includes weights, `yerr` set
    to ``1/self.weights``.
    If model returns complex data, `yerr` is treated the same way that
    weights are in this case.
    If `fig` is None then `matplotlib.pyplot.figure(**fig_kws)` is
    called, otherwise `fig_kws` is ignored.
    """
    from matplotlib import pyplot as plt
    import matplotlib as mpl

    if data_kws is None:
        data_kws = {}
    if fit_kws is None:
        fit_kws = {}
    if init_kws is None:
        init_kws = {}
    if ax_res_kws is None:
        ax_res_kws = {}
    if ax_fit_kws is None:
        ax_fit_kws = {}

    # make a square figure with side equal to the default figure's x-size
    figxsize = plt.rcParams["figure.figsize"][0]
    fig_kws_ = {"figsize": (figxsize, figxsize)}
    if fig_kws is not None:
        fig_kws_.update(fig_kws)

    if len(self.model.independent_vars) != 1:
        print("Fit can only be plotted if the model function has one " "independent variable.")
        return False

    if not isinstance(fig, (plt.Figure, mpl.figure.SubFigure)):
        fig = plt.figure(**fig_kws_)

    gs = plt.GridSpec(nrows=2, ncols=1, height_ratios=[1, 4])
    ax_res = fig.add_subplot(gs[0], **ax_res_kws)
    ax_fit = fig.add_subplot(gs[1], sharex=ax_res, **ax_fit_kws)

    self.plot_fit(
        ax=ax_fit,
        datafmt=datafmt,
        fitfmt=fitfmt,
        yerr=yerr,
        initfmt=initfmt,
        xlabel=xlabel,
        ylabel=ylabel,
        numpoints=numpoints,
        data_kws=data_kws,
        fit_kws=fit_kws,
        init_kws=init_kws,
        ax_kws=ax_fit_kws,
        show_init=show_init,
        parse_complex=parse_complex,
    )
    self.plot_residuals(
        ax=ax_res,
        datafmt=datafmt,
        yerr=yerr,
        data_kws=data_kws,
        fit_kws=fit_kws,
        ax_kws=ax_res_kws,
        parse_complex=parse_complex,
    )
    plt.setp(ax_res.get_xticklabels(), visible=False)
    ax_fit.set_title("")
    if title is not None:
        ax_res.set_title(title)
    return fig, ax_res, ax_fit


def plot_components(
    self,
    ax=None,
    show_combined=True,
    alpha=1,
    log=False,
    keep_ylim=True,
    zorder=-10,
):
    from matplotlib import pyplot as plt

    ylim = None

    # if you are passing in an axis and wish to keep the current view limits:
    if (ax is not None) and (keep_ylim is True):
        ylim = ax.get_ylim()

    x = self.userkws["x"]
    if ax is None:
        ax = plt.gca()
    for p in self.eval_components().values():
        if log:
            p = np.log10(p)
        try:
            ax.plot(x, p, zorder=-10, alpha=alpha)
        except ValueError:
            ax.axhline(p, zorder=-10, alpha=alpha)
    if show_combined:
        ax.plot(x, self.best_fit, "k", zorder=zorder, alpha=alpha)

    if ylim is not None:
        ax.set_ylim(ylim)
    return ax


def stderrsdict(self):
    """Return an ordered dictionary of parameter stderrs.

    Returns
    -------
    dict
        A dictionary of :attr:`name`::attr:`stderr` pairs for each
        Parameter.
    """
    return {p.name: p.stderr for p in self.values()}


def valerrsdict(self):
    """Return a dictionary of parameter [value, stderr].

    Returns
    -------
    dict
        A dictionary of :attr:`name`:[:attr:`value`, :attr:`stderr`] pairs for each
        Parameter.
    """
    result = {}
    for p in self.values():
        result[p.name] = p.value
        result[p.name + "_err"] = p.stderr
    return result


def set_param_hints_endswith(self, name, **kwargs):
    """Set param hints for all model params names ending in `name`.

    Parameters
    ----------
    name : str
        The string to search the param names for, at the end of the name.
    **kwargs : dict
        Passed to :meth:`lmfit.models.Model.set_param_hint` for each matching
        param name.

    See Also
    --------
    :meth:`lmfit.models.Model.set_param_hint`

    Examples
    --------
    >>> from threadcount.models import Const_3GaussModel
    >>> model = Const_3GaussModel()
    >>> model.set_param_hint_endswith("_sigma", min=0.9)
    """
    names = self.param_names
    for this_name in names:
        if this_name.endswith(name):
            self.set_param_hint(this_name, **kwargs)


def mc_iter(self, n_mc_iterations=0, distribution="normal"):
    """ModelResult extension to change the data related to sigma and refit.

    The new data will be randomly `n_mc_iterations` times, using the data
    value and the sigma as inputs to the random number generator.

    `distribution` options may be:

    * "normal", which returns a normal distribution
      where mean = data value and sigma = sigma.
    * "uniform", which returns a uniform distribution in the range
      [data - 2*sigma, data + 2*sigma)

    The original ModelResult will be returned as the 0th element, meaning
    an array of 1+`n_mc_iterations` will be returned.

    Parameters
    ----------
    n_mc_iterations : int, optional
        The number of new data sets to generate and fit, by default 0
    distribution : "normal" or "uniform", optional
        The type of random distribution to use, by default "normal"

    Returns
    -------
    list of ModelResult
        a list of 1+`n_mc_iterations` ModelResults, where the first element
        contains the original ModelResult (i.e. the real measured data.)

    Raises
    ------
    NotImplementedError
        If `distribution` is not recognized, this function cannot complete.
    """  # noqa: D403
    if n_mc_iterations == 0:
        return [self]

    # extract fit input from self
    data = self.data
    sigma = 1 / self.weights
    params = self.init_params
    # print(params)

    if distribution == "normal":
        distribution_fcn = np.random.default_rng(42).normal
        input1 = data  # loc ("center")
        input2 = sigma  # scale ("standard deviation")
    elif distribution == "uniform":
        distribution_fcn = np.random.default_rng(42).uniform
        input1 = data - 2 * sigma  # low
        input2 = data + 2 * sigma  # high
    else:
        raise NotImplementedError("distribution " + distribution + " not implemented.")
    # use the above definitions to calculate the monte carlo iterations.
    mc_data = distribution_fcn(input1, input2, (n_mc_iterations, np.broadcast(input1, input2).size))
    mc_fits = [self]
    for mcd in mc_data:
        modelresult = copy(self)
        modelresult.params = params.copy()
        modelresult.fit(data=mcd)  # , params=params)
        mc_fits += [modelresult]
    return mc_fits


def summary_array(self, fit_info=None, param_info=None):
    if fit_info is None:
        fit_info = []
    if param_info is None:
        param_info = []

    result = [getattr(self, attr) for attr in fit_info]
    d = self.params.valerrsdict()
    result += [d.get(par, None) for par in param_info]

    return np.array(result, dtype=float)


def order_gauss(self, delta_x=0.5):
    # shifting things around can get messy when there are expressions involved.
    # First things first: Lets remove all expressions.
    for p in self.values():
        p.set(expr="")

    ngauss = sum([x.startswith("g") and x.endswith("_sigma") for x in self])
    if ngauss in (0, 1):
        return

    centers = [self.get("g{}_center".format(i + 1)).value for i in range(ngauss)]
    heights = [self.get("g{}_height".format(i + 1)).value for i in range(ngauss)]

    order = np.argsort(centers)
    centers = np.array(centers)[order]
    heights = np.array(heights)[order]

    for i in range(ngauss - 1):
        if centers[i + 1] - centers[i] < delta_x:
            # now we have to compare heights, and set the tallest one second
            if heights[i + 1] < heights[i]:
                # print("order changed from heights")
                heights[i], heights[i + 1] = heights[i + 1], heights[i]
                centers[i], centers[i + 1] = centers[i + 1], centers[i]
                order[i], order[i + 1] = order[i + 1], order[i]
    new_order = list(order)
    paramcopy = self.copy()
    for i, order in enumerate(new_order):
        if i != order:
            # print("reording g{},g{}".format(i + 1, order + 1))
            dest_prefix = "g{}_".format(i + 1)
            orig_prefix = "g{}_".format(order + 1)
            # g{order+1}_ -> g{i+1}_
            for name in self.keys():
                if name.startswith(dest_prefix):
                    suffix = name[len(dest_prefix) :]
                    self[name] = paramcopy[orig_prefix + suffix]


def aic_real(self):
    """Chisqr + 2 * nvarys."""
    try:
        _neg2_log_likel = self.chisqr
        nvarys = getattr(self, "nvarys", len(self.init_vals))
        return _neg2_log_likel + 2 * nvarys
    except (AttributeError, TypeError):
        return None


def bic_real(self):
    """Chisqr + np.log(ndata) * nvarys."""
    try:
        _neg2_log_likel = self.chisqr
        # fallback incase nvarys or ndata not defined.
        nvarys = getattr(self, "nvarys", len(self.init_vals))
        ndata = getattr(self, "ndata", len(self.residual))
        return _neg2_log_likel + np.log(ndata) * nvarys
    except (AttributeError, TypeError):
        return None


def extend_lmfit(lmfit):
    print("function extend_lmfit has been run")

    lmfit.minimizer.MinimizerResult.aic_real = property(aic_real)
    lmfit.minimizer.MinimizerResult.bic_real = property(bic_real)
    # lmfit.model.ModelResult.aic_real = property(aic)
    # lmfit.model.ModelResult.bic_real = property(bic)

    lmfit.model.ModelResult.plot2 = plot2
    lmfit.model.ModelResult.plot_components = plot_components
    lmfit.model.ModelResult.mc_iter = mc_iter
    lmfit.model.ModelResult.summary_array = summary_array

    lmfit.model.Model.set_param_hint_endswith = set_param_hints_endswith

    lmfit.parameter.Parameters.stderrsdict = stderrsdict
    lmfit.parameter.Parameters.valerrsdict = valerrsdict
    lmfit.parameter.Parameters.order_gauss = order_gauss


extend_lmfit(lmfit)
