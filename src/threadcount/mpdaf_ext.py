"""Functions to extend Spectrum, Image, and Cube from package mpdaf."""
import mpdaf.obj.spectrum
import mpdaf.obj.image
import numpy as np
from scipy import signal

# import astropy.units as u
# from mpdaf.obj.plot import FormatCoord, get_plot_norm
# import matplotlib.pyplot as plt


# def plot(  # noqa: C901
#     self,
#     title=None,
#     scale="linear",
#     vmin=None,
#     vmax=None,
#     zscale=False,
#     colorbar=None,
#     var=False,
#     show_xlabel=False,
#     show_ylabel=False,
#     ax=None,
#     unit=u.deg,
#     use_wcs=False,
#     **kwargs,
# ):
#     """Plot the image with axes labeled in pixels.

#     If either axis has just one pixel, plot a line instead of an image.
#     Colors are assigned to each pixel value as follows. First each
#     pixel value, ``pv``, is normalized over the range ``vmin`` to ``vmax``,
#     to have a value ``nv``, that goes from 0 to 1, as follows::

#         nv = (pv - vmin) / (vmax - vmin)

#     This value is then mapped to another number between 0 and 1 which
#     determines a position along the colorbar, and thus the color to give
#     the displayed pixel. The mapping from normalized values to colorbar
#     position, color, can be chosen using the scale argument, from the
#     following options:

#     - 'linear': ``color = nv``
#     - 'log': ``color = log(1000 * nv + 1) / log(1000 + 1)``
#     - 'sqrt': ``color = sqrt(nv)``
#     - 'arcsinh': ``color = arcsinh(10*nv) / arcsinh(10.0)``

#     A colorbar can optionally be drawn. If the colorbar argument is given
#     the value 'h', then a colorbar is drawn horizontally, above the plot.
#     If it is 'v', the colorbar is drawn vertically, to the right of the
#     plot.

#     By default the image is displayed in its own plot. Alternatively
#     to make it a subplot of a larger figure, a suitable
#     ``matplotlib.axes.Axes`` object can be passed via the ``ax`` argument.
#     Note that unless matplotlib interative mode has previously been enabled
#     by calling ``matplotlib.pyplot.ion()``, the plot window will not appear
#     until the next time that ``matplotlib.pyplot.show()`` is called. So to
#     arrange that a new window appears as soon as ``Image.plot()`` is
#     called, do the following before the first call to ``Image.plot()``::

#         import matplotlib.pyplot as plt
#         plt.ion()

#     Parameters
#     ----------
#     title : str
#         An optional title for the figure (None by default).
#     scale : 'linear' | 'log' | 'sqrt' | 'arcsinh'
#         The stretch function to use mapping pixel values to
#         colors (The default is 'linear'). The pixel values are
#         first normalized to range from 0 for values <= vmin,
#         to 1 for values >= vmax, then the stretch algorithm maps
#         these normalized values, nv, to a position p from 0 to 1
#         along the colorbar, as follows:
#         linear:  p = nv
#         log:     p = log(1000 * nv + 1) / log(1000 + 1)
#         sqrt:    p = sqrt(nv)
#         arcsinh: p = arcsinh(10*nv) / arcsinh(10.0)
#     vmin : float
#         Pixels that have values <= vmin are given the color
#         at the dark end of the color bar. Pixel values between
#         vmin and vmax are given colors along the colorbar according
#         to the mapping algorithm specified by the scale argument.
#     vmax : float
#         Pixels that have values >= vmax are given the color
#         at the bright end of the color bar. If None, vmax is
#         set to the maximum pixel value in the image.
#     zscale : bool
#         If True, vmin and vmax are automatically computed
#         using the IRAF zscale algorithm.
#     colorbar : str
#         If 'h', a horizontal colorbar is drawn above the image.
#         If 'v', a vertical colorbar is drawn to the right of the image.
#         If None (the default), no colorbar is drawn.
#     var : bool
#             If true variance array is shown in place of data array
#     ax : matplotlib.axes.Axes
#         An optional Axes instance in which to draw the image,
#         or None to have one created using ``matplotlib.pyplot.gca()``.
#     unit : `astropy.units.Unit`
#         The units to use for displaying world coordinates
#         (degrees by default). In the interactive plot, when
#         the mouse pointer is over a pixel in the image the
#         coordinates of the pixel are shown using these units,
#         along with the pixel value.
#     use_wcs : bool
#         If True, use `astropy.visualization.wcsaxes` to get axes
#         with world coordinates.
#     kwargs : matplotlib.artist.Artist
#         Optional extra keyword/value arguments to be passed to
#         the ``ax.imshow()`` function.

#     Returns
#     -------
#     out : matplotlib AxesImage

#     """
#     cax = None
#     # Default X and Y axes are labeled in pixels.
#     xlabel = "q (pixel)"
#     ylabel = "p (pixel)"

#     if ax is None:
#         if use_wcs:
#             ax = plt.subplot(projection=self.wcs.wcs)
#             xlabel = "ra"
#             ylabel = "dec"
#         else:
#             ax = plt.gca()
#     elif use_wcs:
#         self._logger.warning("use_wcs does not work when giving also an axis (ax)")

#     if var:
#         data_plot = self.var
#     else:
#         data_plot = self.data

#     # If either axis has just one pixel, plot it as a line-graph.
#     if self.shape[1] == 1:
#         # Plot a column as a line-graph
#         yaxis = np.arange(self.shape[0], dtype=float)
#         ax.plot(yaxis, data_plot)
#         xlabel = "p (pixel)"
#         ylabel = self.unit
#     elif self.shape[0] == 1:
#         # Plot a row as a line-graph
#         xaxis = np.arange(self.shape[1], dtype=float)
#         ax.plot(xaxis, data_plot.T)
#         xlabel = "q (pixel)"
#         ylabel = self.unit
#     else:
#         # Plot a 2D image.
#         # get image normalization
#         norm = get_plot_norm(
#             data_plot, vmin=vmin, vmax=vmax, zscale=zscale, scale=scale
#         )

#         # Display the image.
#         cax = ax.imshow(
#             data_plot, interpolation="nearest", origin="lower", norm=norm, **kwargs
#         )

#         # # Create a colorbar
#         if colorbar == "h":
#             # not perfect but it's okay.
#             cbar = plt.colorbar(cax, ax=ax, orientation="horizontal", location="top")
#             for t in cbar.ax.xaxis.get_major_ticks():
#                 t.tick1On = True
#                 t.tick2On = True
#                 t.label1On = False
#                 t.label2On = True
#         elif colorbar == "v":
#             fraction = 0.15 * ax.get_aspect()
#             plt.colorbar(cax, ax=ax, aspect=20, fraction=fraction)
#         # Keep the axis to allow other functions to overplot
#         # the image with contours etc.
#         self._ax = ax

#     # Label the axes if requested.
#     if show_xlabel:
#         ax.set_xlabel(xlabel)
#     if show_ylabel:
#         ax.set_ylabel(ylabel)
#     if title is not None:
#         ax.set_title(title)

#     # Change the way that plt.show() displays coordinates when the pointer
#     # is over the image, such that world coordinates are displayed with the
#     # specified unit, and pixel values are displayed with their native
#     # units.
#     ax.format_coord = FormatCoord(self, data_plot)
#     self._unit = unit
#     return cax


# mpdaf.obj.image.Image.plot = plot


def lmfit(self, model, **kwargs):
    """Fit `model` to :class:`~mpdaf.obj.spectrum.Spectrum` using lmfit.

    This function is an interface between the :class:`~mpdaf.obj.spectrum.Spectrum`
    and :meth:`lmfit.model.Model.fit`. The Spectrum data, variance, and x are passed to
    :meth:`lmfit.model.Model.fit`, along with the other `kwargs`.

    If `params` is not provided in `kwargs`, then :meth:`lmfit.model.Model.guess`
    is called to compute it. If the guess function is not implemented for the `model`,
    the values for all parameters are expected to be provided as keyword arguments.
    If params is given, and a keyword argument for a parameter value is also given,
    the keyword argument will be used.

    Parameters
    ----------
    model : :class:`lmfit.model.Model`
        lmfit Model to use for fitting
    **kwargs : dict
        Any additional keywords and arguments are passed to :meth:`lmfit.model.Model.fit`

    Returns
    -------
    :class:`lmfit.model.ModelResult`, or None
        The fitted ModelResult, or None if the Spectrum was entirely masked.
    """
    mask = self.mask
    if all(mask):
        return None
    data = self.data
    var = self.var
    x = self.wave.coord()

    if var is not None:
        weights = 1 / np.sqrt(np.abs(var))
    else:
        weights = None
    params = kwargs.pop("params", None)

    if params is None:
        try:
            params = model.guess(data, x=x)
        except NotImplementedError:
            # keep params None and perhaps the values will be passed vie kwargs
            # which would still allow the fit function below to complete.
            pass

    try:
        modelresult = model.fit(data, params=params, x=x, weights=weights, **kwargs)
    except ValueError as e:
        if not str(e).contains("infeasible"):
            raise
        else:
            # this happens when the param value is outside the bounds, so lets
            # cyle through the params and set their value to be their value,
            # because lmfit ensures that it's within the bounds.
            for param in params:
                param.set(value=param.value)
            try:
                modelresult = model.fit(
                    data, params=params, x=x, weights=weights, **kwargs
                )
            except ValueError:
                modelresult = None

    return modelresult


mpdaf.obj.spectrum.Spectrum.lmfit = lmfit
"""Create docstring for this."""


def correlate2d_norm(self, other, interp="no"):
    """Return the cross-correlation of the image with an array.

    Uses `scipy.signal.correlate2d`.
    This function normalizes the `other` image and now properly
    carries treats the variance. By that I mean: each element of other is
    squared before it is correlated in scipy. In this way, I hope that propagation
    of errors is done right.

    Parameters
    ----------
    other : 2d-array
        Second 2d-array.
    interp : 'no' | 'linear' | 'spline'
        if 'no', data median value replaced masked values.
        if 'linear', linear interpolation of the masked values.
        if 'spline', spline interpolation of the masked values.

    Returns
    -------
    :class:`mpdaf.obj.image.Image`
    """
    # normalize the `other` image:
    other_norm = other / np.sum(other)
    # Get a copy of the data array with masked values filled.
    data = self._prepare_data(interp)

    res = self.copy()

    res._data = signal.correlate2d(data, other_norm, mode="same", boundary="symm")
    if res._var is not None:
        other_norm_sq = other_norm * other_norm
        res._var = signal.correlate2d(
            res._var, other_norm_sq, mode="same", boundary="symm"
        )
    return res


mpdaf.obj.image.Image.correlate2d_norm = correlate2d_norm
