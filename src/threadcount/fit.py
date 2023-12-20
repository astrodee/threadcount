"""main threadcount module."""
import json
import csv
from types import SimpleNamespace
from collections import OrderedDict, UserList
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import lmfit
import mpdaf.obj
import astropy.units as u
from . import lines
from . import models
from . import mpdaf_ext  # noqa: F401


FLAM16 = u.Unit(1e-16 * u.erg / (u.cm ** 2 * u.s * u.AA))
"""A header["BUNIT"] value we have."""

FLOAT_FMT = ".8g"
"""Default formatting for floats in output files."""

DEFAULT_FIT_INFO = "aic_real bic_real chisqr redchi success".split()
"""Define typical ModelResult information we might want."""


def open_fits_cube(
    data_filename, data_hdu_index=None, var_filename=None, var_hdu_index=None, mask_if_over_n_nans=None, **kwargs
):
    """Load a fits file using :class:`mpdaf.obj.Cube`, and handle variance in separate file.

    I highly recommend being explicit in the parameters and not relying on the
    guessing that mpdaf can perform.

    Parameters
    ----------
    data_filename : str
        Path to file containing data
    data_hdu_index : int, optional
        Index indicating which hdu contains the data (starting with 0), by default None (then the
        :class:`mpdaf.obj.Cube` constructor will attempt
        to guess the correct extension)
    var_filename : str, optional
        Path to file containing variance, by default None (No variance will be
        loaded. Unless `data_hdu_index` = None, and then the
        :class:`mpdaf.obj.Cube` constructor will attempt to
        automatically load variance from `data_filename`)
    var_hdu_index : int, optional
        Index indicating which hdu contains the variance (starting with 0), by
        default None (then the :class:`mpdaf.obj.Cube` constructor will attempt
        to guess the correct extension)
    **kwargs : dict, optional
        Any keyword arguments to pass to :class:`mpdaf.obj.Cube`, such as `unit`

    Returns
    -------
    :class:`mpdaf.obj.Cube`
        A data cube.
    """
    # no variance given:
    if var_filename is None:
        cube = mpdaf.obj.Cube(data_filename, ext=data_hdu_index, **kwargs)
    # data and variance stored in same file:
    elif data_filename == var_filename:
        cube = mpdaf.obj.Cube(
            data_filename, ext=(data_hdu_index, var_hdu_index), **kwargs
        )

    # data and variance stored in different files:
    else:
        cube = mpdaf.obj.Cube(data_filename, ext=data_hdu_index, **kwargs)

        varcube = mpdaf.obj.Cube(var_filename, ext=var_hdu_index, **kwargs)

        # varcube is loaded as masked array.
        cube._var = varcube.data.data
        cube._mask |= varcube.mask
    if mask_if_over_n_nans is not None:
        masksum = cube.mask.sum(axis=0)
        for iy, ix in np.ndindex(masksum.shape):
            if masksum[iy, ix] > mask_if_over_n_nans:
                cube.mask[:,iy,ix] = True
    # test for FLAM16:
    if cube.unit == u.dimensionless_unscaled:
        if cube.data_header.get("BUNIT") == "FLAM16":
            cube.unit = FLAM16
    return cube


def de_redshift(wavecoord, z=0, z_initial=0):
    r"""De-redshift the WaveCoord in-place.

    Parameters
    ----------
    wavecoord : :class:`mpdaf.obj.WaveCoord`
        The wavelength coordinate to be de-redshifted
    z : float, optional
        The redshift of the object whose wavecoord to de-redshift, by default 0 (i.e. no change)
    z_initial : float, optional
        The redshift currently applied to the wavecoord, by default 0 (i.e. none applied)

    Notes
    -----
    I tried to make z a new attribute in `wavecoord`, but due to details in how
    slicing works, this was not a simple change. Therefore z must be stored in
    a variable externally to the wavecoord.

    """
    wavecoord.set_crval(wavecoord.get_crval() * (1 + z_initial) / (1 + z))
    wavecoord.set_step(wavecoord.get_step() * (1 + z_initial) / (1 + z))
    return z


# TODO: Add in a part where the user can input in a redshift and move the
# histogram or center line or whatever around. i.e. user input at the end.
def tweak_redshift(
    cube,
    z_gal,
    center_wavelength=lines.OIII5007,
    wavelength_range=(-15, 15),  # This is in Angstroms
    pixel_mask=None,
):
    """Interactively choose a new redshift.

    This procedure has several steps.

    1. Select which spaxels to use for calculating the redshift via one of these options:

      * use the input parameter `pixel_mask`
      * Select pixels with a high integrated flux value in the selected wavelength range.
        These are likely to be the galaxy. The user will interact with the terminal
        and view a plot to interactively change the lower threshold for the desired
        pixels. To accept the value plotted, leave the entry blank and press enter.

    2. Fit a :class:`~threadcount.models.Const_1GaussModel` to the selected spaxels.
    3. Extract the parameter value for 'g1_center' to get the center wavelength
       of the fitted gaussian and compute the median center.
    4. Calculate the redshift required for the median center to be equal to
       `center_wavelength` using the formula::

           new_z = (median_center / `center_wavelength`) * (1 + `z_gal`) - 1

    5. Display a plot showing the spaxels used and a histogram displaying all the
       center wavelengths (with `center_wavelength` subtracted, so it displays
       the change from ideal)

    Parameters
    ----------
    cube : :class:`mpdaf.obj.Cube`
        A datacube containing the wavelength range set in these parameters
    z_gal : float
        The redshift of the object which has already been applied to the `cube`
    center_wavelength : float, optional
        The center wavelength of the emission line to fit, by default :const:`threadcount.lines.OIII5007`
    wavelength_range : array-like [float, float], optional
        The wavelength range to fit, in Angstroms. These are defined as a change
        from the `center_wavelength`, by default (-15, 15)

    Returns
    -------
    float
        The redshift selected by the user.
    """
    plt.close()
    print("====================================")
    print("Tweak reshift procedure has started.")
    print("====================================\n\n")
    print("Using line {:.4g} +/- {} A".format(center_wavelength, wavelength_range[1]))

    # retrieve spectral subcube from cube.
    subcube = cube.select_lambda(
        center_wavelength + wavelength_range[0],
        center_wavelength + wavelength_range[1],
    )
    fluxmap = subcube.sum(axis=0)
    if pixel_mask is None:
        # use the sum of the flux and mask at value, changed by user interaction.
        plot_title = (
            "Tweak Redshift:\n"
            "Spaxels to fit. Set mask level in console.\n"
            "Val.=sum of spectrum (arb. units)"
        )
        limit = interactive_lower_threshold(fluxmap, title=plot_title)
        pixel_mask = (fluxmap > limit).mask
    fluxmap.mask = pixel_mask

    fig, axs = plt.subplots(ncols=2, gridspec_kw={"top": 0.85})
    fig.suptitle(
        "Tweak z using line {:.4g} +/- {} A".format(
            center_wavelength, wavelength_range[1]
        )
    )
    fluxmap.plot(
        ax=axs[0],
        title="Spaxels included in histogram\nVal.=sum of spectrum (arb. units)",
        colorbar="v",
        zscale=True,
    )
    valid_pixels = np.where(~fluxmap.mask)
    # loop over valid pixels, do the fit, and store in results list.
    results = []
    model = models.Const_1GaussModel()
    params = None
    print("Fitting selected spaxels with gaussian model...")
    for y, x in zip(*valid_pixels):
        this_mr = subcube[:, y, x].lmfit(model, params=params, method="least_squares", nan_policy='omit')
        if params is None:
            params = this_mr.params
        results += [this_mr]
    fit_centers = vget_param_values(results, "g1_center")
    # remove invalid values, specifically centers outside the range given:
    fit_centers = fit_centers[fit_centers < (center_wavelength + wavelength_range[1])]
    fit_centers = fit_centers[fit_centers > (center_wavelength + wavelength_range[0])]

    plt.sca(axs[1])
    plt.hist(fit_centers - center_wavelength, bins=20)
    plt.title("Center wavelength histogram")
    plt.xlabel(r"change from {:.5g} $\AA$ [$\AA$]".format(center_wavelength))
    plt.axvline(0, color="black", label=r"{:.5g} $\AA$".format(center_wavelength))
    plt.axvline(
        np.nanmedian(fit_centers) - center_wavelength, color="red", label="median"
    )
    plt.legend()
    plt.show(block=False)

    print("Redshift from input settings (for reference)        : {}".format(z_gal))
    new_z = (np.nanmedian(fit_centers) / center_wavelength) * (1 + z_gal) - 1
    print("Redshift calculated from the median of the fit centers: {}".format(new_z))

    change_z = input(
        "Do you want to update the redshift with the calculated value {} ([y]/n)?  ".format(
            new_z
        )
    )
    if change_z.lower().startswith("n"):
        return_z = z_gal
        message = "The original redshift has been kept: {}".format(return_z)
    else:
        return_z = new_z
        message = "The redshift has been updated to {}".format(return_z)

    print("Tweak reshift procedure is finished. " + message)
    return return_z


def interactive_lower_threshold(image, title=""):
    """Create plot and interact with user to determine the lower threshold for valid data.

    The image is plotted, with a mask applied which initially masks the lower 95%
    of data.  A prompt is given in the console, asking for user input. If the user
    enters no input and presses <enter>, that indicates the currently shown level
    has been accepted by the user. Otherwise, the user may input a different
    number. The plot will be redrawn and the input is requested again.

    This function is primarily used to determine the cutoff indicating the
    spaxels containing the most flux, hopefully indicating the galaxy center.
    We then will use those pixels to fit an emission line, and find the centers.
    This can be used to tweak the redshift if desired.

    Parameters
    ----------
    image : :class:`mpdaf.obj.image.Image`
        An mpdaf image we wish to threshold interactively
    title : str, optional
        The title given to the plot displaying the `image`, by default ""

    Returns
    -------
    limit : float
        The deterimined threshold for valid data.
    """
    limit = np.quantile(image.data, 0.95)

    m_img = image > limit
    m_img.plot(zscale=True, title=title, colorbar="v")
    fig = plt.gcf()
    plt.show(block=False)

    while True:
        print("Change the threshold for valid pixels.")
        print(
            "You may try multiple thresholds. Leave the entry blank and press Enter to confirm your choice."
        )
        print("current limit: {}".format(limit))
        new_limit = input(
            "Set new limit: (or leave blank and press Enter to continue)  "
        )
        # if input is convertable to float, redo loop, otherwise exit loop
        try:
            limit = float(new_limit)
        except ValueError or TypeError:
            plt.close()
            return limit
        m_img = image > limit
        plt.close(fig)
        m_img.plot(zscale=True, title=title, colorbar="v")
        fig = plt.gcf()
        plt.show(block=False)


def get_param_values(params, param_name, default_value=np.nan):
    """Retrieve parameter value by name from lmfit objects.

    Parameters
    ----------
    params : :class:`lmfit.model.ModelResult` or :class:`lmfit.parameter.Parameters`
        Input object containing the value you wish to extract
    param_name : str
        The :class:`lmfit.parameter.Parameter` name, whose value will be returned.
        Also may be a :class:`lmfit.model.ModelResult` attribute, such as 'chisqr'
    default_value : Any, optional
        The return value if the function cannot find the `param_name`, by default np.nan

    Returns
    -------
    float, bool, str, or type(`default_value`)

        * If type(`params`) is :class:`~lmfit.parameter.Parameters`: `params`.get(`param_name`).value
        * If type('params`) is :class:`~lmfit.model.ModelResult`:
            * Tries first: `params`.params.get(`param_name`).value
            * Tries second: `params`.get(`param_name`), which allows for ModelResult attributes.
        * If all these fail, returns `default_value`

    See Also
    --------
    get_param_values : Use this version of the function on 1 input object
    vget_param_values : Use this version of the function on an array of input objects.
        This is a vectorized version of this function that you can apply to
        arrays (see: https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html)

    Examples
    --------
    >>> import threadcount.fit
    >>> from lmfit.models import GaussianModel
    >>> model = GaussianModel()
    >>> params = model.make_params()
    >>> threadcount.fit.get_param_values(params,'sigma')
    1.0
    >>> # or use the vectorized version:
    >>> params2 = model.make_params(sigma=4)
    >>> a = np.array([params,params2], dtype=object)
    >>> threadcount.fit.vget_param_values(a,"sigma")
    array([1., 4.])
    """
    # Quick test, because I know sometimes `params` will be None.
    if params is None:
        return default_value

    # The order of the following try/except blocks is from most-nested to least-nested
    # extraction.

    # 1st: assume `params` is actually a lmfit ModelResult, and that we are
    # trying to extract the parameter `param_name` value from that modelresult's params.
    try:
        return params.params.get(param_name).value
    except AttributeError:
        pass

    # 2nd: assume `params` is a lmfit Parameters object, and that we are
    # trying to extract the parameter `param_name` value from it.
    try:
        return params.get(param_name).value
    except AttributeError:
        pass

    # 3rd: This works for everything else. If `params` is a modelresult and
    # if `param_name` is a modelresult attribute, this will return it properly
    # If `params` has no attribute `get` (such as if it is type int), then
    # default value is returned.
    try:
        return params.get(param_name, default_value)
    except AttributeError:
        return default_value


vget_param_values = np.vectorize(get_param_values)


def iter_spaxel(image, index=False):
    """Create an iterator over the spaxels of successive image pixels in a 2d numpy array.

    Each call to the iterator returns the value of the array `image` at a spaxel.
    The first spaxel to be addressed of image is
    pixel 0,0. Thereafter the X-axis pixel index is incremented by one
    at each call (modulus the length of the X-axis), and the Y-axis
    pixel index is incremented by one each time that the X-axis index
    wraps back to zero.

    The return value of iter_spaxel() is a python generator that can be
    used in loops

    Parameters
    ----------
    image : 2d `numpy.ndarray`
       The image to be iterated over.
    index : bool
       If False, return just a value at each iteration.
       If True, return both a value and the pixel index
       of that spaxel in the image (a tuple of image-array
       indexes along the axes (y,x)).

    Yields
    ------
    dtype of `image`
    """
    if index:
        for y, x in np.ndindex(image.shape):
            yield image[y, x], (y, x)
    else:
        for y, x in np.ndindex(image.shape):
            yield image[y, x]


def process_settings(default_settings, user_settings_string=""):
    """Combine the default settings with any user settings.

    Process the user settings and override the default if a corresponding user
    setting exists. Print a warning if a there is a missing user setting.

    Parameters
    ----------
    default_settings : dict
        A dictionary containing all required settings for the script to run.
    user_settings_string : str, optional
        A string (created by json.dumps(dictionary) containing user settings.),
        by default ""

    Returns
    -------
    :class:`types.SimpleNamespace`
        A simple namespace containing the settings, for easier access to attributes.
    """
    if user_settings_string == "":
        return SimpleNamespace(**default_settings)

    # otherwise process them.
    user_settings = json.loads(user_settings_string)

    # determine if there are missing settings in the user's and report them.
    missing = {
        k: default_settings[k] for k in default_settings.keys() - user_settings.keys()
    }
    for k, v in missing.items():
        print("Missing setting {}, using default value {}".format(k, v))
    final_settings = SimpleNamespace(**user_settings)
    final_settings.__dict__.update(**missing)
    return final_settings


def process_settings_dict(default_settings, user_settings=None):
    """Combine the default settings with any user settings.

    Process the user settings and override the default if a corresponding user
    setting exists. Print a warning if a there is a missing user setting.

    Parameters
    ----------
    default_settings : dict
        A dictionary containing all required settings for the script to run.
    user_settings : dict, optional
        A dictionary containing user settings, by default None

    Returns
    -------
    :class:`types.SimpleNamespace`
        A simple namespace containing the settings, for easier access to attributes.
    """
    if not user_settings:  # takes care of "", None, and {}
        return SimpleNamespace(**default_settings)

    # determine if there are missing settings in the user's and report them.
    missing = {
        k: default_settings[k] for k in default_settings.keys() - user_settings.keys()
    }
    for k, v in missing.items():
        print("Missing setting {}, using default value {}".format(k, v))
    final_settings = SimpleNamespace(**user_settings)
    final_settings.__dict__.update(**missing)
    return final_settings


def get_region(rx, ry=None):
    """Select pixels in ellipse of radius rx, ry from (0,0).

    Return an array of np.array([row,col]) that are within an ellipse centered
    at [0,0] with radius x of rx and radius y of ry.

    Parameters
    ----------
    rx : number or list of numbers [rx, ry]
    ry : number

    Returns
    -------
    numpy.ndarray

    """
    # try to process a list if it is given as parameter
    try:
        rx, ry = rx[0], rx[1]
    # expect TypeError if rx is not a list.
    except TypeError:
        pass
    # Defaults to a circle if ry=None
    if ry is None:
        ry = rx

    rx = abs(rx)
    ry = abs(ry)

    rx_int = round(rx)
    ry_int = round(ry)

    indicies = (np.mgrid[-ry_int : ry_int + 1, -rx_int : rx_int + 1]).T.reshape(-1, 2)
    # create boolean array of where inside ellipse is:
    rx2 = rx * rx
    ry2 = ry * ry
    # remember python likes row, column convention, so y comes first.
    inside = (
        indicies[:, 0] * indicies[:, 0] / ry2 + indicies[:, 1] * indicies[:, 1] / rx2
        <= 1
    )

    return indicies[inside]


def get_reg_image(region):
    """Create kernel image from list of pixels.

    The input `region` is typically the output of :func:`get_region`.
    This kernel image is used for spatial averaging, and it's values
    are either 1 (if included in `region`) or 0.

    Parameters
    ----------
    region : list of pixel positions (y, x)
        The list of pixel positions relative to an arbitrary point,
        usually (0,0) in the case of output from :func:`get_region`, to
        set to value 1 in the output image

    Returns
    -------
    2d numpy array
        An array consisting of the smallest area that will encompass the list of
        pixels in `region`, with the relative shape of `region` preserved. The
        array is 0 except for `region` pixels are set to 1.
    """
    # calculate the extent of the list of inputs:
    mins = region.min(axis=0)
    maxs = region.max(axis=0)
    shape = maxs - mins + 1

    # initialize output
    output = np.zeros(shape)

    # shift the pixel list by mins to reference the new origin.
    inside = [tuple(pix - mins) for pix in region]
    # set those pixels in the pixel list to 1.
    output[tuple(zip(*inside))] = 1
    return output


def spatial_average(cube, kernel_image, **kwargs):
    """Apply kernel image smoothing on every spatial image in a cube.

    This function will correctly apply a smoothing image `kernel_image` to the
    data and variance arrays in `cube`. The normalization is properly propegated
    to the variance array.

    Parameters
    ----------
    cube : :class:`mpdaf.obj.cube.Cube`
        The data you want smoothed
    kernel_image : 2d numpy array
        The smoothing image to apply
    **kwargs : dict
        key word arguments passed to :func:`.mpdaf_ext.correlate2d_norm`

    Returns
    -------
    :class:`mpdaf.obj.cube.Cube`
        Spatially smoothed cube.
    """
    # determine if variance array of output should be initialized:
    var_init = None
    if cube.var is not None:
        var_init = np.empty
    # initialize empty loop output:
    output = cube.clone(data_init=np.empty, var_init=var_init)

    # loop over all images in cube, and set the output to output.
    for ima, k in mpdaf.obj.iter_ima(cube, index=True):
        output[k, :, :] = ima.correlate2d_norm(kernel_image)

    return output


def get_SNR_map(cube, signal_idx=None, signal_Angstrom=None, nsigma=5, plot=False):
    """Create Image of signal to noise ratio in a given bandwidth.

    This bandwidth may be selected in 3 different ways:

    1. Choose the indices of the wavlength array to include (`signal_idx`)
    2. Choose the wavelengths to include (`signal_Angstrom`)
    3. Have the program fit a gaussian to the data, and choose how many sigmas
       to include (`nsigma`). (Uses function: :func:`get_SignalBW_idx`)

    If multiple of `signal_idx`, `signal_Angstrom`, and `nsigma` are given, the
    order of preference is as follows: `signal_idx` overrides all others, then
    `signal_Angstrom`, and finally the least preferenced is `nsigma`, which will
    only be used if either `signal_idx` or `signal_Angstrom` are not specified.

    Parameters
    ----------
    cube : :class:`mpdaf.obj.Cube`
        The cube containing data, var, and wave attributes
    signal_idx : array [int, int], optional
        The indices of the wavelength array to use, by default None
    signal_Angstrom : array [float, float], optional
        The wavelengths in Angstroms to use, by default None
    nsigma : float, optional
        Fit a gaussian, and use center wavelength +/- `nsigma` * sigma, by default 5
    plot : bool, optional
        Plot the whole image spectrum and highlight the SNR bandwidth,
        by default False. A tool for troubleshooting/setup.

    Returns
    -------
    :class:`mpdaf.obj.Image`
        An Image where the pixel values indicate the signal to noise in the
        selected bandwidth. Given a Spectrum for each spaxel, the SNR for the
        spaxel is calculated by sum(Spectrum.data)/sqrt(sum(Spectrum.var)).

    Examples
    --------
    Given a Cube with name `this_cube`, then the default bandwidth selection
    is to fit a gaussian, and use the gaussian center +/- 5*sigma. This is
    implemented by the following command:

    >>> import threadcount as tc
    >>> snr_image = tc.fit.get_SNR_map(this_cube)

    To use the same method but change the width to, for example,
    gaussian center +/- 3*sigma, (meaning nsigma=3), then use the following:

    >>> snr_image = tc.fit.get_SNR_map(this_cube, nsigma=3)

    If you know the specific wavelengths of the bandwidth you would like to use,
    (for example, 5000-5020 A) then use the following:

    >>> snr_image = tc.fit.get_SNR_map(this_cube, signal_Angstrom=[5000,5020])

    And finally, if you know the pixel indices (for example, indices 31-60).
    Note, this is an inclusive range, meaning in this case pixel 60 will be
    included in the SNR calculation.

    >>> snr_image = tc.fit.get_SNR_map(this_cube, signal_idx=[31,60])

    """
    if signal_idx is None:
        if signal_Angstrom is None:
            signal_idx = get_SignalBW_idx(cube, nsigma=nsigma, plot=plot)
            plot = False  # This is taken care of inside the function.
        else:
            signal_idx = cube.wave.pixel(signal_Angstrom, nearest=True)

    subcube = cube[signal_idx[0] : signal_idx[1] + 1, :, :]
    if plot is True:
        plt.figure()
        spectrum = cube.sum(axis=(1, 2))
        title = "Total image spectrum"
        try:
            title = " ".join([cube.label, title])
        except AttributeError:
            pass
        spectrum.plot(title=title)
        plt.axvspan(
            *cube.wave.coord(signal_idx),
            facecolor=plt.rcParams["axes.prop_cycle"].by_key()["color"][1],
            alpha=0.25,
            label="SNR range",
            zorder=-3,
        )
        plt.legend()
    subcube_sum = subcube.sum(axis=0)
    result_image = subcube[0].clone()
    result_image.data = subcube_sum.data / np.sqrt(subcube_sum.var)
    return result_image


def get_SignalBW_idx(cube, nsigma=5, plot=False):
    """Determine the wavelength indices containing signal.

    This function computes an average spectrum using the whole `cube`. Then,
    fits a gaussian plus constant (:class:`~threadcount.models.Const_1GaussModel`).
    The gaussian center and sigma, along with `nsigma`, are used to compute and
    return the indices corresponding to
    :math:`[center - nsigma*sigma, center + nsigma*sigma]`.

    The plot option may be used for debugging for a visual of the spectrum and
    the fit, and the computed range.

    Parameters
    ----------
    cube : :class:`mpdaf.obj.Cube`
        The cube containing data, var, and wave attributes
    nsigma : float, optional
        The number of sigmas to include on each side of the gaussian center,
        by default 5
    plot : bool, optional
        Dispaly a plot of the spectrum and fit, with the bandwidth highlighted,
        by default False

    Returns
    -------
    array, [int, int]
        The indices of the wavelength array corresponding to the calculated
        bandwidth.
    """
    ydata = np.nanmean(
        np.nanmean(cube.data, axis=2), axis=1
    )  # gives 1d spectrum average for all of data.

    x = cube.wave.coord()

    gauss_model = models.Const_1GaussModel()
    params = gauss_model.guess(data=ydata, x=x)
    mod_result = gauss_model.fit(ydata, params, x=x)

    center = mod_result.values["g1_center"]
    sigma = mod_result.values["g1_sigma"]
    low = center - nsigma * sigma
    high = center + nsigma * sigma
    if plot is True:
        plt.figure()
        mod_result.plot()
        plt.axvspan(
            low,
            high,
            facecolor=plt.rcParams["axes.prop_cycle"].by_key()["color"][2],
            alpha=0.25,
            label="SNR range",
            zorder=-3,
        )
        plt.legend()
        title = "Total image spectrum"
        try:
            title = " ".join([cube.label, title])
        except AttributeError:
            pass
        plt.suptitle(title)
    xrange = [low, high]

    # find index of nearest element
    xrange_idx = cube.wave.pixel(xrange, nearest=True)

    return xrange_idx


def get_index(array, value):
    """Determine the index of 'array' which is closest to `value`.

    Parameters
    ----------
    array : float or array/list/iterable of floats
        The list of numbers to search. Will be processed with np.array(`array`).
    value : float or array/list/iterable of floats
        The value(s) to search for in `array`

    Returns
    -------
    int or list of ints
        The index (or indices) of array where the value is closest to the search
        value.

    Examples
    --------
    >>> get_index([10,11,12,13,14],[13,22])
    [3, 4]
    >>> get_index(4,[3,0])
    [0, 0]
    >>> get_index([4,0],10)
    0
    """
    array = np.array(array)

    # value may be a list of values.
    try:
        value_iter = iter(value)
    except TypeError:
        # This catches anything if value is not a list.
        return (np.abs(array - value)).argmin()

    return [(np.abs(array - this_value)).argmin() for this_value in value_iter]


def get_aic(model, error=np.nan):
    """Return the aic_real of a successful fit.

    Parameters
    ----------
    model : :class:`lmfit.model.ModelResult`
        The modelresult to extract info from.
    error : float, optional
        The numeric value to assign any unsuccessful modelresult, by default np.nan

    Returns
    -------
    float
        The modelresult's aic_real, or `error`
    """
    try:
        if model.success is True:
            return model.aic_real
    except AttributeError:
        pass

    return error


vget_aic = np.vectorize(get_aic, doc="Vectorized :func:`get_aic`.")


def choose_model_aic_single(model_list, d_aic=-150):
    r"""Determine best modelresult in a list, chosen by computing :math:`{\Delta}aic`.

    Note: we now look at `aic_real`, defined in :meth:`threadcount.lmfit_ext.aic_real`

    This function uses the aic (Akaike Information Criterion) to choose between
    several models fit to the same data. Our general philosophy: choose simpler
    models.

    The default change in aic we consider
    significant (-150) is quite high compared to a standard -10 you may see in
    statistics, since we are intending to identify the model components with
    physical processes in the galaxy. This value was chosen by observing
    fits to several different spectra and choosing the desired number of gaussian
    components by eye, then finding a :math:`{\Delta}aic` which came close to
    accomplishing that.
    via wikipedia: The :math:`exp((AIC_{min} − AIC_i)/2)` is known as the
    relative liklihood of model i.

    The numbers returned begin with 1, not 0 as is usual in python. If no results
    in `model_list` are valid, then -1 will be returned.

    The algorithm goes as follows:

    * Lets say `model_list` = [model1, model2] (note the numbers begin with 1).

      * If model2.aic_real - model1.aic_real < `d_aic`:

        * return 2

      * else:

        * return 1.

    * Lets now say `model_list` = [model1, model2, model3].

      * If model2.aic_real - model1.aic_real < `d_aic`:

        * This means that model2 is better. We will eliminate
          model1 as an option, then apply bullet point 1, with [model2, model3],
          returning whichever number is better (so the return value will be 2 or 3).

      * else:

        * This means that model2 is not better than model1. We will eliminate
          model2 as an option, then apply bullet point 1, using [model1, model3],
          returning either 1 or 3.
        * TODO: I think if we get a choice of 3 from this way, we should flag
          it for manual inspection, since it may only be slightly better than
          model2 and so our philosophy of less complex is better would be violated.


    Parameters
    ----------
    model_list : list of :class:`lmfit.model.ModelResult`
        A list of different model results which have been fit to the same data.
        Right now, the length must be no longer than 3. The order of the models
        is assumed to be from least complex -> more complex.
    d_aic : float, optional
        The change in fit aic (Akaike Information Criterion) indicating
        a significantly better fit, by default -150.

    Returns
    -------
    int
        The index+1 of the model chosen with this algorithm. Returns -1 if all
        models are invalid.
    """
    # Python starts counting at 0 (0-based indexing.). choices starts counting at 1.
    # Pay attention that the returned value is the python index +1

    # return -1 for invalid:
    if model_list is None:
        return -1
    # return 1 if only one choice:
    if len(model_list) == 1:
        return 0 + 1

    # model list is assumed to be in order simple -> complex.
    # philosophy: prefer simpler models.
    aic = vget_aic(model_list, error=np.nan)
    # if all nans, then return -1 (invalid)
    if np.all(np.isnan(aic)):
        return -1
    # print(np.array(aic)-aic[0])

    # now we have different ways of choosing based on if 2 or 3 models:
    # TODO: generalize to more than 3. Should be easy enough, given the
    # explanation of the algorithm in the docstring.
    if len(model_list) == 2:
        if (aic[1] - aic[0]) < d_aic:
            return (
                1 + 1
            )  # these +1's are for translation to human interaction indexing....
        else:
            return 0 + 1
    if len(model_list) == 3:
        if (aic[1] - aic[0]) < d_aic:
            # True, this means 2 gaussians better than 1.
            # Eliminates id 0 as option and do more tests:
            if (aic[2] - aic[1]) < d_aic:
                # True, this means 3 gaussians better than 2. choose this.
                return 2 + 1
            else:
                return 1 + 1
        else:
            # False, this means 2 gaussians not better than 1.
            # Eliminates id 1 as option and do more tests:
            if (aic[2] - aic[0]) < d_aic:
                # True, this means 3 gaussians better than 1. choose this.
                return 2 + 1
            else:
                return 0 + 1
    # safest thing to return is 0 i guess?
    return 0 + 1


def choose_model_aic(model_list, d_aic=-150):
    """Broadcast :func:`choose_model_aic_single` over array.

    Parameters
    ----------
    model_list : array-like, containing :class:`lmfit.model.ModelResult`
        Array representing spatial dimensions and the last dimension contains
        the model result for different models fitted to that spaxel. Works also
        for simply a list of model results for one pixel.
    d_aic : float, optional
        The change in fit aic (Akaike Information Criterion) indicating
        a significantly better fit, by default -150.

    Returns
    -------
    array of shape model_list.shape[:-1] containing int, or int
        Spatial array containing the chosen model number, starting with 1.
        invalid entries are given the value -1.

    See Also
    --------
    :func:`choose_model_aic_single` : A detailed discussion of this function.
    """
    # assume the first dimensions of model_list are spatial and the last is
    # the different models.
    # Handle a single pixel:
    model_list = np.array(model_list)
    shape = model_list.shape
    if len(shape) == 1:
        single = choose_model_aic_single(model_list, d_aic=d_aic)
        return single

    # if we have passed that block, we know we have an array of size shape to loop over.

    # create output
    output = np.empty(shape[:-1], dtype=int)
    for index in np.ndindex(*shape[:-1]):
        output[index] = choose_model_aic_single(model_list[index], d_aic=d_aic)

    return output


def marginal_fits(fit_list, choices, flux=0.25, dmu=0.5):
    """Determine which fits should be inspected by hand.

    We noticed at times that when fitting multiple gaussians, there was often
    a case of an "embedded" gaussian. This is when a small flux narrow
    gaussian was fit at the same center wavelength as the highest flux
    gaussian. This function is intended to identify those cases and ask the
    user to verify that this is actually the desired fit.

    This function analyzes the selected model (provided by the combination of
    `fit_list` and `choices`), and determines if user inspection is needed based
    on the relative characteristics of the gaussians. This procedure depends on
    the analyzed components having parameter names ending in "flux" and "center",
    and was originally programmed to analyze a multi-gaussian model.
    Note that ("amplitude" is used as a fallback for "flux",
    because lmfit's gaussian use this name to denote integrated flux).

    The "main" gaussian is selected as the model
    component with highest flux, lets call this main gaussian `g0`.
    For the other components `g#`, we compute `g#_flux/g0_flux` and
    `g#_center` - `g0_center`. If any component has both the following:


    * `g#_flux/g0_flux < flux`
    * `g#_center - g0_center < dmu`

    then that fit will be flagged for examination.

    Parameters
    ----------
    fit_list : array-like of :class:`lmfit.model.ModelResult`
        This assumes a spatial array (n,m) of ModelResults, with an outer dimension
        varying in the model used for that spaxel. e.g. shape = (3,n,m) for 3
        different models.
    choices : array-like of shape `fit_list[0]` containing int
        This will be used to select which model for a given spaxel will be
        analyzed in this function. For example, if there are 3 models, the value
        for choices must be 1,2,or 3 (or negative to indicate invalid).
    flux : float, optional
        The gaussian flux ratio to main gaussian component indicating that
        this should be inspected by hand, by default 0.25
    dmu : float, optional
        dmu in my head meant delta mu, the change in the x center between a
        gaussian component and the main gaussian, by default 0.5.

    Returns
    -------
    numpy array of boolean
        Array of same shape as choices, where True means the user should inspect
        this spaxel's fit, and False means this spaxel should be okay with the
        automatic guessing.
    """
    # returns boolean array of shape choices.
    # True indicates you need to manually look at them.

    # Python starts counting at 0 (0-based indexing.). choices starts counting at 1.
    # subtract 1 from choices to convert between these 2 indexing regimes.
    chosen_models = np.choose(choices - 1, fit_list, mode="clip")
    output = np.empty_like(choices, dtype=bool)
    # get the gaussian component comparison for the chosen models.

    # 2d index iteration.
    for index, modelresult in np.ndenumerate(chosen_models):
        if (
            modelresult is None
        ):  # this means pixel was not fit, because it didn't pass snr test.
            # therefore user does not need to check.
            output[index] = False
            continue

        # if chosen model fit has not succeeded, then user checks.
        if not modelresult.success:
            output[index] = True
            continue  # continues to next iteration in the for loop.

        # if model is a single gaussian, user does not need to check.
        if get_ngaussians(modelresult) == 1:
            output[index] = False
            continue

        # more than 1 gaussian component:
        # do tests based on flux and dmu parameters.

        # array of [g_flux/g0_flux, g_center - g0_center]
        components = get_gcomponent_comparison(modelresult)

        # test if the conditions are met for any component.
        # if component[0] < flux AND component[1] < dmu, then user must decide.
        # (set to True here.)
        user_decides = False
        for component in components:
            if (component[0] < flux) and (np.abs(component[1]) < dmu):
                user_decides = True
                break  # stop the inner loop because we know the answer already.
        output[index] = user_decides

    return output


def get_gcomponent_comparison(fit):
    """Determine component comparison to this highest flux gaussian.

    This function finds the highest flux gaussian (we will name it g0),
    and returns a list for the other components containing for each entry:
    [g#_flux/g0_flux, g#_center - g0_center].

    Parameters
    ----------
    fit : :class:`lmfit.model.ModelResult`
        The ModelResult to analyze the components for.

    Returns
    -------
    list of [float,float]
        A list containing the list [g#_flux/g0_flux, g#_center - g0_center]
        for each component g# that is not g0.
    """
    prefixes = [comp.prefix for comp in fit.components if "gaussian" in comp._name]
    if len(prefixes) == 1:  # means just one gaussian component
        return []
    # initialize lists for loop
    values = []
    centers = []
    for pre in prefixes:
        # values is the value of the flux parameter, and if the flux parameter
        # doesn't exist, it falls back on trying to find the value of the
        # amplitude parameter.
        values += [
            fit.params.get(pre + "flux", fit.params.get(pre + "amplitude")).value
        ]
        centers += [fit.params[pre + "center"].value]
    # ID the index of the maximum flux.
    maxval_idx = np.argmax(values)

    # convert to numpy array for easier math.
    values = np.array(values)
    centers = np.array(centers)

    # Column stack allows us to retrieve, e.g. output[0] = [flux/flux0, center-center0]
    # note that inside here, we remove the maximum-flux gaussian.
    output = np.column_stack(
        [
            np.delete(values / values[maxval_idx], maxval_idx),
            np.delete(centers - centers[maxval_idx], maxval_idx),
        ]
    )
    return output


def get_ngaussians(fit):
    """Determine the number of gaussians in a :class:`lmfit.model.Model`.

    Parameters
    ----------
    fit : :class:`lmfit.model.Model` or :class:`lmfit.model.ModelResult`
        The model to analyze.

    Returns
    -------
    int
        The number of components whose name contains "gaussian".
    """
    return sum(["gaussian" in comp._name for comp in fit.components])


def interactive_user_choice(fits, choices, user_check, baseline_fits=None):
    """Choose best model from a display of all model fits to a pixel.

    This function goes through each spaxel flagged for user verification (via
    `user_check` parameter). It displays a figure showing all fit options,
    highlighting in yellow which fit option the aic algorithm has auto-chosen
    via :func:`choose_model_aic` (the `choices` parameter).

    The user is then prompted in the console to enter the number corresponding
    to the fit they would like to choose, usually a number 1-3. Pressing "enter"
    without inputting a number will keep the default choice.

    At any point, the user may cancel further interaction by entering 'x' instead
    of the number 1-3, and the remaining
    pixels will be set to the auto chosen value (given by `choices` parameter).

    Beware: For an example of 3 fits to choose from, any input that is not
    1, 2, 3, or x will keep the default choice and move on to the next pixel,
    without verification or warning.

    Parameters
    ----------
    fits : numpy array of :class:`lmfit.model.ModelResult`
        array of shape e.g. (3,n,m) for 3 models for spatial shape (n,m)
    choices : numpy array of int
        array of shape (n,m) containing which model, starting with 1, is the
        automatic choice made by the aic algorithm :func:`choose_model_aic`. A
        pixel value of -1 indicates invalid fits.
    user_check : numpy array of bool
        array of shape (n,m) indicating if this procedure should be run on that
        pixel.

    Returns
    -------
    numpy array of int
        array of shape (n,m) contining all the choices, including any changes
        made by the user during this procedure.
    """
    # Initialize output as a copy of the input choices.
    final_choices = choices.copy()

    # get a list of the valid user input strings: e.g. ["1", "2", "3"]
    # remember choices are stored beginning at 1, not 0.
    valid_user_entries = [str(x + 1) for x in range(len(fits))]
    message = (
        "Please enter choice 1-{}, or x to cancel further checking.\n"
        "Invalid options will keep default choice.\n".format(len(fits))
    )
    for this_pix, check in np.ndenumerate(user_check):
        if not check:
            # skip this spaxel if not flagged for checking.
            continue
        else:
            # create figure panel.
            fig = plot_ModelResults_pixel(
                [x[this_pix] for x in fits],
                title=str(this_pix) + " User Check",
                computer_choice=choices[this_pix],
                user_checked=None,
                user_choice=None,
            )
            # create a second fig panel for the baseline fit:
            fig_bl = None
            if baseline_fits is not None:
                if baseline_fits[this_pix]:
                    fig_bl = plot_baseline(baseline_fits[this_pix])
            # display in non-blocking way, to enable console interaction.
            plt.show(block=False)

            # ask user to choose which fit they like.
            this_choice = input(
                message
                + "Leave blank for default for pixel {} [{}]:  ".format(
                    str(this_pix), choices[this_pix]
                )
            )
            plt.close(fig)
            plt.close(fig_bl)

            # process user input --

            # strip whitespace:
            this_choice = this_choice.strip()

            # 3 outcomes for this_choice:
            # cancel further iterations, assign new choice, don't change anything.

            # check for cancellation:
            if this_choice.lower() == "x":
                print("Further user examination is cancelled.")
                break  # breaks further execution of the loop.

            # check for valid options:
            if this_choice in valid_user_entries:
                final_choices[this_pix] = int(this_choice)

            # for all other entries, change nothing... meaning there's nothing to do.
    return final_choices


def plot_ModelResults_pixel(  # noqa: C901
    fitList, title="", computer_choice=None, user_checked=None, user_choice=None
):
    r"""Create a multipanel figure comparing fits to the same data.

    This function will call :meth:`lmfit.model.ModelResult.plot_residuals` and
    :meth:`lmfit.model.ModelResult.plot_fit` for each fit in `fitList`.

    The background of the `computer_choice` fit is highlighted in lightyellow.
    The background of the `user_choice` is highlighted in azure. In cases where
    `user_choice` is the same as `computer_choice`, the background is azure.

    Additional fit information is added as a text box to each plot. This
    information consists of:

    * Fit statistics:

        * :math:`\chi^2` - chi-square
        * :math:`\chi_\nu^2` - reduced chi-square
        * :math:`\Delta aic_{{n,n-1}}` - The Akaike info criterion for
          this panel (`n`) - the AIC for the previous panel (`n-1`). For the
          first panel, there exists no previous panel so I set
          :math:`\Delta aic_{{n,n-1}}` = 0, even though it should probably be
          'n/a'.

    * Information about relative gaussian parameters for models containing more
      than 1 gaussian. The component with the highest flux is defined as
      component 0.

        * :math:`flux_n/flux_0` - ratio of component n flux to component 0 flux.
        * :math:`\Delta\mu_n` - the x center of component n minus the x center
          of componenet 0.

    Note that each component is plotted as the fitted constant + gaussian component.
    This allows for easier visual comparison of the contributions of the shape
    of each gaussian component.

    Parameters
    ----------
    fitList : list of :class:`lmfit.model.ModelResult`
        A list of fits of different models to the same data set.
    title : str, optional
        The figure's suptitle, by default "". The function will append to `title`
        "auto choice _", and if the user has checked the fits,
        "user choice _" is also appended.
    computer_choice : int, optional
        The automatic choice, usually by :func:`choose_model_aic`, by default None.
        If None, :func:`choose_model_aic` will be called.
    user_checked : bool, optional
        If True, indicates this pixel was checked by the user, by default None.
        This is often the output for this spaxel's :func:`marginal_fits`
    user_choice : int, optional
        The fit chosen by the user, if any, by default None

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The figure instance.
    """
    # sometimes the fit will contain None because that's what happens at low SNR.
    # Go ahead and plot something I guess, showing pixel label and no fits.
    no_fits = all([v is None for v in fitList])
    if no_fits:
        fig = plt.figure(figsize=(4 * len(fitList), 1), num="No Fits")
        plt.text(
            0.5,
            0.5,
            "{} {}".format(title, "No Fits"),
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
            fontsize=14,
        )
        return fig

    fig = plt.figure(figsize=(4 * len(fitList), 5))
    gs = plt.GridSpec(nrows=2, ncols=len(fitList), height_ratios=[1, 4])

    # we'll be including delta aic in the plot panels, so compute delta aic
    aic_list = [np.nan if fit is None else fit.aic_real for fit in fitList]
    aic_diff = np.diff(aic_list, prepend=aic_list[0])

    # determine automatic choice:
    if computer_choice is None:
        aic_choice = choose_model_aic(fitList)
    else:
        aic_choice = computer_choice

    # explicitly put into title the automatic choice.
    title_string = "{} auto choice {}".format(title, aic_choice)

    # There's probably a better way to do this.
    # create the panel for each fit.
    for col, this_fit in enumerate(fitList):
        ax_res = fig.add_subplot(gs[0, col])
        ax_fit = fig.add_subplot(gs[1, col], sharex=ax_res)
        # Python starts counting at 0 (0-based indexing.). choices starts counting at 1.
        # compare col+1 to choices to convert between these 2 indexing regimes.

        # set the facecolor corresponding to any "choices"
        if col + 1 == aic_choice:
            ax_fit.set_facecolor("lightyellow")
        if user_checked:
            # implied acceptance of computer choice:
            if (user_choice is None) and (col + 1 == aic_choice):
                ax_fit.set_facecolor("azure")
                title_string += ", user choice {}".format(aic_choice)
            # for an explicitly set user_choice.
            if col + 1 == user_choice:
                ax_fit.set_facecolor("azure")
                title_string += ", user choice {}".format(user_choice)
        if this_fit is None:
            continue

        # plot the residuals and the fit
        this_fit.plot_fit(ax=ax_fit)
        this_fit.plot_residuals(ax=ax_res)

        plt.setp(ax_res.get_xticklabels(), visible=False)
        ax_fit.set_title("")
        plt.sca(ax_fit)

        # setup text info box:
        textstr = "\n".join(
            (
                r"$\chi^2$ {:.2g}".format(this_fit.chisqr),
                r"$\chi_\nu^2$ {:.2g}".format(this_fit.redchi),
                r"$\Delta aic_{{n,n-1}}$ {:.2g}".format(aic_diff[col]),
            )
        )
        components = get_gcomponent_comparison(this_fit)
        for i, comp in enumerate(components):
            textstr += "\n" + r"$flux_{}/flux_0$ {:.2g}".format(i + 1, comp[0])
            textstr += "\n" + r"$\Delta\mu_{}$ {:.2g}".format(i + 1, comp[1])

        ax_fit.text(
            0.025, 0.975, textstr, transform=ax_fit.transAxes, verticalalignment="top"
        )
        plt.legend(loc=1)

        # plot the components, with the constant offset added to every gaussian
        # component.
        comps = this_fit.eval_components()
        constant = comps.get("constant", 0)
        if len(comps) > 2:
            for (k, v) in comps.items():
                try:
                    plt.plot(this_fit.userkws["x"], v + constant, label=k)
                except ValueError:
                    pass
        # set the title of the panel to be the number of gaussians.
        ax_res.set_title("{:.1g} Gaussian".format(get_ngaussians(this_fit)))

        # add a warning if the fit didn't converge.
        if this_fit.success is False:
            ax_fit.set_title("!! NOT CONVERGED")
    fig.suptitle(title_string)
    return fig


def save_fit_stats(
    filename=None, dataset=None, fit_info="auto", model_keys="auto", snr=None
):
    """Compile fit parameter summary and save to file.

    This function will iterate over `dataset` (an array containing
    :class:`lmfit.model.ModelResult` for each spaxel of the image), and extract
    the information for the fit quality (attributes listed in `fit_info`) and the
    fit parameters and their errors (parameter names given by `model_keys`).

    These data entries, along with the spaxel coordinates given in (y, x) form,
    and the signal to noise value (if the `snr` parameter is specified) will
    be written in columns to `filename` (if given), and returned as a list of
    lists. (a list of the rows, where each row contains a list of the column
    entries.)

    The parameter `fit_info` must be a list of the attributes to record, or may
    be the string "auto". If it is "auto", the constant `DEFAULT_FIT_INFO`
    defined in this module will be used.

    The parameter `model_keys` must be a list of the parameter names to record,
    or the special strings "auto" or "all". If either of these special strings
    is indicated, then `model_keys` will be set to the output of the function
    :func:`get_model_keys`, run on the first entry of `dataset` which is not None,
    with the parameter `ignore` = ['fwhm'] or = None (for "auto" and
    "all" respectively.)

    Parameters
    ----------
    filename : str
        Name of output file
    dataset : iterable of lmfit.model.ModelResult
        Loop over this to extract fit information
    fit_info : list of str or "auto", optional
        List of lmfit.model.ModelResult attributes to be recorded, by default "auto"
    model_keys : list of str, "auto", or "all", optional
        The ModelResult.params parameter names to be recorded, by default "auto".
        The default will get this automatically with get_model_keys
    snr : numpy.ndarray or None
        If not None, will save the snr to the parameters file.
    """
    # fill in the default parameters:
    if fit_info == "auto":
        fit_info = DEFAULT_FIT_INFO
    if model_keys == "auto":
        model_keys = get_model_keys(dataset, ignore=["fwhm"])
    elif model_keys == "all":
        model_keys = get_model_keys(dataset, ignore=None)

    fit_stats = {"fit_info": fit_info, "model_keys": model_keys}
    if snr is not None:
        snr_label = ["SNR"]
    else:
        snr_label = []
    header_row = ["pixel_tuple"] + snr_label + get_header_stats(**fit_stats)
    collected_results = [header_row]
    # 2d index iteration.
    snr_data = []  # concatenates empty list if snr is None
    for this_pix, model_result in np.ndenumerate(dataset):
        if snr is not None:
            snr_data = [snr[this_pix]]
        collected_results += [
            [this_pix] + snr_data + collect_stats(model_result, **fit_stats)
        ]
    if filename is not None:
        save_to_file(filename, collected_results)
    return collected_results


def save_choice_fit_stats(
    filename, fit_results, choices, fit_info="auto", model_keys="auto", snr=None
):
    """Compile fit choice and parameter summary, and save to file.

    Parameters
    ----------
    filename : str
        Name of output file
    fit_results : list of iterable of lmfit.model.ModelResult
        All the fit options to choose from
    choices : iterable of int
        Which fit_result to choose for each pixel in iterable
    fit_info : list of str or "auto", optional
        List of lmfit.model.ModelResult attributes to be recorded, by default "auto"
    model_keys : list of str, optional
        The ModelResult.params parameter names to be recorded, by default "auto".
        The default will get this automatically with get_model_keys
    snr : numpy.ndarray or None
        If not None, will save the snr to the parameters file.
    """
    # fill in the default parameters:
    if fit_info == "auto":
        fit_info = DEFAULT_FIT_INFO
    if model_keys == "auto":
        # gets the biggest amount of keys.... assumes the last entry is most # gaussians.
        model_keys = get_model_keys(fit_results[-1], ignore=["fwhm"])
    # Python starts counting at 0 (0-based indexing.). choices starts counting at 1.
    # subtract 1 from choices to convert between these 2 indexing regimes.
    model_choices = np.choose(choices - 1, fit_results, mode="clip")
    fit_stats = {"fit_info": fit_info, "model_keys": model_keys}
    if snr is not None:
        snr_label = ["SNR"]
    else:
        snr_label = []
    header_row = (
        ["pixel_tuple", "num_gaussians"] + snr_label + get_header_stats(**fit_stats)
    )
    collected_results = [header_row]
    snr_data = []  # concatenates empty list if snr is None
    # 2d index iteration.
    for this_pix, model_result in np.ndenumerate(model_choices):
        if snr is not None:
            snr_data = [snr[this_pix]]
        collected_results += [
            [this_pix, choices[this_pix]]
            + snr_data
            + collect_stats(model_result, **fit_stats)
        ]
    save_to_file(filename, collected_results)


def save_to_file(filename, result_list):
    """Convience wrapper of csv.writer."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f, dialect="excel-tab")
        for row in result_list:
            formatted_row = [
                format(x, FLOAT_FMT) if isinstance(x, (np.floating, float)) else x
                for x in row
            ]
            writer.writerow(formatted_row)


def get_model_keys(model_results, ignore=None):
    """Retrieve the names of parameters from model results.

    Allows you to ignore some parameters, typically fwhm and height in gaussian,
    as these are calculated from the other parameters.

    Parameters
    ----------
    model_results : lmfit.model.ModelResult, or list of lmfit.model.ModelResult
        The first entry which is not None will be analyzed.
    ignore : str or list of str, optional
        Parameters ending in these entries will not be returned, by default None.
        If str, ignore will be split to a list on whitespace.

    Returns
    -------
    list of str
        The alphabetized parameter names found in the model_result which do not end in ignored values
    """
    # fix so that the loop doesn't break if model_results is just one
    try:
        model = next((x for x in model_results.flat if x is not None), None)
    except AttributeError:
        if isinstance(model_results, lmfit.model.ModelResult):
            model = model_results
        else:
            try:
                model = next(
                    (x for x in np.array(model_results).flat if x is not None), None
                )
            except AttributeError:
                model = None
    if model is None:
        return []

    # create tuple from ignore_key_endings entry (list/string).
    if ignore is not None:
        if isinstance(ignore, str):
            ignore = tuple(ignore.split())
        else:
            ignore = tuple([x.strip() for x in ignore])
    else:
        ignore = ()
    if isinstance(model, lmfit.Model):
        params = model.make_params()
    else:
        params = model.params
    return sorted([k for k in params.keys() if not k.endswith(ignore)])


def get_header_stats(model_keys=None, fit_info="auto"):
    """Collect list of column names, suitable for a header row.

    Parameters
    ----------
    model_keys : list of str, optional
        List of lmfit.model.ModelResult parameter names, by default None
    fit_info : list of str, or "auto", optional
        List of lmfit.model.ModelResult attributes, by default "auto"

    Returns
    -------
    list of str
        Intended to be column names for an output file

    See Also
    --------
    collect_stats : The data entries to go with this header.
    """
    if fit_info == "auto":
        fit_info = DEFAULT_FIT_INFO
    elif fit_info is None:
        fit_info = []

    header = fit_info.copy()
    try:
        for k in model_keys:
            header += [k, k + "_err"]
    except TypeError:  # expected if model_keys is None.
        pass
    return header


def collect_stats(model_result, model_keys=None, fit_info="auto", empty_value=None):
    """Collect the information from one model_result.

    Parameters
    ----------
    model_result : lmfit.model.ModelResult
        This object contains the information to be extracted.
    model_keys : list of str, optional
        The ModelResult.params parameter names to be recorded, by default None
    fit_info : list of str or "auto", optional
        List of lmfit.model.ModelResult attributes to be recorded, by default "auto"
    empty_value : None, np.nan, or any other, optional
        The value to be recorded for any invalid entries found, by default None

    Returns
    -------
    list of information, mostly numbers or boolean
        The information retrieved from model_result

    See Also
    --------
    get_header_stats : The column labels corresponding to this information
    """
    if fit_info == "auto":
        fit_info = DEFAULT_FIT_INFO
    elif fit_info is None:
        fit_info = []

    if model_keys is None:
        model_keys = []
    try:
        this_row = []
        for info in fit_info:
            this_row += [getattr(model_result, info)]
        for k in model_keys:
            param = model_result.params.get(k)
            if param is None:
                this_row += 2 * [empty_value]
            else:
                this_row += [
                    model_result.params[k].value,
                    model_result.params[k].stderr,
                ]
        return this_row
    except AttributeError:  # expected if model_result is not correct type
        return [empty_value] * (2 * len(model_keys) + len(fit_info))


def save_pdf_plots(  # noqa: C901
    filename,
    fitList,
    computer_choices=None,
    user_check=None,
    final_choices=None,
    onlyChecked=False,
    title="",
):
    """Save plots of each spaxel fit as separate page in 1 pdf file.

    This function saves plots of all the different fits to each spaxel, complete
    with indications of which fit was chosen by the aic algorithm (yellow) and
    which (if any) was chosen by the user (blue). A subset for only the fits
    selected for manual checking may be saved, using the `onlyChecked` = True
    option.

    Parameters
    ----------
    filename : str
        file name to save to.
    fitList : list of array of :class:`lmfit.model.ModelResult`
        A list of fits of different models to the same data set, for a spatial
        shape of nxm and 3 different models, this would have the shape (3,n,m)
    computer_choice : array of int, optional
        The automatic choice, usually by :func:`choose_model_aic`, by default None.
        If None, :func:`choose_model_aic` will be called. Same spatial shape as
        `fitList` (nxm)
    user_check : array of bool, optional
        If True, indicates this pixel was checked by the user, by default None.
        This is often the output for this spaxel's :func:`marginal_fits`.
        Same spatial shape as `fitList` (nxm)
    final_choices : array of int, optional
        The fits chosen by the user, if any, by default None.
        Same spatial shape as `fitList` (nxm)
    onlyChecked : bool, optional
        Whether to save only the fits flagged for manual checking, by default False
    title : str, optional
        The figure's suptitle, by default "". The function will prepend to `title`
        the spaxel's coordinates, and append
        "auto choice _", and if the user has checked the fits,
        "user choice _" is also appended.


    See Also
    --------
    plot_ModelResults_pixel : The plotting function used for each spaxel.
    """
    # I'm sure there's a better way to do this but I can't think about it now.

    # easy case: if onlyChecked is True but user_check is None or contains
    # no True values, then exit function.
    if onlyChecked:
        if (user_check is None) or (not user_check.any()):
            print("No user checked pixels. Skipping save user checked plots.")
            return
    # If the inputs are None, make it so that when looping later,
    # it can be addressed correctly.
    try:
        emptyNone = np.full(fitList[0].shape, None)
    except AttributeError:
        emptyNone = None
    if computer_choices is None:
        computer_choices = emptyNone
    if user_check is None:
        user_check = emptyNone
    if final_choices is None:
        final_choices = emptyNone

    print("Saving plots to {}, this may take a while.".format(filename))
    total = np.size(computer_choices)
    count = 0

    # if there are no fits for a given pixel, record that pixel here
    # to deal with later.
    lst_no_fits = []
    with PdfPages(filename) as pdf:
        # loops over all spaxels:
        for this_pix in np.ndindex(computer_choices.shape):
            if onlyChecked:
                if not user_check[this_pix]:
                    continue  # skip this one.
            this_fitList = [x[this_pix] for x in fitList]
            no_fits = all([v is None for v in this_fitList])
            if no_fits:
                lst_no_fits += [str(this_pix)]
                count += 1
                if count % 100 == 0:
                    print("Saved {}/{}".format(count, total))
                continue

            # create the figure to save.
            fig = plot_ModelResults_pixel(
                this_fitList,
                title=str(this_pix) + " " + title,
                computer_choice=computer_choices[this_pix],
                user_checked=user_check[this_pix],
                user_choice=final_choices[this_pix],
            )

            # intelligent iterrupt to allow you to view the pdf
            try:
                pdf.savefig(fig)
            except KeyboardInterrupt:
                raise
            plt.close(fig)

            # silly progress counter.
            count += 1
            if count % 100 == 0:
                print("Saved {}/{}".format(count, total))

        # # deal with the list of not-fit spaxels.

        # # This section should add a pdf page plot which is just a list of
        # # spaxel coordinates with no plot associated.

        # # The intention here was you
        # # could search for a spaxel coordinates, and still find it even if there
        # # were no fits. My pdf viewer didn't work like that though.... so maybe
        # # this can be eliminated.

        # # add 1 more page if we looped over a spaxel that wasn't plotted.
        # if len(lst_no_fits) > 0:
        #     pix_per_line = int(len(fitList) * 10 / 3.0)  # 10 works well if 3 plots
        #     num_lines = int(2 + np.ceil(len(lst_no_fits) / pix_per_line))
        #     str_no_fits = "\n".join(
        #         [
        #             ", ".join(lst_no_fits[x : x + pix_per_line])
        #             for x in range(0, len(lst_no_fits), pix_per_line)
        #         ]
        #     )
        #     fontsize = 14
        #     fig_height = (num_lines * (1 / 72.0) * (fontsize + 2)) + 2 * 0.04167
        #     fig = plt.figure(figsize=(4 * len(fitList), fig_height), num="No Fits")
        #     plt.text(
        #         0.5,
        #         0.5,
        #         "No Fits\n{}\nEnd of List".format(str_no_fits),
        #         horizontalalignment="center",
        #         verticalalignment="center",
        #         transform=plt.gca().transAxes,
        #         fontsize=fontsize,
        #     )
        #     plt.gca().set_axis_off()
        #     pdf.savefig(fig)
        #     plt.close(fig)
    print("Finished saving to {}".format(filename))


# def compile_spaxel_info_mc(mc_fits, keys_to_save):
#     # handle special case inputs:
#     if mc_fits is None:
#         return None
#     if len(mc_fits) == 0:
#         # gives an entry for every output desired: median, std, and median_err for each key
#         return [None] * 3 * len(keys_to_save)

#     temp = save_fit_stats(None, mc_fits, fit_info=None, model_keys="all")
#     names = temp[0][1:]
#     info = np.ma.masked_equal([t[1:] for t in temp[1:]], None)
#     # These next lines take care of None values for any parameter error (e.g. as when given when the)
#     # fit converges at the parameter limit. If all have None, will enter 0.
#     info.set_fill_value(0)
#     info = info.astype("float")

#     mc_medians = np.ma.median(info, axis=0).filled()
#     mc_std = np.ma.std(info, axis=0).filled()

#     data_row = []
#     for key in keys_to_save:
#         try:
#             ix = names.index(key)
#             # label_row += ["median_{}".format(key), "stdev_{}".format(key)]
#             data_row += [mc_medians[ix], mc_std[ix]]

#             # label_row += ["av_error_{}".format(key)]
#             ix = names.index(key + "_err")
#             data_row += [mc_medians[ix]]
#         except ValueError:
#             data_row += 3 * [None]
#     return data_row


# def create_label_row_mc(keys_to_save):
#     label_row = []
#     for key in keys_to_save:
#         label_row += ["median_{}".format(key), "stdev_{}".format(key)]
#         label_row += ["av_error_{}".format(key)]
#     return label_row


def extract_spaxel_info(
    fit_results, fit_info=None, model_params=None, result_dict=None, names_only=False
):
    """Extract ModelResult attributes into numpy array.

    This function broadcasts :meth:`threadcount.lmfit_ext.summary_array` to all ModelResult
    in `fit_results` and handles None values.

    If `names_only` is True, the output is a list containing the attribute names.
    (simply `fit_info` + `model_params`.)

    If `result_dict` is not None, the output will be appendeded to `result_dict`,
    otherise a new :class:`ResultDict` will be returned. The keys in the dict
    correspond to the entries in `fit_info` and `model_params`.

    Each entry of results[key] will have the same spatial shape as `fit_results`,
    giving an easy way of viewing spatial maps of these parameters.

    Parameters
    ----------
    fit_results : Array of :class:`lmfit.model.ModelResult`
        The set of ModelResults to extract the information from.
    fit_info : list of string
        Any attribute that can return a float value from :class:`lmfit.model.ModelResult`,
        e.g. ["chisqr", "aic", "aic_real", "success"]
        (aic_real is defined in :mod:`threadcount.lmfit_ext` module)
    model_params : list of str
        Options here are the param names, or the param names with "_err" appended.
        As an example, lets say we have a param name = "height", and we wish to
        extract the value of the "height" parameter and it's fit error, then
        `model_params` = ["height", "height_err"]. The full list you can choose
        from for a model is model.make_params().valerrsdict().keys().
    result_dict : :class:`ResultDict`, optional
        Append these results to `result_dict` or create a new one (if None),
        by default None
    names_only : bool, optional
        Return a list of the attribute names this call generates, by default False.
        It is essentially `fit_info` + `model_params`.

    Returns
    -------
    :class:`ResultDict`
        A ResultDict where the keys are the strings in `fit_info` and
        `model_params`, and the values are the numpy arrays of same spatial shape
        as `fit_results`. Essentially images of the parameters.

    See Also
    --------
    :meth:`threadcount.lmfit_ext.valerrsdict`
    :meth:`threadcount.lmfit_ext.aic_real`
    """
    if fit_results is None:
        fit_results = []
    if fit_info is None:
        fit_info = []
    if model_params is None:
        model_params = []

    label_row = fit_info + model_params
    if names_only is True:
        return label_row

    fit_results = np.array(fit_results)
    spatial_shape = fit_results.shape
    # create empty output
    output = np.empty((len(label_row),) + spatial_shape)
    for index in np.ndindex(spatial_shape):
        this_fit = fit_results[index]
        if this_fit is None:
            save_info = np.array(np.broadcast_to(None, (len(label_row),)), dtype=float)
        else:
            save_info = this_fit.summary_array(fit_info, model_params)
        output[(slice(None), *index)] = save_info

    if result_dict is None:
        result_dict = ResultDict(output, names=label_row)
    else:
        for i, label in enumerate(label_row):
            result_dict[label] = output[i]
    return result_dict


def extract_spaxel_info_mc(
    mc_fits, fit_info, model_params, method="median", names_only=False
):
    """Compute the average and standard deviation of the information requested.

    This function takes as input a list of monte carlo iterations of a ModelResult
    and returns the average of the `fit_info`, and average and standard deviation
    of the `model_params`, using method "median" or "mean", provided by `method`.

    For models which have multiple gaussian components, this function also
    incorporates a re-ordering of components
    (see :meth:`threadcount.lmfit_ext.order_gauss`) with the idea that we would
    like to average similar components together.

    It is advisable to also have this function compute the names array for you,
    using `names_only` =True, since the entries in `fit_info` and `model_params`
    will have strings added to the beginning and end.

    Parameters
    ----------
    mc_fits : list of :class:`lmfit.model.ModelResult`
        List of fits to extract and average parameters from.
    fit_info : list of string
        Options here include things like "chisqr","aic_real", "success",
        any attribute that will return a float from ModelResult.attribute
    model_params : list of string
        The list of the model parameter names (don't include "_err", that will be
        added for you.) For example, you could compute this from
        model.make_params().keys()
    method : str, optional
        either "mean" or "median", by default "median". The function used to
        caluclate the average. This will always be "mean" for the "success" info.
    names_only : bool, optional
        Return a list of the "column names" instead of computing the averages,
        by default False

    Returns
    -------
    numpy array of floats, or list of string (in case `names_only` is True)
        A list of the extracted and averaged information.

    Raises
    ------
    ValueError
        If `method` is not "median" or "mean".
    """
    # 1 value for each entry in fit_info, 3 values for each entry in model_params.
    # Therefore the null entry needs to be len(fit_info) + 3*len(model_params)
    # "success" in fit_info will always be mean.
    # model_params will have the param.value, param.stderr medianed and param.value st deviation
    if method not in ("median", "mean"):
        raise ValueError("Function parameter `method` must be 'mean' or 'median'.")

    if names_only is True:
        param_names = []
        for p in model_params:
            # yes yes I know that it might sometimes be 'mean' but I wanted consistent
            # column names and I couldn't think of a term that meant both.
            param_names += ["avg_" + p, "stdev_" + p, "avg_" + p + "_err"]
        return fit_info + param_names
    if mc_fits is None or len(mc_fits) == 0:
        return np.array(
            np.broadcast_to(None, (len(fit_info) + 3 * len(model_params,))), dtype=float
        )
    #         return [None]*(len(fit_info)+3*len(model_params))
    dx = mc_fits[0].userkws["x"][1] - mc_fits[0].userkws["x"][0]
    mc_fits_rec = RecursiveArray(mc_fits)
    mc_fits_rec.params.order_gauss(delta_x=dx)
    if method == "median":
        avgfunc = np.nanmedian
    else:
        avgfunc = np.nanmean
    result = []
    for info in fit_info:
        this_array = np.array(getattr(mc_fits_rec, info).data, dtype=float)
        if info == "success":
            this_result = np.nanmean(this_array)
        else:
            this_result = avgfunc(this_array)
        result += [this_result]

    available_params = sorted(mc_fits[0].model.make_params().keys())
    for p in model_params:
        if p not in available_params:
            result += 3 * [None]
        else:
            this_param = mc_fits_rec.params.get(p)
            this_array = np.array(this_param.value.data, dtype=float)
            result += [avgfunc(this_array), np.nanstd(this_array)]

            this_array = np.array(this_param.stderr.data, dtype=float)
            result += [avgfunc(this_array)]
    return np.array(result, dtype=float)


# def save_cube_txt(filename, header_row, cube_info_to_save, pixel_label="pixel_tuple"):
#     # I will insert the pixel_label to beginning of header_row
#     # This can be set to None to skip this step.

#     # assumes 1st 2 dimensions cube_info_to_save are the spatial dimensions.
#     # and the header row labels the 3rd dimension.

#     # creates numpy array if not alreay one.
#     if not isinstance(cube_info_to_save, np.ndarray):
#         cube_info_to_save = np.array(cube_info_to_save)
#     spatial_shape = cube_info_to_save.shape[0:2]
#     output_obj_array = np.empty(spatial_shape, dtype="object")
#     for pixel in np.ndindex(spatial_shape):
#         try:
#             output_obj_array[pixel] = [pixel] + list(cube_info_to_save[pixel])
#         except TypeError:
#             output_obj_array[pixel] = [pixel] + [cube_info_to_save[pixel]]

#     if pixel_label is not None:
#         final_header = [pixel_label] + header_row
#     else:
#         final_header = header_row

#     if filename is None:
#         return [final_header] + output_obj_array.flatten().tolist()
#     else:
#         save_to_file(filename, [final_header] + output_obj_array.flatten().tolist())


class ResultDict(OrderedDict):
    """Container for ordered dict of numpy ndarrays with save/load functionality."""

    DIM_NAMES = ("panel", "row", "col")
    """Default dimension labels, outer -> inner names."""

    def __init__(
        self,
        data_array=None,
        names=None,
        data_dict=None,
        generate_pixel_coordinates=True,
        comment="",
    ):
        """Create a new ResultDict and possibly generate row/column images.

        This class is intended to give you a way to easily access 'images' of
        various parameters, whether it's a signal-to-noise image or a map of how
        many gaussians are in the best fit to your spaxel, or a map of fit
        parameters. The class also provides methods for saving and loading:
        :meth:`savetxt` and :meth:`loadtxt`.

        A big caveat here is that I have not implemented internal checks to make
        sure that all the data are the same spatial shape.

        The intention is for each entry in the arrays should contain information
        consistently for a given spaxel.

        This constructor will create an OrderedDict from data_dict, and then
        update that dict with key, value pairs calculated from
        (for i in range(len(names))) names[i] : data_array[i]. This means that
        the order of keys will be of data_dict.keys(), then any new entries in
        `names` (any overlapping keys will be overridden with the value from
        `data_array`).

        The default behavior is (`generate_pixel_coordinates` is True) to
        determine the shape of the first value (array) in the OrderedDict,
        and create a coordinate 'image' for all the dimensions necessary, and
        named from :const:`DIM_NAMES`, and moved to the beginning of the dict.
        This will give "columns" in the output file called "row", "col" and contain
        the indices of each spaxel.

        A comment string may be provided, and is saved as a header during
        savetxt, or is read from the header in loadtxt.

        Parameters
        ----------
        data_array : numpy ndarray, optional
            Array where data_array[0] corresponds to names[0], by default None
        names : list of string, optional
            The dictionary keys that will correspond to `data_array`, by default None
        data_dict : dict of numpy ndarray, optional
            Dictionary to start with, by default None
        generate_pixel_coordinates : bool, optional
            Whether to generate "row", "col" entries, by default True
        comment : str, optional
            A string that will be saved/loaded as a header from the txt
            file, by default ""

        Raises
        ------
        ValueError
            When there's a mismatch between the length of `names` and the number of
            entries in `data_array`
        """
        # data_dict will be used to initialize OrderedDict, then data_array along with
        # names will update the ordered dict.
        if data_dict is not None:
            super().__init__(**data_dict)
        else:
            super().__init__()

        # now... make sure names exists and it's length is the length of the first dimension of data_array
        # and insert into self.
        if data_array is not None:
            data_array = np.array(data_array)
            if names is None:
                names = ["data_{}".format(i) for i in range(data_array.shape[0])]
            if data_array.shape[0] != len(names):
                raise ValueError("mismatch between names length and data shape")
            for i in range(len(names)):
                self[names[i]] = data_array[i]

        if (len(self) > 0) and generate_pixel_coordinates:
            example = np.array(next(iter(self.values())))  # this is the first value.
            ndim = example.ndim
            shape = example.shape
            indices = np.indices(shape)

            for i in range(ndim):
                label = self.DIM_NAMES[-1 - i]
                self[label] = indices[
                    -1 - i
                ]  # go backward through label and through indices.

                self.move_to_end(label, last=False)
        self.comment = comment

    def names(self):
        """Return a list of all the keys in the dictionary.

        Returns
        -------
        list of string
            A list of the dictionary keys.
        """
        return list(self.keys())

    def data(self):
        """Prepare for saving with :func:`numpy.savetxt`.

        Returns
        -------
        numpy ndarray
            A flattened numpy array, transposed such that each spaxel's
            information will be on one line in the output file.
        """
        images = np.array(list(self.values()))
        return images.reshape(len(self), -1).T

    def savetxt(self, fname, delimiter="\t", header="", fmt="%" + FLOAT_FMT, **kwargs):
        r"""Save this information to a text file.

        This function translates between this class and :func:`numpy.savetxt`.

        The header will consist of `header` + self.comment + computed label row.
        The `delimiter`, `header`, and `fmt` are updated into the `kwargs` dict
        which is passed straight to numpy.savetxt().

        Parameters
        ----------
        fname : string
            Filename path string
        delimiter : str, optional
            string delimiter to separate columns, by default "\\t"
        header : str, optional
            Any additional header string, by default ""
        fmt : str, optional
            See :func:`numpy.savetxt`, by default "%"+FLOAT_FMT
        **kwargs : dict, optional
            Any further keyword arguments passed to :func:`numpy.savetxt`.
        """
        if kwargs is None:
            kwargs = {}
        label_row = delimiter.join(self.names())
        savetxt_kwargs = {
            "delimiter": delimiter,
            "fmt": fmt,
            "header": "\n".join(filter(None, [header, self.comment, label_row])),
        }
        savetxt_kwargs.update(**kwargs)
        np.savetxt(fname, self.data(), **savetxt_kwargs)

    @classmethod
    def loadtxt(cls, fname, comments="#", delimiter=None, **kwargs):
        """Load a file which has been saved by :meth:`savetxt`.

        This should ideally load that text file into one of these
        ResultDict objects. The function reads in the header comment string,
        and adds all but the final line to the ResultDict.comment attribute.
        The final line is used to create the keys for the entries.

        :func:`numpy.loadtxt` is used to load the data. The keys are searched for
        any entry in DIM_NAMES to search for labels like row or col. If found,
        the data is sorted on those columns, and then each column is reshaped
        into a spatial image. The row and/or col indices arrays are created again
        and compared to the read-in values to ensure our spatial placement was
        accurate.

        Parameters
        ----------
        fname : str
            filename to read in
        comments : str, optional
            The :func:`numpy.loadtxt` keyword, indicating the start of a comment,
            by default "#"
        delimiter : str, optional
            The :func:`numpy.loadtxt` keyword, indicating the string used to
            separate values, by default None (meaning whitespace).

        Returns
        -------
        :class:`ResultDict`
            A reconstructed ResultDict.

        Raises
        ------
        ValueError
            If generated row/col indices do not match the read-in indices.
            This will likely be caused by missing row/col entries, as the method
            I use to reshape the arrays does not deal with that possibility yet.
        """
        if kwargs is None:
            kwargs = {}
        loadtxt_kwargs = {"comments": comments, "delimiter": delimiter}
        loadtxt_kwargs.update(**kwargs)

        # get final line of comments, assumed to be column names.
        with open(fname) as f:
            len_comment = len(comments)
            comment_str = ""
            lastcomment = ""
            for line in f:
                if line.startswith(comments):
                    comment_str += lastcomment + "\n"
                    lastcomment = line[len_comment:].strip()
                else:
                    break
        names = lastcomment.split(delimiter)
        data = np.loadtxt(fname, **loadtxt_kwargs)

        # search for our dimension names in the names array:
        indices = [names.index(label) for label in cls.DIM_NAMES if label in names]

        # # sort by row and column:
        ordering = np.lexsort(tuple([data[:, index] for index in reversed(indices)]))
        data = data[ordering]

        images = data.T

        max_entries = [int(images[index][-1]) for index in indices]

        if len(indices) == 0:
            result = cls(images, names, generate_pixel_coordinates=False)
        else:
            new_shape = [len(names)] + [x + 1 for x in max_entries]
            images = images.reshape(*new_shape)

            suffix = "_readin"
            for index in indices:
                names[index] += suffix
            result = cls(images, names)
            # ensure the entries we had match the ones we made.
            for index in indices:
                readin_name = names[index]
                base_name = readin_name[: -len(suffix)]
                if (result[base_name] != result[readin_name]).any():
                    raise ValueError(
                        "Error matching {} with read in values".format(base_name)
                    )
                else:
                    del result[readin_name]

        result.comment = comment_str.strip()
        return result

    def apply_mask(self, mask, fill_value=np.nan):
        """Fill all non-DIM_NAMES images with fill_value where mask = True.

        Parameters
        ----------
        mask : boolean numpy array of same shape as data
            Where this is True, the data value will be replace by fill value
        fill_value : float, optional
            The value to fill in, by default np.nan
        """
        for k, v in self.items():
            if k in self.DIM_NAMES:
                continue
            else:
                v[mask] = fill_value


class RecursiveArray(UserList):
    """Subclass of list which distributes attribute and function calls recursively."""

    def __init__(self, array=None):
        super().__init__(array)
        if isinstance(self.data[0], (list, np.ndarray)):
            self.data = [self.__class__(x) for x in self.data]

    def __getattr__(self, name):
        """Recursively apply getattr.

        Parameters
        ----------
        name : str
            The attribute to get.

        Returns
        -------
        :class:`RecursiveArray`
            Containing the results of recursively getting attribute.
        """
        result = [getattr(x, name, None) for x in self.data]
        return self.__class__(result)
        # if isinstance(result[0], (str, int, float, bool)):
        #     return result
        # else:
        #     return self.__class__(result)

    def __call__(self, *args, **kwargs):
        """Recursively apply the call function.

        Returns
        -------
        :class:`RecursiveArray`
            Containing the recursive results of the call.
        """
        return self.__class__(
            [x(*args, **kwargs) if x is not None else None for x in self.data]
        )

    def aslist(self):
        """Recursively remove the RecursiveArray class and return list of lists.

        Returns
        -------
        list
            Should be the internal list object, if nested RecursiveArray then will
            be list of lists.
        """
        if isinstance(self.data[0], self.__class__):
            return [x.aslist() for x in self.data]
        else:
            return self.data

    def array(self, dtype=float, **kwargs):
        """Easily convert to numpy array.

        Parameters
        ----------
        dtype : data-type, optional
            numpy dtype, see :func:`numpy.array`, by default float

        Returns
        -------
        numpy ndarray
            Array representing this whole RecursiveArray.
        """
        return np.array(self.data, dtype=dtype, **kwargs)


def remove_baseline(
    cube, subcube_av, this_baseline_range, baseline_fit_type, iterate_over_pixels
):
    # modifies subcube_av in place.
    # returns 2d array fit_results.

    spatial_shape = subcube_av.shape[1:]
    subcube_wave_range = subcube_av.wave.get_range()

    buffer_outside = 5.
    baseline_subcube = cube.select_lambda(
        this_baseline_range[0][0] - buffer_outside,
        this_baseline_range[1][1] + buffer_outside,
    )

    fit_results = np.full(spatial_shape, None, dtype=object)

    for pix in iterate_over_pixels:
        this_spectrum = baseline_subcube[:, pix[0], pix[1]]
        this_fitresult = fit_baseline(
            this_spectrum, this_baseline_range, baseline_fit_type
        )

        # set the output array value
        fit_results[pix] = this_fitresult

        # subtract from subcube_av in place.
        if this_fitresult:
            baseline_model = this_spectrum.clone()
            baseline_model.data = this_fitresult.best_fit
            subcube_av[:, pix[0], pix[1]] -= baseline_model.subspec(*subcube_wave_range)

    return fit_results


def fit_baseline(spectrum, this_baseline_range, baseline_fit_type):
    # mask only the appropriate regions:
    this_spectrum = spectrum.copy()
    this_spectrum.unmask()
    this_spectrum.mask_region(
        lmin=this_baseline_range[0][1], lmax=this_baseline_range[1][0], unit=u.angstrom
    )
    this_spectrum.mask_region(lmin=this_baseline_range[1][1], unit=u.angstrom)
    this_spectrum.mask_region(lmax=this_baseline_range[0][0], unit=u.angstrom)

    if baseline_fit_type == "linear":
        baseline_model = lmfit.models.LinearModel()
    elif baseline_fit_type == "quadratic":
        baseline_model = lmfit.models.QuadraticModel()
    else:
        print("No valid baseline fit type, skipping baseline fitting.")
        return None

    fitresult = this_spectrum.lmfit(baseline_model, method="least_squares")

    return fitresult


def plot_baseline(fitresult):
    x = fitresult.userkws["x"]
    orig_y = fitresult.data
    mask = fitresult.weights.mask
    fit = fitresult.best_fit
    orig_y_nomask = np.ma.array(orig_y, mask=False, keep_mask=False)
    new_y = orig_y_nomask - fit

    all_y = np.array(
        [
            *np.ma.array(orig_y, mask=mask).compressed(),
            *np.ma.array(new_y, mask=mask).compressed(),
            *fit,
        ]
    )
    ax_buffer = 0.1 * (all_y.max() - all_y.min())
    ylim = [all_y.min() - ax_buffer, all_y.max() + ax_buffer]

    fig = plt.figure()
    plt.plot(
        x,
        np.ma.array(orig_y, mask=mask),
        "o",
        color="mediumblue",
        label="fitted points",
        zorder=5,
    )
    plt.plot(x, fit, "--k", label="fit")
    # plt.gca().autoscale(enable=False, axis='y')
    plt.plot(
        x,
        orig_y_nomask,
        color="orangered",
        zorder=-2,
        label="original data",
        linewidth=3,
        alpha=0.8,
    )
    plt.plot(
        x, new_y, color="forestgreen", zorder=-1, label="new", linewidth=3, alpha=0.7
    )
    plt.gca().set_ylim(ylim)
    plt.axhline(0, color="k")
    return fig
