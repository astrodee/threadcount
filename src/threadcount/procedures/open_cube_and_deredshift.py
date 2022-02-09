"""Open a fits data cube and de-redshift it along with any continuum cube."""
import threadcount as tc

# import matplotlib as mpl


def run(user_settings):
    """Open a fits data cube and de-redshift it, as well as any continuum cube.

    The data and variance files are combined into a single cube, and the
    continuum file (if available) is kept separate. An interactive tweak redshift
    procedure may be activated, if `setup_parameters` is True.

    New dict keys are added to the user_settings and returned, with
    user_settings['comment'] having comments appended to it.

    Parameters
    ----------
    user_settings : dict
        required keys:

            * "data_filename": str, filename of data cube.
            * "z": float, estimation of object redshift.

        optional keys:

            * "data_hdu_index": int, index of the data hdu. Default None -- mpdaf
              will attempt to guess.
            * "var_filename": str, filename of the variance cube. Default None.
            * "var_hdu_index": int, index of the variance hdu. Default None -- mpdaf
              will attempt to guess.
            * "continuum_filename": str, filename of the variance cube. Default None.
            * "setup_parameters": bool, run interactive tweak redshift. Default False.
            * "comment": str, default "". Any comment to add to the header of saved
              files.

    Returns
    -------
    (dict, :class:`mpdaf.obj.Cube`, :class:`mpdaf.obj.Cube` or None)
        dict : an updated settings dictionary.
        Cube : the data cube
        Cube 2 : the continuum cube, if continuum filename given, otherwise None.


    Examples
    --------

    >>> from threadcount.procedures import open_cube_and_deredshift
    >>> my_settings = {
    ...     "data_filename": "MRK1486_red_metacube.fits",
    ...     "z": 0.0339,
    ... }
    >>> my_settings = open_cube_and_deredshift.run(my_settings)
    >>> # examine my_settings to see the information added.
    >>> cube = my_settings["cube"]
    """
    # %%
    default_settings = {
        "data_filename": "",
        "data_hdu_index": None,
        "var_filename": None,
        "var_hdu_index": None,
        "continuum_filename": None,  # Empty string or None if not supplied.
        "z": 0.0339,
        "setup_parameters": False,
        "comment": "",
    }

    s = tc.fit.process_settings_dict(default_settings, user_settings)  # s for settings.

    # %%
    cube = tc.fit.open_fits_cube(
        s.data_filename, s.data_hdu_index, s.var_filename, s.var_hdu_index
    )
    if s.continuum_filename not in ("", None):
        continuum_cube = tc.fit.open_fits_cube(s.continuum_filename)
    else:
        continuum_cube = None
    # extract info from wcs and save to settings and mpl.rcParams:
    s.observed_delta_lambda = cube.wave.get_step()
    s.wcs_step = cube.wcs.get_step(tc.fit.u.arcsec)  # returns (dy,dx)
    s.image_aspect = s.wcs_step[0] / s.wcs_step[1]
    # mpl.rcParams["image.aspect"] = s.image_aspect

    # de-redshift:
    s.z_set = tc.fit.de_redshift(cube.wave, s.z, z_initial=0)

    # Fine Adjust redshift:
    if s.setup_parameters is True:
        z_tweak = tc.fit.tweak_redshift(cube, s.z_set)
        s.z_set = tc.fit.de_redshift(cube.wave, z_tweak, s.z_set)
    # de-redshift continuum cube.
    if continuum_cube:
        _ = tc.fit.de_redshift(continuum_cube.wave, s.z_set, z_initial=0)

    # add to the comments string for saving.
    if s.comment and not s.comment.endswith("\n"):
        s.comment += "\n"
    comment_keys = [
        "data_filename",
        "z_set",
        "image_aspect",
        "wcs_step",
        "observed_delta_lambda",
    ]
    s.comment += "\n".join(
        ["{}: {}".format(x, s.__dict__.get(x, None)) for x in comment_keys]
    )
    # put units after finished with cube manipulation.
    # s.comment += "\nunits: " + cube.unit.to_string()

    # maybe do this instead?
    s.cube = cube
    s.continuum_cube = continuum_cube
    return s.__dict__
    # return s.__dict__, cube, continuum_cube
