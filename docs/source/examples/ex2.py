# examples/ex2.py

import threadcount as tc
from threadcount.procedures import (
    open_cube_and_deredshift,
    set_rcParams,
    fit_lines,
)

load_settings = {
    # open_cube_and_deredshift procedure parameters:
    "data_filename": "MRK1486_red_metacube.fits",
    "data_hdu_index": 0,
    "var_filename": "MRK1486_red_varcube.fits",
    "var_hdu_index": 0,
    "continuum_filename": "Red_Cont_PPXF_original.fits",  # Empty string or None if not supplied.
    "z": 0.03386643885613516,
    "tweak_redshift": False,
    "tweak_redshift_line": tc.lines.L_OIII5007,
    "comment": "",
}

fit_settings = {
    # fit_lines procedure parameters:
    #
    # If setup_parameters is True, only the monitor_pixels will be fit. (faster)
    "setup_parameters": True,
    "monitor_pixels": [(40, 40), (28, 62), (48, 58), (52, 56)],
    #
    # output options
    "output_filename": "ex2",  # saved files will begin with this.
    "save_plots": True,  # If True, can take up to 1 hour to save each line's plots.
    #
    # Prep image, and global fit settings.
    "region_averaging_radius": 1.5,  # smooth image by this many PIXELS.
    "instrument_dispersion": 0.8,  # in Angstroms. Will set the minimum sigma for gaussian fits.
    "lmfit_kwargs": {
        "method": "least_squares"
    },  # arguments to pass to lmfit. Should not need to change this.
    "snr_lower_limit": 10,  # spaxels with SNR below this for the given line will not be fit.
    #
    # Which emission lines to fit, and which models to fit them.
    #
    # "lines" must be a list of threadcount.lines.Line class objects.
    # There are some preset ones, or you can also define your own.
    "lines": [tc.lines.L_OIII5007, tc.lines.L_Hb4861],
    #
    # "models" is a list of lists.  The line lines[i] will be fit with the lmfit
    # models contained in the list at models[i].
    # This means the simplest entry here is: "models": [[tc.models.Const_1GaussModel()]]
    # which corresponds to one model being fit to one line.
    "models": [  # This is a list of lists, 1 list for each of the above lines.
        # 5007 models
        [
            tc.models.Const_1GaussModel(),
            tc.models.Const_2GaussModel(),
            tc.models.Const_3GaussModel(),
        ],
        # hb models
        [tc.models.Const_1GaussModel()],
    ],
    #
    # If any models list has more than one entry, a "best" model is going to be
    # chosen. This has the options to choose automatically, or to include an
    # interactive portion.
    "d_aic": -150,  # starting point for choosing models based on delta aic.
    # Plot fits and have the user choose which fit is best in the terminal
    "interactively_choose_fits": True,
    # Selection of pixels to always view and choose, but only if the above line is True.
    "always_manually_choose": [
        (28, 62),
        (29, 63),
    ],  # interactively_choose_fits must also be True for these to register.
    #
    # Options to include Monte Carlo iterations on the "best" fit.
    "mc_snr": 25,  # SNR below which the monte carlo is run.
    "mc_n_iterations": 20,  # number of monte carlo fits for each spaxel.
}

# Open and de-redshift cube
settings = open_cube_and_deredshift.run(load_settings)

# update the rcparams in case of any non-square pixels.
set_rcParams.set_params({"image.aspect": settings["image_aspect"]})

# add the fit_settings to the settings we received back from opening the cube
settings.update(fit_settings)

# Run the fit_lines procedure, which saves output files.
settings = fit_lines.run(settings)
