# examples/ex3.py

import matplotlib.pyplot as plt
import threadcount as tc
from threadcount.procedures import (  # noqa: F401
    open_cube_and_deredshift,
    set_rcParams,
    fit_lines,
    analyze_outflow_extent,
)

set_rcParams.set_params()
settings = {
    # open_cube_and_deredshift parameters:
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
    "setup_parameters": False,
    "monitor_pixels": [(40, 40), (28, 62), (48, 58), (52, 56)],
    #
    # output options
    "output_filename": "ex3_full",  # saved files will begin with this.
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
    "interactively_choose_fits": False,
    # Selection of pixels to always view and choose, but only if the above line is True.
    "always_manually_choose": [
        # (28, 62),
        # (29, 63),
    ],  # interactively_choose_fits must also be True for these to register.
    #
    # Options to include Monte Carlo iterations on the "best" fit.
    "mc_snr": 25,  # SNR below which the monte carlo is run.
    "mc_n_iterations": 20,  # number of monte carlo fits for each spaxel.
}

# combine these settings dictionaries into 1 dict. sadly, I repeated a key,
# and "setup_parameters" will be overridden by its value in fit_settings.
settings.update(fit_settings)


# run the long version of the code. If you have already run this once and the
# line fitting output files are okay, you may
# comment this line out to save time!!
settings = fit_lines.run(settings)


# Create the settings dict for measuring the outflow extent.
#
# I will want to refer to the value of "line", so lets start it out with this
# and update the dict.
#
analyze_settings = {"line": settings["lines"][0]}
analyze_settings.update(
    {
        "one_gauss_input_file": "_".join(
            filter(
                None,
                [
                    settings["output_filename"],
                    analyze_settings["line"].save_str,
                    "simple_model.txt",  # current options here: simple_model.txt, best_fit.txt, mc_best_fit.txt
                ],
            )
        ),
        "velocity_mask_limit": 60,
        # manual_galaxy_region format
        # [min_row, max_row, min_column, max_column] --> array[min_row:max_row+1,min_column:max_column+1]
        # 'None' means it will continue to image edge.
        "manual_galaxy_region": [30, 39, None, None],
        "verbose": True,
        "vertical_average_region": 1,
        "contour_levels": [0.5, 0.9],
        # Select from contour_levels list which contour to choose to define outflow region.
        "outflow_contour_level": 0.5,
        "output_base_name": settings["output_filename"],
        "galaxy_center_pixel": [35, 62],  # row,col
        "velocity_vmax": 140,
        "arcsec_per_pixel": settings.get("wcs_step", "header"),
        "units": "header",
    }
)

analyze_outflow_extent.run(analyze_settings)
# %%
print("Finished with script.")
set_rcParams.reset_params()
plt.show()


# %%
