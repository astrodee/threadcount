# examples/ex3.py
import time
import matplotlib.pyplot as plt
import threadcount as tc
from threadcount.procedures import (  # noqa: F401
    open_cube_and_deredshift,
    set_rcParams,
    fit_lines,
    analyze_outflow_extent,
)

start = time.time()
start_cpu = time.process_time()


L_Halpha = tc.lines.Line(6562.819, plus=20, minus=8, label="Halpha")

set_rcParams.set_params()
settings = {
    # open_cube_and_deredshift parameters:
    "data_filename": "eso120_10x10.fits",
    "data_hdu_index": 1,
    "var_filename": "eso120_10x10.fits",
    "var_hdu_index": 2,
    # "continuum_filename": "Red_Cont_PPXF_original.fits",  # Empty string or None if not supplied.
    "mask_spaxel_if_this_many_nans": 20,
    "z": 0.012611957403472651,
    "tweak_redshift": False,
    "tweak_redshift_line": tc.lines.L_OIII5007,
    "setup_parameters": False,
    "comment": "",
}

L_Halpha = tc.lines.Line(6562.819, plus=15, minus=12, label="Halpha")
L_HalphaNII = tc.lines.Line(6562.819, plus=55, minus=45, label="HalphaNII")
L_SII = tc.lines.Line(6730.81, plus=35, minus=45, label="SII")
L_HBeta_small = tc.lines.Line(tc.lines.Hb4861, plus=8, minus=8, label="HbNarrow")
L_4363_small = tc.lines.Line(tc.lines.OIII4363, plus=13, minus=13, label="4363Narrow")


def HalphaNIImodparams():
    # model = tc.models.Const_3GaussModel()
    model = tc.models.Const_3GaussModel_fast()
    model.set_param_hint("deltax", value=-14, vary=True, max=-13, min=-17)
    # model.set_param_hint('deltax', value=-14, vary=False, max=-1, min=-17)
    model.set_param_hint("g1_center", expr="g2_center+deltax")
    model.set_param_hint("g1_sigma", max=3.0)
    model.set_param_hint("g2_center", value=6562.0, vary=True, min=6558.0, max=6570.0)
    model.set_param_hint("g2_sigma", max=3.0)
    model.set_param_hint("deltaxhi", value=21, vary=True, max=23, min=19)
    # model.set_param_hint('deltaxhi', value=21, vary=False)
    model.set_param_hint("g3_center", expr="g2_center+deltaxhi")
    model.set_param_hint("g3_sigma", max=3.0)
    return model


def SIImodparams():
    model = tc.models.Const_2GaussModel()
    model.set_param_hint("deltax", value=-14, vary=True, max=-13, min=-16)
    # model.set_param_hint('deltax', value=-14, vary=False, max=-1, min=-17)
    model.set_param_hint("g1_center", expr="g2_center+deltax")
    model.set_param_hint("g1_sigma", max=5.0)
    model.set_param_hint("g2_center", value=6730.810, vary=True, max=6726, min=6735)
    model.set_param_hint("g2_sigma", max=5.0)
    return model


##### Which Model setup do you want to use
model = HalphaNIImodparams()
# model = SIImodparams()

inst_wave = 6565.0
sigma_inst = (5.866e-8 * inst_wave**2 - 9.187e-4 * inst_wave + 6.040) / 2.3553


### Define a block of monitor pixels
pixels = []
for y in range(295, 300 + 1):
    for x in range(260, 266 + 1):
        pixels.append((y, x))

fit_settings = {
    # fit_lines procedure parameters:
    #
    # If setup_parameters is True, only the monitor_pixels will be fit. (faster)
    "setup_parameters": False,
    "monitor_pixels": pixels,
    #
    # output options
    "output_filename": "eso120_HalphaNII_10x10_mc80",  # saved files will begin with this.
    "save_plots": False,  # If True, can take up to 1 hour to save each line's plots.
    #
    # Prep image, and global fit settings.
    "region_averaging_radius": 2.0,  # smooth image by this many PIXELS.
    "instrument_dispersion": 0.66
    * sigma_inst,  # in Angstroms. Will set the minimum sigma for gaussian fits.
    "lmfit_kwargs": {
        "method": "fast_nelder",
    },  # arguments to pass to lmfit. Should not need to change this.
    "snr_lower_limit": 3.0,  # spaxels with SNR below this for the given line will not be fit.
    #
    # Which emission lines to fit, and which models to fit them.
    #
    # "lines" must be a list of threadcount.lines.Line class objects.
    # There are some preset ones, or you can also define your own.
    "lines": [L_HalphaNII],  # tc.lines.L_OIII5007, tc.lines.L_Hb4861],
    #
    # "models" is a list of lists.  The line lines[i] will be fit with the lmfit
    # models contained in the list at models[i].
    # This means the simplest entry here is: "models": [[tc.models.Const_1GaussModel()]]
    # which corresponds to one model being fit to one line.
    "models": [  # This is a list of lists, 1 list for each of the above lines.
        # 5007 models
        [
            # tc.models.Const_1GaussModel(),
            # tc.models.Const_3GaussModel(),
            model,
        ],
        # hb models
        # [tc.models.Const_1GaussModel()],
    ],
    #
    # If any models list has more than one entry, a "best" model is going to be
    # chosen. This has the options to choose automatically, or to include an
    # interactive portion.
    "d_aic": 0,  # -150,  # starting point for choosing models based on delta aic.
    # Plot fits and have the user choose which fit is best in the terminal
    "interactively_choose_fits": False,
    # Selection of pixels to always view and choose, but only if the above line is True.
    "always_manually_choose": pixels,  # interactively_choose_fits must also be True for these to register.
    #
    # Options to include Monte Carlo iterations on the "best" fit.
    "mc_snr": 80.0,  # SNR below which the monte carlo is run.
    "mc_n_iterations": 5,  # number of monte carlo fits for each spaxel.
    "parallel": True,  # whether invoke multiprocess
    "n_process": 4,  # number of processes used in multiprocessing; activated when parallel is True
}

# combine these settings dictionaries into 1 dict. sadly, I repeated a key,
# and "setup_parameters" will be overridden by its value in fit_settings.
settings.update(fit_settings)


# run the long version of the code. If you have already run this once and the
# line fitting output files are okay, you may
# comment this line out to save time!!
settings = fit_lines.run(settings)


explore_settings = {
    "input_file": "trash_6563_mc_best_fit.txt",  # Threadcount output file, can be any threadcout output txt file
    "data_filename": None,  # file to read the spectra, should be the same fits file you use for the threadcount fit. if set to None it will get the file from the header of the input_file.
    "data_hdu_index": 0,  # the hdu index from where to read the spectral data of data_filename
    "line": L_HalphaNII,  # the line you want to plot, should be a tc.lines object, the sabe you used for the fit
    "plot_map": map,  # an array with values of the map to display, must have the same spatial shape as the data cube.
    # If None it will show a map of the highest fitted gaussian in each spaxel.
    "plot_map_log": True,  # True to show the map in log scale, False to show it in linear scale
}


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
        "manual_galaxy_region": [175, 197, None, None],
        "verbose": True,
        "vertical_average_region": 1,
        "contour_levels": [0.5, 0.9],
        # Select from contour_levels list which contour to choose to define outflow region.
        "outflow_contour_level": 0.5,
        "output_base_name": settings["output_filename"],
        "galaxy_center_pixel": [186, 165],  # row,col
        "velocity_vmax": 200,
        "arcsec_per_pixel": settings.get("wcs_step", "header"),
        "units": "header",
    }
)

# analyze_outflow_extent.run(analyze_settings)
# %%
print("Finished with script.")
set_rcParams.reset_params()
plt.show()

end = time.time()
end_cpu = time.process_time()
print("total running time: ", end - start)
print("total cpu time: ", end_cpu - start_cpu)
# %%
