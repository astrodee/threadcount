import threadcount as tc
from threadcount.procedures import fit_line, open_cube_and_deredshift, set_rcParams


def update_settings(s):
    # "de-redshift" instrument dispersion:
    s.instrument_dispersion_rest = s.instrument_dispersion / (1 + s.z_set)
    if s.always_manually_choose is None:
        s.always_manually_choose = []

    # create a kernel image to use for region averaging.
    region_pixels = tc.fit.get_region(s.region_averaging_radius)
    k = tc.fit.get_reg_image(region_pixels)
    s.kernel = k

    # add to the comments string for saving.
    if s.comment and not s.comment.endswith("\n"):
        s.comment += "\n"
    comment_keys = [
        "instrument_dispersion",
        "region_averaging_radius",
        "snr_lower_limit",
        "mc_snr",
        "mc_n_iterations",
        "lmfit_kwargs",
        "d_aic",
    ]
    s.comment += "\n".join(
        ["{}: {}".format(x, s.__dict__.get(x, None)) for x in comment_keys]
    )
    s.comment += "\nunits: " + s.cube.unit.to_string()


def fit_lines(s):
    num_lines = len(s.lines)
    for i in range(num_lines):
        s._i = i
        s = fit_line.run(s)
    return s


def run(user_settings):
    default_settings = {
        # If setup_parameters is True, only the monitor_pixels will be fit. (faster)
        "setup_parameters": False,
        "monitor_pixels": [  # the pixels to always save.
            # (40, 40),
            # (28, 62),
            # (48, 58),
            # (52, 56),
        ],
        "baseline_subtract": None,  # Options: None, "linear", "quadratic"
        "baseline_fit_range": None,  # a list of: [[left_begin, left_end],[right_begin, right_end]], one for each line.
        #
        # output options
        "output_filename": "example_output",  # saved files will begin with this.
        "save_plots": False,  # If True, can take up to 1 hour to save each line's plots.
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
        "lines": [
            tc.lines.L_OIII5007,
            # tc.lines.L_Hb4861,
        ],
        #
        # "models" is a list of lists.  The line lines[i] will be fit with the lmfit
        # models contained in the list at models[i].
        # This means the simplest entry here is: "models": [[tc.models.Const_1GaussModel()]]
        # which corresponds to one model being fit to one line.
        "models": [  # This is a list of lists, 1 list for each of the above lines.
            # 5007 models
            [
                tc.models.Const_1GaussModel(),
                # tc.models.Const_2GaussModel(),
                # tc.models.Const_3GaussModel(),
            ],
            # hb models
            # [tc.models.Const_1GaussModel()],
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
        # Options to include Monte Carlo iterations on the "best" fit.
        "mc_snr": 25,  # SNR below which the monte carlo is run.
        "mc_n_iterations": 20,  # number of monte carlo fits for each spaxel.
        "parallel": False,  # whether invoke multiprocess (not available on windows)
        "n_process": 4,  # number of processes used in multiprocessing; activated when parallel is True
        "chop_bandwidth": False,  # if fit fails, cut down the x axis by 5AA on each side and try again.
        "SNR_HalfBW": 9,  # calculate the SNR in Line center +/- SNR_HalfBW
    }

    # test if cube has been opened. if not, open cube and deredshift.
    if "cube" not in user_settings.keys():
        user_settings = open_cube_and_deredshift.run(user_settings)
        set_rcParams.set_params({"image.aspect": user_settings["image_aspect"]})
    s = tc.fit.process_settings_dict(default_settings, user_settings)  # s for settings.

    update_settings(s)
    s = fit_lines(s)
    return s.__dict__
