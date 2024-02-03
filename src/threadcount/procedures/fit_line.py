import numpy as np
import threadcount as tc
from itertools import tee
import multiprocessing as mp
from functools import partial

mpException = None
try:
    ctx = mp.get_context("fork")
except ValueError as exc:
    mpException = exc
    ctx = None


def run(s):  # noqa: C901
    if s.parallel and ctx is None:
        raise ValueError(
            'Cannot run parallel, please set setting "parallel" to False in runner file.'
        ) from mpException
    cube = s.cube
    continuum_cube = s.continuum_cube
    k = s.kernel

    index = s._i
    this_line = s.lines[index]
    print("Processing line {}".format(this_line.label))
    models = s.models[index]
    # initialize baseline results:
    if not hasattr(s, "baseline_results"):
        s.baseline_results = [None] * len(s.lines)

    # create a subcube, and region average each of the wavelength channels.
    # creating a subcube like this is a view into the original cube, and doesn't copy
    # the data, so any change to subcube changes cube.
    subcube = cube.select_lambda(this_line.low, this_line.high)
    subcube_av = tc.fit.spatial_average(subcube, k)

    # repeat for continuum cube if it exists.
    if continuum_cube:
        # subcontinuum is a view, not new data.
        subcontinuum = continuum_cube.select_lambda(this_line.low, this_line.high)
        subcontinuum_av = tc.fit.spatial_average(subcontinuum, k)
    else:
        subcontinuum = None
        subcontinuum_av = None

    # Determine the SNR for each spaxel.
    # This default way to get the snr image:
    snr_image = tc.fit.get_SNR_map(subcube_av)

    # Subtract the continuum:
    if subcontinuum_av:
        subcube_av -= subcontinuum_av

    # clear memory, we are finished with the continuum for this line.
    del subcontinuum
    del subcontinuum_av

    # now, we fit a list of functions to each spaxel in subcube_av, if SNR is high
    # I don't know what setting to put this under, I feel like this should be
    # different for each line?

    # Set limits for these model instances, especially min sigmas.
    for model in models:
        model.set_param_hint_endswith("sigma", min=s.instrument_dispersion_rest)

    snr_threshold = s.snr_lower_limit  # 5
    # mod = tc.models.Const_1GaussModel()
    # create a list of length(models) where each entry is an empty "image" to place
    # model results for each spaxel. fit_results[0] will be the results corresponding
    # to models[0], etc.
    spatial_shape = subcube_av.shape[1:]
    fit_results = np.array([np.full(spatial_shape, None, dtype=object)] * len(models))
    # transpose fit_results for easy addressing by spaxel indices inside loop:
    fit_results_T = fit_results.transpose((1, 2, 0))

    # iterate over every spaxel (i.e. retrieve each spectrum):
    # (choose this direct way of iteration in order to eventually use example pixels)
    iterate_full = np.ndindex(*spatial_shape)
    if len(s.monitor_pixels) > 0 and (s.setup_parameters):
        iterate_full = s.monitor_pixels

    iterate, bl_iterate = tee(iterate_full, 2)

    baseline_fitresults = None
    if s.baseline_subtract:
        this_baseline_range = s.baseline_fit_range[s._i]
        baseline_fit_type = s.baseline_subtract
        baseline_fitresults = tc.fit.remove_baseline(
            cube, subcube_av, this_baseline_range, baseline_fit_type, bl_iterate
        )
        s.baseline_results[s._i] = baseline_fitresults

    if s.parallel:
        pool = ctx.Pool(processes=s.n_process)
        print("start pooling")
        results = pool.map(
            partial(
                process_single_spectrum,
                subcube_av,
                snr_image,
                snr_threshold,
                models,
                s,
            ),
            iterate,
        )
        print("finish pooling")
        fit_results_T[:] = np.array(results, dtype=object).reshape(fit_results_T.shape)
    else:
        for idx in iterate:
            out = process_single_spectrum(
                subcube_av, snr_image, snr_threshold, models, s, idx
            )
            fit_results_T[idx] = out

    s.model_results = fit_results
    print("Finished the fits.")

    # %%
    # make model choices.
    # ugh I just auto-refactored this, but it could have been better, sorry.
    chosen_models, final_choices, auto_aic_choices, user_check = choose_best_fits(
        models, fit_results_T, s, fit_results
    )

    # %%
    # Now do monte carlo iterations if required.

    # create colunm names to save and corresponding labels.
    keys_to_save = tc.fit.get_model_keys(models[-1], ignore=["fwhm"])
    # label_row = tc.fit.create_label_row_mc(keys_to_save)
    fit_info = ["success", "chisqr", "redchi"]
    mc_label_row = tc.fit.extract_spaxel_info_mc(
        None, fit_info, keys_to_save, names_only=True
    )
    # create output arrays:
    img_modelresults = np.empty(spatial_shape, dtype=object)
    img_mc_output = np.empty((len(mc_label_row),) + spatial_shape)

    for index, chosen_model in np.ndenumerate(chosen_models):
        # choose how many iterations of mc:
        if snr_image[index] < s.mc_snr:
            mc_n_iterations = s.mc_n_iterations
        else:
            mc_n_iterations = 0
        if chosen_model is not None:
            fit_list = chosen_model.mc_iter(mc_n_iterations)
            img_modelresults[index] = fit_list

        img_mc_output[(slice(None), *index)] = tc.fit.extract_spaxel_info_mc(
            img_modelresults[index], fit_info, keys_to_save
        )

        # img_mc_output[index] = tc.fit.compile_spaxel_info_mc(fit_list, keys_to_save)

    # %%
    # save files.

    # ugh just auto-refactored this, sorry for the mess.
    save_files(
        s,
        this_line,
        models,
        snr_image,
        fit_results,
        final_choices,
        chosen_models,
        mc_label_row,
        img_mc_output,
        auto_aic_choices,
        user_check,
    )
    return s


def process_single_spectrum(subcube_av, snr_image, snr_threshold, models, s, idx):
    print("index: ", idx)
    sp = subcube_av[(slice(None), *idx)]
    # this below line is how I originally tried this, and it works.
    # for sp, idx in mpdaf.obj.iter_spe(subcube_av, index=True):
    # Test if it passes the SNR test:
    if (snr_image[idx] < snr_threshold) or (np.isnan(snr_image[idx]) is True):
        # fit_results_T[idx] = [None] * len(models)
        return [None]

    # Fit the least complex model, and make sure of success.
    spec_to_fit = sp
    f = spec_to_fit.lmfit(models[0], **s.lmfit_kwargs)
    if f is None:
        # fit_results_T[idx] = [None] * len(models)
        return [None]

    if f.success is False:
        # One reason we saw for 1 gaussian fit to fail includes the iron line when
        # fitting 5007. Therefore, if there is a failure to fit 1 gaussian, I will
        # cut down the x axis by 5AA on each side and try again.
        wave_range = sp.get_range()
        print("cutting spectrum by +/- 5A for pixel {}".format(idx))
        cut_sp = sp.subspec(wave_range[0] + 5, wave_range[1] - 5)
        spec_to_fit = cut_sp
        f = spec_to_fit.lmfit(models[0], **s.lmfit_kwargs)
        if f.success is False:
            # fit_results_T[idx] = [None] * len(models)
            return [None]

    # at this point: if the first model has failed to fit both times, we don't
    # even reach this point, the loop continues. However, if the first model
    # fit the first time, then spec_to_fit = sp. If the first model failed the
    # first time and succeeded the second time, then spec_to_fit = cut_sp.

    # continue with the rest of the models.
    rest = [spec_to_fit.lmfit(model, **s.lmfit_kwargs) for model in models[1:]]
    return [f] + rest


def save_files(
    s,
    this_line,
    models,
    snr_image,
    fit_results,
    final_choices,
    chosen_models,
    mc_label_row,
    img_mc_output,
    auto_aic_choices,
    user_check,
):
    # %%
    # save files.

    # make the output filenames
    # the below line: if s.output_filename is falsey (e.g. None or ""), then
    # base_filename is just this_line.save_str, otherwise, it is joined
    # with "_" as usual.
    base_filename = "_".join(filter(None, [s.output_filename, this_line.save_str]))

    oneG_filename = base_filename + "_simple_model.txt"
    bestG_filename = base_filename + "_best_fit.txt"
    mc_filename = base_filename + "_mc_best_fit.txt"

    # plot output filenames
    user_checked_filename = base_filename + "_fits_user_checked.pdf"
    all_fits_filename = base_filename + "_fits_all.pdf"

    fit_info_to_save = tc.fit.DEFAULT_FIT_INFO
    # save all one-gaussian fits:
    # tc.fit.save_fit_stats(oneG_filename, fit_results[0], snr=snr_image)
    oneG_param_names = sorted(models[0].make_params().valerrsdict().keys())
    oneG_result_dict = tc.fit.ResultDict(data_dict={"snr": snr_image.data.data})

    oneG_result_dict = tc.fit.extract_spaxel_info(
        fit_results[0], fit_info_to_save, oneG_param_names, oneG_result_dict
    )
    oneG_result_dict.comment = s.comment
    oneG_result_dict.savetxt(oneG_filename)

    if len(models) > 1:
        most_param_names = sorted(models[-1].make_params().valerrsdict().keys())
        choice_result_dict = tc.fit.ResultDict(
            data_dict={"snr": snr_image.data.data, "choice": final_choices}
        )
        choice_result_dict = tc.fit.extract_spaxel_info(
            chosen_models, fit_info_to_save, most_param_names, choice_result_dict
        )
        choice_result_dict.comment = s.comment
        choice_result_dict.savetxt(bestG_filename)
        # # save the best choice fits:
        # tc.fit.save_choice_fit_stats(
        #     bestG_filename, fit_results, final_choices, snr=snr_image
        # )

    mc_result_dict = tc.fit.ResultDict(data_dict={"snr": snr_image.data.data})
    if final_choices is not None:
        mc_result_dict["choice"] = final_choices
    # result_array = np.array(img_mc_output, dtype=float)
    mc_result_dict.update({key: val for key, val in zip(mc_label_row, img_mc_output)})
    mc_result_dict.comment = s.comment
    mc_result_dict.savetxt(mc_filename)
    # tc.fit.save_cube_txt(mc_filename, label_row, img_mc_output)

    if s.save_plots:
        tc.fit.save_pdf_plots(
            user_checked_filename,
            fit_results,
            auto_aic_choices,
            user_check,
            final_choices,
            onlyChecked=True,
            title="",
        )
        tc.fit.save_pdf_plots(
            all_fits_filename,
            fit_results,
            auto_aic_choices,
            user_check,
            final_choices,
            onlyChecked=False,
            title="",
        )


def choose_best_fits(models, fit_results_T, s, fit_results):
    # %%
    if len(models) == 1:
        auto_aic_choices = None
        user_check = None
        final_choices = None
    else:
        auto_aic_choices = tc.fit.choose_model_aic(fit_results_T, d_aic=s.d_aic)

        # oh man I wasn't consistent here using transposed/untransposed. Sorry.
        user_check = tc.fit.marginal_fits(fit_results, auto_aic_choices)
        for pixel in s.always_manually_choose:
            user_check[pixel] = True

        interactive_choices = s.interactively_choose_fits
        if interactive_choices is True:
            message = (
                "==============================================\n"
                "====== Manual Checking procedure =============\n"
                "==============================================\n"
                "The software has determined there are {} fits to check. "
                "If cancelled, the automatic choice will be "
                "selected. At any point in the checking process you may cancel and "
                "use the automatic choice by entering x instead of the choice number.\n"
                "Would you like to continue manual checking?  [y]/n   "
            ).format(user_check.sum())
            verify_continue = input(message)
            if verify_continue.lower() == "n":
                final_choices = auto_aic_choices
            else:
                final_choices = tc.fit.interactive_user_choice(
                    fit_results,
                    auto_aic_choices,
                    user_check,
                    baseline_fits=s.baseline_results[s._i],
                )
        else:
            final_choices = auto_aic_choices

        # get an array of the chosen models.
    # Python starts counting at 0 (0-based indexing.). choices starts counting at 1.
    # subtract 1 from choices to convert between these 2 indexing regimes.
    if final_choices is None:
        chosen_models = fit_results[0]
    else:
        chosen_models = np.choose(final_choices - 1, fit_results, mode="clip")

    return chosen_models, final_choices, auto_aic_choices, user_check
