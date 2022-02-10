import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import threadcount as tc
from threadcount.procedures import set_rcParams


def run(user_settings):
    # these are the inputs that will be used if there are no command line arguments.
    # The command line argument will take the form of a json string and override any of these options.
    default_settings = {
        "one_gauss_input_file": "ex_5007_simple_model.txt",
        "velocity_mask_limit": 60,
        # manual_galaxy_region format
        # [min_row, max_row, min_column, max_column] --> array[min_row:max_row+1,min_column:max_column+1]
        # 'None' means it will continue to image edge.
        "manual_galaxy_region": [30, 39, None, None],
        "verbose": True,
        "vertical_average_region": 1,
        "contour_levels": [0.5, 0.9],
        "outflow_contour_level": 0.5,  # Select from contour_levels list which contour to choose to define outflow region.
        "output_base_name": "",
        "arcsec_per_pixel": 0.291456,
        "galaxy_center_pixel": [35, 62],  # row,col
        "velocity_vmax": 140,
        "units": None,
    }
    s = tc.fit.process_settings_dict(default_settings, user_settings)  # s for settings.

    # load file
    input_data = tc.fit.ResultDict.loadtxt(s.one_gauss_input_file)

    # parse file header if necessary
    comment_lines = input_data.comment.split("\n")
    s.arcsec_per_pixel = process_arcsecs(s.arcsec_per_pixel, comment_lines)
    s.units = process_units(s.units, comment_lines)

    sigma = input_data["g1_sigma"]
    center = input_data["g1_center"]
    flux = input_data["g1_flux"]
    snr = input_data["snr"]
    snr_cutoff = np.nanmedian(snr)

    # converts sigma to velocity
    velocity = ((sigma + center) * u.AA).to(
        u.km / u.s, equivalencies=u.doppler_optical(center * u.AA)
    )

    flux_avg = boxcar_average_1d(flux, s.vertical_average_region)

    # compute the maximum row --- here I compute the image I will use to calculate
    # the row containing the galaxy center.
    m_flux_avg = np.ma.masked_where(np.isnan(flux_avg) | (snr < snr_cutoff), flux_avg)
    # I decided to use a snr cutoff, of the median snr of the whole image, for
    # valid pixels to compute the center row.
    gal_center_row = compute_gal_center_row(m_flux_avg)

    # calculate the "extent" with arcsec per pixel, for viewing in matplotlib imshow
    default_extent = (-0.5, flux.shape[1] - 0.5, -0.5, flux.shape[0] - 0.5)
    extent = (
        s.arcsec_per_pixel
        * np.array(
            [
                default_extent[0] - s.galaxy_center_pixel[1],
                default_extent[1] - s.galaxy_center_pixel[1],
                default_extent[2] - gal_center_row,
                default_extent[3] - gal_center_row,
            ]
        )
    ).tolist()
    if s.verbose:
        im_to_plot = velocity.to_value(u.km / u.s)
        vmax = s.velocity_vmax
        # blurred = gaussian_filter(temp_masked.filled(50), sigma=2)
        lim = [55, 60, 65, 70, 75]
        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        axis = iter(axes.flatten())
        plt.sca(next(axis))
        plt_image_extent(im_to_plot, extent, vmax=vmax, horizontal0=False)
        cbar = plt.colorbar()
        cbar.set_label("km/s")
        plt.contour(im_to_plot, levels=lim, extent=extent, cmap="plasma_r")
        cbar = plt.colorbar()
        cbar.set_label("km/s")
        # temp2 = plt.contour(blurred,levels=[lim],colors='r')
        plt.sca(next(axis))
        plt_image_extent(im_to_plot, extent, horizontal0=False, vmax=vmax)
        cbar = plt.colorbar()
        cbar.set_label("km/s")

        plt.sca(next(axis))
        plt.gca().set_aspect("equal")
        plt.contourf(im_to_plot, levels=lim, cmap="plasma_r", extent=extent)
        plt.xlabel("arcsec")
        plt.ylabel("arcsec")
        cbar = plt.colorbar()
        cbar.set_label("km/s")
        plt.sca(next(axis))
        plt_image_extent(
            np.ma.masked_where(
                (
                    (im_to_plot < s.velocity_mask_limit)
                    # |(rel_err > 0.08)
                ),
                im_to_plot,
            ),
            extent,
            vmax=vmax,
            horizontal0=False,
        )
        plt.gca().set_title("mask < {} km/s".format(s.velocity_mask_limit))
        cbar = plt.colorbar()
        cbar.set_label("km/s")
        plt.tight_layout()

    # arrays starting with the word mask are boolean masks.
    maskv = ~(velocity.to_value(u.km / u.s) > s.velocity_mask_limit)
    mask_manual = np.full_like(maskv, False)
    g = s.manual_galaxy_region
    if g[1] is not None:
        g[1] += 1
    if g[3] is not None:
        g[3] += 1
    mask_manual[slice(g[0], g[1]), slice(g[2], g[3])] = True

    # arrays starting with m are masked arrays
    # mvelocity = np.ma.array(velocity.value, mask=maskv)
    # mflux = np.ma.array(flux, mask=maskv)
    mask_flux_avg = maskv | mask_manual | np.isnan(flux_avg)
    mflux_avg = np.ma.array(flux_avg, mask=mask_flux_avg)

    flux_image = mflux_avg  # easily switch which flux map to use for calculating contours and plotting.

    contour_output = calculate_contours(
        flux_image, clip_max=2, levels=s.contour_levels, center_row=gal_center_row
    )
    contour_output_arcsec = contours_to_arcsec(
        contour_output, gal_center_row, s.galaxy_center_pixel[1], s.arcsec_per_pixel
    )

    outflow_mask = create_outflow_mask(
        contour_output,
        s.contour_levels,
        which_contour=s.outflow_contour_level,
        output_shape=maskv.shape,
    )

    save_contours(
        s.output_base_name + "_outflow_width_prof.txt",
        contour_output,
        gal_center_row,
        s.contour_levels,
        s.arcsec_per_pixel,
        s.galaxy_center_pixel[1],
    )
    upper_outflow = np.full_like(outflow_mask, False)
    upper_outflow[:gal_center_row] = True
    upper_outflow = upper_outflow | outflow_mask

    upper = np.full_like(outflow_mask, False)
    upper[:gal_center_row] = True
    upper = upper | outflow_mask

    # fill upper down to gal_center_row:
    # find lowest row and repeat that row till center.
    bottom_row = np.argwhere(np.sum(~upper, axis=1) > 0)[0][0]
    upper[gal_center_row:bottom_row] = upper[bottom_row]

    lower_outflow = np.full_like(outflow_mask, False)
    lower_outflow[gal_center_row:] = True
    lower_outflow = lower_outflow | outflow_mask

    lower = np.full_like(outflow_mask, False)
    lower[gal_center_row:] = True
    lower = lower | outflow_mask

    # fill upper down to gal_center_row:
    # find lowest row and repeat that row till center.
    upper_row = np.argwhere(np.sum(~lower, axis=1) > 0)[-1][-1]
    lower[upper_row : gal_center_row + 1] = lower[upper_row]

    if s.verbose:
        fig, axes = plt.subplots(1, 4, figsize=(12, 4))
        iterax = iter(axes)
        ax = next(iterax)
        plt.sca(ax)
        plt_image_extent(upper_outflow, extent, title="upper_outflow")
        ax = next(iterax)
        plt.sca(ax)
        plt_image_extent(upper, extent, title="upper")
        ax = next(iterax)
        plt.sca(ax)
        plt_image_extent(lower_outflow, extent, title="lower_outflow")
        ax = next(iterax)
        plt.sca(ax)
        plt_image_extent(lower, extent, title="lower")
        plt.tight_layout()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.sca(axes[0])
    plt.imshow(velocity.value, vmax=s.velocity_vmax, extent=extent)
    it_colors = iter(["r--", "r"])
    plt.plot(
        contour_output_arcsec[1],
        contour_output_arcsec[0],
        "black",
        label="median peak flux",
    )
    for this_plot in range(len(s.contour_levels)):
        color = next(it_colors)
        plt.plot(
            contour_output_arcsec[1] - contour_output_arcsec[this_plot + 2],
            contour_output_arcsec[0],
            color,
        )
        plt.plot(
            contour_output_arcsec[1] + contour_output_arcsec[this_plot + 2],
            contour_output_arcsec[0],
            color,
            label="{:g}% width".format(100 * s.contour_levels[this_plot]),
        )
    # plt.plot(max_row_in_column, "white", label="max flux in column")
    plt.axhline(0, label="galaxy midplane")
    # plt.axhline(gal_center_row, label="galaxy midplane")
    plt.gca().set_title("Velocity Dispersion")
    plt.xlabel("arcsec")
    plt.ylabel("arcsec")
    plt.legend()
    cbar = plt.colorbar(orientation="horizontal")
    cbar.set_label(r"velocity dispersion [km s$^{-1}$]")
    ##################
    plt.sca(axes[1])

    plt.imshow(np.log10(flux_image), extent=extent)
    it_colors = iter(["r--", "r"])
    plt.plot(
        contour_output_arcsec[1],
        contour_output_arcsec[0],
        "black",
        label="median peak flux",
    )
    for this_plot in range(len(s.contour_levels)):
        color = next(it_colors)
        plt.plot(
            contour_output_arcsec[1] - contour_output_arcsec[this_plot + 2],
            contour_output_arcsec[0],
            color,
        )
        plt.plot(
            contour_output_arcsec[1] + contour_output_arcsec[this_plot + 2],
            contour_output_arcsec[0],
            color,
            label="{:g}% width".format(100 * s.contour_levels[this_plot]),
        )
    # plt.plot(max_row_in_column, "white", label="max flux in column")
    plt.axhline(0, label="galaxy midplane")
    # plt.axhline(gal_center_row, label="galaxy midplane")
    plt.gca().set_title("Line Flux")
    plt.xlabel("arcsec")
    plt.ylabel("arcsec")
    plt.legend()
    cbar = plt.colorbar(orientation="horizontal")
    cbar.set_label(r"$log_{10}$(flux) [arb.]")

    plt.savefig(s.output_base_name + "_outflow_width_prof.pdf")
    # we have already calculated contour output and gal_center_row
    upper_outflow_center = contour_output[:, contour_output[0] > gal_center_row][
        1
    ].mean()
    lower_outflow_center = contour_output[:, contour_output[0] < gal_center_row][
        1
    ].mean()

    # upper_origin = [int(gal_center_row), int(upper_outflow_center)]
    # lower_origin = [int(gal_center_row), int(lower_outflow_center)]
    upper_origin = s.arcsec_per_pixel * np.array([gal_center_row, upper_outflow_center])
    lower_origin = s.arcsec_per_pixel * np.array([gal_center_row, lower_outflow_center])

    this_data = flux_avg
    grid = s.arcsec_per_pixel * np.indices(this_data.shape)

    if s.verbose:
        which_side = "upper"
        if which_side == "upper":
            this_mask = upper_outflow
            this_origin = upper_origin
        else:
            this_mask = lower_outflow
            this_origin = lower_origin
        x_im = distance(
            np.ma.masked_where(this_mask, grid[0]),
            np.ma.masked_where(this_mask, grid[1]),
            this_origin,
        )
        data_im = np.ma.masked_where(this_mask, this_data)

        x, data = sort_data(x_im, data_im)
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        iteraxes = iter(axes.flatten())
        ax = next(iteraxes)
        plt.sca(ax)
        plt.plot(x_im, np.log10(data_im), "x-")
        plt.xlabel("distance to {} origin [arcsec]".format(which_side))
        plt.ylabel(r"$log_{10}$(flux) [arb.]")
        plt.title("{} outflow".format(which_side))
        # plt.gca().set_xlim(left=0)
        # plt.gca().set_ylim(top=1.5)
        ax = next(iteraxes)
        plt.sca(ax)
        plt.plot(
            np.ma.masked_where(
                this_mask, np.abs(grid[0] - s.arcsec_per_pixel * gal_center_row)
            ),
            np.log10(data_im),
            "x-",
        )
        plt.xlabel("distance to galaxy midplane [arcsec]")
        plt.ylabel(r"$log_{10}$(flux) [arb.]")
        plt.title("{} outflow".format(which_side))

        which_side = "lower"
        if which_side == "upper":
            this_mask = upper_outflow
            this_origin = upper_origin
        else:
            this_mask = lower_outflow
            this_origin = lower_origin
        x_im = distance(
            np.ma.masked_where(this_mask, grid[0]),
            np.ma.masked_where(this_mask, grid[1]),
            this_origin,
        )
        data_im = np.ma.masked_where(this_mask, this_data)

        x, data = sort_data(x_im, data_im)

        ax = next(iteraxes)
        plt.sca(ax)
        plt.plot(x_im, np.log10(data_im), "x-")
        plt.xlabel("distance to {} origin [arcsec]".format(which_side))
        plt.ylabel(r"$log_{10}$(flux) [arb.]")
        plt.title("{} outflow".format(which_side))
        # plt.gca().set_xlim(left=0)
        # plt.gca().set_ylim(top=1.5)
        ax = next(iteraxes)
        plt.sca(ax)
        plt.plot(
            np.ma.masked_where(
                this_mask, np.abs(grid[0] - s.arcsec_per_pixel * gal_center_row)
            ),
            np.log10(data_im),
            "x-",
        )
        plt.xlabel("distance to galaxy midplane [arcsec]")
        plt.ylabel(r"$log_{10}$(flux) [arb.]")
        plt.title("{} outflow".format(which_side))
        plt.tight_layout()
    string_to_output = (
        "Log10 Double exponential model:\n"
        "f(x) = log10(\n"
        "             e1_amplitude * exp( -x / e1_decay ) +\n"
        "             e2_amplitude * exp( -x / e2_decay )\n"
        "       )\n\n"
    )

    set_rcParams.set_params({"axes.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True, sharex=True)
    iterax = iter(axes)
    mins = [np.nan, np.nan]
    maxs = [np.nan, np.nan]

    for which_side in ["upper", "lower"]:
        if which_side == "upper":
            this_mask = upper_outflow
            this_origin = upper_origin
        else:
            this_mask = lower_outflow
            this_origin = lower_origin
        x_im = distance(
            np.ma.masked_where(this_mask, grid[0]),
            np.ma.masked_where(this_mask, grid[1]),
            this_origin,
        )
        data_im = np.ma.masked_where(this_mask, this_data)

        x, y = sort_data(x_im, data_im)

        model = tc.models.Log10_DoubleExponentialModel()
        params = model.guess(np.log10(y), x)

        fitresult = model.fit(np.log10(y), x=x, params=params, method="least_sq")
        summary_string = (
            "\n".join(radius_at_fraction(x, y, [50, 90], return_string=True)) + "\n\n"
        )
        description_string = "##### {} outflow #####\n\n".format(which_side)
        string_to_output += (
            description_string + summary_string + fitresult.fit_report() + "\n\n"
        )

        # fitresult.plot(ylabel=r"$log_{10}$(flux)")
        # set_title()
        # comps = list(fitresult.eval_components(x=x).items())
        # for key,data in comps:
        #     plt.plot(x,np.log10(data),label="decay: {:.3g}".format(fitresult.params[key+"decay"].value))
        # plt.gca().set_ylim(bottom=-2.5);
        # plt.legend()
        plt.sca(next(iterax))
        a = plt.hist2d(x, np.log10(y), bins=40, cmap="Blues", cmin=1)
        mins = min([b.min() for b in a[1:3]], mins)
        maxs = max([b.max() for b in a[1:3]], maxs)
        comps = list(fitresult.eval_components(x=x).items())
        for key, data in comps:
            plt.plot(
                x,
                np.log10(data),
                "red",
                label="decay: {:.3g}".format(fitresult.params[key + "decay"].value),
            )
        plt.plot(x, fitresult.best_fit, "red", label="bestfit")
        # plt.gca().set_ylim(bottom=-2.2);
        # plt.legend()
        plt.title("{}".format(which_side))
        plt.ylabel(r"$log_{10}$(flux) [arb.]")
        plt.xlabel("distance to {} origin [arcsec]".format(which_side))
        textstr = "Inner scale length: {:.3g}\nOuter scale length: {:.3g}".format(
            *sorted(
                [p.value for k, p in fitresult.params.items() if k.endswith("decay")]
            )
        )
        plt.gca().text(
            0.97,
            0.97,
            textstr,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="right",
        )
    # plt.autoscale()
    dx = maxs[0] - mins[0]
    dy = maxs[1] - mins[1]
    margin = 0.05
    plt.gca().set_xlim(mins[0] - margin * dx, maxs[0] + margin * dx)
    plt.gca().set_ylim(mins[1] - margin * dy, maxs[1] + margin * dy)
    plt.savefig(s.output_base_name + "_sb_prof_fitresults.pdf")
    with open(s.output_base_name + "_sb_prof_fitresults.txt", "w") as f:
        f.write(string_to_output)

    plt.show()


def extract_wcs(comment_lines):
    search_string = "wcs_step:"
    wcs_line = [x for x in comment_lines if x.startswith(search_string)][0]
    return eval(wcs_line[len(search_string) :].strip().replace(" ", ","))


def process_arcsecs(input_arcsecs, comment_lines):
    if (input_arcsecs is None) or (
        isinstance(input_arcsecs, str) and input_arcsecs in ("header", "auto")
    ):
        try:
            output_arcsecs = extract_wcs(comment_lines)
        except IndexError:
            raise ValueError("auto determination of arcsec_per_pixel failed.")
    else:
        output_arcsecs = input_arcsecs
    # since we right now are only equipped to handle square pixels, I will take
    # the first element of any arcsec_per_pixel list.

    if not isinstance(output_arcsecs, (float, int)):
        output_arcsecs = output_arcsecs[0]

    return output_arcsecs


def process_units(input_units, comment_lines):
    if input_units in (None, "header", "auto"):
        search_string = "units:"
        units_line = [x for x in comment_lines if x.startswith(search_string)][0]
        output_units = units_line[len(search_string) :].strip()
    else:
        output_units = input_units

    # convert string to astropy units.
    if isinstance(output_units, str):
        output_units = u.Unit(output_units)

    return output_units


def boxcar_average_1d(input_array, width, axis=0):
    return np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(width), "same") / width,
        axis=axis,
        arr=input_array,
    )


def plt_image_extent(data, extent, title="", horizontal0=True, **kwargs):
    plt.imshow(data, extent=extent, **kwargs)
    plt.gca().set_title(title)
    if horizontal0:
        plt.gca().axhline(0, label="galaxy midplane")
    plt.xlabel("arcsec")
    plt.ylabel("arcsec")


def contours_to_arcsec(
    contour_output, galaxy_center_row=0, galaxy_center_col=0, arcsec_to_pixel=1
):
    output = contour_output.copy()
    output[0] -= galaxy_center_row
    output[1] -= galaxy_center_col
    return output * arcsec_to_pixel


def save_contours(
    filename,
    contour_output,
    gal_center_row,
    contour_levels,
    arcsec_per_pixel=None,
    gal_center_col=None,
):
    temp = contour_output.copy()  # ensures there's no modifying of contour_output
    # make row# = minor axis distance: i.e. subtract gal_center_row
    temp[0] -= gal_center_row
    if gal_center_col is not None:
        temp[1] -= gal_center_col
    temp[0].mask = temp[1].mask
    # switch row col --> col,row = x,y for human reading
    temp[[0, 1]] = temp[[1, 0]]

    if arcsec_per_pixel is not None:
        fmt = "%.5g"
        temp = arcsec_per_pixel * temp
        header = ["dx_arcsec", "dy_arcsec"]
    else:
        fmt = "%.i"
        header = ["x_px", "y_px"]

    delimiter = " "
    np.savetxt(
        filename,
        np.ma.compress_cols(temp).T,
        fmt=fmt,
        delimiter=delimiter,
        header=delimiter.join(
            header + ["{:.2g}pc_halfwidth".format(lev * 100) for lev in contour_levels]
        ),
    )


def distance(row, col, origin):
    # origin: order is row,col
    return np.sqrt((row - origin[0]) ** 2 + (col - origin[1]) ** 2)


def sort_data(x, data):
    # ## ASSUMES SAME MASK
    x = np.ma.compressed(x)
    data = np.ma.compressed(data)
    sort_increasing = np.argsort(x)
    x = x[sort_increasing]
    data = data[sort_increasing]
    return x, data


def calculate_contours(flux_masked_array, levels=None, clip_max=3, center_row=35):
    if levels is None:
        levels = [0.5, 0.9]
    levels = np.sort(levels)
    max_rows = row_max(flux_masked_array, clip=clip_max, center_row=center_row)

    contour_output = []
    for row, max_col in zip(*max_rows):
        if np.ma.is_masked(max_col):
            output_row = [row, max_col]
            output_row += [max_col] * len(levels)
            contour_output += [output_row]
            continue
        this_row = flux_masked_array[row]
        total = this_row.sum()

        loop_total = this_row[max_col]
        count = 0
        output_row = [row, max_col]
        for goal in levels * total:
            while loop_total < goal:
                count += 1
                loop_total += this_row[max_col - count] + this_row[max_col + count]

            output_row += [count]
        contour_output += [output_row]
        # start at max_col and step out 1 either side until reach % flux contained. Add the coordinates to output.
        # output will look like:
        # [row,left_col,right_col]
        # for each % level.
    contour_output = np.array(contour_output).T
    return np.ma.masked_where(np.isnan(contour_output), contour_output.astype(int))


def create_outflow_mask(contour_output, contour_levels, which_contour, output_shape):
    for i, val in enumerate(contour_levels):
        if val == which_contour:
            idx = i
            break
    outflow_mask = np.full(output_shape, True)
    for line in zip(
        *contour_output
    ):  # row, center_column, half_width@contour_level[0], half_width@contour_level[1]....
        row = line[0]
        center_col = line[1]
        if np.ma.is_masked(center_col):
            continue
        col_span = line[2 + idx]  # the half-width at desired contour
        outflow_mask[row, center_col - col_span : center_col + col_span + 1] = False
    return outflow_mask


def row_max(flux_masked_array, clip=3, center_row=35):
    """
    """
    rowmax = np.argmax(flux_masked_array, axis=1)
    # we know that entries will be 0 for masked values.
    # eliminate those and calc. mean and std so we can eliminate outliers.
    mean, std = rowmax[rowmax > 0].mean(), rowmax[rowmax > 0].std()
    # determine rows where the entry is mean += clip*std
    valid = (rowmax < mean + clip * std) & (rowmax > mean - clip * std)
    mask = ~valid

    # rowmax[~valid] = 0
    rowmax = np.ma.masked_where(mask, rowmax)
    rowmax.data[0:center_row] = np.ma.median(rowmax[0:center_row])
    rowmax.data[center_row:] = np.ma.median(rowmax[center_row:])

    return np.arange(0, len(rowmax)), rowmax
    # valid = np.where((rowmax < mean +clip*std)&(rowmax > mean -clip*std))
    # return valid[0], rowmax[valid] # row#, col of max flux for that row.


def compute_gal_center_row(image):
    # for each column, computes the row index of the maximum value
    # Ideally, the answer would then be the median of these values. However,
    # we must ensure that the flux is above a certain amount, to eliminate edge
    # contributions.

    # So, we compute the value of the max flux for each column, and apply a filter
    # for whether that column gets included in the median. If a column's max flux
    # is > 1% of the image maximum, then it's row index is included in the median.
    max_row_in_column = np.argmax(image, axis=0)

    max_flux_in_column = np.array(
        [image[x] for x in zip(max_row_in_column, range(len(max_row_in_column)))]
    )
    gal_center_row = int(
        np.median(
            max_row_in_column[max_flux_in_column > 0.01 * np.nanmax(max_flux_in_column)]
        )
    )
    return gal_center_row


def radius_at_fraction(x, y, values, return_string=False):
    sort_increasing = np.argsort(x)
    x = x[sort_increasing]
    y = y[sort_increasing]
    try:
        values = np.array(sorted(values))
    except TypeError:
        values = np.array([values])
    if values[0] > 1:
        values = values / 100.0
    goals = np.array(values) * y.sum()
    results = []
    it = iter(goals)
    this_goal = next(it)
    total = 0
    for idx, yval in enumerate(y):
        total += yval
        if total > this_goal:
            results += [idx]
            try:
                this_goal = next(it)
            except StopIteration:
                break
    return_array = np.column_stack([values, x[results]])
    if return_string is False:
        return return_array
    else:
        return ["r_{:.3g} = {:.8g}".format(e[0] * 100, e[1]) for e in return_array]


if __name__ == "__main__":
    run(None)
