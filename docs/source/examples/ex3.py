# examples/ex3.py

import threadcount as tc
from threadcount.procedures import (  # noqa: F401
    set_rcParams,
    analyze_outflow_extent,
)

set_rcParams.set_params()

# Create the settings dict for measuring the outflow extent.
analyze_settings = {
    "one_gauss_input_file": "ex3_5007_simple_model.txt",
    "line": tc.lines.L_OIII5007,
    # https://mpdaf.readthedocs.io/en/latest/api/mpdaf.obj.Image.html#mpdaf.obj.Image.mask_region
    # each entry in the list is a dictionary, where they keys are the
    # parameters for the function mask_region() in mpdaf, and the values
    # are the corresponding values. For now, both "unit_center" and "unit_radius"
    # MUST be included and MUST have the value None. (i.e. it only works in
    # pixels).
    "mask_region_arguments": [],
    "maximum_sigma_A": 50,  # maximum allowed sigma in Angstroms.
    "velocity_mask_limit": 60,
    # manual_galaxy_region format
    # [min_row, max_row, min_column, max_column] --> array[min_row:max_row+1,min_column:max_column+1]
    # 'None' means it will continue to image edge.
    "manual_galaxy_region": [30, 39, None, None],
    "verbose": False,
    "vertical_average_region": 1,
    "contour_levels": [0.5, 0.9],
    # Select from contour_levels list which contour to choose to define outflow region.
    "outflow_contour_level": 0.5,
    "output_base_name": "ex3output",
    "galaxy_center_pixel": [35, 62],  # row,col
    "velocity_vmax": 140,  # sets the image display maximum value.
    "arcsec_per_pixel": "header",
    "units": "header",
}


analyze_outflow_extent.run(analyze_settings)

print("Finished with script.")
set_rcParams.reset_params()
