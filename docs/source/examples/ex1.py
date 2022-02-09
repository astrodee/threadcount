# examples/ex1.py

from threadcount.procedures import set_rcParams, open_cube_and_deredshift

user_settings = {
    "data_filename": "MRK1486_red_metacube.fits",
    "data_hdu_index": None,
    "var_filename": None,
    "var_hdu_index": None,
    "continuum_filename": None,  # Empty string or None if not supplied.
    "z": 0.0339,
    "setup_parameters": True,
    "comment": "",
}

user_settings = open_cube_and_deredshift.run(user_settings)
