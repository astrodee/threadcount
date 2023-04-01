Changelog
=========

0.0.6 (26/03/2023)
------------------

* include package *numdifftools* in setup.cfg
* change setting *region_averaging_radius* to understand that an input of 0 radius
  means no region averaging. Inside the code, 0 radius is reset a radius of 0.5 (1 pixel diameter region).


0.0.2 (23/02/2022)
------------------

---
**NOTE**

**Settings have changed** in procedures: open_cube_and_deredshift.py and analyze_outflow_extent.py.

open_cube_and_deredshift:
* changed setup_parameters --> tweak_redshift
* added tweak_redshift_line

analyze_outflow_extent, Added:
* line
* maximum_sigma_A
* mask_region_arguments

---

### Enhancements:

* Masking added to procedures/analyze_outflow_extent.py
* Catch errors in fitting so the entire code doesn't crash during line fitting.
* remove pdf page containing list of un-fit pixels when plots are saved in procedures/fit_lines.py.
* Added option to choose which Line is used to set redshift in procedures/open_cube_and_deredshift.py
* Updated settings name in procedures/open_cube_and_deredshift.py to avoid clash.

### Bug fixes:

* procedures/analyze_outflow_extent.py
  * Processing setting for arcsec_per_pixel
* procedures/fit_line.py
  * Fixed case for no continuum cube provided.
* fit.py
  * Fixed tweak_redshift plot not updating. (affected procedures/open_cube_and_deredshift.py)
