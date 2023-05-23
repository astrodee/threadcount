Changelog
=========

0.1.0 (23/05/2023)
------------------

### Enhancement:
* Add new guess function for 3 gaussians. Make use of it in your runner file by:
  ```python
  model = tc.models.Const_3GaussModel()
  model.guess = tc.models._guess_multiline3  # uses the default parameters 

  # to specify parameters, e.g. here specifying centers and absolute_centers: 
  model.guess = lambda data, x : tc.models._guess_multiline3(self = model, data = data, x = x,
      centers = (-14, 0, 21),
      absolute_centers = True
  )
  ```
* Add new guess function for 2 gaussians. Make use of it in your runner file by:
  ```python
  model = tc.models.Const_2GaussModel()
  model.guess = tc.models._guess_multiline2  # uses the default parameters 

  # to specify parameters, e.g. here specifying centers and absolute_centers: 
  model.guess = lambda data, x : tc.models._guess_multiline2(self = model, data = data, x = x,
      centers = (-1, 0),
      absolute_centers = True
  )
  ```  
* Add troubleshooting model_results to settings dictionary
* Exception handling in src/threadcount/mpdaf_ext.py, do not stop the looping but returns None.


0.0.6 (11/05/2023)
------------------

### Enhancement:
* Add explore_results procedure and example


0.0.5 (20/08/2022)
------------------

### Bug fixes:
* src/threadcount/lmfit_ext.py
  * update order_gauss to fix bug


0.0.4 (05/06/2022)
------------------

### Bug fixes:

* src/threadcount/procedures/fit_line.py
  * fixed bug in fit iterator
* src/threadcount/fit.py
  * fixed bug in calculating new_y


0.0.3 (05/06/2022)
------------------

### Enhancements:

* Add baseline fitting, removing, and plotting

### Bug fixes:

* src/threadcount/mpdaf_ext.py
  * fixed bug in fit error catching


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