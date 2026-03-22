Changelog
=========

0.2.0 (22/03/2026)
-------------------

### Packaging / requirements:
* Minimum Python version raised from 3.6 to 3.10. Python 3.10, 3.11, 3.12, and 3.13 are now declared as supported classifiers.
* NumPy upper-cap `<2` removed. NumPy 2.x is now supported; minimum version raised from 1.17.0 to 1.24.
* Project URLs corrected to current repository location.

### Bug fixes:
* `set_param_hints_endswith` in `lmfit_ext` unconditionally overwrote existing parameter `min`/`max` hints, silently loosening user-set tighter bounds. Now merges conservatively: takes `max(existing_min, new_min)` and `min(existing_max, new_max)`.
* `ResultDict.loadtxt` crashed with `IndexError` on single-spaxel (1×1 spatial grid) files because `numpy.loadtxt` returns a 1D array for single-row files. Promoted to 2D immediately after loading.
* `ResultDict.loadtxt` crashed with `TypeError: need sequence of keys with len > 0` when loading files saved with `generate_pixel_coordinates=False`, because `numpy.lexsort(())` was called unconditionally before the empty-indices guard. The sort is now skipped when no dimension columns are present.
* `process_single_spectrum` in `fit_line` did not skip spaxels whose SNR is `NaN`. The guard used `np.isnan(...) is True` (identity comparison against the Python `True` singleton), which always evaluates to `False` for `numpy.bool_` values. Replaced with a plain truthiness check.
* `process_single_spectrum` in `fit_line` raised `AttributeError` when the chop-bandwidth retry path received `None` from `lmfit` (fully-masked chopped spectrum). The `None` guard present on the first fit call was missing on the retry. Guard added.
* `create_outflow_mask` in `analyze_outflow_extent` raised `UnboundLocalError` instead of a descriptive error when `which_contour` was not found in `contour_levels`. Replaced the `for/break` search with `list.index()` wrapped in a `try/except` that raises a clear `ValueError`.
* `get_region(0)` triggered a division-by-zero: both `rx2` and `ry2` were `0`, making the inside-ellipse check produce `nan`, which evaluates to `False`, so all pixels were excluded and an empty array was returned instead of `[[0, 0]]`. Added a special-case guard before the ellipse computation.
* `RecursiveArray.__init__` and `aslist()` accessed `self.data[0]` unconditionally, raising `IndexError` on empty input. Added early-return guards for the empty-list case in both methods.
* `get_param_values` branch 3 (extracting a `ModelResult` attribute such as `redchi` or `chisqr`) was unreachable because it relied on `ModelResult.get()`, which does not exist in `lmfit`. Replaced `params.get(param_name, default_value)` with `getattr(params, param_name, default_value)`.


0.1.18 (22/08/2025)
-------------------

### Bug fix:
* reapply_certain_model_hints was improperly called.


0.1.17 (02/02/2025)
-------------------

### Enhancement:
* Moved the package information to pyproject.toml and updated the dependencies. Installation of dependencies should be automatic now.

### Update:
* Because we are using an old version of lmfit, I have constrained Numpy version to be <2 (v2 broke some functionality and I didn't fix it yet.)

### Bug fix:
* All threadcount guess functions now keep certain model param_hints, to aid in how we use the params in the fitting scripts.  Previously, the guess function would overwrite any vary=False value with the guess, and remove any expr constraint.  Now, if a model param hint has set vary = False and a value is present, the guessed params preserve that value.  Also, if model param hint includes expr, then that expr is preserved.


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