# Threadcount Improvement Roadmap

## Guiding Principles

- **Backwards compatibility first** — every phase preserves the existing user-facing API until a deliberate, versioned breaking change.
- **Tests before dependency changes** — the lmfit fork migration and pyproject.toml modernisation both carry real breakage risk. Build the test safety net first, then make those changes with confidence.
- **Small, reviewable chunks** — each item is a single PR/commit that can be merged and validated independently.

---

## Phase 0a — Safe Tooling (no risk, do first)

*These tasks change no runtime behaviour and can be done immediately.*

### 0a.1 — Add `ruff` and `pre-commit` ✅
- Add a `[tool.ruff]` section in `pyproject.toml` for linting and formatting. Start permissive (disable rules you can't fix yet) and tighten gradually.
- Add a `.pre-commit-config.yaml` running `ruff --fix` and `ruff format` on every commit.
- Add `dev` extras to `[project.optional-dependencies]` (`pytest`, `pytest-cov`, `mypy`, `ruff`, `pre-commit`) so contributors can do `pip install -e ".[dev]"`.
- This stops the codebase from drifting further while you improve it.

### 0a.2 — Add a minimal CI pipeline ✅
- A GitHub Actions workflow (`.github/workflows/ci.yml`) that runs `pytest` and `ruff` on every push/PR.
- Start with the current Python/numpy versions; expand after the dependency work below.

---

---

## Phase 1 — Test Infrastructure (must come before Phase 0b)

*The existing test suite is a single 35-line file with one test. Build coverage to ~50% here — this is the safety net that makes the dependency changes in Phase 0b safe to attempt.*

### 1.1 — Synthetic data fixtures ✅
Create `tests/conftest.py` with `pytest` fixtures that build:
- A small synthetic FITS cube (e.g. 10×10 spatial, 200-wavelength) with known gaussian emission lines injected at known parameters.
- A pre-built `SimpleNamespace` settings object with all defaults filled in.

These become reusable inputs for every subsequent test.

### 1.2 — Tests for settings processing ✅
Cover `fit.py` `process_settings` / `process_settings_dict`:
- Defaults are applied when a key is absent.
- User overrides replace defaults.
- Invalid types raise a clear error (currently they silently produce wrong behaviour).

### 1.3 — Tests for `Line` and `lines.py` ✅
Cover `lines.py`:
- Constructing a `Line` with explicit values stores them correctly.
- Pre-defined constants (`L_OIII5007`, etc.) have the expected wavelength values.

### 1.4 — Tests for model functions ✅
Extend `tests/test_model_function.py`:
- One test per model class (fast and standard): pass synthetic flux array, verify the fit recovers the injected gaussian parameters within tolerance.
- Fix the existing `test_numba_accuracy` to use a seeded RNG for reproducibility.

**Optimizer finding**: When fitting closely-spaced components (≤1σ separation) from an auto-guess starting point, `method='leastsq'` (Levenberg–Marquardt, the scipy default) reliably lands in local minima.  A full survey of all lmfit-available optimizers on a 3-Gaussian model with 1σ spacing shows:
- `least_squares` (scipy TRF): redchi ~3×10⁻¹⁸, 0 % height error — **best by far**
- `bfgs`: redchi ~7×10⁻¹³, 0 % error (but reports `success=False`)
- All others (`nelder`, `powell`, `leastsq`, `slsqp`, `tnc`, `cobyla`, `cg`): redchi ≥10⁻⁵, height errors 20–260 %

`least_squares` is already the production default in `fit_lines.py` and is recommended in the docs.  The `test_fit_from_auto_guess` tests for standard composite models (e.g. `Const_3GaussModel`) must use `method='least_squares'` explicitly, since `model.fit()` defaults to `leastsq`.

**Findings from implementing the first five classes** (`GaussianModelH`, `Const_1GaussModel`, `Const_1GaussModel_fast`, `Const_2GaussModel`, `Const_2GaussModel_fast`):
- `test_numba_accuracy` previously used `random.uniform` with no seed → non-reproducible. Fixed to `np.random.default_rng(RNG_SEED)`. `np.array_equal` replaced with `np.allclose` (numba may reorder FP ops).
- `lmfit.models.fwhm_expr` formats the FWHM factor as `:.7f` (`2.3548200`), truncating the exact value `2.3548200450309493`. The resulting systematic inaccuracy in the constrained `fwhm` parameter is ~4.5×10⁻⁸ × σ per Å (see §2.12).
- Standard composite models (`Const_1GaussModel`, etc.) silently ignore their `prefix` constructor argument — they print a warning and continue. `Const_1GaussModel_fast` correctly supports `prefix` because it uses `lmfit.Model` directly. See §2.13.
- `_guess_2gauss` default `centers=(-2, 0)` is designed for the narrow+broad same-centre use-case. It does **not** work for well-separated doublets (e.g. [NII] 6548/6563 at 10 σ separation). Users must pass `absolute_centers=True` with explicit centres or use `_guess_multiline2`. This is currently undocumented. See §2.14.

**Code-review gaps and bugs found after completing all 22 test sections** (see §1.4a below for the full fix plan):

- **Missing tests for `_guess_multiline2` / `_guess_multiline3`** (standard models) and **`_guess_multiline2_d` / `_guess_multiline3_d` / `_guess_multiline4_d` / `_guess_multiline6_d`** (fast models) — these guess helpers are exported in `__all__` and used in production, but have zero direct test coverage.
- **No parity tests for `Const_2GaussModel_fast` and `Const_3GaussModel_fast` vs. their standard counterparts** — only `Const_1GaussModel_fast` has a parity test. Discrepancies in the fast-model deltax parameterisation could go undetected.
- **`test_fit_from_auto_guess` for 4G and 6G classes asserts only `g4_center` and `c`** — individual deltax values and all component heights are unchecked; the redchi threshold is `1e-2` (loose) not `1e-4`. Same looseness in `TestConst4GaussModelConstrainedSIIFast.test_fit_from_auto_guess` and `TestConst6GaussModelConstrainedHaNIIFast.test_fit_from_auto_guess`.
- **`TestConst6GaussModelConstrainedHaNIIFast.test_fit_recovers_parameters` never checks `g1_h_factor`, `g3_h_factor`, `g5_h_factor`** — these are the novel outflow-ratio parameters central to the model's purpose.
- **`TestConst4GaussModelFast.test_fit_recovers_parameters` and `TestConst6GaussModelFast.test_fit_recovers_parameters` do not check individual component heights or all deltax values** — only the reference component (g4) and a subset of offsets are verified.
- **`Quadratic_*GaussModel` tests do not recover non-zero quadratic coefficients from a genuine quadratic baseline** — `test_eval_quadratic_baseline_additive` checks eval correctly, but `test_fit_recovers_parameters` always supplies `a=b=0` as the initial guess and does not inject a spectrum with a real quadratic continuum. The baseline-removal capability is untested in a fit.
- **`fwhm_expr_fast` / `flux_expr_fast` use `:.7f` format** — the fast-model `fwhm` truncation error is documented in `test_fit_fwhm_consistent` via a loose `< 1e-6` tolerance, but there is no regression test that will *enforce* the corrected `< 1e-12` precision once §2.12 is fixed. The standard `flux_expr` in `models.py` has the same `:.7f` bug and also lacks a precision regression test.
- **`GaussianModelH` with a non-empty `prefix` is untested** — unlike the composite model classes that silently drop the prefix (§2.13), `GaussianModelH` inherits from `lmfit.Model` directly and should propagate the prefix correctly. No test verifies this.
- **`set_common_limits` is only tested on `Const_1GaussModel`** — multi-component models (2G, 3G) and the fast variants are not exercised by `TestSetCommonLimits`.

**Bugs already documented but not yet fixed (fix tasks are in Phase 2):**
- `GaussianModelH` `height min=0` clamps absorption lines — `test_guess_negative_profile` documents the broken behaviour but there is no corresponding fix task. Fix is in §2.14.
- `fwhm_expr` / `flux_expr` / `fwhm_expr_fast` / `flux_expr_fast` truncate constants at `:.7f` — the resulting error (~2×10⁻⁷ Å in fwhm, ~10⁻⁸ relative in flux) is far below any SNR-limited measurement precision and is **not worth fixing**. The loose tolerances in `test_fit_fwhm_consistent` and `test_flux_numerical_value` document this correctly and should not be tightened. §2.12 is retained as a cosmetic note only.

---

### 1.4a — Fill the test coverage gaps documented by the §1.4 code review

This item adds the tests identified as missing in §1.4.  Source-code bug fixes belong in Phase 2 (see §2.14, §2.13, etc.); this item contains only new or improved tests.  Tests that depend on a Phase 2 source fix (e.g. absorption-line tests for `GaussianModelH`) are bundled with their Phase 2 item, not here.

**A. Add parity tests for `Const_2GaussModel_fast` and `Const_3GaussModel_fast`** ✅
- Extend the `TestConst2GaussModelFast` and `TestConst3GaussModelFast` classes with `test_parity_with_standard` methods matching the pattern already in `TestConst1GaussModelFast`.

**B. Tighten 4G and 6G `test_fit_from_auto_guess` assertions** ✅
- `TestConst4GaussModelFast.test_fit_from_auto_guess`: add assertions on `deltax1`, `deltax2`, `deltax3` and at least `g4_height`; tighten redchi threshold to `1e-3`.
- `TestConst6GaussModelFast.test_fit_from_auto_guess`: add assertions on `deltax2`, `deltax3`, `deltax5`; tighten redchi threshold to `1e-3`.
- `TestConst4GaussModelConstrainedSIIFast.test_fit_from_auto_guess`: add assertions on `g2_height`, `deltax12`; tighten redchi threshold to `1e-3`.
- `TestConst6GaussModelConstrainedHaNIIFast.test_fit_from_auto_guess`: add assertions on `g2_height`, `g6_height`; tighten redchi threshold to `1e-3`.

**C. Check `g{1,3,5}_h_factor` in `Const_6GaussModel_constrained_HaNII_fast` fit-recovery test** ✅
- Extend `TestConst6GaussModelConstrainedHaNIIFast.test_fit_recovers_parameters` to assert `g1_h_factor`, `g3_h_factor`, `g5_h_factor` within 5 %; also assert all six `g{n}_sigma` values.
- Use `SIG_OUT = 1.8` (outflow, g1/g3/g5) vs `SIG_SYS = 1.2` (systemic, g2/g4/g6) in the fixture to break the outflow/systemic degeneracy.  Set `G5_H_FACTOR = 0.35` (distinct from `G1_H_FACTOR = 0.30`) so all three ratios are individually identifiable.
- Assertions use the named parameters directly (`g1_h_factor`, `g1_sigma`, etc.) — no sorting required because the model's constraint expressions (`g1_height = g1_h_factor * g2_height`, etc.) tie each h_factor to its specific pair unambiguously.
- Same sigma and h_factor assertions added to `test_fit_from_auto_guess`.

**D. Check all deltax and height values in 4G/6G `test_fit_recovers_parameters`** ✅
- `TestConst4GaussModelFast`: add assertions for `g1_height`, `g2_height`, `g3_height`.
- `TestConst6GaussModelFast`: add assertions for `deltax2`, `deltax3`, `deltax5` and `g1_height` through `g6_height`.

**E. Add a `Quadratic_*` fit test with a genuine non-zero continuum background** ✅
- Added `test_fit_recovers_quadratic_baseline` to `TestQuadratic1GaussModel`, `TestQuadratic2GaussModel`, and `TestQuadratic3GaussModel`.  Injects a combined arch+tilt baseline (`a_arch*(x−x_mid)²  + slope*(x−x_mid)`, arch ~25% of peak height, tilt ~12.5% edge-to-edge), starts the fit with `a=b=0` and `c=C_orig`, and asserts the recovered `a` and `b` are within 10% of truth.

**F. Add direct tests for `_guess_multiline2` and `_guess_multiline3`** ✅
- Add a `TestGuessMultiline2` class: call `model.guess(y, x=x)` after patching the bound method, verify the returned parameters are finite and the center offsets follow the formula.
- Add a `TestGuessMultiline3` class similarly.
- For the fast variants add `TestGuessMultiline2D`, `TestGuessMultiline3D`, `TestGuessMultiline4D`, `TestGuessMultiline6D` in the same style.

**G. Add `GaussianModelH` prefix test** ✅
- Add `test_prefix_propagation` to `TestGaussianModelH`: construct `GaussianModelH(prefix="ha_")` and verify that all parameter names are prefixed and `eval` returns the same values as the un-prefixed model.

**H. Extend `set_common_limits` tests to multi-component models** ✅
- Add `TestSetCommonLimits` fixture variants for `Const_2GaussModel` and `Const_3GaussModel`, verifying that all `g{n}_height`, `g{n}_sigma`, and `g{n}_center` parameters receive appropriate bounds.

---

### 1.5 — Tests for parameter extraction utilities ✅
Cover `fit.py` `get_param_values` and `lmfit_ext.py` `summary_array`:
- Given a known `ModelResult`, the extraction returns the correct values.
- Tests for the "try three different extraction methods" fallback chain — each branch should be individually testable.

### 1.6 — Tests for `ResultDict` ✅
Cover `fit.py` `ResultDict`:
- `savetxt` / `loadtxt` round-trip produces identical data.
- Works with NaN-containing arrays (the normal case for masked cubes).

**Code-review gaps and bugs found after completing the initial test set:**

- **`loadtxt` crashes on a 1×1 spatial grid (single-spaxel file)** — `np.loadtxt` returns a 1D array when the file contains exactly one data row. `loadtxt` then does `data[:, index]` which raises `IndexError: too many indices for array`. Any run on a cube with a single fitted spaxel will fail at the save/reload step. `test_result_dict.py::TestResultDictLoadtxtErrors::test_single_spaxel_round_trip` is marked `xfail(strict=True)` to document this. Fix in §2.19.

- **Missing `loadtxt` ValueError test** — the class docstring listed "loadtxt raises ValueError when row/col indices cannot be matched" as a coverage target, but no test existed. Added `test_mismatched_row_col_raises`: writes four rows that cover only the (0,1) and (1,0) pairs in a nominally 2×2 grid, causing the col comparison inside `loadtxt` to fail.

- **Missing no-coordinate round-trip test, and a second `loadtxt` crash discovered** — no test exercised the `len(indices) == 0` branch (files saved with `generate_pixel_coordinates=False`). Adding those tests exposed a second bug: `np.lexsort(())` raises `TypeError` when `indices` is empty because the guard `if len(indices) == 0` is in the wrong place — the `lexsort` call is unconditional and runs before the guard is checked. Both no-coordinate tests are marked `xfail(strict=True)`. Fix in §2.19.

**Bugs already documented but not yet fixed (fix tasks are in Phase 2):**
- `loadtxt` 1D-array crash on single-spaxel (1×1) files — fix in §2.19.
- `loadtxt` unconditional `np.lexsort(())` crash when no dimension columns (no-coordinate files) — fix in §2.19.

**Additional code-review gaps found during the §1.6 post-completion review (addressed in §1.6a):**

- **`test_mismatched_row_col_raises` tests a reshape failure, not the comparison branch** — the test data has rows `(0,1),(0,1),(1,0),(1,0)`. After lexsort the last col value is `0`, giving `max_col=0` and `new_shape=[3,2,1]`. Reshaping 12 elements into `(3,2,1)=6` elements raises `ValueError` *before* the comparison branch is ever reached. The test comment ("col_readin [[1,1],[0,0]] ≠ col_regenerated [[0,1],[0,1]]") describes a scenario that never occurs with this data. The actual comparison check `result[base_name] != result[readin_name]` is completely untested. To exercise it, the file must have exactly `(max_row+1)*(max_col+1)` rows but with duplicate diagonal pairs, e.g. `(0,0),(0,0),(1,1),(1,1)` — these reshape to `(3,2,2)` successfully but yield `col_readin=[[0,0],[1,1]] ≠ col_regenerated=[[0,1],[0,1]]`.

- **3D spatial arrays (panel dimension) are completely untested** — `DIM_NAMES=("panel","row","col")` explicitly supports 3D spatial arrays, but no test constructs or round-trips a `(n_maps, n_panels, n_rows, n_cols)` shaped array. The `loadtxt` reshape path for 3D grids is also untested.

- **`data_dict` + `data_array` combined construction is untested** — the constructor docstring explicitly describes the merge pattern (data_dict initialises the OrderedDict, then data_array/names update it, with overlapping keys overridden), but no test exercises this combination.

- **`names=None` auto-generation is untested** — when `data_array` is provided without `names`, the constructor silently generates `["data_0", "data_1", ...]`. No test verifies this path.

- **`generate_pixel_coordinates=True` (default) with `data_dict`-only construction is untested** — `test_from_data_dict` always passes `generate_pixel_coordinates=False`. The default `True` path (where coordinates are generated from the first value in a dict-only input) has no test.

---

### 1.6a — Fill the test coverage gaps documented by the §1.6 code review ✅

This item adds the tests identified as missing in §1.6. Source-code bug fixes remain in Phase 2 (§2.19). Tests that depend on a Phase 2 fix (single-spaxel, no-coordinate round-trips) remain `xfail` until §2.19 is implemented.

**A. Fix `test_mismatched_row_col_raises` to actually exercise the comparison branch**
- Replace the current four-row data (which has duplicated `(0,1)` and `(1,0)` pairs) with data containing duplicate diagonal pairs `(0,0),(0,0),(1,1),(1,1)`. These produce `max_row=1, max_col=1`, reshape to `(3,2,2)` successfully, and then fail the comparison because `col_readin=[[0,0],[1,1]] ≠ col_regenerated=[[0,1],[0,1]]`.
- Update the test docstring to accurately describe the failure mechanism.

**B. Add a 3D spatial (panel + row + col) round-trip test**
- Add `TestResultDictRoundTrip3D`: construct a `ResultDict` from a `(n_maps, 2, 3, 4)` array (2 panels, 3 rows, 4 cols), verify that `"panel"`, `"row"`, and `"col"` keys are inserted, and that a `savetxt`/`loadtxt` round-trip recovers identical data including all three coordinate arrays.

**C. Add constructor edge-case tests to `TestResultDictConstruction`**
- `test_combined_data_dict_and_data_array`: call `ResultDict(data_array=arr, names=["c"], data_dict={"a": x, "b": y})` and assert all three keys exist with correct values, and that the overlapping-key override behaviour is correct.
- `test_names_none_autogenerates`: call `ResultDict(data_array=arr, names=None, generate_pixel_coordinates=False)` and assert keys are `["data_0", "data_1", ...]`.
- `test_generate_pixel_coordinates_with_data_dict_only`: call `ResultDict(data_dict={"flux": arr2d})` (default `generate_pixel_coordinates=True`) and assert `"row"` and `"col"` are present with correct shapes.

### 1.7 — Integration smoke test for `fit_lines`

Broken into five sequential sub-steps.  Each can be implemented and run independently.  All live in a new file `tests/test_smoke_fit_lines.py`.

#### 1.7a — `update_settings` unit test ✅

Call `fit_lines.update_settings(s)` with a copy of the `default_settings` fixture (to avoid mutating the session fixture) and assert:

- `s.kernel` is a 2-D `numpy.ndarray` with `sum > 0` and odd side lengths centered on the origin.
- `s.instrument_dispersion_rest == s.instrument_dispersion` (because `z_set = 0`).
- `s.comment` is a non-empty string containing the key names `"instrument_dispersion"`, `"snr_lower_limit"`, and `"units"`.

This is pure Python with no file I/O and runs in milliseconds.

#### 1.7b — Single-spaxel fit test ✅

Without calling the full pipeline, manually build the inputs that `fit_line.run()` would produce for one spaxel and call `fit_line.process_single_spectrum()` directly:

1. Create the spatially-averaged subcube by calling `tc.fit.spatial_average(subcube, kernel)` on `synthetic_cube.select_lambda(L_OIII5007.low, L_OIII5007.high)`.
2. Build a trivial `snr_image` that passes every spaxel (all values = 999).
3. Call `process_single_spectrum(subcube_av, snr_image, snr_threshold=3, models=[Const_1GaussModel()], s=settings_ns, idx=(5, 5))`.

Assert:
- The result is a list of length 1.
- `result[0]` is not `None`.
- `result[0].success is True`.
- `result[0].params["g1_center"].value` is within 0.5 Å of `LINE_CENTER`.

Use a plain `SimpleNamespace` for `s` — only needs `lmfit_kwargs`, `chop_bandwidth`, and `instrument_dispersion_rest`.

**Bugs and gaps found while implementing 1.7b:**

- **`synthetic_cube` in conftest.py lacked a spatial WCS** — `tc.fit.spatial_average` clones the input cube and assigns mpdaf Images back into it. mpdaf's `Cube.__setitem__` checks WCS compatibility whenever *both* objects have `_has_wcs = True`. A `Cube` created without an explicit `wcs=` argument gets `_has_wcs = True` but `wcs = None` (an inconsistent internal state), causing `AttributeError: 'NoneType' object has no attribute 'get_step'`. Fixed in conftest.py by passing `wcs=MpdafWCS(cdelt=(0.2, 0.2), crval=(0, 0))`.

- **`is True` identity check for numpy bool in `process_single_spectrum`** — The NaN-SNR guard `np.isnan(snr_image[idx]) is True` uses Python identity comparison. `np.isnan()` returns `numpy.bool_` (confirmed numpy 1.26.4), which is **not** the Python `True` singleton, so the expression always evaluates to `False`. Spaxels with `snr_image[idx] = np.nan` are therefore not skipped — they are passed straight to the fitter. Added `test_snr_nan_skips_spaxel` marked `xfail(strict=True)` to document this. Fix: replace `is True` with a plain truthiness check `np.isnan(snr_image[idx])`. Fix target: Phase 2.

- **Missing test for SNR below threshold** — The happy-path test uses `snr_image = 999.0` everywhere. Added `test_snr_below_threshold_returns_none_list` that verifies `snr_image = 1.0 < threshold = 3` correctly returns `[None]` without fitting.

#### 1.7c — `fit_line.run()` for a single line, no MC, no file I/O ✅

Build a settings `SimpleNamespace` that mirrors the `default_settings` fixture but sets `mc_n_iterations=0`, `save_plots=False`, and uses a 3×3 subregion of `synthetic_cube` (to keep runtime short).  Call `fit_line.run(s)` with `s.output_filename` pointing to `tmp_path`.

Assert:
- Returns without raising (verified via `hasattr(run_state, "model_results")`).
- `s.model_results` has shape `(1, 3, 3)` (one model, 3×3 spatial).
- All 9 entries in `s.model_results[0]` are not `None` — the clean synthetic SNR is >> threshold, so every spaxel should be fitted.
- All 9 non-`None` entries are `lmfit.model.ModelResult` instances.
- All 9 entries have `success=True`.

Does not assert file content — that is covered in 1.7d.

**Code-review gaps fixed after initial implementation:**

- **`assert run_state is not None` was trivially true** — a `SimpleNamespace` returned by a pytest fixture can never be `None`. An erroring fixture shows as `ERROR`, not `FAIL`. Replaced with `assert hasattr(run_state, "model_results")`, which actually confirms `fit_line.run()` completed its main assignment.
- **`any()` was weaker than the data warrants** — the synthetic cube injects a clean high-SNR signal in every spaxel, so all 9 should fit. Using `any()` would let a run where 8/9 spaxels silently returned `None` pass undetected. Changed to `all()` in both result assertions.
- **No type check on fitted results** — added `test_all_results_are_model_result_instances` to assert every entry is `lmfit.model.ModelResult`. Without this, a non-result object deposited into `model_results` would pass the shape and `all(...)` checks.

#### 1.7d — Output files are created ✅

Using the same setup as 1.7c (but with the full 10×10 `synthetic_cube` and `mc_n_iterations=0`), assert that after `fit_line.run(s)` all three expected `.txt` files exist in `tmp_path`:

- `{output_filename}_5007_simple_model.txt`
- `{output_filename}_5007_best_fit.txt` — only when `len(models) > 1` (test both cases: single model and two models).
- `{output_filename}_5007_mc_best_fit.txt`

Also assert each file contains at least one non-comment, non-blank data row (stronger than a byte-count check, which would pass a header-only file).

**Code-review gaps fixed after initial implementation:**

- **`_LINE_SAVE_STR` was a magic string** — the constant `"5007"` was not tied to `tc.lines.L_OIII5007.save_str`. A failing run would produce a confusing `FileNotFoundError` rather than a clear assertion failure. Added a module-level `assert tc.lines.L_OIII5007.save_str == _LINE_SAVE_STR` guard that fails immediately with an explanatory message if the `save_str` ever changes.
- **`_make_run_settings` `_i` was hardcoded to 0** — added `_i=0` as an explicit parameter. A runtime `assert s._i < len(s.lines) and s._i < len(s.models)` fires immediately if a caller passes an out-of-range index, with a message directing them to extend the lines/models lists or use a dedicated fixture.
- **File "non-empty" check was `stat().st_size > 0`** — `ResultDict.savetxt` writes comment lines (starting with `#`) before any data. A file consisting entirely of comment lines would pass the size check but contain no usable data. Replaced all six `_nonempty_` tests with `_assert_has_data_rows()`, which strips comment/blank lines and asserts at least one data row remains.
- **`baseline_subtract=None` mutation risk** — if `baseline_subtract` were set to a non-None value, `fit_line.run` would subtract in-place from a subcube that is a *view* into the session-scoped `synthetic_cube`, silently corrupting shared test state. Added `baseline_subtract=None` as an explicit parameter with a runtime `assert baseline_subtract is None` guard; the error message directs future callers to pass `synthetic_cube.copy()` directly instead.

#### 1.7e — `ResultDict.loadtxt` round-trip ✅

After the 1.7d run, load each `.txt` file with `ResultDict.loadtxt` and assert:

- The returned object is a `ResultDict` (i.e. an `OrderedDict` subclass).
- `"row"` and `"col"` keys are present.
- The spatial shape inferred from `max(result["row"]) + 1` and `max(result["col"]) + 1` matches `(CUBE_NY, CUBE_NX)` = `(10, 10)`.
- All spaxels on a clean SNR=25 cube produce finite `g1_center` values (`np.all`, not `np.any`).
- All finite `g1_center` values are within 0.5 Å of the injected line centre (end-to-end physical correctness check).
- `.comment` is reconstructed as a `str` by `loadtxt`.

Both `simple_model.txt` and `mc_best_fit.txt` are checked. `best_fit.txt` is implicitly covered by the single-model fixture (only two files are written) — a dedicated two-model round-trip is deferred to 1.8 (covered as part of the row-count assertion in §1.7d item 5).

`mc_best_fit.txt` uses `avg_`-prefixed column names (e.g. `avg_g1_center`, not `g1_center`) because `extract_spaxel_info_mc` mediates storage. The mc assertions check the `avg_g1_center` key, finite-values, and proximity to the injected line centre. Since `mc_n_iterations=0`, `mc_iter(0)` returns `[self]` (the original fit only), so the median equals the direct fit value.

**Code-review findings applied after initial implementation:**

1. **Docstring corrected** — class docstring mistakenly claimed the fit ran only once; pytest does not share class-scoped fixtures across classes, so it runs a second time.
2. `np.any` → `np.all` for the finite-values assertion (same issue as 1.7c: `np.any` only catches total collapse).
3. **Vacuous-truth guard** — added `assert finite.size > 0` before the `np.all(abs(...) < 0.5)` proximity check; without it an all-NaN failure passes silently.
4. **`mc_result` semantic tests** — none existed; added key-presence (`avg_g1_center`), finite-values, and proximity-to-line checks.
5. **`.comment` tests** — `loadtxt` reconstructs the attribute; added `test_simple_model_comment_is_str` and `test_mc_result_comment_is_str`.

---

### 1.8 — Tests for `fit.py` utility functions

These are pure-logic functions with no GUI or mpdaf dependencies.  They are called throughout the pipeline and are the most likely things to silently break after a numpy or lmfit dependency bump.  Group them into three new test files.

#### A. `test_fit_utilities.py` — small helper functions ✅

**`get_index`**
- Single scalar `value`, single-element `array` → returns `0`.
- Exact match returns the correct index.
- Off-grid value returns the index of the nearest element (both directions, including tie-breaking towards the first element).
- Vectorised call: `value` is a list → returns a list of ints of the same length.
- Degenerate: `array` has one element, any `value` → always returns `0`.

**`iter_spaxel`**
- `index=False`: iterates over every pixel in row-major order, values match `image[y, x]`.
- `index=True`: second yield element is the `(y, x)` tuple, and the full set of index tuples equals `set(np.ndindex(*image.shape))`.
- Works for non-square arrays (3×5).
- Works for a 1×1 array (single spaxel).

**`get_region`**
- Circle (`rx=2`, `ry=None`): all returned pixels satisfy `row²+col² ≤ rx²`.
- Ellipse (`rx=3, ry=1`): correct pixel count and all pixels satisfy the ellipse inequality.
- Passing a two-element list `[rx, ry]` as the first argument gives the same result as passing `rx` and `ry` separately.
- `rx=0` returns only the origin `[0, 0]`. ⚠️ **Bug found (§2.21):** actual behaviour is to return an empty array — marked `xfail(strict=True)`.

**`get_reg_image`**
- Output shape is `(max_row − min_row + 1, max_col − min_col + 1)`.
- All pixels in `region` are set to 1; all others are 0.
- Round-trip with `get_region`: `get_reg_image(get_region(2)).sum()` equals `len(get_region(2))`.

**`de_redshift`**
- `z=0, z_initial=0` → `crval` and `step` unchanged.
- `z=z_initial` → no change.
- Known analytic case: `crval_out = crval_in * (1 + z_initial) / (1 + z)`.  Use a mock `WaveCoord`-like object with `get_crval`/`set_crval`/`get_step`/`set_step` to avoid an mpdaf dependency.
- Return value is `z`.

#### B. `test_aic.py` — AIC model-selection logic ✅

**`get_aic`**
- Returns `model.aic_real` when `model.success is True`.
- Returns `error` (default `np.nan`) when `model.success is False`.
- Returns `error` when `model` has no `aic_real` attribute (`AttributeError`).
- Returns `error` when `model` is `None` (no attribute at all).

**`choose_model_aic_single`**
- `model_list=None` → returns `-1`.
- Single-element list → always returns `1`.
- Two-model list, `aic[1] - aic[0] < d_aic` → returns `2`.
- Two-model list, difference ≥ `d_aic` → returns `1`.
- Two-model list, both AICs `nan` → returns `-1`.
- Three-model list: all four branches of the decision tree (2-better-than-1-and-3-better-than-2, 2-better-than-1-but-3-not, 3-better-than-1-skipping-2, none-better) each return the correct index.
- Custom `d_aic` threshold is respected.
- Exact boundary tie (`aic[1] - aic[0] == d_aic`): `<` is strict, so the simpler model always wins on a tie.
- List of 4 models: `test_more_than_three_models_raises` is `xfail(strict=True)` — sees today's silent fallthrough; fix in §2.22.
- Only the simpler model (index 0) has `NaN` AIC: `test_only_simpler_model_fails_returns_complex` is `xfail(strict=True)` — `NaN - valid = NaN < d_aic` is `False` in numpy, so the failed model 1 is returned instead of the only valid model 2; fix in §2.22.

**`choose_model_aic`**
- Single list (shape `(n_models,)`) delegates to `choose_model_aic_single` and returns a scalar.
- 2D spatial array `(ny, nx, n_models)`: output shape is `(ny, nx)` and each element matches independent `choose_model_aic_single` calls.
- Invalid spaxel (all-NaN AIC column) is assigned `-1` in the output array.
- Custom `d_aic` is propagated to `choose_model_aic_single` through the spatial broadcast loop.

**`get_ngaussians`**
- Model with 0, 1, 2, 3 gaussian components returns the correct count.
- Non-gaussian components (constant) are not counted.

**`get_gcomponent_comparison`**
- Single gaussian → returns `[]`.
- Two gaussians: ratio and delta-center are correct to floating-point tolerance.
- The main component (highest flux) is excluded from the output list.
- When the second gaussian has higher flux, it becomes the main and the first becomes the secondary (tests the `np.argmax` path in the other direction).

**`marginal_fits`**
- `None` model (un-fitted spaxel) → `False` (no user check needed).
- `model.success = False` → `True` (flag for inspection).
- Single-gaussian model → `False`.
- Two-gaussian model where secondary flux ratio and delta-centre are below both thresholds → `True`.
- Two-gaussian model where either condition is not met → `False`.
- `choices=-1` (invalid spaxel): `np.choose(choices-1, fit_list, mode="clip")` clips `-2` to `0`, selecting `fit_list[0]`; in production `fit_list[0]` is `None` so the result is `False` via the `None`-path (documented in `test_choices_minus_one_not_flagged`).
- Custom `flux` threshold: `flux=0.35` raises the flag boundary from default `0.25`.
- Custom `dmu` threshold: `dmu=0.35` narrows the separation-flagging window from default `0.5`.

**Code-review bugs found, not yet fixed (fix tasks in §2.22):**
- `choose_model_aic_single` with `len > 3` silently falls through to `return 1` — should raise `ValueError`. The docstring has a matching `TODO: generalize to more than 3`. (`test_more_than_three_models_raises`, `xfail strict`)
- `choose_model_aic_single` when only `aic[0]` is `NaN` (simpler model failed): `aic[1] - aic[0] = NaN`, and `NaN < d_aic` is `False`, so the function returns model 1 (the failed one) instead of model 2 (the only successful one). (`test_only_simpler_model_fails_returns_complex`, `xfail strict`)

#### C. `test_stats_collection.py` — result harvesting chain ✅

**`get_model_keys`**
- Single `ModelResult` → sorted list of parameter names, all present.
- Array of `ModelResult` objects (some `None`) → uses the first non-None entry.
- `ignore="fwhm height"` (string) → no returned key ends with any ignored suffix.
- `ignore=["fwhm", "height"]` (list) → same result as the string form.
- All-`None` numpy array → returns `[]`.
- Scalar `None` → returns `[]`.
- Returned list is always sorted alphabetically.

**`get_header_stats`**
- `fit_info="auto"` → first columns are `DEFAULT_FIT_INFO`, followed by `key` / `key_err` pairs.
- `fit_info=None` → only `key` / `key_err` columns (no `DEFAULT_FIT_INFO` entries).
- `model_keys=None` → only the `fit_info` columns, no crash; length equals `len(DEFAULT_FIT_INFO)`.
- `fit_info=None, model_keys=None` → returns `[]`.
- Length is `len(fit_info) + 2 * len(model_keys)` for all combinations.
- Each `key` entry is immediately followed by `key_err`.

**`collect_stats`**
- Valid result + matching keys: each `[value, stderr]` pair is finite and matches `result.params[k]` directly.
- `fit_info="auto"`: `result.aic_real` is the first element.
- Missing key (not in `result.params`) → two `empty_value` entries.
- `model_result=None`: returns `[empty_value] * (len(fit_info) + 2 * len(model_keys))`.
- Custom `empty_value` (e.g. `-999.0`) is propagated correctly for `None` results.
- `get_header_stats` and `collect_stats` return matching lengths for every combination of `model_keys` and `fit_info` (including both `None`).

**`RecursiveArray`**
- Attribute access on a flat list distributes over elements and returns a `RecursiveArray`.
- Call distributes over callable elements and returns a `RecursiveArray`.
- `None` elements in a call are passed through as `None` rather than raising.
- Nested (2D) construction wraps inner lists as `RecursiveArray` instances.
- `aslist()` on a flat array returns a plain `list`; on a nested array returns a list of lists.
- `array()` converts to a `numpy.ndarray` with the correct dtype; `dtype=bool` also works.
- Missing attribute name (`getattr` default) returns `None` for each element.

**Code-review bugs found, not yet fixed (fix task in §2.23):**
- `RecursiveArray.__init__` accesses `self.data[0]` unconditionally — `IndexError` on empty list or `None` argument. (`test_empty_list_does_not_raise`, `xfail strict`)
- `RecursiveArray.aslist()` has the **same `self.data[0]` access** — even after the `__init__` fix, calling `aslist()` on an empty instance still crashes. The fix must guard both methods. (`test_aslist_empty_list_does_not_raise`, `xfail strict`; test bypasses `__init__` by injecting `ra.data = []` directly to isolate the `aslist` bug independently.)

**Additional gaps found during code review and added as tests:**
- `get_model_keys` with a plain `lmfit.Model` (not yet fitted): exercises the `isinstance(model, lmfit.Model) → model.make_params()` source branch.
- `get_model_keys` with a 2D spatial numpy array `(2, 3, 4)`: real production arrays are `(n_models, ny, nx)`; `.flat` handles it but was untested.
- `get_header_stats` with a custom `fit_info` list (not `"auto"` or `None`): exercises the fall-through else branch.
- `get_header_stats` custom `fit_info` does not mutate the caller's list (the `.copy()` guard).
- `collect_stats` with a custom `fit_info` list.
- `collect_stats` with an invalid `fit_info` attribute name: `AttributeError` mid-loop causes the `except` block to discard all previously-collected values and return all-`empty_value` — silent data loss documented as expected behavior.
- `RecursiveArray.__call__` with a positional argument: `*args/**kwargs` forwarding was untested.

#### D. `test_lmfit_ext.py` — `lmfit_ext` extensions ✅

These require a real `ModelResult` from a trivial fit (single gaussian on synthetic data); use the fixture pattern already established in `test_model_function.py`.

**`aic_real` / `bic_real`**
- `aic_real = chisqr + 2 * nvarys` (exact).
- `bic_real = chisqr + log(ndata) * nvarys` (exact).
- Both return `None` when called on an object missing `chisqr`.

**`stderrsdict`**
- Keys equal `params.keys()`.
- Values equal the per-parameter `stderr` from the fit result.

**`valerrsdict`**
- For each parameter name `k`, both `k` (value) and `k + "_err"` (stderr) are present.
- Values match those from the `ModelResult.params` directly.

**`set_param_hints_endswith`**
- After calling `model.set_param_hints_endswith("_sigma", min=0.5)` on a `Const_2GaussModel`, every parameter ending in `_sigma` has `min=0.5`; others are unchanged.
- A suffix that matches no parameter is a no-op (no exception).

**`order_gauss`**
- Two-gaussian result where g2 has a smaller center than g1: after `order_gauss`, `g1_center < g2_center`.
- When centers differ by less than `delta_x`, the taller component is placed second.
- Single-gaussian result: returns immediately with no changes.
- All expressions are cleared before reordering (no leftover `expr` strings).

**`summary_array`**
- With `fit_info=["redchi"]` and `param_info=["g1_center", "g1_center_err"]`: returns a 3-element float array matching `[result.redchi, result.params["g1_center"].value, result.params["g1_center"].stderr]`.
- With empty `fit_info` and `param_info`: returns a zero-length array.

**Code-review bugs found, not yet fixed (fix tasks in §2.16 and §2.24):**
- `set_param_hint_endswith` (registered as singular on `Model`, despite the function being named `set_param_hints_endswith`) unconditionally overwrites existing `min`/`max` hints — pre-set stricter bounds are silently replaced. (`test_preexisting_stricter_min_not_overwritten`, `xfail strict`; fix in §2.16)
- `order_gauss` assumes `g{n}_height` parameters exist and calls `.value` on `self.get("g{n}_height")`, crashing with `AttributeError: 'NoneType' object has no attribute 'value'` when height parameters are absent (e.g. standard lmfit `GaussianModel` which uses `amplitude`). (`test_order_gauss_missing_height_param_does_not_crash`, `xfail strict`; fix in §2.24)

**Additional gaps found during code review and added as tests:**
- `aic_real` / `bic_real`: `bic > aic` verified for `ndata=120 > e²`.
- `aic_real` / `bic_real`: `None + int → TypeError` path is caught (alongside `AttributeError`).
- `set_param_hint_endswith`: multiple kwargs (`min` + `max`) are applied together.
- `order_gauss`: 3-gaussian full reorder (g1=5020, g2=5000, g3=5010 → sorted ascending).
- `order_gauss`: no-gaussian params (only `c`) — function returns immediately.
- `summary_array`: missing `param_info` key → `d.get` returns `None` → `np.array(dtype=float)` → `NaN`.
- `summary_array`: a misspelt `fit_info` attribute raises `AttributeError` (no silent `NaN` — asymmetric with `param_info` handling; documented as expected behaviour).

---

### 1.9 — Tests for procedure helper functions

The five procedures (`open_cube_and_deredshift`, `fit_lines`, `fit_line`, `analyze_outflow_extent`, `explore_results`) divide into four categories:

- **`open_cube_and_deredshift`** — pure orchestration glue (settings + mpdaf calls); no pure-logic helpers to extract. Its settings-processing behaviour is already covered by 1.2 and the smoke test in 1.7.
- **`fit_lines`** — the top-level pipeline; covered end-to-end by 1.7.
- **`explore_results`** — interactive matplotlib widget; not unit-testable.
- **`set_rcParams`** — two wrappers around `mpl.rcParams`; trivial.

That leaves two procedures with testable logic:

#### A. `fit_line` sub-functions (`tests/test_procedures_fit_line.py`) ✅

`process_single_spectrum` contains the only branching logic not exercised by the 1.7 smoke test.  Use the synthetic cube fixture from 1.1 and real `Const_1GaussModel` / `Const_2GaussModel` instances so that the lmfit path is exercised without mocking.

**`process_single_spectrum`**
- SNR below threshold → returns `[None]` without fitting.
- `snr_image` value is `np.nan` → returns `[None]`.
- Single model, fit succeeds → returns a one-element list containing a successful `ModelResult`.
- Two models, both succeed → returns a two-element list.
- First model fails and `s.chop_bandwidth = False` → returns `[None]` immediately.
- First model fails and `s.chop_bandwidth = True` → spectrum is narrowed by ±5 Å and the fit is retried; a successful second attempt returns a one-element list.

**`choose_best_fits`**
- Single model in list → `auto_aic_choices`, `user_check`, and `final_choices` are all `None`; no exception.
- Multiple models, `interactively_choose_fits=False` → `final_choices` is populated from `choose_model_aic`; `always_manually_choose` entries in `user_check` are `True`.
- `always_manually_choose=[]` (empty) → `user_check` is entirely determined by `marginal_fits`.

**Code-review bugs found, not yet fixed (fix tasks in §2.25 and §2.26):**
- `np.isnan(snr_image[idx]) is True` uses Python identity comparison against the `True` singleton, but `np.isnan` returns `numpy.bool_` (not Python `bool`). `numpy.bool_(True) is True` evaluates to `False`, so the NaN gate never fires. NaN SNR pixels are incorrectly forwarded to fitting instead of returning `[None]`. (`test_snr_nan_returns_none_list`, `xfail strict`; fix in §2.25)
- The chop-bandwidth retry path does not guard against `lmfit` returning `None` (all-masked chopped spectrum). After the retry, the code proceeds directly to `if f.success is False:`, raising `AttributeError: 'NoneType' object has no attribute 'success'`. The first call has this guard; the retry is missing it. (`test_first_model_fail_chop_true_retry_returns_none_on_masked_spectrum`, `xfail strict`; fix in §2.26)

**Additional gaps found during code review and added as tests:**
- SNR gate is strict less-than: a pixel with SNR exactly equal to the threshold proceeds to fitting. Test `test_snr_equal_to_threshold_proceeds_to_fitting` documents this boundary.
- `lmfit` returning `None` (all-masked spectrum) on the first call → `[None]` via the existing guard.
- Retry after chop where both original and chopped fits fail with `success=False` → `[None]`.
- Different spatial indices produce independent results (`test_result_spaxel_index_applies_correctly`).
- chop-bandwidth retry call count verified to be exactly 2.

**Additional gaps found during fresh code review (3 new tests added, 1 cleanup):**
- `auto_aic_choices` values verified: `(auto == 1).all()` when model 1 has better AIC (`test_auto_aic_choices_all_one_when_model_one_wins`).
- `chosen_models.shape` verified in the multi-model case (`test_multiple_models_chosen_models_shape_matches_spatial`).
- Model-2 selection path exercised: `aic2<<aic1` → `(auto == 2).all()` and `chosen_models` contains 2G objects from `fit_results[1]` (`test_auto_aic_chooses_model_two_when_significantly_better`; `tc.fit.marginal_fits` patched to isolate AIC path from mock `.params` requirement).
- Cleaned up `is True or == True` pattern in `test_always_manually_choose_sets_user_check_true` → `assert user[(0, 0)]`.

**Final state after fresh review:** 26 passed, 2 xfailed.

#### B. `analyze_outflow_extent` helpers (`tests/test_procedures_analyze_outflow_extent.py`) ✅ (complete)

All functions here are pure numpy; no mpdaf or matplotlib dependency.

**`distance`** ✅ (implemented)
- `distance(origin[0], origin[1], origin)` → `0.0`.
- Known Pythagorean triple: `distance(3, 4, [0, 0])` → `5.0`.
- Works element-wise on numpy arrays of the same shape.

**`sort_data`** ✅ (implemented)
- Unordered `x` → output `x` is monotonically increasing.
- Masked array input: masked entries are removed (compressed) from both `x` and `data`.
- `x` and `data` maintain the same correspondence after sorting.

**`boxcar_average_1d`** ✅ (implemented)
- `width=1` → output equals input (identity).
- `width=3` on a constant array → interior rows unchanged (edges are zero-padded by `np.convolve(mode='same')`).
- `width=3` on a step function: interior transition value averaged correctly.
- `axis=1` applies smoothing along columns rather than rows.

**`row_max`** ✅ (implemented)
- Returns two arrays of equal length (one per row of input).
- First returned array is `np.arange(0, n_rows)`.
- Obvious outlier column (single row with a far-off peak) is masked after sigma clipping.
- Values above/below `center_row` are replaced by their respective medians.

**`compute_gal_center_row`** ✅ (implemented)
- Synthetic image with a horizontal bright stripe at a known row → returns that row index.
- Noisy low-flux columns at image edges do not shift the result (the 1% flux threshold filters them).

**`radius_at_fraction`** ✅ (implemented)
- `values=[50]` on a uniform array of length `n` → returned radius is the first x where cumsum strictly exceeds 50% of total (strict `>` comparison).
- `values` given as percentages (>1) are divided by 100 before use.
- `return_string=True` → returns a list of strings of the form `"r_50 = ..."`.
- Scalar `values` (not a list) does not raise (wraps to array internally).

**`calculate_contours`** ✅ (implemented)
- Synthetic 5-row flux array with a known peak column per row → returned contour widths match the manually computed cumulative-sum half-widths.
- A fully masked row produces a masked entry in the output (not a crash). `np.array()` correctly propagates `np.ma.masked` constants from the Python list, creating a masked array before `.T` and `.astype(int)`.
- `levels` are sorted ascending before processing regardless of input order.
- Peak near right edge with spread flux (so the goal isn't met at count=0) → the expansion loop tries `this_row[max_col + count]` beyond the array bounds and raises `IndexError`. (`test_near_edge_peak_does_not_crash_or_wrap`, `xfail strict`; fix in §2.31)

**`contours_to_arcsec`** ✅ (implemented)
- Default arguments (center=0, scale=1) → output equals input.
- Input array is not modified in-place (function copies before modifying).
- Row channel (index 0) is reduced by `galaxy_center_row`; col channel (index 1) by `galaxy_center_col`.
- `arcsec_to_pixel` multiplies the entire shifted array.

**`create_outflow_mask`** ✅ (implemented)
- All pixels outside the contour region are `True`; pixels inside are `False`.
- `which_contour` selects the correct level from `contour_levels`.
- A masked `center_col` entry → that row is left fully masked (`True`) rather than raising.

**`extract_wcs` / `process_arcsecs` / `process_units`** ✅ (implemented) (header parsing)
- `extract_wcs`: the pipeline comment format is `"wcs_step: [dy dx]"` (square brackets, space-separated). The function strips the key, replaces spaces with commas, and `eval()`s the result to get a Python list `[dy, dx]`.
- `process_arcsecs`: numeric input is passed through unchanged; `"header"`/`"auto"`/`None` triggers `extract_wcs`; missing `"wcs_step"` line raises `ValueError` (wraps `IndexError`). Non-scalar results (list/tuple) return only `[0]`.
- `process_units`: string `"header"`/`"auto"`/`None` reads from `"units: <string>"` comment line; explicit string is converted to `astropy.units.Unit`; an already-Unit object is returned unchanged.

**Code-review bugs found (fix tasks in §2.27, §2.28, §2.29, §2.30, §2.31):**
- `sort_data` is annotated `# ## ASSUMES SAME MASK` but does not enforce it. When `x` and `data` have *different* masks, `np.ma.compressed(x)` and `np.ma.compressed(data)` produce arrays of different lengths. The sort indices computed on the shorter array are silently applied to `data`, truncating or misaligning values without raising. (`test_mismatched_masks_raises_not_silent_corruption`, `xfail strict`; fix in §2.27)
- `row_max` filters raw `argmax` results with `rowmax[rowmax > 0]` to exclude masked-row artefacts (which `argmax` defaults to 0). This also silently excludes genuine peaks at column 0 from the mean/std calculation used for sigma-clipping, biasing outlier detection. When *all* rows peak at column 0 the filtered array is empty, `mean()` and `std()` produce NaN, and the function emits `RuntimeWarning: Mean of empty slice` before proceeding with NaN-contaminated logic. (`test_genuine_peak_at_column_zero_not_excluded_from_stats`, `xfail strict`; fix in §2.28)
- `radius_at_fraction` collects hit indices into `results` inside a loop and calls `np.column_stack([values, x[results]])` at the end. If the cumulative sum of `y` never reaches a goal (e.g. a fraction > 1.0 is requested, or y sums to zero), the loop ends before all goals are satisfied, `len(results) < len(values)`, and `column_stack` raises `ValueError`. (`test_unreachable_fraction_does_not_crash`, `xfail strict`; fix in §2.29)
- `create_outflow_mask` finds `which_contour` with a `for` loop + `break`; if `which_contour` is not in `contour_levels`, `idx` is never set. The next line `col_span = line[2 + idx]` raises `UnboundLocalError` instead of a descriptive `ValueError`. (`test_unknown_which_contour_raises_not_silent`, `xfail strict`; fix in §2.30)
- `calculate_contours` expands outward symmetrically with `this_row[max_col - count]` and `this_row[max_col + count]`. When the peak is near the right edge, `max_col + count` exceeds the array length and raises `IndexError`. When near the left edge, `max_col - count < 0` silently wraps (Python negative indexing) and adds flux from the wrong end. (`test_near_edge_peak_does_not_crash_or_wrap`, `xfail strict`; fix in §2.31)

**Additional gaps found during code review and added as tests:**
- `boxcar_average_1d` edge behavior documented: `mode='same'` zero-pads the boundary, so the first and last entries are lower than the constant value; interior rows are unchanged (test renamed to `test_width_three_constant_interior_unchanged`).
- `distance` called with masked array inputs (production usage): mask propagates correctly to output (`test_masked_array_inputs_propagate_mask`).
- `radius_at_fraction`: 50% goal uses strict `>`, so the radius is the x value at the first index where cumsum *exceeds* the goal — documented and corrected from the off-by-one initial assumption.
- `radius_at_fraction`: `return_array.shape == (n_values, 2)` verified; first column contains the fractional values.
- `compute_gal_center_row`: return type is `int` (explicit `int(...)` cast).
- `extract_wcs`: ROADMAP spec had wrong format. Actual pipeline comment is `"wcs_step: [dy dx]"` (brackets, space-separated); the function does `.replace(" ", ",")` then `eval()` to get a Python list, not a tuple. Using `"wcs_step: (0.2, 0.2)"` (comma-space) would produce `"(0.2,,0.2)"` — a SyntaxError.
- `extract_wcs` uses `eval()` on a string from file headers, which is a security concern if comment lines can be attacker-controlled. Design choice, not a crash bug; documented but not xfailed.
- `process_units` raises raw `IndexError` for a missing `"units:"` line, while `process_arcsecs` wraps the equivalent `IndexError` in a `ValueError`. The inconsistency is noted.
- `create_outflow_mask`: correctly handles masked `center_col` (skips row, left all-True). `np.ma.masked` in Python list → `np.array()` correctly creates a masked array (mask is not lost through `.T` and `.astype(int)`).
- `contours_to_arcsec`: `count` reuse was initially a concern (the `count` variable in `calculate_contours` is not reset between level goals); verified that accumulation is intentional (each successive level goal is larger, so count only grows).

**Current state:** 62 passed, 5 xfailed (§2.27, §2.28, §2.29, §2.30, §2.31). Phase 1.9B complete.

---

### 1.10 — Smoke tests for matplotlib-dependent functions ✅

The Phase 1 safety net has no coverage of any code that calls matplotlib directly.
Before updating matplotlib (Phase 0b), add minimal smoke tests that catch API renames,
removed functions, and changed return types — without trying to assert pixel-level rendering.

**Scope:** The three lowest-cost, highest-value targets:

1. **`plot2` (lmfit_ext)** — monkey-patched onto `ModelResult`; calls `plt.GridSpec`,
   `fig.add_subplot`, `plt.setp`, `plt.rcParams`. Tested with the module-level `_RESULT`
   fixture already used in `test_lmfit_ext.py`.
   - Returns `(fig, ax_res, ax_fit)` tuple (types verified).
   - Accepts an existing `Figure` and reuses it.

2. **`plot_components` (lmfit_ext)** — monkey-patched onto `ModelResult`; calls
   `ax.plot`, `ax.axhline`. Tested with same `_RESULT` fixture.
   - Returns an `Axes` object.
   - Accepts and returns an existing `Axes`.

3. **`plt_image_extent` (analyze_outflow_extent)** — calls `plt.imshow`, `plt.gca()`,
   `plt.xlabel`, `plt.ylabel`, `axhline`. Pure-pyplot stateful style; no fig/ax passed in.
   - Does not raise with a synthetic 2-D array and numeric extent.
   - `horizontal0=False` adds no `"galaxy midplane"` axhline.
   - `horizontal0=True` (default) adds exactly one `"galaxy midplane"` axhline.

**Not added (too coupled to mpdaf objects, low ROI):**
- `save_pdf_plots`, `plot_ModelResults_pixel`, `plot_baseline` in `fit.py`.
- All `interactive_*` functions (block on `plt.show`).

**Files modified:**
- `tests/test_lmfit_ext.py` — new classes `TestPlot2Smoke`, `TestPlotComponentsSmoke`
- `tests/test_procedures_analyze_outflow_extent.py` — new class `TestPltImageExtentSmoke`

All tests use `matplotlib.use("Agg")` (non-interactive backend, safe in CI).

---

## Phase 0b — Dependency Modernisation (requires Phase 1 safety net)

*These tasks carry real risk of breaking behaviour. The Phase 1 tests are your safety net — run the full suite after each step.*

### 0b.1 — Resolve the custom `lmfit` fork dependency ✅
The `pyproject.toml` pinned `lmfit` to `sebusch/light-lmfit-py@light_dev` — a private fork installed from GitHub.

**Audit findings (fork vs upstream 1.3.4):**
- `parameter.py`: fork added `from numba import njit` and JIT-compiled 3 bound-transform helpers — performance only, numba already required by `fast_models.py`.
- `minimizer.py`: fork changed `.aic`/`.bic` formula to linear `chisqr + 2*nvarys`. Threadcount never reads `.aic`/`.bic` directly; it exclusively uses `aic_real` from `lmfit_ext.py`, which already implements that same formula as a monkey-patch.
- `parameter.py`: fork had a typo `sdterr` instead of `stderr` in uncertainty propagation — switching to upstream *fixes* this latent bug.
- `model.py`: fork used deprecated `np.asfarray()` (removed in numpy 2); upstream uses safe `np.asarray(..., dtype=float64)`.
- Everything else: reformatting only (single → double quotes, whitespace).

**Resolution:** switched to upstream `lmfit >= 1.3.4`. All fork-specific behaviour was either already vendored in `lmfit_ext.py` or was a bug. The numba JIT performance patch for `Parameter.setup_bounds` (which accelerates the bound-transform hot path to correlate with `fast_models.py`) has been re-implemented as a monkey-patch in `lmfit_ext.py` (`_numba_setup_bounds`). The original lmfit method is saved as `_original_setup_bounds` immediately before `extend_lmfit` replaces it, making it available for two-branch comparison in tests. Tests added in `test_lmfit_ext.py`: `TestNumbaKernels` (kernel math correctness against analytic formulas), `TestNumbaMatchesLmfit` (two-branch comparison: our patch vs `_original_setup_bounds` at a grid of probe values for all 4 bound cases — catches formula divergence independently of self-consistency), `TestNumbaSetupBounds` (self-consistent round-trips confirming invertibility), `TestNumbaSetupBoundsIntegration` (bounded fit convergence). Full test suite: **57 passed, 2 xfailed** in `test_lmfit_ext.py`.

### 0b.2 — Modernise `pyproject.toml`
- Python `>= 3.6` is EOL. Raise the floor to `>= 3.10` (f-strings, `match`, `dataclasses`, `typing` improvements become available without backports).
- Run the test suite against **numpy 2** (install it in a fresh env). If all tests pass, drop the `numpy < 2` upper-bound pin entirely. If they don't, the failures pinpoint exactly what needs fixing before the pin can be removed.
- Add a lower bound `numpy >= 1.23` regardless, since the current constraint is one-sided and underspecified.
- Expand the CI matrix (from 0a.2) to cover the newly-supported Python and numpy versions.

> **Process note — test-first methodology**: Before upgrading a pinned dependency, first confirm the current full test suite is green as a baseline. Then upgrade and run the suite again. Any new failures are unambiguously caused by the version bump. This is the correct order regardless of whether you expect failures; it turns the upgrade into a structured experiment with a clear before/after. (In 0b.1, the lmfit fork was swapped without first writing characterisation tests against the fork. This worked because the fork's contract was mathematically identical to upstream — but the `sdterr` typo and `.aic`/`.bic` formula differences in the fork would have been caught earlier had fork-specific tests existed first.)

---

## Phase 2 — Non-Breaking Code Quality (safe to do in any order within the phase)

*Improvements that change no public API and cannot break user scripts.*

### 2.1 — Replace `print()` with `logging`
Throughout `fit.py`, `procedures/fit_lines.py` and others, all diagnostic output uses `print()`. This cannot be silenced or redirected.
- Add `import logging; logger = logging.getLogger(__name__)` to each module.
- Replace `print(...)` with `logger.info(...)` / `logger.debug(...)` / `logger.warning(...)`.
- Users who want console output call `logging.basicConfig()` as they already would in any Python app.
- **Backwards compatible**: `print` output disappears, but no user script calls `print` directly — they just observe it.

### 2.2 — Fix bare `except` / overly broad exception catching
Replace `except:` and `except Exception:` with specific exception types. Where a broad catch is truly intentional (e.g. third-party libraries raising unexpected types), add a comment explaining why, and log at `WARNING` level before continuing.

### 2.3 — Replace `.format()` strings with f-strings
Mechanical find-and-replace across all files. `ruff` can automate most of this (`UP032`). Zero semantic change, improves readability.

### 2.4 — Document hardcoded physics constants
In `models/fast_models.py`:
- `gaussian4CH_constrained_SII_d_DELTAX24 = -14.37` — add a comment with its origin (vacuum wavelength difference between [S II] λ6731 and λ6717 in Å).
- `gaussian6CH_constrained_HaNII_d_DELTAX24 = -14.769` — similarly document Hα/[N II] separations.
- Consider promoting these to named module-level constants with `ALL_CAPS` names and docstrings, rather than embedding them silently inside function defaults.

### 2.27 — Fix `sort_data` silent data corruption on mismatched masks
`sort_data(x, data)` is annotated `# ## ASSUMES SAME MASK` but does not enforce this precondition. When `x` and `data` have different masks, `np.ma.compressed` returns arrays of different lengths. The argsort indices from the shorter `x` array are applied to `data`, silently truncating or misaligning values with no error. In production, `x` and `data` are constructed from the same boolean mask so this is not triggered, but the lack of a guard makes the function unsafe to call in other contexts.
- **Fix**: add `if ma.getmaskarray(x).shape != ma.getmaskarray(data).shape or not np.array_equal(ma.getmaskarray(x), ma.getmaskarray(data)): raise ValueError("x and data must share the same mask")` before the `compressed` calls, or derive a common mask via `ma.mask_or` and apply it to both before compressing.
- (`test_mismatched_masks_raises_not_silent_corruption`, `xfail strict`)

### 2.28 — Fix `row_max` column-0 peaks excluded from sigma-clip statistics
`row_max` filters raw `argmax` results with `rowmax[rowmax > 0]` before computing the mean and std used for sigma-clipping. The intent is to exclude masked-row artefacts (for which `np.argmax` returns 0 by default), but the filter also silently excludes genuine peaks that sit at column 0, biasing the outlier detection threshold. When *all* row peaks are at column 0 the filtered array is empty: `mean()` and `std()` produce NaN and the function emits `RuntimeWarning: Mean of empty slice` before proceeding with NaN-contaminated logic, masking all or none of the rows unpredictably.
- **Fix**: instead of filtering by value `> 0`, construct a proper masked array from the flux input and use `np.ma.compressed(rowmax)` on a version masked where the entire row is masked in the flux array (i.e. `rowmax = np.ma.masked_where(flux_masked_array.mask.all(axis=1), rowmax)`). This separates "row was entirely masked" (real artefact) from "row peaked at column 0" (valid data).
- (`test_genuine_peak_at_column_zero_not_excluded_from_stats`, `xfail strict`)

### 2.29 — Fix `radius_at_fraction` crash on unreachable cumulative fractions
`radius_at_fraction` iterates over fraction goals and appends an index to `results` only when the cumulative sum of `y` first exceeds the goal. If a goal is never reached (e.g. a fraction > 1.0 is passed, or `y` sums to zero/NaN), the loop ends with `len(results) < len(values)`. The subsequent `np.column_stack([values, x[results]])` then raises `ValueError: all the input array dimensions except for the concatenation axis must match exactly`.
- **Fix**: initialise `results` with `np.full(len(values), np.nan)` (float placeholder) and replace the append loop with indexed assignment; or after the loop check `len(results) == len(values)` and pad missing entries with `np.nan` before calling `column_stack`.
- (`test_unreachable_fraction_does_not_crash`, `xfail strict`)

### 2.30 — Fix `create_outflow_mask` UnboundLocalError on unknown `which_contour`
`create_outflow_mask` searches for `which_contour` in `contour_levels` using a `for` loop with `break`, and stores the matching index in `idx`. If `which_contour` is not present in `contour_levels`, the loop completes without executing `break` and `idx` is never assigned. The next reference to `idx` in `col_span = line[2 + idx]` raises `UnboundLocalError: local variable 'idx' referenced before assignment` rather than a descriptive error.
- **Fix**: replace the loop with `idx = list(contour_levels).index(which_contour)`, which naturally raises `ValueError: <value> is not in list` if the value is absent. Alternatively, add an `else` clause to the `for` loop or an explicit guard before the loop.
- (`test_unknown_which_contour_raises_not_silent`, `xfail strict`)

### 2.31 — Fix `calculate_contours` out-of-bounds expansion near array edges
`calculate_contours` expands outward from the row's peak column with `this_row[max_col - count] + this_row[max_col + count]`. When the peak is near the **right** edge, `max_col + count` exceeds the array length and raises `IndexError`. When near the **left** edge, `max_col - count < 0` silently wraps via Python's negative-indexing convention and adds flux from the far end of the row, producing incorrect half-widths without any error.
- **Fix**: before the expansion loop, clamp the accessible range to `[0, len(this_row) - 1]` and treat out-of-range positions as zero flux — e.g. use `np.pad(this_row.filled(0), max_col)` to create a symmetric window centred on `max_col`, or guard inside the loop with `count = min(count, max_col, len(this_row) - 1 - max_col)` before the index access.
- (`test_near_edge_peak_does_not_crash_or_wrap`, `xfail strict`)

### 2.5 — Remove dead code
- Large blocks of commented-out functions in `fit.py` (`compile_spaxel_info_mc`, `create_label_row_mc`, etc.) — delete them. They are in version control history if needed.
- `_guess_2gauss_old()` and `_guess_3gauss_old()` in `models/models.py` — delete or keep with an `_old` deprecation warning.
- Incomplete `set_component_param_hints()` stub in models — either complete it or remove it.
- Commented-out `__copy__` / `copy` methods in `lines.py` — delete them.

### 2.6 — Fix the seeded RNG in Monte Carlo
In `lmfit_ext.py` `mc_iter`, `np.random.default_rng(42)` is re-created every call with the same seed. This means repeated calls produce **identical** noise draws, which defeats the purpose of Monte Carlo.
- Move the RNG creation to module level (one RNG per session), or accept an optional `seed` parameter for reproducible testing.

### 2.7 — Fix global pixel-position state in `explore_results`
In `procedures/explore_results.py`, `p` and `q` (current pixel position) are module-level globals mutated by callback functions. This is not thread-safe and makes testing impossible.
- Wrap them in a small state object (a dataclass with two int fields) that is closed over by the callbacks.

### 2.8 — Lazy `matplotlib` import in `lines.py`
`import matplotlib.pyplot as plt` sits at the top of `lines.py`, so importing any wavelength constant (e.g. `from threadcount.lines import L_OIII5007`) silently drags matplotlib into the process. Move the import inside `Line.plot()` so it only loads when plotting is actually requested.

### 2.9 — Make `Line.low` / `Line.high` live-computed properties
`low` and `high` are computed once in `__init__` from `center`, `plus`, and `minus`. Mutating any of those attributes afterwards leaves `low`/`high` stale. Convert them to `@property` so they are always `center - minus` and `center + plus` respectively. The existing `self.low` / `self.high` assignments in `__init__` are simply removed.

### 2.10 — Add `Line.__eq__`
`Line` is a data-holding object, but two instances with identical parameters are not equal (`Line(5006.843) == Line(5006.843)` is `False`). Add `__eq__` comparing `__dict__` so equality works naturally in tests and user code.

### 2.11 — Guard `Line(**kwargs)` against overwriting core attributes
`Line.__init__` stores extra keyword arguments via `self.__dict__.update(**kwargs)`, which silently overwrites core attributes if a caller passes e.g. `center=9999`. Add a check that raises `TypeError` for any kwarg whose name collides with a core attribute (`center`, `plus`, `minus`, `low`, `high`, `label`, `save_str`).

### 2.12 — Fix truncated FWHM factor in `fwhm_expr` / `flux_expr`
Both `lmfit.models.fwhm_expr` and the local `flux_expr` in `models/models.py` format the scaling factor with `:.7f`.  For FWHM this yields `2.3548200` instead of the exact `2.3548200450309493`, a relative error of ~1.9×10⁻⁸.  For flux the factor is `2.5066283` vs exact `2.5066282746310002`.
- The truncation error is ~2×10⁻⁷ Å for `fwhm` and ~10⁻⁸ relative for `flux` — both are 4–6 orders of magnitude below SNR-limited measurement precision. This is a cosmetic tidiness issue only, not a scientific one.
- If desired, replace `:.7f` with `:.15g` in `flux_expr`, `fwhm_expr_fast`, and `flux_expr_fast`. Saved `.txt` files would shift by at most those tiny amounts.
- **No regression test should be added** for sub-1e-11 precision — it would test numerical trivia rather than physical correctness. The existing loose tolerances in `test_fit_fwhm_consistent` and `test_flux_numerical_value` are appropriate.

### 2.13 — Raise instead of silently ignoring `prefix` on composite models
All composite model classes (`Const_1GaussModel`, `Const_2GaussModel`, `Const_3GaussModel`, `Quadratic_*`, `Log10_DoubleExponentialModel`) accept a `prefix` argument but print a warning and ignore it, because `lmfit.model.CompositeModel` does not propagate prefix cleanly.
- Short-term: change the `print()` to `raise NotImplementedError(...)` so callers get a clear error instead of silently wrong behaviour.
- Long-term (Phase 3): migrate these models to `lmfit.Model` (like the fast variants) to gain proper prefix support.

### 2.14 — Support absorption-line fitting in `GaussianModelH`
`GaussianModelH._set_paramhints_prefix()` hard-codes `height min=0`, which means `guess(negative=True)` is silently broken: `guess_from_peak()` returns the correct negative height, but `reapply_certain_model_hints()` immediately clamps it back to 0.  The model cannot currently fit absorption features.
- Remove the `height min=0` hint from `_set_paramhints_prefix()` (or make it conditional on a constructor argument `absorption=False`).
- Ensure `guess(negative=True)` propagates a negative starting value cleanly.
- `tests/test_model_function.py::TestGaussianModelH::test_guess_negative_profile` already documents the current (broken) clamping behaviour and must be updated to assert `height.value < 0` once this is fixed.
- Add a new `test_fit_absorption_line` covering a negative-height fit round-trip.

### 2.15 — Document `_guess_2gauss` / `_guess_3gauss` applicability
`_guess_2gauss` default `centers=(-2, 0)` is designed for the narrow+broad same-centre outflow decomposition (both components at the same wavelength).  It **fails silently** for well-separated doublets (e.g. [NII] 6548/6563) where the peaks are >5 σ apart — the initial guess for the weaker component ends up far from its true position.
- Add a one-sentence docstring note on this constraint to `_guess_2gauss` and `_guess_2gauss_d`.
- Add a cross-reference to `_guess_multiline2` / `_guess_multiline2_d` as the correct choice for spatially-offset doublets.
- Consider adding a `separate_lines=False` flag that routes to the multiline variant automatically.

### 2.16 — Fix `set_param_hint_endswith` silently overwriting stricter user bounds
`lmfit_ext.set_param_hints_endswith` calls `model.set_param_hint(name, **kwargs)` unconditionally for every matching parameter. When `fit_line.py` uses it to apply the instrument-dispersion floor (`min=instrument_dispersion_rest`) it overwrites any tighter `min` or looser `max` the user already set on a specific parameter — e.g. a custom model with `g2_sigma min=2` would silently have that minimum replaced by `~0.77`.

**Workaround** (in user scripts): add the following monkey-patch at the top of the script, before importing threadcount procedures:
```python
import lmfit

def _set_param_hints_endswith_conservative(self, name, **kwargs):
    for this_name in self.param_names:
        if this_name.endswith(name):
            merged = dict(kwargs)
            existing = self.param_hints.get(this_name, {})
            if "min" in merged and "min" in existing:
                merged["min"] = max(merged["min"], existing["min"])
            if "max" in merged and "max" in existing:
                merged["max"] = min(merged["max"], existing["max"])
            self.set_param_hint(this_name, **merged)

lmfit.Model.set_param_hint_endswith = _set_param_hints_endswith_conservative
```

**Fix**: replace the body of `set_param_hints_endswith` in `lmfit_ext.py` with the same logic above — when a parameter already has a hint, take `max(existing_min, new_min)` and `min(existing_max, new_max)` before calling `set_param_hint`. Parameters with no prior hint are unaffected.
- Add a test to `test_model_function.py` (or a new `test_lmfit_ext.py`): set `g2_sigma min=2` on a `Const_2GaussModel_fast`, call `set_param_hint_endswith("sigma", min=0.77)`, and assert `g2_sigma min` is still `2`.

### 2.17 — Improve `_guess_*gauss` initial parameter estimates for non-flat continua

The current `_guess_1gauss`, `_guess_2gauss`, and `_guess_3gauss` functions estimate the
baseline constant `c` from the minimum value of the supplied spectrum.  When the true
continuum has a non-zero quadratic shape (e.g. a concave-down arch over the window), the
wing values are depressed below the flat-baseline level, so `c` is underestimated and the
Gaussian heights are overestimated.  More critically, `a` and `b` are never touched by
the guess — they remain at 0 — so the optimizer must discover all quadratic curvature
from a flat starting point.  For strong curvature this leads to poor or failed convergence
when starting from an auto-guess (`test_fit_recovers_quadratic_baseline` therefore uses
near-truth Gaussian params rather than `model.guess()`).

**Proposed fix**:
1. **Wing-based quadratic pre-fit**: before estimating the Gaussian parameters, mask out
   the central ±2σ region around the brightest peak and fit a degree-2 polynomial to the
   remaining "continuum" pixels using `numpy.polyfit`.  Use the polynomial coefficients as
   starting values for `a`, `b`, and `c`.  Subtract the polynomial from the spectrum
   before estimating peak height, center, and sigma.
2. **Sigma estimate for masking**: use a rough first-pass σ estimate (e.g. half the
   half-width at half-max of the smoothed spectrum) to define the mask width.  The mask
   need only be approximate — the goal is to exclude the obvious line core.
3. **Flag for disabling**: expose a `polynomial_baseline=True` keyword on each guess
   function (default `True` to improve behaviour; set `False` to restore the current
   flat-baseline behaviour for backwards compatibility and unit-test isolation).

**Testing**:
- Add `test_fit_from_auto_guess_quadratic_baseline` to each `TestQuadratic_*GaussModel`
  class: inject the same arch+tilt spectrum as `test_fit_recovers_quadratic_baseline`,
  call `model.guess(y, x=x)` for the initial params (no manual seeding of Gaussian
  params), fit with `method='least_squares'`, and assert `redchi < 1e-4` plus `a`/`b`
  within 10% of truth.
- Add a unit test for the wing-masking logic in isolation (given a known polynomial +
  Gaussian, verify the pre-fit coefficients are within 5% of truth).

**Scope**: `src/threadcount/models/models.py` (`_guess_1gauss`, `_guess_2gauss`,
`_guess_3gauss`) and the corresponding fast-model functions in `models/fast_models.py`
(`_guess_1gauss_d`, `_guess_2gauss_d`, `_guess_3gauss_d`).  The `Const_*GaussModel`
family also uses these guess functions and will benefit automatically; the improvement is
not limited to `Quadratic_*` models.

### 2.18 — Fix `get_param_values` branch 3 unreachable for `ModelResult` attributes

The docstring for `get_param_values` in `fit.py` states:

> If type('params') is `ModelResult`: Tries second: `params`.get(`param_name`), which allows for ModelResult attributes.

In practice, the `ModelResult` class in this lmfit fork does **not** implement `.get()`, so `params.get(param_name, default_value)` immediately raises `AttributeError` and the `except AttributeError` guard returns `default_value`.  Branch 3 is therefore completely unreachable for `ModelResult` inputs — callers who rely on it to extract attributes like `redchi` or `chisqr` silently receive `nan` instead.

**Fix**: replace the branch-3 `try` block with an explicit `getattr` call:
```python
try:
    return getattr(params, param_name, default_value)
except Exception:
    return default_value
```
This correctly handles both `ModelResult` attributes and any other objects that may not have the requested attribute.

**Test**: `tests/test_param_extraction.py::TestGetParamValuesModelResultAttribute` is marked `xfail(strict=True)` and will automatically turn green once this fix is applied.

### 2.19 — Fix two `ResultDict.loadtxt` crashes on edge-case files

Two separate code-path bugs exist in `loadtxt`, both discovered during the §1.6 test-coverage audit.  Neither can be triggered through normal multi-spaxel usage, which is why they went undetected.

**Bug A — 1×1 spatial grid (single-row file)**
`np.loadtxt` returns a 1D array when the file has exactly one data row. `loadtxt` then does `data[:, index]` which raises `IndexError: too many indices for array`.

*Fix*: promote a 1D result to 2D immediately after the `np.loadtxt` call:
```python
data = np.loadtxt(fname, **loadtxt_kwargs)
if data.ndim == 1:
    data = data[np.newaxis, :]   # single row → shape (1, n_cols)
```

**Bug B — No-coordinate files (`generate_pixel_coordinates=False`)**
When no `row`/`col` columns are present, `indices = []`.  The code then unconditionally calls `np.lexsort(tuple([...]))` which evaluates to `np.lexsort(())` and raises `TypeError: need sequence of keys with len > 0`.  The `if len(indices) == 0` guard that would skip the reshape path comes *after* this crash and is therefore unreachable.

*Fix*: guard the `lexsort` and subsequent sort with `if indices:`:
```python
if indices:
    ordering = np.lexsort(tuple([data[:, index] for index in reversed(indices)]))
    data = data[ordering]
```

**Tests**: the following `xfail(strict=True)` tests will turn green once both fixes are applied:
- `tests/test_result_dict.py::TestResultDictLoadtxtErrors::test_single_spaxel_round_trip` (Bug A)
- `tests/test_result_dict.py::TestResultDictRoundTripNoCoordinates::test_no_coordinates_round_trip` (Bug B)
- `tests/test_result_dict.py::TestResultDictRoundTripNoCoordinates::test_no_coordinates_nan_round_trip` (Bug B)

---

### 2.20 — Fix NaN-SNR guard in `process_single_spectrum`

In `procedures/fit_line.py`, the guard that skips spaxels with a NaN SNR value reads:

```python
if (snr_image[idx] < snr_threshold) or (np.isnan(snr_image[idx]) is True):
```

`np.isnan()` returns `numpy.bool_`, which is **not** the Python `True` singleton, so the `is True` identity comparison always evaluates to `False`.  Spaxels whose SNR is `NaN` are therefore never skipped — they are passed straight to the fitter, which can produce a spurious fit or crash depending on the spectrum.

**Fix**: replace the identity check with a plain truthiness test:

```python
if (snr_image[idx] < snr_threshold) or np.isnan(snr_image[idx]):
```

**Tests**: `tests/test_smoke_fit_lines.py::TestSingleSpaxelFit::test_snr_nan_skips_spaxel` is marked `xfail(strict=True)` and will turn green once this fix is applied.

---

### 2.21 — Fix `get_region(rx=0)` returning an empty array instead of `[[0, 0]]`

Found during Phase 1.8A test implementation. In `fit.py`, `get_region` computes `rx2 = rx * rx` before the ellipse-inequality check:

```python
inside = (
    indicies[:, 0] ** 2 / ry2 + indicies[:, 1] ** 2 / rx2 <= 1
)
```

When `rx=0` (and therefore `ry=0` after the `ry = rx` default), both `rx2` and `ry2` are `0.0`. Dividing by zero yields `nan` for every pixel (including the origin, which would give `0/0`), and `nan <= 1` evaluates to `False` in numpy. All pixels are therefore excluded and an empty array is returned instead of `[[0, 0]]`.

**Fix**: add an early-return special case before the division:
```python
if rx == 0:
    return np.array([[0, 0]])
```

Apply this immediately after the `rx = abs(rx)` / `ry = abs(ry)` normalization lines so negative-zero inputs are also handled.

**Test**: `tests/test_fit_utilities.py::TestGetRegion::test_rx_zero_returns_only_origin` is marked `xfail(strict=True)` and will turn green once this fix is applied.

---

### 2.22 — Fix `choose_model_aic_single` silent fallthrough and NaN-AIC bias

Two bugs found during Phase 1.8B code review, both documented as `xfail(strict=True)` tests in `tests/test_aic.py`.

**Bug A — silent fallthrough for `len > 3`**

When `model_list` has more than 3 elements the function falls through all `if` branches and executes the comment-labelled "safest thing" `return 0+1`. The docstring has a matching `TODO: generalize to more than 3`. The silent return of `1` regardless of the AIC values is wrong and invisible to the caller.

**Fix**: replace the trailing `return 0 + 1` with:
```python
raise ValueError(
    f"choose_model_aic_single only supports up to 3 models, got {len(model_list)}."
)
```

Once this is in place the `TODO` comment and `# safest thing to return is 0 i guess?` comment can both be removed. The docstring parameter description (`"Right now, the length must be no longer than 3"`) should be updated to `"The length must be no longer than 3"`.

**Test**: `tests/test_aic.py::TestChooseModelAicSingle::test_more_than_three_models_raises` is `xfail(strict=True)` and will turn green once this fix is applied.

**Bug B — NaN AIC bias when only the simpler model fails**

When `aic[0]` is `NaN` (model 0 failed) and `aic[1]` is valid, `aic[1] - aic[0]` is `NaN`, and `NaN < d_aic` evaluates to `False` in numpy. The function therefore returns model 1 (the failed one) instead of model 2 (the only successful one). The same logic occurs in the 3-model branch whenever the model selected for comparison has a `NaN` AIC.

**Fix**: after computing `aic = vget_aic(model_list, error=np.nan)` and the all-`NaN` check, add a valid-count guard. For the 2-model case a minimal fix is:
```python
if len(model_list) == 2:
    valid = ~np.isnan(aic)
    if valid.sum() == 1:
        return int(np.where(valid)[0][0]) + 1
    # ... existing comparison ...
```
A more general approach would loop over the models, filter `NaN` entries first, and then apply the sequential comparison logic. Either way the fix must also be applied to the 3-model branches.

**Test**: `tests/test_aic.py::TestChooseModelAicSingle::test_only_simpler_model_fails_returns_complex` is `xfail(strict=True)` and will turn green once this fix is applied.

---

### 2.23 — Fix `RecursiveArray` crashes on empty data (`__init__` and `aslist`)

Found during Phase 1.8C code review.

**Two methods unconditionally access `self.data[0]`.**

`RecursiveArray.__init__`:
```python
def __init__(self, array=None):
    super().__init__(array)
    if isinstance(self.data[0], (list, np.ndarray)):   # ← IndexError if array is []
        self.data = [self.__class__(x) for x in self.data]
```

`RecursiveArray.aslist`:
```python
def aslist(self):
    if isinstance(self.data[0], self.__class__):       # ← IndexError if data is []
        return [x.aslist() for x in self.data]
    else:
        return self.data
```

When `array` is `[]` or `None` (both produce `self.data = []` via `UserList.__init__`), `self.data[0]` raises `IndexError`. The same crash cascades through `__getattr__` and `__call__`, which both construct new `RecursiveArray` instances from list comprehensions — an empty source array produces an empty comprehension result, which immediately fails again on construction. After fixing `__init__`, an empty `RecursiveArray` can exist, but `aslist()` will still crash without its own guard.

**Fix**: add an early-return guard in both methods:

```python
def __init__(self, array=None):
    super().__init__(array)
    if not self.data:
        return
    if isinstance(self.data[0], (list, np.ndarray)):
        self.data = [self.__class__(x) for x in self.data]

def aslist(self):
    if not self.data:
        return []
    if isinstance(self.data[0], self.__class__):
        return [x.aslist() for x in self.data]
    else:
        return self.data
```

**Tests**:
- `tests/test_stats_collection.py::TestRecursiveArray::test_empty_list_does_not_raise` is `xfail(strict=True)` — covers `__init__`.
- `tests/test_stats_collection.py::TestRecursiveArray::test_aslist_empty_list_does_not_raise` is `xfail(strict=True)` — covers `aslist()`; bypasses `__init__` by injecting `ra.data = []` directly so it tests the `aslist()` guard independently.

Both will turn green once this fix is applied.

### 2.24 — Fix `order_gauss` crash when `g{n}_height` parameters are absent

`lmfit_ext.order_gauss` reads heights via:
```python
heights = [self.get("g{}_height".format(i + 1)).value for i in range(ngauss)]
```
`Parameters.get(name)` returns `None` when the key is absent; calling `.value` on `None` raises `AttributeError: 'NoneType' object has no attribute 'value'`.

This crashes silently for any model that names peak strength as `amplitude` rather than `height` (e.g. the stock lmfit `GaussianModel`).  In the threadcount model suite every gaussian component carries an expression-constrained `g{n}_height` parameter, so production usage is unaffected — but any user who calls `order_gauss` on params from a custom or standard-lmfit model will hit this.

**Fix**: guard the `get` call and skip (or raise a user-friendly error) when `g{n}_height` is absent:
```python
h_param = self.get("g{}_height".format(i + 1))
if h_param is None:
    raise KeyError(
        "order_gauss requires a 'g{n}_height' parameter for each gaussian "
        "component; 'g{}_height' not found.".format(i + 1)
    )
heights.append(h_param.value)
```

Alternatively, if sorting by height is not needed, make center-only sorting the fallback when height params are absent.

**Tests**:
- `tests/test_lmfit_ext.py::TestOrderGauss::test_order_gauss_missing_height_param_does_not_crash` is `xfail(strict=True)` — currently raises `AttributeError`.

Will turn green (xpass → pass) once the guard is added.

### 2.25 — Fix `process_single_spectrum` NaN SNR gate

In `fit_line.process_single_spectrum`:
```python
if (snr_image[idx] < snr_threshold) or (np.isnan(snr_image[idx]) is True):
    return [None]
```
`snr_image[idx]` is a `numpy.float64` scalar. `np.isnan(numpy.float64(nan))` returns `numpy.bool_(True)`, not Python's `bool` singleton `True`. Therefore `numpy.bool_(True) is True` is `False` and the NaN gate never fires. By IEEE 754, `nan < threshold` also returns `False`. The net effect: NaN SNR pixels pass the gate unchanged and proceed to fitting.

**Fix**: replace the `is True` identity check with a truthy check:
```python
if (snr_image[idx] < snr_threshold) or np.isnan(snr_image[idx]):
    return [None]
```

**Tests**:
- `tests/test_procedures_fit_line.py::TestProcessSingleSpectrum::test_snr_nan_returns_none_list` is `xfail(strict=True)`.

Will turn green once the `is True` identity check is removed.

### 2.26 — Fix missing `None` guard on chop-bandwidth retry in `process_single_spectrum`

In `fit_line.process_single_spectrum`, a second `lmfit` call is made after chopping the spectrum by ±5 Å:
```python
cut_sp = sp.subspec(wave_range[0] + 5, wave_range[1] - 5)
spec_to_fit = cut_sp
f = spec_to_fit.lmfit(models[0], **s.lmfit_kwargs)
if f.success is False:          # ← BUG: no guard if f is None
    return [None]
```
The first `lmfit` call is guarded by `if f is None: return [None]`, but the retry is not.  If the chopped spectrum is entirely masked (e.g. all variance = 0 or the wave range collapses), `lmfit` returns `None` and the `f.success` access raises `AttributeError`.

**Fix**: add the same guard after the retry call:
```python
f = spec_to_fit.lmfit(models[0], **s.lmfit_kwargs)
if f is None:
    return [None]
if f.success is False:
    return [None]
```

**Tests**:
- `tests/test_procedures_fit_line.py::TestProcessSingleSpectrum::test_first_model_fail_chop_true_retry_returns_none_on_masked_spectrum` is `xfail(strict=True)`.

Will turn green once the guard is added.

---

## Phase 3 — Structural Refactoring (requires Phase 1 safety net)

*Internal restructuring. Public API unchanged. Run the full test suite after each item.*

### 3.1 — Split `fit.py` into focused sub-modules
At ~3,100 lines, `fit.py` is a monolith. Proposed split:

| New file | Responsibility |
|---|---|
| `fit/_io.py` | `open_fits_cube`, `save_fit_stats`, `save_choice_fit_stats`, `save_to_file`, `ResultDict` |
| `fit/_snr.py` | `get_SNR_map`, `get_SignalBW_idx` |
| `fit/_model_selection.py` | `choose_model_aic`, `choose_model_aic_single`, `get_aic`, `marginal_fits` |
| `fit/_extraction.py` | `extract_spaxel_info`, `extract_spaxel_info_mc`, `get_param_values`, `collect_stats` |
| `fit/_spatial.py` | `spatial_average`, `get_region`, `get_reg_image` |
| `fit/_plot.py` | `save_pdf_plots`, `plot_ModelResults_pixel`, `interactive_user_choice` |
| `fit/_utils.py` | `iter_spaxel`, `RecursiveArray`, `de_redshift`, `tweak_redshift` |

Re-export everything from `fit/__init__.py` so `from threadcount.fit import get_SNR_map` continues to work unchanged.

### 3.2 — Break up long functions
Priority targets (by length / complexity):
- `save_pdf_plots()` (~150 lines): extract `_build_page_layout()` and `_plot_single_spaxel()` helpers.
- `interactive_user_choice()` (~100 lines): extract `_draw_interactive_panel()` and `_handle_keypress()`.
- `extract_spaxel_info()`: the triple try/except chain in `get_param_values` should be replaced by a single, well-documented extraction helper with explicit fallback cases.

### 3.3 — Formalise the monkey-patching approach
Both `lmfit_ext.py` and `mpdaf_ext.py` call `extend_lmfit()` / equivalent at import time, silently mutating third-party classes.
- Keep the behaviour, but make it explicit and opt-out: expose `threadcount.lmfit_ext.extend_lmfit()` and `threadcount.mpdaf_ext.extend_mpdaf()` as public functions.
- Call them from `threadcount/__init__.py` so they still run automatically for users.
- Document clearly in the API reference that these extensions are applied.
- This also makes it possible to write tests that run *without* the patches applied.

### 3.4 — Extract constants to a `constants.py` module
Move `FLAM16`, `FLOAT_FMT`, `DEFAULT_FIT_INFO` (from `fit.py`) and the physics wavelength deltas (from `models/fast_models.py`) into `threadcount/constants.py`.
- Re-import them in the original locations with `from threadcount.constants import ...` so nothing breaks.
- Now users can also `from threadcount.constants import FLAM16` if they need it.
- While moving the scalar wavelength constants from `lines.py` (e.g. `OIII5007`, `Hb4861`, `Hgamma`), normalise their names to a consistent convention (e.g. all-caps `OIII_5007`, `H_BETA_4861`) to match the constants already named `FLAM16` etc.

### 3.5 — Deduplicate model `_guess` functions
`models/models.py` has `_guess_1gauss`, `_guess_2gauss`, `_guess_3gauss` that share large blocks of identical setup logic. Extract the shared preamble into `_base_guess(spectrum)` returning `(peak, center, sigma, baseline)`. Each specific function then only handles its unique logic.

---

## Phase 4 — Type Hints (can be done incrementally)

*Add type hints progressively. Use `mypy --ignore-missing-imports` in CI from Phase 0.4.*

### 4.1 — Annotate `lines.py` and `constants.py`
Start with the simplest, most stable modules. These have no external type dependencies.

### 4.2 — Annotate `lmfit_ext.py` and `mpdaf_ext.py`
These are small (~100–400 lines) and have clear inputs/outputs.

### 4.3 — Annotate settings and I/O functions
The `process_settings` functions and `ResultDict` are the most user-facing. Annotating them first gives the best IDE autocompletion benefit.

### 4.4 — Annotate model classes
Annotate the public `guess()`, `fit()`, and `eval()` signatures on all model classes.

### 4.5 — Annotate the rest of `fit.py`
The largest module — do it last, when the split from Phase 3.1 makes it manageable.

---

## Phase 5 — User-Facing API Simplification (optional, semver minor version bump)

*These are opt-in improvements. Old dict-based settings continue to work.*

### 5.1 — Replace `SimpleNamespace` settings with a `dataclass`
Currently, settings are a dict that gets converted to a `SimpleNamespace`. This gives no validation, no default documentation, and no autocompletion.
- Create a `FitSettings` dataclass in a new `threadcount/settings.py` with all fields typed and defaulted.
- `process_settings` / `process_settings_dict` continue to accept a plain dict and convert it to `FitSettings`.
- Users can now also construct settings directly: `FitSettings(output_filename="out", snr_lower_limit=3.0)` with full IDE support.

### 5.2 — Add validation and helpful error messages to settings
When users pass nonsensical settings (e.g., `snr_lower_limit=-1`, `n_process=0`, a `lines` list that doesn't match the `models` list length), raise a `ValueError` with a clear human-readable message at settings-processing time, rather than failing silently or crashing deep inside the fit loop.

### 5.3 — Simplify model specification
Currently users must write:
```python
"models": [[Const_1GaussModel(), Const_2GaussModel(), Const_3GaussModel()], ...]
```
Provide a helper:
```python
"models": [tc.models.gauss_sequence(n_max=3, baseline="constant"), ...]
```
The old list-of-objects form remains valid.

### 5.4 — Add a `verbose` / `progress` option
Replace the current all-or-nothing print output with an optional `tqdm` progress bar for the spaxel-fitting loop, and a `verbose=False` default that suppresses most informational output unless requested. Tie this into the logging infrastructure from Phase 2.1.

---

## Phase 6 — Robustness and Performance

### 6.1 — Reproducible Monte Carlo
Accept a `random_seed` parameter in the user settings (default `None` for non-reproducible, any int for reproducible) and pass it through to `mc_iter`. Document in the settings dataclass.

### 6.2 — Memory-efficient large-cube handling
The `iter_spaxel` loop loads the entire cube into memory. For large cubes (>10 GB), add an optional chunked iterator and document memory requirements.

### 6.3 — Replace `multiprocessing` with `joblib`
The current parallel path uses `multiprocessing` directly. Replace it with `joblib.Parallel` / `joblib.delayed`:
- **Cross-platform consistency**: `joblib` handles the `spawn` vs `fork` start-method difference between Windows and Linux transparently, so the same code works on both without `if __name__ == "__main__"` guards or platform-specific workarounds.
- **Better error propagation**: exceptions raised in worker processes are re-raised in the main process with the original traceback, unlike the current code where worker exceptions are silently swallowed.
- **Backend flexibility**: `joblib` supports `loky` (default, robust), `threading`, and `multiprocessing` backends. Switching backend for profiling or debugging requires only one keyword change.
- **Automatic `chunksize` heuristics**: joblib batches tasks adaptively, so small cubes don't pay full parallelisation overhead.
- Add `joblib` to `[project.dependencies]` in `pyproject.toml` and to the `[dev]` extras.
- Replace the `n_process` multiprocessing pool calls in `fit_lines.py` with `Parallel(n_jobs=n_process)(delayed(fit_spaxel)(...) for ...)`.
- The `n_process` user-facing setting name stays unchanged.

**Also parallelise the Monte Carlo loop**: the MC iterations are currently entirely sequential — two nested loops with no parallelism at all:
1. In `fit_line.py`, the outer `for index, chosen_model in np.ndenumerate(chosen_models)` loop calls `chosen_model.mc_iter()` for each spaxel one at a time.
2. Inside `lmfit_ext.mc_iter`, the inner `for mcd in mc_data` loop re-fits each noise-draw sequentially.

Both levels are embarrassingly parallel and can be parallelised with `joblib.Parallel`:
- The per-spaxel MC work (outer loop) is the higher-value target: dispatch each spaxel's `mc_iter` call as a separate job alongside the main fit jobs so the whole spaxel pipeline (fit + MC) runs in parallel across spaxels.
- The per-draw inner loop inside `mc_iter` can optionally use `Parallel` too, but only makes sense when `mc_n_iterations` is large (≥ ~50) and the number of free parameters is high; for typical values (10–30 iterations) the joblib overhead may dominate. Add a `parallel_mc=False` flag to `mc_iter` to keep it opt-in.
- Ensure the RNG fix from §6.1 (seeded per-spaxel, not module-level) is done first, since parallelising `mc_iter` will expose the seeding bug immediately.

---

## Suggested Order Summary

| Phase | Effort | Risk | Unlock |
|---|---|---|---|
| 0a — Safe tooling | Low | None | CI, linting baseline |
| 1 — Tests | Medium | None | Phase 0b and all refactoring |
| 0b — Dependency modernisation | Low/Medium | Medium (mitigated by tests) | Clean deps, numpy 2, PyPI |
| 2 — Non-breaking quality | Low/Medium | Very low | Cleaner base for Phase 3 |
| 3 — Structural refactoring | High | Low (with tests) | Maintainability |
| 4 — Type hints | Medium | None | IDE support, mypy |
| 5 — API simplification | Medium | Low (additive) | Better UX |
| 6 — Robustness/performance | Medium | Low | Production use on large cubes |

Strict ordering: **0a → 1 → 0b → 2**. Within Phase 2, items 2.1–2.7 are independent. Phases 3 and 4 can interleave. Phase 5 requires Phase 3 to be complete. Phase 6 is independent throughout.

> **numpy 2 note**: Do not drop the `numpy < 2` pin speculatively. Run the Phase 1 tests with numpy 2 installed and let the results decide. A passing suite means the pin can be removed in 0b.2; failures become a concrete to-do list.
