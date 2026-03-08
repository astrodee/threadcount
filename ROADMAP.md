# Threadcount Improvement Roadmap

## Guiding Principles

- **Backwards compatibility first** — every phase preserves the existing user-facing API until a deliberate, versioned breaking change.
- **Tests before dependency changes** — the lmfit fork migration and pyproject.toml modernisation both carry real breakage risk. Build the test safety net first, then make those changes with confidence.
- **Small, reviewable chunks** — each item is a single PR/commit that can be merged and validated independently.

---

## Phase 0a — Safe Tooling (no risk, do first)

*These tasks change no runtime behaviour and can be done immediately.*

### 0a.1 — Add `ruff` and `pre-commit`
- Add a `[tool.ruff]` section in `pyproject.toml` for linting and formatting. Start permissive (disable rules you can't fix yet) and tighten gradually.
- Add a `.pre-commit-config.yaml` running `ruff --fix` and `ruff format` on every commit.
- Add `dev` extras to `[project.optional-dependencies]` (`pytest`, `pytest-cov`, `mypy`, `ruff`, `pre-commit`) so contributors can do `pip install -e ".[dev]"`.
- This stops the codebase from drifting further while you improve it.

### 0a.2 — Add a minimal CI pipeline
- A GitHub Actions workflow (`.github/workflows/ci.yml`) that runs `pytest` and `ruff` on every push/PR.
- Start with the current Python/numpy versions; expand after the dependency work below.

---

---

## Phase 1 — Test Infrastructure (must come before Phase 0b)

*The existing test suite is a single 35-line file with one test. Build coverage to ~50% here — this is the safety net that makes the dependency changes in Phase 0b safe to attempt.*

### 1.1 — Synthetic data fixtures
Create `tests/conftest.py` with `pytest` fixtures that build:
- A small synthetic FITS cube (e.g. 10×10 spatial, 200-wavelength) with known gaussian emission lines injected at known parameters.
- A pre-built `SimpleNamespace` settings object with all defaults filled in.

These become reusable inputs for every subsequent test.

### 1.2 — Tests for settings processing
Cover `fit.py` `process_settings` / `process_settings_dict`:
- Defaults are applied when a key is absent.
- User overrides replace defaults.
- Invalid types raise a clear error (currently they silently produce wrong behaviour).

### 1.3 — Tests for `Line` and `lines.py`
Cover `lines.py`:
- Constructing a `Line` with explicit values stores them correctly.
- Pre-defined constants (`L_OIII5007`, etc.) have the expected wavelength values.

### 1.4 — Tests for model functions
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

**A. Add parity tests for `Const_2GaussModel_fast` and `Const_3GaussModel_fast`**
- Extend the `TestConst2GaussModelFast` and `TestConst3GaussModelFast` classes with `test_parity_with_standard` methods matching the pattern already in `TestConst1GaussModelFast`.

**B. Tighten 4G and 6G `test_fit_from_auto_guess` assertions**
- `TestConst4GaussModelFast.test_fit_from_auto_guess`: add assertions on `deltax1`, `deltax2`, `deltax3` and at least `g4_height`; tighten redchi threshold to `1e-3`.
- `TestConst6GaussModelFast.test_fit_from_auto_guess`: add assertions on `deltax2`, `deltax3`, `deltax5`; tighten redchi threshold to `1e-3`.
- `TestConst4GaussModelConstrainedSIIFast.test_fit_from_auto_guess`: add assertions on `g2_height`, `deltax12`; tighten redchi threshold to `1e-3`.
- `TestConst6GaussModelConstrainedHaNIIFast.test_fit_from_auto_guess`: add assertions on `g2_height`, `g6_height`; tighten redchi threshold to `1e-3`.

**C. Check `g{1,3,5}_h_factor` in `Const_6GaussModel_constrained_HaNII_fast` fit-recovery test**
- Extend `TestConst6GaussModelConstrainedHaNIIFast.test_fit_recovers_parameters` to assert `g1_h_factor`, `g3_h_factor`, `g5_h_factor` within 5 %.

**D. Check all deltax and height values in 4G/6G `test_fit_recovers_parameters`**
- `TestConst4GaussModelFast`: add assertions for `g1_height`, `g2_height`, `g3_height`.
- `TestConst6GaussModelFast`: add assertions for `deltax2`, `deltax3`, `deltax5` and `g1_height` through `g6_height`.

**E. Add a `Quadratic_*` fit test with a genuine non-zero continuum background**
- Add `test_fit_recovers_quadratic_baseline` to `TestQuadratic1GaussModel` (and 2G/3G variants): inject a spectrum with `a=1e-4, b=-1.0` on top of the Gaussian flux, start the fit with `a=b=0`, and assert the recovered `a` and `b` are within 10 % of truth.

**F. Add direct tests for `_guess_multiline2` and `_guess_multiline3`**
- Add a `TestGuessMultiline2` class: call `model.guess(y, x=x)` after patching the bound method, verify the returned parameters are finite and the center offsets follow the formula.
- Add a `TestGuessMultiline3` class similarly.
- For the fast variants add `TestGuessMultiline2D`, `TestGuessMultiline3D`, `TestGuessMultiline4D`, `TestGuessMultiline6D` in the same style.

**G. Add `GaussianModelH` prefix test**
- Add `test_prefix_propagation` to `TestGaussianModelH`: construct `GaussianModelH(prefix="ha_")` and verify that all parameter names are prefixed and `eval` returns the same values as the un-prefixed model.

**H. Extend `set_common_limits` tests to multi-component models**
- Add `TestSetCommonLimits` fixture variants for `Const_2GaussModel` and `Const_3GaussModel`, verifying that all `g{n}_height`, `g{n}_sigma`, and `g{n}_center` parameters receive appropriate bounds.

---

### 1.5 — Tests for parameter extraction utilities
Cover `fit.py` `get_param_values` and `lmfit_ext.py` `summary_array`:
- Given a known `ModelResult`, the extraction returns the correct values.
- Tests for the "try three different extraction methods" fallback chain — each branch should be individually testable.

### 1.6 — Tests for `ResultDict`
Cover `fit.py` `ResultDict`:
- `savetxt` / `loadtxt` round-trip produces identical data.
- Works with NaN-containing arrays (the normal case for masked cubes).

### 1.7 — Integration smoke test for `fit_lines`
Using the synthetic cube fixture from 1.1, run a minimal `fit_lines.run()` end-to-end. Assert:
- Output `.txt` files are created.
- At least one spaxel was fitted without error.
- Loading the results back via `ResultDict.loadtxt` succeeds.

---

## Phase 0b — Dependency Modernisation (requires Phase 1 safety net)

*These tasks carry real risk of breaking behaviour. The Phase 1 tests are your safety net — run the full suite after each step.*

### 0b.1 — Resolve the custom `lmfit` fork dependency
The `pyproject.toml` pins `lmfit` to `sebusch/light-lmfit-py@light_dev` — a private fork installed from GitHub. This is fragile (if the branch disappears, the package breaks for all users) and blocks publication to PyPI.
- **Task**: Audit what differs in the fork vs upstream `lmfit`. Either upstream the changes, vendor the delta as monkey-patches (already partially done in `lmfit_ext.py`), or pin a specific commit SHA as a fallback.
- Run the full test suite after switching to confirm no regressions.

### 0b.2 — Modernise `pyproject.toml`
- Python `>= 3.6` is EOL. Raise the floor to `>= 3.10` (f-strings, `match`, `dataclasses`, `typing` improvements become available without backports).
- Run the test suite against **numpy 2** (install it in a fresh env). If all tests pass, drop the `numpy < 2` upper-bound pin entirely. If they don't, the failures pinpoint exactly what needs fixing before the pin can be removed.
- Add a lower bound `numpy >= 1.23` regardless, since the current constraint is one-sided and underspecified.
- Expand the CI matrix (from 0a.2) to cover the newly-supported Python and numpy versions.

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
