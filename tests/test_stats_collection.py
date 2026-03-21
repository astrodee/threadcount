"""Unit tests for stats-collection pipeline in threadcount.fit (Phase 1.8C).

Covers: get_model_keys, get_header_stats, collect_stats, RecursiveArray.
"""

import lmfit
import numpy as np
import pytest

import threadcount as tc
import threadcount.fit as tf

# ---------------------------------------------------------------------------
# Module-level real ModelResult (Const_1GaussModel on clean synthetic data)
# ---------------------------------------------------------------------------
# Const_1GaussModel parameters: c, g1_amplitude, g1_center, g1_fwhm,
#                                g1_flux, g1_height, g1_sigma  (7 total)

_X = np.linspace(4990.0, 5025.0, 120)
_Y = 2.0 + 50.0 * np.exp(-((_X - 5006.843) ** 2) / (2 * 1.0**2))

_MODEL = tc.models.Const_1GaussModel()
_RESULT = _MODEL.fit(_Y, _MODEL.guess(_Y, x=_X), x=_X, method="least_squares")

# Sorted keys with nothing ignored
_ALL_KEYS = sorted(_RESULT.params.keys())

# Keys remaining after ignoring "fwhm" and "height" suffixes
_KEYS_NO_FWHM_HEIGHT = sorted(
    k for k in _ALL_KEYS if not k.endswith("fwhm") and not k.endswith("height")
)


# ---------------------------------------------------------------------------
# get_model_keys
# ---------------------------------------------------------------------------


class TestGetModelKeys:
    """1.8C — get_model_keys: extract sorted parameter names."""

    def test_single_model_result_returns_sorted_keys(self):
        keys = tf.get_model_keys(_RESULT)
        assert isinstance(keys, list)
        assert keys == sorted(keys)
        assert set(keys) == set(_ALL_KEYS)

    def test_array_with_none_at_front_uses_first_non_none(self):
        """First non-None entry is used; None elements are skipped."""
        arr = np.array([None, None, _RESULT], dtype=object)
        keys = tf.get_model_keys(arr)
        assert set(keys) == set(_ALL_KEYS)

    def test_ignore_string_filters_fwhm_and_height(self):
        keys = tf.get_model_keys(_RESULT, ignore="fwhm height")
        assert all(not k.endswith("fwhm") for k in keys)
        assert all(not k.endswith("height") for k in keys)
        assert set(keys) == set(_KEYS_NO_FWHM_HEIGHT)

    def test_ignore_list_same_as_string(self):
        keys_str = tf.get_model_keys(_RESULT, ignore="fwhm height")
        keys_list = tf.get_model_keys(_RESULT, ignore=["fwhm", "height"])
        assert keys_str == keys_list

    def test_all_none_returns_empty_list(self):
        arr = np.array([None, None], dtype=object)
        assert tf.get_model_keys(arr) == []

    def test_none_scalar_returns_empty_list(self):
        assert tf.get_model_keys(None) == []

    def test_result_is_sorted(self):
        """Returned list must be sorted alphabetically regardless of param order."""
        keys = tf.get_model_keys(_RESULT)
        assert keys == sorted(keys)

    def test_plain_lmfit_model_returns_keys(self):
        """A bare lmfit.Model (not yet fitted) uses model.make_params() internally."""
        model = tc.models.Const_1GaussModel()
        keys = tf.get_model_keys(model)
        assert isinstance(keys, list)
        assert len(keys) > 0
        assert keys == sorted(keys)
        # Must include the same parameter names as a fitted result from the same model
        assert set(keys) == set(_ALL_KEYS)

    def test_2d_spatial_array_uses_first_non_none(self):
        """Real production arrays are (n_models, ny, nx); .flat must reach a result."""
        arr = np.full((2, 3, 4), None, dtype=object)
        arr[0, 1, 2] = _RESULT  # one non-None entry buried in the 2D grid
        keys = tf.get_model_keys(arr)
        assert set(keys) == set(_ALL_KEYS)


# ---------------------------------------------------------------------------
# get_header_stats
# ---------------------------------------------------------------------------


class TestGetHeaderStats:
    """1.8C — get_header_stats: build column-name header list."""

    def test_auto_fit_info_starts_with_defaults(self):
        header = tf.get_header_stats(model_keys=["g1_center"], fit_info="auto")
        for name in tf.DEFAULT_FIT_INFO:
            assert name in header
        # DEFAULT_FIT_INFO entries come first
        assert header[: len(tf.DEFAULT_FIT_INFO)] == tf.DEFAULT_FIT_INFO

    def test_none_fit_info_omits_defaults(self):
        header = tf.get_header_stats(model_keys=["g1_center"], fit_info=None)
        for name in tf.DEFAULT_FIT_INFO:
            assert name not in header
        assert "g1_center" in header
        assert "g1_center_err" in header

    def test_none_model_keys_no_crash(self):
        """model_keys=None must not raise; only fit_info columns are returned."""
        header = tf.get_header_stats(model_keys=None, fit_info="auto")
        assert header == list(tf.DEFAULT_FIT_INFO)

    def test_header_length_with_auto_and_keys(self):
        keys = ["g1_center", "g1_sigma"]
        header = tf.get_header_stats(model_keys=keys, fit_info="auto")
        assert len(header) == len(tf.DEFAULT_FIT_INFO) + 2 * len(keys)

    def test_header_length_with_none_fit_info(self):
        keys = ["g1_center"]
        header = tf.get_header_stats(model_keys=keys, fit_info=None)
        assert len(header) == 2 * len(keys)

    def test_key_err_pairs_follow_key(self):
        """Each key must be immediately followed by key_err."""
        keys = ["g1_center", "g1_sigma"]
        header = tf.get_header_stats(model_keys=keys, fit_info=None)
        assert header == ["g1_center", "g1_center_err", "g1_sigma", "g1_sigma_err"]

    def test_both_none_returns_empty(self):
        header = tf.get_header_stats(model_keys=None, fit_info=None)
        assert header == []

    def test_custom_fit_info_list(self):
        """A user-supplied fit_info list replaces DEFAULT_FIT_INFO exactly."""
        header = tf.get_header_stats(
            model_keys=["g1_center"], fit_info=["redchi", "success"]
        )
        assert header == ["redchi", "success", "g1_center", "g1_center_err"]

    def test_custom_fit_info_does_not_mutate_original(self):
        """Passing a custom list must not mutate the caller's list."""
        custom = ["redchi"]
        tf.get_header_stats(model_keys=["g1_center"], fit_info=custom)
        assert custom == ["redchi"]


# ---------------------------------------------------------------------------
# collect_stats
# ---------------------------------------------------------------------------


class TestCollectStats:
    """1.8C — collect_stats: extract data row from one ModelResult."""

    def test_valid_result_values_finite(self):
        keys = ["g1_center", "g1_sigma"]
        row = tf.collect_stats(_RESULT, model_keys=keys, fit_info=None)
        assert len(row) == 2 * len(keys)
        assert all(np.isfinite(v) for v in row)

    def test_valid_result_values_correct(self):
        """Value and stderr must match result.params directly."""
        keys = ["g1_center"]
        row = tf.collect_stats(_RESULT, model_keys=keys, fit_info=None)
        assert row[0] == pytest.approx(_RESULT.params["g1_center"].value)
        assert row[1] == pytest.approx(_RESULT.params["g1_center"].stderr)

    def test_auto_fit_info_included(self):
        """DEFAULT_FIT_INFO attributes come first in the row."""
        keys = ["g1_center"]
        row = tf.collect_stats(_RESULT, model_keys=keys, fit_info="auto")
        assert len(row) == len(tf.DEFAULT_FIT_INFO) + 2 * len(keys)
        # aic_real is first DEFAULT_FIT_INFO entry
        assert row[0] == pytest.approx(_RESULT.aic_real)

    def test_missing_key_returns_empty_value(self):
        """A key not present in result.params yields two empty_value entries."""
        row = tf.collect_stats(
            _RESULT, model_keys=["nonexistent_param"], fit_info=None, empty_value=np.nan
        )
        assert len(row) == 2
        assert np.isnan(row[0])
        assert np.isnan(row[1])

    def test_none_result_returns_empty_value_list(self):
        keys = ["g1_center", "g1_sigma"]
        row = tf.collect_stats(
            None, model_keys=keys, fit_info="auto", empty_value=np.nan
        )
        expected_len = len(tf.DEFAULT_FIT_INFO) + 2 * len(keys)
        assert len(row) == expected_len
        assert all(np.isnan(v) for v in row)

    def test_none_result_custom_empty_value(self):
        row = tf.collect_stats(
            None, model_keys=["g1_center"], fit_info=None, empty_value=-999.0
        )
        assert row == [-999.0, -999.0]

    def test_header_and_stats_same_length(self):
        """get_header_stats and collect_stats must always return matching lengths."""
        for model_keys in [None, [], ["g1_center"], ["g1_center", "g1_sigma"]]:
            for fit_info in ["auto", None]:
                header = tf.get_header_stats(model_keys=model_keys, fit_info=fit_info)
                row = tf.collect_stats(
                    _RESULT, model_keys=model_keys, fit_info=fit_info
                )
                assert len(header) == len(row), (
                    f"Length mismatch for model_keys={model_keys}, fit_info={fit_info}: "
                    f"header={len(header)}, row={len(row)}"
                )

    def test_header_and_stats_same_length_for_none_result(self):
        """Length consistency must hold even when model_result=None."""
        keys = ["g1_center", "g1_sigma"]
        header = tf.get_header_stats(model_keys=keys, fit_info="auto")
        row = tf.collect_stats(None, model_keys=keys, fit_info="auto")
        assert len(header) == len(row)

    def test_custom_fit_info_list(self):
        """A custom fit_info list is used as-is; values match the named attributes."""
        row = tf.collect_stats(_RESULT, model_keys=None, fit_info=["redchi", "success"])
        assert len(row) == 2
        assert row[0] == pytest.approx(_RESULT.redchi)
        assert row[1] == _RESULT.success

    def test_invalid_fit_info_attr_silently_returns_all_empty(self):
        """Bug: if any fit_info attribute raises AttributeError, ALL collected
        values are discarded and the full row is filled with empty_value.
        Previously-gathered values (e.g. aic_real) are silently lost."""
        # fit_info has two entries; the second one does not exist on ModelResult.
        # The first entry (aic_real) is gathered successfully, then AttributeError
        # fires on the second.  The except block returns all-empty_value.
        row = tf.collect_stats(
            _RESULT,
            model_keys=["g1_center"],
            fit_info=["aic_real", "nonexistent_attr"],
            empty_value=np.nan,
        )
        expected_len = 2 + 2  # 2 fit_info + 2 for g1_center pair
        assert len(row) == expected_len
        # All values are empty_value — the successfully-collected aic_real is gone
        assert all(np.isnan(v) for v in row)


# ---------------------------------------------------------------------------
# RecursiveArray
# ---------------------------------------------------------------------------


class _Obj:
    """Minimal helper with a known attribute and callable methods."""

    def __init__(self, val):
        self.val = val

    def double(self):
        return self.val * 2

    def multiply(self, factor):
        return self.val * factor


class TestRecursiveArray:
    """1.8C — RecursiveArray: distributed attribute access and calls."""

    def test_attr_access_distributes(self):
        ra = tf.RecursiveArray([_Obj(1), _Obj(2), _Obj(3)])
        result = ra.val
        assert isinstance(result, tf.RecursiveArray)
        assert list(result) == [1, 2, 3]

    def test_call_distributes(self):
        ra = tf.RecursiveArray([_Obj(2), _Obj(3)])
        doubled = ra.double()
        assert isinstance(doubled, tf.RecursiveArray)
        assert list(doubled) == [4, 6]

    def test_none_elements_pass_through_call(self):
        """None elements must not raise; they are preserved as None."""
        ra = tf.RecursiveArray([_Obj(5), None])
        doubled = ra.double()
        assert doubled[0] == 10
        assert doubled[1] is None

    def test_nested_wraps_inner_lists(self):
        """2D construction: inner lists are wrapped as RecursiveArray instances."""
        ra = tf.RecursiveArray([[_Obj(1), _Obj(2)], [_Obj(3), _Obj(4)]])
        assert isinstance(ra[0], tf.RecursiveArray)
        assert isinstance(ra[1], tf.RecursiveArray)

    def test_aslist_flat_returns_plain_list(self):
        ra = tf.RecursiveArray([_Obj(1), _Obj(2)])
        plain = ra.val.aslist()
        assert type(plain) is list
        assert plain == [1, 2]

    def test_aslist_nested_returns_list_of_lists(self):
        ra = tf.RecursiveArray([[_Obj(1), _Obj(2)], [_Obj(3), _Obj(4)]])
        plain = ra.val.aslist()
        # val distributes over the nested RecursiveArrays → list of lists
        assert type(plain) is list
        assert type(plain[0]) is list

    def test_array_converts_to_ndarray(self):
        ra = tf.RecursiveArray([_Obj(1.5), _Obj(2.5), _Obj(3.5)])
        arr = ra.val.array()
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == float
        np.testing.assert_array_equal(arr, [1.5, 2.5, 3.5])

    def test_array_custom_dtype(self):
        ra = tf.RecursiveArray([_Obj(1), _Obj(0), _Obj(1)])
        arr = ra.val.array(dtype=bool)
        assert arr.dtype == bool
        assert list(arr) == [True, False, True]

    def test_missing_attr_returns_none_elements(self):
        """getattr with a missing name falls back to None for each element."""
        ra = tf.RecursiveArray([_Obj(1), _Obj(2)])
        result = ra.nonexistent_attr
        assert list(result) == [None, None]

    def test_call_with_positional_args(self):
        """*args/**kwargs must be forwarded to each element's call."""
        ra = tf.RecursiveArray([_Obj(2), _Obj(3)])
        result = ra.multiply(5)
        assert isinstance(result, tf.RecursiveArray)
        assert list(result) == [10, 15]

    @pytest.mark.xfail(
        reason=(
            "Bug: RecursiveArray.__init__ accesses self.data[0] unconditionally. "
            "An empty list raises IndexError: list index out of range."
        ),
        strict=True,
    )
    def test_empty_list_does_not_raise(self):
        """RecursiveArray([]) should produce an empty container, not IndexError."""
        ra = tf.RecursiveArray([])
        assert list(ra) == []

    @pytest.mark.xfail(
        reason=(
            "Bug: RecursiveArray.aslist() accesses self.data[0] unconditionally, "
            "same root cause as the __init__ bug (§2.23). Even after __init__ is "
            "fixed, aslist() will still raise IndexError on an empty instance."
        ),
        strict=True,
    )
    def test_aslist_empty_list_does_not_raise(self):
        """aslist() on an empty RecursiveArray should return [], not IndexError.

        This requires both the __init__ fix AND an aslist() guard:
            def aslist(self):
                if not self.data:
                    return []
                ...
        """
        # Bypass __init__ crash by injecting data directly
        ra = tf.RecursiveArray.__new__(tf.RecursiveArray)
        ra.data = []
        assert ra.aslist() == []
