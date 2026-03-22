"""Unit tests for analyze_outflow_extent helper functions (Phase 1.9B).

Scope: distance, sort_data, boxcar_average_1d, row_max, compute_gal_center_row,
       radius_at_fraction, calculate_contours, create_outflow_mask,
       extract_wcs, process_arcsecs, process_units, contours_to_arcsec.
All functions are pure numpy — no mpdaf or matplotlib dependency.

Source: threadcount.procedures.analyze_outflow_extent
"""

import astropy.units as u
import numpy as np
import numpy.ma as ma
import pytest

from threadcount.procedures.analyze_outflow_extent import (
    boxcar_average_1d,
    calculate_contours,
    compute_gal_center_row,
    contours_to_arcsec,
    create_outflow_mask,
    distance,
    extract_wcs,
    process_arcsecs,
    process_units,
    radius_at_fraction,
    row_max,
    sort_data,
)

# ---------------------------------------------------------------------------
# distance
# ---------------------------------------------------------------------------


class TestDistance:
    """distance(row, col, origin) — Pythagorean distance."""

    def test_origin_to_origin_is_zero(self):
        """distance from origin to itself is 0."""
        origin = [3.0, 7.0]
        result = distance(origin[0], origin[1], origin)
        assert result == 0.0

    def test_pythagorean_triple(self):
        """3-4-5 triple: distance from (3, 4) to (0, 0) is 5."""
        result = distance(3.0, 4.0, [0.0, 0.0])
        assert result == pytest.approx(5.0)

    def test_elementwise_on_arrays(self):
        """Works element-wise when row/col are numpy arrays."""
        rows = np.array([0.0, 3.0, 0.0])
        cols = np.array([0.0, 4.0, 5.0])
        origin = [0.0, 0.0]
        result = distance(rows, cols, origin)
        expected = np.array([0.0, 5.0, 5.0])
        np.testing.assert_allclose(result, expected)

    def test_non_integer_coordinates(self):
        """Floating-point coordinates give exact Pythagorean result."""
        # 0.6-0.8-1.0 triple
        result = distance(0.6, 0.8, [0.0, 0.0])
        assert result == pytest.approx(1.0)

    def test_non_origin_reference_point(self):
        """Distance is measured from the given origin, not from (0,0)."""
        # vector (3, 4) from origin (1, 1) → row=4, col=5
        result = distance(4.0, 5.0, [1.0, 1.0])
        assert result == pytest.approx(5.0)

    def test_masked_array_inputs_propagate_mask(self):
        """Masked input arrays produce masked output (production usage pattern).

        In run(), distance is called with np.ma.masked_where(..., grid[0])
        as row/col arguments.  The mask must propagate to the output so that
        downstream sort_data can compress the masked entries.
        """
        rows = ma.array([0.0, 3.0, 0.0], mask=[False, False, True])
        cols = ma.array([0.0, 4.0, 5.0], mask=[False, False, True])
        origin = [0.0, 0.0]
        result = distance(rows, cols, origin)
        assert isinstance(result, ma.MaskedArray)
        assert not result.mask[0]  # unmasked entry 0→0,0: distance 0
        assert not result.mask[1]  # unmasked entry 3,4: distance 5
        assert result.mask[2]  # masked entry propagated
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# sort_data
# ---------------------------------------------------------------------------


class TestSortData:
    """sort_data(x, data) — compress masked arrays and sort by x."""

    def test_output_x_is_monotonically_increasing(self):
        """After sort_data, x values are in ascending order."""
        x = ma.array([3.0, 1.0, 4.0, 1.5])
        data = ma.array([10.0, 20.0, 30.0, 40.0])
        x_out, _ = sort_data(x, data)
        assert np.all(np.diff(x_out) >= 0)

    def test_output_is_plain_ndarray(self):
        """sort_data returns plain ndarrays, not masked arrays."""
        x = ma.array([2.0, 1.0])
        data = ma.array([5.0, 6.0])
        x_out, data_out = sort_data(x, data)
        assert isinstance(x_out, np.ndarray)
        assert not isinstance(x_out, ma.MaskedArray)
        assert isinstance(data_out, np.ndarray)
        assert not isinstance(data_out, ma.MaskedArray)

    def test_masked_entries_removed(self):
        """Masked entries in x/data are removed (compressed) from output."""
        x = ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
        data = ma.array([10.0, 99.0, 30.0], mask=[False, True, False])
        x_out, data_out = sort_data(x, data)
        assert len(x_out) == 2
        assert 2.0 not in x_out
        assert 99.0 not in data_out

    def test_sort_preserves_x_data_correspondence(self):
        """After sorting, each x value is paired with its original data value."""
        x = ma.array([3.0, 1.0, 2.0])
        data = ma.array([30.0, 10.0, 20.0])
        x_out, data_out = sort_data(x, data)
        # x=1.0 → data=10.0, x=2.0 → data=20.0, x=3.0 → data=30.0
        np.testing.assert_array_equal(x_out, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(data_out, [10.0, 20.0, 30.0])

    def test_all_masked_returns_empty_arrays(self):
        """All entries masked → both returned arrays are empty."""
        x = ma.array([1.0, 2.0], mask=[True, True])
        data = ma.array([10.0, 20.0], mask=[True, True])
        x_out, data_out = sort_data(x, data)
        assert len(x_out) == 0
        assert len(data_out) == 0

    @pytest.mark.xfail(
        reason=(
            "Bug §2.27: sort_data assumes x and data share the same mask "
            "(documented by '# ## ASSUMES SAME MASK' comment). When masks differ, "
            "np.ma.compressed(x) and np.ma.compressed(data) return different-length "
            "arrays. The argsort indices are computed on the shorter x array and "
            "applied to data, silently truncating or misaligning values instead of "
            "raising. Fix: add an assertion or normalise to a common mask before "
            "compressing."
        ),
        strict=True,
    )
    def test_mismatched_masks_raises_not_silent_corruption(self):
        """sort_data with different masks on x and data should raise, not silently corrupt.

        Currently: x is compressed to 2 elements, data to 3; sort_increasing
        has indices [0, 1] which silently slices data to its first 2 elements —
        the correspondence is broken without any error.
        """
        x = ma.array([3.0, 1.0, 2.0], mask=[False, False, True])  # 2 unmasked
        data = ma.array([30.0, 10.0, 20.0], mask=[False, False, False])  # 3 unmasked
        # With same-length compressed arrays this would work; with mismatched lengths
        # it should raise ValueError (or equivalent) to alert the caller.
        x_out, data_out = sort_data(x, data)
        # We reach here only if no exception was raised — force the test to fail
        # to mark this as the buggy (silent) path.
        raise AssertionError("Expected sort_data to raise on mismatched masks")


# ---------------------------------------------------------------------------
# boxcar_average_1d
# ---------------------------------------------------------------------------


class TestBoxcarAverage1d:
    """boxcar_average_1d(input_array, width, axis=0) — uniform convolution."""

    def test_width_one_is_identity(self):
        """width=1 → output equals input exactly."""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = boxcar_average_1d(arr, width=1)
        np.testing.assert_array_equal(result, arr)

    def test_width_three_constant_interior_unchanged(self):
        """width=3 on a constant array → interior rows equal input.

        Edge rows are zero-padded by np.convolve(mode='same'), so only the
        interior (rows 1:-1) are expected to equal the constant value.
        """
        arr = np.full((5, 4), 7.0)
        result = boxcar_average_1d(arr, width=3)
        np.testing.assert_allclose(result[1:-1, :], arr[1:-1, :])

    def test_width_three_step_function_interior(self):
        """width=3 on a 1D step function: interior transition averaged correctly.

        Step [0, 0, 1, 1, 1] with width=3 (convolution mode 'same'):
           index 2: avg([0, 1, 1]) = 2/3
           index 3: avg([1, 1, 1]) = 1
        The interior point at index 2 is the average of its two neighbours and itself.
        """
        arr = np.array([[0.0, 0.0, 1.0, 1.0, 1.0]])  # shape (1, 5) so axis=0 is trivial
        # Use axis=1 to apply along the 5-element dimension
        result = boxcar_average_1d(arr, width=3, axis=1)
        # index 2 is the step point: [0, 1, 1] average = 2/3
        assert result[0, 2] == pytest.approx(2.0 / 3.0)
        # index 3: all ones
        assert result[0, 3] == pytest.approx(1.0)

    def test_axis_one_averages_along_columns(self):
        """axis=1 applies smoothing along columns.

        A single-row array with a spike at column 2 and zeros elsewhere:
        after width-3 smoothing along axis=1, the spike centre becomes
        (0 + spike + 0) / 3 = spike/3.
        """
        spike = 9.0
        arr = np.array([[0.0, 0.0, spike, 0.0, 0.0]])
        result = boxcar_average_1d(arr, width=3, axis=1)
        assert result[0, 2] == pytest.approx(spike / 3.0)

    def test_axis_zero_versus_axis_one_differ_on_asymmetric_array(self):
        """axis=0 and axis=1 produce different results on a non-symmetric array."""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result_axis0 = boxcar_average_1d(arr, width=3, axis=0)
        result_axis1 = boxcar_average_1d(arr, width=3, axis=1)
        # They should differ: axis=0 averages rows, axis=1 averages columns
        assert not np.allclose(result_axis0, result_axis1)


# ---------------------------------------------------------------------------
# row_max
# ---------------------------------------------------------------------------


def _make_flux_image(n_rows, n_cols, peak_col, center_row):
    """Return a float64 array where every row peaks at peak_col."""
    arr = np.zeros((n_rows, n_cols))
    arr[:, peak_col] = 10.0
    return np.ma.array(arr, mask=np.zeros((n_rows, n_cols), dtype=bool))


class TestRowMax:
    """row_max(flux_masked_array, clip, center_row) — sigma-clip then median-smooth."""

    def test_returns_two_arrays(self):
        """Returns a tuple of two objects."""
        img = _make_flux_image(6, 10, peak_col=5, center_row=3)
        result = row_max(img, center_row=3)
        assert len(result) == 2

    def test_first_array_is_arange(self):
        """First returned array is np.arange(0, n_rows)."""
        n_rows = 8
        img = _make_flux_image(n_rows, 12, peak_col=6, center_row=4)
        rows, _ = row_max(img, center_row=4)
        np.testing.assert_array_equal(rows, np.arange(0, n_rows))

    def test_second_array_length_equals_n_rows(self):
        """Second returned array has the same length as the input row count."""
        n_rows = 7
        img = _make_flux_image(n_rows, 10, peak_col=5, center_row=3)
        rows, cols = row_max(img, center_row=3)
        assert len(cols) == n_rows

    def test_outlier_row_is_masked(self):
        """A row whose peak is a clear outlier (far from other rows) is masked.

        9 rows peak at column 5; 1 row peaks at column 20 in a 30-col image.
        With clip=1, the outlier row must be masked.
        """
        n_rows, n_cols = 10, 30
        arr = np.zeros((n_rows, n_cols))
        arr[:, 5] = 10.0  # most rows peak at col 5
        arr[4, 5] = 0.0
        arr[4, 20] = 10.0  # outlier: row 4 peaks at col 20
        img = np.ma.array(arr, mask=np.zeros_like(arr, dtype=bool))
        _, cols = row_max(img, clip=1, center_row=5)
        assert cols.mask[4]  # outlier row must be masked

    def test_above_center_replaced_by_median(self):
        """Rows above center_row are replaced by the median of that half."""
        n_rows, n_cols = 10, 20
        arr = np.zeros((n_rows, n_cols))
        arr[:, 8] = 10.0  # all rows peak at col 8
        img = np.ma.array(arr, mask=np.zeros_like(arr, dtype=bool))
        center_row = 5
        _, cols = row_max(img, center_row=center_row)
        # All rows in upper half (0:center_row) should have the same data value
        upper_data = cols.data[0:center_row]
        assert np.all(upper_data == upper_data[0])

    def test_below_center_replaced_by_median(self):
        """Rows below center_row are replaced by the median of that half."""
        n_rows, n_cols = 10, 20
        arr = np.zeros((n_rows, n_cols))
        arr[:, 8] = 10.0
        img = np.ma.array(arr, mask=np.zeros_like(arr, dtype=bool))
        center_row = 5
        _, cols = row_max(img, center_row=center_row)
        lower_data = cols.data[center_row:]
        assert np.all(lower_data == lower_data[0])

    @pytest.mark.xfail(
        reason=(
            "Bug §2.28: row_max filters with 'rowmax[rowmax > 0]' to exclude "
            "masked-row argmax artefacts (which default to 0), but this also "
            "excludes genuine peaks at column 0. When valid data peaks at column "
            "0, these rows are removed from the mean/std calculation used for "
            "sigma-clipping, biasing the outlier detection."
        ),
        strict=True,
    )
    def test_genuine_peak_at_column_zero_not_excluded_from_stats(self):
        """Rows whose genuine peak is at column 0 should not be excluded from stats.

        Currently 'rowmax[rowmax > 0]' silently drops them from the mean/std
        computation, biasing sigma-clipping when column-0 is a real peak column.
        """
        n_rows, n_cols = 6, 10
        arr = np.zeros((n_rows, n_cols))
        arr[:, 0] = 10.0  # all rows genuinely peak at column 0
        img = np.ma.array(arr, mask=np.zeros_like(arr, dtype=bool))
        _, cols = row_max(img, center_row=3)
        # None of the rows should be masked — they all peak at the same column
        assert not np.any(cols.mask)


# ---------------------------------------------------------------------------
# compute_gal_center_row
# ---------------------------------------------------------------------------


class TestComputeGalCenterRow:
    """compute_gal_center_row(image) — column-max argmax filtered by 1% flux threshold."""

    def test_bright_stripe_returns_correct_row(self):
        """A horizontal bright stripe at a known row → returns that row index."""
        n_rows, n_cols = 20, 15
        img = np.ones((n_rows, n_cols)) * 0.1
        bright_row = 7
        img[bright_row, :] = 100.0
        result = compute_gal_center_row(img)
        assert result == bright_row

    def test_low_flux_edge_columns_do_not_shift_result(self):
        """Columns whose max flux is ≤ 1% of global max are excluded from the median.

        With the bright stripe at row 7 but the edge columns having max flux
        at a different row (and low absolute flux), the result should still be 7.
        """
        n_rows, n_cols = 20, 15
        img = np.zeros((n_rows, n_cols))
        bright_row = 7
        img[bright_row, 1:-1] = 100.0  # bright stripe, excluding edge columns
        # Edge columns have a tiny peak at a different row — flux << 1% of max
        img[3, 0] = 0.5
        img[3, -1] = 0.5
        result = compute_gal_center_row(img)
        assert result == bright_row

    def test_returns_integer(self):
        """Return value is a Python int (cast via int(...))."""
        n_rows, n_cols = 10, 8
        img = np.zeros((n_rows, n_cols))
        img[4, :] = 5.0
        result = compute_gal_center_row(img)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# radius_at_fraction
# ---------------------------------------------------------------------------


class TestRadiusAtFraction:
    """radius_at_fraction(x, y, values, return_string) — cumulative-sum half-radius."""

    def _uniform(self, n=20):
        """Uniform y=1 array on x=[0..n-1]."""
        return np.arange(float(n)), np.ones(n)

    def test_fifty_percent_uniform_midpoint(self):
        """values=[50] on uniform array: radius = x at first index where cumsum > 50%.

        For 20 ones, 50% goal = 10.0. The cumulative sum exceeds 10 at index 10
        (total = 11 > 10, strict >), so the returned x value is x[10] = 10.0.
        """
        x, y = self._uniform(20)
        result = radius_at_fraction(x, y, [50])
        assert result[0, 1] == pytest.approx(10.0)

    def test_values_greater_than_one_treated_as_percent(self):
        """values > 1 are divided by 100 before use."""
        x, y = self._uniform(20)
        result_pct = radius_at_fraction(x, y, [50])
        result_frac = radius_at_fraction(x, y, [0.50])
        np.testing.assert_allclose(result_pct, result_frac)

    def test_return_string_true_returns_list_of_strings(self):
        """return_string=True → list of formatted strings."""
        x, y = self._uniform(20)
        result = radius_at_fraction(x, y, [50], return_string=True)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], str)
        # Format is "r_50 = ..."
        assert result[0].startswith("r_50")

    def test_return_string_multiple_values(self):
        """return_string=True with two values → two strings."""
        x, y = self._uniform(20)
        result = radius_at_fraction(x, y, [25, 75], return_string=True)
        assert len(result) == 2

    def test_return_array_shape(self):
        """Default return is a 2D array with shape (len(values), 2)."""
        x, y = self._uniform(20)
        result = radius_at_fraction(x, y, [25, 50, 75])
        assert result.shape == (3, 2)

    def test_first_column_contains_fractions(self):
        """First column of result contains the requested fractions (in [0,1])."""
        x, y = self._uniform(20)
        result = radius_at_fraction(x, y, [25, 75])
        np.testing.assert_allclose(result[:, 0], [0.25, 0.75])

    def test_scalar_value_does_not_raise(self):
        """Scalar values (not a list) is handled via the TypeError except path."""
        x, y = self._uniform(20)
        # Should not raise; internally wraps to np.array([values])
        result = radius_at_fraction(x, y, 50)
        assert result.shape == (1, 2)

    @pytest.mark.xfail(
        reason=(
            "Bug §2.29: radius_at_fraction collects results in a list and calls "
            "np.column_stack([values, x[results]]) at the end. If the cumulative "
            "sum of y never reaches one of the goals (e.g. a fraction > 1.0 after "
            "normalisation, or y values that sum to less than the goal), the loop "
            "ends before all goals are satisfied, results is shorter than values, "
            "and column_stack raises ValueError: 'all input arrays must have the "
            "same shape'. Fix: guard that len(results) == len(values) before "
            "column_stack, or break out of the outer loop cleanly."
        ),
        strict=True,
    )
    def test_unreachable_fraction_does_not_crash(self):
        """A fraction whose cumulative goal exceeds the total sum of y should not crash.

        values=[50, 200] → after /100 → [0.5, 2.0].  goals = [0.5*sum, 2.0*sum].
        The second goal can never be reached (it requires twice the total flux),
        so results ends up with only 1 entry, and column_stack raises ValueError.
        """
        x = np.arange(5.0)
        y = np.ones(5)
        # Should return gracefully (e.g. only the reachable fractions), not crash
        result = radius_at_fraction(x, y, [50, 200])
        assert result is not None


# ---------------------------------------------------------------------------
# calculate_contours
# ---------------------------------------------------------------------------


class TestCalculateContours:
    """calculate_contours(flux_masked_array, levels, clip_max, center_row).

    Internally calls row_max, then expands outward from the peak column of each
    row until the cumulative flux fraction goal is met, recording the half-width.
    Output shape: (2 + len(levels), n_rows).
    """

    # Flux: 5 rows × 11 cols.  Rows 0-3 spike at col 5; row 4 spikes at col 6.
    # The col-6 entry makes std(argmax) > 0 so row_max's σ-clip stays finite.
    # After row_max with center_row=3: upper-half median=5, lower-half
    # median=int(median([5,6]))=int(5.5)=5 → all rows get center_col=5.
    _rows, _cols = 5, 11
    _center_row = 3

    def _make_flux(self):
        flux = np.ma.zeros((self._rows, self._cols))
        flux[0:4, 5] = 10.0
        flux[4, 6] = 10.0
        return flux

    def test_output_shape_two_plus_nlevels_by_nrows(self):
        """Result shape is (2 + len(levels), n_rows)."""
        result = calculate_contours(
            self._make_flux(), levels=[0.5, 0.9], center_row=self._center_row
        )
        assert result.shape == (4, self._rows)

    def test_first_row_is_sequential_row_indices(self):
        """result[0] equals np.arange(n_rows)."""
        result = calculate_contours(
            self._make_flux(), levels=[0.5], center_row=self._center_row
        )
        np.testing.assert_array_equal(result[0], np.arange(self._rows))

    def test_half_widths_match_manual_cumsum_computation(self):
        """Single-spike rows: entire flux at the peak column → any fraction
        goal is met immediately (count=0), so half_width=0 for rows 0-3.

        Row 4 is intentionally different to keep std(argmax) > 0 for the
        sigma-clip inside row_max; only rows 0-3 are verified here.
        """
        flux = np.ma.zeros((self._rows, self._cols))
        flux[0:4, 5] = (
            10.0  # total=10 for rows 0-3; any fraction goal ≤ 10 met at count=0
        )
        flux[4, 6] = 10.0  # ensures std(argmax) > 0
        result = calculate_contours(
            flux, levels=[0.5, 0.9], center_row=self._center_row
        )
        for row_i in range(4):
            assert result[2, row_i] == 0, f"row {row_i}: 0.5 half-width should be 0"
            assert result[3, row_i] == 0, f"row {row_i}: 0.9 half-width should be 0"

    def test_levels_sorted_ascending_regardless_of_input_order(self):
        """Passing levels in reverse order produces the same output as sorted."""
        flux = self._make_flux()
        result_asc = calculate_contours(
            flux, levels=[0.5, 0.9], center_row=self._center_row
        )
        result_desc = calculate_contours(
            flux, levels=[0.9, 0.5], center_row=self._center_row
        )
        np.testing.assert_array_equal(result_asc, result_desc)

    def test_fully_masked_row_produces_masked_output_not_crash(self):
        """A row fully masked in the input produces masked entries in the output.

        numpy creates a masked array from a list containing np.ma.masked constants,
        preserving the mask through the .T and astype(int) steps in calculate_contours.
        """
        flux = self._make_flux()
        flux[2, :] = np.ma.masked  # fully mask row 2
        result = calculate_contours(flux, levels=[0.5], center_row=self._center_row)
        # np.ma.is_masked returns a plain bool; center_col and half-width for row 2 are masked.
        assert np.ma.is_masked(result[1:, 2])

    @pytest.mark.xfail(
        reason=(
            "§2.31: calculate_contours expands outward from max_col with "
            "'this_row[max_col - count] + this_row[max_col + count]'. When the "
            "peak is near the right edge of the array, max_col + count goes out of "
            "bounds and raises IndexError. Similarly, max_col - count < 0 silently "
            "wraps around to the far end of the array, using flux from the wrong "
            "side. Fix: clamp count to the valid expansion range before accessing, "
            "or use np.pad to add zero-boundary padding."
        ),
        strict=True,
    )
    def test_near_edge_peak_does_not_crash_or_wrap(self):
        """Peak near the right edge with spread flux should not raise IndexError.

        flux[0]: [1, 1, 3, 1, 4] → argmax=4 (right edge), total=10, 50% goal=5.
        The peak alone (4.0) < goal (5.0), so the expansion loop runs and tries
        to access this_row[4 + 1] = this_row[5] → IndexError (5-col array).
        flux[1]: isolated spike at col 2 → gives std(argmax) > 0 so sigma-clip
        inside row_max stays finite.
        """
        flux = np.ma.zeros((2, 5))
        flux[0, :] = [
            1.0,
            1.0,
            3.0,
            1.0,
            4.0,
        ]  # peak at right edge (col 4), spread flux
        flux[1, 2] = 5.0  # different peak → std(argmax) > 0
        result = calculate_contours(flux, levels=[0.5], center_row=1)
        assert result is not None


# ---------------------------------------------------------------------------
# create_outflow_mask
# ---------------------------------------------------------------------------


class TestCreateOutflowMask:
    """create_outflow_mask(contour_output, contour_levels, which_contour, output_shape).

    contour_output shape: (2 + n_levels, n_rows).  Starts all-True; sets
    centre ± half_width to False for the chosen contour level.
    """

    @staticmethod
    def _make_co(rows_cols_hws):
        """Build a (2+n_levels, n_rows) masked array from a list of per-row tuples."""
        return np.ma.array(np.array(rows_cols_hws).T)

    def test_pixels_inside_contour_are_false(self):
        """Columns within center ± half_width are False (inside the outflow region)."""
        # 1 row: row=0, center_col=5, half_width=2 → cols 3,4,5,6,7 set to False
        co = self._make_co([(0, 5, 2)])
        result = create_outflow_mask(co, [0.5], 0.5, (1, 11))
        assert not result[0, 3]  # left edge
        assert not result[0, 5]  # center
        assert not result[0, 7]  # right edge

    def test_pixels_outside_contour_are_true(self):
        """Columns outside center ± half_width remain True."""
        co = self._make_co([(0, 5, 2)])
        result = create_outflow_mask(co, [0.5], 0.5, (1, 11))
        assert result[0, 2]  # just outside left
        assert result[0, 8]  # just outside right

    def test_which_contour_selects_correct_level(self):
        """which_contour=0.5 uses the smaller half-width; 0.9 uses the larger."""
        # contour_output columns: [row, center_col, hw_50%, hw_90%]
        co = self._make_co([(0, 5, 1, 3)])
        # 50%: center±1 → cols 4,5,6 False.  90%: center±3 → cols 2..8 False.
        result_50 = create_outflow_mask(co, [0.5, 0.9], 0.5, (1, 12))
        result_90 = create_outflow_mask(co, [0.5, 0.9], 0.9, (1, 12))
        assert not result_50[0, 5]  # inside 50%
        assert result_50[0, 3]  # outside 50% (but inside 90%)
        assert not result_90[0, 3]  # inside 90%
        assert result_90[0, 1]  # outside 90%

    def test_masked_center_col_row_is_left_all_true(self):
        """A row with a masked center_col is skipped; all its pixels stay True."""
        # shape (3, 2): row_indices=[0,1], center_cols=[5,masked], half_widths=[2,masked]
        co = np.ma.array(
            [[0, 1], [5, 5], [2, 2]],
            mask=[[False, False], [False, True], [False, True]],
        )
        result = create_outflow_mask(co, [0.5], 0.5, (2, 11))
        assert not result[0, 5]  # row 0: inside contour
        assert result[1, 5]  # row 1: masked center → left True
        assert result[1, 0]  # row 1: also True at far column

    def test_unknown_which_contour_raises_not_silent(self):
        """which_contour not in contour_levels should raise ValueError, not UnboundLocalError."""
        co = self._make_co([(0, 5, 2)])
        with pytest.raises((ValueError, KeyError)):
            create_outflow_mask(co, [0.5], 0.7, (1, 11))


# ---------------------------------------------------------------------------
# extract_wcs
# ---------------------------------------------------------------------------


class TestExtractWcs:
    """extract_wcs(comment_lines) — parse the wcs_step pixel scale from header comments.

    The comment format written by the pipeline is 'wcs_step: [dy dx]' (square
    brackets, space-separated values).  extract_wcs strips the key, replaces
    spaces with commas, and eval()s the result.
    """

    def test_extracts_list_from_wcs_step_line(self):
        """'wcs_step: [0.2 0.2]' returns [0.2, 0.2]."""
        result = extract_wcs(["wcs_step: [0.2 0.2]"])
        assert result == pytest.approx([0.2, 0.2])

    def test_ignores_non_matching_lines(self):
        """Lines not starting with 'wcs_step:' are silently skipped."""
        lines = ["units: erg", "wcs_step: [0.1 0.3]", "other: foo"]
        result = extract_wcs(lines)
        assert result == pytest.approx([0.1, 0.3])

    def test_missing_wcs_step_raises_index_error(self):
        """No matching line → IndexError from [0] on empty list."""
        with pytest.raises(IndexError):
            extract_wcs(["units: erg", "other: foo"])


# ---------------------------------------------------------------------------
# process_arcsecs
# ---------------------------------------------------------------------------


class TestProcessArcsecs:
    """process_arcsecs(input_arcsecs, comment_lines) — resolve pixel scale to a float."""

    _WCS_LINES = ["wcs_step: [0.2 0.2]"]

    def test_numeric_float_input_passed_through_unchanged(self):
        """A numeric float is returned as-is without consulting comment_lines."""
        assert process_arcsecs(0.5, []) == pytest.approx(0.5)

    def test_numeric_int_input_passed_through_unchanged(self):
        """A numeric int is also returned unchanged."""
        assert process_arcsecs(1, []) == 1

    def test_header_string_triggers_extract_wcs(self):
        """'header' reads wcs_step from comment_lines and returns only the first element."""
        result = process_arcsecs("header", self._WCS_LINES)
        assert result == pytest.approx(0.2)

    def test_auto_string_behaves_same_as_header(self):
        """'auto' is treated identically to 'header'."""
        result = process_arcsecs("auto", self._WCS_LINES)
        assert result == pytest.approx(0.2)

    def test_none_triggers_extract_wcs(self):
        """None behaves the same as 'header'."""
        result = process_arcsecs(None, self._WCS_LINES)
        assert result == pytest.approx(0.2)

    def test_missing_wcs_step_line_raises_value_error(self):
        """Missing 'wcs_step:' comment line raises ValueError (not raw IndexError)."""
        with pytest.raises(ValueError):
            process_arcsecs("header", [])

    def test_tuple_input_returns_first_element(self):
        """Non-scalar input (e.g. tuple) returns only the first element."""
        result = process_arcsecs((0.3, 0.3), [])
        assert result == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# process_units
# ---------------------------------------------------------------------------


class TestProcessUnits:
    """process_units(input_units, comment_lines) — resolve flux units to an astropy Unit."""

    _UNIT_LINES = ["units: erg"]

    def test_header_reads_units_from_comment_line(self):
        """'header' extracts the units string from comment_lines and converts to Unit."""
        result = process_units("header", self._UNIT_LINES)
        assert isinstance(result, u.UnitBase)

    def test_header_units_match_expected_unit(self):
        """Units parsed from 'units: erg' equal u.Unit('erg')."""
        result = process_units("header", self._UNIT_LINES)
        assert result == u.Unit("erg")

    def test_explicit_string_converted_to_astropy_unit(self):
        """An explicit string bypasses header lookup and is converted to a Unit."""
        result = process_units("erg/s", [])
        assert isinstance(result, u.UnitBase)
        assert result == u.Unit("erg/s")

    def test_none_triggers_header_read(self):
        """None behaves the same as 'header'."""
        result = process_units(None, self._UNIT_LINES)
        assert isinstance(result, u.UnitBase)

    def test_already_a_unit_object_returned_unchanged(self):
        """An astropy Unit object passed directly is returned as-is (no string conversion)."""
        result = process_units(u.erg, [])
        assert result == u.erg

    def test_missing_units_line_raises_index_error(self):
        """Missing 'units:' comment line raises IndexError (contrast: process_arcsecs raises ValueError)."""
        with pytest.raises(IndexError):
            process_units("header", [])


# ---------------------------------------------------------------------------
# contours_to_arcsec
# ---------------------------------------------------------------------------


class TestContoursToArcsec:
    """contours_to_arcsec(contour_output, galaxy_center_row, galaxy_center_col, arcsec_to_pixel).

    Converts pixel-space contour_output to arcseconds:
      - subtracts galaxy_center_row from row channel (index 0)
      - subtracts galaxy_center_col from col channel (index 1)
      - multiplies the entire array by arcsec_to_pixel
    """

    @staticmethod
    def _make_co():
        """Minimal 3-channel contour_output: rows, cols, half-widths."""
        return np.ma.array(
            [
                [10.0, 11.0, 12.0],  # row pixel coords
                [5.0, 6.0, 7.0],  # col pixel coords
                [2.0, 2.0, 2.0],
            ],  # half-widths (not touched by this function)
        )

    def test_default_args_return_copy_equal_to_input(self):
        """All defaults (center=0, scale=1) → output equals input values."""
        co = self._make_co()
        result = contours_to_arcsec(co)
        np.testing.assert_array_equal(result, co)

    def test_does_not_modify_original(self):
        """Input array is not modified in-place (copy is taken)."""
        co = self._make_co()
        original = co.copy()
        contours_to_arcsec(co, galaxy_center_row=5)
        np.testing.assert_array_equal(co, original)

    def test_row_channel_shifted_by_galaxy_center_row(self):
        """Row channel (index 0) is reduced by galaxy_center_row."""
        co = self._make_co()
        result = contours_to_arcsec(co, galaxy_center_row=5)
        np.testing.assert_array_equal(result[0], [5.0, 6.0, 7.0])  # 10,11,12 − 5
        np.testing.assert_array_equal(result[1], co[1])  # col unchanged

    def test_col_channel_shifted_by_galaxy_center_col(self):
        """Col channel (index 1) is reduced by galaxy_center_col."""
        co = self._make_co()
        result = contours_to_arcsec(co, galaxy_center_col=3)
        np.testing.assert_array_equal(result[1], [2.0, 3.0, 4.0])  # 5,6,7 − 3
        np.testing.assert_array_equal(result[0], co[0])  # row unchanged

    def test_scale_multiplies_entire_output(self):
        """arcsec_to_pixel multiplies all channels of the shifted array."""
        co = self._make_co()
        scale = 0.2
        result = contours_to_arcsec(co, galaxy_center_row=10, arcsec_to_pixel=scale)
        # row channel: (co[0] - 10) * 0.2 = [0, 0.2, 0.4]
        np.testing.assert_allclose(result[0], [0.0, 0.2, 0.4])
        # col channel: co[1] * 0.2 = [1.0, 1.2, 1.4]
        np.testing.assert_allclose(result[1], [1.0, 1.2, 1.4])


# ---------------------------------------------------------------------------
# plt_image_extent — matplotlib smoke test (Phase 1.10)
# ---------------------------------------------------------------------------


class TestPltImageExtentSmoke:
    """1.10 — plt_image_extent: smoke test using the Agg backend.

    The function calls plt.imshow / plt.gca() in stateful global-pyplot style
    (no fig/ax passed in).  The only contract tested here is that it does not
    raise when called with valid inputs — which is enough to catch matplotlib
    API renames or signature changes during a dependency upgrade.
    """

    def setup_method(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_does_not_raise_with_default_args(self):
        """Calling with a 2-D array and a valid extent does not raise."""
        import matplotlib.pyplot as plt

        from threadcount.procedures.analyze_outflow_extent import plt_image_extent

        data = np.ones((5, 5))
        extent = [-1.0, 1.0, -1.0, 1.0]  # [left, right, bottom, top] in arcsec
        plt_image_extent(data, extent, title="test")
        plt.close("all")

    def test_horizontal0_false_skips_axhline(self):
        """horizontal0=False should not add any axhline to the current axes."""
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        from threadcount.procedures.analyze_outflow_extent import plt_image_extent

        data = np.ones((5, 5))
        plt_image_extent(data, [-1, 1, -1, 1], horizontal0=False)
        ax = plt.gca()
        hlines = [
            c
            for c in ax.get_children()
            if isinstance(c, Line2D) and c.get_label() == "galaxy midplane"
        ]
        assert len(hlines) == 0
        plt.close("all")

    def test_horizontal0_true_adds_axhline(self):
        """horizontal0=True (default) should add an axhline labelled 'galaxy midplane'."""
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        from threadcount.procedures.analyze_outflow_extent import plt_image_extent

        data = np.ones((5, 5))
        plt_image_extent(data, [-1, 1, -1, 1], horizontal0=True)
        ax = plt.gca()
        hlines = [
            c
            for c in ax.get_children()
            if isinstance(c, Line2D) and c.get_label() == "galaxy midplane"
        ]
        assert len(hlines) == 1
        plt.close("all")
