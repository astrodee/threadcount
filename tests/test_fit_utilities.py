"""Unit tests for threadcount.fit utility functions (Phase 1.8A).

Covers: get_index, iter_spaxel, get_region, get_reg_image, de_redshift.
No mpdaf, lmfit, or I/O dependencies — pure-logic tests only.
"""

import numpy as np
import pytest

import threadcount.fit as tf

# ---------------------------------------------------------------------------
# Helper — minimal WaveCoord stand-in (no mpdaf dependency)
# ---------------------------------------------------------------------------


class _MockWaveCoord:
    """Minimal stand-in for mpdaf.obj.WaveCoord."""

    def __init__(self, crval, step):
        self._crval = float(crval)
        self._step = float(step)

    def get_crval(self):
        return self._crval

    def set_crval(self, v):
        self._crval = float(v)

    def get_step(self):
        return self._step

    def set_step(self, v):
        self._step = float(v)


# ---------------------------------------------------------------------------
# get_index
# ---------------------------------------------------------------------------


class TestGetIndex:
    """1.8A — get_index: nearest-element index lookup."""

    def test_single_element_array_returns_zero(self):
        """Single-element array: only index 0 is ever possible."""
        assert tf.get_index([5.0], 3.0) == 0

    def test_exact_match(self):
        assert tf.get_index([10, 11, 12, 13, 14], 13) == 3

    def test_off_grid_closer_to_lower(self):
        # 12.3 is 0.3 from 12 (idx 2) and 0.7 from 13 (idx 3)
        assert tf.get_index([10, 11, 12, 13, 14], 12.3) == 2

    def test_off_grid_closer_to_upper(self):
        # 12.7 is 0.7 from 12 (idx 2) and 0.3 from 13 (idx 3)
        assert tf.get_index([10, 11, 12, 13, 14], 12.7) == 3

    def test_tie_breaks_towards_first_element(self):
        # 12.5 is equidistant; argmin returns the first (lower) occurrence
        assert tf.get_index([10, 11, 12, 13, 14], 12.5) == 2

    def test_vectorised_returns_list_of_same_length(self):
        result = tf.get_index([10, 11, 12, 13, 14], [13, 22])
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == 3  # exact match
        assert result[1] == 4  # 22 clamps to last element (14)

    def test_degenerate_single_element_any_value(self):
        """Regardless of value, a length-1 array always returns index 0."""
        assert tf.get_index([99.0], -1e9) == 0
        assert tf.get_index([99.0], 1e9) == 0

    def test_unsorted_array(self):
        """argmin works on non-monotonic arrays; value 21 is nearest to 20 (idx 3)."""
        assert tf.get_index([50, 10, 30, 20, 40], 21) == 3


# ---------------------------------------------------------------------------
# iter_spaxel
# ---------------------------------------------------------------------------


class TestIterSpaxel:
    """1.8A — iter_spaxel: generator over 2-D array spaxels."""

    def test_index_false_values_match_row_major_order(self):
        image = np.arange(15).reshape(3, 5)
        values = list(tf.iter_spaxel(image, index=False))
        assert values == list(image.ravel())

    def test_index_true_indices_cover_all_spaxels(self):
        image = np.zeros((3, 5))
        _, indices = zip(*tf.iter_spaxel(image, index=True))
        assert set(indices) == set(np.ndindex(*image.shape))

    def test_index_true_value_matches_image_at_each_index(self):
        image = np.arange(15).reshape(3, 5)
        for val, idx in tf.iter_spaxel(image, index=True):
            assert val == image[idx]

    def test_single_spaxel_array(self):
        image = np.array([[42]])
        result = list(tf.iter_spaxel(image, index=False))
        assert result == [42]

    def test_index_true_single_spaxel(self):
        """index=True on a 1x1 array must yield exactly (value, (0, 0))."""
        image = np.array([[42]])
        result = list(tf.iter_spaxel(image, index=True))
        assert len(result) == 1
        val, idx = result[0]
        assert val == 42
        assert idx == (0, 0)


# ---------------------------------------------------------------------------
# get_region
# ---------------------------------------------------------------------------


class TestGetRegion:
    """1.8A — get_region: pixel list inside a circle or ellipse."""

    def test_circle_all_pixels_satisfy_inequality(self):
        region = tf.get_region(rx=2)
        rows, cols = region[:, 0], region[:, 1]
        assert np.all(rows**2 + cols**2 <= 2**2)

    def test_ellipse_pixel_count_and_inequality(self):
        # rx=3, ry=1: row²/1 + col²/9 ≤ 1
        # row=0: col ∈ {-3,..,3} → 7 pixels; row=±1: col=0 → 2 pixels; total=9
        region = tf.get_region(rx=3, ry=1)
        rows, cols = region[:, 0], region[:, 1]
        assert len(region) == 9
        assert np.all(rows**2 / 1.0**2 + cols**2 / 3.0**2 <= 1.0 + 1e-12)

    def test_list_argument_equals_separate_rx_ry(self):
        region_list = tf.get_region([3, 1])
        region_sep = tf.get_region(3, 1)
        np.testing.assert_array_equal(region_list, region_sep)

    def test_negative_rx_equals_positive(self):
        """abs() normalisation: get_region(-2) must equal get_region(2)."""
        np.testing.assert_array_equal(tf.get_region(-2), tf.get_region(2))

    @pytest.mark.xfail(
        reason=(
            "Bug: get_region(0) triggers a division-by-zero when rx2=ry2=0. "
            "The inside-ellipse check computes col²/rx2 = 0/0 = nan, making "
            "nan <= 1 evaluate to False, so all pixels are excluded and an empty "
            "array is returned instead of [[0, 0]]. Fix: special-case rx==0 "
            "before computing rx2."
        ),
        strict=True,
    )
    def test_rx_zero_returns_only_origin(self):
        region = tf.get_region(0)
        np.testing.assert_array_equal(region, [[0, 0]])


# ---------------------------------------------------------------------------
# get_reg_image
# ---------------------------------------------------------------------------


class TestGetRegImage:
    """1.8A — get_reg_image: kernel image from region pixel list."""

    def test_output_shape(self):
        region = tf.get_region(rx=2)
        img = tf.get_reg_image(region)
        rows, cols = region[:, 0], region[:, 1]
        expected = (int(rows.max() - rows.min()) + 1, int(cols.max() - cols.min()) + 1)
        assert img.shape == expected

    def test_region_pixels_are_one_others_are_zero(self):
        region = tf.get_region(rx=2)
        img = tf.get_reg_image(region)
        mins = region.min(axis=0)
        # every region pixel maps to 1
        for pix in region:
            assert img[tuple(pix - mins)] == 1.0
        # total non-zero count equals region size
        assert np.count_nonzero(img) == len(region)

    def test_round_trip_sum_equals_region_length(self):
        region = tf.get_region(rx=2)
        assert tf.get_reg_image(region).sum() == len(region)

    def test_single_pixel_region(self):
        """A one-pixel region must produce a 1x1 image with a single 1."""
        region = np.array([[0, 0]])
        img = tf.get_reg_image(region)
        assert img.shape == (1, 1)
        assert img[0, 0] == 1.0


# ---------------------------------------------------------------------------
# de_redshift
# ---------------------------------------------------------------------------


class TestDeRedshift:
    """1.8A — de_redshift: in-place WaveCoord de-redshifting."""

    def test_both_zero_no_change(self):
        wc = _MockWaveCoord(crval=5007.0, step=0.3)
        tf.de_redshift(wc, z=0, z_initial=0)
        assert wc.get_crval() == pytest.approx(5007.0)
        assert wc.get_step() == pytest.approx(0.3)

    def test_z_equals_z_initial_no_change(self):
        wc = _MockWaveCoord(crval=5007.0, step=0.3)
        tf.de_redshift(wc, z=0.15, z_initial=0.15)
        assert wc.get_crval() == pytest.approx(5007.0)
        assert wc.get_step() == pytest.approx(0.3)

    def test_known_analytic_values(self):
        crval_in, step_in = 5000.0, 0.5
        z_initial, z = 0.1, 0.0
        wc = _MockWaveCoord(crval=crval_in, step=step_in)
        tf.de_redshift(wc, z=z, z_initial=z_initial)
        assert wc.get_crval() == pytest.approx(crval_in * 1.1 / 1.0)
        assert wc.get_step() == pytest.approx(step_in * 1.1 / 1.0)

    def test_known_analytic_values_reverse_direction(self):
        """Adding redshift (z_initial=0, z=0.1): crval and step must decrease."""
        crval_in, step_in = 5000.0, 0.5
        z_initial, z = 0.0, 0.1
        wc = _MockWaveCoord(crval=crval_in, step=step_in)
        tf.de_redshift(wc, z=z, z_initial=z_initial)
        assert wc.get_crval() == pytest.approx(crval_in * 1.0 / 1.1)
        assert wc.get_step() == pytest.approx(step_in * 1.0 / 1.1)

    def test_return_value_is_z(self):
        wc = _MockWaveCoord(crval=5000.0, step=0.3)
        assert tf.de_redshift(wc, z=0.05, z_initial=0.1) == pytest.approx(0.05)
