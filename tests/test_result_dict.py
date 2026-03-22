"""Tests for ResultDict (roadmap item 1.6).

Covers:
  - Construction from data_array + names.
  - Construction from data_dict.
  - generate_pixel_coordinates inserts row / col entries.
  - comment attribute is stored.
  - names() returns keys in insertion order.
  - data() produces a correctly shaped, transposed float array.
  - savetxt / loadtxt round-trip for clean (non-NaN) cubes.
  - savetxt / loadtxt round-trip for NaN-containing arrays (masked cubes).
  - comment string survives the round-trip.
  - Custom delimiter survives the round-trip.
  - apply_mask fills non-DIM_NAMES entries and leaves DIM_NAMES intact.
  - Mismatched names / data raises ValueError.
  - loadtxt raises ValueError when row/col indices cannot be matched.
"""

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest

import threadcount.fit as tc_fit

ResultDict = tc_fit.ResultDict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple(shape=(3, 4), n_maps=2, seed=0):
    """Return (data_array, names) for a simple ResultDict with `n_maps` maps."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_maps, *shape))
    names = [f"map{i}" for i in range(n_maps)]
    return data, names


def _round_trip(rd, *, delimiter="\t", tmpdir):
    """Save rd to a temp file and reload it; return the loaded ResultDict."""
    fpath = Path(tmpdir) / "result.txt"
    rd.savetxt(str(fpath), delimiter=delimiter)
    return ResultDict.loadtxt(str(fpath), delimiter=delimiter)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestResultDictConstruction:
    def test_from_data_array_and_names(self):
        data, names = _make_simple()
        rd = ResultDict(data, names, generate_pixel_coordinates=False)
        assert rd.names() == names
        assert all(np.array_equal(rd[n], data[i]) for i, n in enumerate(names))

    def test_from_data_dict(self):
        a = np.ones((3, 4))
        b = np.zeros((3, 4))
        rd = ResultDict(data_dict={"a": a, "b": b}, generate_pixel_coordinates=False)
        assert list(rd.keys()) == ["a", "b"]
        np.testing.assert_array_equal(rd["a"], a)

    def test_generate_pixel_coordinates_inserts_row_col(self):
        data, names = _make_simple(shape=(3, 4))
        rd = ResultDict(data, names)
        assert "row" in rd
        assert "col" in rd
        # row/col are the first two keys
        key_list = rd.names()
        assert key_list.index("row") < key_list.index(names[0])
        assert key_list.index("col") < key_list.index(names[0])

    def test_generate_pixel_coordinates_values(self):
        data, names = _make_simple(shape=(3, 4))
        rd = ResultDict(data, names)
        expected_rows, expected_cols = np.indices((3, 4))
        np.testing.assert_array_equal(rd["row"], expected_rows)
        np.testing.assert_array_equal(rd["col"], expected_cols)

    def test_no_generate_pixel_coordinates(self):
        data, names = _make_simple()
        rd = ResultDict(data, names, generate_pixel_coordinates=False)
        assert "row" not in rd
        assert "col" not in rd

    def test_comment_stored(self):
        data, names = _make_simple()
        rd = ResultDict(data, names, comment="hello world")
        assert rd.comment == "hello world"

    def test_names_length_mismatch_raises(self):
        data = np.ones((3, 4, 5))
        with pytest.raises(ValueError):
            ResultDict(data, ["only_one_name"], generate_pixel_coordinates=False)

    def test_empty_construction(self):
        rd = ResultDict()
        assert len(rd) == 0
        assert rd.names() == []

    def test_names_none_autogenerates(self):
        """When data_array is given without names, keys default to data_0, data_1, ..."""
        data = np.ones((3, 4, 5))
        rd = ResultDict(data, names=None, generate_pixel_coordinates=False)
        assert rd.names() == ["data_0", "data_1", "data_2"]
        np.testing.assert_array_equal(rd["data_0"], data[0])

    def test_combined_data_dict_and_data_array(self):
        """data_dict initialises first; data_array/names update it; overlapping key is overridden."""
        x = np.ones((3, 4))
        y = np.zeros((3, 4))
        z_orig = np.full((3, 4), 99.0)
        z_new = np.full((3, 4), 7.0)
        rd = ResultDict(
            data_array=np.stack([z_new]),
            names=["z"],
            data_dict={"x": x, "y": y, "z": z_orig},
            generate_pixel_coordinates=False,
        )
        # All three keys exist
        assert set(rd.names()) == {"x", "y", "z"}
        # data_array value overrides data_dict value for the shared key
        np.testing.assert_array_equal(rd["z"], z_new)
        np.testing.assert_array_equal(rd["x"], x)
        np.testing.assert_array_equal(rd["y"], y)

    def test_generate_pixel_coordinates_with_data_dict_only(self):
        """Default generate_pixel_coordinates=True adds row/col from a data_dict input."""
        arr = np.ones((3, 4))
        rd = ResultDict(
            data_dict={"flux": arr}
        )  # default generate_pixel_coordinates=True
        assert "row" in rd
        assert "col" in rd
        expected_rows, expected_cols = np.indices((3, 4))
        np.testing.assert_array_equal(rd["row"], expected_rows)
        np.testing.assert_array_equal(rd["col"], expected_cols)


# ---------------------------------------------------------------------------
# data()
# ---------------------------------------------------------------------------


class TestResultDictData:
    def test_shape_is_transposed(self):
        data, names = _make_simple(shape=(3, 4), n_maps=2)
        rd = ResultDict(data, names, generate_pixel_coordinates=False)
        # 2 maps over 3x4 spatial → data() should be (12, 2)
        d = rd.data()
        assert d.shape == (12, 2)

    def test_values_match(self):
        data, names = _make_simple(shape=(3, 4), n_maps=2)
        rd = ResultDict(data, names, generate_pixel_coordinates=False)
        d = rd.data()
        flat0 = data[0].ravel()
        flat1 = data[1].ravel()
        np.testing.assert_array_equal(d[:, 0], flat0)
        np.testing.assert_array_equal(d[:, 1], flat1)


# ---------------------------------------------------------------------------
# Round-trip: clean data
# ---------------------------------------------------------------------------


class TestResultDictRoundTripClean:
    def test_basic_round_trip(self, tmp_path):
        rng = np.random.default_rng(1)
        data = rng.standard_normal((2, 5, 6))
        names = ["alpha", "beta"]
        rd = ResultDict(data, names)
        rd2 = _round_trip(rd, tmpdir=tmp_path)

        for name in names:
            np.testing.assert_allclose(rd2[name], rd[name], rtol=1e-6)

    def test_row_col_survive_round_trip(self, tmp_path):
        data, names = _make_simple(shape=(4, 5))
        rd = ResultDict(data, names)
        rd2 = _round_trip(rd, tmpdir=tmp_path)
        np.testing.assert_array_equal(rd2["row"], rd["row"])
        np.testing.assert_array_equal(rd2["col"], rd["col"])

    def test_names_preserved(self, tmp_path):
        data, names = _make_simple()
        rd = ResultDict(data, names)
        rd2 = _round_trip(rd, tmpdir=tmp_path)
        for name in names:
            assert name in rd2

    def test_comment_survives_round_trip(self, tmp_path):
        data, names = _make_simple()
        rd = ResultDict(data, names, comment="test comment line")
        rd2 = _round_trip(rd, tmpdir=tmp_path)
        assert "test comment line" in rd2.comment

    def test_custom_delimiter(self, tmp_path):
        data, names = _make_simple(shape=(3, 3))
        rd = ResultDict(data, names)
        rd2 = _round_trip(rd, delimiter=",", tmpdir=tmp_path)
        for name in names:
            np.testing.assert_allclose(rd2[name], rd[name], rtol=1e-6)

    def test_single_map_round_trip(self, tmp_path):
        rng = np.random.default_rng(7)
        data = rng.standard_normal((1, 6, 8))
        names = ["flux"]
        rd = ResultDict(data, names)
        rd2 = _round_trip(rd, tmpdir=tmp_path)
        np.testing.assert_allclose(rd2["flux"], rd["flux"], rtol=1e-6)

    def test_integer_like_values_survive(self, tmp_path):
        data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        names = ["a", "b"]
        rd = ResultDict(data, names)
        rd2 = _round_trip(rd, tmpdir=tmp_path)
        np.testing.assert_allclose(rd2["a"], rd["a"])
        np.testing.assert_allclose(rd2["b"], rd["b"])


# ---------------------------------------------------------------------------
# Round-trip: NaN-containing data (masked cubes)
# ---------------------------------------------------------------------------


class TestResultDictRoundTripNaN:
    """NaN values are the normal case for masked spaxels in fitted cubes."""

    def test_all_nan_map_round_trip(self, tmp_path):
        data = np.full((2, 4, 5), np.nan)
        names = ["p", "q"]
        rd = ResultDict(data, names)
        rd2 = _round_trip(rd, tmpdir=tmp_path)
        assert np.all(np.isnan(rd2["p"]))
        assert np.all(np.isnan(rd2["q"]))

    def test_partial_nan_round_trip(self, tmp_path):
        """A realistic masked cube: some spaxels fitted, others NaN."""
        rng = np.random.default_rng(3)
        data = rng.standard_normal((3, 5, 5))
        # Mask a 2x2 corner
        data[:, :2, :2] = np.nan
        names = ["vel", "sigma", "flux"]
        rd = ResultDict(data, names)
        rd2 = _round_trip(rd, tmpdir=tmp_path)
        for name in names:
            orig = rd[name]
            reloaded = rd2[name]
            nan_mask = np.isnan(orig)
            # NaN locations stay NaN
            assert np.all(np.isnan(reloaded[nan_mask]))
            # Non-NaN locations recover correctly
            np.testing.assert_allclose(reloaded[~nan_mask], orig[~nan_mask], rtol=1e-6)

    def test_nan_and_finite_mixed_single_column(self, tmp_path):
        data = np.array([[[1.0, np.nan], [np.nan, 4.0]]])
        names = ["v"]
        rd = ResultDict(data, names)
        rd2 = _round_trip(rd, tmpdir=tmp_path)
        assert np.isnan(rd2["v"][0, 1])
        assert np.isnan(rd2["v"][1, 0])
        assert rd2["v"][0, 0] == pytest.approx(1.0)
        assert rd2["v"][1, 1] == pytest.approx(4.0)

    def test_inf_values_survive(self, tmp_path):
        """np.savetxt / loadtxt handle inf; verify it's preserved."""
        data = np.array([[[1.0, np.inf], [-np.inf, 4.0]]])
        names = ["w"]
        rd = ResultDict(data, names)
        rd2 = _round_trip(rd, tmpdir=tmp_path)
        assert np.isposinf(rd2["w"][0, 1])
        assert np.isneginf(rd2["w"][1, 0])


# ---------------------------------------------------------------------------
# apply_mask
# ---------------------------------------------------------------------------


class TestApplyMask:
    def test_mask_fills_data_maps(self):
        data = np.ones((2, 4, 4))
        names = ["a", "b"]
        rd = ResultDict(data, names)
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True
        rd.apply_mask(mask)
        assert np.isnan(rd["a"][0, 0])
        assert np.isnan(rd["b"][0, 0])
        assert rd["a"][1, 1] == pytest.approx(1.0)

    def test_mask_leaves_dim_names_intact(self):
        data, names = _make_simple(shape=(3, 4))
        rd = ResultDict(data, names)
        orig_row = rd["row"].copy()
        orig_col = rd["col"].copy()
        mask = np.ones((3, 4), dtype=bool)
        rd.apply_mask(mask)
        np.testing.assert_array_equal(rd["row"], orig_row)
        np.testing.assert_array_equal(rd["col"], orig_col)

    def test_mask_custom_fill_value(self):
        data = np.ones((2, 3, 3))
        names = ["x", "y"]
        rd = ResultDict(data, names)
        mask = np.zeros((3, 3), dtype=bool)
        mask[2, 2] = True
        rd.apply_mask(mask, fill_value=-999.0)
        assert rd["x"][2, 2] == pytest.approx(-999.0)

    def test_full_mask_all_nan(self):
        data = np.ones((1, 2, 2))
        names = ["z"]
        rd = ResultDict(data, names)
        rd.apply_mask(np.ones((2, 2), dtype=bool))
        assert np.all(np.isnan(rd["z"]))


# ---------------------------------------------------------------------------
# Round-trip: 3D spatial (panel + row + col)
# ---------------------------------------------------------------------------


class TestResultDictRoundTrip3D:
    """ResultDict.DIM_NAMES includes 'panel' for 3D spatial arrays.
    Exercises the panel+row+col coordinate generation and loadtxt reshape.
    """

    def test_panel_row_col_coordinates_inserted(self):
        """3D spatial array generates panel, row, and col coordinate maps."""
        rng = np.random.default_rng(11)
        data = rng.standard_normal((2, 2, 3, 4))  # 2 maps, panels=2, rows=3, cols=4
        names = ["flux", "vel"]
        rd = ResultDict(data, names)
        assert "panel" in rd
        assert "row" in rd
        assert "col" in rd
        # Coordinate arrays have the correct spatial shape
        assert rd["panel"].shape == (2, 3, 4)
        assert rd["row"].shape == (2, 3, 4)
        assert rd["col"].shape == (2, 3, 4)
        # Values are correct
        expected_panel, expected_row, expected_col = np.indices((2, 3, 4))
        np.testing.assert_array_equal(rd["panel"], expected_panel)
        np.testing.assert_array_equal(rd["row"], expected_row)
        np.testing.assert_array_equal(rd["col"], expected_col)

    def test_round_trip_3d(self, tmp_path):
        """savetxt/loadtxt round-trip for a 3D spatial ResultDict."""
        rng = np.random.default_rng(12)
        data = rng.standard_normal((2, 2, 3, 4))  # 2 maps, panels=2, rows=3, cols=4
        names = ["flux", "vel"]
        rd = ResultDict(data, names)
        rd2 = _round_trip(rd, tmpdir=tmp_path)
        for name in names:
            np.testing.assert_allclose(rd2[name], rd[name], rtol=1e-6)
        # Coordinate maps survive
        np.testing.assert_array_equal(rd2["panel"], rd["panel"])
        np.testing.assert_array_equal(rd2["row"], rd["row"])
        np.testing.assert_array_equal(rd2["col"], rd["col"])

    def test_round_trip_3d_with_nan(self, tmp_path):
        """NaN values survive a 3D round-trip."""
        rng = np.random.default_rng(13)
        data = rng.standard_normal((1, 2, 3, 4))
        data[0, 0, :, :] = np.nan  # blank out the first panel
        names = ["sigma"]
        rd = ResultDict(data, names)
        rd2 = _round_trip(rd, tmpdir=tmp_path)
        assert np.all(np.isnan(rd2["sigma"][0, :, :]))
        np.testing.assert_allclose(
            rd2["sigma"][1, :, :], rd["sigma"][1, :, :], rtol=1e-6
        )


# ---------------------------------------------------------------------------
# Round-trip: no pixel coordinates
# ---------------------------------------------------------------------------


class TestResultDictRoundTripNoCoordinates:
    """Round-trip through savetxt/loadtxt when generate_pixel_coordinates=False."""

    def test_no_coordinates_round_trip(self, tmp_path):
        """Files without row/col columns should survive a clean round-trip."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((3, 4, 5))
        names = ["vel", "sigma", "flux"]
        rd = ResultDict(data, names, generate_pixel_coordinates=False)
        assert "row" not in rd
        assert "col" not in rd

        fpath = tmp_path / "no_coords.txt"
        rd.savetxt(str(fpath))
        rd2 = ResultDict.loadtxt(str(fpath))

        assert rd2.names() == names
        for name in names:
            np.testing.assert_allclose(rd2[name], rd[name].ravel(), rtol=1e-6)

    def test_no_coordinates_nan_round_trip(self, tmp_path):
        """NaN values must survive when no pixel coordinates are present."""
        data = np.array([[[1.0, np.nan], [np.nan, 4.0]]])
        names = ["v"]
        rd = ResultDict(data, names, generate_pixel_coordinates=False)

        fpath = tmp_path / "no_coords_nan.txt"
        rd.savetxt(str(fpath))
        rd2 = ResultDict.loadtxt(str(fpath))

        assert np.isnan(rd2["v"][1])  # second element was NaN
        assert rd2["v"][0] == pytest.approx(1.0)
        assert rd2["v"][3] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Error handling in loadtxt
# ---------------------------------------------------------------------------


class TestResultDictLoadtxtErrors:
    """loadtxt error paths documented in the class docstring."""

    def test_mismatched_row_col_raises(self, tmp_path):
        """loadtxt raises ValueError when regenerated col indices don't match
        the col column read from the file.

        Four rows with duplicate diagonal pairs (0,0),(0,0),(1,1),(1,1) produce
        max_row=1, max_col=1 → reshape to (3,2,2) succeeds.  But the read-in
        col array [[0,0],[1,1]] differs from the regenerated col [[0,1],[0,1]],
        so the comparison branch raises ValueError.
        """
        fpath = tmp_path / "bad_coords.txt"
        # Duplicate diagonal pairs: reshape succeeds (3,2,2) but
        # col_readin [[0,0],[1,1]] ≠ col_regenerated [[0,1],[0,1]]
        content = "# \n# row\tcol\tflux\n0\t0\t1.0\n0\t0\t2.0\n1\t1\t3.0\n1\t1\t4.0\n"
        fpath.write_text(content)
        with pytest.raises(ValueError):
            ResultDict.loadtxt(str(fpath), delimiter="\t")

    def test_single_spaxel_round_trip(self, tmp_path):
        """A 1×1 spatial grid produces a single-row file that crashes loadtxt."""
        data = np.array([[[1.5]], [[2.5]]])  # shape (2, 1, 1)
        names = ["flux", "vel"]
        rd = ResultDict(data, names)
        fpath = tmp_path / "single_spaxel.txt"
        rd.savetxt(str(fpath))
        rd2 = ResultDict.loadtxt(str(fpath))
        np.testing.assert_allclose(rd2["flux"], rd["flux"])
        np.testing.assert_allclose(rd2["vel"], rd["vel"])
