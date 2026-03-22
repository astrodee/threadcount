"""Tests for threadcount.lines — Line construction and pre-defined constants."""

import pytest

import threadcount.lines as lines
from threadcount.lines import Line

# ---------------------------------------------------------------------------
# Line construction
# ---------------------------------------------------------------------------


class TestLineConstruction:
    """Line.__init__ stores all attributes correctly."""

    def test_center_stored(self):
        line = Line(5006.843)
        assert line.center == 5006.843

    def test_plus_stored(self):
        line = Line(5006.843, plus=20)
        assert line.plus == 20

    def test_minus_stored(self):
        line = Line(5006.843, minus=10)
        assert line.minus == 10

    def test_defaults_plus_minus(self):
        line = Line(5006.843)
        assert line.plus == 15
        assert line.minus == 15

    def test_high_computed(self):
        line = Line(5000.0, plus=20, minus=10)
        assert line.high == 5020.0

    def test_low_computed(self):
        line = Line(5000.0, plus=20, minus=10)
        assert line.low == 4990.0

    def test_label_stored(self):
        line = Line(5006.843, label="[OIII] 5007")
        assert line.label == "[OIII] 5007"

    def test_label_default_empty(self):
        line = Line(5006.843)
        assert line.label == ""

    def test_save_str_explicit(self):
        line = Line(5006.843, save_str="OIII5007")
        assert line.save_str == "OIII5007"

    def test_save_str_default_is_rounded_center(self):
        line = Line(5006.843)
        assert line.save_str == "5007"

    def test_save_str_default_rounds_correctly(self):
        line = Line(4861.333)
        assert line.save_str == "4861"

    def test_abs_plus(self):
        """plus is stored as abs value so negative input still gives positive range."""
        line = Line(5000.0, plus=-20)
        assert line.plus == 20
        assert line.high == 5020.0

    def test_abs_minus(self):
        line = Line(5000.0, minus=-10)
        assert line.minus == 10
        assert line.low == 4990.0

    def test_extra_kwargs_stored(self):
        line = Line(5006.843, foo="bar", answer=42)
        assert line.foo == "bar"
        assert line.answer == 42

    def test_repr_contains_class_name(self):
        line = Line(5006.843)
        assert repr(line).startswith("Line(")

    def test_repr_contains_center(self):
        line = Line(5006.843)
        assert "center=5006.843" in repr(line)

    def test_repr_contains_label(self):
        line = Line(5006.843, label="[OIII] 5007")
        assert "label='[OIII] 5007'" in repr(line)

    def test_repr_contains_all_keys(self):
        line = Line(5000.0, plus=20, minus=10, label="test", save_str="5000")
        r = repr(line)
        for key in ("center", "plus", "minus", "low", "high", "label", "save_str"):
            assert key in r

    def test_repr_roundtrippable_center(self):
        """The repr value for center can be parsed back to the original float."""
        line = Line(4861.333)
        r = repr(line)
        # extract "center=<value>" and eval just that value
        import re

        match = re.search(r"center=([\d.]+)", r)
        assert match is not None
        assert float(match.group(1)) == pytest.approx(4861.333)


# ---------------------------------------------------------------------------
# Wavelength scalar constants
# ---------------------------------------------------------------------------


class TestWavelengthConstants:
    """Module-level float constants have the documented air-wavelength values."""

    def test_OIII5007(self):
        assert lines.OIII5007 == pytest.approx(5006.843)

    def test_OIII4959(self):
        assert lines.OIII4959 == pytest.approx(4958.911)

    def test_OIII4363(self):
        assert lines.OIII4363 == pytest.approx(4363.210)

    def test_OII3726(self):
        assert lines.OII3726 == pytest.approx(3726.032)

    def test_OII3729(self):
        assert lines.OII3729 == pytest.approx(3728.815)

    def test_Hb4861(self):
        assert lines.Hb4861 == pytest.approx(4861.333)

    def test_Hgamma(self):
        assert lines.Hgamma == pytest.approx(4340.471)

    def test_Hdelta(self):
        assert lines.Hdelta == pytest.approx(4101.742)

    def test_NeIII(self):
        assert lines.NeIII == pytest.approx(3868.760)


# ---------------------------------------------------------------------------
# Pre-defined Line constants (L_* objects)
# ---------------------------------------------------------------------------


class TestPredefinedLines:
    """L_* module constants are Line instances with correct center wavelengths."""

    def test_L_OIII5007_is_Line(self):
        assert isinstance(lines.L_OIII5007, Line)

    def test_L_OIII5007_center(self):
        assert lines.L_OIII5007.center == pytest.approx(lines.OIII5007)

    def test_L_OIII5007_bandwidth(self):
        assert lines.L_OIII5007.plus == 15
        assert lines.L_OIII5007.minus == 15

    def test_L_OIII5007_high_low(self):
        assert lines.L_OIII5007.high == pytest.approx(lines.OIII5007 + 15)
        assert lines.L_OIII5007.low == pytest.approx(lines.OIII5007 - 15)

    def test_L_OIII4959_center(self):
        assert lines.L_OIII4959.center == pytest.approx(lines.OIII4959)

    def test_L_OIII4959_bandwidth(self):
        assert lines.L_OIII4959.plus == 15
        assert lines.L_OIII4959.minus == 15

    def test_L_OIII4363_center(self):
        assert lines.L_OIII4363.center == pytest.approx(lines.OIII4363)

    def test_L_OII3727d_center(self):
        expected = (lines.OII3726 + lines.OII3729) / 2
        assert lines.L_OII3727d.center == pytest.approx(expected)

    def test_L_OII3727d_bandwidth(self):
        assert lines.L_OII3727d.plus == 16
        assert lines.L_OII3727d.minus == 16

    def test_L_OII3727d_save_str(self):
        assert lines.L_OII3727d.save_str == "3727"

    def test_L_Hb4861_center(self):
        assert lines.L_Hb4861.center == pytest.approx(lines.Hb4861)

    def test_L_Hb4861_save_str(self):
        assert lines.L_Hb4861.save_str == "Hbeta"

    def test_L_Hgamma_center(self):
        assert lines.L_Hgamma.center == pytest.approx(lines.Hgamma)

    def test_L_Hgamma_save_str(self):
        assert lines.L_Hgamma.save_str == "Hgamma"

    def test_L_Hdelta_center(self):
        assert lines.L_Hdelta.center == pytest.approx(lines.Hdelta)

    def test_L_Hdelta_save_str(self):
        assert lines.L_Hdelta.save_str == "Hdelta"

    def test_L_NeIII_center(self):
        assert lines.L_NeIII.center == pytest.approx(lines.NeIII)

    def test_L_NeIII_label(self):
        assert lines.L_NeIII.label == "[Ne III] 3869"


# ---------------------------------------------------------------------------
# Line.plot()
# ---------------------------------------------------------------------------


class TestLinePlot:
    """Line.plot() draws an axvline at the correct position and returns the axes."""

    def setup_method(self):
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend, safe in CI
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_returns_axes(self):
        import matplotlib.pyplot as plt

        line = Line(5006.843, label="test")
        ax = line.plot()
        assert ax is plt.gca()
        plt.close("all")

    def test_returns_provided_axes(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        line = Line(5006.843)
        returned = line.plot(ax=ax)
        assert returned is ax
        plt.close("all")

    def test_vline_at_center(self):
        import matplotlib.pyplot as plt

        center = 5006.843
        line = Line(center, label="")
        ax = line.plot()
        vlines = [
            c
            for c in ax.get_children()
            if hasattr(c, "get_xdata")
            and len(c.get_xdata()) == 2
            and c.get_xdata()[0] == pytest.approx(center)
        ]
        assert len(vlines) == 1
        plt.close("all")

    def test_autolabel_uses_line_label(self):
        import matplotlib.pyplot as plt

        line = Line(5006.843, label="[OIII] 5007")
        ax = line.plot()
        labels = [c.get_label() for c in ax.get_children() if hasattr(c, "get_label")]
        assert "[OIII] 5007" in labels
        plt.close("all")

    def test_autolabel_false_gives_no_label(self):
        import matplotlib.pyplot as plt

        line = Line(5006.843, label="[OIII] 5007")
        ax = line.plot(autolabel=False)
        # axvline with label=None gets the auto-generated "_line0" style label
        labels = [c.get_label() for c in ax.get_children() if hasattr(c, "get_label")]
        assert "[OIII] 5007" not in labels
        plt.close("all")

    def test_explicit_label_overrides_autolabel(self):
        import matplotlib.pyplot as plt

        line = Line(5006.843, label="[OIII] 5007")
        ax = line.plot(label="custom")
        labels = [c.get_label() for c in ax.get_children() if hasattr(c, "get_label")]
        assert "custom" in labels
        assert "[OIII] 5007" not in labels
        plt.close("all")

    def test_uses_gca_when_no_ax_given(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        # Make ax the current axes
        plt.sca(ax)
        line = Line(5006.843)
        returned = line.plot()
        assert returned is ax
        plt.close("all")
