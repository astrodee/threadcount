"""Library of line wavelengths."""
import matplotlib.pyplot as plt

# These are the air wavelengths in A
# Here's some resources for finding more lines.
# http://astronomy.nmsu.edu/drewski/tableofemissionlines.html
# http://www.pa.uky.edu/~peter/atomic/index.html

# define some constants in Angstroms.
OIII5007 = 5006.843  #: = 5006.843 # [O III] 5007
OIII4959 = 4958.911  #: = 4958.911 # [O III] 4959
OIII4363 = 4363.210  #: = 4363.210 # [O III] 4363

OII3726 = 3726.032  #: = 3726.032 # [O II] 3727 doublet, line 1
OII3729 = 3728.815  #: = 3728.815 # [O II] 3727 doublet, line 2

Hb4861 = 4861.333  #: = 4861.333 # Hβ
Hgamma = 4340.471  #: = 4340.471 # Hγ
Hdelta = 4101.742  #: = 4101.742 # Hδ

NeIII = 3868.760  #: = 3868.760 # [Ne III] 3869


# TODO: implement flexibility for other units.
# TODO: implement composite line, and ability to change attributes on this
# one after createion.
class Line(object):
    """Line object containing center and wavelength range."""

    def __init__(self, center, plus=15, minus=15, label="", save_str="", **kwargs):
        self.center = center
        self.plus = abs(plus)
        self.minus = abs(minus)
        self.low = center - self.minus
        self.high = center + self.plus
        self.label = label
        if save_str == "":
            save_str = str(round(center))  # default: rounded center wavelength.
        self.save_str = save_str
        self.__dict__.update(**kwargs)

    def __repr__(self):
        """Print Useful representation, copied from :class:`types.SimpleNamespace`."""
        items = (f"{k}={v!r}" for k, v in self.__dict__.items())
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def plot(self, ax=None, linestyle="--", color="black", autolabel=True, **kwargs):
        """Plot a vertical line at the Line center using :meth:`matplotlib.axes.Axes.axvline`.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`, optional
            axis to plot on. If none, use current axis, by default None
        linestyle : str, optional
            valid linestyle passed to :meth:`~matplotlib.axes.Axes.axvline`, by default "--"
        color : str, optional
            valid color passed to :meth:`~matplotlib.axes.Axes.axvline`, by default "black"
        autolabel : bool, optional
            Whether to use `self.label` for the axvline label, by default True.
            Overridden by any explicit `label` keyword.
        **kwargs : dict, optional
            keword arguments passed to :meth:`matplotlib.axes.Axes.axvline`

        Returns
        -------
        :class:`matplotlib.pyplot.axes.Axes`
        """
        # check for explicit label in kwargs:
        if "label" in kwargs:
            label = kwargs.pop("label")
        else:
            if autolabel:
                label = self.label
            else:
                label = None
        # select the axes:
        if ax is None:
            ax = plt.gca()
        ax.axvline(self.center, linestyle=linestyle, color=color, label=label)
        return ax

    # def __copy__(self):
    #     a = Line(**self.__dict__)
    #     return a

    # def copy(self):
    #     self.__copy__()


L_OIII5007 = Line(OIII5007, plus=15, minus=15, label="[OIII] 5007")
"""Line instance, OIII5007 +/- 15"""

L_OIII4959 = Line(OIII4959, plus=15, minus=15, label="[OIII] 4959")
"""Line instance, OIII4959 +/- 15"""

L_OIII4363 = Line(OIII4363, plus=15, minus=15, label="[OIII] 4363")
"""Line instance, OIII4363 +/- 15"""

L_OII3727d = Line(
    (OII3726 + OII3729) / 2,
    plus=16,
    minus=16,
    label="[O II] 3727 doublet",
    save_str="3727",
)
"""Line instance, (OII3726 + OII3729)/2 +/- 16"""

L_Hb4861 = Line(Hb4861, plus=15, minus=15, label="Hβ", save_str="Hbeta")
"""Line instance, Hb4861 +/- 15"""

L_Hgamma = Line(Hgamma, plus=15, minus=15, label="Hγ", save_str="Hgamma")
"""Line instance, Hgamma +/- 15"""

L_Hdelta = Line(Hdelta, plus=15, minus=15, label="Hδ", save_str="Hdelta")
"""Line instance, Hdelta +/- 15"""

L_NeIII = Line(NeIII, plus=15, minus=15, label="[Ne III] 3869")
"""Line instance, NeIII +/- 15"""
