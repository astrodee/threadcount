"""Tools for data cube peak fitting and analysis for the DUVET project."""

from . import fit
from . import lines
from . import models

from . import mpdaf_ext
from . import lmfit_ext

__all__ = ["fit", "lines", "models", "mpdaf_ext", "lmfit_ext"]

__version__ = "0.1.16"
