"""Tools for data cube peak fitting and analysis for the DUVET project."""
import logging
from . import fit
from . import lines
from . import models

from . import mpdaf_ext
from . import lmfit_ext

__all__ = ["fit", "lines", "models", "mpdaf_ext", "lmfit_ext"]

__version__ = "0.0.6"

# set up logger for the package
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s : %(message)s")
file_handler = logging.FileHandler("log.log", "a")
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
