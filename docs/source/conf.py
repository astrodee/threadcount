# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))


# -- Project information -----------------------------------------------------

project = "Threadcount"
copyright = "2021, SEB"
author = "SEB"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    # "numpydoc",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "sphinx_automodapi.automodsumm",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Options for Sphinx extensions ------------------------

# sphinx_automodapi settings
automodapi_toctreedirnm = "api"
automodapi_inheritance_diagram = False

# numpydoc settings
numpydoc_show_class_members = False

# shpinx.ext.autodoc settings
autoclass_content = "both"
autosummary_imported_members = False
autodoc_member_order = "bysource"

# shpinx.ext.intersphinx settings
intersphinx_mapping = {
    "py": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
    "lmfit": ("https://lmfit.github.io/lmfit-py/", None),
    "ipython": ("https://ipython.readthedocs.io/en/stable/", None,),
    "specutils": ("https://specutils.readthedocs.io/en/stable", None),
    "mpdaf": ("https://mpdaf.readthedocs.io/en/stable", None),
}
