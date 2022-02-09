.. Threadcount documentation master file, created by
   sphinx-quickstart on Fri Dec  3 09:59:50 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Threadcount's documentation!
==========================================

This package contains tools for fitting spectral data cubes, designed for 
internal use by the Fisher Group.

It relies heavily on other packages, including 
`lmfit <https://lmfit.github.io/lmfit-py/index.html>`_ and 
`mpdaf <https://mpdaf.readthedocs.io/en/latest/>`_.

The idea behind this is to be a set of analysis scripts for doing common tasks, 
with easy configuration.

We have added some common custom models to supplement lmfit's builtin models, 
have extended lmfit to handle monte-carlo iterations, and have extended mpdaf 
to interface with lmfit to fit spectra with ease.

.. toctree::
   :maxdepth: 4
   :caption: Contents:
   
   installation
   getting_started

.. toctree::
   :maxdepth: 2

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

