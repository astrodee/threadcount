***************
Getting Started
***************

Note regarding Jupyter Notebooks
================================

I highly recommend using iPython, not Jupyter
---------------------------------------------

Displaying and updating the images was unreliable in Jupyter notebooks -- it
sometimes worked and sometimes didn't depending on versions. (suspect maybe
something to do with interaction with javascript).

The images are refreshed/redrawn, so the ability to update a drawn figure is
necessary. If you are still attempting to use a jupyter notebook 
and notice the figures are not behaving as you would expect, then
please try with iPython.


Procedures
==========

There are a few "procedures" included in threadcount. These are recipes
for common tasks, with an easily configurable interface.


.. toctree::
   :maxdepth: 2

   procedure_open_cube
   procedure_fit_lines
   procedure_outflow_extent
   procedure_rcparams


Models for use with lmfit
==========================

The Classes in the :mod:`threadcount.models` module gives a list of pre-defined
lmfit Models this package makes available for use. These are used just as
the built-in fitting models from lmfit would be used, and I have implemented a
guess method for most (if not all) of them.

Line objects
============

The Variables in the :mod:`threadcount.lines` module shows the predefined
wavelengths and Line instances. The variables beginning with "L\_" are the Line
objects used in the settings for the fit_lines procedure.



Extensions to mpdaf and lmfit
=============================

:mod:`threadcount.mpdaf_ext`

The most notable extension here is the :meth:`~threadcount.mpdaf_ext.lmfit` method
added to the Spectrum class.

:mod:`threadcount.lmfit_ext`

The most notable extensions for lmfit are the ModelResult class:

* Attributes :obj:`~threadcount.lmfit_ext.aic_real` and :obj:`~threadcount.lmfit_ext.bic_real`.
* Method :meth:`~threadcount.lmfit_ext.mc_iter` to perform Monte Carlo iterations
  of the ModelResult.