# Threadcount

Read the documentation: [https://threadcount.readthedocs.io](https://threadcount.readthedocs.io)

This package contains tools for fitting spectral data cubes, designed for internal use by Deanne Fisher's Research Group.

It relies heavily on other packages, including [lmfit](https://lmfit.github.io/lmfit-py/index.html) and [mpdaf](https://mpdaf.readthedocs.io/en/latest/).

The idea behind this is to be a set of analysis scripts for doing common tasks, with easy configuration.

We have added some common custom models to supplement lmfit's builtin models, have extended lmfit to handle monte-carlo iterations, and have extended mpdaf to interface with lmfit to fit spectra with ease. It uses an AIC function that is not the built-in lmfit AIC, due to the difference between lmfit version and the common definition of AIC.
