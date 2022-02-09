============================
Downloading and Installation
============================

Prerequisites
-------------

Threadcount works with Python versions 3.6 and higher.

Threadcount requires the following packages. The versions given are what I
used when developing the code, and I don't know if they are actual requirements.

    * lmfit >= 1.0.1
    * matplotlib >= 3.4.3
    * mpdaf >= 3.5 (should be installed via pip -- the conda version is outdated.)
    * numpy >= 1.17.0
    * astropy >= 3.2.1

pip is used for installation.

These dependencies should be installed upon installing threadcount, if they are
not present already.

Installation
------------

You may install threadcount through pip directly from github if you have a
working git executable::

  python -m pip install git+https://github.com/astrodee/threadcount#egg=threadcount

If you would like to start from a fresh conda environemnt (in this case, I named
it "tc" but you can use anything you like instead), you may do the following,
and pip will install all the dependencies::

  conda create -n tc pip
  conda activate tc
  (tc) python -m pip install git+https://github.com/astrodee/threadcount#egg=threadcount


Install an editable version
---------------------------

Inside the directory you wish to keep the source code, run this::

  python -m pip install -e git+https://github.com/astrodee/threadcount#egg=threadcount

Now, the package source files will be stored in ./src/threadcount where you can
edit them if necessary. The package will still be available to import as usual
from any location.




