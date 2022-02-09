Set common matplotlib rcParams
------------------------------

:mod:`threadcount.procedures.set_rcParams`

Easy script with some presets, which also allows resetting to whatever was set
before this script started modifying things.

The presets include::

  {
    "image.aspect": "equal",
    "image.origin": "lower",
    "image.interpolation": "nearest",
    "image.cmap": "Blues_r",
    "axes.facecolor": "whitesmoke",
  }

The basics of the script takes this dictionary, and updates it with the dictionary
passed into the set_params() function. Then updates the matplotlib.rcParams with
that resulting dictionary. See below for usage examples::

  # The first time this is imported, set_rcParams.set_params() is run, which sets
  # the rcParams to the presets.
  from threadcount.procedures import set_rcParams

  # reset the rcParams to the presets:
  set_rcParams.set_params()

  # update the rcParams cmap with "Reds_r":
  set_rcParams.set_params({"image.cmap" : "Reds_r"})

  # reset to whatever was originally set:
  set_rcParams.reset_params()
