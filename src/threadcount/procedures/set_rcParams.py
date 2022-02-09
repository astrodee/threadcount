"""Interface to set and reset some common matplotlib rcParams."""
import matplotlib as mpl

default_rcParams = {
    "image.aspect": "equal",
    "image.origin": "lower",
    "image.interpolation": "nearest",
    "image.cmap": "Blues_r",
    "axes.facecolor": "whitesmoke",
}

my_rcParams = default_rcParams.copy()

orig_rcParams = {}


def set_params(kwargs=None):
    if kwargs is None:
        kwargs = default_rcParams
    global orig_rcParams
    mpl.rcParams.update(orig_rcParams)
    my_rcParams.update(kwargs)
    # store the rcparams I will change, so I can restore them at the end... I dont want to
    # mess with peoples settings.
    orig_rcParams = {key: mpl.rcParams.get(key) for key in my_rcParams.keys()}
    mpl.rcParams.update(my_rcParams)


def reset_params():
    mpl.rcParams.update(orig_rcParams)


set_params()
