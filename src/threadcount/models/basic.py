import numpy as np

import lmfit

tiny = lmfit.models.tiny  # 1.0e-15


def guess_from_peak(y, x, negative=False):
    """Estimate starting values from 1D peak data and return (height,center,sigma).

    Parameters
    ----------
    y : array-like
        y data
    x : array-like
        x data
    negative : bool, optional
        determines if peak height is positive or negative, by default False

    Returns
    -------
    (height, center, sigma) : (float, float, float)
        Estimates of 1 gaussian line parameters.
    """
    sort_increasing = np.argsort(x)
    x = x[sort_increasing]
    y = y[sort_increasing]

    # find the max/min values of x and y, and the x value at max(y)
    maxy, miny = max(y), min(y)
    maxx, minx = max(x), min(x)

    height = maxy - miny

    # set a backup sigma, and center in case using the halfmax calculation doesn't work.
    # The backup sigma = 1/6 the full x range and the backup center is the
    # location of the maximum
    sig = (maxx - minx) / 6.0
    cen = x[np.argmax(y)]

    # the explicit conversion to a NumPy array is to make sure that the
    # indexing on line 65 also works if the data is supplied as pandas.Series

    # find the x positions where y is above (ymax+ymin)/2
    x_halfmax = np.array(x[y > (maxy + miny) / 2.0])
    if negative:
        height = -(maxy - miny)
        # backup center for if negative.
        cen = x[np.argmin(y)]
        x_halfmax = x[y < (maxy + miny) / 2.0]

    # calculate sigma and center based on where y is above half-max:
    if len(x_halfmax) > 2:
        sig = (x_halfmax[-1] - x_halfmax[0]) / 2.0
        cen = x_halfmax.mean()

    return height, cen, sig


def mean_edges(y, x=None, edge_fraction=0.1):
    """Compute the mean of the outer points of y.

    Mean the first and last n points in y, where n is given by
    len(y)*edge_fraction

    An edge_fraction = 0 will return the mean of the first and last points.
    An edge_fraction = 0.5 will return the mean of all y.
    """
    if edge_fraction > 0.5:
        raise ValueError("Edge fraction must be <= 0.5")
    # determine number of points in edge_fraction.
    limit = int(round(len(y) * edge_fraction))
    if limit == 0:
        limit = 1

    # ensure that y is arranged by increasing x.
    if x is not None:
        sort_increasing = np.argsort(x)
        y = y[sort_increasing]

    # extract the edge data points
    y_edges = np.r_[y[:limit], y[-limit:]]

    # return the mean of the edge points.
    return y_edges.mean()
