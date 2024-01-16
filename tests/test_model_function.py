import numpy as np
import random
import numba
import lmfit

tiny = lmfit.models.tiny


def gaussian3CH_d(
    x,
    g1_height=1.0,
    deltax=0.0,
    g1_sigma=1.0,
    g2_height=1.0,
    g2_center=0.0,
    g2_sigma=1.0,
    g3_height=1.0,
    deltaxhi=0.0,
    g3_sigma=1.0,
    c=0.0,
):
    """Return a 3-Gaussian function in 1-dimension."""
    f = (
        g1_height * np.exp(-((1.0 * x - g2_center - deltax) ** 2) / max(tiny, (2 * g1_sigma**2)))
        + g2_height * np.exp(-((1.0 * x - g2_center) ** 2) / max(tiny, (2 * g2_sigma**2)))
        + g3_height
        * np.exp(-((1.0 * x - g2_center - deltaxhi) ** 2) / max(tiny, (2 * g3_sigma**2)))
        + c
    )
    return f


def test_numba_accuracy():
    x = np.linspace(6517, 6618, 80)
    g1_height = random.uniform(1.0, 6.0)
    deltax = random.uniform(-17.0, -13.0)
    g1_sigma = random.uniform(1.0, 3.0)
    g2_height = random.uniform(4.0, 7.0)
    g2_center = random.uniform(6558.0, 6570.0)
    g2_sigma = random.uniform(1.0, 3.0)
    g3_height = random.uniform(0.0, 4.0)
    deltaxhi = random.uniform(19.0, 23.0)
    g3_sigma = random.uniform(1.0, 3.0)
    res1 = gaussian3CH_d(
        x,
        g1_height,
        deltax,
        g1_sigma,
        g2_height,
        g2_center,
        g2_sigma,
        g3_height,
        deltaxhi,
        g3_sigma,
    )
    res2 = numba.jit(gaussian3CH_d)(
        x,
        g1_height,
        deltax,
        g1_sigma,
        g2_height,
        g2_center,
        g2_sigma,
        g3_height,
        deltaxhi,
        g3_sigma,
    )
    assert np.array_equal(res1, res2)
