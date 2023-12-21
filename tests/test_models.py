import pytest

import random
import numpy as np
from threadcount.models import (
    gaussianH,
    Const_2GaussModel,
    Const_2GaussModel_fast,
    Const_3GaussModel,
    Const_3GaussModel_fast,
)


@pytest.mark.parametrize(
    "model",
    [
        [Const_2GaussModel, Const_2GaussModel_fast, 2],
        [Const_3GaussModel, Const_3GaussModel_fast, 3],
    ],
)
class TestCompareModel:
    def test_compare_param_names(self, model):
        m1 = model[0]()
        m2 = model[1]()
        assert m1.param_names == m2.param_names

    def test_compare_param_hints(self, model):
        m1 = model[0]()
        m2 = model[1]()
        assert m1.param_hints == m2.param_hints

    def test_compare_guess_eval(self, model):
        m1 = model[0]()
        m2 = model[1]()
        n = model[2]
        x, data = generate_test_data(n)
        guess1 = m1.guess(data, x)
        guess2 = m2.guess(data, x)
        assert guess1 == guess2

        res1 = m1.eval(params=guess1, x=x)
        res2 = m2.eval(params=guess2, x=x)
        assert np.array_equal(res1, res2)


def generate_test_data(n):
    x = np.linspace(-10, 10, 101)
    data = np.random.normal(0, 0.2, x.size)
    for i in range(n):
        height = random.uniform(0.2, 1.2)
        center = random.uniform(-5, 5)
        sigma = random.uniform(0.3, 1.3)
        data = data + gaussianH(x, height, center, sigma)
    return x, data
