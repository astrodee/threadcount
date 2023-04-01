from abc import ABC, abstractclassmethod, abstractmethod


class CubeFitter(ABC):
    @abstractmethod
    def __init__(self, data, weight, x, model, fit_method):
        self._data = data
        self._weight = weight
        self.x = x
        self.model = model
        self.method = fit_method
        self.result = None

    @abstractclassmethod
    def fit_cube(self):
        """method to fit data cube"""


class CubeFitterMPDAF(ABC):
    @abstractmethod
    def __init__(self, cube, models, **lmfit_kwargs):
        self._cube = cube
        self.models = models
        self.lmfit_kwargs = lmfit_kwargs
        self.result = None

    @abstractclassmethod
    def fit_cube(self):
        """method to fit data cube"""
