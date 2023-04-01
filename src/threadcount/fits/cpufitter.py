__all__ = ["CubeFitterMPDAFLM"]

import logging
import math
from multiprocessing.pool import ThreadPool as Pool
import os
from functools import partial
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
from typing import List

import numpy as np
from threadcount import mpdaf_ext

# from tqdm import tqdm

from .base import CubeFitterMPDAF

logger = logging.getLogger(__name__)

# CPU COUNT
CPU_COUNT = os.cpu_count()
logger.info(f"The number of cpus: {CPU_COUNT}")


class CubeFitterMPDAFLM(CubeFitterMPDAF):
    """use extensions methods in mpdaf_ext.py"""

    def __init__(
        self,
        cube,
        spaxel_iterator,
        models,
        snr_image,
        snr_threshold,
        nprocess=CPU_COUNT,
        **lmfit_kwargs,
    ):
        """**kwargs: refer to lmfit.model.fit"""
        self._cube = cube
        self.iterator = spaxel_iterator
        self.models = models
        self.nprocess = nprocess if isinstance(nprocess, int) else CPU_COUNT
        self.snr = snr_image
        self.snr_threshold = snr_threshold
        self.lmfit_kwargs = lmfit_kwargs
        self.result = None
        self.fit_results_T = None
        self._create_result_container()

    def _create_result_container(self):
        """create result array with nan value"""
        spatial_shape = self._cube.shape[1:]
        self.result = np.array(
            [np.full(spatial_shape, None, dtype=object)] * len(self.models)
        )
        # transpose fit_results for easy addressing by spaxel indices inside loop:
        self.fit_results_T = self.result.transpose((1, 2, 0))

    def _fit_single_spaxel(
        self,
        cube: mpdaf_ext.mpdaf.obj.Cube,
        snr_image: np.ndarray,
        snr_threshold: float,
        models: List,
        lmfit_kwargs: dict,
        resm: shared_memory.SharedMemory,
        idx: List[int],
    ):

        # Test if it passes the SNR test:
        if snr_image[idx] < snr_threshold:
            return

        sp = cube[(slice(None), *idx)]

        # Fit the least complex model, and make sure of success.
        spec_to_fit = sp
        f = spec_to_fit.lmfit(models[0], **lmfit_kwargs)
        if f is None:
            return

        if f.success is False:
            # One reason we saw for 1 gaussian fit to fail includes the iron line when
            # fitting 5007. Therefore, if there is a failure to fit 1 gaussian, I will
            # cut down the x axis by 5AA on each side and try again.
            wave_range = sp.get_range()
            logger.info("cutting spectrum by +/- 5A for pixel {}".format(idx))
            cut_sp = sp.subspec(wave_range[0] + 5, wave_range[1] - 5)
            spec_to_fit = cut_sp
            f = spec_to_fit.lmfit(models[0], **lmfit_kwargs)
            if f.success is False:
                return

        # at this point: if the first model has failed to fit both times, we don't
        # even reach this point, the loop continues. However, if the first model
        # fit the first time, then spec_to_fit = sp. If the first model failed the
        # first time and succeeded the second time, then spec_to_fit = cut_sp.

        # continue with the rest of the models.
        rest = [spec_to_fit.lmfit(model, **lmfit_kwargs) for model in models[1:]]
        out = [f] + rest
        # read fitting result
        result = np.ndarray(
            self.fit_results_T.shape, self.fit_results_T.dtype, buffer=resm.buf
        )

        result[idx] = out

        # display fitting process
        name = os.getpid()
        logger.info(f"subprocess: {name}; pixel: {idx}")

    def _set_default_chunksize(self, ncpu):
        return math.ceil(self._cube.shape[1] / ncpu)

    def fit_cube(self, nprocess=None, chunksize=None):
        """Fit data cube parallelly"""
        if nprocess is None:
            nprocess = self.nprocess
        if chunksize is None:
            chunksize = self._set_default_chunksize(nprocess)

        # initialize result back to none's.
        self._create_result_container()

        with SharedMemoryManager() as smm:
            logger.debug("Put data into shared memory")
            shm_r = smm.SharedMemory(size=self.fit_results_T.nbytes)
            sr = np.ndarray(
                self.fit_results_T.shape,
                dtype=self.fit_results_T.dtype,
                buffer=shm_r.buf,
            )
            sr[:] = self.fit_results_T[:]

            logger.info("Start pooling ...")
            with Pool(processes=nprocess) as pool:
                pool.map(
                    partial(
                        self._fit_single_spaxel,
                        self._cube,
                        self.snr,
                        self.snr_threshold,
                        self.models,
                        self.lmfit_kwargs,
                        shm_r,
                    ),
                    self.iterator,
                    chunksize=chunksize,
                )
                logger.info("Finish pooling.")

        self.fit_results_T[:] = sr[:]
