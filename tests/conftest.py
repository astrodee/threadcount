"""Shared pytest fixtures for threadcount tests."""

from types import SimpleNamespace

import mpdaf.obj
import numpy as np
import pytest

import threadcount as tc

# ---------------------------------------------------------------------------
# Known injection parameters — imported by any test that needs them.
# ---------------------------------------------------------------------------
CUBE_NX = 10  # spatial pixels (x)
CUBE_NY = 10  # spatial pixels (y)
CUBE_NW = 200  # wavelength channels
WAVE_START = 4990.0  # Angstroms, first channel
WAVE_STEP = 0.3  # Angstroms per channel

# Gaussian emission-line parameters injected into every spaxel
LINE_CENTER = 5006.843  # [O III] 5007 rest wavelength in Angstroms
LINE_HEIGHT = 50.0  # peak flux (in whatever cube units)
LINE_SIGMA = 1.0  # Gaussian sigma in Angstroms
CONTINUUM = 2.0  # flat continuum level


@pytest.fixture(scope="session")
def synthetic_cube():
    """Return a 10x10 x 200-wavelength mpdaf Cube with a known Gaussian line.

    Every spaxel contains an identical spectrum:
        flux(λ) = CONTINUUM + LINE_HEIGHT * exp(-(λ - LINE_CENTER)^2 / (2 * LINE_SIGMA^2))

    The WaveCoord starts at WAVE_START with step WAVE_STEP so the line sits
    well inside the bandpass.
    """
    rng = np.random.default_rng(42)

    wavelengths = WAVE_START + np.arange(CUBE_NW) * WAVE_STEP  # shape (NW,)

    # Build a single clean spectrum
    spectrum = CONTINUUM + LINE_HEIGHT * np.exp(
        -((wavelengths - LINE_CENTER) ** 2) / (2 * LINE_SIGMA**2)
    )

    # Tile across all spaxels: shape (NW, NY, NX)
    data = np.broadcast_to(
        spectrum[:, np.newaxis, np.newaxis], (CUBE_NW, CUBE_NY, CUBE_NX)
    ).copy()

    # Add Gaussian noise targeting peak SNR ≈ 25 (matches the mc_snr threshold)
    noise_sigma = 2.0
    data += rng.normal(0.0, noise_sigma, data.shape)

    # Variance array (uniform, matching the noise level)
    var = np.full_like(data, noise_sigma**2)

    # Build the mpdaf WaveCoord and Cube
    wave = mpdaf.obj.WaveCoord(crval=WAVE_START, cdelt=WAVE_STEP, cunit="Angstrom")
    cube = mpdaf.obj.Cube(data=data, var=var, wave=wave, unit=tc.fit.FLAM16)

    return cube


@pytest.fixture(scope="session")
def default_settings(synthetic_cube):
    """Return a SimpleNamespace with all fit_lines defaults, using the synthetic cube.

    This mirrors exactly what ``threadcount.procedures.fit_lines.run()`` builds
    before calling ``update_settings``, so individual unit tests can exercise
    settings processing in isolation without running the full pipeline.
    """
    default_settings_dict = {
        "setup_parameters": False,
        "monitor_pixels": [],
        "baseline_subtract": None,
        "baseline_fit_range": None,
        "output_filename": "test_output",
        "save_plots": False,
        "region_averaging_radius": 1.5,
        "instrument_dispersion": 0.8,
        "lmfit_kwargs": {"method": "least_squares"},
        "snr_lower_limit": 3,
        "lines": [tc.lines.L_OIII5007],
        "models": [[tc.models.Const_1GaussModel()]],
        "d_aic": -150,
        "interactively_choose_fits": False,
        "always_manually_choose": [],
        "mc_snr": 25,
        "mc_n_iterations": 20,
        "parallel": False,
        "n_process": 4,
        "chop_bandwidth": False,
        "SNR_HalfBW": 9,
        "SNR_Baseline_q": 0.15,
        # Cube is supplied directly so no file-loading step is needed
        "cube": synthetic_cube,
        "z_set": 0,
        "comment": "",
    }
    return SimpleNamespace(**default_settings_dict)
