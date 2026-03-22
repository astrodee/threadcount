"""Visualise the synthetic test cube defined in tests/conftest.py.

Run from the repo root:
    python script/view_synthetic_cube.py
"""

import sys
from pathlib import Path

# Allow imports from src/ and tests/ without installing extras
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mpdaf.obj
import numpy as np
from conftest import (
    CONTINUUM,
    CUBE_NW,
    CUBE_NX,
    CUBE_NY,
    LINE_CENTER,
    LINE_HEIGHT,
    LINE_SIGMA,
    WAVE_START,
    WAVE_STEP,
)

import threadcount as tc

# ── Build the cube (same logic as the fixture) ──────────────────────────────
rng = np.random.default_rng(42)
wavelengths = WAVE_START + np.arange(CUBE_NW) * WAVE_STEP
clean_spectrum = CONTINUUM + LINE_HEIGHT * np.exp(
    -((wavelengths - LINE_CENTER) ** 2) / (2 * LINE_SIGMA**2)
)
data = np.broadcast_to(
    clean_spectrum[:, np.newaxis, np.newaxis], (CUBE_NW, CUBE_NY, CUBE_NX)
).copy()
noise_sigma = 2.0
data += rng.normal(0.0, noise_sigma, data.shape)
var = np.full_like(data, noise_sigma**2)

wave = mpdaf.obj.WaveCoord(crval=WAVE_START, cdelt=WAVE_STEP, cunit="Angstrom")
cube = mpdaf.obj.Cube(data=data, var=var, wave=wave, unit=tc.fit.FLAM16)

# ── Derived quantities ───────────────────────────────────────────────────────
# Index of the channel closest to the line centre
peak_chan = int(np.argmin(np.abs(wavelengths - LINE_CENTER)))

# Integrated flux map (sum over wavelength axis)
flux_map = cube.data.sum(axis=0)  # (NY, NX)

# Peak-flux map (max over wavelength axis)
peak_map = cube.data.max(axis=0)

# Representative spectrum: central spaxel
cy, cx = CUBE_NY // 2, CUBE_NX // 2
spaxel_spectrum = cube.data[:, cy, cx]

# ── Plot ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9))
fig.suptitle("Synthetic test cube — [O III] 5007 Å", fontsize=13, fontweight="bold")

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. Central-spaxel spectrum
ax_spec = fig.add_subplot(gs[0, :])
ax_spec.plot(wavelengths, spaxel_spectrum, lw=1, color="steelblue", label="noisy")
ax_spec.plot(
    wavelengths,
    clean_spectrum,
    lw=1.5,
    ls="--",
    color="tomato",
    label="injected (clean)",
)
ax_spec.axvline(LINE_CENTER, color="grey", lw=0.8, ls=":", label=f"λ = {LINE_CENTER} Å")
ax_spec.set_xlabel("Wavelength (Å)")
ax_spec.set_ylabel(f"Flux [{tc.fit.FLAM16}]")
ax_spec.set_title(f"Spectrum at central spaxel ({cy}, {cx})")
ax_spec.legend(fontsize=9)

# 2. Channel map at line peak
ax_chan = fig.add_subplot(gs[1, 0])
im_chan = ax_chan.imshow(cube.data[peak_chan], origin="lower", cmap="viridis")
fig.colorbar(im_chan, ax=ax_chan, shrink=0.85)
ax_chan.set_title(f"Channel map\nλ ≈ {wavelengths[peak_chan]:.2f} Å (peak channel)")
ax_chan.set_xlabel("x (spaxel)")
ax_chan.set_ylabel("y (spaxel)")

# 3. Integrated flux map
ax_flux = fig.add_subplot(gs[1, 1])
im_flux = ax_flux.imshow(flux_map, origin="lower", cmap="plasma")
fig.colorbar(im_flux, ax=ax_flux, shrink=0.85)
ax_flux.set_title("Integrated flux map\n(sum over wavelength)")
ax_flux.set_xlabel("x (spaxel)")
ax_flux.set_ylabel("y (spaxel)")

# 4. Noise estimate from variance
ax_snr = fig.add_subplot(gs[1, 2])
snr_map = peak_map / np.sqrt(var[peak_chan])
im_snr = ax_snr.imshow(snr_map, origin="lower", cmap="cividis")
fig.colorbar(im_snr, ax=ax_snr, shrink=0.85)
ax_snr.set_title("Peak-channel SNR map")
ax_snr.set_xlabel("x (spaxel)")
ax_snr.set_ylabel("y (spaxel)")

plt.show()
