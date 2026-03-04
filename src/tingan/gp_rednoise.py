"""
Rednoise from Gaussian process.

Use this module to extract rednoise power spectra from pulsar timing data
and simulate new ones.

"""

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from scipy.stats import _continuous_distns, gaussian_kde, norm

SECONDS_PER_DAY = 86400.0
F_1YR = 1 / (365.25 * SECONDS_PER_DAY)  # in per second


def load_gammas_and_amplitudes(psrs: Iterator[Path]) -> tuple:
    """
    Load Gaussian process fit parameters from file.

    :param psrs: Pulsar folders that contain
     timing fit results (model_params.json and residuals.npz).
    """
    npsrs = psrs.__sizeof__()
    gammas, amplitudes, tstart, tspans = (
        np.zeros(npsrs),
        np.zeros(npsrs),
        np.zeros(npsrs),
        np.zeros(npsrs),
    )
    resid, time = [], []
    for i, psr in enumerate(psrs):
        psr_model_path = Path(f"{psr}/model_params.json")
        model_params = json.load(psr_model_path.open())
        gammas[i] = model_params["TNRedGam"]
        amplitudes[i] = model_params["TNRedAmp"]
        residuals = np.load(f"{psr}/residuals.npz")
        dat_t = residuals["mjd"]  # MJD
        tstart[i] = dat_t.min()
        tspans[i] = dat_t.max() - tstart[i]

        dat_y = residuals["residual"]  # in seconds
        p = np.polyfit(dat_t, dat_y, 2)
        dat_y -= np.polyval(p, dat_t)
        f2 = model_params["F2"]  # in seconds per second^2
        f0 = model_params["F0"]  # in seconds per second
        pepoch = model_params["pepoch"]  # in MJD
        dat_y_no_f2 = (
            dat_y - (f2 / 6.0 / f0) * ((dat_t - pepoch) * SECONDS_PER_DAY) ** 3
        )
        p = np.polyfit(dat_t, dat_y_no_f2, 2)
        dat_y_no_f2 -= np.polyval(p, dat_t)
        resid.append(dat_y_no_f2)
        time.append(dat_t)
    i += 1
    return gammas[:i], amplitudes[:i], tstart[:i], tspans[:i], resid[:i], time[:i]


def gaussian_kde_1d(data: list | np.ndarray, size: int = 100) -> np.ndarray:
    """
    Fit a 1D PDF using gaussian kernel.

    :param data: data to fit.
    :param size: number of points to evaluate the PDF at.
    """
    x = np.linspace(np.min(data), np.max(data), size)
    kernel = gaussian_kde(data)
    return kernel(x)


def gaussian_kde_2d(
    data_x: list | np.ndarray, data_y: list | np.ndarray, size: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a 2D PDF using gaussian kernel.

    :param data_x: data to fit along first dimension.
    :param data_y: data to fit along second dimension.
    :param size: number of points to evaluate the PDF at, per dimension.
    """
    x, y = np.mgrid[
        np.min(data_x) : np.max(data_x) : size, np.min(data_y) : np.max(data_y) : size
    ]
    positions = np.vstack([x.ravel(), y.ravel()])
    data = np.vstack([data_x, data_y])
    kernel = gaussian_kde(data)
    z = np.reshape(kernel(positions).T, x.shape)
    return z, x, y


def marginalize_2d_kde(kde: np.ndarray, dim: int, data: np.ndarray) -> np.ndarray:
    """
    Marginalize 2D PDF.

    :param kde: evaluation of the 2D PDF on a grid.
    :param dim: dimension to marginalize.
    :param data: data to marginalize.
    """
    kde_pdf = np.sum(np.rot90(kde), axis=dim)
    kde_pdf /= simpson(kde_pdf, data)
    return kde_pdf


def gaussian_dist(data: list | np.ndarray) -> _continuous_distns:
    """
    Return a Gaussian distribution with same mean and standard deviation as data.

    :param data: data to mimic.
    """
    return norm(loc=np.mean(data), scale=np.std(data))


def simulate_power_spectrum(
    gammas: np.ndarray | list,
    amplitudes: list | np.ndarray,
    nreal: int = 100,
    npoints: int = 1024,
    dt: float | np.ndarray = 1800000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate red noise power spectra.

    :param gammas: indices of the power spectrum.
    :param amplitudes: amplitudes of the power spectrum in log10.
    :param nreal: number of realizations.
    :param npoints: number of points per realization.
    :param dt: time step in seconds.
    """
    try:
        dt = np.ndarray([dt]).reshape(1, -1)
    except TypeError:
        contextlib.suppress(TypeError)
    gammas = np.array(gammas)
    n = nreal * npoints
    freq = np.array(np.fft.rfftfreq(n)).reshape(-1, 1) / dt
    freq = np.array(freq)
    df = freq[1] - freq[0]
    power = (
        ((np.power(10, amplitudes)) ** 2)
        / 12.0
        / np.pi**2
        * np.power(F_1YR, gammas - 3)
        * np.power(freq, -gammas)
        * df
    ) / 4.0
    power[0] = 0
    return power, freq


def simulate_noise_from_power_spectrum(
    power: list | np.ndarray,
    freq: list | np.ndarray,
    nreal: int = 100,
    npoints: int = 1024,
) -> list:
    """
    Simulate noise from power spectra through a Gaussian process.

    :param power: powers of the power spectrum.
    :param freq: frequencies of the power spectrum.
    :param nreal: number of realizations.
    :param npoints: number of points per realization.
    """
    rng = np.random.default_rng()
    w = rng.normal(0, 1, size=len(freq)) + 1j * rng.normal(
        0, 1, size=len(freq)
    )  # Complex white noise.
    spectrum = w.reshape(-1, 1) * np.sqrt(power)
    signals = []
    for i in range(spectrum.shape[1]):
        signal = np.fft.irfft(spectrum[:, i], norm="forward").reshape(nreal, npoints)
        signals.append(signal)
    return signals
