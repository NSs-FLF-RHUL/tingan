"""
Rednoise from Gaussian process.

Use this module to extract rednoise power spectra from pulsar timing data
and simulate new ones.

"""

import contextlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
from astropy.table import QTable
from astropy.timeseries import TimeSeries
from scipy.integrate import simpson
from scipy.stats import gaussian_kde
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from timellm.data_provider.data_factory import data_provider

from tingan.datasets import toas_to_timellm_format

SECONDS_PER_DAY = 86400.0
F_1YR = 1 / (365.25 * SECONDS_PER_DAY)  # in per second


def load_gammas_and_amplitudes(psrs: tuple, *, no_f2: bool = True) -> tuple:
    """
    Load Gaussian process fit parameters from file.

    :param psrs: Pulsar folders that contain
     timing fit results (model_params.json and residuals.npz).
     :param no_f2: Wether to include F2
    """
    npsrs = len(psrs)
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
        if no_f2:
            dat_y_no_f2 = (
                dat_y - (f2 / 6.0 / f0) * ((dat_t - pepoch) * SECONDS_PER_DAY) ** 3
            )
            p = np.polyfit(dat_t, dat_y_no_f2, 2)
            dat_y_no_f2 -= np.polyval(p, dat_t)
            resid.append(dat_y_no_f2)
        else:
            resid.append(dat_y)
        time.append(dat_t)

    return gammas, amplitudes, tstart, tspans, resid, time


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
    data_x: list | np.ndarray,
    data_y: list | np.ndarray,
    size: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a 2D PDF using gaussian kernel.

    :param data_x: data to fit along first dimension.
    :param data_y: data to fit along second dimension.
    :param size: size of each output dimension.
    """
    x, y = np.meshgrid(
        np.linspace(np.min(data_x), np.max(data_x), size),
        np.linspace(np.min(data_y), np.max(data_y), size),
        indexing="ij",
    )
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
) -> list[np.ndarray]:
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


def gp_dataset_for_timellm(
    args: Namespace, *, save_npy: bool = False
) -> tuple[Namespace, list, list]:
    """
    Create red noise train and validation datasets in timellm format.

    Red noise is read from real-data files. For each pulsar, a dataset is created and
    added to a list. The function returns an updated config containing the path to the
    last pulsar data file.

    :param args: config file, which contains path to data.
    :param save_npy: whether to also save datasets in npy format.
    """
    train_data, vali_data = [], []

    psrs = tuple(Path(args.root_path).glob("[JB]*"))
    _, _, _, _, resid, tm = load_gammas_and_amplitudes(psrs)

    for p, t, r in zip(psrs, tm, resid, strict=False):
        tr = np.stack((t, r), axis=-1)[np.argsort(t)]
        mjds = toas_to_timellm_format(tr[:, 0])

        newp = p.parent / "processed" / p.name
        if save_npy:
            np.save(f"{newp}.npy", tr)

        ts = TimeSeries(
            time=mjds,
            data={"resid_s": r, "err_s": 0 * r},
        )
        ts = QTable(ts)
        ts.rename_column("time", "date")
        ts.write(f"{newp}.csv", format="csv", overwrite=True)

        if len(t) > 2 * args.seq_len:
            args.data_path = f"{newp}.csv"

            td, _ = data_provider(
                args, "train", seed=args.seed if args.use_seed else None, scale=True
            )
            vd, _ = data_provider(
                args, "val", seed=args.seed if args.use_seed else None, scale=True
            )
            train_data.append(td)
            vali_data.append(vd)

    return args, train_data, vali_data


def fit_residual_gp(  # noqa: PLR0913
    t_list: np.ndarray,
    residual_list: np.ndarray,
    nu_mattern: float = 1.5,
    n_restarts: int = 5,
    err_list: np.ndarray | None = None,
    *,
    normalize_y: bool = True,
) -> tuple[GaussianProcessRegressor, float, float, float]:
    """
    Fit a Matérn GP to the residuals.

    The Matérn kernel with smoothness nu is more flexible than RBF and
    can capture a range of spectral shapes.

    :param t_list: timestamps for each series
    :param residual_list: residuals
    :param nu_mattern: Matérn smoothness parameter
    :param n_restarts: optimizer restarts for kernel hyperparameters
    :param err_list: errors on residuals
    """
    t_all = np.reshape(t_list, (-1, 1))
    r_all = np.reshape(residual_list, (-1, 1))
    if normalize_y:
        r_mean = r_all.mean()
        r_std = r_all.std()
    else:
        r_mean = 0
        r_std = 1
    r_norm = (r_all - r_mean) / r_std
    r_norm.reshape(-1, 1)

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=10, length_scale_bounds=(1e-3, 1e3), nu=nu_mattern
    ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-3, 1e3))

    alpha = (
        np.power(np.array(err_list) / r_std, 2).reshape((-1, 1))
        if err_list is not None
        else 1e-10
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=False,
        random_state=0,
        alpha=alpha,
    )
    gp.fit(t_all, r_norm)
    score = gp.score(t_all, r_norm)

    return gp, score, r_mean, r_std


def sample_residual(
    t: np.ndarray,
    residual_gp: GaussianProcessRegressor,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Draw a single sample from the fitted residual GP on timestamps t.

    :param t: timestamps
    :param residual_gp: fitted GaussianProcessRegressor
    :param rng: numpy random Generator
    """
    if rng is None:
        rng = np.random.default_rng()

    random_state = rng.integers(0, 2**31)
    return residual_gp.sample_y(
        t.reshape(-1, 1), n_samples=1, random_state=int(random_state)
    ).squeeze()


def check_kernel(kernel_dict: dict) -> None:
    """Check kernel hyperparameters validity."""
    for param, value in kernel_dict.items():
        if f"{param}_bounds" in kernel_dict:
            low, high = kernel_dict[f"{param}_bounds"]
            if value < low or value > high:
                err_msg = f"{param}={value} is out of bounds [{low}-{high}]."
                raise ValueError(err_msg)
