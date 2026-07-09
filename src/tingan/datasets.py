"""tingan's datasets."""

import json
from pathlib import Path

import numpy as np
import torch


class TimingNoise(torch.utils.data.Dataset):
    """
    Timing Noise dataset in PyTorch format.

    This toy dataset generates Gaussian noise, with mean and standard deviation related.
    """

    def __init__(self, size: tuple[int, ...]) -> None:
        """Initialize the dataset."""
        self.size = size
        self.std = torch.randn(self.size)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get an item from the dataset."""
        return self.std[index] ** 2 * torch.randn(64) + self.std[index]

    def __len__(self) -> tuple[int, ...]:
        """Get the length of the dataset."""
        return self.size


class RealTimingNoise(torch.utils.data.Dataset):
    """
    Real timing noise dataset in PyTorch format.

    This toy dataset is read from a real database.
    """

    def __init__(
        self,
        data_path: str = "/home/jberteaud/Science/EOS/tingan/data/real/",
        *,
        ic: bool = False,
    ) -> None:
        """Initialize the dataset."""
        min_length = np.inf
        max_length = 0
        mjds, resids = np.zeros((10000, 10000)), np.zeros((10000, 10000))
        i = 0
        for i, psr in enumerate(Path(data_path).glob("[J,B]*")):
            m, r, _ = load_residuals(str(psr / Path("residuals.npz")))
            if not (sorted(m) != m).any():
                b, nwav = load_rednoise_model(str(psr / "tempo2_fit_info.npz"))
                h, epoch = load_harmonic_series(f"{psr}/model_params.json", nwav, m)
                local_length = len(m)
                mjds[i, :local_length] = m
                resids[i, :local_length] = r - h.dot(b)
                resids[i, :local_length] -= resids[i, :local_length].mean()
                min_length = min(min_length, local_length)
                max_length = max(max_length, local_length)
        n = (10000 // min_length) * min_length
        i += 1
        mjds, resids = mjds[:i], resids[:i]
        mjds, resids = mjds[:, :n], resids[:, :n]
        mjds2, resids2 = (
            np.copy(mjds)[:, min_length // 2 : -(min_length - min_length // 2)],
            np.copy(resids)[:, min_length // 2 : -(min_length - min_length // 2)],
        )
        mjds = np.hstack((mjds, mjds2))
        resids = np.hstack((resids, resids2))
        mjds, resids = (
            mjds.reshape((len(mjds), -1, int(min_length))),
            resids.reshape((len(mjds), -1, int(min_length))),
        )
        mjds, resids = (
            mjds.reshape((-1, int(min_length))),
            resids.reshape((-1, int(min_length))),
        )
        ii = ~np.any(mjds == 0, axis=1)
        mjds = mjds[ii]
        resids = resids[ii]
        if ic:
            mjds[:, 1:] -= mjds[:, :-1].copy()  # inverse cumsum
        self.mjds_mean = mjds.mean()
        self.mjds_std = mjds.std()
        mjds = np.array((mjds - self.mjds_mean) / self.mjds_std)
        resids = np.array(resids)

        self.mjds = torch.Tensor(np.concatenate((mjds, -mjds[:, ::-1])))
        self.resids = torch.Tensor(np.concatenate((resids, resids[:, ::-1])))
        self.ic = ic

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get an item from the dataset."""
        m, r = self.mjds[index], self.resids[index]
        mr = torch.zeros((2, len(m)))
        mr[0], mr[1] = m, r
        return mr

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.resids)


def load_residuals(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load real-life timing noise residuals.

    Each pulsar file contains dates in MJD format, residuals in seconds, and uncertainty
    of residuals in seconds.
    """
    residuals = np.load(
        path
    )  # MJD, timing Residual in seconds, uncertainty of Residual in seconds
    return residuals["mjd"], residuals["residual"], residuals["error"]


def load_rednoise_model(path: str) -> tuple[np.ndarray, int]:
    """Load red noise model decomposition."""
    tempo2_fit_info = np.load(path)
    lab, val = (
        tempo2_fit_info["lab"],
        tempo2_fit_info["beta"],
    )  # Parameter labels and values

    cosidx = lab[1] == "param_red_cos"
    sinidx = lab[1] == "param_red_sin"

    # Extract the relevant parameters for the red noise variations.
    # This will be used to construct the GP model.
    beta_mod = val[np.logical_or(sinidx, cosidx)]  # red noise only

    return beta_mod, np.sum(sinidx)


def load_harmonic_series(
    path: str, nwav: int, data_mjd: np.ndarray
) -> tuple[np.ndarray, float]:
    """Load harmonic series decomposition from file and build it."""
    model_params = load_model_parameters(path)
    reference_epoch_for_gp_model_subtraction = model_params[
        "epoch"
    ]  # Reference epoch for GP model subtraction in MJD
    angular_freq_of_0th_oder_gp_mode = model_params["omega"]

    tn_red_log = model_params[
        "TNRedLog"
    ]  # Number of logarithmic components below 1/Tspan
    tn_red_log_factor = model_params[
        "TNRedLog_factor"
    ]  # Logarithmic factor for components below 1/Tspan...

    # Creating modes.
    hi_modes = angular_freq_of_0th_oder_gp_mode * np.arange(
        1, nwav + 1 - tn_red_log
    )  # linear spaced above Tspan # multiples of the 0th order frequency
    if tn_red_log > 0:
        low_modes = angular_freq_of_0th_oder_gp_mode * tn_red_log_factor ** -np.arange(
            1, tn_red_log + 1
        )  # log spaced below Tspan
        modes = np.concatenate((hi_modes, low_modes))
    else:
        modes = hi_modes

    # Create red noise model.
    harmonic_sercies_matrix_data_spacing = np.zeros(
        (2 * nwav, len(data_mjd))
    )  # data spacing, not necessarily linear

    for i, omegai in enumerate(modes):
        harmonic_sercies_matrix_data_spacing[i] = np.sin(
            omegai * (data_mjd - reference_epoch_for_gp_model_subtraction)
        )
        harmonic_sercies_matrix_data_spacing[i + nwav] = np.cos(
            omegai * (data_mjd - reference_epoch_for_gp_model_subtraction)
        )

    return harmonic_sercies_matrix_data_spacing.T, model_params["epoch"]


def load_model_parameters(path: str) -> dict:
    """Load model parameters from file."""
    with Path(path).open() as file:
        return json.load(file)
