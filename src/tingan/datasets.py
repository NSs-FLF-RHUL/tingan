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
        use_inverse_cumsum: bool = False,
        augment: bool = True,
    ) -> None:
        """
        Initialize the real timing dataset.

        Each dataset element is a couple of (MJDs, residuals). The neural network
        requires these elements to have the same length, so we first identify the
        lenght of the smallest dataset element and chop longer elements into
        chunks of this size. The original dataset can be augmented.
        """
        self.min_length = 10000
        self.max_length = 0
        self.ic = use_inverse_cumsum

        self.mjds, self.resids = np.zeros((10000, 10000)), np.zeros((10000, 10000))
        i_psr = 0
        for i_psr, psr in enumerate(Path(data_path).glob("[J,B]*")):
            m, r, _ = load_residuals(str(psr / Path("residuals.npz")))
            if not (sorted(m) != m).any():
                b, nwav = load_rednoise_model(str(psr / "tempo2_fit_info.npz"))
                h, epoch = load_harmonic_series(f"{psr}/model_params.json", nwav, m)
                local_length = len(m)  # data size for current pulsar
                self.mjds[i_psr, :local_length] = m  # fill MJDs
                self.resids[i_psr, :local_length] = r - h.dot(b)  # fill residuals
                self.resids[i_psr, :local_length] -= self.resids[
                    i_psr, :local_length
                ].mean()  # subtact mean
                self.min_length = min(
                    self.min_length, local_length
                )  # update min_length
                self.max_length = max(
                    self.max_length, local_length
                )  # update max_length

        # Largest integer multiple of min_length smaller than 10000
        n = int((10000 // self.min_length) * self.min_length)

        # Truncate arrays along first dimension at last pulsar
        self.mjds, self.resids = self.mjds[: i_psr + 1], self.resids[: i_psr + 1]
        # Truncate arrays along second dimension at largest multiple of min_length
        self.mjds, self.resids = self.mjds[:, :n], self.resids[:, :n]
        if augment:
            self.mjds, self.resids = self.shift_and_merge()
        # Divide arrays in smaller chunks of size min_length
        self.mjds, self.resids = (
            self.mjds.reshape((len(self.mjds), -1, int(self.min_length))),
            self.resids.reshape((len(self.mjds), -1, int(self.min_length))),
        )
        self.mjds, self.resids = (
            self.mjds.reshape((-1, int(self.min_length))),
            self.resids.reshape((-1, int(self.min_length))),
        )

        self.exclude_chunks_with_zeros()

        if use_inverse_cumsum:
            self.mjds[:, 1:] -= self.mjds[:, :-1].copy()
        self.mjds = np.array((self.mjds - self.mjds_mean) / self.mjds_std)
        self.resids = np.array(self.resids)

        if augment:
            self.mjds, self.resids = self.reverse_and_merge()

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get an item from the dataset."""
        m, r = self.mjds[index], self.resids[index]
        mr = torch.zeros((2, len(m)))
        mr[0], mr[1] = m, r
        return mr

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.resids)

    def exclude_chunks_with_zeros(self) -> None:
        """Exclude the shortest time-series."""
        i_zeros = ~np.any(self.mjds == 0, axis=1)
        self.mjds = self.mjds[i_zeros]
        self.resids = self.resids[i_zeros]

    def shift_and_merge(self) -> tuple[np.ndarray, np.ndarray]:
        """Shift and merge the dataset."""
        mjds2, resids2 = (
            np.copy(self.mjds)[
                :, self.min_length // 2 : -(self.min_length - self.min_length // 2)
            ],
            np.copy(self.resids)[
                :, self.min_length // 2 : -(self.min_length - self.min_length // 2)
            ],
        )
        return np.hstack((self.mjds, mjds2)), np.hstack((self.resids, resids2))

    def reverse_and_merge(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reverse and merge the dataset."""
        return torch.Tensor(
            np.concatenate((self.mjds, -self.mjds[:, ::-1]))
        ), torch.Tensor(np.concatenate((self.resids, self.resids[:, ::-1])))

    @property
    def mjds_mean(self) -> float:
        """Compute the mean of the MJDs of the dataset."""
        return float(self.mjds.mean())

    @property
    def mjds_std(self) -> float:
        """Compute the standard deviation of the MJDs of the dataset."""
        return float(self.mjds.std())


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
    """
    Load red noise model decomposition.

    param path: path to red noise model
    return: Fourier coefficients and number of components of the red noise model,
    """
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
    model_params = load_json(path)
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


def load_json(path: str) -> dict:
    """Load JSON file."""
    with Path(path).open() as file:
        return json.load(file)
