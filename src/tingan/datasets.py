"""tingan's datasets."""

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
