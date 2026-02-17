"""tingan's datasets."""

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
