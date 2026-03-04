"""tingan's networks."""

import torch


class Generator(torch.nn.Module):
    """
    Generator network.

    Tries to generate realistic (fake) noise.
    """

    def __init__(self, nz: int) -> None:
        """Initialize the generator."""
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(nz, 2 * nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2 * nz, 4 * nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4 * nz, 8 * nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(8 * nz, 4 * nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4 * nz, 2 * nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2 * nz, nz),
        )

    def forward(self, random_noise: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.main(random_noise)


class Discriminator(torch.nn.Module):
    """
    Discriminator network.

    Tries to discriminate between timing noise and fake noise.
    """

    def __init__(self, nz: int) -> None:
        """Initialize the discriminator."""
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(nz, 2 * nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2 * nz, 4 * nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4 * nz, 8 * nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(8 * nz, 4 * nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4 * nz, 2 * nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2 * nz, nz),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(nz, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.main(noise)
