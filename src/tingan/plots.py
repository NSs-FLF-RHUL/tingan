"""tingan's plots."""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_timing_noise(
    tin: list | np.ndarray | torch.Tensor, tin_type: str = ""
) -> None:
    """
    Plot several examples of timing noise.

    :param tin: a set of timing noise examples.
    :param tin_type: type of timing noise.
    """
    plt.figure(figsize=(10, 10))
    for i in range(15):
        plt.subplot(5, 3, i + 1)
        plt.plot(tin[i])
    plt.suptitle(tin_type + " noise")
    plt.tight_layout()
    plt.show()


def plot_timing_noise_properties(noises: list | tuple) -> None:
    """
    Plot properties of real and/or fake timing noise.

    Useful to assess the realism of the Generator.

    :param noises: a set of timing noises to compare.
    """
    nnoise = len(noises)

    plt.figure(figsize=(10, 10))

    for n in range(nnoise):
        plt.subplot(nnoise, 2, 2 * n + 1)
        plt.hist(
            np.mean(noises[n], axis=1),
            bins=np.linspace(0, 8, 9),
            histtype="step",
            label="mean",
        )
        plt.hist(
            np.std(noises[n], axis=1),
            bins=np.linspace(0, 8, 9),
            histtype="step",
            label="std",
        )
        plt.legend()

        plt.subplot(nnoise, 2, 2 * n + 2)
        plt.scatter(np.mean(noises[n], axis=1), np.std(noises[n], axis=1))
        plt.xlabel("mean")
        plt.ylabel("std")

    plt.tight_layout()
    plt.show()


def plot_losses(
    generator_loss: list | np.ndarray | torch.Tensor,
    discriminator_loss: list | np.ndarray | torch.Tensor,
) -> None:
    """
    Plot generator and discriminator losses.

    :param generator_loss: the generator loss.
    :param discriminator_loss: the discriminator loss.
    """
    plt.figure(figsize=(10, 10))
    plt.plot(generator_loss, label="Generator loss")
    plt.plot(discriminator_loss, label="Discriminator loss")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()
