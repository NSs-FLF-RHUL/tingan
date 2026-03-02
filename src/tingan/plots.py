"""tingan's plots."""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_timing_noise(
    tin: list | np.ndarray | torch.Tensor, tin_type: str = ""
) -> plt.Figure:
    """
    Plot several examples of timing noise.

    :param tin: a set of timing noise examples.
    :param tin_type: type of timing noise.
    """
    fig, ax = plt.subplots(nrows=5, ncols=3, figsize=(10, 10))
    ax = ax.flatten()
    for i in range(15):
        ax[i].plot(tin[i])
    fig.suptitle(tin_type + " noise")
    fig.tight_layout()
    return fig


def plot_timing_noise_properties(noises: list | tuple) -> plt.Figure:
    """
    Plot properties of real and/or fake timing noise.

    Useful to assess the realism of the Generator.

    :param noises: a set of timing noises to compare.
    """
    nnoise = len(noises)

    fig = plt.figure(figsize=(10, 10))

    for n in range(nnoise):
        ax = fig.add_subplot(nnoise, 2, 2 * n + 1)
        ax.hist(
            np.mean(noises[n], axis=1),
            bins=np.linspace(0, 8, 9),
            histtype="step",
            label="mean",
        )
        ax.hist(
            np.std(noises[n], axis=1),
            bins=np.linspace(0, 8, 9),
            histtype="step",
            label="std",
        )
        ax.legend()

        ax = fig.add_subplot(nnoise, 2, 2 * n + 2)
        ax.scatter(np.mean(noises[n], axis=1), np.std(noises[n], axis=1))
        ax.set_xlabel("mean")
        ax.set_ylabel("std")

    fig.tight_layout()
    return fig


def plot_losses(
    generator_loss: list | np.ndarray | torch.Tensor,
    discriminator_loss: list | np.ndarray | torch.Tensor,
) -> plt.Figure:
    """
    Plot generator and discriminator losses.

    :param generator_loss: the generator loss.
    :param discriminator_loss: the discriminator loss.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(generator_loss, label="Generator loss")
    ax.plot(discriminator_loss, label="Discriminator loss")
    ax.legend()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    return fig
