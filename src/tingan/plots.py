import matplotlib.pyplot as plt
import numpy as np


def plot_timing_noise(tin, tin_type=""):
    plt.figure(figsize=(10, 10))
    for i in range(15):
        plt.subplot(5, 3, i + 1)
        plt.plot(tin[i])
    plt.suptitle(tin_type + " noise")
    plt.tight_layout()
    plt.show()


def plot_timing_noise_properties(noises):
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


def plot_losses(generator_loss, discriminator_loss):
    plt.figure(figsize=(10, 10))
    plt.plot(generator_loss, label="Generator loss")
    plt.plot(discriminator_loss, label="Discriminator loss")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()
