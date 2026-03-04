from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tingan.gp_rednoise import (
    SECONDS_PER_DAY,
    gaussian_kde_1d,
    gaussian_kde_2d,
    gaussian_pdf,
    load_gammas_and_amplitudes,
    marginalize_2d_kde,
    simulate_noise_from_power_spectrum,
    simulate_power_spectrum,
)

save = False
plot = True
nsim = 100

psrs = Path("/home/jberteaud/Science/EOS/tingan/data/real/").glob("[JB]*")
gammas, amplitudes, tstart, tspans, resid, time = load_gammas_and_amplitudes(
    psrs,
)
kde_2d, x, y = gaussian_kde_2d(gammas, amplitudes)
kde_gammas = marginalize_2d_kde(kde_2d, 0, x[:, 0])
kde_amplitudes = marginalize_2d_kde(kde_2d, 1, y[0, :])
gauss_gammas = gaussian_pdf(gammas).pdf(x[:, 0])
gauss_amplitudes = gaussian_pdf(amplitudes).pdf(y[0, :])

if plot:
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.ravel()

    ax = axes[3]
    ax.imshow(
        np.rot90(kde_2d),
        extent=[np.min(gammas), np.max(gammas), np.min(amplitudes), np.max(amplitudes)],
        cmap="Blues",
    )
    ax.plot(gammas, amplitudes, "o", color="tab:orange")
    ax.set_xlim([np.min(gammas), np.max(gammas)])
    ax.set_ylim([np.min(amplitudes), np.max(amplitudes)])
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"Amplitudes")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_aspect("auto")

    ax = axes[1]
    ax.hist(
        gammas, bins=5, density=True, label="Samples", color="tab:orange", alpha=0.5
    )
    ax.plot(x[:, 0], kde_gammas, label="Marginalized 2D KDE", color="tab:blue", ls="--")
    ax.plot(x[:, 0], gaussian_kde_1d(gammas), label="1D KDE", color="tab:blue", ls=":")
    ax.plot(
        x[:, 0],
        gauss_gammas,
        label=r"N($\mu_\mathrm{samp}$,$\sigma_\mathrm{samp}$)",
        color="tab:blue",
    )
    ax.set_ylabel("PDF")
    ax.legend()

    ax = axes[2]
    ax.hist(
        amplitudes,
        bins=5,
        density=True,
        orientation="horizontal",
        color="tab:orange",
        alpha=0.5,
    )
    ax.plot(kde_amplitudes, y[0, ::-1], color="tab:blue", ls="--")
    ax.plot(gaussian_kde_1d(amplitudes), y[0, :], color="tab:blue", ls=":")
    ax.plot(gauss_amplitudes, y[0, :], color="tab:blue")
    ax.set_xlabel("PDF")
    xmin, xmax = ax.get_xlim()
    ax.set_xlim([xmax, xmin])

    fig.delaxes(axes[0])

    plt.tight_layout()
    plt.show()

rng = np.random.default_rng()
gammas_sim = gaussian_pdf(gammas).rvs(nsim)
amplitudes_sim = gaussian_pdf(amplitudes).rvs(nsim)
tstart_sim = rng.uniform(tstart.min(), tstart.max(), nsim)
tspans_sim = rng.uniform(tspans.min(), tspans.max(), nsim)

power, freq = simulate_power_spectrum(
    gammas_sim, amplitudes_sim, dt=(SECONDS_PER_DAY * tspans_sim) / 1024
)
noise = simulate_noise_from_power_spectrum(power, freq)

power_data, freq_data = simulate_power_spectrum(
    gammas, amplitudes, dt=(SECONDS_PER_DAY * tspans) / 1024
)

res = np.zeros((nsim, 100, 2 * 1024))

if save or plot:
    for i in range(nsim):
        t = tstart_sim[i] + np.linspace(0, tspans_sim[i], 1024)
        for j in range(100):
            s = noise[i][j, :]
            p = np.polyfit(t, s, 2)
            s -= np.polyval(p, t)
            res[i, j, :1024] = t
            res[i, j, 1024:] = s

    if save:
        np.save(
            "/home/jberteaud/Science/EOS/tingan/data/simulated/gp_simulated_rednoise.npy",
            res.reshape((100 * nsim, 2 * 1024)),
        )

    if plot:
        for i in range(min(nsim, 500)):
            plt.loglog(
                freq[:, i],
                power[:, i],
                alpha=0.1,
                color="tab:blue",
                label="Simulated" if i == 0 else None,
            )
        for i in range(power_data.shape[1]):
            plt.loglog(
                freq_data[:, i],
                power_data[:, i],
                color="tab:orange",
                label="Fitted to data" if i == 0 else None,
            )
        plt.title("Red noise power spectra")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.legend(loc=1)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))

        for i in range(min(nsim, 100)):
            plt.plot(
                res[i, 0, :1024],
                res[i, 0, 1024:],
                alpha=0.5,
                color="tab:blue",
                label="Simulated" if i == 0 else None,
            )
        for i in range(len(gammas)):
            plt.plot(
                time[i],
                resid[i],
                ".",
                color="tab:orange",
                label="Fitted to data" if i == 0 else None,
            )
        plt.legend()
        plt.ylim(-0.3, 0.3)
        plt.xlabel("MJD")
        plt.ylabel("Timing residual (s)")
        plt.title("Random draws from the power-law GP model")
        plt.tight_layout()
        plt.show()
