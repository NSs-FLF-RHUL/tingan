import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tingan.gp_rednoise import (
    SECONDS_PER_DAY,
    gaussian_kde_1d,
    gaussian_kde_2d,
    load_gammas_and_amplitudes,
    marginalize_2d_kde,
    simulate_noise_from_power_spectrum,
    simulate_power_spectrum,
)
from tingan.utils import bin_min_max, gaussian_dist, load_latex_table

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--plot", default=True, help="Create plots when running script."
)
parser.add_argument(
    "-n",
    "--nsim",
    type=int,
    nargs=1,
    default=(100,),
    help="Number of simulations to run.",
)
parser.add_argument(
    "-s", "--save", type=str, nargs=1, help="Save location for generated data."
)
parser.add_argument(
    "-d",
    "--data",
    type=str,
    nargs=1,
    default=("/home/jberteaud/Science/EOS/tingan/data/real/",),
    help="Read location for real data.",
)
parser.add_argument(
    "-ds",
    "--dataset",
    type=str,
    nargs=1,
    default=("keith",),
    help="Dataset to used (keith, arthasarathy, both)",
)
args = parser.parse_args()
# If you want to then re-scope the original variables you were using
save_location = Path(args.save[0]) if args.save else None
nsim = args.nsim[0]
plot = args.plot

t1, e1 = load_latex_table(args.data[0] + "table1.txt")
t3, e3 = load_latex_table(args.data[0] + "table3.txt")

psr_names, idx1, idx3 = np.intersect1d(t1[:, 0], t3[:, 0], return_indices=True)
t1, t3 = t1[idx1, :], t3[idx3, :]

amp_arthasarathy = t3[:, 3].astype(float)
gam_arthasarathy = -t3[:, 4].astype(float)  # use same definition for spectral index
tspan_arthasarathy = 365.25 * t1[:, 7].astype(float)  # convert froms years to days

igood_arthasarathy = ~np.isnan(amp_arthasarathy)
amp_arthasarathy, gam_arthasarathy, tspan_arthasarathy = (
    amp_arthasarathy[igood_arthasarathy],
    gam_arthasarathy[igood_arthasarathy],
    tspan_arthasarathy[igood_arthasarathy],
)

psrs = tuple(Path(args.data[0]).glob("[JB]*"))
gammas_keith, amplitudes_keith, tstart, tspans_keith, resid, time = (
    load_gammas_and_amplitudes(
        psrs,
    )
)

if args.dataset[0] == "both":
    gammas = np.concatenate((gammas_keith, gam_arthasarathy))
    amplitudes = np.concatenate((amplitudes_keith, amp_arthasarathy))
    tspans = np.concatenate((tspans_keith, tspan_arthasarathy))
elif args.dataset[0] == "keith":
    gammas = gammas_keith
    amplitudes = amplitudes_keith
    tspans = tspans_keith
else:
    gammas = gam_arthasarathy
    amplitudes = amp_arthasarathy
    tspans = tspan_arthasarathy

kde_2d, x, y = gaussian_kde_2d(gammas, amplitudes)
kde_gammas = marginalize_2d_kde(kde_2d, 0, x[:, 0])
kde_amplitudes = marginalize_2d_kde(kde_2d, 1, y[0, :])
gauss_gammas = gaussian_dist(gammas).pdf(x[:, 0])
gauss_amplitudes = gaussian_dist(amplitudes).pdf(y[0, :])

bin_g = bin_min_max((gammas,), nbins=5)
bin_a = bin_min_max((amplitudes,), nbins=5)

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
    ax.plot(gam_arthasarathy, amp_arthasarathy, ".", color="tab:purple")
    ax.plot(gammas_keith, amplitudes_keith, ".", color="tab:green")
    ax.set_xlim([np.min(gammas), np.max(gammas)])
    ax.set_ylim([np.min(amplitudes), np.max(amplitudes)])
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"Amplitudes")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_aspect("auto")

    ax = axes[1]
    ax.hist(
        gammas,
        bins=bin_g,
        density=True,
        label="Samples",
        color="tab:orange",
        alpha=0.5,
    )
    ax.hist(
        gam_arthasarathy,
        bins=bin_g,
        density=True,
        label="Arthasarathy",
        color="tab:purple",
        histtype="step",
        lw=2.0,
    )
    ax.hist(
        gammas_keith,
        bins=bin_g,
        density=True,
        label="Keith",
        color="tab:green",
        histtype="step",
        lw=2.0,
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
        bins=bin_a,
        density=True,
        orientation="horizontal",
        color="tab:orange",
        alpha=0.5,
    )
    ax.hist(
        amp_arthasarathy,
        bins=bin_a,
        density=True,
        label="Samples",
        color="tab:purple",
        histtype="step",
        lw=2.0,
        orientation="horizontal",
    )
    ax.hist(
        amplitudes_keith,
        bins=bin_a,
        density=True,
        label="Samples",
        color="tab:green",
        histtype="step",
        lw=2.0,
        orientation="horizontal",
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
gammas_sim = gaussian_dist(gammas).rvs(nsim)
amplitudes_sim = gaussian_dist(amplitudes).rvs(nsim)
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

if save_location is not None or plot:
    for i in range(nsim):
        t = tstart_sim[i] + np.linspace(0, tspans_sim[i], 1024)
        for j in range(100):
            s = noise[i][j, :]
            p = np.polyfit(t, s, 2)
            s -= np.polyval(p, t)
            res[i, j, :1024] = t
            res[i, j, 1024:] = s

    if save_location is not None:
        np.save(save_location, res.reshape((100 * nsim, 2 * 1024)))

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
        for i in range(len(time)):
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
