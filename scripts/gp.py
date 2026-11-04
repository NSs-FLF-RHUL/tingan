from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.time import TimeDelta
from pint import toa
from pint.fitter import WLSFitter
from pint.logging import setup as setup_log
from pint.pulsar_mjd import Time
from pint.residuals import Residuals
from pint.simulation import make_fake_toas_fromMJDs
from pint.utils import find_optimal_nharms, plrednoise_from_wavex, wavex_setup

from tingan.datasets import get_model_from_json, load_residuals
from tingan.gp_rednoise import (
    SECONDS_PER_DAY,
    check_kernel,
    fit_residual_gp,
    sample_residual,
    simulate_noise_from_power_spectrum,
    simulate_power_spectrum,
)
from tingan.plots import plot_kernels, plot_residuals

setup_log(level="WARNING")


parser = ArgumentParser()

parser.add_argument(
    "--err", required=False, default=False, type=bool, help="Use TOAs errors in GP fit"
)
parser.add_argument(
    "--nu", required=False, default=1.5, type=float, help="Matérn smoothness parameter"
)
parser.add_argument(
    "--npred", required=False, default=200, type=int, help="Number of TOA predictions"
)
parser.add_argument(
    "--dpred", required=False, default=14, type=float, help="Days between predictions"
)
parser.add_argument(
    "--obs", required=False, default="jodrell", type=str, help="Observatory"
)
parser.add_argument("--data_path", required=True, type=str, help="Path to data files")

args = parser.parse_args()

# Get pulsar data paths
psrs = tuple(Path(args.data_path).glob("[JB]*"))

for i in range(len(psrs)):
    # Load residuals
    residuals = np.load(psrs[i] / "res.npy")

    if len(residuals) > 500:
        continue

    # Load errors
    _, _, err = load_residuals(psrs[i] / "residuals.npz")

    # Fit a Gaussian process to the residuals
    gpr, score, rm, rs = fit_residual_gp(
        t_list=residuals[:, 0],
        residual_list=residuals[:, 2],
        n_restarts=5,
        nu_mattern=args.nu,
        err_list=err if args.err else None,
    )
    check_kernel(gpr.get_params())

    if score < 0.5:
        continue

    fig = plot_kernels(gpr, score=score)
    fig.savefig(psrs[i] / "kernels.png")

    fig_real_resid, ax = plt.subplots()
    plot_residuals(residuals[:, 0], residuals[:, 2], ax, err_s=err, ls="", marker="x")
    fig_real_resid.savefig(psrs[i] / "data_residuals.png")

    # Future observation times
    mjd_pred = 61406 + np.arange(args.npred) * args.dpred
    # Past and future observation times
    mjd_all = np.concatenate((residuals[:, 0], mjd_pred)).reshape(-1, 1)

    tm = get_model_from_json(str(psrs[i]), obs=args.obs)
    power, freq = simulate_power_spectrum(
        tm.TNREDGAM.value,
        tm.TNREDAMP.value,
        nreal=1,
        npoints=args.npred,
        dt=(SECONDS_PER_DAY * (mjd_pred.max() - mjd_pred.min())) / args.npred,
    )
    noise = simulate_noise_from_power_spectrum(power, freq, nreal=1, npoints=args.npred)
    print("Model fitted to real data:")
    print(tm)

    # Create TOAs for future observation given the timing model we just created
    tm.write_parfile(psrs[i] / "real_parfile.par")
    tm.remove_component("PLRedNoise")
    toas_no_resid = make_fake_toas_fromMJDs(
        Time(mjd_pred, scale="utc", format="pulsar_mjd", precision=9),
        tm,
        obs=args.obs,
    )

    # Sample residuals from the fitted Gaussian Process
    rng1 = np.random.default_rng(0)
    resid_pred = sample_residual(t=mjd_all, residual_gp=gpr) * rs + rm

    # Add residuals to future TOAs
    fig_no_noise, ax = plt.subplots()
    plot_residuals(
        dates_mjd=toas_no_resid.get_mjds(),
        resid_s=Residuals(toas_no_resid, tm).time_resids.to(u.s),
        err_s=toas_no_resid.get_errors().to(u.s),
        ax=ax,
        fmt="x",
    )
    fig_no_noise.tight_layout()
    fig_no_noise.savefig(psrs[i] / "mock_nonoise_resid.png")
    toas_with_resid = toas_no_resid.get_mjds(high_precision=True)

    for itoa in range(len(toas_with_resid)):
        toas_with_resid[itoa] += TimeDelta(resid_pred[-args.npred :] * u.s)[itoa]
    toas_temp = np.array(
        [toas_with_resid[itoa].value for itoa in range(len(toas_with_resid))],
        dtype=np.float64,
    )
    toas_temp = Time(toas_temp, scale="utc", format="pulsar_mjd")
    toas_temp = toa.get_TOAs_array(toas_temp, obs=args.obs, errors=err[-1] * u.s)
    fig_resid, ax = plt.subplots()
    plot_residuals(
        dates_mjd=toas_temp.get_mjds(),
        resid_s=Residuals(toas_temp, tm).time_resids.to(u.s),
        err_s=toas_temp.get_errors().to(u.s),
        ax=ax,
        fmt="x",
    )
    fig_resid.tight_layout()
    fig_resid.savefig(psrs[i] / "mock_resid.png")

    for itoa in range(len(toas_with_resid)):
        toas_with_resid[itoa] += noise[0][0][itoa] * u.s
        toas_with_resid[itoa] = toas_with_resid[itoa].value
    toas_with_resid = np.array(toas_with_resid, dtype=np.float64)
    toas_with_resid = Time(toas_with_resid, scale="utc", format="pulsar_mjd")
    toas_with_resid = toa.get_TOAs_array(
        toas_with_resid, obs=args.obs, errors=err[-1] * u.s
    )
    fig_resid, ax = plt.subplots()
    plot_residuals(
        dates_mjd=toas_with_resid.get_mjds(),
        resid_s=Residuals(toas_with_resid, tm).time_resids.to(u.s),
        err_s=toas_with_resid.get_errors().to(u.s),
        ax=ax,
        fmt="x",
    )
    fig_resid.tight_layout()
    fig_resid.savefig(psrs[i] / "mock_resid_rn.png")

    # Fit timing model to future TOAs
    tm_temp = deepcopy(tm)
    print("Computing optimal number of harmonics...")
    nharm_opt, d_aics = find_optimal_nharms(
        tm_temp,
        toas_with_resid,
        "WaveX",
        30,  # tm.TNREDC.value
    )
    print("Optimal number of harmonics:", nharm_opt)

    tm_fit = deepcopy(tm_temp)
    Tspan = toas_with_resid.get_mjds().max() - toas_with_resid.get_mjds().min()
    wavex_setup(tm_fit, T_span=Tspan, n_freqs=nharm_opt, freeze_params=False)

    f = WLSFitter(toas_with_resid, tm_fit)
    f.fit_toas()
    tm_fitted = f.model

    tm_fitted_pl = plrednoise_from_wavex(tm_fitted)
    print("Model fitted to fake data:")
    print(tm_fitted_pl)
    tm_fitted.write_parfile(psrs[i] / "mockfit_parfile_wavex.par")
    tm_fitted_pl.write_parfile(psrs[i] / "mockfit_parfile_PL.par")

    fig_post_fit, ax = plt.subplots()
    plot_residuals(
        dates_mjd=toas_with_resid.get_mjds(),
        resid_s=f.resids.time_resids.to(u.s),
        err_s=toas_with_resid.get_errors().to(u.s),
        ax=ax,
        fmt="x",
    )
    fig_post_fit.tight_layout()
    fig_post_fit.savefig(psrs[i] / "mock_postfit_resid.png")

    fig_pre_fit, ax = plt.subplots()
    plot_residuals(
        dates_mjd=toas_with_resid.get_mjds(),
        resid_s=Residuals(toas_with_resid, tm_fit).time_resids.to(u.s),
        err_s=toas_with_resid.get_errors().to(u.s),
        ax=ax,
        fmt="x",
    )
    fig_pre_fit.tight_layout()
    fig_pre_fit.savefig(psrs[i] / "mock_pretfit_resid.png")

    plt.show()
