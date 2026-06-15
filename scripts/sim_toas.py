import argparse
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from pint.models import get_model
from pint.residuals import Residuals
from pint.simulation import make_fake_toas_uniform
from scipy.integrate import simpson
from scipy.stats import ks_2samp

from tingan.gp_rednoise import gaussian_kde_1d

rng = np.random.default_rng()

parser = argparse.ArgumentParser()
parser.add_argument(
    "-pe", default=61206.0, help="Pulsar spin period reference epoch [MJD]."
)
parser.add_argument("-p0", default=1.0, help="Pulsar spin period [s].")
parser.add_argument("-p1", default=-1e-16, help="Pulsar spin period derivative [s/s].")
parser.add_argument(
    "-p2", default=-1e-30, help="Pulsar spin period second derivative [s/s**2]."
)
parser.add_argument("-amp", default=-12.0, help="Red noise amplitude.")
parser.add_argument("-gam", default=5.0, help="Red noise index.")
parser.add_argument("-psr", default="B0331+45", help="Pulsar noise to mimic.")
args = parser.parse_args()

par_sim = f"""
    PSR           MOCK
    PEPOCH        {args.pe}
    F0            {args.p0}     1
    F1            {args.p1}     1
    F2            {args.p2}     1
    TNREDAMP      {args.amp}
    TNREDGAM      {args.gam}
    TNREDC        400
    UNITS         TDB
"""

m = get_model(StringIO(par_sim))

par_sim = f"""
    PSR           MOCK
    PEPOCH        {args.pe}
    F0            {args.p0}     1
    F1            {args.p1}     1
    F2            {args.p2}     1
    TNREDAMP      {-10.0}
    TNREDGAM      {args.gam}
    TNREDC        400
    UNITS         TDB
"""

m_no_tn = get_model(StringIO(par_sim))

resid = np.load(f"../data/real/{args.psr}/residuals.npz")

errors_sim = np.sort(rng.uniform(resid["error"].min(), resid["error"].max(), int(1e5)))
errors_sim_pdf = gaussian_kde_1d(resid["error"], size=None)(errors_sim)
errors_sim_pdf /= simpson(errors_sim_pdf, errors_sim)
errors_sim_mc = rng.uniform(0, errors_sim_pdf.max(), errors_sim.size)
plt.hist(resid["error"], density=True, bins=30, label=f"{args.psr} errors")
plt.plot(errors_sim, errors_sim_pdf, label=f"KDE fit to {args.psr} errors")
errors_sim = errors_sim[errors_sim_mc < errors_sim_pdf]
rng.shuffle(errors_sim)
nsim = len(errors_sim) // len(resid["error"])
errors_sim = errors_sim[: nsim * len(resid["error"])].reshape(
    (nsim, len(resid["error"]))
)
pv = []
for i in range(nsim):
    plt.hist(
        errors_sim[i],
        density=True,
        bins=30,
        alpha=0.1,
        color="tab:grey",
        label=f"Simulations ({nsim})" if i == 0 else None,
    )
    pv.append(ks_2samp(errors_sim[i], resid["error"]).pvalue)
errors_sim = errors_sim[np.argmax(pv)]
plt.hist(
    errors_sim,
    density=True,
    bins=30,
    histtype="step",
    label="Best simulation",
    color="tab:green",
    lw=2,
)
plt.xlabel("Residual error [s]")
plt.ylabel("PDF")
plt.legend()
plt.show()

plt.hist(pv, bins=np.logspace(-6, 0, 13), label="Simulations")
plt.xlabel("KS-test p-value")
plt.ylabel("N")
plt.axvline(np.max(pv), ls=":", label="Maximum p-value", color="tab:green")
plt.xscale("log")
plt.legend()
plt.show()

t1 = make_fake_toas_uniform(
    resid["mjd"].min(),
    resid["mjd"].max(),
    len(errors_sim),
    m,
    error=errors_sim * u.s,
    add_noise=True,
    add_correlated_noise=True,
)

plt.figure(figsize=(10, 6))
plt.title(f"{args.psr}")
plt.xlabel("MJD")
plt.ylabel("Residual (seconds)")
plt.grid(visible=True)

for mm in [m, m_no_tn]:
    print(mm)
    rs = Residuals(t1, mm)
    mjd_times = t1.get_mjds()
    time_residuals = rs.time_resids.value  # Gets residuals in seconds
    print(time_residuals)
    plt.errorbar(mjd_times, time_residuals, yerr=t1.get_errors().to("s").value, fmt="x")

plt.show()

t1.compute_pulse_numbers(m)
x = t1.get_pulse_numbers()
print(x[:3])

t1.compute_pulse_numbers(m_no_tn)
y = t1.get_pulse_numbers()
print(y[:3])

plt.hist(x - y)
plt.show()
