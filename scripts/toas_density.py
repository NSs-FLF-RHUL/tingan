import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from tingan.datasets import load_json, load_residuals

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--data_path",
    default="/home/jberteaud/Science/EOS/tingan/data/real/",
    help="Path to data directory",
)
args = parser.parse_args()

slopes = []

for _, psr in enumerate(Path(args.data_path).glob("[J,B]*")):
    model_parameters = load_json(f"{psr}/model_params.json")
    m, r, _ = load_residuals(str(psr / Path("residuals.npz")))
    f0, f1 = model_parameters["F0"], model_parameters["F1"]
    mjd_min, mjd_max = m.min(), m.max()

    delta = (m[1:] - m[:-1]) * 86400
    time_span = (m[-1] - m[0]) * 86400
    res = linregress(m[1:], delta)
    slopes.append(res.slope)
    breaks = delta > 3 * np.median(delta)

    fig, ax = plt.subplots()  # create new figure canvas / reference
    ax.plot(m[1:], delta, ".")  # you can use ax wherever you were using plt before,
    # to apply changes to the axes referenced by ax
    ax.plot(m[1:], delta, ".")
    ax.plot(m[1:], m[1:] * res.slope + res.intercept)
    ax.axhline(3 * np.median(delta))
    for d in m[1:][breaks]:
        ax.axvline(d)

plt.show()

print(len(slopes), np.sum(np.array(slopes) > 0))
plt.hist(slopes)
plt.show()
