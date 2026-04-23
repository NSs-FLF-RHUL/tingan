"""Some useful functions for data manipulation and analysis."""

import numpy as np


def load_latex_table(texfile: str) -> tuple:
    """
    Load data table from file in LaTeX format.

    param texfile: path to latex table file
    return: parameter values and (min-max)
    """
    t = np.loadtxt(texfile, dtype=object, delimiter="&", skiprows=3)
    tshape = t.shape
    rows = tshape[0]
    cols = tshape[1]
    e = np.zeros((rows, cols, 2))
    for i in range(rows):
        for j in range(cols):
            tij = (
                t[i, j]
                .replace(" ", "")
                .replace("\\", "")
                .replace("{", "")
                .replace("}", "")
                .replace("$", "")
            )
            if "(" in tij and ")" in tij and ":" not in tij:
                if tij[0] != "(":
                    tij = tij.split("(")
                    eij = tij[1].split(")")[0]
                    t[i, j] = float(tij[0])
                    e[i, j, :] = eij
                else:
                    tij = tij[1:-1].split(",")
                    t[i, j] = np.nan
                    e[i, j, :] = tij
            elif "^" in tij and "_" in tij and ":" not in tij:
                tij = tij.split("^")
                t[i, j] = float(tij[0])
                tij = tij[1].split("_")
                e[i, j, :] = t[i, j] - np.array(tij[::-1], dtype=float)
            elif tij == "NA":
                t[i, j] = np.nan
            else:
                t[i, j] = tij

    return t, e


def bin_min_max(arrays: tuple, nbins: int = 10) -> np.ndarray:
    """
    Create a regular binning that spans a set of arrays.

    param arrays: list of arrays
    param nbins: number of bins

    return: bin edges
    """
    array = np.concatenate(arrays).flatten()
    return np.linspace(np.nanmin(array), np.nanmax(array), nbins + 1)
