import numpy as np

from tingan import utils


def test_bin_min_max_standard() -> None:
    """Test that bin edges are correct."""
    array1 = [1, 2, 3, 4, 5]
    array2 = [6, 7, 8, 9]
    bin_edges = utils.bin_min_max((array1, array2))
    assert bin_edges[0] == array1[0]
    assert bin_edges[-1] == array2[-1]


def test_bin_min_max_with_nans() -> None:
    """Test that NaNs are exlucded from binning."""
    array1 = [1, 2, 3, np.nan, 5]
    array2 = [6, np.nan, 8, 9]
    bin_edges = utils.bin_min_max((array1, array2))
    assert bin_edges[0] == array1[0]
    assert bin_edges[-1] == array2[-1]


def test_gaussian_dist_basic() -> None:
    """Test that data statistics are correctly extracted."""
    data = 10 * [1]
    dist = utils.gaussian_dist(data)
    assert np.isclose(dist.rvs(), data[0])


def test_laplace_dist_basic() -> None:
    """Test that data statistics are correctly extracted."""
    data = 10 * [1]
    dist = utils.laplace_dist(data)
    assert np.isclose(dist.rvs(), data[0])
