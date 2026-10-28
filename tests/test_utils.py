from pathlib import Path

import numpy as np
import pytest

from tingan import utils
from tingan._tests.fake_data import fake_concatenated_file


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


def test_split_file_at_string_nchunks() -> None:
    """Test that a file is split into the correct number of chunks."""
    path = fake_concatenated_file(3, "PSR_chunks")
    n = utils.split_file_at_string(path, "PSR")
    assert n == 3


def test_split_file_at_string_chunk_exists() -> None:
    """Test that chunked files are created."""
    path = fake_concatenated_file(2, "PSR_exists")
    utils.split_file_at_string(path, "PSR")
    sfx = path.suffix
    for i in range(2):
        assert Path(str(path)[: -len(sfx)] + f"_{i}" + sfx).exists()


def test_split_file_at_string_string_middle() -> None:
    """Test a file is split only when the string appears at the beginning of a line."""
    path = fake_concatenated_file(1, "random")
    n = utils.split_file_at_string(path, "random")
    assert n == 1


def test_updates_str_basic() -> None:
    """Test list/array conversion to string."""
    updates_per_batch = [1, 2, 3]
    updates_epoch = np.array([0, 1, 2])
    updates_per_batch_str, updates_epochs_str = utils.updates_str(
        updates_per_batch, updates_epoch
    )
    assert updates_per_batch_str == "1-2-3"
    assert updates_epochs_str == "0-1-2"


def test_updates_str_empty() -> None:
    """Test empty list/array conversion to a string."""
    updates_per_batch_str, updates_epochs_str = utils.updates_str([], [], check=False)
    assert updates_per_batch_str == ""
    assert updates_epochs_str == ""


def test_updates_str_low_train_epochs() -> None:
    """Test if error is raised when there are too few train_epochs."""
    updates_per_batch = [1, 2]
    updates_epoch = np.array([0, 2])
    train_epochs = 1
    with pytest.raises(ValueError, match="train_epochs *") as err_info:
        utils.updates_str(updates_per_batch, updates_epoch, train_epochs=train_epochs)
    assert "train_epochs should be larger than last updates_epochs." in str(
        err_info.value
    )


def test_updates_str_diff_length() -> None:
    """Test if error is raised when input lists/arrays have different lengths."""
    updates_per_batch = [1, 2, 3]
    updates_epoch = [0, 2]
    train_epochs = 5
    with pytest.raises(
        ValueError, match="updates_per_batch and updates_epochs *"
    ) as err_info:
        utils.updates_str(updates_per_batch, updates_epoch, train_epochs=train_epochs)
    assert "updates_per_batch and updates_epochs should have the same length." in str(
        err_info.value
    )


def test_updates_str_first_epoch_not_0() -> None:
    """Test if error is raised when first update epoch is not zero."""
    updates_per_batch = [1, 3]
    updates_epoch = [1, 2]
    with pytest.raises(ValueError, match="updates_epochs *") as err_info:
        utils.updates_str(updates_per_batch, updates_epoch)
    assert "updates_epochs should start from 0." in str(err_info.value)


def test_proba_to_sigma() -> None:
    """Test conversion between probabilities and standard deviations."""
    proba = np.array([0.683, 0.954, 0.9973])
    true_sigma = np.array([1, 2, 3])
    assert np.allclose(true_sigma, utils.proba_to_sigma(proba, 2), atol=1e-2)
