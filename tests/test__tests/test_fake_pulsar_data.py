import json
from pathlib import Path

import numpy as np
import pytest

from tingan._tests.fake_data import fake_pulsar_data


def test_fake_pulsars_subdirs(
    tmp_path: Path,
    n_boring_pulsars,
    n_pulsars: int = 3,
) -> None:
    """Test that fake_pulsar_data creates the expected subdirectories."""
    fake_pulsars = n_boring_pulsars(n_pulsars)

    subdirs_created = fake_pulsar_data(tmp_path, *fake_pulsars)

    assert len(subdirs_created) == n_pulsars
    for i in range(n_pulsars):
        expected_path = tmp_path / f"psr-{i}"
        assert expected_path in subdirs_created
        assert expected_path.exists()


def test_fake_pulsars_model_params(
    tmp_path: Path,
    n_boring_pulsars,
    n_pulsars: int = 2,
    expected_param_keys: tuple[str, ...] = (
        "TNRedGam",
        "TNRedAmp",
        "F2",
        "F0",
        "pepoch",
    ),
) -> None:
    """Test that fake_pulsar_data correctly writes model parameters."""
    fake_pulsars = list(n_boring_pulsars(n_pulsars))
    # Change data values in 2nd pulsar entry so we can distinguish later
    fake_pulsars[1] = {key: 2 * value for key, value in fake_pulsars[1].items()}

    subdirs_created = fake_pulsar_data(tmp_path, *fake_pulsars)

    for i, subdir in enumerate(subdirs_created):
        model_params_file = subdir / "model_params.json"
        assert model_params_file.exists()

        with Path.open(model_params_file) as f:
            saved_data: dict = json.load(f)

        # All model parameter keys are present
        assert set(expected_param_keys).issubset(set(saved_data.keys()))
        # Correct values were saved
        expected_data_dumped = fake_pulsars[i]
        for key, value in saved_data.items():
            assert value == expected_data_dumped[key]


def test_fake_pulsars_npz_file(
    tmp_path: Path,
    n_boring_pulsars,
    n_pulsars: int = 2,
    expected_npz_arrays: tuple[str, ...] = ("mjd", "residual"),
) -> None:
    """Test that fake_pulsar_data correctly writes residuals.npz."""
    fake_pulsars = list(n_boring_pulsars(n_pulsars))
    # Change data values in 2nd pulsar entry so we can distinguish later
    fake_pulsars[1] = {key: 2 * value for key, value in fake_pulsars[1].items()}

    subdirs_created = fake_pulsar_data(tmp_path, *fake_pulsars)

    for i, subdir in enumerate(subdirs_created):
        residuals_file = subdir / "residuals.npz"
        assert residuals_file.exists()

        saved_data: dict = np.load(residuals_file)

        # All model parameter keys are present
        assert set(expected_npz_arrays).issubset(set(saved_data.keys()))
        # Correct values were saved
        expected_data_dumped = fake_pulsars[i]
        for key, array in saved_data.items():
            assert np.allclose(array, expected_data_dumped[key])


@pytest.mark.parametrize(
    ("extra_keys", "missing_keys", "error_msg"),
    [
        pytest.param(
            (),
            ("mjd",),
            "Extra keys: \nMissing keys: mjd",
            id="mjd missing",
        ),
        pytest.param(
            ("not a key",),
            (),
            "Extra keys: not a key\nMissing keys: ",
            id="Extra key present",
        ),
        pytest.param(
            ("not a key",),
            ("F0", "F2"),
            "Extra keys: not a key\nMissing keys: F0, F2",
            id="Extra + missing keys identified at once",
        ),
    ],
)
def test_fake_pulsars_bad_keys(
    extra_keys: tuple[str],
    missing_keys: tuple[str],
    error_msg: KeyError,
    tmp_path: Path,
    raises_context,
    n_boring_pulsars,
    n_pulsars: int = 1,
) -> None:
    """Confirm that KeyErrors are thrown when keys mismatch."""
    fake_pulsar: dict = n_boring_pulsars(n_pulsars)[0]

    # Artificially add and remove keys from the data we generated
    for key in missing_keys:
        fake_pulsar.pop(key)
    for key in extra_keys:
        fake_pulsar[key] = 0.0

    with raises_context(KeyError(error_msg)):
        fake_pulsar_data(tmp_path, fake_pulsar)
