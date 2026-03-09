import json
from pathlib import Path
from typing import Literal

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


@pytest.mark.parametrize("file", ["model_params.json", "residuals.npz"])
def test_fake_pulsars_files(
    tmp_path: Path,
    n_boring_pulsars,
    file: Literal["model_params.json", "residuals.npz"],
    n_pulsars: int = 2,
) -> None:
    """
    Test that fake_pulsar_data creates files with the expected structure.

    Also confirms that the pulsars are saved in the order / directory structure
    that is to be expected.
    """
    fake_pulsars = list(n_boring_pulsars(n_pulsars))
    # Change data values in 2nd pulsar entry so we can distinguish later
    fake_pulsars[1] = {key: 2 * value for key, value in fake_pulsars[1].items()}
    subdirs_created = fake_pulsar_data(tmp_path, *fake_pulsars)

    expected_keys: tuple[str, ...]  # mypy things
    if file == "model_params.json":
        expected_keys = (
            "TNRedGam",
            "TNRedAmp",
            "F2",
            "F0",
            "pepoch",
        )

        def load_saved_data(f: Path) -> dict:
            with Path.open(f) as ff:
                return json.load(ff)
    else:
        expected_keys = ("mjd", "residual")
        load_saved_data = np.load

    for i, subdir in enumerate(subdirs_created):
        results_file = subdir / file
        assert results_file.exists()

        saved_data: dict = load_saved_data(results_file)

        # All model parameter keys are present
        assert set(expected_keys).issubset(set(saved_data.keys()))
        # Correct values were saved
        expected_data_dumped = fake_pulsars[i]
        for key, value in saved_data.items():
            assert np.allclose(value, expected_data_dumped[key])


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
