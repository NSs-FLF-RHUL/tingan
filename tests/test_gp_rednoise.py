from pathlib import Path

import numpy as np
import pytest

from tingan import gp_rednoise as gp
from tingan._tests.fake_data import fake_pulsar_data


def test_load_gammas_and_amplitudes(
    tmp_path: Path,
    n_boring_pulsars,
    expected_n_pulsars: int = 21,
) -> None:
    """Test that the correct number of parameters are loaded."""
    created_pulsars = n_boring_pulsars(expected_n_pulsars)
    fake_pulsar_dirs = fake_pulsar_data(tmp_path, *created_pulsars)

    gammas, *_ = gp.load_gammas_and_amplitudes(fake_pulsar_dirs)
    assert len(gammas) == expected_n_pulsars
@pytest.fixture
def rng(seed: int = 0) -> np.random.Generator:
    """Set the RNG for the entire test suite."""
    return np.random.default_rng(seed)


def test_load_gammas_and_amplitudes() -> None:
    """Test that the correct number of parameters are loaded."""
    # I know this is specific to my machine, but I got a bug with npsrs when passing
    # psrs as a function argument. __sizeof__() used to return the number of pulsars,
    # 21, but inside the function it returms 32. So I just want to make sure I loaded
    # data for the correct number of pulsars.
    psrs = tuple(Path("/home/jberteaud/Science/EOS/tingan/data/real/").glob("[JB]*"))
    gammas, *_ = gp.load_gammas_and_amplitudes(psrs)
    number_of_psrs_on_my_machine = 21
    assert len(gammas) == number_of_psrs_on_my_machine


def test_marginalize_2d_kde_with_ones() -> None:
    """Marginalize a flat 2D distribution."""
    kde = np.ones((10, 10))
    data = np.linspace(0, 1, 10)
    kde_pdf = gp.marginalize_2d_kde(kde, 0, data)
    assert np.allclose(kde_pdf, 10 * [1.0])


def test_marginalize_2d_kde_with_rand(rng: np.random.Generator) -> None:
    """Marginalize a distribution that is already 1D."""
    kde = rng.random(size=(1, 10))
    data = np.linspace(0, 1, 10)
    kde_pdf = gp.marginalize_2d_kde(kde, 1, data)
    assert np.isclose(np.std(kde[0][::-1] / kde_pdf), 0.0)
