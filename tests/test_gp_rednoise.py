from pathlib import Path

import numpy as np

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


def test_marginalize_2d_kde_with_ones() -> None:
    """Marginalize a flat 2D distribution."""
    kde = np.ones((10, 10))
    data = np.linspace(0, 1, 10)
    kde_pdf = gp.marginalize_2d_kde(kde, 0, data)
    assert np.allclose(kde_pdf, 10 * [1.0])


def test_marginalize_2d_kde_with_rand() -> None:
    """Marginalize a distribution that is already 1D."""
    kde = np.random.default_rng().random(size=(1, 10))
    data = np.linspace(0, 1, 10)
    kde_pdf = gp.marginalize_2d_kde(kde, 1, data)
    assert np.isclose(np.std(kde[0][::-1] / kde_pdf), 0.0)
