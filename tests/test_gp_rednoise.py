from pathlib import Path

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, WhiteKernel

from tingan import gp_rednoise as gp
from tingan._tests.fake_data import fake_pulsar_data


@pytest.mark.parametrize(
    "expected_n_pulsars",
    [pytest.param(21, id="Some pulsars"), pytest.param(0, id="No pulsars")],
)
def test_load_gammas_and_amplitudes(
    tmp_path: Path,
    n_boring_pulsars,
    expected_n_pulsars: int,
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


def test_fit_residual_gp() -> None:
    """Test Gaussian Process fit to simple data."""
    x = np.linspace(0, 10, 101)
    y = np.sin(x)
    rng = np.random.default_rng(100)
    y += rng.uniform(low=-0.1, high=0.1, size=101)
    gpr, score, mean, std = gp.fit_residual_gp(x, y)
    assert np.isclose(
        gpr.kernel_.get_params()["k1__k1__constant_value"], 6.012942163094865
    )
    assert np.isclose(gpr.kernel_.get_params()["k1__k2__nu"], 1.5)
    assert np.isclose(
        gpr.kernel_.get_params()["k1__k2__length_scale"], 5.319496504948747
    )
    assert np.isclose(gpr.kernel_.get_params()["k2__noise_level"], 0.005939190406783511)
    assert np.isclose(score, 0.9953953125232637)
    assert np.isclose(mean, 0.18859336795848478)
    assert np.isclose(std, 0.6765636601856538)


@pytest.mark.parametrize(
    ("kernel", "seed", "expected_answer"),
    [
        pytest.param(
            WhiteKernel(1.5),
            44,
            np.array(
                [
                    -3.36470385,
                    1.82627574,
                    -0.73718275,
                    -0.30328934,
                    0.87848262,
                    1.2130447,
                    -0.80404935,
                    1.30235749,
                    -0.84394868,
                    -0.00987443,
                    -0.24607337,
                ]
            ),
            id="white",
        ),
        pytest.param(
            ConstantKernel(1.0), 42, np.array(11 * [-1.38408539]), id="constant"
        ),
    ],
)
def test_sample_residual(
    kernel: Kernel, seed: int, expected_answer: np.ndarray
) -> None:
    """Test sampling form Gaussian Process."""
    x = np.linspace(0, 1, 11)
    gpr = GaussianProcessRegressor(kernel=kernel)
    rng = np.random.default_rng(seed)
    assert np.allclose(gp.sample_residual(x, gpr, rng), expected_answer)


def test_check_kernel_corr() -> None:
    """Test identification of correct kernel."""
    kernel = ConstantKernel(1.0, (1e-3, 1e3))
    gp.check_kernel(kernel.get_params())


def test_check_kernel_incorr() -> None:
    """Test identification of incorrect kernel."""
    kernel = ConstantKernel(1e4, (1e-3, 1e3))
    with pytest.raises(ValueError, match="is out of bounds"):
        gp.check_kernel(kernel.get_params())
