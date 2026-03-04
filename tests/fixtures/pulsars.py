from collections.abc import Callable

import numpy as np
import pytest


@pytest.fixture
def n_boring_pulsars(
    n_residuals: int = 10,
) -> Callable[[int], tuple[dict[str, float | np.ndarray]]]:
    """
    Create `n` identical pulsars displaying no interesting features.

    All model parameters are set to 1. Residuals are all set to 0 (so the data is
    "exact") and `mjd` values are set to an appropriate `np.arange` span.

    Fixture is designed for use with the `fake_pulsar_data` function.
    """

    def _inner(n_pulsars: int = 3) -> tuple[dict[str, float | np.ndarray]]:
        pulsar = {
            "TNRedGam": 1.0,
            "TNRedAmp": 1.0,
            "F2": 1.0,
            "F0": 1.0,
            "pepoch": 1.0,
            "mjd": np.arange(n_residuals, dtype=float),
            "residual": np.zeros((10,)),
        }
        return (pulsar,) * n_pulsars

    return _inner
