import io

import numpy as np
import pytest
from pint.toa import get_TOAs

from tingan import datasets


@pytest.mark.parametrize(
    "toas",
    [
        pytest.param([58000, 58500], id="list"),
        pytest.param(np.array([58000, 58500]), id="array"),
        pytest.param(
            get_TOAs(
                io.StringIO(
                    "FORMAT 1\n"
                    "58000.ft 1376.00000000 57999.999999999679810 2.02300 Hobart\n"
                    "58500.ft 1376.00000000 58499.999999999679701 7.04000 Hobart"
                )
            ),
            id="pint",
        ),
    ],
)
def test_toas_to_timellm_format_list(toas) -> None:
    """Test TOA conversion from MJD to timellm format."""
    time = datasets.toas_to_timellm_format(toas)
    np.testing.assert_array_equal(
        time,
        np.array(
            ["2017-09-04T00:00:00.000000000", "2019-01-17T00:00:00.000000000"],
            dtype="<U29",
        ),
    )
