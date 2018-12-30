"""
    This is the underlying module providing a simple risk calculation
    (expected loss) under a standard normal distribution.
    Also, this provides a fast implementation based on a pre-computed
    lookup.

    This is used by the bayesianAB.risk module, which is a wrapper that
    extends it to normals with non-standard parameters
"""
from typeguard import typechecked
from scipy.stats import truncnorm, norm
import numpy as np
from pytest import approx


@typechecked
def standard_risk(x: float) -> float:
    """
        given a standard normal,
            Y ~ N(0,1)
        this returns
            mean(min(Y,x))
    """
    if x >= 0:
        # These three lines could be used for both positive
        # and negative values of 'x', but we use this if
        # in order to avoid having to use a special case
        # for values of 'x' less then -100
        t = truncnorm.mean(-100, x)
        p = norm.cdf(x)
        return p * t + (1 - p) * x
    else:
        return x + standard_risk(-x)


_precomputed_array_of_standard_risk = None
_Precomputed_array_of_standard_risk = None


def _get_precomputed_array_of_standard_risk() -> np.array:
    global _precomputed_array_of_standard_risk
    global _Precomputed_array_of_standard_risk
    STEP = 0.001
    if _precomputed_array_of_standard_risk is None:
        xs = list(np.arange(0, 10 + STEP / 2, STEP))
        _precomputed_array_of_standard_risk = np.array([standard_risk(x) for x in xs])
        xs = list(np.arange(-10, 10 + STEP / 2, STEP))
        _Precomputed_array_of_standard_risk = np.array([standard_risk(x) for x in xs])
    return _precomputed_array_of_standard_risk


@typechecked
def fast_standard_risk(x: float):
    pre = _get_precomputed_array_of_standard_risk()
    N = len(_Precomputed_array_of_standard_risk)
    n = len(pre)
    assert pre.dtype.name == 'float64'
    assert N == 2 * n -1

    assert pre[0] == approx(_Precomputed_array_of_standard_risk[(N-1)//2])

    if np.isnan(x):
        return x

    if x < -10:
        return x

    # Now to compute the index corresponding to 'x'
    # pre[0] corresponds to standard_risk(x=-10)
    # pre[n-1] corresponds to standard_risk(x=10)
    Index = int((N-1)//2 + round((N - 1) * x / 20))
    # If 'x' is greater than 10, then pull the index down
    Index = min(N - 1, Index)
    Index = max(0, Index)
    return _Precomputed_array_of_standard_risk[Index]
