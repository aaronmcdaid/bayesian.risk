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
    if np.isnan(x):
        return x
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


def _get_precomputed_array_of_standard_risk() -> np.array:
    global _precomputed_array_of_standard_risk
    STEP = 0.001
    if _precomputed_array_of_standard_risk is None:
        xs = list(np.arange(0, 10 + STEP / 2, STEP))
        _precomputed_array_of_standard_risk = np.array([standard_risk(x) for x in xs])
    return _precomputed_array_of_standard_risk


@typechecked
def fast_standard_risk(x: float):
    pre = _get_precomputed_array_of_standard_risk()
    n = len(pre)
    assert pre.dtype.name == 'float64'

    if np.isnan(x):
        return x

    if x < -10:
        return x

    if x < 0:
        return x + fast_standard_risk(-x)

    index = int(round((n - 1) * x / 10))
    # If 'x' is greater than 10, then pull the index down
    index = min(n - 1, index)
    index = max(0, index)
    return pre[index]


@typechecked
def fast_standard_risks(x: np.array):
    # Make sure it's floating-point, not integer
    x = x.astype('float', copy=False)
    # Remember where the NaNs are
    indices_of_nan = np.isnan(x)
    # Convert NaNs to zero
    x = np.nan_to_num(x)
    # Remember where the 'x' are negative
    negative_indices = x < 0
    # Prepare the result vector
    result = x * 0
    # Store the results for the non-negative xs
    result[~ negative_indices] = _only_non_negative_fast_standard_risks(x[~ negative_indices])
    # Store the results for the negative xs
    result[negative_indices] = x[negative_indices] + _only_non_negative_fast_standard_risks(-x[negative_indices])
    # Write NaN in for the NaNs
    result[indices_of_nan] = np.nan
    return result



@typechecked
def _only_non_negative_fast_standard_risks(x: np.array):
    assert (x >= 0).all()
    pre = _get_precomputed_array_of_standard_risk()
    n = len(pre)
    indices = ((n - 1) * x / 10).round()
    # If 'x' is greater than 10, then pull the index down
    indices = np.minimum(n - 1, indices).astype(int)
    return pre[indices]
