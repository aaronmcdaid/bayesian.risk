from typeguard import typechecked
from bayesianAB.core_standard_risk import fast_standard_risk, fast_standard_risks
import numpy as np


@typechecked
def risk(x: float, loc: float = 0, scale: float = 1) -> float:
    """
        given a standard normal,
            Y ~ N(loc,scale^2)
        this returns
            mean(min(Y,x))
    """
    return loc + scale * fast_standard_risk((x - loc) / scale)


@typechecked
def risks(x: np.array, loc: np.array = 0.0, scale: np.array = 1.0) -> np.array:
    """
        given a standard normal,
            Y ~ N(loc,scale^2)
        this returns
            mean(min(Y,x))
    """
    return loc + scale * fast_standard_risks((x - loc) / scale)
