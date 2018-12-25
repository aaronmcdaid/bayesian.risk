from typeguard import typechecked
from scipy.stats import truncnorm, norm
import numpy as np

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
        return p*t + (1-p) * x
    else:
        return x + standard_risk(-x)
