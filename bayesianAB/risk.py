from typeguard import typechecked
from scipy.stats import truncnorm, norm
import numpy as np

@typechecked
def standard_risk(x: float):
    """
        given a standard normal,
            Y ~ N(0,1)
        this returns
            mean(min(Y,x))
    """
    # scipy.stats.truncnorm doesn't like very negative bounds,
    # so we clip 'x' so that it's not so extreme
    VERY_NEGATIVE = -8

    if x <= VERY_NEGATIVE:
        return x
    else:
        t = truncnorm.mean(VERY_NEGATIVE, x)
        p = norm.cdf(x)
        return p*t + (1-p) * x
