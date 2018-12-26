from typeguard import typechecked
from bayesianAB.core_standard_risk import fast_standard_risk


@typechecked
def risk(x: float, loc: float = 0, scale: float = 1) -> float:
    """
        given a standard normal,
            Y ~ N(loc,scale^2)
        this returns
            mean(min(Y,x))
    """
    return loc + scale * fast_standard_risk((x - loc) / scale)
