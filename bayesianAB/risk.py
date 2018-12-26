from typeguard import typechecked
from bayesianAB.core_standard_risk import standard_risk


@typechecked
def risk(x: float, loc: float = 0, scale: float = 1) -> float:
    """
        given a standard normal,
            Y ~ N(loc,scale^2)
        this returns
            mean(min(Y,x))
    """
    return loc + scale * standard_risk((x - loc)/scale)
