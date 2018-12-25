import pytest
from pytest import approx
from bayesianAB.risk import standard_risk
import numpy as np

def test_standard_risk():
    # For large 'x', there is essentially no clipping,
    # and therefore it's basically a standard normal
    # with mean 0.
    assert standard_risk(1000) == approx(0)
    assert standard_risk(100) == approx(0)
    # As 'x' decreases, so does the standard_risk(x).
    # standard_risk(x) is always less than 'x'.
    assert standard_risk(3) == approx(-0.0003821543170)
    assert standard_risk(1) == approx(-0.08331547)
    assert standard_risk(0) == approx(-0.3989422)
    assert standard_risk(-1) == approx(-1.083315470)
    assert standard_risk(-2) == approx(-2.008490702)
    assert standard_risk(-3) == approx(-3.000382154)
    assert standard_risk(-4) == approx(-4.000007145)
    assert standard_risk(-5) == approx(-5)
    assert standard_risk(-7) == approx(-7)
    assert standard_risk(-100) == approx(-100)

