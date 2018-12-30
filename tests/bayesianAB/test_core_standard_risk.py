from typeguard import typechecked
from bayesianAB.core_standard_risk import standard_risk, fast_standard_risk, fast_standard_risks
from pytest import approx
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
    # for very negative 'x', they become equal
    assert standard_risk(-5) == approx(-5)
    assert standard_risk(-7) == approx(-7)
    assert standard_risk(-100) == approx(-100)


def test_standard_risk__symmetry():
    # There is a symmetry property
    assert standard_risk(2) - standard_risk(-2) == approx(2)
    assert standard_risk(3) - standard_risk(-3) == approx(3)


def test_precompute():
    # a couple of checks around x=0
    assert fast_standard_risk( 1e-4) == fast_standard_risk(0)
    assert fast_standard_risk(-2e-3) <  fast_standard_risk(0)
    assert fast_standard_risk( 2e-3) >  fast_standard_risk(0)
    assert fast_standard_risk(0) == approx(standard_risk(0))

    for x in np.arange(-13, 13, 0.1):
        assert fast_standard_risk(x) == approx(standard_risk(x), abs=1e-3)
    for x in [-1.939123180390296, 1.939123180390296,]:
        assert fast_standard_risk(x) == approx(standard_risk(x), abs=1e-3), (x, standard_risk(x), fast_standard_risk(x))


def test_fast_standard_risks_positive():
    xs = np.array([0,1,2,10,15])
    ys = fast_standard_risks(xs)
    verify = np.array([standard_risk(x) for x in xs.tolist()])
    assert (ys == approx(verify))


def test_fast_standard_risks_nan():
    xs = np.array([0,1,2,np.nan,10,15])
    ys = fast_standard_risks(xs)
    verify = np.array([standard_risk(x) for x in xs.tolist()])
    assert (np.isnan(ys) == np.isnan(verify)).all()
    assert (np.nan_to_num(ys) == approx(np.nan_to_num(verify)))


def test_fast_standard_risks_negative():
    xs = np.array([-1, -2, -15])
    ys = fast_standard_risks(xs)
    verify = np.array([standard_risk(x) for x in xs.tolist()])
    assert (np.nan_to_num(ys) == approx(np.nan_to_num(verify)))


def test_fast_standard_risks_mix():
    xs = np.array([0,1,-2,np.nan,10,-13,15])
    ys = fast_standard_risks(xs)
    verify = np.array([standard_risk(x) for x in xs.tolist()])
    assert (np.isnan(ys) == np.isnan(verify)).all()
    assert (np.nan_to_num(ys) == approx(np.nan_to_num(verify)))
