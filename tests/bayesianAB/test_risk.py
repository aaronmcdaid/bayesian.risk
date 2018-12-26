import pytest
from pytest import approx
from bayesianAB.risk import risk
import numpy as np


def test_risk():
    # first, just confirm the standard normal case
    assert risk(1, 0, 1) == approx(-0.08331547)
    assert risk(-1, 0, 1) == approx(-1.083315470)


def test_risk__translated():
    # then, location translated by a constant. Simply
    # adding 5 to each of the two last tests.
    assert risk(5 + 1, 5, 1) == approx(5 - 0.08331547)
    assert risk(5 - 1, 5, 1) == approx(5 - 1.083315470)


def test_risk__scaled_not_translated():
    # Back to location zero, but scaling instead
    assert risk(2, 0, 2) == approx(2 * -0.08331547)
    assert risk(-2, 0, 2) == approx(2 * -1.083315470)
    assert risk(3, 0, 3) == approx(3 * -0.08331547)
    assert risk(-3, 0, 3) == approx(3 * -1.083315470)


def test_risk__scaled_and_translated():
    # scaled and translated
    assert risk(100 + 3, 100, 3) == approx(100 + 3 * -0.08331547)


def test_risk_default_params():
    # check the default params work
    assert risk(7, 0, 1) == risk(7)
