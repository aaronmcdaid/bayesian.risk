from bayesianAB.event_stream import SimulationParams, run_simulation_until_the_end
import pandas as pd
import numpy as np
from bayesianAB.prior import FlatPrior, FixedPrior
from pytest import approx


def test_very_strong_prior():
    # A very strong prior (small stdev), therefore
    # the posterior will be close to the prior.
    prior = FixedPrior(5, 0.01)

    sim_params = SimulationParams(
        weights = [0.5, 0.5],
        means = [1, 1],
        stdevs = [1, 1],
        stopping_condition = 'total_sample_size >= 10',
        min_sample_size = 0,
        seeds = (1,2),
        prior = prior,
        )
    df = run_simulation_until_the_end(sim_params)
    assert df.posterior_mean.iloc[-1] == approx(prior.prior_mean, abs=0.01)
    assert df.posterior_stdev.iloc[-1] == approx(prior.prior_stdev, abs=0.01)


def test_very_weak_prior():
    # A very weak prior (large stdev), therefore
    # the posterior will be close the likelihood.
    # i.e. the posterior_mean should be closer to 'true_diff' than
    # it is to 'prior_mean':
    prior = FixedPrior(5, 1000)
    true_diff = -3

    sim_params = SimulationParams(
        weights = [0.5, 0.5],
        means = [1, 1+true_diff],
        stdevs = [1, 1],
        stopping_condition = 'total_sample_size >= 10',
        min_sample_size = 0,
        seeds = (1,2),
        prior = prior,
        )
    df = run_simulation_until_the_end(sim_params)
    assert df.posterior_mean.iloc[-1] == approx(true_diff, abs=0.5)


def test_very_large_sample_size():
    # With a large sample size, the prior doesn't matter. Even if
    # it is a strong prior.
    prior = FixedPrior(5, 0.1)
    true_diff = -3

    sim_params = SimulationParams(
        weights = [0.5, 0.5],
        means = [1, 1+true_diff],
        stdevs = [1, 1],
        stopping_condition = 'total_sample_size >= 100000',
        min_sample_size = 0,
        seeds = (1,2),
        prior = prior,
        )
    df = run_simulation_until_the_end(sim_params)
    assert df.posterior_mean.iloc[-1] == approx(true_diff, abs=0.1)
