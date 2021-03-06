from bayesianAB.event_stream import random_variants, one_column_per_variant, seeded_RandomStates, seeded_RandomState, random_standard_normals, \
        simulate_many_draws_for_many_variants, generate_cumulative_dataframes, generate_cumulative_dataframes_with_extra_columns, \
        SimulationParamsForOneChunk, one_simulation_until_stopping_condition
import itertools as it
import numpy as np
import pandas as pd
from pytest import approx
from bayesianAB.prior import FlatPrior


def test_random_variants():
    rng = seeded_RandomState(1337)
    x = random_variants(rng, [0.3, 0.6, 0.1], 1000)
    counts = x.value_counts()
    assert counts[0] == 306
    assert counts[1] == 594
    assert counts[2] == 100


def test_make_one_column_per_variant():
    rng = seeded_RandomState(1337)
    weights = [0.3, 0.6, 0.1]
    M = len(weights)
    x = random_variants(rng, weights, 1000)
    df = one_column_per_variant(M, x)
    counts = df.agg('sum')
    assert counts.to_dict() == x.value_counts().to_dict()


def test_simulate_many_normals():
    rng = seeded_RandomState(1234)
    x = random_standard_normals(rng, 100)
    assert x.mean() == approx(0, abs=0.4)
    assert x.std() == approx(1, abs=0.01)


def test_simulate_many_draws_for_many_variants():
    two_rngs = seeded_RandomStates(1337, 1234)
    n = 10000
    M = 2
    weights = [0.3, 0.7]
    means = [3, 5]
    stdevs = [2, 4]
    params = SimulationParamsForOneChunk(n, M, weights, means, stdevs)
    simulated_dataframes = simulate_many_draws_for_many_variants(
            two_rngs,
            params,
            )
    sample_sizes = simulated_dataframes.assignment.agg('sum')
    sums = simulated_dataframes.metric.agg('sum')
    sumOfSquares = simulated_dataframes.metric_squared.agg('sum')

    estimated_means = sums / sample_sizes
    estimated_variances = sumOfSquares / sample_sizes - estimated_means ** 2

    assert sample_sizes[0] == 3006 # approximately n * weights[0]
    assert sample_sizes[1] == 6994 # approximately n * weights[1]
    assert estimated_means[0] == approx(means[0], abs=0.1)
    assert estimated_means[1] == approx(means[1], abs=0.1)
    assert np.sqrt(estimated_variances[0]) == approx(stdevs[0], abs=0.1)
    assert np.sqrt(estimated_variances[1]) == approx(stdevs[1], abs=0.1)


def test_generate_cumulative_dataframes():
    two_rngs = seeded_RandomStates(1337, 1234)
    n = 10
    M = 2
    NUMBER_OF_CHUNKS = 2
    weights = [0.3, 0.7]
    means = [3, 5]
    stdevs = [2, 4]
    params = SimulationParamsForOneChunk(n, M, weights, means, stdevs)

    dfs= list(it.islice(generate_cumulative_dataframes(
            two_rngs,
            params,
            ), NUMBER_OF_CHUNKS))
    df = pd.concat(dfs, axis = 0).reset_index(drop=True)

    total_sample_sizes = df['sample_size_0'] + df['sample_size_1']
    assert total_sample_sizes.tolist() == [s+1 for s in range(n * NUMBER_OF_CHUNKS)]


def test_inserting_extra_columns():
    two_rngs = seeded_RandomStates(1337, 1234)
    n = 1000
    M = 2
    NUMBER_OF_CHUNKS = 10
    weights = [0.3, 0.7]
    means = [3, 5]
    stdevs = [2, 4]
    params = SimulationParamsForOneChunk(n, M, weights, means, stdevs)

    g = generate_cumulative_dataframes_with_extra_columns(
            two_rngs,
            params,
            FlatPrior(),
            )
    for _ in range(NUMBER_OF_CHUNKS-1):
        next(g) # discard the first NUMBER_OF_CHUNKS-1 chunks
    last_row = next(g).iloc[-1,]
    assert last_row['sample_size_0'] + last_row['sample_size_1'] == n * NUMBER_OF_CHUNKS
    assert last_row['estimated_mean_0'] == approx(means[0], abs=0.1)
    assert last_row['estimated_mean_1'] == approx(means[1], abs=0.1)
    assert np.sqrt(last_row['estimated_variance_0']) == approx(stdevs[0], abs=0.1)
    assert np.sqrt(last_row['estimated_variance_1']) == approx(stdevs[1], abs=0.1)


def test_inserting_columns_and_correctness():
    two_rngs = seeded_RandomStates(1337, 1234)
    n = 100
    M = 2
    weights = [0.3, 0.7]
    means = [3, 5]
    stdevs = [3, 3] # must be equal to each other for this test to work
    params = SimulationParamsForOneChunk(n, M, weights, means, stdevs)

    many_estimates_of_the_difference = []
    many_estimates_of_the_estimatorvariance = []
    for _ in range(100):
        g = generate_cumulative_dataframes_with_extra_columns(
            two_rngs,
            params,
            FlatPrior(),
            )
        last_row = next(g).iloc[-1,]
        many_estimates_of_the_difference.append(last_row['difference_of_means'])
        many_estimates_of_the_estimatorvariance.append(last_row['variance_of_estimator'])

    central_estimate = np.mean(many_estimates_of_the_difference)
    variance_of_many_means = np.var(many_estimates_of_the_difference)
    central_variance = np.mean(many_estimates_of_the_estimatorvariance)

    assert central_estimate == approx(means[1] - means[0], abs=0.1)
    assert variance_of_many_means == approx(central_variance, abs=0.02)


def test_one_simulation_until_stopping_condition__sample_size():
    weights = [0.3, 0.7]
    means = [3, 5]
    stdevs = [2, 4]
    # In this test, we use 'min_sample_size=0' in order to disable the special
    # rule that says that every variant must have at least 5 entries. In this
    # test, 11 controls and 4 treatments is OK
    df = one_simulation_until_stopping_condition(weights, means, stdevs, 'total_sample_size >= 15', min_sample_size=0)
    assert df.iloc[-1,]['total_sample_size'] == 15


def test_one_simulation_until_stopping_condition__min_sample_size():
    """
        The min_sample_size means the early rows should be skipped entirely.
        So in this test we check that the first row has the minimum number of
        samples
    """
    weights = [0.3, 0.7]
    means = [3, 5]
    stdevs = [2, 4]
    MIN_SAMPLE_SIZE=4
    # In this test, we use 'min_sample_size=0' in order to disable the special
    # rule that says that every variant must have at least 5 entries. In this
    # test, 11 controls and 4 treatments is OK
    df = one_simulation_until_stopping_condition(weights, means, stdevs, 'total_sample_size >= 100', min_sample_size=MIN_SAMPLE_SIZE)
    smsz0, smsz1 = df.iloc[0,][['sample_size_0', 'sample_size_1']]
    print(smsz0, smsz1)
    assert min(smsz0, smsz1) == MIN_SAMPLE_SIZE


def test_one_simulation_until_stopping_condition__risk():
    weights = [0.3, 0.7]
    means = [3, 4]
    stdevs = [2, 4]
    RISK_THRESHOLD_TO_WAIT_FOR = -0.01
    # fixed seeds are needed here, or else sometimes the second assert fails
    df = one_simulation_until_stopping_condition(weights, means, stdevs, 'EL >= {}'.format(RISK_THRESHOLD_TO_WAIT_FOR), seeds=(1,2))
    assert df['EL'].iloc[-1] >= RISK_THRESHOLD_TO_WAIT_FOR
    assert df['EL'].iloc[-2] <  RISK_THRESHOLD_TO_WAIT_FOR


def test_one_simulation_until_stopping_condition__regret():
    weights = [0.3, 0.7]
    means = [4, 3]
    stdevs = [2, 4]
    REGRET_THRESHOLD_TO_WAIT_FOR = 0.0001
    df = one_simulation_until_stopping_condition(weights, means, stdevs, 'EG <= {}'.format(REGRET_THRESHOLD_TO_WAIT_FOR), seeds=(1,2))
    # fixed seeds are needed here, or else sometimes the second assert fails
    assert df['EG'].iloc[-1] <= REGRET_THRESHOLD_TO_WAIT_FOR
    assert df['EG'].iloc[-2] >  REGRET_THRESHOLD_TO_WAIT_FOR


def test_one_simulation_until_stopping_condition__min_sample_size():
    # Test the 'min_sample_size' parameter to 'one_simulation_until_stopping_condition',
    # which ensures that both variants have at least that many samples, regardless
    # of any other condition.
    weights = [0.3, 0.7]
    means = [3, 5]
    stdevs = [2, 4]
    df = one_simulation_until_stopping_condition(weights, means, stdevs, 'total_sample_size >= 0', min_sample_size = 13)
    last_row = df.iloc[-1,]
    assert min(last_row.sample_size_0, last_row.sample_size_1) == 13
