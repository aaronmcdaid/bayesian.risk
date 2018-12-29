from bayesianAB.event_stream import gen_normals, TrackOneStream, ABtest, random_variants, \
        one_column_per_variant, seeded_RandomStates, seeded_RandomState, random_standard_normals, \
        simulate_many_draws_for_many_variants, generate_cumulative_dataframes, generate_cumulative_dataframes_with_extra_columns, \
        SimulationParams, simple_dataframe_with_all_stats
import itertools as it
import numpy as np
import pandas as pd
from pytest import approx


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
    params = SimulationParams(n, M, weights, means, stdevs)
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
    params = SimulationParams(n, M, weights, means, stdevs)

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
    params = SimulationParams(n, M, weights, means, stdevs)

    g = generate_cumulative_dataframes_with_extra_columns(
            two_rngs,
            params,
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
    params = SimulationParams(n, M, weights, means, stdevs)

    many_estimates_of_the_difference = []
    many_estimates_of_the_estimatorvariance = []
    for _ in range(100):
        g = generate_cumulative_dataframes_with_extra_columns(
            two_rngs,
            params,
            )
        last_row = next(g).iloc[-1,]
        many_estimates_of_the_difference.append(last_row['difference_of_means'])
        many_estimates_of_the_estimatorvariance.append(last_row['variance_of_estimator'])

    central_estimate = np.mean(many_estimates_of_the_difference)
    variance_of_many_means = np.var(many_estimates_of_the_difference)
    central_variance = np.mean(many_estimates_of_the_estimatorvariance)

    assert central_estimate == approx(means[1] - means[0], abs=0.1)
    assert variance_of_many_means == approx(central_variance, abs=0.02)


def test_simple_dataframe_with_all_stats__sample_size():
    weights = [0.3, 0.7]
    means = [3, 5]
    stdevs = [2, 4]
    df = simple_dataframe_with_all_stats(weights, means, stdevs, 'total_sample_size >= 15')
    assert df.iloc[-1,]['total_sample_size'] == 15


def test_simple_dataframe_with_all_stats__risk():
    weights = [0.3, 0.7]
    means = [3, 4]
    stdevs = [2, 4]
    RISK_THRESHOLD_TO_WAIT_FOR = -0.01
    df = simple_dataframe_with_all_stats(weights, means, stdevs, 'risk >= {}'.format(RISK_THRESHOLD_TO_WAIT_FOR))
    assert df['risk'].iloc[-1] >= RISK_THRESHOLD_TO_WAIT_FOR
    assert df['risk'].iloc[-2] <  RISK_THRESHOLD_TO_WAIT_FOR


def test_simple_dataframe_with_all_stats__regret():
    weights = [0.3, 0.7]
    means = [4, 3]
    stdevs = [2, 4]
    REGRET_THRESHOLD_TO_WAIT_FOR = 0.01
    df = simple_dataframe_with_all_stats(weights, means, stdevs, 'regret <= {}'.format(REGRET_THRESHOLD_TO_WAIT_FOR))
    assert df['regret'].iloc[-1] <= REGRET_THRESHOLD_TO_WAIT_FOR
    assert df['regret'].iloc[-2] >  REGRET_THRESHOLD_TO_WAIT_FOR


def test_gen_normals():
    g = gen_normals(5, 3, 1337)
    x = list(it.islice(g, 1000))
    assert np.mean(x) == approx(4.9234279241459875)
    assert np.std(x) == approx(2.944864624211224)


def test_TrackOneStream():
    for mean in [-3, 0, 5]:
        for stdev in [0.1, 1, 10]:
            tos = TrackOneStream(gen_normals(mean, stdev, 1337))
            tos.advance(steps = 100000)
            m, s = tos.get_mean_and_sddev()
            assert 0.99 < s / stdev < 1.01
            assert -0.01 < (mean - m) / stdev < 0.01


def test_ABtest_weights():
    # Generate with a 30/70 split, and check that is
    # the observed split.
    x = [
        (3, gen_normals(0, 1, 1337)),
        (7, gen_normals(0, 1, 1337)),
    ]
    ab = ABtest.from_list_of_pairs(x, 1337)
    ab.advance(steps = 1000)
    ns, _, _ = ab.get_ns_means_sddevs()
    assert ns == [306, 694]


def test_ABtest_means_and_sddevs():
    # Generate with a few different means and stdevs,
    # and verify the results are approximately as
    # expected.
    x = [
        (3, gen_normals(0, 1, 1337)),
        (7, gen_normals(3, 2, 1337)),
        (7, gen_normals(8, 3, 1337)),
    ]
    ab = ABtest.from_list_of_pairs(x, 1337)
    ab.advance(steps = 1000)
    _, means, stdevs = ab.get_ns_means_sddevs()
    assert means == approx([0, 3, 8], abs=0.1)
    assert stdevs == approx([1, 2, 3], abs=0.1)


def test_ABtest_at_least_two():
    x = [
        (3, gen_normals(0, 1, 1337)),
        (7, gen_normals(0, 1, 1337)),
    ]
    ab = ABtest.from_list_of_pairs(x, 1337)
    # Even though 'advance' isn't called here, we expect that
    # each variant has two observations in it already
    ns, _, _ = ab.get_ns_means_sddevs()
    assert ns == [2, 2]


def test_ABtest_simple():
    # Generate with a few different means and stdevs,
    # and verify the results are approximately as
    # expected.
    ab = ABtest.from_simple_args(true_diff=3, sd=2, weight=0.3, seed=1337, steps=1000)
    ns, means, stdevs = ab.get_ns_means_sddevs()
    assert ns == [306, 694]
    assert means[1] - means[0] == approx(3, abs=0.1)
    assert stdevs == approx([2, 2], abs=0.1)
