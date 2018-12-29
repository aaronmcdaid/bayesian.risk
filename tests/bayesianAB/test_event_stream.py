from bayesianAB.event_stream import gen_normals, TrackOneStream, ABtest, random_variants, \
        one_column_per_variant, seeded_RandomState, random_standard_normals, \
        simulate_many_draws_for_many_variants, cumulate
import itertools as it
import numpy as np
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
    rng_variant = seeded_RandomState(1337)
    rng_normals = seeded_RandomState(1234)
    n = 1000
    weights = [0.3, 0.7]
    means = [3, 5]
    stdevs = [1, 2]
    df = simulate_many_draws_for_many_variants(
            rng_variant,
            rng_normals,
            n,
            2, # M: number of variants
            weights,
            means,
            stdevs,
            )
    print()
    print(df)
    summed_df = cumulate(df)
    print(summed_df)
    assert summed_df['sample_size_0'] == 306 # approximately n * weights[0]
    assert summed_df['sample_size_1'] == 694 # approximately n * weights[1]
    assert summed_df['sum_0'] / summed_df['sample_size_0'] == approx(means[0], abs=0.1)
    assert summed_df['sum_1'] / summed_df['sample_size_1'] == approx(means[1], abs=0.1)


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
