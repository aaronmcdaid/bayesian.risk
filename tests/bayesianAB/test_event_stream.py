from bayesianAB.event_stream import gen_normals, TrackOneStream, ABtest
import itertools as it
import numpy as np
from pytest import approx


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
    ab = ABtest.from_simple_args(true_diff=3, sd=2, weight=0.3, steps=1000, seed=1337)
    ns, means, stdevs = ab.get_ns_means_sddevs()
    assert ns == [306, 694]
    assert means[1] - means[0] == approx(3, abs=0.1)
    assert stdevs == approx([2, 2], abs=0.1)
