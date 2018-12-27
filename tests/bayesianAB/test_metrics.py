from bayesianAB.event_stream import gen_normals, ABtest
from bayesianAB import metrics as bm
import numpy as np
from pytest import approx

def test_difference_of_means__as_a_likelihood_function():
    # simulate 100 experiments with the same true difference.
    # this test checks that the means are clustered around
    # the true difference.
    # Also, each experiment returns an estimate of its
    # own variance. So we also check here that the variation
    # of the 100 estimates is as expected

    # Feel free to change these next couple of variables.
    # The teset should still pass regardless
    true_mean_diff = 5
    true_sd = 3

    diffs, sds = ([], [])
    for seed in range(1000, 1500):
        params = [
            (3, gen_normals(0, true_sd, seed + 20000)),
            (7, gen_normals(true_mean_diff, true_sd, seed + 10000)),
        ]
        ab = ABtest.from_list_of_pairs(params, seed)
        ab.advance(steps = 100)

        m, s = bm.get_difference_of_means__and_its_sd(ab)
        diffs.append(m)
        sds.append(s)

    assert np.mean(diffs) == approx(true_mean_diff, abs=0.02)
    assert np.std(diffs) == approx(np.mean(sds), abs=0.04)


def test_ABtest_many_iterations():
    ab = ABtest.from_simple_args(true_diff=3, sd=2, weight=0.3)
    df = bm.many_iterations(ab, 100)
    (l, g) = df.iloc[99,]
    assert l == approx(0)
    assert g == approx(3, abs=0.5) # very rough
    assert df.shape == (100, 2)
