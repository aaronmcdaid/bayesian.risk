from bayesianAB.event_stream import gen_normals, TrackOneStream
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
            for x in range(100000):
                tos.advance()
            m, s = tos.get_mean_and_sddev()
            assert 0.99 < s/stdev < 1.01
            assert -0.01 < (mean - m) / stdev < 0.01
