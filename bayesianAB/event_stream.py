from typeguard import typechecked
import numpy as np

@typechecked
def gen_normals(loc: float, scale: float, seed: int):
    rng = np.random.RandomState()
    rng.seed(seed)
    while True:
        xs = rng.normal(loc, scale, 1000)
        for x in xs:
            yield x


class TrackOneStream:

    def __init__(self, g):
        self._g = g
        self._n = 0
        self._sum = 0.0
        self._sum_squares = 0.0

    def advance(self):
        x = next(self._g)
        self._n += 1
        self._sum += x
        self._sum_squares += x*x

    def get_mean_and_sddev(self):
        m = self._sum / self._n
        v = self._sum_squares / self._n - m**2
        return m, np.sqrt(v)
