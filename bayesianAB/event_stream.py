from typeguard import typechecked, List, Any, Tuple
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

    def get_n_mean_and_sddev(self):
        n = self._n
        m = self._sum / self._n
        v = self._sum_squares / self._n - m**2
        return n, m, np.sqrt(v)

    def get_mean_and_sddev(self):
        _, m, s = self.get_n_mean_and_sddev()
        return m, s


class ABtest:
    @typechecked
    def __init__(self, weights: List[float], generators: List[Any]):
        assert len(generators) == len(weights)
        self._weights = np.array(weights) / sum(weights)
        self._trackers = [TrackOneStream(g) for g in generators]
        self._rng = np.random.RandomState()
        self._rng.seed(1337)

    @classmethod
    @typechecked
    def from_list_of_pairs(cls, lop: List[Tuple[float, Any]]):
        # converts a list-of-pairs to a pair-of-lists, and then
        # constructs an ABtest from them.
        args = [list(z) for z in zip(*lop)]
        return cls(*args)

    def advance(self):
        variant = self._rng.choice(len(self._weights), p=self._weights)
        self._trackers[variant].advance()

    @typechecked
    def get_ns_means_sddevs(self) -> List[List[Any]]:
        list_of_triples = [t.get_n_mean_and_sddev() for t in self._trackers]
        return [list(z) for z in zip(*list_of_triples)]
