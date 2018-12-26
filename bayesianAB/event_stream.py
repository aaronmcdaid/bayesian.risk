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
    def __init__(self, generator):
        self._g = generator
        self._sum = 0.0
        self._sum_squares = 0.0
        self._n = 0
        self._requested_n = 0

        # We force two elements in, so that we can get a
        # Bessel-corrected variance estimate.
        # This increases self._n, but not self._requested_n
        self._advance_impl()
        self._advance_impl()
        assert 2, 0 == (self._n, self._requested_n)

    def advance(self, steps: int = 1):
        # Increase the requested_n by 'steps'.
        # Then, actually call _advance_impl if needed.
        #
        # We need this strange design to ensure that the
        # first two calls to 'advance' are ignored.
        assert self._n == max(2, self._requested_n)
        self._requested_n += steps
        while self._n < self._requested_n:
            self._advance_impl()

    def _advance_impl(self):
            x = next(self._g)
            self._n += 1
            self._sum += x
            self._sum_squares += x * x

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
    def __init__(self, weights: List[float], generators: List[Any], seed: int):
        assert len(generators) == len(weights)
        self._M = len(generators)
        self._weights = np.array(weights) / sum(weights)
        self._trackers = [TrackOneStream(g) for g in generators]
        self._rng = np.random.RandomState()
        self._rng.seed(seed)

    @classmethod
    @typechecked
    def from_list_of_pairs(cls, lop: List[Tuple[float, Any]], seed: int):
        # converts a list-of-pairs to a pair-of-lists, and then
        # constructs an ABtest from them.
        args = [list(z) for z in zip(*lop)]
        return cls(*args, seed)

    @classmethod
    @typechecked
    def from_simple_args(cls,
                        true_diff: float,
                        sd: float,
                        weight: float,
                        seed: int = 1337,
                        steps: int = 0,
                        ):
        # For the typical case with just two variants and the same standard deviation
        x = [
            (weight, gen_normals(0, sd, seed + 100000)),
            (1-weight, gen_normals(true_diff, sd, seed + 200000)),
        ]
        ab = ABtest.from_list_of_pairs(x, seed)
        ab.advance(steps = steps)
        return ab

    def advance(self, steps: int = 1):
        for _ in range(steps):
            variant = self._rng.choice(len(self._weights), p=self._weights)
            self._trackers[variant].advance()

    @typechecked
    def get_ns_means_sddevs(self) -> List[List[Any]]:
        list_of_triples = [t.get_n_mean_and_sddev() for t in self._trackers]
        return [list(z) for z in zip(*list_of_triples)]

    @typechecked
    def get_number_of_variants(self) -> int:
        return self._M
