from typeguard import typechecked, List, Any, Tuple
import numpy as np
import pandas as pd

"""
    Given:
        - two pre-seeded instances of np.RandomState
        - a set of variants, with weights
        - (optionally) summary of any history
    We will generally a large chunk (thousands) of
    events with all the necessary sufficient statistics.
"""


@typechecked
def seeded_RandomState(seed: int):
    rng = np.random.RandomState()
    rng.seed(seed)
    return rng

@typechecked
def random_variants(rng: np.random.RandomState, weights: List[float], n: int) -> pd.Series:
    variants = rng.choice(len(weights), p=weights, size=n)
    return pd.Series(variants, name='variant')


@typechecked
def one_column_per_variant(M: int, variants: pd.Series) -> pd.DataFrame:
    """
    M is the number of distinct variants. Although, remember that (by chance)
    some variants might be missing from 'variants'.
    Returns a DataFrame with M columns, one for each variant.
    The cells are 0/1, depending on which variant a given user has, i.e.
    df[i,j] = 1 is variants[j] == i
    """
    columns = [0 + (variants == v) for v in range(M)]
    df = pd.concat(columns, axis=1)
    df.columns = list(range(M))
    return df


@typechecked
def random_standard_normals(rng: np.random.RandomState, n: int) -> pd.Series:
    """
        These are all standard normals, i.e. N(0, 1). In another part of this
        system, these standard normals will be scaled and translated according
        to the variant.
    """
    norms = rng.normal(size=n)
    return pd.Series(norms, name='standard_normal')


@typechecked
def simulate_many_draws_for_many_variants(
        rng_variant: np.random.RandomState,
        rng_normals: np.random.RandomState,
        n: int,
        M: int,
        weights: List[float],
        means: List[float],
        stdevs: List[float],
        ) -> pd.DataFrame:
    vs = random_variants(rng_variant, weights, n)
    standard_normals = random_standard_normals(rng_normals, n)

    assignment_matrix = one_column_per_variant(M, vs)
    assignment_matrix_renamed = assignment_matrix.rename(lambda col_name: 'assignment_' + str(col_name), axis=1)
    df = pd.concat([assignment_matrix_renamed], axis = 1)

    """
        At this point, df will look something like this, but we will
        add more columns later:

            variant  assignment_0  assignment_1
        0         0             1             0
        1         0             1             0
        2         0             1             0
        3         1             0             1
        4         1             0             1
        5         1             0             1
        6         0             1             0
        7         1             0             1
        8         1             0             1
        9         0             1             0
    """

    # next, append one column per variant, with the value of random metric.
    observed_metrics = [
            (df['assignment_' + str(j)] * (standard_normals * stdevs[j] + means[j])).rename('observation_' + str(j))
        for j in range(M)]
    # and also, the square of each metric
    observed_metrics_squared = [
            ((df['assignment_' + str(j)] * (standard_normals * stdevs[j] + means[j])) ** 2).rename('squared_observation_' + str(j))
        for j in range(M)]
    df = pd.concat([df] + observed_metrics + observed_metrics_squared, axis = 1)
    return df


@typechecked
def cumulate(df: pd.DataFrame): # -> pd.DataFrame:
    # drop the variant column, and return the rest aggregated
    # Also, rename them in order to make the names more meaningful
    df = df.agg('sum')
    def renamer(col_name):
        if col_name.startswith('assignment_'):
            return col_name.replace('assignment_', 'sample_size_')
        if col_name.startswith('observation_'):
            return col_name.replace('observation_', 'sum_')
        if col_name.startswith('squared_observation_'):
            return col_name.replace('squared_observation_', 'sumOfSquares_')
        return col_name
    df = df.rename(renamer)
    return df



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
