from typeguard import typechecked, List, Any, Tuple
import numpy as np
import pandas as pd
from collections import namedtuple
from bayesianAB.core_standard_risk import fast_standard_risk
from bayesianAB.risk import risk

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


class SimulationDataFrames(namedtuple('SimulationDataFrames', 'assignment metric metric_squared')):
    pass


@typechecked
def simulate_many_draws_for_many_variants(
        rng_variant: np.random.RandomState,
        rng_normals: np.random.RandomState,
        n: int,
        M: int,
        weights: List[float],
        means: List[float],
        stdevs: List[float],
        ) -> SimulationDataFrames:
    """
        Returns three dataframes. Each dataframe has 'M' columns, where
        'M' is the number of variants. The number of rows in each is 'n',
        the number of samples that we draw.

        Three returned dataframes:
         - 'assignment_matrix' - if an observation 'i' is in variant 'j', then
           assignment_matrix[i,j] = 1. Otherwise, it's zero.
         - 'metric_matrix' - if an observation 'i' is in variant 'j', then
           metric_matrix[i,j] is a draw from the distribution for variant 'j'.
           Otherwise, it's zero.
         - 'metric_squared_matrix' - Same as 'metric_matrix', but squared.

        The reason we want this matrices is that we can perform a cumulative
        sum on them in order to get a running total, as each observation is
        observed, of all the relevant statistics (sample sizes, means,
        variances)
    """
    vs = random_variants(rng_variant, weights, n)
    standard_normals = random_standard_normals(rng_normals, n)

    assignment_matrix = one_column_per_variant(M, vs)
    metric_matrix = pd.concat([
            (assignment_matrix[j] * (standard_normals * stdevs[j] + means[j]))
        for j in range(M)], axis=1)
    metric_squared_matrix = metric_matrix ** 2

    return SimulationDataFrames(assignment_matrix, metric_matrix, metric_squared_matrix)


@typechecked
def generate_cumulative_dataframes(
        rng_variant: np.random.RandomState,
        rng_normals: np.random.RandomState,
        n: int,
        M: int,
        weights: List[float],
        means: List[float],
        stdevs: List[float],
        ):
    """
        We don't know in advance how many samples we'll need, so this returns
        a generator of dataframes, where each dataframe is cumulative and builds
        on the last dataframe.
    """
    last_row = None
    while True:
        simulated_dataframes = simulate_many_draws_for_many_variants(
                rng_variant,
                rng_normals,
                n,
                M,
                weights,
                means,
                stdevs,
                )
        assignment = simulated_dataframes.assignment.agg('cumsum')
        metric = simulated_dataframes.metric.agg('cumsum')
        metric_squared = simulated_dataframes.metric_squared.agg('cumsum')

        assignment = assignment.rename(lambda j: 'sample_size_' + str(j), axis=1)
        metric = metric.rename(lambda j: 'sum_' + str(j), axis=1)
        metric_squared = metric_squared.rename(lambda j: 'sumOfSquares_' + str(j), axis=1)
        one_chunk = pd.concat([assignment, metric, metric_squared], axis=1)
        if last_row is not None:
            one_chunk = one_chunk + last_row
        last_row = one_chunk.iloc[-1,]
        yield one_chunk
        #print(last_row)


def _insert_the_mean_and_variance_columns(df):
    for j in range(2):
        df.eval('estimated_mean_{} = sum_{} / sample_size_{}'.format(j, j, j), inplace=True)
        df.eval('estimated_variance_{} = sumOfSquares_{} / sample_size_{} - estimated_mean_{} ** 2'.format(j, j, j, j), inplace=True)


def _insert_the_ttest_columns(df):
    df.eval('difference_of_means = estimated_mean_1 - estimated_mean_0', inplace=True)
    pooled_variance = df.eval("""(  estimated_variance_0 * (sample_size_0-1) \
                                  + estimated_variance_1 * (sample_size_1-1) \
                                 ) / (sample_size_0 + sample_size_1 - 2)
            """)
    df.eval('variance_of_estimator = @pooled_variance * (1/sample_size_0 + 1/sample_size_1)', inplace=True)


def _insert_the_risk_regret_columns(df):
    # this is equivalent to calling
    #   risk(0, difference_of_means, sqrt(variance_of_estimator))
    # for every item
    diff_of_means = df['difference_of_means']
    stdev_of_estimator = np.sqrt(df['variance_of_estimator'])
    shifted_score_to_pass_to_standard_risk = - diff_of_means / stdev_of_estimator
    Risk = df.apply(lambda one_row: risk(0, one_row['difference_of_means'], np.sqrt(one_row['variance_of_estimator'])), axis=1)
    regret = - df.apply(lambda one_row: risk(0, - one_row['difference_of_means'], np.sqrt(one_row['variance_of_estimator'])), axis=1)
    df['risk'] = Risk
    df['regret'] = regret




def generate_cumulative_dataframes_with_extra_columns(*l, **kw):
    # Assuming exactly two variants for now, no idea how to extend this!
    for df in generate_cumulative_dataframes(*l, **kw):
        df = df.copy(deep=False)
        assert 'sample_size_0' in df.columns
        assert 'sample_size_1' in df.columns
        assert 'sample_size_2' not in df.columns
        _insert_the_mean_and_variance_columns(df)
        _insert_the_ttest_columns(df)
        _insert_the_risk_regret_columns(df)
        yield df


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
