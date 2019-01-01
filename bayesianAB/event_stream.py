from typeguard import typechecked, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from collections import namedtuple
from bayesianAB.risk import risk, risks
from bayesianAB.prior import Prior, FlatPrior

"""
    Given:
        - two pre-seeded instances of np.RandomState
        - a set of variants, with weights
        - (optionally) summary of any history
    We will generally a large chunk (thousands) of
    events with all the necessary sufficient statistics.
"""


ITEMS_PER_CHUNK = 100000
DEFAULT_MIN_SAMPLE_SIZE = 5 # every variant must have at least this many observations


class SimulationParams:
    """
        This records all the parameters needed to define and run a simulation.
        - The number of variants
        - The weights (normally 50/50) of the variants
        - The means and standard deviations of the metric, for each variant
        - The stopping condition
        - The two seeds (may None)
    """

    @typechecked
    def __init__(self,
                weights: List[float],
                means: List[float],
                stdevs: List[float],
                stopping_condition: str,
                seeds: Optional[Tuple[int, int]] = None,
                min_sample_size = DEFAULT_MIN_SAMPLE_SIZE,
                prior = FlatPrior(),
            ):
        # The three lists must be of the same size
        assert len(weights) == len(means)
        assert len(weights) == len(stdevs)

        self.weights = weights
        self.means = means
        self.stdevs = stdevs
        self.stopping_condition = stopping_condition
        self.raw_seeds = seeds
        self.min_sample_size = min_sample_size
        self.prior = prior


    @typechecked
    def get_two_seeded_generators(self) -> Tuple[np.random.RandomState, np.random.RandomState]:
        seeds = self.raw_seeds
        if seeds is None:
            seeds = (np.random.randint(10000), np.random.randint(10000))
        return seeded_RandomStates(seeds[0], seeds[1])


    @typechecked
    def to_SimulationParamsForOneChunk(self) -> 'SimulationParamsForOneChunk':
        return SimulationParamsForOneChunk(
                n = ITEMS_PER_CHUNK,
                M = len(self.weights),
                weights = self.weights,
                means = self.means,
                stdevs = self.stdevs,
                )

class SimulationParamsForOneChunk(namedtuple('SimulationParamsForOneChunk', 'n M weights means stdevs')):
    """ This is an 'internal' class, which describes there bare minimum
    needed to compute one 'chunk' of observations. It doesn't know anything
    about stopping conditions though."""
    pass



@typechecked
def seeded_RandomState(seed: int) -> np.random.RandomState:
    rng = np.random.RandomState()
    rng.seed(seed)
    return rng


def seeded_RandomStates(*seeds) -> tuple:
    return tuple([seeded_RandomState(seed) for seed in seeds])

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
        two_rngs: Tuple[np.random.RandomState, np.random.RandomState],
        params: SimulationParamsForOneChunk,
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
    rng_variant, rng_normals = two_rngs
    n = params.n
    M = params.M
    weights = params.weights
    means = params.means
    stdevs = params.stdevs

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
        two_rngs: Tuple[np.random.RandomState, np.random.RandomState],
        params: SimulationParamsForOneChunk,
        ):
    """
        We don't know in advance how many samples we'll need, so this returns
        a generator of dataframes, where each dataframe is cumulative and builds
        on the last dataframe.
    """
    last_row = None
    while True:
        simulated_dataframes = simulate_many_draws_for_many_variants(
                two_rngs,
                params,
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
    """
        This basically is the 'likelihood', just the estimate of the mean
        and the variance of that estimate.
        Later, this likelihood will be combined with the prior
        to compute the posterior.
    """
    difference_of_means = df.eval('estimated_mean_1 - estimated_mean_0')
    pooled_variance = df.eval("""(  estimated_variance_0 * (sample_size_0-1) \
                                  + estimated_variance_1 * (sample_size_1-1) \
                                 ) / (sample_size_0 + sample_size_1 - 2)
            """)
    variance_of_estimator = df.eval('@pooled_variance * (1/sample_size_0 + 1/sample_size_1)')

    df['difference_of_means'] = difference_of_means
    df['variance_of_estimator'] = variance_of_estimator

def _compute_the_prior_and_insert_the_posterior(df: pd.DataFrame, prior: Prior):

    # Compute the prior
    prior_mean, prior_stdev = prior.compute_prior(df)


    # Apply the prior
    # ===============
    #
    # Remember, the 'precision' is the reciprocal of the variance:
    #    https://en.wikipedia.org/wiki/Precision_(statistics)
    # In the Gaussian conjugate prior, the posterior precision is the sum of
    # the prior precision and the likelihood precision. And the posterior
    # mean is a weighted average (weighted by precision) of the prior mean
    # and the likelihood mean.
    # See 'Normal with known precision τ' here:
    #    https://en.wikipedia.org/wiki/Conjugate_prior#Continuous_distributions

    # Compute the precisions:
    prior_precision = 1/(prior_stdev**2)
    likelihood_precision = 1/df.variance_of_estimator
    posterior_precision = prior_precision + likelihood_precision

    posterior_mean_numerator = df.difference_of_means * likelihood_precision + prior_mean * prior_precision
    posterior_mean_denominator = likelihood_precision + prior_precision
    posterior_mean = posterior_mean_numerator / posterior_mean_denominator

    # Store the posterior
    df['posterior_mean'] = posterior_mean
    df['posterior_stdev'] = np.sqrt(1/posterior_precision)


def _insert_the_risk_regret_columns(df):
    diffs = df['posterior_mean']
    stdev_of_estimator = df['posterior_stdev']
    df['expected_loss'] = risks(0, diffs, stdev_of_estimator)
    df['expected_gain'] = - risks(0, - diffs, stdev_of_estimator)


@typechecked
def generate_cumulative_dataframes_with_extra_columns(two_rngs, params, prior: Prior):
    # Assuming exactly two variants for now, no idea how to extend this!
    for df in generate_cumulative_dataframes(two_rngs, params):
        df = df.copy(deep=False)
        assert 'sample_size_0' in df.columns
        assert 'sample_size_1' in df.columns
        assert 'sample_size_2' not in df.columns
        df.eval('total_sample_size = sample_size_0 + sample_size_1', inplace=True)
        _insert_the_mean_and_variance_columns(df)
        _insert_the_ttest_columns(df)
        _compute_the_prior_and_insert_the_posterior(df, prior)
        _insert_the_risk_regret_columns(df)
        yield df


def _generator_for_simple_dataframe_with_all_stats(sim_params: SimulationParams):
    # If either 'seeds' value is 'None', replace it with a random value
    seeds = sim_params.raw_seeds
    if seeds is None:
        seeds = (np.random.randint(10000), np.random.randint(10000))
    two_rngs = sim_params.get_two_seeded_generators() #seeded_RandomStates(seeds[0], seeds[1])
    params = sim_params.to_SimulationParamsForOneChunk()
    adjusted_stopping_condition = _adjust_condition_for_min_sample_size(sim_params.stopping_condition, sim_params.min_sample_size)
    prior = sim_params.prior
    for df in generate_cumulative_dataframes_with_extra_columns(two_rngs, params, prior):
        matching_indices = df.index[df.eval(adjusted_stopping_condition)].tolist()
        if matching_indices == []:
            yield df
        else:
            first_matching_index = matching_indices[0]
            yield df.iloc[0:first_matching_index+1,]
            break


@typechecked
def _adjust_condition_for_min_sample_size(stopping_condition: str, min_sample_size: int) -> str:
    if min_sample_size > 0:
        stopping_condition = '({stopping_condition}) & sample_size_0 >= {min_sample_size} & sample_size_1 >= {min_sample_size}'.format(**locals())
    return stopping_condition


def simple_dataframe_with_all_stats(*l, **kw) -> pd.DataFrame:
    """

            DEPRECATE this? It's a bit redundant. Maybe just call
            'run_simulation_until_the_end' instead.

        Keep generating cumulative dataframes until one row matching
        'stopping_condition' is found. Then concat all rows up to and including
        the first matching row and return the DataFrame

        Also, this will wait until every variant has at least
        'min_sample_size' samples.  The 'min_sample_size' parameter should
        always be at least 2, in order for the variance estimates to be
        reasonable.

        NOTE: The args are simply directly to the constructor for
        SimulationParams. This allows us to add more parameters
        without having to edit this function
    """
    sim_params = SimulationParams(*l, **kw)
    return run_simulation_until_the_end(sim_params)

def run_simulation_until_the_end(sim_params: SimulationParams) -> pd.DataFrame:
    """
        Keep generating cumulative dataframes until one row matching
        'stopping_condition' is found. Then concat all rows up to and including
        the first matching row and return the DataFrame

        Also, this will wait until every variant has at least
        'min_sample_size' samples.  The 'min_sample_size' parameter should
        always be at least 2, in order for the variance estimates to be
        reasonable.

        NOTE: The args are simply directly to the constructor for
        SimulationParams. This allows us to add more parameters
        without having to edit this function
    """
    gen = _generator_for_simple_dataframe_with_all_stats(sim_params)
    return pd.concat(gen).reset_index(drop=True)
