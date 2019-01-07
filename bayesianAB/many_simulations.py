from typeguard import typechecked, Optional, List
from bayesianAB.event_stream import one_simulation_until_stopping_condition
import pandas as pd

@typechecked
def _one_simulation(run: int, trace: bool, **kw):
    """
        To save a little memory, we return only four columns from the simulation:
         - EL
         - EG
         - total_sample_size
         - 'run' (this is just the 'run' parameter to this function)
    """
    if trace: print(run, end=' ')
    sim = one_simulation_until_stopping_condition(**kw)
    data_to_keep = sim[['total_sample_size', 'EL', 'EG']].copy()
    data_to_keep['run'] = run
    return data_to_keep

@typechecked
def many_simulations_to_the_stopping_condition(runs: int, trace: bool = False, seed: Optional[int] = None):
    """
    Note that this is a 'curried' function. You call it like this:
        many_simulations_to_the_stopping_condition(3)(..other args...)
    where '3' is the number of runs. To get tracing for debugging purposes:
        many_simulations_to_the_stopping_condition(3, trace=True)(..other args...)

    - 'run' is the number of simulations runs to do.
    - 'trace' prints the run number between each simulation.
    - The '..other args..' are passed directly to 'one_simulation_until_stopping_condition'.
    - seed: If not None, then pass a different seed to each run, deterministically based
            on 'seed'
    """
    def _curried(**kw):
        many_runs = [_one_simulation(
                        run = run,
                        trace = trace,
                        seeds = None if seed is None else (seed + 2*run, seed + 2*run +1),
                        **kw,
                        )
                    for run in range(runs)]
        if trace: print()
        return pd.concat(many_runs).reset_index(drop=True)
    return _curried

@typechecked
def get_one_row_per_simulation(df: pd.DataFrame, new_stopping_condition: Optional[str] = None):
    """
    Given a dataframe like that from 'many_simulations_to_the_stopping_condition',
    return one row per 'run'.

    By default, we return the last row for each simulation, as that is the first
    row that satisfied the original stopping condition.

    However, if 'new_stopping_condition' is not None, then we retun the first
    row from each run that satisfies the 'new_stopping_condition'.

    NOTE: The assumption is that the 'new_stopping_condition' is stricter than the
    original stopping condition. In other words that there will be at least one
    row per 'run' that satisfies the 'new_stopping_condition'.
    """
    if new_stopping_condition is None:
        res = df.groupby('run').tail(1)
    else:
        res = df.query(new_stopping_condition).groupby('run').head(1)

    assert res['run'].nunique() == df['run'].nunique()

    return res

@typechecked
def _one_simulation_many_stopping_conditions(
        run: int,
        stopping_conditions: List[str],
        trace: bool,
        **kw):
    strictest_stopping_condition = stopping_conditions[0]
    sim = one_simulation_until_stopping_condition(
            stopping_condition = strictest_stopping_condition,
            **kw)
    if trace: print('{}({})'.format(run, sim.shape[0]), end=' ')

    one_row_per_stopping_condition = []
    for stop in stopping_conditions:
        x = sim.query(stop).head(1)
        assert x.shape[0] == 1, stopping_conditions # otherwise, the first condition isn't the strictest?
        x.insert(0, 'run', run)
        x.insert(0, 'stopping_condition', stop)
        one_row_per_stopping_condition.append(x)
    return pd.concat(one_row_per_stopping_condition)

@typechecked
def many_sims_many_stopping_conditions(
        runs: int,
        stopping_conditions: List[str],
        trace: bool = False,
        seed: Optional[int] = None,
        ):
    # the conditions should be unique
    assert len(stopping_conditions) == len(set(stopping_conditions))
    def _curried(**kw):
        many_runs = [_one_simulation_many_stopping_conditions(
                        run = run,
                        stopping_conditions = stopping_conditions,
                        trace = trace,
                        seeds = None if seed is None else (seed + 2*run, seed + 2*run +1),
                        **kw,
                        )
                    for run in range(runs)]
        if trace: print()
        return pd.concat(many_runs).reset_index(drop=True)
    return _curried
