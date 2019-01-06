from typeguard import typechecked, Optional
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
