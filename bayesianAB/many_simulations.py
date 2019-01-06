from bayesianAB.event_stream import one_simulation_until_stopping_condition
import pandas as pd

def many_simulations_to_the_stopping_condition(runs, stopping_condition, means, stdevs, weights):

    def _one_simulation(run):
        print(run, end=' ')
        sim = one_simulation_until_stopping_condition(
            weights = weights,
            # the absolute value of the means doesn't matter, just the difference between them
            means = means,
            stdevs = stdevs,
            stopping_condition = stopping_condition,
            min_sample_size = 100,
        )
        data_to_keep = sim[['total_sample_size', 'EL', 'EG']].copy()
        data_to_keep['run'] = run
        print(data_to_keep.head())
        print(data_to_keep.tail())
        print(stopping_condition)
        return data_to_keep

    many_runs = [_one_simulation(run) for run in range(runs)]
    print()
    return pd.concat(many_runs).reset_index(drop=True)
