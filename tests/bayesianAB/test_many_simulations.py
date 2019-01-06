from bayesianAB.many_simulations import many_simulations_to_the_stopping_condition
def test_many_simulations_to_the_stopping_condition():
    df = many_simulations_to_the_stopping_condition(10, seed = 1337)(
            min_sample_size = 100,
            stopping_condition = 'EL > -0.01',
            means = [7, 7],
            stdevs = [1,1],
            weights = [0.5, 0.5],
            )
    print(df.groupby('run').tail(1))
