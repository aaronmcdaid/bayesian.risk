from bayesianAB.many_simulations import many_simulations_to_the_stopping_condition
def test_many_simulations_to_the_stopping_condition__seeded():
    RUNS = 100
    l = lambda seed: many_simulations_to_the_stopping_condition(RUNS, seed = seed)(
            min_sample_size = 10,
            stopping_condition = 'total_sample_size >= 25',
            means = [7, 7],
            stdevs = [1,1],
            weights = [0.5, 0.5],
            )
    df1 = l(1337)
    df2 = l(1337)
    df3 = l(None)
    df4 = l(None)
    EL1 = df1.groupby('run').tail(1).EL.values
    EL2 = df2.groupby('run').tail(1).EL.values
    EL3 = df3.groupby('run').tail(1).EL.values
    EL4 = df4.groupby('run').tail(1).EL.values

    # The first two are seeded the same, and hence should have identical results
    assert (EL1 == EL2).all()
    # The second two are unseeded, and will be different from each other
    # (unless we're extremely unlucky)
    assert not (EL3 == EL4).all()
