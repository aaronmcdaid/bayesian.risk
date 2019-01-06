from bayesianAB.many_simulations import \
    many_simulations_to_the_stopping_condition, get_one_row_per_simulation, many_sims_many_stopping_conditions


def test_many_simulations_to_the_stopping_condition__seeded():
    RUNS = 10
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


def test_many_simulations_to_the_stopping_condition__stricter_stopping():
    ORIG_STOPPING_CONDITION = 'EL >= -0.01'
    NEW_STOPPING_CONDITION = 'EL >= -0.1'
    RUNS = 10
    df = many_simulations_to_the_stopping_condition(RUNS, seed = 1337)(
            min_sample_size = 10,
            stopping_condition = ORIG_STOPPING_CONDITION,
            means = [7, 7],
            stdevs = [1,1],
            weights = [0.5, 0.5],
            )
    orig_sample_sizes = get_one_row_per_simulation(df).total_sample_size.values
    new_sample_sizes = get_one_row_per_simulation(df, NEW_STOPPING_CONDITION).total_sample_size.values

    # As the second stopping condition is less strict, the simulations will
    # be shorter (or, very rarely, the same length).
    assert (new_sample_sizes <= orig_sample_sizes).all()
    assert new_sample_sizes.sum() < orig_sample_sizes.sum()


def test_many_sims_many_stopping_conditions():
    stopping_conditions = ['EL >= -0.01', 'EL >= -0.1', 'EL >= -0.3']
    strictest_stopping_condition = stopping_conditions[0]
    other_stopping_conditions = stopping_conditions[1:]
    RUNS = 10

    df = many_sims_many_stopping_conditions(
            RUNS,
            stopping_conditions = stopping_conditions,
            seed = 1337,
            )(
            min_sample_size = 10,
            means = [7, 7],
            stdevs = [1,1],
            weights = [0.5, 0.5],
            )
    # For each run, and each stopping_condition, there should be exactly one row
    assert df.shape[0] == RUNS * len(stopping_conditions)

    # The first stopping_condition is the strictest, therefore it should have
    # the largest sample sizes
    strictest_set = df.query('stopping_condition == @strictest_stopping_condition')
    for one_stopping_condition in other_stopping_conditions:
        one_set = df.query('stopping_condition == @one_stopping_condition')
        assert one_set.shape[0] == RUNS
        assert (one_set.total_sample_size.values <= strictest_set.total_sample_size.values).all()
        assert sum(one_set.total_sample_size.values) < sum(strictest_set.total_sample_size.values)
