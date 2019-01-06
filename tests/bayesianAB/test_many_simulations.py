from bayesianAB.many_simulations import many_simulations_to_the_stopping_condition
def test_many_simulations_to_the_stopping_condition():
    many_simulations_to_the_stopping_condition(
            3,
            stopping_condition = 'EL < 0.01',
            means = [7, 7],
            stdevs = [1,1],
            weights = [0.5, 0.5]
            )



