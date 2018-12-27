from typeguard import typechecked, Tuple
from bayesianAB.event_stream import ABtest
from bayesianAB.risk import risk
import numpy as np
import pandas as pd

def get_difference_of_means__and_its_sd(ab: ABtest) -> Tuple[float, float]:
    assert ab.get_number_of_variants() == 2
    (n1, n2), (m1, m2), (s1, s2) = ab.get_ns_means_sddevs()
    assert n1 > 1
    assert n2 > 1
    difference_of_means = m2 - m1
    pooled_variance = ( (s1 ** 2) * (n1 - 1) + (s2 ** 2) * (n2 - 1) ) / (n1 + n2 - 2)
    variance_of_estimate = pooled_variance / n1 + pooled_variance / n2
    return difference_of_means, np.sqrt(variance_of_estimate)


def get_expected_loss(ab: ABtest) -> float:
    diff, sd_of_diff = get_difference_of_means__and_its_sd(ab)
    return risk(0, diff, sd_of_diff)


def get_expected_gain(ab: ABtest) -> float:
    diff, sd_of_diff = get_difference_of_means__and_its_sd(ab)
    return -risk(0, -diff, sd_of_diff)


@typechecked
def many_iterations(ab: ABtest, steps: int) -> pd.DataFrame:
    rows = []
    for _ in range(steps):
        ab.advance()
        expected_loss = get_expected_loss(ab)
        expected_gain = get_expected_gain(ab)
        rows.append((expected_loss, expected_gain))
    df = pd.DataFrame.from_records(rows, columns = 'expected_loss expected_gain'.split())
    return df
