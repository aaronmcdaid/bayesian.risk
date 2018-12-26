from typeguard import typechecked
from bayesianAB.event_stream import ABtest
import numpy as np

def get_difference_of_means__as_a_likelihood_function(ab: ABtest):
    assert ab.get_number_of_variants() == 2
    (n1, n2), (m1, m2), (s1, s2) = ab.get_ns_means_sddevs()
    assert n1 > 1
    assert n2 > 1
    difference_of_means = m2 - m1
    pooled_variance = ( (s1 ** 2) * (n1 - 1) + (s2 ** 2) * (n2 - 1) ) / (n1 + n2 - 2)
    variance_of_estimate = pooled_variance / n1 + pooled_variance / n2
    return difference_of_means, np.sqrt(variance_of_estimate)
