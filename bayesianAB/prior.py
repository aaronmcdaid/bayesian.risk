from typeguard import typechecked, Tuple
from collections import namedtuple
import pandas as pd
import numpy as np

class Prior:
    # subclasses of this will specify a method
    #
    #   compute_prior(df: pd.DataFrame) -> Tuple[float, float]:
    #
    # which will return the mean and stdev of the prior. The
    # dataframe is provided in order to allow that the prior
    # might actually be dependent on the data (e.g. it might
    # be a "unit information prior":
    #   https://aaronmcdaid.github.io/blog.posts/unit.information.priors/
    pass

class FlatPrior(Prior):
    @staticmethod
    def compute_prior(df: pd.DataFrame) -> Tuple[float, float]:
        return (0, float('inf'))

class FixedPrior(Prior):
    def __init__(self, prior_mean, prior_stdev):
        self.prior_mean = prior_mean
        self.prior_stdev = prior_stdev

    def compute_prior(self, df: pd.DataFrame) -> Tuple[float, float]:
        return (self.prior_mean, self.prior_stdev)
