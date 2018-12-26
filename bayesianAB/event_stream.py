from typeguard import typechecked
from numpy.random import RandomState

@typechecked
def gen_normals(loc: float, scale: float, seed: int):
    rng = RandomState()
    rng.seed(seed)
    while True:
        xs = rng.normal(loc, scale, 1000)
        for x in xs:
            yield x
