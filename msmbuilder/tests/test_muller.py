import numpy as np
from msmbuilder.example_datasets import MullerPotential, load_muller
from msmbuilder.utils import array2d


def test_func():
    xx = load_muller(random_state=1110102)['trajectories']
    assert len(xx) == 10
    assert xx[0].ndim == 2
    assert xx[0].shape[1] == 2
    array2d(xx)


def test_class():
    xx = MullerPotential(random_state=123122).get()['trajectories']
    assert len(xx) == 10
    assert xx[0].ndim == 2
    assert xx[0].shape[1] == 2
    array2d(xx)


def test_propagate_thin_not_dividing_n_steps():
    # A frame is saved at steps 0, thin, 2*thin, ...: ceil(n_steps / thin) of
    # them. msmbuilder2022 raised IndexError when thin did not divide n_steps.
    from msmbuilder.example_datasets._muller import propagate
    xs = propagate(n_steps=10, thin=3, random_state=0)
    assert xs.shape == (4, 2)
    full = propagate(n_steps=10, thin=1, random_state=0)
    np.testing.assert_array_equal(xs, full[::3])
