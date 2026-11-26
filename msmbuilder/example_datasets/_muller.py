"""Brownian dynamics on the Muller-Brown potential.

Numba port of the former _muller.pyx. Random numbers are drawn with NumPy
before the loop, as before, so trajectories for a given random_state are
unchanged. The kernel releases the GIL, which load_muller relies on to run
trajectories in a thread pool.
"""
import math

import numpy as np
from numba import njit
from sklearn.utils import check_random_state

_AA = np.array([-200.0, -100.0, -170.0, 15.0])
_aa = np.array([-1.0, -1.0, -6.5, 0.7])
_bb = np.array([0.0, 0.0, 11.0, 0.6])
_cc = np.array([-10.0, -10.0, -6.5, 0.7])
_XX = np.array([1.0, 0.0, -0.5, -1.0])
_YY = np.array([0.0, 0.5, 1.5, 1.0])


@njit(cache=True, nogil=True)
def _muller_potential(x, y, beta):
    value = 0.0
    for j in range(4):
        dx = x - _XX[j]
        dy = y - _YY[j]
        value += _AA[j] * math.exp(_aa[j] * dx * dx + _bb[j] * dx * dy + _cc[j] * dy * dy)
    return beta * value


@njit(cache=True, nogil=True)
def _propagate(r, x, y, thin, beta, dt, dt_sqrt_2d, min_x, max_x, min_y, max_y):
    n_steps = r.shape[0]
    saved = np.zeros(((n_steps + thin - 1) // thin, 2))
    save_index = 0
    for i in range(n_steps):
        gx = 0.0
        gy = 0.0
        for j in range(4):
            dx = x - _XX[j]
            dy = y - _YY[j]
            term = _AA[j] * math.exp(_aa[j] * dx * dx + _bb[j] * dx * dy + _cc[j] * dy * dy)
            gx += (2 * _aa[j] * dx + _bb[j] * dy) * term
            gy += (_bb[j] * dx + 2 * _cc[j] * dy) * term
        gx *= beta
        gy *= beta

        # Brownian update, reflecting at the box edges
        x = x - dt * gx + dt_sqrt_2d * r[i, 0]
        y = y - dt * gy + dt_sqrt_2d * r[i, 1]
        if x > max_x:
            x = 2 * max_x - x
        if x < min_x:
            x = 2 * min_x - x
        if y > max_y:
            y = 2 * max_y - y
        if y < min_y:
            y = 2 * min_y - y

        if i % thin == 0:
            saved[save_index, 0] = x
            saved[save_index, 1] = y
            save_index += 1
    return saved


def propagate(n_steps=5000, x0=None, thin=1, kT=1.5e4, dt=0.1, D=0.010,
              random_state=None, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
              max_y=np.inf):
    """Simulate Brownian dynamics on the Muller-Brown potential.

    Returns an array of shape (ceil(n_steps / thin), 2): the position after
    every ``thin``-th step, starting with the first. (Before MSMBuilder Neo
    this raised IndexError when ``thin`` did not divide ``n_steps``.)
    """
    random = check_random_state(random_state)
    if x0 is None:
        x0 = (-0.5, 0.5)
    n_steps, thin = int(n_steps), int(thin)
    dt, D = float(dt), float(D)
    r = random.randn(n_steps, 2)
    return _propagate(r, float(x0[0]), float(x0[1]), thin, 1.0 / float(kT), dt,
                      dt * math.sqrt(2 * D), float(min_x), float(max_x),
                      float(min_y), float(max_y))


def muller_potential(x, y, beta=1):
    """Muller potential.

    This function can take vectors (x,y) as input and will broadcast
    """
    return np.vectorize(lambda x, y, beta: _muller_potential(float(x), float(y), float(beta)))(x, y, beta)
