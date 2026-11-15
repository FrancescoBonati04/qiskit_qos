"""
Synthetic dataset that satisfies the hypotheses of Theorem F.13 (used by the final table of
binary_classification when real data are not available).
"""

import numpy as np
from jax import random

import small_toy_exp as ste
from predict import select_tests


def dataset_sintetico(key, N=2**12, D=2**8, s=4, lam=0.1, m=5, overlap_min=0.05, n_cand=2000):
    """Synthetic sparse instance (X, y, lambda, tests).

    lambda is chosen directly in (0, 1]: kappa_target does not work for tall well-conditioned
    matrices. overlap_min is small because in dimension D the typical overlap of a sparse test
    with w is ~ 1/sqrt(D).

    Args:
        key (jax.Array): PRNG key.
        N (int): number of rows (power of 2).
        D (int): number of columns.
        s (int): sparsity of X.
        lam (float): regularization in (0, 1] (needed by the element oracle).
        m (int): number of tests.
        overlap_min (float): minimum |<x^|w^>| of the tests.
        n_cand (int): number of candidate tests.

    Returns:
        tuple: (X, y, lam, tests) as NumPy arrays and float.
    """
    assert 0 < lam <= 1, f"serve 0 < lambda <= 1 (oracolo elemento), lambda = {lam}"
    k_x, k_t = random.split(key)
    X, y, _ = ste.small_instance_generator(k_x, N, D, s)           # only X and y are used
    tests, _ = select_tests(k_t, X, y, lam, m, overlap_min, n_cand)
    return np.asarray(X), np.asarray(y), float(lam), np.asarray(tests)
