"""
Generators of small toy instances for Theorem F.13: sparse matrices, flat labels, sparse
test vectors and instances with a small sketched condition number.
"""

import numpy as np
import jax.numpy as jnp
from jax import random

from utilities import Connect
from reference import classical_reference

key = Connect()
import utils


def compute_lambda(sigma_max, sigma_min, k):
    """Regularization that gives condition number k: (s_max^2 + l)/(s_min^2 + l) = k^2.

    Args:
        sigma_max (float): largest singular value of X.
        sigma_min (float): smallest singular value of X (0 if D >= N).
        k (float): target condition number.

    Returns:
        float: lambda.
    """
    return (((sigma_min**2) * (k**2) - (sigma_max**2)))/(1 - k**2)


def small_instance_generator(key, N=4, D=6, s=2, kappa_target=None, k_reg_=None):
    """Random toy instance (X, y, lambda) for Theorem F.13.

    Args:
        key (jax.Array): PRNG key.
        N (int): number of rows (power of 2).
        D (int): number of columns.
        s (int): sparsity (the matrix has s * min(N, D) non-zero entries).
        kappa_target (float | None): if given, lambda gives exactly this k_reg; otherwise
            lambda is uniform in the interval that gives 2 <= k_reg <= 4.
        k_reg_ (bool | None): if truthy, k_reg is returned too.

    Returns:
        tuple: (X, y, _lambda), or (X, y, _lambda, k_reg) if k_reg_ is truthy.
    """
    assert N > 0 and np.log2(N).is_integer(), "N deve essere potenza di 2"

    key, subkey = random.split(key)
    A = utils.random_sparse_matrix(subkey, (N, D), s * min(N, D))
    key, subkey = random.split(key)
    y = utils.random_flat_vector(subkey, N).reshape(N)

    # singular values -> lambda range for 2 <= k_reg <= 4
    U, S, Vh = jnp.linalg.svd(A)
    S_max = S[0]
    S_min = S[-1] if D < N else 0
    a = compute_lambda(S_max, S_min, 2)
    b = compute_lambda(S_max, S_min, 4)
    if kappa_target is None:
        key, subkey = random.split(key)
        _lambda = random.uniform(subkey, shape=(), minval=min(a, b), maxval=max(a, b))
    else:
        _lambda = compute_lambda(S_max, S_min, kappa_target)

    k_reg = jnp.sqrt((S_max**2 + _lambda)/(S_min**2 + _lambda))
    if k_reg_:
        return A, y, _lambda, k_reg
    return A, y, _lambda


def sparse_vector_gen(key, N=6, s=2):
    """Random unit vector with exactly N - s entries set to zero.

    Args:
        key (jax.Array): PRNG key.
        N (int): dimension.
        s (int): number of non-zero entries.

    Returns:
        jax.Array: (1, N) s-sparse vector.
    """
    assert s <= N, "s deve essere minore o uguale alla dimensione del vettore"
    vec = utils.random_unit_vector(key, N)
    key, subkey = random.split(key)
    pos = random.choice(subkey, N, shape=(N-s,), replace=False)
    return vec.at[0, pos].set(0)


def istanza_kappa_piccolo(key, N=4, D=2, lam=0.9, kappa_poly_max=4, tentativi=500, m=5, overlap_min=0.45):
    """Instance with a small sketched condition number kappa_s = s_r s_c / sigma_min(A_aug).

    Draws instances until ceil(kappa_s) <= kappa_poly_max and m tests with overlap >= overlap_min exist.

    Args:
        key (jax.Array): PRNG key.
        N (int): number of rows.
        D (int): number of columns.
        lam (float): regularization (the one of small_instance_generator is ignored).
        kappa_poly_max (int): maximum ceil(kappa_s).
        tentativi (int): maximum number of draws.
        m (int): number of tests.
        overlap_min (float): minimum |<x^|w^>| of the tests.

    Returns:
        tuple: (X, y, lam, tests, r) with r the ClassicalReferenceResult on the tests.

    Raises:
        RuntimeError: if no instance is found in `tentativi` draws.
    """
    from predict import select_tests          # local import: predict imports this module

    for _ in range(tentativi):
        key, k_x, k_t = random.split(key, 3)
        X, y, _ = small_instance_generator(k_x, N, D)
        A_aug = np.vstack([np.asarray(X), np.sqrt(lam) * np.eye(D)])
        s_r = int((A_aug != 0).sum(1).max())
        s_c = int((A_aug != 0).sum(0).max())
        kappa_poly = int(np.ceil(s_r * s_c / np.linalg.svd(A_aug, compute_uv=False).min() - 1e-6))
        if kappa_poly > kappa_poly_max:
            continue
        try:
            tests, r = select_tests(k_t, X, y, lam, m, overlap_min)
        except AssertionError:                # not enough tests with the required overlap
            continue
        print(f"istanza: s_r = {s_r}, s_c = {s_c}, kappa_poly = {kappa_poly}, kappa_reg = {float(r.k_reg):.3f}")
        return X, y, lam, tests, r
    raise RuntimeError(f"nessuna istanza con kappa_poly <= {kappa_poly_max} in {tentativi} tentativi")


"""
Tests
"""

def _test_small_toy_exp(key):
    """Prints one instance and ten sparse test vectors.

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    key, subkey = random.split(key)
    A, y, _lambda = small_instance_generator(subkey)
    print("Matrice A: ", A)
    print("y = ", y)
    print("lambda = ", _lambda)
    print("---"*10)
    for i in range(10):
        key, subkey = random.split(key)
        v = sparse_vector_gen(subkey)
        print("\n", v)


def _test_classical_reference(key):
    """Prints the classical reference of one instance with one test vector.

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    key, subkey = random.split(key)
    A, y, _lambda = small_instance_generator(subkey)
    key, subkey = random.split(key)
    x_test = sparse_vector_gen(subkey)
    res = classical_reference(A, y, _lambda, x_test)
    print(res)


if __name__ == "__main__":
    print('---'*10)
    print("small_toy_exp.py")
    print('---'*10)
    #_test_small_toy_exp(key)
    _test_classical_reference(key)
