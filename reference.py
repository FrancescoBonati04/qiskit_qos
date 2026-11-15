"""
Classical reference for the comparisons with the quantum algorithm: exact ridge solution,
condition number, margins, paper estimates (F58, F59, F69) and classical cost.
"""

import math
import time
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from sklearn.linear_model import RidgeClassifier

from utilities import Connect

key = Connect()
from utils import suppress_stdout_stderr as suppress


def get_exact_num_qubits(D, m, gamma_test, delta, s, b, k_reg):
    """Number of qubits of Eq. F58 (all constants set to 1).

    Args:
        D (int): number of features.
        m (int): number of test vectors.
        gamma_test (float): minimum test margin.
        delta (float): failure probability.
        s (int): sparsity.
        b (int): bits per entry.
        k_reg (float): regularized condition number.

    Returns:
        jax.Array: upper bound on the number of qubits.
    """
    return b + ((s*(jnp.log(D)**2)*(min(s*jnp.log(D), jnp.log(m)) + jnp.power(jnp.log(1/delta), 2.5)))/(gamma_test**2)) + (jnp.log(s*D)*jnp.log(k_reg) + jnp.power(jnp.log(k_reg), 2.5))


def get_exact_num_samples(R, N, s, k_reg, gamma_test, D, m, delta):
    """Number of samples of Eq. F59 (all constants set to 1).

    Args:
        R (float): multiplicative constant of F59.
        N (int): number of training points.
        s (int): sparsity.
        k_reg (float): regularized condition number.
        gamma_test (float): minimum test margin.
        D (int): number of features.
        m (int): number of test vectors.
        delta (float): failure probability.

    Returns:
        jax.Array: number of samples.
    """
    return (R*N*(s**7)*(k_reg**2)/(gamma_test**4 * delta))*(min(s**2 * jnp.power(jnp.log(D), 2), jnp.log(m)**2))


def compute_F69(k_reg, eps):
    """Query complexity of Eq. F69: k log(k) log(k/eps).

    Args:
        k_reg (float | array): condition number.
        eps (float | array): target precision.

    Returns:
        jax.Array: F69 value.
    """
    k_reg = jnp.asarray(k_reg, dtype=float)
    return k_reg * jnp.log(k_reg) * jnp.log(k_reg/eps)


@dataclass
class ClassicalReferenceResult:
    """Classical reference values of a ridge instance.

    Attributes:
        w: ridge solution (X^T X + lambda I)^-1 X^T y.
        w_ket: normalized w.
        y_pred: sign(<x'_j, w>) for every test.
        k_reg: sqrt(lambda_max / lambda_min) of X^T X + lambda I.
        gamma_j: margins |<x'_j, w>| / ||w||.
        gamma_test: minimum margin.
        s: sparsity of X (max over rows and columns).
        inner_product: <x'_j/||x'_j||, w_ket>.
        epsilon: gamma_test / (3 sqrt(s)).
    """
    w: jnp.ndarray
    w_ket: jnp.ndarray
    y_pred: jnp.ndarray
    k_reg: float
    gamma_j: jnp.ndarray
    gamma_test: jnp.ndarray
    s: int
    inner_product: float
    epsilon: float


def classical_reference(X: np.ndarray, y: np.ndarray, _lambda: float, test_vec: np.ndarray) -> ClassicalReferenceResult:
    """Exact classical solution of ridge regression and quantities of Theorem F.13.

    Args:
        X (np.ndarray): (N, D) training matrix with ||X|| <= 1 and |X_ij| <= 1.
        y (np.ndarray): (N,) labels.
        _lambda (float): regularization strength.
        test_vec (np.ndarray): (m, D) test vectors with ||x'||_inf <= 1.

    Returns:
        ClassicalReferenceResult: reference values.

    Raises:
        AssertionError: if the hypotheses of F.13 are not satisfied.
    """
    X = jnp.array(X)
    y = jnp.array(y)
    test_vec = jnp.array(test_vec)
    N, D = X.shape

    # hypotheses of F.13
    norm_X_2 = jnp.linalg.norm(X, 2)
    assert norm_X_2 <= 1+1e-6, f"||X|| > 1! ({norm_X_2})"
    max_X = jnp.max(jnp.abs(X))
    assert max_X <= 1+1e-6, f"max(X_ij) > 1! ({max_X})"
    max_test_vec = jnp.max(jnp.abs(test_vec))
    assert max_test_vec <= 1+1e-6, f"||x'_j||_inf > 1! ({max_test_vec})"
    assert math.log2(N).is_integer(), f"N must be a power of 2! Got {N}"

    # sparsity: max over rows and columns; tests must be at most s-sparse
    nnz_rows = jnp.count_nonzero(X, axis=1)
    nnz_cols = jnp.count_nonzero(X, axis=0)
    s = int(jnp.maximum(jnp.max(nnz_rows), jnp.max(nnz_cols)))
    for x in test_vec:
        nnz_test = jnp.count_nonzero(x)
        assert nnz_test <= s, f"x' must have at most {s} sparsity"

    # ridge solution and its normalized state
    X_T = X.T
    A = X_T @ X + _lambda * jnp.eye(D)
    w = jnp.linalg.solve(A, X_T @ y)
    w_norm = jnp.linalg.norm(w)
    w_ket = np.array(w / w_norm)
    y_i = jnp.sign(jnp.dot(test_vec, w))

    # condition number of the normal equations (square root -> of A_aug)
    eigen = jnp.linalg.eigvalsh(A)
    k_reg = float(jnp.sqrt(jnp.max(eigen)/jnp.min(eigen)))

    # margins and overlaps with the normalized tests
    gamma_j = jnp.abs(jnp.dot(test_vec, w)/w_norm)
    gamma_test = min(gamma_j)
    test_vec_norm = jnp.linalg.norm(test_vec, axis=1, keepdims=True)
    test_vec_normalized = test_vec / jnp.maximum(test_vec_norm, 1e-12)
    inner_product = test_vec_normalized @ w_ket
    epsilon = gamma_test/(3.0*jnp.sqrt(s))

    return ClassicalReferenceResult(
        w=w,
        w_ket=w_ket,
        y_pred=y_i,
        k_reg=k_reg,
        gamma_j=gamma_j,
        gamma_test=gamma_test,
        s=s,
        inner_product=inner_product,
        epsilon=epsilon,
    )


def classical_cost(X, y, _lambda, tests, ripetizioni=5):
    """Time and memory of the classical solution (sklearn RidgeClassifier).

    Args:
        X (np.ndarray): (N, D) training matrix.
        y (np.ndarray): (N,) labels.
        _lambda (float): regularization strength.
        tests (np.ndarray): (m, D) test vectors.
        ripetizioni (int): number of timed repetitions (the median is returned).

    Returns:
        dict: tempo_classico_s (median fit+predict time), mem_streaming_float (streaming lower
            bound, >= D floats, paper Sec. A), mem_sparsa_float (sparse-matrix algorithm, >= nnz
            floats), mem_streaming_bit (64 D), mem_csr_byte (X in CSR + w, y excluded).
    """
    tempi = []
    for _ in range(ripetizioni):
        t0 = time.perf_counter()
        RidgeClassifier(alpha=float(_lambda), fit_intercept=False).fit(X, y).predict(tests)
        tempi.append(time.perf_counter() - t0)
    N, D = X.shape
    nnz = int(np.count_nonzero(X))
    return dict(tempo_classico_s=float(np.median(tempi)),
                mem_streaming_float=D,
                mem_sparsa_float=nnz,
                mem_streaming_bit=64 * D,
                mem_csr_byte=12 * nnz + 4 * (N + 1) + 8 * D)


"""
Tests
"""

def _test_classical_reference_against_sklearn(seed=42):
    """Cross-check of classical_reference against sklearn RidgeClassifier on a random instance.

    Args:
        seed (int): seed of the random instance.

    Returns:
        float: max |w - w_sklearn|.
    """
    np.random.seed(seed)
    N = 8
    D = 4
    _lambda = 0.5

    # random instance rescaled to ||X|| = 1/2
    X_raw = np.random.randn(N, D)
    X = X_raw / (2.0 * np.linalg.norm(X_raw, 2))
    y = np.random.choice([-1.0, 1.0], size=N)
    X_test = np.random.uniform(-0.8, 0.8, size=(3, D))
    res = classical_reference(X, y, _lambda, X_test)

    clf = RidgeClassifier(alpha=_lambda, fit_intercept=False, solver="cholesky")
    clf.fit(X, y)
    sklearn_pred = clf.predict(X_test)

    err = float(np.max(np.abs(np.asarray(res.w) - clf.coef_.ravel())))
    np.testing.assert_array_equal(res.y_pred, sklearn_pred, err_msg="Le predizioni non coincidono con RidgeClassifier!")

    print("Controllo incrociato con sklearn completato con successo!")
    print(f"Sparsità s: {res.s}")
    print(f"Condition number k_reg: {res.k_reg:.4f}")
    print(f"Gamma test: {res.gamma_test:.4f}")
    return err


if __name__ == "__main__":
    print('---'*10)
    print("reference.py")
    print('---'*10)
    print(_test_classical_reference_against_sklearn())
    n_rep = 10000
    res = []
    for i in tqdm(range(n_rep)):
        with suppress():
            res.append(_test_classical_reference_against_sklearn(seed=i))   # a different instance each time

    print(f"Test superati: {len(res)}/{n_rep}")
    print(f"Errore medio: {np.mean(res):.3e}")
    print(f"Errore massimo: {np.max(res):.3e}")
    print(f"Deviazione standard: {np.std(res):.3e}")
