"""
Interferometric classical shadow: estimates Re<x'_j|w> for m test vectors from random
Clifford measurements of |w~> = (|0>|0> + |1>|w>)/sqrt(2), without reading |w> and without
a Hadamard test per test vector.
"""

import time

import numpy as np
import jax.numpy as jnp
from jax import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_clifford
from qiskit.circuit.library import StatePreparation

from utilities import Connect
import small_toy_exp as ste
from reference import classical_reference as ref
from ridge_solver import build_W, amplify_fixed_point

key = Connect()
from utils import random_unit_vector, suppress_stdout_stderr as suppress


def w_tilde_prep(N, D, w: jnp.ndarray):
    """Circuit that prepares (|0>|0> + |1>|w^>)/sqrt(2), with w in the slots [N+D, N+2D).

    Args:
        N (int): number of rows.
        D (int): number of columns.
        w (jnp.ndarray): (D,) normalized vector.

    Returns:
        QuantumCircuit: circuit on n_sys + 1 qubits (control = qubit 0).
    """
    n_sys = int(jnp.ceil(np.log2(N + 2*D)))
    w_exp = np.zeros(2**n_sys, dtype=complex)
    w_exp[N+D:N+2*D] = w
    cV = StatePreparation(w_exp).control(1)
    qc = QuantumCircuit(n_sys + 1)
    qc.h(0)
    qc.append(cV, [0] + list(range(1, n_sys + 1)))
    return qc


def shadow_measure(qc: QuantumCircuit, rng, T, stato=None):
    """T random-Clifford measurements: for each shot a Clifford C and the outcome b ~ |<b|C|psi>|^2.

    Args:
        qc (QuantumCircuit | None): circuit that prepares the state (ignored if stato is given).
        rng (np.random.Generator): random generator (Cliffords and outcomes).
        T (int): number of shots.
        stato (Statevector | None): state to measure, if already available.

    Returns:
        list[tuple]: T pairs (Clifford, outcome index b).
    """
    n = qc.num_qubits if stato is None else stato.num_qubits
    dati = []
    if stato is None:
        stato = Statevector(qc)
    for i in range(T):
        C = random_clifford(n, seed=rng)
        out = stato.evolve(C.to_circuit())
        b = rng.choice(2**n, p=np.abs(out.data)**2)
        dati.append((C, b))
    return dati


def classical_result(Xp, Xm, n_tot, dati: list):
    """Single-shot estimators (2^n + 1)(|<x+|C^dag|b>|^2 - |<x-|C^dag|b>|^2) of Re<x|w>.

    Args:
        Xp (np.ndarray): (m, 2^n_tot) states |x+> = (|0>|0> + |1>|x>)/sqrt(2).
        Xm (np.ndarray): (m, 2^n_tot) states |x-> = (|0>|0> - |1>|x>)/sqrt(2).
        n_tot (int): number of measured qubits.
        dati (list[tuple]): (Clifford, outcome) pairs of shadow_measure.

    Returns:
        jax.Array: (T, m) estimates, one row per shot.
    """
    results = []
    for (C, b) in dati:
        # C^dagger |b>: the snapshot state of this shot
        b_state = Statevector.from_int(b, dims=(2,) * n_tot)
        phi = jnp.asarray(b_state.evolve(C.adjoint().to_circuit()).data)
        sp = Xp @ phi.conj()
        sm = Xm @ phi.conj()
        results.append((2**n_tot + 1)*(abs(sp)**2 - abs(sm)**2))
    return jnp.stack(results, axis=0)


def embed_tests(x_batch, N, D, n_sys):
    """States |x+-> of the tests in the layout of w_tilde_prep (x in [N+D, N+2D), control = qubit 0).

    Args:
        x_batch (np.ndarray): (m, D) normalized tests.
        N (int): number of rows.
        D (int): number of columns.
        n_sys (int): number of system qubits.

    Returns:
        tuple: (X_p, X_m), each (m, 2^(n_sys+1)).
    """
    m = x_batch.shape[0]
    x_exp = np.zeros((m, 2**n_sys), dtype=complex)
    x_exp[:, N+D:N+2*D] = x_batch
    X_p = np.zeros((m, 2**(n_sys + 1)), dtype=complex)
    X_m = np.zeros_like(X_p)
    # control qubit 0 is the least significant bit: |0>|0> -> index 0, |1>|x> -> odd indices
    X_p[:, 0] = X_m[:, 0] = 1 / np.sqrt(2)
    X_p[:, 1::2] = x_exp / np.sqrt(2)
    X_m[:, 1::2] = -x_exp / np.sqrt(2)
    return X_p, X_m


def grafico(Ts, std_dev, mean_err):
    """Plots the per-shot std (vs sqrt(3)) and the RMS error of the mean with the fit err ~ sigma T^-nu.

    Args:
        Ts (np.ndarray): numbers of shots.
        std_dev (list[float]): per-shot standard deviation at each T.
        mean_err (list[float]): RMS error of the mean at each T.

    Returns:
        None
    """
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))

    axs[0].scatter(Ts, std_dev, marker='o', label="std per colpo")
    axs[0].axhline(np.sqrt(3), color='red', linestyle='--', label=r"$\sqrt{3}$")
    axs[0].set_xscale('log')
    axs[0].set_xlabel("shots")
    axs[0].legend()
    axs[0].set_title("Andamento della devazione standard")
    axs[0].grid(True, alpha=0.3)

    # fit log(err) = log(sigma) - nu log(T)
    (pend, log_s), cov = np.polyfit(np.log(Ts), np.log(mean_err), 1, cov=True)
    nu, dnu = -pend, float(np.sqrt(cov[0, 0]))
    print(f"nu = {nu:.3f} ± {dnu:.3f}, sigma = {np.exp(log_s):.3f}")

    axs[1].scatter(Ts, mean_err, marker='o', label="errore RMS della media")
    axs[1].plot(Ts, std_dev[-1] / np.sqrt(Ts), 'r--', label=r"$\sigma/\sqrt{T}$")
    axs[1].plot(Ts, np.exp(log_s) * np.asarray(Ts, float)**(-nu), 'k-', label=rf"fit: $\nu={nu:.2f}\pm{dnu:.2f}$")
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlabel("shots")
    axs[1].set_title(r"Andamento dell'errore della media: " r"$\epsilon = |\hat{o} - \operatorname{Re}[\langle x_i \mid w \rangle]|$")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


"""
Tests
"""

def _verify_proof(key):
    """Checks the identity Tr(O_x rho_w~) = Re<x|w> with O_x = |x+><x+| - |x-><x-|.

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    dim = 10
    key, subkey1, subkey2 = random.split(key, 3)
    w_ = jnp.asarray(random.normal(subkey1, (dim,)) + 1j * random.normal(subkey2, (dim,)))
    key, subkey1, subkey2 = random.split(key, 3)
    x_ = jnp.asarray(random.normal(subkey1, (dim,)) + 1j * random.normal(subkey2, (dim,)))
    zero = jnp.zeros((1,), dtype=jnp.complex64)
    one = jnp.ones((1,), dtype=jnp.complex64)

    # (|0>|0> +- |1>|v>)/sqrt(2) written as block vectors
    x_plus = jnp.concatenate((one, jnp.zeros((dim,), dtype=jnp.complex64), zero, x_,))/jnp.sqrt(2)
    x_minus = jnp.concatenate((one, jnp.zeros((dim,), dtype=jnp.complex64), zero, -x_,))/jnp.sqrt(2)
    w_tilde = jnp.concatenate((one, jnp.zeros((dim,), dtype=jnp.complex64), zero, w_,))/jnp.sqrt(2)
    O_x = (jnp.outer(x_plus, x_plus.conj()) - jnp.outer(x_minus, x_minus.conj()))

    Re_x_w = jnp.real(jnp.vdot(x_, w_))
    rho = jnp.outer(w_tilde, w_tilde.conj())
    trace = jnp.trace(O_x @ rho)
    assert jnp.allclose(Re_x_w, trace, atol=1e-6), f"Assert error: Re_x_w not close to trace!"
    print("_verify_proof() concluso con successo")


def _test_w_tilde_prep(key):
    """w_tilde_prep prepares the expected state.

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    key, subkey = random.split(key)
    X, y, _lambda = ste.small_instance_generator(subkey)
    N, D = X.shape
    key, subkey1, subkey2 = random.split(key, 3)
    w = jnp.asarray(random.normal(subkey1, (D,)) + 1j * random.normal(subkey2, (D,)))
    w = w/jnp.linalg.norm(w)

    qc = w_tilde_prep(N, D, w)
    stato = Statevector.from_instruction(qc).data

    # expected: 1/sqrt(2) at index 0, w_j/sqrt(2) at index 2(N+D+j)+1
    dim_sys = N + 2*D
    w_expected = np.zeros(2 * dim_sys, dtype=complex)
    w_expected[0] = 1 / np.sqrt(2)
    for j in range(D):
        i = N + D + j
        w_expected[2*i + 1] = np.asarray(w)[j] / np.sqrt(2)

    assert jnp.allclose(stato, w_expected, atol=1e-6), "State not prep. well!"
    print("_test_w_tilde_prep() concluso con successo")


def _test_classical_result(key):
    """Statistics of the estimator: per-shot std and decay of the error of the mean with T (plot).

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    key, subkey = random.split(key)
    X, y, _lambda = ste.small_instance_generator(subkey)
    N, D = X.shape
    key, subkey1, subkey2 = random.split(key, 3)
    w = jnp.asarray(random.normal(subkey1, (D,)) + 1j * random.normal(subkey2, (D,)))
    w = w/jnp.linalg.norm(w)

    qc = w_tilde_prep(N, D, w)
    n_tot = qc.num_qubits
    key, subkey = random.split(key)
    m = 10
    x_batch = np.array(random_unit_vector(subkey, D, m))
    Xp, Xm = embed_tests(x_batch, N, D, n_tot-1)
    expected = jnp.real(x_batch.conj() @ w)

    T_max = 3000
    n_T = 200
    data = shadow_measure(qc, np.random.default_rng(42), T_max)
    oper_array = np.array(classical_result(Xp, Xm, n_tot, data))
    # error and std on the first T shots of one trajectory
    Ts = np.unique(np.logspace(1, np.log10(T_max), n_T).astype(int))
    mean_err = [np.sqrt(np.mean((oper_array[:T].mean(axis=0) - expected)**2)) for T in Ts]
    std_dev = [oper_array[:T].std(axis=0).mean() for T in Ts]

    grafico(Ts, std_dev, mean_err)


def _test_interferometric_classical_shadow(key):
    """Checks Eq. F65: failure rate of the median of means at the theoretical K and N_g.

    The shadow is measured once (pool of T_pool shots) and resampled, to keep the cost low.

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    key, subkey = random.split(key)
    X, y, _lambda = ste.small_instance_generator(subkey)
    N, D = X.shape
    key, subkey1, subkey2 = random.split(key, 3)
    w = jnp.asarray(random.normal(subkey1, (D,)) + 1j * random.normal(subkey2, (D,)))
    w = w/jnp.linalg.norm(w)

    qc = w_tilde_prep(N, D, w)
    n_tot = qc.num_qubits
    key, subkey = random.split(key)
    m = 10
    x_batch = np.array(random_unit_vector(subkey, D, m))
    Xp, Xm = embed_tests(x_batch, N, D, n_tot-1)
    expected = jnp.real(x_batch.conj() @ w)

    Delta = [0.1, 0.05]
    Eps = [0.3, 0.2, 0.1]
    results = []
    # approximation: one pool of shots resampled inside the loops (T_pool = 10000)
    T_pool = 10000
    data = shadow_measure(qc, np.random.default_rng(42), T_pool)
    pool = np.asarray(classical_result(Xp, Xm, n_tot, data))
    n_rep = 50
    rng = np.random.default_rng(0)

    for delta in tqdm(Delta):
        for eps in tqdm(Eps):
            K = int(np.ceil(2*np.log(2*m/delta)))           # number of groups of the median
            N_g = int(np.ceil(34 * 6 / (eps**2)))           # shots per group
            errors = []
            cont_sup = 0
            for rep in tqdm(range(n_rep)):
                idx = rng.integers(0, T_pool, size=(K, N_g))
                estimate = np.median(pool[idx].mean(axis=1), axis=0)
                max_error = np.max(np.abs(estimate - expected))
                errors.append(max_error)
                if max_error > eps:
                    cont_sup += 1

            results.append({
                "eps": eps,
                "delta": delta,
                "K": K,
                "N_g": N_g,
                "failure_rate": cont_sup/n_rep,
                "mean_max_error": np.mean(errors),
                "max_error_observed": np.max(errors),
            })

    print(results)
    """
    results:
    [{'eps': 0.3, 'delta': 0.1, 'K': 11, 'N_g': 2267, 'failure_rate': 0.0, 'mean_max_error': np.float64(0.04841750009942704), 'max_error_observed': np.float64(0.07866212502267439)}, {'eps': 0.2, 'delta': 0.1, 'K': 11, 'N_g': 5100, 'failure_rate': 0.0, 'mean_max_error': np.float64(0.04662065654085408), 'max_error_observed': np.float64(0.06946420980437795)}, {'eps': 0.1, 'delta': 0.1, 'K': 11, 'N_g': 20400, 'failure_rate': 0.0, 'mean_max_error': np.float64(0.045360492827389784), 'max_error_observed': np.float64(0.05694647535336361)}, {'eps': 0.3, 'delta': 0.05, 'K': 12, 'N_g': 2267, 'failure_rate': 0.0, 'mean_max_error': np.float64(0.04597001005718928), 'max_error_observed': np.float64(0.0706946687036124)}, {'eps': 0.2, 'delta': 0.05, 'K': 12, 'N_g': 5100, 'failure_rate': 0.0, 'mean_max_error': np.float64(0.044879283759733185), 'max_error_observed': np.float64(0.06409059315120802)}, {'eps': 0.1, 'delta': 0.05, 'K': 12, 'N_g': 20400, 'failure_rate': 0.0, 'mean_max_error': np.float64(0.045699741559972314), 'max_error_observed': np.float64(0.05758178752749099)}]
    """


def _test_predictions(key):
    """Shadow on the exact |w>: F64 (|o_hat - overlap| <= eps) and classical signs on 3 instances.

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    m = 10
    start = time.perf_counter()
    for i in range(3):
        key, subkey = random.split(key)
        X, y, _lambda = ste.small_instance_generator(subkey)
        N, D = X.shape
        keys = random.split(key, 200)
        matr = np.array([np.asarray(ste.sparse_vector_gen(k))[0] for k in keys])

        # keep m candidate tests with a large overlap
        class_ref = ref(X, y, _lambda, matr)
        idx = np.where(np.abs(np.asarray(class_ref.inner_product)) >= 0.9)[0][:m]
        matr = matr[idx]

        qc = w_tilde_prep(N, D, class_ref.w_ket)
        n_tot = qc.num_qubits
        x_norm = np.array([v / jnp.linalg.norm(v) for v in matr])
        Xp, Xm = embed_tests(x_norm, N, D, n_tot-1)

        class_ref = ref(X, y, _lambda, matr)
        overlap = np.asarray(class_ref.inner_product)
        eps = np.min(np.abs(overlap)) / 3
        delta = 0.4
        # theoretical N_g
        N_g = int(np.ceil(204/(eps**2)))
        K = int(np.ceil(2*np.log(2*m/delta)))
        print(eps)
        print(N_g)
        print(K)
        print("INIZIO shadow_measure", flush=True)
        data = shadow_measure(qc, np.random.default_rng(42), N_g*K)
        print("FINE shadow_measure", flush=True)
        print("INIZIO classical_result", flush=True)
        est = np.asarray(classical_result(Xp, Xm, n_tot, data))
        print("FINE classical_result", flush=True)
        # median of means
        o_hat = np.median(est.reshape(K, N_g, -1).mean(axis=1), axis=0)
        y_hat = np.sign(o_hat)

        assert np.all(np.abs(o_hat - overlap) <= eps), "F64 not verified!"
        assert np.all(y_hat == class_ref.y_pred), "Predictions dont match with classical result!"

    fine = time.perf_counter()
    print("_test_prediction() concluso con successo")
    minuti, secondi = divmod(fine - start, 60)
    print(f"Tempo di esecuzione del test: {int(minuti)} min e {secondi:.2f}s")


def _test_real_resolver(key):
    """Full pipeline on the exact oracles: W -> fixed-point amplification -> shadow -> signs.

    Tested with N_g = 125 (sensitive to shadow_measure and classical_result).
    Expected output: Re P = 0.990, eps = 0.150, T = 1375, errore max = 0.070.

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    start = time.perf_counter()
    key, subkey = random.split(key)
    X, y, _lambda = ste.small_instance_generator(subkey)
    N, D = X.shape
    keys = random.split(key, 200)
    m = 10
    matr = np.array([np.asarray(ste.sparse_vector_gen(k))[0] for k in keys])

    class_ref = ref(X, y, _lambda, matr)
    idx = np.where(np.abs(np.asarray(class_ref.inner_product)) >= 0.45)[0][:m]
    matr = matr[idx]
    x_norm = np.array([v / jnp.linalg.norm(v) for v in matr])
    class_ref = ref(X, y, _lambda, matr)

    # amplified state V|0>
    with suppress():
        W, info = build_W(X, y, _lambda, 0.05)
    qc_V, info_V = amplify_fixed_point(W, info, 61, 0.1, 0.99)
    v = Statevector(qc_V).data
    print("\nV builded")
    n_V = qc_V.num_qubits

    # |w~> = (|0>|0> + |1> V|0>)/sqrt(2)
    w_tilde = np.zeros(2 ** (n_V + 1), dtype=complex)
    w_tilde[0] = 1 / np.sqrt(2)
    w_tilde[1::2] = v / np.sqrt(2)

    Xp, Xm = embed_tests(x_norm, N, D, n_V)
    x_exp = np.sqrt(2) * Xp[:, 1::2]                  # x^ in [N+D, N+2D), ancillas in 0
    target = np.real(x_exp.conj() @ v)                # = Re P * <x^|w^>
    w_exp = np.zeros(2 ** n_V)
    w_exp[N+D:N+2*D] = np.asarray(class_ref.w_ket)
    print("\nEmved_tests fatta")

    N_g = 125
    delta = 0.1
    K = int(np.ceil(2 * np.log(2 * m / delta)))
    print("\nComincio shadow")
    data = shadow_measure(None, np.random.default_rng(42), N_g*K, stato=Statevector(w_tilde))
    print("\nShadow finito")
    est = np.asarray(classical_result(Xp, Xm, n_V+1, data))
    print("\nClassical result")
    o_hat = np.median(est.reshape(K, N_g, m).mean(axis=1), axis=0)
    print("Mediana fatta")

    assert np.real(w_exp @ v) > 0, "Re P negativo: la fase di V|0> non e' controllata"
    print("\n primo test passato")

    eps = np.min(np.abs(target)) / 3
    assert np.all(np.abs(o_hat - target) <= eps), "F64 non verificata (shadow)"
    assert np.all(np.sign(o_hat) == np.asarray(class_ref.y_pred)), "predizioni diverse dal classico"
    print(f"_test_real_resolver() concluso: Re P = {np.real(w_exp @ v):.3f}, eps = {eps:.3f}, "
          f"T = {K*N_g}, errore max = {np.max(np.abs(o_hat - target)):.3f}")

    fine = time.perf_counter()
    minuti, secondi = divmod(fine - start, 60)
    print(f"Tempo di esecuzione del test: {int(minuti)} min e {secondi:.2f}s")


if __name__ == "__main__":
    print("---"*10)
    print("shadow.py")
    print("---"*10)
    #_verify_proof(key)
    #_test_w_tilde_prep(key)
    _test_classical_result(key)
    #_test_interferometric_classical_shadow(key)
    #_test_predictions(key)
    #_test_real_resolver(key)
