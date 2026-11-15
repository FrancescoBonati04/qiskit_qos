"""
Exact (dense) block encoding of A_aug = [X; sqrt(lambda) I] through Hermitian and Halmos
dilations, and state preparation of y. Ideal black boxes used before the sketched oracles.
"""

import numpy as np
import jax.numpy as jnp
from jax import random
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator, state_fidelity
from qiskit.circuit.library import UnitaryGate, XGate, HGate, StatePreparation

from utilities import Connect
import small_toy_exp as ste

key = Connect()


def _is_unitary(circ):
    """Checks unitarity up to 1e-6.

    Args:
        circ (QuantumCircuit | np.ndarray): circuit or matrix.

    Returns:
        bool: True if U U^dagger = I.
    """
    U = Operator(circ).data
    return np.allclose(U @ U.conj().T, np.eye(U.shape[0]), atol=1e-6)


def extract_block(circ: QuantumCircuit, n_sys: int):
    """Top-left block of the unitary of circ (ancillas = most significant qubits in |0>).

    Args:
        circ (QuantumCircuit): circuit.
        n_sys (int): number of system qubits.

    Returns:
        np.ndarray: (2^n_sys, 2^n_sys) block.
    """
    U = Operator(circ).data
    return U[:2**n_sys, :2**n_sys]


def hermitian_dilation(X: jnp.ndarray):
    """Hermitian dilation [[0, X], [X^T, 0]] padded with zeros to the next power of 2.

    Args:
        X (jnp.ndarray): (N, D) matrix.

    Returns:
        jax.Array: Hermitian matrix of size 2^ceil(log2(N+D)).
    """
    N, D = X.shape
    zeros_down = jnp.zeros((D, D))
    zeros_up = jnp.zeros((N, N))
    next_power = 2 ** int(jnp.ceil(jnp.log2(N + D)))
    X_ = jnp.block([[zeros_up, X], [X.T, zeros_down]])
    padding = next_power - (N + D)
    return jnp.pad(X_, (0, padding))


def halmos_dilation_eigh(H):
    """Halmos unitary dilation [[H, S], [S, -H]] with S = sqrt(1 - H^2) (H Hermitian, ||H|| <= 1).

    Args:
        H (np.ndarray): Hermitian matrix with spectrum in [-1, 1].

    Returns:
        np.ndarray: unitary of twice the size whose top-left block is H.
    """
    w, V = np.linalg.eigh(H)
    S = (V * np.sqrt(np.clip(1 - w**2, 0, None))) @ V.conj().T
    return np.block([[H, S], [S, -H]])


def block_encoding(X: jnp.ndarray, _lambda: float):
    """Exact block encoding of the Hermitian dilation of A_aug = [X; sqrt(lambda) I] divided by alpha.

    Args:
        X (jnp.ndarray): (N, D) matrix.
        _lambda (float): regularization strength.

    Returns:
        tuple: (qc, alpha, H) with qc the circuit of the unitary (ancilla = last qubit),
            alpha = ||A_aug||_2 the normalization and H the (unnormalized) Hermitian dilation.
    """
    N, D = X.shape
    X = jnp.vstack([X, jnp.sqrt(_lambda) * jnp.eye(D)])
    # normalization so that ||H / alpha|| = 1
    alpha = float(jnp.linalg.norm(X, ord=2))

    X_hermitian = hermitian_dilation(X)
    X_halmos = np.asarray(halmos_dilation_eigh(X_hermitian / alpha))
    gate = UnitaryGate(X_halmos)
    assert _is_unitary(X_halmos), "The matrix is not unitary"

    num_qubits = int(np.log2(X_halmos.shape[0]))
    qc = QuantumCircuit(num_qubits)
    qc.append(gate, qc.qubits)
    return qc, alpha, np.asarray(X_hermitian)


def check_block_encoding(qc, expected_block, alpha):
    """Checks that the top-left block of qc equals expected_block / alpha.

    Args:
        qc (QuantumCircuit): block-encoding circuit.
        expected_block (np.ndarray): unnormalized block.
        alpha (float): normalization.

    Returns:
        bool: True if the blocks agree up to 1e-6.
    """
    expected_block = np.asarray(expected_block, dtype=complex)
    n_sys = int(np.log2(expected_block.shape[0]))
    actual_block = extract_block(qc, n_sys)
    return np.allclose(actual_block, expected_block / alpha, atol=1e-6)


def state_prep(X: jnp.ndarray, y: jnp.ndarray):
    """Exact preparation of |y~> = (y/||y||, 0) on ceil(log2(N + 2D)) qubits.

    Args:
        X (jnp.ndarray): (N, D) matrix (only its shape is used).
        y (jnp.ndarray): (N,) labels.

    Returns:
        tuple: (StatePreparation gate, y_enc amplitude vector).
    """
    N, D = X.shape
    y = np.asarray(y, dtype=complex).reshape(N)
    next_power = 2 ** int(np.ceil(np.log2(N + 2*D)))
    y_enc = np.zeros(next_power, dtype=complex)
    y_enc[:N] = y / np.linalg.norm(y)
    return StatePreparation(y_enc), y_enc


"""
Tests
"""

def _test_():
    """Checks _is_unitary and extract_block on a Hadamard.

    Returns:
        None
    """
    qc = QuantumCircuit(1)
    qc.h(0)
    assert _is_unitary(qc)
    A_comp = extract_block(qc, 1)
    A_expected = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]])
    assert np.allclose(A_comp, A_expected)
    print("Test passato")


def _test_qiskit():
    """Shows that the qubit order passed to append matters (X kron H on [q1, q0]).

    Returns:
        None
    """
    U = np.kron(Operator(XGate()).data, Operator(HGate()).data)
    qc = QuantumCircuit(2)
    q0 = qc.qubits[0]
    q1 = qc.qubits[1]
    qc.append(UnitaryGate(U), [q1, q0])
    print(np.allclose(Operator(qc).data, U, atol=1e-6))


def _test_block_encoding(key):
    """Unitarity and correct block on 5 random instances (N=4, D=6, s=2).

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    for i in range(5):
        key, subkey = random.split(key)
        A, y, _lambda = ste.small_instance_generator(subkey)
        qc, alpha, H = block_encoding(A, _lambda)
        assert _is_unitary(qc), f"istanza {i}: non unitaria"
        assert check_block_encoding(qc, H, alpha), f"istanza {i}: blocco != H/alpha"
    print("Test passato su un istanza di 5 unitarie con N=4, D=6, s=2")


def _test_state_prep(key):
    """Checks that U_A |y~> has X^T y / (||y|| alpha) in the w register on 5 instances.

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    for i in range(5):
        key, subkey = random.split(key)
        X, y, _lambda = ste.small_instance_generator(subkey)
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        N, D = X.shape
        qc_BE, alpha, H = block_encoding(X, _lambda)
        y_prep, _ = state_prep(X, y)
        n_sys = y_prep.num_qubits

        # prepare |y~>, apply the block encoding, keep the ancilla-0 part
        qc = QuantumCircuit(n_sys + 1)
        qc.append(y_prep, range(n_sys))
        qc.compose(qc_BE, inplace=True)
        out = Statevector(qc).data[:2**n_sys]

        expected = np.zeros(2**n_sys, dtype=complex)
        expected[N + D: N + 2*D] = X.T @ y / np.linalg.norm(y) / alpha
        assert np.allclose(out, expected, atol=1e-6), f"istanza {i}: U|y> != X^T y/alpha"

    print("Test passato su 5/5 instanze")


def _test_state_prep_2(key):
    """State preparation: correct state, inverse back to |0>, controlled version.

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    key, subkey = random.split(key)
    X, y, lam = ste.small_instance_generator(subkey)
    prep, y_enc = state_prep(X, y)
    n = prep.num_qubits
    zero = np.zeros(2**n, dtype=complex)
    zero[0] = 1

    assert np.isclose(state_fidelity(Statevector(prep), Statevector(y_enc)), 1.0)

    # prep followed by its inverse is the identity
    qc = QuantumCircuit(n)
    qc.append(prep, range(n))
    qc.append(prep.inverse(), range(n))
    assert np.isclose(state_fidelity(Statevector(qc), Statevector(zero)), 1.0)

    # controlled version: control 0 -> |0>, control 1 -> |y~>
    cprep = prep.control(1)
    for c in (0, 1):
        qc = QuantumCircuit(n + 1)
        if c == 1:
            qc.x(0)
        qc.append(cprep, range(n + 1))
        target = y_enc if c == 1 else zero
        expected = np.zeros(2**(n + 1), dtype=complex)
        expected[c::2] = target
        assert np.isclose(state_fidelity(Statevector(qc), Statevector(expected)), 1.0)
    print("state prep: preparazione, inversa, controllata ok")


if __name__ == "__main__":
    print('---'*10)
    print("block_encoding.py")
    print('---'*10)
    #_test_()
    #_test_qiskit()
    _test_block_encoding(key)
    _test_state_prep(key)
    _test_state_prep_2(key)
