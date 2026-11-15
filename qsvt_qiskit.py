"""
QSVT circuits in Qiskit: projector phases, QSVT sequence, real part through an LCU
(Hadamard test) and fast versions with the rzz trick. Checked against apply_qsvt of the repo.
"""

from collections.abc import Sequence

import numpy as np
import jax.numpy as jnp
from jax import random
from tqdm import tqdm
from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import UnitaryGate

from utilities import Connect
import small_toy_exp as ste
from block_encoding import block_encoding, extract_block, halmos_dilation_eigh

key = Connect()
from qsvt import get_qsvt_angles, apply_qsvt, get_qsvt_angles_sign, get_qsvt_angles_inverse
from utils import suppress_stdout_stderr as suppress


def projector_phase(qc, phi, ancille):
    """Projector phase Pi(phi) = e^{i phi (2 Pi - I)}: e^{i phi} if all ancillas are 0, e^{-i phi} otherwise.

    Args:
        qc (QuantumCircuit): circuit modified in place.
        phi (float): phase.
        ancille (Sequence[Qubit]): ancilla qubits that define the projector.

    Returns:
        QuantumCircuit: qc.
    """
    phi = float(phi)
    qc.global_phase -= phi
    # flip so that "all ancillas 0" becomes "all ones", then a multi-controlled phase 2 phi
    for anc in ancille:
        qc.x(anc)
    qc.mcp(2 * phi, ancille[:-1], ancille[-1])
    for anc in ancille:
        qc.x(anc)
    return qc


def qsvt_circuit(be_circ: QuantumCircuit, angles: jnp.ndarray, list_ancillas_idx: jnp.ndarray | None = None, conj: bool = False):
    """QSVT sequence Pi(phi_0) U Pi(phi_1) U ... U Pi(phi_d) with the repo angle convention.

    Args:
        be_circ (QuantumCircuit): block encoding (Hermitian, so U is used at every step).
        angles (array): d + 1 QSVT angles.
        list_ancillas_idx (array | None): indices of the block-encoding ancillas (default: last qubit).
        conj (bool): sign of the global phase (True for the sequence with angles -Phi).

    Returns:
        QuantumCircuit: QSVT circuit on the same qubits as be_circ.
    """
    num_qubits = be_circ.num_qubits
    angles = np.asarray(angles, dtype=float)
    if list_ancillas_idx is None:
        list_ancillas_idx = np.array([num_qubits - 1])
    qc = QuantumCircuit(num_qubits)
    ancs = [qc.qubits[i] for i in list_ancillas_idx]

    # angles are applied from phi_d down to phi_1, each followed by U; phi_0 last
    ang_zero = angles[0]
    angles = angles[::-1]
    for ang in angles[:-1]:
        qc = projector_phase(qc, ang, ancs)
        qc.compose(be_circ, inplace=True)
    projector_phase(qc, ang_zero, ancs)

    # global phase of the repo convention
    sign = +1 if conj else -1
    qc.global_phase += sign * jnp.pi/2*len(angles)
    return qc


def qsvt_circuit_real(be_circ: QuantumCircuit, angles: jnp.ndarray, list_ancillas_idx: jnp.ndarray | None = None):
    """Real part of the QSVT polynomial: LCU (V_Phi + V_-Phi)/2 with one extra qubit.

    Args:
        be_circ (QuantumCircuit): block encoding.
        angles (array): QSVT angles.
        list_ancillas_idx (array | None): ancillas of the block encoding (default: last qubit).

    Returns:
        QuantumCircuit: circuit on be_circ.num_qubits + 1 qubits (LCU qubit = last).
    """
    angles = np.asarray(angles, dtype=float)
    num_qubits = be_circ.num_qubits

    V = qsvt_circuit(be_circ, angles, list_ancillas_idx, conj=False).to_gate(label="V_Phi")
    V_conj = qsvt_circuit(be_circ, -angles, list_ancillas_idx, conj=True).to_gate(label="V_-Phi")

    # Hadamard test: control 0 -> V_Phi, control 1 -> V_-Phi
    qc = QuantumCircuit(num_qubits + 1)
    qc.h(num_qubits)
    qc.append(V.control(1, ctrl_state=0), [num_qubits] + list(range(num_qubits)))
    qc.append(V_conj.control(1, ctrl_state=1), [num_qubits] + list(range(num_qubits)))
    qc.h(num_qubits)
    return qc


def qsvt_circuit_real_fast_list(be_list: jnp.ndarray, angles: jnp.ndarray):
    """Fast real-part QSVT with a different block encoding at every query.

    The pair of projector phases controlled by the LCU qubit is a single rzz between the
    block-encoding ancilla (last qubit of each block) and the LCU qubit.

    Args:
        be_list (list[QuantumCircuit]): d block encodings with the same number of qubits.
        angles (array): d + 1 QSVT angles.

    Returns:
        QuantumCircuit: circuit on be_list[0].num_qubits + 1 qubits (LCU qubit = last).
    """
    angles = np.asarray(angles, dtype=float)
    num_qubits = be_list[0].num_qubits

    qc = QuantumCircuit(num_qubits + 1)
    qc.h(num_qubits)
    for ang, be in zip(angles[::-1][:-1], be_list[::-1]):
        qc.rzz(-2*ang, num_qubits-1, num_qubits)
        qc.compose(be, range(num_qubits), inplace=True)
    qc.rzz(-2*angles[0], num_qubits-1, num_qubits)
    # global phase of the repo convention, as a relative phase on the LCU qubit
    qc.rz(np.pi*len(angles), num_qubits)
    qc.h(num_qubits)
    return qc


"""
Helpers of the tests
"""

def poli_test_1(x):
    """Test polynomial 2x^2 - 1 (Chebyshev T_2).

    Args:
        x (float | array): argument.

    Returns:
        float | array: 2x^2 - 1.
    """
    return 2 * x**2 - 1


def be_from_hermitian(A):
    """Exact Halmos block encoding of a Hermitian A with ||A|| <= 1 (ancilla = last qubit).

    Args:
        A (np.ndarray): Hermitian matrix.

    Returns:
        QuantumCircuit: block-encoding circuit.
    """
    U = halmos_dilation_eigh(np.asarray(A))
    qc = QuantumCircuit(int(np.log2(len(U))))
    qc.append(UnitaryGate(U), qc.qubits)
    return qc


def hermitian_with_spectrum(eigs, seed=0):
    """Real symmetric matrix Q diag(eigs) Q^T with random orthogonal Q.

    Args:
        eigs (list[float]): eigenvalues.
        seed (int): seed of Q.

    Returns:
        np.ndarray: matrix with the given spectrum.
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(len(eigs), len(eigs))))
    return Q @ np.diag(eigs) @ Q.T


"""
Tests
"""

def _test_projector_phase(k_1: int, n_sys=1):
    """Compares Pi(phi) with the expected diagonal matrix.

    Args:
        k_1 (int): number of ancillas.
        n_sys (int): number of system qubits.

    Returns:
        None
    """
    qc = QuantumCircuit(n_sys + k_1)
    ancs: Sequence[Qubit] = qc.qubits[n_sys:]
    theta = jnp.pi/2 * 0.37
    qc = projector_phase(qc, theta, ancs)
    # +theta only on the all-zero ancilla block
    mask = jnp.concatenate([jnp.array([1.0]), -jnp.ones((2**k_1) - 1)])
    diag = np.diag(np.exp(1j * theta * np.repeat(mask, 2**n_sys)))

    assert jnp.allclose(Operator(qc).data, diag, atol=1e-6), f"Confronto Pi(phi) e diag errato"
    print(f"_test_projector_phase(), num ancilla qubit = {k_1}: Test concluso con successo")


def _test_qsvt_circuit_vs_repo(key, angles, batch: int):
    """qsvt_circuit vs apply_qsvt of the repo on `batch` random block encodings.

    Args:
        key (jax.Array): PRNG key.
        angles (array): QSVT angles.
        batch (int): number of instances.

    Returns:
        None
    """
    print("\n === Test confronto con repo ===\n")
    for i in tqdm(range(batch)):
        key, subkey = random.split(key)
        A, y, _lambda = ste.small_instance_generator(subkey)
        be_circ, alpha, H = block_encoding(A, _lambda)

        num_ancillas = 1                                  # the block-encoding ancilla
        V = apply_qsvt(Operator(be_circ).data, num_ancillas, angles)
        qc_qsvt = qsvt_circuit(be_circ, jnp.asarray(angles))
        err = Operator(qc_qsvt).data - V

        print(f"Errore massimo assoluto: {jnp.max(jnp.abs(err)):.3e}")
        print(f"Errore norma di Frobenius: {jnp.linalg.norm(err):.3e}")
        assert jnp.allclose(Operator(qc_qsvt).data, V, atol=1e-6), "qc_qsvt is not right implemented - test 1"

    print("\nTest concluso con successo")


def _test_qsvt_circuit_vs_spectral(key, angles, func, batch: int):
    """QSVT block vs func applied to the spectrum of H/alpha.

    Args:
        key (jax.Array): PRNG key.
        angles (array): QSVT angles of func.
        func (callable): polynomial implemented by the angles.
        batch (int): number of instances.

    Returns:
        None
    """
    print("\n === Test confronto con versione teorica a partire da funzione === \n")
    for i in tqdm(range(batch)):
        key, subkey = random.split(key)
        A, y, _lambda = ste.small_instance_generator(subkey)
        be_circ, alpha, H = block_encoding(A, _lambda)

        qc = qsvt_circuit(be_circ, angles, [be_circ.num_qubits-1])
        QSVT_block = jnp.real(extract_block(qc, be_circ.num_qubits-1))

        # func(H/alpha) through the eigendecomposition
        w, V = jnp.linalg.eigh(jnp.asarray(H/alpha))
        expected = (V * func(w)) @ V.conj().T
        err = np.abs(QSVT_block - expected).max()
        assert err < 1e-8, "qc_qsvt is not right implemented - test 2"

    print(f"\n Test concluso con successo")


def _test_sign():
    """Sign function with QSVT + LCU: (a) vs the repo, (b) vs scale * sign(A).

    Returns:
        None
    """
    with suppress():
        angles, scale = get_qsvt_angles_sign(degree=21, threshold=0.2)
    angles = jnp.asarray(angles)

    A = hermitian_with_spectrum([-0.9, -0.4, 0.35, 0.8])
    be_circ = be_from_hermitian(A)
    n_sys = be_circ.num_qubits - 1
    dim = 2**n_sys
    qc = qsvt_circuit_real(be_circ, angles, list_ancillas_idx=[be_circ.num_qubits-1])
    qc_block = extract_block(qc, n_sys)

    # (a) same as the real part of the repo QSVT
    V = np.asarray(apply_qsvt(jnp.asarray(Operator(be_circ).data), 1, jnp.asarray(angles)))
    err_a = np.abs(qc_block - np.real(V[:dim, :dim])).max()
    assert err_a < 1e-8, f"segno (a): errore {err_a:.2e}"

    # (b) close to the ideal sign function (eigenvalues away from the threshold)
    w, _V = np.linalg.eigh(np.asarray(A))
    tol = 5e-2
    err_b = np.abs(np.real(qc_block) - scale * ((_V * jnp.sign(w)) @ _V.conj().T)).max()
    assert err_b < tol, f"segno (b): errore {err_b:.2e} > {tol}"

    imag = np.abs(np.imag(V[:dim, :dim])).max()
    print(f"segno: query = {len(angles) - 1}, (a) {err_a:.1e}, (b) {err_b:.1e}, " f"|Im| senza LCU = {imag:.1e}")


def _test_inversa():
    """Inverse function with QSVT + LCU: (a) vs the repo, (b) vs scale * A^-1.

    Returns:
        None
    """
    with suppress():
        angles, scale = get_qsvt_angles_inverse(kappa=3, epsilon=0.05)

    A = hermitian_with_spectrum([-0.9, -0.4, 0.35, 0.8])
    be_circ = be_from_hermitian(A)
    n_sys = be_circ.num_qubits - 1
    dim = 2**n_sys
    qc = qsvt_circuit_real(be_circ, angles, list_ancillas_idx=[be_circ.num_qubits-1])
    qc_block = extract_block(qc, n_sys)

    # (a) same as the real part of the repo QSVT
    V = np.asarray(apply_qsvt(jnp.asarray(Operator(be_circ).data), 1, jnp.asarray(angles)))
    err_a = np.abs(qc_block - np.real(V[:dim, :dim])).max()
    assert err_a < 1e-8, f"ivnversa (a): errore {err_a:.2e}"

    # (b) close to scale * A^-1 (|eigenvalues| >= 1/kappa)
    w, _V = np.linalg.eigh(np.asarray(A))
    tol = 5e-2
    err_b = np.abs(np.real(qc_block) - scale * ((_V * (1/w))) @ _V.conj().T).max()
    assert err_b < tol, f"inversa (b): errore {err_b:.2e} > {tol}"

    imag = np.abs(np.imag(V[:dim, :dim])).max()
    print(f"inversa: query = {len(angles) - 1}, (a) {err_a:.1e}, (b) {err_b:.1e}, " f"|Im| senza LCU = {imag:.1e}")


def _test_qsvt_circuit(key, func, batch: int):
    """Runs the QSVT tests: vs repo on 2x^2 - 1, sign, inverse (spectral test optional).

    Args:
        key (jax.Array): PRNG key.
        func (callable): polynomial of the spectral test.
        batch (int): number of instances.

    Returns:
        None
    """
    with suppress():
        angles = get_qsvt_angles(poli_test_1, degree=2, rescale=1.0, ensure_bounded=False)
    _test_qsvt_circuit_vs_repo(key, angles, batch)
    #_test_qsvt_circuit_vs_spectral(key, angles, func, batch)
    _test_sign()
    _test_inversa()


if __name__ == "__main__":
    print("---"*10)
    print("qsvt_qiskit.py")
    print("---"*10)
    #_test_projector_phase(1)
    #_test_projector_phase(3)
    _test_qsvt_circuit(key, poli_test_1, 5)
