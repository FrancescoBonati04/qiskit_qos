"""
Pure-NumPy statevector simulator (no Aer, no transpile) for the circuits of
binary_classification. Written because AerSimulator crashes at native level on Windows
(exit code -1073741819).

It enters recursively the boxes (W, U_A, O_col, StatePreparation, ...) and applies without
dense matrices the two large gates of the circuit, both diagonal:
    - mcphase  (projector phases on all qubits)
    - diagonal (aggregated kicks of the oracles)
Small gates (<= max_mat qubits) are applied with their matrix. The global phases of the
definitions are included (they matter for the real part extracted by the LCU).

Convention: Qiskit little-endian (qubit 0 = least significant bit of the index).
"""

import numpy as np
from qiskit import QuantumCircuit


def _applica_matrice(psi, mat, qubits, n):
    """Applies a 2^k x 2^k matrix (Qiskit convention on `qubits`) to the n-qubit state psi.

    Args:
        psi (np.ndarray): (2^n,) state.
        mat (np.ndarray): (2^k, 2^k) matrix.
        qubits (list[int]): target qubits (qubits[0] = least significant bit of the matrix).
        n (int): total number of qubits.

    Returns:
        np.ndarray: (2^n,) new state.
    """
    k = len(qubits)
    psi = psi.reshape([2] * n)                                  # axis a <-> qubit n-1-a
    m = np.asarray(mat, dtype=complex).reshape([2] * (2 * k))   # axes: (out_{k-1}..out_0, in_{k-1}..in_0)
    # contract the input axes of the matrix with the axes of the target qubits
    psi = np.tensordot(m, psi, axes=([2 * k - 1 - j for j in range(k)], [n - 1 - q for q in qubits]))
    psi = np.moveaxis(psi, list(range(k)), [n - 1 - q for q in reversed(qubits)])
    return psi.reshape(-1)


def _indice_locale(qubits, n):
    """For every basis state, its index restricted to `qubits` (qubits[0] = least significant bit).

    Args:
        qubits (list[int]): qubits of the gate.
        n (int): total number of qubits.

    Returns:
        np.ndarray: (2^n,) local indices.
    """
    idx = np.arange(2**n)
    loc = np.zeros(2**n, dtype=np.int64)
    for j, q in enumerate(qubits):
        loc |= ((idx >> q) & 1) << j
    return loc


def _esegui(psi, qc, qmap, n, max_mat=8):
    """Evolves psi with the circuit qc.

    Args:
        psi (np.ndarray): (2^n,) state.
        qc (QuantumCircuit): circuit to apply.
        qmap (list[int]): qmap[i] = global qubit of qubit i of qc.
        n (int): total number of qubits.
        max_mat (int): gates with at most this many qubits are applied with their matrix.

    Returns:
        np.ndarray: (2^n,) evolved state.

    Raises:
        ValueError: on measure / reset / save_statevector.
    """
    if qc.global_phase:
        psi = psi * np.exp(1j * float(qc.global_phase))
    for ins in qc.data:
        op = ins.operation
        qs = [qmap[qc.find_bit(q).index] for q in ins.qubits]

        if op.name == "barrier":
            continue

        if op.name == "mcphase":                                   # phase only where all qubits are 1
            idx = np.arange(2**n)
            mask = np.ones(2**n, dtype=bool)
            for q in qs:
                mask &= ((idx >> q) & 1) == 1
            psi = psi.copy()
            psi[mask] *= np.exp(1j * float(op.params[0]))
            continue

        if op.name == "diagonal":                                  # element-wise product
            psi = psi * np.asarray(op.params, dtype=complex)[_indice_locale(qs, n)]
            continue

        if op.name in ("measure", "reset", "save_statevector"):
            raise ValueError(f"'{op.name}' non è supportata: il simulatore restituisce lo statevector")

        # small gate: matrix; otherwise recurse into its definition
        mat = None
        if op.num_qubits <= max_mat:
            try:
                mat = op.to_matrix()
            except Exception:                                      # boxes without a matrix
                mat = None
        if mat is not None:
            psi = _applica_matrice(psi, mat, qs, n)
        else:
            assert op.definition is not None, f"nessuna matrice né definizione per '{op.name}'"
            psi = _esegui(psi, op.definition, qs, n, max_mat)
    return psi


def simulate(qc, max_mat=8):
    """Statevector of qc from |0...0> (same ordering as Statevector(qc).data).

    Args:
        qc (QuantumCircuit): circuit.
        max_mat (int): size threshold for applying gates by matrix.

    Returns:
        np.ndarray: (2^n,) final state.
    """
    n = qc.num_qubits
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1
    return _esegui(psi, qc, list(range(n)), n, max_mat)


def evolvi(sv, pezzo, n_qubit, max_mat=8):
    """Applies the circuit `pezzo` to the statevector sv: simulation piece by piece.

    Args:
        sv (np.ndarray | None): current state (None = |0...0>).
        pezzo (QuantumCircuit): piece of circuit on n_qubit qubits.
        n_qubit (int): number of qubits.
        max_mat (int): size threshold for applying gates by matrix.

    Returns:
        np.ndarray: (2^n_qubit,) evolved state.
    """
    if sv is None:
        sv = np.zeros(2**n_qubit, dtype=complex)
        sv[0] = 1
    return _esegui(np.asarray(sv, dtype=complex), pezzo, list(range(n_qubit)), n_qubit, max_mat)


"""
Test
"""

def _test_simulatore(seed=3, n=9):
    """Comparison with Qiskit Statevector on a circuit with the same gate types as binary_classification.

    Args:
        seed (int): seed of the random angles.
        n (int): number of qubits.

    Returns:
        None
    """
    from qiskit.circuit.library import DiagonalGate, StatePreparation
    from qiskit.quantum_info import Statevector

    rng = np.random.default_rng(seed)
    # box with global phase, diagonal gate and inverse state preparation
    riquadro = QuantumCircuit(4, name="box")
    riquadro.global_phase = 0.7
    riquadro.h(0)
    riquadro.rzz(0.4, 0, 3)
    riquadro.append(DiagonalGate(np.exp(1j * rng.uniform(0, 6, 8))), [1, 2, 3])
    riquadro.append(StatePreparation(np.ones(4) / 2).inverse(), [0, 2])
    riquadro.sdg(1)
    riquadro.ch(2, 0)
    g = riquadro.to_gate()

    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(rng.uniform(0, 3), q)
    qc.append(g, [5, 1, 7, 2])                       # box with global phase
    qc.append(g.inverse(), [0, 8, 3, 4])             # its inverse
    qc.append(g.control(1), [6, 0, 1, 2, 3])         # its controlled version
    qc.mcp(0.9, list(range(7)), 8)                   # multi-controlled phase
    qc.append(StatePreparation(np.ones(4) / 2), [4, 6])
    qc.crz(0.3, 2, 7)
    qc.global_phase = 0.2

    err = np.max(np.abs(Statevector(qc).data - simulate(qc)))
    assert err < 1e-10, f"simulatore sbagliato: errore {err:.2e}"

    # piece-by-piece simulation = whole simulation
    meta = len(qc.data) // 2
    a, b = QuantumCircuit(n), QuantumCircuit(n)
    for ins in qc.data[:meta]:
        a.append(ins.operation, [qc.find_bit(q).index for q in ins.qubits])
    for ins in qc.data[meta:]:
        b.append(ins.operation, [qc.find_bit(q).index for q in ins.qubits])
    b.global_phase = qc.global_phase
    err_pezzi = np.max(np.abs(evolvi(evolvi(None, a, n), b, n) - simulate(qc)))
    assert err_pezzi < 1e-10, f"evolvi a pezzi sbagliato: errore {err_pezzi:.2e}"

    print(f"_test_simulatore() ok: errore {err:.1e}, a pezzi {err_pezzi:.1e}")


if __name__ == "__main__":
    _test_simulatore()
