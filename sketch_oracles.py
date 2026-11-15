"""
Core of Quantum Oracle Sketching: the black boxes used so far are replaced by circuits built
from classical samples:
    - U_y (so far a StatePreparation) -> flat state sketch of y (phase kicks);
    - U   (so far the Halmos UnitaryGate) -> block encoding U_A built from the sketched
      sparse oracles: element, row index and column index.
Also: gate/depth costs, the sweep over the number of samples M and its fit.
"""

import json
import time
from collections import Counter

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2, norm
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import StatePreparation, DiagonalGate, RZGate, IntegerComparatorGate

from utilities import Connect, save_angles
from qsvt_qiskit import qsvt_circuit_real_fast_list

key = Connect()
from utils import suppress_stdout_stderr as suppress, random_sparse_matrix, block_encoding_from_sparse_oracles, int_dtype, real_dtype
from qos_sampling import q_state_sketch_flat_unitary, q_state_sketch_flat, q_oracle_sketch_matrix_element, q_oracle_sketch_matrix_row_index, q_oracle_sketch_matrix_index
from data_generation import matrix_data, vector_data
from qsvt import get_qsvt_angles_sign, apply_qsvt

BASIS = ("cx", "rz", "sx", "x")
NOMI = ["U_y: infidelity", "elemento: max|sinB-A|", "indice riga: 1-|amp|^2", "indice colonna: 1-|amp|^2", "block encoding: ||.||_2 rel."]
P_ATTESO = [1.0, 0.5, 1.0, 1.0, 0.5]       # expected exponents of error ~ M^-p
EPS_COL = 1e-3                             # error budget of the sign in O_col (over the n_row steps)


def grado_segno_col(s, n_row, eps=EPS_COL, d_min=11, d_max=1001):
    """Degree, threshold and rescaling of the sign polynomial in O_col.

    Requires Re P(x) >= 1 - eps/n_row on ALL possible inputs x_m = |sin theta_m|,
    theta_m = pi (m + 1/2)/(2s + 1), m = 0..s-1 (not only on x_min).

    Args:
        s (int): column sparsity.
        n_row (int): number of row bits (steps of the binary search).
        eps (float): total error budget of the n_row steps.
        d_min (int): first degree tried.
        d_max (int): last degree tried (step 10).

    Returns:
        tuple: (degree, threshold, rescale).

    Raises:
        ValueError: if no degree <= d_max is enough.
    """
    x = np.sin(np.pi * (np.arange(s) + 0.5) / (2 * s + 1))
    x_min = float(x.min())
    eps_passo = eps / n_row
    thr = 0.5 * x_min
    rescale = 1 - eps_passo / 2
    for d in range(d_min, d_max + 1, 10):
        with suppress():
            ang, _ = save_angles(get_qsvt_angles_sign, degree=d, threshold=thr, rescale=rescale)
        ang = jnp.asarray(ang)
        # minimum of Re P over the possible inputs (repo QSVT on the Halmos dilation of x_m)
        P_min = min(complex(np.asarray(apply_qsvt(jnp.asarray(np.array([[xm, np.sqrt(1 - xm**2)],
                                                                         [np.sqrt(1 - xm**2), -xm]])), 1, ang))[0, 0]).real
                    for xm in x)
        if P_min >= 1 - eps_passo:
            return d, thr, rescale
    raise ValueError(f"nessun grado <= {d_max} basta per s = {s}, n_row = {n_row}, eps = {eps}")


def phase_kick(j, theta: float, n: int):
    """Single phase kick e^{i theta} on the basis state |j> of n qubits.

    Args:
        j (int): basis index.
        theta (float): phase.
        n (int): number of qubits.

    Returns:
        QuantumCircuit: X on the zero bits of j, multi-controlled phase, X again.
    """
    qc = QuantumCircuit(n, name="kick")
    zeros = [q for q in range(n) if not ((j >> q) & 1)]
    if zeros:
        qc.x(zeros)
    qc.mcp(theta, list(range(n-1)), n-1)          # phase on |1...1>
    if zeros:
        qc.x(zeros)
    return qc


def cost(kick, n_qubits, n_kicks):
    """Transpiled gate count of n_kicks kicks.

    Args:
        kick (bool): True = phase kick on n_qubits; False = RZ with n_qubits zero-controls
            (one sample of an LCU oracle).
        n_qubits (int): number of qubits of the kick (controls for kick=False).
        n_kicks (int): number of kicks.

    Returns:
        Counter: gates in BASIS.
    """
    if kick:
        qc = phase_kick(0, 0.1, n_qubits)
    else:
        qc = QuantumCircuit(n_qubits + 1)
        qc.append(RZGate(0.1).control(n_qubits, ctrl_state=0), range(n_qubits+1))
    cont = Counter(transpile(qc, basis_gates=list(BASIS)).count_ops())
    return Counter({g: k * n_kicks for g, k in cont.items()})


def profondita(qc):
    """Two-qubit depth and remaining depth of qc transpiled to BASIS.

    Args:
        qc (QuantumCircuit): circuit.

    Returns:
        np.ndarray: [depth of 2-qubit gates, total depth - 2-qubit depth].
    """
    qc_t = transpile(qc, basis_gates=list(BASIS))
    d = qc_t.depth()
    d2 = qc_t.depth(lambda ins: ins.operation.num_qubits == 2)
    return np.array([d2, d - d2], dtype=float)


def depth_kick(kick, n_qubits):
    """Depth of a single kick (same circuits as cost).

    Args:
        kick (bool): phase kick (True) or zero-controlled RZ (False).
        n_qubits (int): number of qubits / controls.

    Returns:
        np.ndarray: [2-qubit depth, other depth].
    """
    if kick:
        qc = phase_kick(0, 0.1, n_qubits)
    else:
        qc = QuantumCircuit(n_qubits + 1)
        qc.append(RZGate(0.1).control(n_qubits, ctrl_state=0), range(n_qubits + 1))
    return profondita(qc)


def depth_UA_query(n_campioni, dims, row_sparsity, kicks, degree=51, controlled=False):
    """Sequential depth of one query of U_A (element + row + column oracle, kicks in series).

    Args:
        n_campioni (int): samples per oracle (M).
        dims (tuple): (rows, columns) of the encoded matrix.
        row_sparsity (int): row sparsity.
        kicks (int): number of kicks of the row oracle.
        degree (int): degree of the sign in O_col.
        controlled (bool): one more control on every kick (shadow control).

    Returns:
        np.ndarray: [2-qubit depth, other depth].
    """
    N, D = dims
    c = int(controlled)
    n_rs = max(1, int(np.ceil(np.log2(row_sparsity))))
    n_row = int(np.ceil(np.log2(N)))
    n_col = int(np.ceil(np.log2(D)))
    comp = IntegerComparatorGate(n_row, 2**n_row - 1, geq=True)
    qc_c = QuantumCircuit(comp.num_qubits)
    qc_c.append(comp, range(comp.num_qubits))
    return (depth_kick(False, n_row + n_col + c) * n_campioni                    # element oracle
            + depth_kick(True, n_row + n_rs + 1 + c) * kicks                     # row oracle
            + depth_kick(False, n_col + 2 + c) * n_row * degree * n_campioni      # cumulative counters
            + profondita(qc_c) * 2 * n_row * degree * n_campioni)                 # comparators


def quantum_state_sketching(data, dim: int, kick: bool):
    """Flat state sketch U_y: H^n then the diagonal of the kicks e^{i pi dim/M} on the samples with value -1.

    Args:
        data (tuple): (sampled indices, sampled values +-1) of vector_data.get_data.
        dim (int): dimension of the vector (power of 2).
        kick (bool): not used (the kicks are always aggregated in one DiagonalGate).

    Returns:
        tuple: (gate "U_y", Counter of the gates of the individual kicks).
    """
    n_qubits = int(np.ceil(np.log2(dim)))
    assert dim == 2**n_qubits, f"Assert error: dimensione errarta in qss"
    sampled_index, sampled_data = data
    n_campioni = sampled_index.shape[0]
    qc = QuantumCircuit(n_qubits, name="U_y")
    qc.h(range(n_qubits))
    # each sample with value -1 adds a phase pi dim / M on its index
    phase = np.zeros(dim)
    np.add.at(phase, sampled_index, (1 - sampled_data) / 2)
    phase *= np.pi * dim / n_campioni
    qc.append(DiagonalGate(np.exp(1j*phase)), range(n_qubits))

    n_kicks = int(np.sum(sampled_data == -1))
    cont = cost(True, n_qubits, n_kicks)
    return qc.to_gate(), cont


def LCU_sin(phase, n_qubits, name: str):
    """Block encoding of diag(sin(phase)) with one ancilla: LCU of e^{i phase} and e^{-i phase}.

    Args:
        phase (np.ndarray): (2^n_qubits,) phases.
        n_qubits (int): number of system qubits.
        name (str): name of the circuit.

    Returns:
        QuantumCircuit: circuit on n_qubits + 1 qubits (ancilla = last).
    """
    qc = QuantumCircuit(n_qubits + 1, name=name)
    V = DiagonalGate(np.concatenate([np.exp(1j*phase), np.exp(-1j*phase)]))
    qc.sdg(n_qubits)
    qc.h(n_qubits)
    qc.x(n_qubits)
    qc.append(V, list(range(n_qubits)) + [n_qubits])
    qc.h(n_qubits)
    qc.sdg(n_qubits)
    qc.x(n_qubits)
    return qc


def element_oracle(row: int, col: int, val, dim, nnz):
    """Sketched element oracle: phases (nnz/M) sum arcsin(A_ij), turned into sin by the LCU.

    Args:
        row (array): sampled row indices.
        col (array): sampled column indices.
        val (array): sampled values (clipped to [-1, 1] before the arcsin).
        dim (tuple): (rows, columns).
        nnz (int): number of non-zero entries.

    Returns:
        tuple: (circuit on n_row + n_col + 1 qubits, dict with kicks and kick_qubits).
    """
    n_campioni = len(row)
    n_row = int(np.ceil(np.log2(dim[0])))
    n_col = int(np.ceil(np.log2(dim[1])))
    n_qubits = n_row + n_col
    phase = np.zeros((2**n_row, 2**n_col))
    np.add.at(phase, (row, col), np.arcsin(np.clip(np.asarray(val, dtype=float), -1.0, 1.0)))
    phase *= float(nnz) / n_campioni
    qc = LCU_sin(phase.ravel(), n_qubits, "element_oracle")
    return qc, dict(kicks=n_campioni, kick_qubits=n_row + n_col + 1)


def row_index_oracle(data, dims, s):
    """Sketched row-index oracle |i>|k>|0> -> |i>|k>|j(i,k)>: one Boolean phase oracle per output bit
    between two Hadamards (phase pi on the sampled rows whose k-th non-zero has that bit at 1).

    Indices beyond the row sparsity are padded with 2^b - 1.

    Args:
        data (tuple): (sampled rows, sampled row vectors) of matrix_data.get_row_data.
        dims (tuple): (N, D).
        s (int): row sparsity.

    Returns:
        tuple: (circuit "O_row" on b + k + i qubits with registers [out | K | I],
            dict with kicks and kick_qubits).
    """
    sampled_rows, sampled_vec = data
    N, D = dims
    n_campioni = len(sampled_rows)
    i = int(np.ceil(np.log2(N)))
    b = int(np.ceil(np.log2(D)))
    k = max(1, int(np.ceil(np.log2(s))))
    n_qubits = b + k + i
    out = list(range(b))
    K = list(range(b, b+k))
    I = list(range(b+k, n_qubits))

    qc = QuantumCircuit(n_qubits, name='O_row')

    # first s non-zero column indices of every sampled row (padding 2^b - 1), in bits
    max_v = 2**b-1
    nz = np.array([np.r_[np.flatnonzero(vec), np.full(s, max_v)][:s] for vec in sampled_vec])
    bits = (nz[..., None] >> np.arange(b - 1, -1, -1)) & 1
    bits = np.pad(bits, ((0, 0), (0, 2**k - s), (0, 0)))
    row_bits = np.zeros((2**i, 2**k, b))
    np.add.at(row_bits, sampled_rows, bits)
    phase = row_bits * np.pi*N/n_campioni

    # bit p: H, phase oracle controlled by the output qubit, H
    for p in range(b):
        gate = DiagonalGate(np.concatenate([np.ones(2**(i + k)), np.exp(1j*phase[:, :, p].ravel())]))
        qc.h(out[b-1-p])
        qc.append(gate, K + I + [out[b-1-p]])
        qc.h(out[b-1-p])

    return qc, dict(kicks=int(bits.sum()), kick_qubits=i+k+1)


def cumulative_counter_oracle(rows, cols, n_campioni, k, n_row, n_col, s, nnz):
    """Block encoding of sin(t C_j(l) - pi (k + 1/2)/(2s + 1)), with C_j(l) the sketched number of
    non-zeros of column j in rows < l.

    Args:
        rows (array): row indices of the samples.
        cols (array): column indices of the samples.
        n_campioni (int): number of samples (M).
        k (int): qubits of the rank register.
        n_row (int): row qubits.
        n_col (int): column qubits.
        s (int): column sparsity.
        nnz (int): number of non-zero entries.

    Returns:
        QuantumCircuit: LCU circuit on k + n_row + n_col + 1 qubits.
    """
    t = np.pi * nnz / (n_campioni * (2*s + 1))
    k_factor = np.pi / (2*s + 1)
    cont = np.zeros((2**n_col, 2**n_row))
    np.add.at(cont, (cols, rows), 1)
    C = np.cumsum(cont, axis=1) - cont                 # exclusive cumulative count over the rows
    k_range = np.arange(2**k)
    phase = t * C[:, None, :] - ((k_range[None, :, None] + 0.5)*k_factor)
    return LCU_sin(phase.ravel(), k+n_row+n_col, name="cumulative_counter_oracle")


def col_index_oracle(data, key, s: int, dims, nnz: int, n_campioni, degree: int = None):
    """Sketched column-index oracle |j>|k>|0> -> |j>|k>|i(j,k)> by in-place binary search.

    For every row bit (from the most significant) a sign QSVT on fresh cumulative counters decides
    whether the k-th non-zero of column j lies above the current prefix; a fresh ancilla pair
    (anc, lcu) is used for every bit.

    Args:
        data (matrix_data): sample generator of the matrix (uses _nz_rows, _nz_cols, _nnz and
            increases num_generated_samples).
        key (jax.Array): PRNG key of the samples.
        s (int): column sparsity.
        dims (tuple): (N, D).
        nnz (int): number of non-zero entries.
        n_campioni (int): samples per cumulative counter (M).
        degree (int | None): imposed degree of the sign (threshold pi/(4s+2), rescale 0.99, as in
            the repo); None = grado_segno_col.

    Returns:
        tuple: (circuit "O_col" on n_row + n_col + k + 1 + 2 n_row qubits, dict with blocks,
            samples_per_block, degree).
    """
    N, D = dims
    k = max(1, int(np.ceil(np.log2(s))))
    n_row = int(np.ceil(np.log2(N)))
    n_col = int(np.ceil(np.log2(D)))
    o = n_row+n_col+k
    L = list(range(n_row))
    K = list(range(n_row, n_row + k))
    J = list(range(n_row + k, o))

    if degree is None:
        degree, thr, rescale = grado_segno_col(s, n_row)
    else:                                             # imposed degree (repo convention, figures)
        thr, rescale = np.pi / (4 * s + 2), 0.99
    with suppress():
        angles, _ = save_angles(get_qsvt_angles_sign, degree=degree, threshold=thr, rescale=rescale)

    nnz_rows = np.asarray(data._nz_rows)
    nnz_cols = np.asarray(data._nz_cols)
    n_qubits = o + 2*n_row + 1
    qc = QuantumCircuit(n_qubits, name='O_col')

    for i, bit in enumerate(range(n_row-1, -1, -1)):
        # sign QSVT on `degree` fresh cumulative counters
        blocks = []
        for m in range(degree):
            key, subkey = random.split(key)
            idx = random.randint(subkey, (n_campioni,), 0, data._nnz, dtype=int_dtype)
            blocks.append(cumulative_counter_oracle(nnz_rows[idx], nnz_cols[idx], n_campioni, k, n_row, n_col, s, nnz))
        O_t = qsvt_circuit_real_fast_list(blocks, angles)
        anc, lcu = o + 1 + 2*i, o + 2 + 2*i
        # binary-search step: set the bit, controlled sign test on qubit o, swap the result into the bit
        qc.x(L[bit])
        qc.h(o)
        qc.compose(add_control(O_t), L + K + J + [anc, lcu] + [o], inplace=True)
        qc.h(o)
        qc.x(L[bit])
        qc.swap(L[bit], o)
        data.num_generated_samples += n_campioni * degree
    return qc, dict(blocks=degree * n_row, samples_per_block=n_campioni, degree=degree)


def q_oracle_sketch_matrix_index_grado(data_gen, key, unit_sample_size, dims, axis, sparsity, nnz, degree, threshold, rescale):
    """Copy of qos_sampling.q_oracle_sketch_matrix_index with degree, threshold and rescale of the
    sign as parameters (the repo fixes them to 51, pi/(4s+2), 0.99). Pure JAX, fast.

    Args:
        data_gen (matrix_data): sample generator of the matrix.
        key (jax.Array): PRNG key.
        unit_sample_size (int): samples per cumulative counter (M).
        dims (tuple): (N, D).
        axis (int): 0 = row-index oracle, 1 = column-index oracle.
        sparsity (int): sparsity along the chosen axis.
        nnz (int): number of non-zero entries.
        degree (int): degree of the sign.
        threshold (float): threshold of the sign.
        rescale (float): rescaling of the sign.

    Returns:
        jax.Array: (dims[axis], sparsity, dims[1-axis]) amplitudes of the oracle on |0>.
    """
    num_rows = dims[axis]
    bitlength_col = int(jnp.ceil(jnp.log2(dims[1 - axis])))
    num_cols = 2**bitlength_col
    unit_sample_size = int(unit_sample_size)
    k_indices = jnp.arange(sparsity, dtype=real_dtype) + 1
    t = jnp.pi * nnz / (2 * sparsity + 1)
    k_phase_scale = jnp.pi / (2 * sparsity + 1)
    nz_rows, nz_cols, nnz_entries = data_gen._nz_rows, data_gen._nz_cols, data_gen._nnz

    def _fase(sample_key):
        # phases of one sketched cumulative counter
        idx = random.randint(sample_key, shape=(unit_sample_size,), minval=0, maxval=nnz_entries, dtype=int_dtype)
        r, c = nz_rows[idx], nz_cols[idx]
        counts = jnp.zeros((num_rows, num_cols), dtype=real_dtype)
        counts = counts.at[r, c].add(1) if axis == 0 else counts.at[c, r].add(1)
        phase = (counts.cumsum(axis=1) - counts) * (t / unit_sample_size)
        phase = jnp.repeat(phase[:, None, :], sparsity, axis=1)
        phase = phase - (k_indices[None, :, None] - 0.5) * k_phase_scale
        return phase.reshape(num_rows * sparsity * num_cols)
    _fase = jax.jit(_fase)

    with suppress():
        angle_set, _ = save_angles(get_qsvt_angles_sign, degree=degree, threshold=threshold, rescale=rescale)
    angle_set = jnp.asarray(angle_set, dtype=real_dtype)
    hadamard = jnp.array([[1, 1], [1, -1]], dtype=real_dtype) / jnp.sqrt(2)
    state_lo = jnp.zeros((num_cols, 2), dtype=real_dtype).at[0, 0].set(1.0)
    state_lo = jnp.tile(state_lo[None, None], (num_rows, sparsity, 1, 1))
    dim = num_rows * sparsity * num_cols
    qsp_mask = jnp.array([1.0, -1.0], dtype=angle_set.dtype)
    phase0 = jnp.exp(1j * angle_set[0] * qsp_mask)
    global_phase = jnp.exp(1j * (-jnp.pi / 2) * angle_set.shape[0])

    for bit in range(bitlength_col - 1, -1, -1):
        # QSVT of the sign, one 2x2 block per basis state
        qsvt_circ = jnp.broadcast_to((global_phase * jnp.diag(phase0))[:, :, None], (2, 2, dim))
        for m in range(degree):
            key, subkey = random.split(key)
            ph = _fase(subkey)
            s_, c_ = jnp.sin(ph), jnp.cos(ph)
            u_gate = jnp.stack([jnp.stack([s_, c_], axis=0), jnp.stack([c_, -s_], axis=0)], axis=0)
            qsvt_circ = jnp.einsum("abm,bcm->acm", qsvt_circ, u_gate)
            qsvt_circ = qsvt_circ * jnp.exp(1j * angle_set[m + 1] * qsp_mask)[None, :, None]
        data_gen.num_generated_samples += unit_sample_size * degree
        # binary-search step on the classical array of amplitudes
        be_ = jnp.real(qsvt_circ[0, 0]).reshape(num_rows, sparsity, num_cols)
        xor = jnp.einsum("am,ijkm,mn->ijkan", hadamard, jnp.stack([jnp.ones_like(be_), be_], axis=-1), hadamard)
        high, low = 1 << (bitlength_col - bit - 1), 1 << bit
        sl = state_lo.reshape(num_rows, sparsity, high, 2, low, 2)[:, :, :, ::-1, :, :].reshape(num_rows, sparsity, num_cols, 2)
        sl = jnp.matvec(xor, sl)
        sl = sl.reshape(num_rows, sparsity, high, 2, low, 2)[:, :, :, ::-1, :, :]
        state_lo = sl.transpose(0, 1, 2, 5, 4, 3).reshape(num_rows, sparsity, num_cols, 2)
    return state_lo[:, :, : dims[1 - axis], 0]


def statepreparation(s, n):
    """Uniform superposition over the first s basis states of n qubits.

    Args:
        s (int): number of states.
        n (int): number of qubits.

    Returns:
        StatePreparation: the gate.
    """
    return StatePreparation(np.r_[np.ones(s), np.zeros(2**n - s)] / np.sqrt(s))


def block_encoding_from_oracles(subkey, n_campioni, data, data_row, data_ele, dims, col_sparsity, row_sparsity, nnz, stampa=False, degree_col=None):
    """Block encoding U_A of A from the sketched oracles (element, row index, column index).

    The index oracles do NOT uncompute the rank register (|j>|k>|0> -> |j>|k>|i(j,k)>): K_c and K_r
    are prepared uniform and projected back on the uniform state, so
        (<i|_I <0|_rest) U_A (|0>_I |j>_J |0>_rest) = A_ij / (s_r s_c),
    i.e. alpha = s_r s_c (Gilyen Lemma 48 / repo: sqrt(s_r s_c); Lemma D.22: s = max(s_r, s_c)).
    Valid if the padding indices point to zero entries (padding systematic).

    Args:
        subkey (jax.Array): PRNG key of the column oracle.
        n_campioni (int): samples per oracle (M).
        data (matrix_data): sample generator of the matrix (column oracle).
        data_row (tuple): samples of the row oracle.
        data_ele (tuple): samples (row, col, val) of the element oracle.
        dims (tuple): (N, D).
        col_sparsity (int): column sparsity.
        row_sparsity (int): row sparsity.
        nnz (int): number of non-zero entries.
        stampa (bool): print the build time.
        degree_col (int | None): imposed degree of the sign in O_col (51 = repo convention, for
            the comparisons); None = grado_segno_col.

    Returns:
        QuantumCircuit: "U_A" with registers [I | J | K_r | K_c | element anc | O_col ancillas].
    """
    start = time.perf_counter()
    N, D = dims
    row, col, val = data_ele
    O_ele, _ = element_oracle(row, col, val, dims, nnz)
    O_row, _ = row_index_oracle(data_row, dims, row_sparsity)
    O_col, _ = col_index_oracle(data, subkey, col_sparsity, dims, nnz, n_campioni, degree=degree_col)

    n_cs = max(1, int(np.ceil(np.log2(col_sparsity))))
    n_rs = max(1, int(np.ceil(np.log2(row_sparsity))))
    n_row = int(np.ceil(np.log2(N)))
    n_col = int(np.ceil(np.log2(D)))
    # registers
    I = list(range(n_row))
    J = list(range(n_row, n_row+n_col))
    Kr = list(range(n_row+n_col, n_row + n_col+n_rs))
    Kc = list(range(n_row + n_col+n_rs, n_row + n_col+n_rs+n_cs))
    anc = list(range(n_row + n_col+n_rs+n_cs+1, n_row + n_col+n_rs+n_cs+2+(2*n_row)))
    n_qubits = n_row + n_col+n_rs+n_cs+2+(2*n_row)
    a = n_row + n_col+n_rs+n_cs                      # ancilla of the element oracle

    # |j> -> sum_k |i(j,k)> -> A_ij -> uncompute towards the row index -> project the ranks
    qc = QuantumCircuit(n_qubits, name='U_A')
    qc.append(statepreparation(col_sparsity, n_cs), Kc)
    qc.compose(O_col.to_gate(label="O_col"), I+Kc+J+anc, inplace=True)
    qc.compose(O_ele.to_gate(label="O_ele"), J+I+[a], inplace=True)
    qc.append(statepreparation(row_sparsity, n_rs), Kr)
    qc.compose(O_row.inverse().to_gate(label="O_row^dagger"), J+Kr+I, inplace=True)
    qc.append(statepreparation(col_sparsity, n_cs).inverse(), Kc)
    qc.append(statepreparation(row_sparsity, n_rs).inverse(), Kr)

    if stampa:
        print(f"Tempo di esecuzione del block encoding dagli oracoli: {time.perf_counter() - start:.4f} secondi")
    return qc


"""
Factories of oracles built every time from fresh samples
"""

def make_Uy(vec_gen, n_campioni):
    """Factory of fresh U_y sketches.

    Args:
        vec_gen (vector_data): sample generator of y.
        n_campioni (int): samples per sketch (M).

    Returns:
        callable: new_Uy(key) -> (gate U_y, Counter of the gates).
    """
    dim = vec_gen.length
    def new_Uy(key):
        data = vec_gen.get_data(key, n_campioni)
        return quantum_state_sketching(data, dim, True)
    return new_Uy


def make_UA(matr_gen, n_campioni, dims, row_sparsity, col_sparsity, nnz):
    """Factory of fresh U_A block encodings.

    Args:
        matr_gen (matrix_data): sample generator of the matrix.
        n_campioni (int): samples per oracle (M).
        dims (tuple): (N, D).
        row_sparsity (int): row sparsity.
        col_sparsity (int): column sparsity.
        nnz (int): number of non-zero entries.

    Returns:
        callable: new_UA(key) -> QuantumCircuit U_A.
    """
    def new_UA(key):
        key, subkey1, subkey2 = random.split(key, 3)
        data_ele = matr_gen.get_matrix_element_data(subkey1, n_campioni)
        data_row = matr_gen.get_row_data(subkey2, n_campioni)
        return block_encoding_from_oracles(key, n_campioni, matr_gen, data_row, data_ele, dims, col_sparsity, row_sparsity, nnz)
    return new_UA


def controlled_gate(qc):
    """Singly-controlled version of a circuit or gate.

    Args:
        qc (QuantumCircuit | Gate): circuit or gate.

    Returns:
        ControlledGate: controlled gate (control = first qubit).
    """
    gate = qc.to_gate() if isinstance(qc, QuantumCircuit) else qc
    return gate.control(1)


def comulative_counter_cost(n, k):
    """Gate count of k comparators on n qubits (cost of the cumulative counters).

    Args:
        n (int): number of qubits compared.
        k (int): number of comparators.

    Returns:
        Counter: gates in BASIS.
    """
    gate = IntegerComparatorGate(n, 2**n-1, geq=True)
    qc = QuantumCircuit(gate.num_qubits)
    qc.append(gate, range(gate.num_qubits))
    cost = Counter(transpile(qc, basis_gates=BASIS).count_ops())
    return Counter({g: v * k for g, v in cost.items()})


def cost_UA_query(n_campioni, dims, row_sparsity, kicks, degree=51, controlled=False):
    """Gate count of one query of U_A (the three oracles, cumulative counters included).

    Args:
        n_campioni (int): samples per oracle (M).
        dims (tuple): (rows, columns) of the encoded matrix.
        row_sparsity (int): row sparsity.
        kicks (int): kicks of the row oracle.
        degree (int): degree of the sign in O_col.
        controlled (bool): one more control on every kick.

    Returns:
        Counter: gates in BASIS.
    """
    N, D = dims
    c = int(controlled)
    n_rs = max(1, int(np.ceil(np.log2(row_sparsity))))
    n_row = int(np.ceil(np.log2(N)))
    n_col = int(np.ceil(np.log2(D)))
    tot = Counter()
    tot += (cost(False, n_row+n_col+c, n_campioni)                            # element oracle
            + cost(True, n_row+n_rs+1+c, kicks)                               # row oracle
            + cost(False, n_col+2+c, n_row*degree*n_campioni)                 # cumulative counters
            + comulative_counter_cost(n_row, 2*n_row*degree*n_campioni))      # comparators
    return tot


def add_control(qc: QuantumCircuit):
    """Controlled version of qc built gate by gate (control = last qubit).

    DiagonalGate and rzz are controlled explicitly (cheaper than .control()); the global phase
    becomes a phase gate on the control.

    Args:
        qc (QuantumCircuit): circuit.

    Returns:
        QuantumCircuit: circuit on qc.num_qubits + 1 qubits.
    """
    n_qubits = qc.num_qubits
    out = QuantumCircuit(n_qubits + 1, name=f"{qc.name}")

    for instance in qc.data:
        oper = instance.operation
        idx = [qc.find_bit(q).index for q in instance.qubits]
        if oper.name == "diagonal":
            # identity when the control is 0
            d = np.asarray(oper.params, dtype=complex)
            out.append(DiagonalGate(np.concatenate([np.ones_like(d), d])), idx + [n_qubits])
        elif oper.name == "rzz":
            a, b = idx
            out.cx(a, b)
            out.crz(oper.params[0], n_qubits, b)
            out.cx(a, b)
        else:
            out.append(oper.control(1), [n_qubits] + idx)

    if qc.global_phase:
        out.p(qc.global_phase, n_qubits)
    return out


"""
Scaling with the number of samples
"""

def oracle_index_error(oracle, A):
    """Mean of 1 - |amplitude|^2 on the correct index of an index oracle (padding excluded).

    Args:
        oracle (np.ndarray): (rows, s, columns) amplitudes of the oracle.
        A (np.ndarray): matrix (rows, columns).

    Returns:
        float: mean error.
    """
    _, D = A.shape
    s = oracle.shape[1]
    expected = np.sort(np.where(A != 0, np.arange(D), D), axis=1)[:, :s]
    valid = expected < D
    amp = np.take_along_axis(oracle, np.minimum(expected, D - 1)[..., None], axis=-1)[..., 0]
    return float(np.mean((1 - np.abs(amp)**2)[valid]))


def to_jsonable(x):
    """Recursively converts arrays and NumPy scalars to Python types for json.dump.

    Args:
        x: object (dict, list, tuple, array, scalar).

    Returns:
        object: JSON-serializable copy.
    """
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x


def sweep_M(key, R=10, N=8, D=8, list_n_camp=(300, 1000, 3000, 10000, 30000, 100000), filename="sweep_M.json"):
    """Error of the five sketched objects vs the number of samples M, R repetitions per M.

    The tests show that the Qiskit oracles coincide with the JAX ones of the repo, so the sweep
    uses the repo functions (faster, larger M at the same computing time).

    Args:
        key (jax.Array): PRNG key.
        R (int): repetitions per M.
        N (int): rows of the random matrix.
        D (int): columns of the random matrix.
        list_n_camp (tuple): values of M.
        filename (str): output JSON.

    Returns:
        tuple: (meta dict, list of {M, mean, std} with one entry per object in NOMI order).
    """
    key, subkey, subkey2 = random.split(key, 3)
    A = random_sparse_matrix(subkey, (N, D), 2 * min(N, D))
    y = random.randint(subkey2, (N,), minval=0, maxval=2) * 2 - 1
    matr_data = matrix_data(A)
    vec_data = vector_data(y)
    nnz = jnp.count_nonzero(A)
    row_sparsity = int(jnp.max(jnp.sum(A != 0, axis=1)))
    col_sparsity = int(jnp.max(jnp.sum(A != 0, axis=0)))
    results = []

    for n_campioni in list_n_camp:
        rep_err_list = []
        for r in tqdm(range(R)):
            key, subkey, subkey2, subkey3, subkey4 = random.split(key, 5)
            err_list = []
            with suppress():
                # U_y: infidelity with the flat state
                state_oracle = np.asarray(q_state_sketch_flat(vec_data.get_data(subkey2, n_campioni), N))
                err_list.append(1 - abs(np.vdot(y / np.sqrt(N), state_oracle))**2)

                # element oracle: max |sin B - A|
                ele_oracle = np.asarray(q_oracle_sketch_matrix_element(matr_data.get_matrix_element_data(subkey3, n_campioni), (N, D), nnz))
                err_list.append(float(np.max(np.abs(ele_oracle - A.ravel()))))

                # index oracles: 1 - |amplitude|^2 on the correct index
                row_oracle = np.asarray(q_oracle_sketch_matrix_row_index(matr_data.get_row_data(subkey4, n_campioni), (N, D), row_sparsity))
                err_list.append(oracle_index_error(row_oracle, A))

                d_c, thr_c, resc_c = grado_segno_col(col_sparsity, int(np.ceil(np.log2(N))))
                col_oracle = np.asarray(q_oracle_sketch_matrix_index_grado(matr_data, subkey, n_campioni, (N, D), 1, col_sparsity, nnz, d_c, thr_c, resc_c))
                err_list.append(oracle_index_error(col_oracle, A.T))

                # block encoding: relative spectral error
                block_encoding_oracle = np.asarray(block_encoding_from_sparse_oracles(row_oracle, col_oracle, ele_oracle))
                err_list.append(float(np.linalg.norm(block_encoding_oracle * np.sqrt(row_sparsity * col_sparsity) - A, 2) / np.linalg.norm(A, 2)))

            rep_err_list.append(err_list)

        rep_err_list = np.array(rep_err_list)
        results.append(dict(M=int(n_campioni), mean=rep_err_list.mean(0).tolist(), std=rep_err_list.std(0).tolist()))

    meta = dict(N=N, D=D, R=R, nnz=nnz, row_sparsity=row_sparsity, col_sparsity=col_sparsity, frac_meno1=float(np.mean(y == -1)))
    with open(filename, "w") as f:
        json.dump(to_jsonable(dict(meta=meta, results=results)), f, indent=2)
    print(f"Risultati salvati in: {filename}")
    return meta, results


def func(M, a, p, c):
    """Fit model in log scale: log(a M^-p + c).

    Args:
        M (array): number of samples.
        a (float): amplitude.
        p (float): exponent.
        c (float): floor.

    Returns:
        array: log of the model.
    """
    return np.log(a * M**(-p) + c)


def grafico(meta=None, results=None, grafico_Name="sweep_M.png", filename="sweep_M.json"):
    """Plots the sweep over M with the fit a M^-p + c for each object and tests the exponents.

    Prints the deviation of every exponent from P_ATTESO in sigma, the chi^2 of the four oracles
    and, separately, the deviation of the block encoding.

    Args:
        meta (dict | None): meta of sweep_M (read from filename if None).
        results (list | None): results of sweep_M (read from filename if None).
        grafico_Name (str): output PNG.
        filename (str): JSON of sweep_M.

    Returns:
        list[dict]: fit parameters (nome, a, p, dp, c, dc, p_atteso) per object.
    """
    if meta is None or results is None:
        with open(filename, "r") as f:
            saved = json.load(f)
        meta = saved["meta"]
        results = saved["results"]

    Ms = np.array([r["M"] for r in results])
    mean = np.array([r["mean"] for r in results])
    std = np.array([r["std"] for r in results])

    Mf = np.logspace(np.log10(Ms.min()), np.log10(Ms.max()), 1000)
    fit = []

    # 2 x 3 grid: three plots on top, two below
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 6)
    axs = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]
    fig.text(0.015, 0.72, "Errore", va="center", ha="center", rotation="vertical", fontsize=11)
    fig.text(0.015, 0.28, "Errore", va="center", ha="center", rotation="vertical", fontsize=11)

    for q, ax in enumerate(axs[:5]):
        yq = mean[:, q]
        p0 = (yq[0] * Ms[0]**P_ATTESO[q], P_ATTESO[q], yq[-1] * 0.1)
        (a, p, c), cov = curve_fit(func, Ms, np.log(yq), p0=p0, bounds=([0, 0, 0], [np.inf, 3, np.inf]), maxfev=20000)

        dp = float(np.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) else float("nan")
        dc = float(np.sqrt(cov[2, 2])) if np.isfinite(cov[2, 2]) else float("nan")
        fit.append(dict(nome=NOMI[q], a=a, p=p, dp=dp, c=c, dc=dc, p_atteso=P_ATTESO[q]))

        ax.errorbar(Ms, yq, yerr=std[:, q], fmt="o", capsize=3, label="dati")
        ax.plot(Mf, np.exp(func(Mf, a, p, c)), "-", label=f"fit: p={p:.2f}±{dp:.2f}, c={c:.1e}")
        if q == 0:
            # analytic infidelity of the flat state sketch
            ax.plot(Mf, meta["frac_meno1"] * np.pi**2 * meta["N"] / Mf, "--", label="f π² N / M")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("M (campioni per query)")
        ax.set_title(f"{NOMI[q]}\n(p atteso {P_ATTESO[q]})")
        ax.legend(fontsize=8)
        print(f"{NOMI[q]}: p={p:.2f} ± {dp:.2f}, c={c:.1e} ± {dc:.1e}")

    fig.suptitle(f"Sweep su M: N={meta['N']}, D={meta['D']}, "f"s_r={meta['row_sparsity']}, s_c={meta['col_sparsity']}, R={meta['R']}")

    # compatibility of the exponents with the expected ones (four oracles, block encoding excluded)
    oracoli = fit[:4]
    z = np.array([(f["p"] - f["p_atteso"]) / f["dp"] for f in oracoli])
    for f, zi in zip(oracoli, z):
        p_bil = 2 * norm.sf(abs(zi))                       # two-sided p-value of the deviation
        print(f"{f['nome']}: scarto = {zi:+.2f} sigma, p bilaterale = {p_bil:.3f}")
    chi2_val, ndof = float(np.sum(z**2)), len(z)
    p_chi2 = float(chi2.sf(chi2_val, ndof))
    print(f"chi^2 = {chi2_val:.2f} con {ndof} gradi di liberta', p = {p_chi2:.2f}")

    # block encoding on its own
    f_be = fit[4]
    z_be = (f_be["p"] - f_be["p_atteso"]) / f_be["dp"]
    print(f"{f_be['nome']}: scarto = {z_be:+.2f} sigma, p bilaterale = {2 * norm.sf(abs(z_be)):.2e}")

    fig.tight_layout(rect=[0.04, 0, 1, 0.94])
    fig.subplots_adjust(hspace=0.5)
    fig.savefig(grafico_Name, dpi=300, bbox_inches="tight")
    plt.show()
    return fit


def tabella_costi(key, sizes=((4, 4), (8, 8), (16, 16), (64, 64)), n_campioni=1000, filename="costi_query.json"):
    """cx per query of U_A (plain and controlled) on random matrices of several sizes.

    Args:
        key (jax.Array): PRNG key.
        sizes (tuple): (N, D) sizes.
        n_campioni (int): samples per oracle (M).
        filename (str): output JSON.

    Returns:
        None
    """
    rows = []
    for N, D in sizes:
        key, kA, kr = random.split(key, 3)
        A = random_sparse_matrix(kA, (N, D), 2 * min(N, D))
        gen = matrix_data(A)
        row_sparsity = int(jnp.max(jnp.sum(A != 0, axis=1)))
        col_sparsity = int(jnp.max(jnp.sum(A != 0, axis=0)))
        _, info_r = row_index_oracle(gen.get_row_data(kr, n_campioni), (N, D), row_sparsity)
        c0 = cost_UA_query(n_campioni, (N, D), row_sparsity, info_r["kicks"])
        c1 = cost_UA_query(n_campioni, (N, D), row_sparsity, info_r["kicks"], controlled=True)
        n_row = int(np.ceil(np.log2(N)))
        n_col = int(np.ceil(np.log2(D)))
        n_cs = max(1, int(np.ceil(np.log2(col_sparsity))))     # as in block_encoding_from_oracles
        n_rs = max(1, int(np.ceil(np.log2(row_sparsity))))

        rows.append(dict(
            N=N,
            D=D,
            row_sparsity=row_sparsity,
            col_sparsity=col_sparsity,
            n_campioni=n_campioni,
            qubit_UA=n_row + n_col + n_cs + n_rs + 1 + 1 + 2 * n_row,
            cx_query=int(c0.get("cx", 0)),
            cx_query_ctrl=int(c1.get("cx", 0)),
            cx_per_campione=c0.get("cx", 0) / n_campioni,
        ))
        print(rows[-1])

    with open(filename, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Risultati salvati in {filename}")


"""
Tests

The Qiskit oracles are compared with the repo ones (data_generation.matrix_data, vector_data)
using the same subkey given to qos_sampling.py.
"""

def _test_phase_kick(key):
    """U_y diagonal: DiagonalGate and single kicks vs q_state_sketch_flat_unitary of the repo.

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    dim = 8
    n_campioni = 5
    key, subkey1, subkey2 = random.split(key, 3)
    vec_data = vector_data(random.randint(subkey1, (dim,), minval=0, maxval=2) * 2 - 1)
    data = vec_data.get_data(subkey2, n_campioni)
    sampled_indices, sampled_values = data
    diag = q_state_sketch_flat_unitary(data, dim=dim)

    # Qiskit with one DiagonalGate
    n_qubits = int(np.ceil(np.log2(dim)))
    qc = QuantumCircuit(n_qubits)
    phase = np.zeros(2**n_qubits)
    np.add.at(phase, np.asarray(sampled_indices), (1 - np.asarray(sampled_values)) / 2)
    phase *= np.pi * dim / n_campioni
    qc.append(DiagonalGate(np.exp(1j * phase)), range(n_qubits))
    # Qiskit with single phase kicks
    qc_kicks = QuantumCircuit(n_qubits)
    for j, v in zip(sampled_indices, sampled_values):
        if v == -1:
            qc_kicks.compose(phase_kick(int(j), np.pi * dim / n_campioni, n_qubits), inplace=True)

    qiskit_diag = np.diag(Operator(qc).data)
    qc_kicks_diag = np.diag(Operator(qc_kicks))
    err = jnp.max(jnp.abs(qiskit_diag - diag))
    err_2 = jnp.max(jnp.abs(qiskit_diag - qc_kicks_diag))
    assert err < 1e-8, f"Assert error: Qiskit not correctly implemented! Errore = {err}"
    assert err_2 < 1e-8, f"Assert error: Errore in phase_kick!"
    print(f"_test_phase_kick() concluso con successo, errore = {err:.2e}, {err_2:.2e}")


def _test_quantum_state_sketching(key):
    """quantum_state_sketching vs q_state_sketch_flat of the repo (dim 64, 1e5 samples).

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    dim = 64
    n_campioni = int(1e5)
    key, subkey1, subkey2 = random.split(key, 3)
    vec = random.randint(subkey1, (dim,), minval=0, maxval=2) * 2 - 1
    vec_data = vector_data(vec)
    data = vec_data.get_data(subkey2, n_campioni)
    state = q_state_sketch_flat(data, dim=dim)

    qc, cost = quantum_state_sketching(data, dim, True)

    err = jnp.max(jnp.abs(Statevector(qc).data - state))
    assert err < 1e-8, f"Assert error: QSS not correctly implemented! Error = {err}"
    print(f"_test_quantum_state_sketching() concluso con successo, errore = {err:.2e}, costo = {cost}")


def _test_element_oracle(key):
    """element_oracle vs q_oracle_sketch_matrix_element of the repo (non-power-of-2 D).

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    key, subkey, subkey2 = random.split(key, 3)
    N = 4
    D = 6
    s = 2
    n_campioni = int(1e3)
    A = random_sparse_matrix(subkey, (N, D), s * min(N, D))
    matr_data = matrix_data(A)
    data = matr_data.get_matrix_element_data(subkey2, n_campioni)
    row, col, val = data
    nnz = jnp.count_nonzero(A)
    jax_oracle = np.asarray(q_oracle_sketch_matrix_element(data, (N, D), nnz))

    qc, _ = element_oracle(row, col, val, (N, D), nnz)
    n_sys = qc.num_qubits - 1
    oracle = Operator(qc).data[:2**n_sys, :2**n_sys]
    # diagonal on the padded grid -> keep the N x D part
    oracle_diag = np.diag(oracle).reshape(-1, 2**int(np.ceil(np.log2(D))))[:N, :D].ravel()

    err = jnp.max(jnp.abs(jax_oracle - oracle_diag))
    assert err < 1e-8, f"Assert error: Element Oracle not right implemented! Error = {err}"
    print(f"_test_element_oracle() concluso con successo, con errore = {err:.2e}")


def _test_row_index_oracle(key):
    """row_index_oracle vs q_oracle_sketch_matrix_row_index of the repo.

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    key, subkey, subkey2 = random.split(key, 3)
    N = 4
    D = 6
    s = 2
    n_campioni = int(1e3)
    A = random_sparse_matrix(subkey, (N, D), s * min(N, D))
    matr_data = matrix_data(A)
    data = matr_data.get_row_data(subkey2, n_campioni)
    row_sparsity = int(jnp.max(jnp.sum(A != 0, axis=1)))
    # register sizes as in row_index_oracle (rank register from the actual row sparsity)
    i = int(np.ceil(np.log2(N)))
    b = int(np.ceil(np.log2(D)))
    k = max(1, int(np.ceil(np.log2(row_sparsity))))
    jax_oracle = np.asarray(q_oracle_sketch_matrix_row_index(data, (N, D), row_sparsity))

    # uniform superposition on K and I, then the oracle
    circ, _ = row_index_oracle(data, (N, D), row_sparsity)
    qc = QuantumCircuit(circ.num_qubits)
    qc.h(range(b, b + k + i))
    qc.compose(circ, inplace=True)
    oracle = Statevector(qc).data*np.sqrt(2**(i+k))
    oracle_reshaped = oracle.reshape(2**i, 2**k, 2**b)[:N, :row_sparsity, :D]

    err = jnp.max(jnp.abs(jax_oracle - oracle_reshaped))
    assert err < 1e-8, f"Assert error: Row index oracle not right implemented! Error = {err}"
    print(f"_test_row_oracle() concluso con successo, con errore = {err:.2e}")


def _test_col_index_oracle(key):
    """col_index_oracle (degree 51, repo convention) vs q_oracle_sketch_matrix_index of the repo.

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    key, subkey, subkey2 = random.split(key, 3)
    N = 4
    D = 4
    s = 2
    n_campioni = int(1e3)
    n_row = int(np.ceil(np.log2(N)))
    n_col = int(np.ceil(np.log2(D)))
    A = random_sparse_matrix(subkey, (N, D), s * min(N, D))
    matr_data = matrix_data(A)
    nnz = jnp.count_nonzero(A)
    col_sparsity = int(jnp.max(jnp.sum(A != 0, axis=0)))
    k = max(1, int(np.ceil(np.log2(col_sparsity))))       # as in col_index_oracle
    o = n_row+n_col+k
    K = list(range(n_row, n_row + k))
    J = list(range(n_row + k, o))
    # same sampling subkey for Qiskit and JAX
    key, subkey = random.split(key)

    with suppress():
        jax_oracle = q_oracle_sketch_matrix_index(matr_data, subkey, n_campioni, (N, D), 1, col_sparsity, nnz)

    circ, _ = col_index_oracle(matr_data, subkey, col_sparsity, (N, D), nnz, n_campioni, degree=51)
    qc = QuantumCircuit(circ.num_qubits)
    # superposition first, then the oracle
    qc.h(K + J)
    qc.compose(circ, inplace=True)
    oracle_reshaped = Statevector(qc).data[:2**o].reshape(2**n_col, 2**k, 2**n_row)[:D, :col_sparsity, :N] * np.sqrt(2**(k + n_col))

    err = jnp.max(jnp.abs(oracle_reshaped - jax_oracle))
    assert err < 1e-8, f"Assert error: Row index oracle not right implemented! Error = {err}"
    print(f"_test_col_oracle() concluso con successo, con errore={err:.2}")


def _test_block_encoding_from_oracles(key):
    """U_A vs block_encoding_from_sparse_oracles of the repo, at equal normalization.

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    N = 4
    D = 6
    s = 2
    n_campioni = int(1e3)
    n_row = int(np.ceil(np.log2(N)))
    n_col = int(np.ceil(np.log2(D)))
    key, subkey, subkey2, subkey3 = random.split(key, 4)
    A = random_sparse_matrix(subkey, (N, D), s * min(N, D))
    matr_data = matrix_data(A)
    data_row = matr_data.get_row_data(subkey2, n_campioni)
    data_ele = matr_data.get_matrix_element_data(subkey3, n_campioni)
    nnz = jnp.count_nonzero(A)
    col_sparsity = int(jnp.max(jnp.sum(A != 0, axis=0)))
    row_sparsity = int(jnp.max(jnp.sum(A != 0, axis=1)))

    with suppress():
        O_row = q_oracle_sketch_matrix_row_index(data_row, (N, D), row_sparsity)
        O_col = q_oracle_sketch_matrix_index(matr_data, subkey, n_campioni, (N, D), 1, col_sparsity, nnz)
        O_ele = q_oracle_sketch_matrix_element(data_ele, (N, D), nnz)
    jax_oracle = block_encoding_from_sparse_oracles(O_row, O_col, O_ele)

    be_oracle = block_encoding_from_oracles(subkey, n_campioni, matr_data, data_row, data_ele, (N, D), col_sparsity, row_sparsity, nnz, degree_col=51)
    # column j of the block: prepare |j> on J, apply U_A, read the I register
    block = np.zeros((N, D), dtype=complex)
    for j in range(D):
        qc = QuantumCircuit(be_oracle.num_qubits)
        for q in range(n_col):
            if (j >> q) & 1:
                qc.x(n_row + q)
        qc.compose(be_oracle, inplace=True)
        block[:, j] = Statevector(qc).data[:2**n_row][:N]

    # same samples => same oracles; different normalizations:
    #   our U_A: B = A/(s_r s_c);   repo: O_repo = A/sqrt(s_r s_c)   (Lemma 48)
    alpha = row_sparsity * col_sparsity
    alpha_repo = np.sqrt(row_sparsity * col_sparsity)
    err = float(np.max(np.abs(block * alpha - np.asarray(jax_oracle) * alpha_repo)))
    assert err < 1e-8, f"Assert error: U_A diverso dal repo a parita' di normalizzazione, errore = {err:.2e}"
    err_A = np.linalg.norm(block * alpha - np.asarray(A), 2) / np.linalg.norm(np.asarray(A), 2)
    print(f"_test_block_encoding_from_oracles() ok: alpha = s_r*s_c = {alpha}, " f"max|alpha*B - alpha_repo*O_repo| = {err:.2e}, ||alpha*B - A||_2/||A||_2 = {err_A:.2e} (sketching + padding)")


def _test_circuit_costs(key):
    """Sample consumption of U_A, unitarity of U_A, controlled U_y.

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    N = 4
    D = 6
    s = 2
    n_campioni = int(1e3)
    key, subkey, subkey2, subkey3, subkey4, subkey5 = random.split(key, 6)
    A = random_sparse_matrix(subkey, (N, D), s * min(N, D))
    matr_data = matrix_data(A)
    nnz = jnp.count_nonzero(A)
    col_sparsity = int(jnp.max(jnp.sum(A != 0, axis=0)))
    row_sparsity = int(jnp.max(jnp.sum(A != 0, axis=1)))
    n_rows = int(np.ceil(np.log2(N)))
    degree = grado_segno_col(col_sparsity, n_rows)[0]

    # samples used by one U_A: M (element) + M (row) + n_row * degree * M (column)
    new_UA = make_UA(matr_data, n_campioni, (N, D), row_sparsity, col_sparsity, nnz)
    n_campioni_inizio = int(matr_data.num_generated_samples)
    U_A1 = new_UA(subkey2)
    n_campioni_fine = int(matr_data.num_generated_samples)
    n_campioni_usati = n_campioni_fine - n_campioni_inizio
    expected = (2*n_campioni) + (n_rows*degree*n_campioni)
    assert expected == n_campioni_usati, f"Assert error: the number of used samples is not right increasing! used={n_campioni_usati}, expected={expected}"
    print("test consumo campioni block enconding di matrice sparsa concluso con successo")

    # a second block with another key
    U_A2 = new_UA(subkey3)

    # U_A U_A^dagger = I at machine precision
    qc = QuantumCircuit(U_A1.num_qubits)
    qc.x(n_rows)
    qc.compose(U_A1, inplace=True)
    qc.compose(U_A1.inverse(), inplace=True)
    state = Statevector(qc).data[2**n_rows]
    assert abs(1-state) < 1e-10, f"Assert error: U_A is not unitary matrix at machine precision!"
    print(f"test U_A unitaria concluso con successo, errore = {abs(1-state)}")

    # controlled U_y
    dim = 8
    vec_data = vector_data(random.randint(subkey4, (dim,), 0, 2) * 2 - 1)
    n = int(np.ceil(np.log2(dim)))
    new_Uy = make_Uy(vec_data, n_campioni)
    U_y, _ = new_Uy(subkey5)
    cU_y = controlled_gate(U_y)
    c_state = Statevector(U_y).data

    # control 0: identity
    qc = QuantumCircuit(U_y.num_qubits + 1)
    qc.append(cU_y, [n] + list(range(n)))
    state = Statevector(qc).data.reshape(2, 2**n)[0]
    assert jnp.allclose(state, np.eye(2**n)[0], atol=1e-8), f"Assert error: U_y controlled to 0 not right implemented!"
    # control 1: U_y
    qc = QuantumCircuit(U_y.num_qubits + 1)
    qc.x(n)
    qc.append(cU_y, [n] + list(range(n)))
    state = Statevector(qc).data.reshape(2, 2**n)[1]
    error = jnp.max(jnp.abs(state - c_state))
    print(f"Errore massimo: {error:.12e}")
    assert jnp.allclose(state, c_state, atol=1e-8), f"Assert error: U_y controlled to 1 not right implemented! Errore massimo: {error:.12e}"
    print("test U_y controllata concluso con successo")


def _test_copia_col_oracle(key):
    """With the repo parameters the copy must coincide with q_oracle_sketch_matrix_index.

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    key, subkey, k_s = random.split(key, 3)
    N, D, s = 8, 8, 2
    A = random_sparse_matrix(subkey, (N, D), s * min(N, D))
    md = matrix_data(A)
    nnz = jnp.count_nonzero(A)
    cs = int(jnp.max(jnp.sum(A != 0, axis=0)))
    with suppress():
        o_repo = np.asarray(q_oracle_sketch_matrix_index(md, k_s, 1000, (N, D), 1, cs, nnz))
        o_copia = np.asarray(q_oracle_sketch_matrix_index_grado(md, k_s, 1000, (N, D), 1, cs, nnz, 51, np.pi / (4 * cs + 2), 0.99))
    err = float(np.max(np.abs(o_repo - o_copia)))
    assert err < 1e-10, f"copia diversa dal repo: {err:.2e}"
    print(f"_test_copia_col_oracle() ok, errore = {err:.2e}")


if __name__ == "__main__":
    print("---"*10)
    print("sketch_oracles.py")
    print("---"*10)
    #_test_phase_kick(key)
    #_test_quantum_state_sketching(key)
    #_test_element_oracle(key)
    #_test_row_index_oracle(key)
    #_test_col_index_oracle(key)
    #_test_block_encoding_from_oracles(key)
    #_test_circuit_costs(key)
    #_test_copia_col_oracle(key)
    #meta, results = sweep_M(key)
    #tabella_costi(key)
    grafico()
