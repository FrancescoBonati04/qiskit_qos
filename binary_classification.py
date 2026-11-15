"""
Theorem F.13 with sketched oracles: rectangular QSVT of 1/x on U_A (circuit W), fixed-point
amplification (circuit V), interferometric shadow, sweep over the number of samples M and
resource estimates (samples, gates, depth, physical and fault-tolerant times).
"""

import os
import json
import gc
import time
from collections import Counter

import numpy as np
import jax.numpy as jnp
from jax import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from utilities import Connect, save_angles, stampa_risultati
import small_toy_exp as ste
from data_set_prep import dataset_sintetico
from shadow import shadow_measure, classical_result
from reference import classical_reference as ref, get_exact_num_qubits, get_exact_num_samples, compute_F69, classical_cost
from ridge_solver import projector_phase_lcu, amplify_fixed_point, scegli_deg_sign
import sketch_oracles
from sketch_oracles import make_UA, make_Uy, cost, cost_UA_query, to_jsonable, func, row_index_oracle, profondita, depth_kick, depth_UA_query, grado_segno_col, EPS_COL
from predict import select_tests
from simulate import simulate, evolvi

key = Connect()
from utils import suppress_stdout_stderr as suppress, random_sparse_matrix
from data_generation import matrix_data, vector_data
from qsvt import get_qsvt_angles_inverse

BASIS = ["cx", "rz", "sx", "x"]
K_MEDIANA = lambda m, delta: int(np.ceil(2 * np.log(2 * m / delta)))    # groups of the median of means
# degree of the sign for the amplification: Re P >= 0.98 only at sqrt(p), degrees up to 401
SIGN_KW = dict(d_max=401, banda=None)
# (t_1q, t_2q, T2) in seconds
platform = {
    "superconductors": (25e-9, 42e-9, 89e-6),   # Google Willow
    "ion_traps":       (135e-6, 600e-6, 1.0),   # IonQ Aria
    "neutral_atoms":   (1e-6, 264e-9, 1.34),    # Harvard/MIT/QuEra (Rb)
}


"""
RAM-efficient functions
"""

def simula_W_a_pezzi(X, y, _lambda, n_campioni, key, eps=0.05):
    """Statevector of W (U_y + rectangular QSVT of 1/x) simulated query by query.

    Same circuit as build_W_rect, but every U_A is built, applied to the state and discarded:
    the memory stays that of a single U_A.

    Args:
        X (np.ndarray): (N, D) matrix.
        y (np.ndarray): (N,) labels.
        _lambda (float): regularization in (0, 1].
        n_campioni (int): samples per oracle (M).
        key (jax.Array): PRNG key of the samples.
        eps (float): precision of 1/x.

    Returns:
        tuple: (statevector of W, info dict with sparsities, alpha, s_min, kappa_s, kappa_poly,
            deg_inv, scale, register sizes, qubit_W, Q_U, Q_y).
    """
    N, D = X.shape
    assert 0 < _lambda <= 1, f"serve 0 < lambda <= 1 (oracolo elemento), lambda = {_lambda}"
    A_aug = jnp.vstack([X, jnp.sqrt(_lambda) * jnp.eye(D)])
    matr_data, vec_data = matrix_data(A_aug), vector_data(y)
    nnz = jnp.count_nonzero(A_aug)
    row_sparsity = int(jnp.max(jnp.sum(A_aug != 0, axis=1)))
    col_sparsity = int(jnp.max(jnp.sum(A_aug != 0, axis=0)))
    # sketched normalization alpha = s_r s_c -> kappa_s and degree of 1/x
    alpha = row_sparsity * col_sparsity
    s_min = float(np.linalg.svd(np.asarray(A_aug), compute_uv=False).min())
    kappa_poly = int(np.ceil(alpha / s_min - 1e-6))
    with suppress():
        angles, scale = save_angles(get_qsvt_angles_inverse, kappa_poly, eps)
    scale = float(np.asarray(scale).ravel()[0])
    angles = np.asarray(angles, dtype=float)
    d = len(angles) - 1

    print("==="*40)
    print(f"kappa_poly = {kappa_poly}, d = {d}")
    print("==="*40)

    new_UA = make_UA(matr_data, n_campioni, A_aug.shape, row_sparsity, col_sparsity, nnz)
    new_Uy = make_Uy(vec_data, n_campioni)
    key, k_y, k_u = random.split(key, 3)
    keys_U = random.split(k_u, d)

    n_row = int(np.ceil(np.log2(N + D)))
    n_col = int(np.ceil(np.log2(D)))
    sv, n_tot = None, None
    for j, ang in enumerate(tqdm(angles[::-1][:-1], desc="query U_A")):
        U = new_UA(keys_U[j]).to_gate()
        if sv is None:                                         # first step: U_y and H on the LCU
            n_U = U.num_qubits
            n_tot, lcu = n_U + 1, n_U
            I, J, anc = list(range(n_row)), list(range(n_row, n_row + n_col)), list(range(n_row + n_col, n_U))
            inizio = QuantumCircuit(n_tot)
            U_y, _ = new_Uy(k_y)
            inizio.append(U_y, range(int(np.log2(N))))
            inizio.h(lcu)
            sv = evolvi(None, inizio, n_tot)
        # projector on the input (J + anc) before U^dagger, on the output (I + anc) before U
        pezzo = QuantumCircuit(n_tot)
        projector_phase_lcu(pezzo, ang, J + anc if j % 2 == 0 else I + anc, lcu)
        pezzo.append(U.inverse() if j % 2 == 0 else U, range(n_U))
        sv = evolvi(sv, pezzo, n_tot)
        del U, pezzo
        if j % 20 == 0:
            gc.collect()

    # last phase, global phase, H on the LCU
    fine = QuantumCircuit(n_tot)
    projector_phase_lcu(fine, angles[0], I + anc, lcu)
    fine.rz(np.pi * len(angles), lcu)
    fine.h(lcu)
    sv = evolvi(sv, fine, n_tot)

    info = dict(N=N, D=D, row_sparsity=row_sparsity, col_sparsity=col_sparsity, nnz=nnz, alpha=alpha,
                s_min=s_min, kappa_s=alpha / s_min, kappa_poly=kappa_poly, deg_inv=d, scale=scale,
                n_row=n_row, n_col=n_col, n_U=n_U, qubit_W=n_tot, Q_U=d, Q_y=1)
    return sv, info


def qsvt_rect(U_list, angles, proj_in, proj_out):
    """Real part of the rectangular QSVT, alternating U^dagger and U (a different block per query).

    Args:
        U_list (list[Gate]): len(angles) - 1 block encodings.
        angles (array): QSVT angles.
        proj_in (list[int]): qubits of the input projector (J + ancillas).
        proj_out (list[int]): qubits of the output projector (I + ancillas).

    Returns:
        QuantumCircuit: "QSVT_rect" on U_list[0].num_qubits + 1 qubits (LCU = last).
    """
    assert len(U_list) == len(angles) - 1, f"Assert error: U_list in qsvt_rect has wrong size!"
    angles = np.array(angles, dtype=float)
    n_qubits = U_list[0].num_qubits
    qc = QuantumCircuit(n_qubits + 1, name="QSVT_rect")
    lcu = n_qubits

    qc.h(lcu)
    for j, ang in enumerate(angles[::-1][:-1]):
        U = U_list[j]
        projector_phase_lcu(qc, ang, proj_in if j % 2 == 0 else proj_out, lcu)
        qc.append(U.inverse() if j % 2 == 0 else U, range(n_qubits))
    projector_phase_lcu(qc, angles[0], proj_out, lcu)
    qc.rz(jnp.pi * len(angles), lcu)
    qc.h(lcu)
    return qc


def build_W_rect(X, y, _lambda, n_campioni, key, eps=0.05, angles=None):
    """Circuit W with sketched oracles: U_y followed by the rectangular QSVT of 1/x on U_A.

    Args:
        X (np.ndarray): (N, D) matrix.
        y (np.ndarray): (N,) labels.
        _lambda (float): regularization in (0, 1].
        n_campioni (int): samples per oracle (M).
        key (jax.Array): PRNG key of the samples.
        eps (float): precision of 1/x.
        angles (array | None): imposed QSVT angles (None = angles of 1/x from pyqsp).

    Returns:
        tuple: (W on n_U + 1 qubits, info dict with sparsities, alpha, s_min, kappa_s,
            kappa_poly, deg_inv, scale, anc, register sizes).
    """
    N, D = X.shape
    assert 0 < _lambda <= 1, f"serve 0 < lambda <= 1 (oracolo elemento), lambda = {_lambda}"
    A_aug = jnp.vstack([X, jnp.sqrt(_lambda) * jnp.eye(D)])
    matr_data = matrix_data(A_aug)
    vec_data = vector_data(y)
    nnz = jnp.count_nonzero(A_aug)
    row_sparsity = int(jnp.max(jnp.sum(A_aug != 0, axis=1)))
    col_sparsity = int(jnp.max(jnp.sum(A_aug != 0, axis=0)))

    alpha = row_sparsity * col_sparsity
    s_min = np.linalg.svd(A_aug, compute_uv=False).min()
    kappa_poly = int(np.ceil(alpha / s_min - 1e-6))

    scale = None
    if angles is None:
        with suppress():
            angles, scale = save_angles(get_qsvt_angles_inverse, kappa_poly, eps)
            scale = float(np.asarray(scale).ravel()[0])
    d = len(angles) - 1

    # d fresh block encodings
    new_UA = make_UA(matr_data, n_campioni, A_aug.shape, row_sparsity, col_sparsity, nnz)
    new_Uy = make_Uy(vec_data, n_campioni)
    U_list = []
    for i in tqdm(range(d)):
        key, subkey = random.split(key)
        U_list.append(new_UA(subkey).to_gate())

    # registers of U_A: I (rows of A_aug), J (columns), ancillas
    n_row = int(np.ceil(np.log2(N+D)))
    n_col = int(np.ceil(np.log2(D)))
    n_U = U_list[0].num_qubits
    I = list(range(n_row))
    J = list(range(n_row, n_row+n_col))
    anc = list(range(n_row+n_col, n_U))

    qc_qsvt_rect = qsvt_rect(U_list, angles, J + anc, I + anc)
    qc = QuantumCircuit(n_U + 1, name="W")
    key, subkey = random.split(key)
    U_y, _ = new_Uy(subkey)
    qc.append(U_y, range(int(np.log2(N))))              # |y~> in the first N rows of I
    qc.compose(qc_qsvt_rect, range(n_U + 1), inplace=True)

    info = dict(
        N=N,
        D=D,
        row_sparsity=row_sparsity,
        col_sparsity=col_sparsity,
        nnz=nnz,
        alpha=alpha,
        s_min=s_min,
        kappa_s=alpha/s_min,
        kappa_poly=kappa_poly,
        deg_inv=d,
        scale=scale,
        anc=I+anc+[n_U],
        n_row=n_row,
        n_col=n_col,
        n_U=n_U,
    )
    return qc, info


def cx_proiettore(n_proj, controlled=False):
    """cx of one projector phase (projector_phase_lcu) on n_proj qubits.

    Args:
        n_proj (int): qubits of the projector.
        controlled (bool): one more control (shadow control).

    Returns:
        int: number of cx.
    """
    n_p = n_proj + int(controlled)
    qc = QuantumCircuit(n_p + 1)
    projector_phase_lcu(qc, 0.3, list(range(n_p)), n_p)
    return int(Counter(transpile(qc, basis_gates=BASIS).count_ops()).get("cx", 0))


def depth_proiettore(n_proj, controlled=False):
    """Depth of one projector phase (projector_phase_lcu) on n_proj qubits.

    Args:
        n_proj (int): qubits of the projector.
        controlled (bool): one more control (shadow control).

    Returns:
        np.ndarray: [2-qubit depth, other depth].
    """
    n_p = n_proj + int(controlled)
    qc = QuantumCircuit(n_p + 1)
    projector_phase_lcu(qc, 0.3, list(range(n_p)), n_p)
    return profondita(qc)


def singol_instance(key, key_campioni, n_campioni=1000, N=4, D=8, kappa_target=1.5, simulation=3, n_select_test=5, overlap_min=0.45, delta=0.1, c_pratico=65, R_F59=1.0):
    """Sketched case of predict.predict on one instance, without shadow and without amplification.

    Since the measured quantity is a sign, the amplification only multiplies the state by Re P;
    its cost is added analytically (degree_sign from scegli_deg_sign).

    Args:
        key (jax.Array): PRNG key of the instance (istanza_kappa_piccolo).
        key_campioni (jax.Array): PRNG key of the samples.
        n_campioni (int): samples per oracle (M).
        N (int): rows.
        D (int): columns.
        kappa_target (float): not used (kept for compatibility).
        simulation (int): independent simulations of W (averaged).
        n_select_test (int): not used (istanza_kappa_piccolo selects 5 tests).
        overlap_min (float): not used (istanza_kappa_piccolo uses 0.45).
        delta (float): failure probability.
        c_pratico (float): practical constant of the shots per group.
        R_F59 (float): constant of F59.

    Returns:
        dict: sizes, sparsities, kappas, degrees, qubits, queries, infidelity (mean and std), p and
            its formula, fraction of correct signs, minimum margin, samples per W, S_tot, T_pratico,
            F59_ref, degree_sign, time and minimum overlap of the tests.
    """
    key, subkey, subkey2 = random.split(key, 3)
    X, y, _lambda, tests, ref = ste.istanza_kappa_piccolo(subkey, N, D)
    test_norm = tests / np.linalg.norm(tests, axis=1, keepdims=True)

    inizio = time.time()
    fidelities = []
    w_list = []
    probs = []
    for i in range(simulation):
        key_campioni, subkey = random.split(key_campioni)
        v, info = simula_W_a_pezzi(X, y, _lambda, n_campioni, subkey)
        # w is in the J register with I = 0 and all ancillas = 0
        w = v[(2**info["n_row"]) * np.arange(D)]
        w_list.append(w)
        probs.append(float(np.sum(np.abs(w)**2)))
        fidelities.append(abs(np.vdot(np.asarray(ref.w_ket), w / np.linalg.norm(w)))**2)

    target = np.real(test_norm @ np.mean(w_list, axis=0))
    b = info["n_row"]
    Q_U = info["Q_U"]
    # samples of one W: per U_A M (element) + M (row) + d_col b M (column); plus M for U_y
    d_col = grado_segno_col(info["col_sparsity"], b)[0]
    campioni_W = Q_U * (2 * n_campioni + d_col * b * n_campioni) + n_campioni

    p = float(np.mean(probs))
    degree_sign, _ = scegli_deg_sign(np.sqrt(p), **SIGN_KW)
    m = tests.shape[0]
    eps_pr = float(np.min(np.abs(np.asarray(ref.inner_product)))) / 3
    T_pr = K_MEDIANA(m, delta) * int(np.ceil(c_pratico / eps_pr**2))
    S_tot = None if degree_sign is None else T_pr * degree_sign * campioni_W
    F59_ref = float(get_exact_num_samples(R_F59, N, ref.s, ref.k_reg, ref.gamma_test, D, m, delta))

    """
    Results
        {'N': 4, 'D': 2, 'M': 10000, 'simulation': 1, 'row_sparsity': 1, 'col_sparsity': 3, 'kappa_reg': 1.4311431503352174, 'kappa_s': 3.1147810982274673, 'deg_inv': 139, 'qubit_W': 16, 'Q_U': 139, 'Q_y': 1, 'infid': 0.012044882085130104, 'infid_std': 0.0, 'p': 0.0015590122773890395, 'p_formula': 0.007415049819856645, 'correct': 1.0, 'min_margin': 0.01956275891922493, 'campioni_W': 215460000, 'S_tot': 819377143200000, 'T_pratico': 18920, 'F59_ref': 210605.92091646016, 'degree_sign': 201, 'tempo_s': 532.5875360965729}
        tempo totale: 537 s
    """

    return dict(N=N,
                D=D,
                M=n_campioni,
                simulation=simulation,
                row_sparsity=info["row_sparsity"],
                col_sparsity=info["col_sparsity"],
                kappa_reg=float(ref.k_reg),
                kappa_s=info["kappa_s"],
                deg_inv=info["deg_inv"],
                qubit_W=info["qubit_W"],
                Q_U=Q_U,
                Q_y=info["Q_y"],
                infid=1 - float(np.mean(fidelities)),
                infid_std=float(np.std(fidelities)),
                p=p,
                p_formula=float((info["scale"] * info["alpha"] * np.linalg.norm(ref.w) / np.linalg.norm(y))**2),
                correct=float(np.mean(np.sign(target) == np.asarray(ref.y_pred))),
                min_margin=float(np.min(np.abs(target))),
                campioni_W=campioni_W,
                S_tot=S_tot,
                T_pratico=T_pr,
                F59_ref=F59_ref,
                degree_sign=degree_sign,
                tempo_s=time.time() - inizio,
                overlap_test=float(np.min(np.abs(np.asarray(ref.inner_product)))),
                )


def sweep_M(key, sizes=((4, 2), (4, 4)), Ms=(100, 300, 1000, 3000, 10000, 30000, 100000), filename="sweep_m_simulazioni.json", aggiungi=True, n_catena=5):
    """singol_instance for every size and every M, saving after each point.

    The sample keys of the first n_catena values of M come from the same key chain as the first
    run; further M use fold_in(k_inst, M). The instance is checked (kappa_s) before simulating.

    Args:
        key (jax.Array): PRNG key.
        sizes (tuple): (N, D) sizes.
        Ms (tuple): values of M.
        filename (str): JSON with the rows.
        aggiungi (bool): append to an existing file, skipping the (N, D, M) already done.
        n_catena (int): length of the key chain of the first run.

    Returns:
        list[dict]: all rows (old and new).
    """
    rows = []
    if aggiungi and os.path.exists(filename):
        with open(filename) as f:
            rows = json.load(f)
    fatti = {(r["N"], r["D"], r["M"]) for r in rows}
    for N, D in sizes:
        key, k_inst = random.split(key)
        catena = []
        for _ in range(n_catena):                      # same chain as the first run (5 M)
            key, k = random.split(key)
            catena.append(k)

        # check the instance BEFORE simulating (same key used by singol_instance)
        _, k_x, _ = random.split(k_inst, 3)
        X, y, lam, _, _ = ste.istanza_kappa_piccolo(k_x, N, D)
        A_aug = np.vstack([np.asarray(X), np.sqrt(lam) * np.eye(D)])
        kappa_s = (A_aug != 0).sum(1).max() * (A_aug != 0).sum(0).max() / np.linalg.svd(A_aug, compute_uv=False).min()
        prec = [r["kappa_s"] for r in rows if (r["N"], r["D"]) == (N, D)]
        assert not prec or abs(kappa_s - prec[0]) < 1e-6, f"({N},{D}): istanza diversa (kappa_s {kappa_s} vs {prec[0]})"

        for i, n_campioni in enumerate(tqdm(Ms)):
            k_samp = catena[i] if i < n_catena else random.fold_in(k_inst, n_campioni)
            if (N, D, n_campioni) in fatti:
                continue
            rows.append(singol_instance(k_inst, k_samp, n_campioni, N, D))
            print(rows[-1])
            with open(filename, "w") as f:
                json.dump(to_jsonable(rows), f, indent=2)
    return rows


def build_V_full(X, y, _lambda, M, key, deg_sign, threshold, fresh=False, angles_W=None):
    """Circuit V = fixed-point amplification of W (sketched).

    Args:
        X (np.ndarray): (N, D) matrix.
        y (np.ndarray): (N,) labels.
        _lambda (float): regularization in (0, 1].
        M (int): samples per oracle.
        key (jax.Array): PRNG key.
        deg_sign (int): degree of the sign polynomial.
        threshold (float): threshold of the sign polynomial.
        fresh (bool): a fresh W for every query of the amplification (True) or the same W (False).
        angles_W (array | None): imposed QSVT angles of W.

    Returns:
        tuple: (V, info of amplify_fixed_point).
    """
    keys = random.split(key, deg_sign)
    if fresh:
        built = [build_W_rect(X, y, _lambda, M, k, angles=angles_W) for k in keys]
        W, info = [b[0].to_gate(label="W") for b in built], built[0][1]
    else:
        W, info = build_W_rect(X, y, _lambda, M, keys[0], angles=angles_W)
        W = W.to_gate(label="W")                                    # "W" box in the drawing
    V, info_V = amplify_fixed_point(W, info, deg_sign, threshold, 0.99)
    return V, info_V


def embed_tests_rect(x_hat, n_row, n_V):
    """States |x+-> of the tests in the rectangular layout (x_j at index 2^n_row * j, control = qubit 0).

    Args:
        x_hat (np.ndarray): (m, D) normalized tests.
        n_row (int): qubits of the I register (the J register starts at bit n_row).
        n_V (int): qubits of V.

    Returns:
        tuple: (Xp, Xm), each (m, 2^(n_V+1)).
    """
    m, D = x_hat.shape
    x_exp = np.zeros((m, 2**n_V), dtype=complex)
    x_exp[:, (2**n_row) * np.arange(D)] = x_hat
    Xp = np.zeros((m, 2**(n_V + 1)), dtype=complex)
    Xm = np.zeros_like(Xp)
    Xp[:, 0] = Xm[:, 0] = 1 / np.sqrt(2)
    Xp[:, 1::2], Xm[:, 1::2] = x_exp / np.sqrt(2), -x_exp / np.sqrt(2)
    return Xp, Xm


def binary_classification(X, y, _lambda, tests, M, key, deg_sign, threshold, gamma, s, delta=0.1, c=65, T_max=None, angles_W=None):
    """Full algorithm of Theorem F.13: for every shot a fresh V, one random-Clifford measurement,
    then the median of means of the shadow estimators.

    Computationally very expensive: it needs small T_max (10-20) and small instances; demonstrative.

    Args:
        X (np.ndarray): (N, D) matrix (N power of 2, ||X|| <= 1, |X_ij| <= 1).
        y (np.ndarray): (N,) labels +-1.
        _lambda (float): regularization in (0, 1].
        tests (np.ndarray): (m, D) s-sparse tests with ||x'||_inf <= 1.
        M (int): samples per oracle.
        key (jax.Array): PRNG key.
        deg_sign (int): degree of the sign polynomial of the amplification.
        threshold (float): threshold of the sign polynomial.
        gamma (float): minimum margin (eps = gamma / (3 sqrt(s))).
        s (int): sparsity.
        delta (float): failure probability.
        c (float): constant of the shots per group.
        T_max (int | None): maximum number of shots.
        angles_W (array | None): imposed QSVT angles of W.

    Returns:
        np.ndarray: (m,) predicted signs.
    """
    N, D = X.shape
    assert np.log2(N).is_integer() and np.max(np.abs(X)) <= 1 and np.linalg.norm(np.asarray(X), 2) <= 1 + 1e-9, f"Assert error 1"
    assert set(np.unique(np.asarray(y))) <= {-1, 1}, f"Assert error 2"
    assert np.all(np.count_nonzero(tests, axis=1) <= s) and np.max(np.abs(tests)) <= 1, f"Assert error 3"

    m = tests.shape[0]
    x_hat = tests / np.linalg.norm(tests, axis=1, keepdims=True)
    # number of shots: K groups of N_g (possibly capped by T_max)
    eps = gamma / (3 * np.sqrt(s))
    K = int(np.ceil(2 * np.log(2 * m / delta)))
    N_g = int(np.ceil(c / eps**2))
    T = K * N_g if T_max is None else min(K * N_g, T_max)
    assert T >= K, f"servono almeno K = {K} shot (uno per gruppo della mediana delle medie)"
    N_g = T // K
    T = K * N_g
    rng = np.random.default_rng(0)

    est = []
    for i, k in enumerate(random.split(key, T)):
        start = time.time()
        # fresh V for this shot, statevector, one shadow measurement
        V, info = build_V_full(X, y, _lambda, M, k, deg_sign, threshold, fresh=True, angles_W=angles_W)
        v = simulate(V)
        w_tilde = np.zeros(2 * v.size, dtype=complex)
        w_tilde[0] = 1 / np.sqrt(2)
        w_tilde[1::2] = v / np.sqrt(2)
        Xp, Xm = embed_tests_rect(x_hat, info["n_row"], V.num_qubits)
        dati = shadow_measure(None, rng, 1, stato=Statevector(w_tilde))
        est.append(np.asarray(classical_result(Xp, Xm, V.num_qubits + 1, dati))[0])
        print(f"shot {i + 1}/{T}: {time.time() - start:.1f} s")
    est = np.asarray(est).reshape(K, N_g, m)
    return np.sign(np.median(est.mean(axis=1), axis=0))


"""
Costs
"""

def fit_costi(key, sizes=((4, 2), (8, 4), (16, 8), (64, 32), (256, 128), (1024, 512)), s=2, n_campioni=1000, n_misure=10, filename="fit_costi.json", png="fit_costi.png"):
    """cx per sample of the controlled U_A vs the number of index qubits n, with the fit b0 n^b1.

    Args:
        key (jax.Array): PRNG key.
        sizes (tuple): (N, D) sizes.
        s (int): sparsity of the random matrices.
        n_campioni (int): samples per oracle (M).
        n_misure (int): measurements per size (different row samples).
        filename (str): output JSON.
        png (str): output figure.

    Returns:
        tuple: (rows, fit dict with b0, b1 and their standard errors).
    """
    rows = []
    for N, D in tqdm(sizes):
        key, subkey = random.split(key)
        X = random_sparse_matrix(subkey, (N, D), s*min(N, D))
        # lambda fixed to 0.5
        A_aug = jnp.vstack([X, jnp.sqrt(0.5) * jnp.eye(D)])
        matr_data = matrix_data(A_aug)
        row_sparsity = int(jnp.max(jnp.sum(A_aug != 0, axis=1)))
        col_sparsity = int(jnp.max(jnp.sum(A_aug != 0, axis=0)))

        costi_UA, calci_riga = [], []
        for _ in range(n_misure):
            key, subkey = random.split(key)
            _, info_r = row_index_oracle(matr_data.get_row_data(subkey, n_campioni), A_aug.shape, row_sparsity)
            cost_UA = int(cost_UA_query(n_campioni, A_aug.shape, row_sparsity, info_r["kicks"], grado_segno_col(col_sparsity, int(np.ceil(np.log2(N + D))))[0], controlled=True).get("cx", 0))
            costi_UA.append(cost_UA / n_campioni)
            calci_riga.append(info_r["kicks"] / n_campioni)

        media_UA, std_UA = float(np.mean(costi_UA)), float(np.std(costi_UA, ddof=1))
        media_calci, std_calci = float(np.mean(calci_riga)), float(np.std(calci_riga, ddof=1))
        cost_Uy = int(cost(True, int(np.log2(N)) + 1, n_campioni // 2).get("cx", 0))
        n = int(np.ceil(np.log2(N + D))) + int(np.ceil(np.log2(D)))
        rows.append(dict(N=N, D=D, n=n, row_sparsity=row_sparsity, col_sparsity=col_sparsity, M=n_campioni, n_misure=n_misure, cx_UA_media=media_UA, cx_UA_std=std_UA, cx_Uy=cost_Uy, cx_Uy_per_campione=cost_Uy / n_campioni, calci_riga_per_campione_media=media_calci, calci_riga_per_campione_std=std_calci))
        print(rows[-1])

    # unweighted log-log fit
    n = np.array([r["n"] for r in rows], float)
    g = np.array([r["cx_UA_media"] for r in rows], float)
    sigma = np.array([r["cx_UA_std"] for r in rows], float)
    b1, logb0 = np.polyfit(np.log(n), np.log(g), 1)
    fit_log = dict(b0=float(np.exp(logb0)), b1=float(b1))
    print(f"Fit logaritmico: cx ~ {fit_log['b0']:.3g} * n^{fit_log['b1']:.2f}")

    # weighted fit with curve_fit: sigma is a std, not the standard error of the mean
    def modello(n, b0, b1):
        return b0 * n**b1

    sigma_fit = np.where(sigma > 0, sigma, np.finfo(float).eps)
    popt, pcov = curve_fit(modello, n, g, p0=[fit_log["b0"], fit_log["b1"]], sigma=sigma_fit, absolute_sigma=True, maxfev=10000)
    fit = dict(b0=float(popt[0]), b1=float(popt[1]), std_b0=float(np.sqrt(pcov[0, 0])), std_b1=float(np.sqrt(pcov[1, 1])))
    print(f"Fit pesato: cx ~ {fit['b0']:.3g} * n^{fit['b1']:.2f}")
    print(f"Errore standard parametri: std(b0)={fit['std_b0']:.3g}, std(b1)={fit['std_b1']:.3g}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(n, g, yerr=sigma, fmt="o", capsize=4, label=f"media di {n_misure} misure")
    nf = np.linspace(n.min(), n.max(), 100)
    ax.plot(nf, modello(nf, *popt), "-", label=f"curve_fit: {fit['b0']:.2g} n^{fit['b1']:.2f}")
    ax.set_xlabel("n = qubit di indice")
    ax.set_ylabel("cx per campione (U_A controllata)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(png, dpi=200)
    plt.show()

    with open(filename, "w") as f:
        json.dump(to_jsonable(dict(rows=rows, fit=fit, fit_log=fit_log)), f, indent=2)
    return rows, fit


def fit_gradi(kappas=(2, 3, 4, 5, 6, 8, 10), sqrt_ps=(0.5, 0.3, 0.2, 0.1, 0.05), eps=0.05):
    """Degrees measured with pyqsp and fitted: deg_inv ~ a_inv k^2 ln(k/eps), scale ~ c/k, deg_sign ~ a_s/sqrt(p) + b_s.

    Args:
        kappas (tuple): condition numbers for 1/x.
        sqrt_ps (tuple): amplitudes for the sign.
        eps (float): precision of 1/x.

    Returns:
        dict: eps, a_inv, c_scale, a_s, b_s and the measured points.
    """
    deg_inv, scales = [], []
    for k in kappas:
        with suppress():
            ang, sc = save_angles(get_qsvt_angles_inverse, k, eps)
        deg_inv.append(len(ang) - 1)
        scales.append(float(np.asarray(sc).ravel()[0]))
    kap = np.array(kappas, float)
    feat = kap**2 * np.log(kap / eps)                              # pyqsp form for 1/x
    a_inv = float(np.sum(feat * np.array(deg_inv)) / np.sum(feat**2))
    c_scale = float(np.mean(np.array(scales) * kap))                # scale ~ c / kappa

    sp_ok, d_ok = [], []
    for sp in sqrt_ps:
        d, _ = scegli_deg_sign(sp, **SIGN_KW)
        if d is not None:
            sp_ok.append(sp)
            d_ok.append(d)
    Amat = np.vstack([1 / np.array(sp_ok), np.ones(len(sp_ok))]).T
    (a_s, b_s), *_ = np.linalg.lstsq(Amat, np.array(d_ok, float), rcond=None)

    out = dict(eps=eps, a_inv=a_inv, c_scale=c_scale, a_s=float(a_s), b_s=float(b_s),
               misure_inv=dict(kappa=list(kappas), deg=deg_inv, scale=scales),
               misure_sign=dict(sqrt_p=sp_ok, deg=d_ok))
    print(out)
    return out


def grado_inversa(kappa_poly, eps, gradi, kappa_max_pyqsp=10):
    """Degree of the TRUNCATED polynomial of 1/x (Childs-Kothari-Somma, arXiv:1511.02306).

    b and j0 as in pyqsp.PolyOneOverX: b = floor(k^2 ln(k/eps)), j0 = floor(sqrt(b ln(4b/eps))),
    degree = 2 j0 + 1. The simulations use the full pyqsp polynomial (degree 2b - 1): to be stated.
    The scale is the pyqsp one (same approximated function up to eps).

    Args:
        kappa_poly (int): condition number (integer).
        eps (float): precision.
        gradi (dict): output of fit_gradi (c_scale used for kappa > kappa_max_pyqsp).
        kappa_max_pyqsp (int): largest kappa for which pyqsp computes the scale.

    Returns:
        tuple: (degree, scale, source string).
    """
    b = int(kappa_poly**2 * np.log(kappa_poly / eps))
    j0 = int(np.sqrt(b * np.log(4 * b / eps)))
    deg = 2 * j0 + 1
    if kappa_poly <= kappa_max_pyqsp:
        with suppress():
            _, scale = save_angles(get_qsvt_angles_inverse, kappa_poly, eps)
        return deg, float(np.asarray(scale).ravel()[0]), "CKS troncato (scala pyqsp)"
    return deg, gradi["c_scale"] / kappa_poly, "CKS troncato (scala dal fit)"


def stima_fault_tolerant(d2q, d1q, n_qubit, delta=0.1, t_ciclo=1e-6, p_fis=1e-3, p_soglia=1e-2):
    """Time per shot on logical qubits (surface code).

    Assumptions to cite: cycle ~1 us, p_fis = 1e-3, threshold ~1e-2; every logical layer lasts d
    cycles; every 1-qubit layer is an arbitrary rotation synthesized with ~3 log2(1/eps_s) T layers
    (Ross-Selinger).

    Args:
        d2q (float): 2-qubit depth per shot.
        d1q (float): 1-qubit depth per shot.
        n_qubit (int): logical qubits.
        delta (float): failure probability (logical error <= delta/6).
        t_ciclo (float): duration of a code cycle in seconds.
        p_fis (float): physical error rate.
        p_soglia (float): threshold of the surface code.

    Returns:
        dict: distanza_codice, strati_logici, cicli_per_shot, t_shot_FT_s, qubit_fisici.
    """
    n_rot = max(float(d1q), 1.0)
    eps_s = delta / (6 * n_rot)                            # synthesis error per rotation
    R = 3 * np.log2(1 / eps_s)                             # T layers per rotation
    strati = float(d2q) + float(d1q) * R                   # sequential logical layers
    # smallest odd distance with total logical error <= delta/6
    d = 3
    while 0.1 * (p_fis / p_soglia) ** ((d + 1) / 2) * n_qubit * d * strati > delta / 6:
        d += 2
    cicli = strati * d
    return dict(distanza_codice=d, strati_logici=strati, cicli_per_shot=cicli,
                t_shot_FT_s=cicli * t_ciclo, qubit_fisici=int(n_qubit * 2 * d**2))


def stima_risorse(X, y, _lambda, tests, gradi, piattaforme=None, rapporto_p=1.0, C_M=1.0, eps=0.05, delta=0.1, c_teorico=204, c_pratico=65, R_F59=1.0, b_F58=1, kappa_max_pyqsp=10, sqrt_p_min_pyqsp=0.05):
    """Bottom-up resource estimate of the full algorithm on one dataset (no simulation).

    M is fixed by the error budget Q_tot eps1 <= eps_shot with M = C_M N_eff / eps1; the degree of
    the sign in O_col is updated self-consistently with eps1.

    Args:
        X (np.ndarray): (N, D) matrix.
        y (np.ndarray): (N,) labels.
        _lambda (float): regularization in (0, 1].
        tests (np.ndarray): (m, D) tests.
        gradi (dict): output of fit_gradi.
        piattaforme (dict | None): {name: (t_1q, t_2q, T2)} (default: platform).
        rapporto_p (float): correction p_measured / p_formula applied to sqrt(p).
        C_M (float): constant of M = C_M N_eff / eps1.
        eps (float): precision of 1/x.
        delta (float): failure probability.
        c_teorico (float): theoretical constant of the shots per group.
        c_pratico (float): practical constant of the shots per group.
        R_F59 (float): constant of F59.
        b_F58 (int): bits per entry in F58.
        kappa_max_pyqsp (int): largest kappa for pyqsp.
        sqrt_p_min_pyqsp (float): smallest sqrt(p) for pyqsp (below: fit).

    Returns:
        dict: instance data, degrees, shots, M, total samples, qubits, memory, cx, depths,
            paper estimates (F58, F59, F69), fault-tolerant estimate and physical times per platform.
    """
    if piattaforme is None:
        piattaforme = platform
    X, y, tests = np.asarray(X, float), np.asarray(y, float), np.asarray(tests, float)
    N, D = X.shape
    lam = float(_lambda)
    assert 0 < lam <= 1, f"serve 0 < lambda <= 1 (oracolo elemento), lambda = {lam}"
    r = ref(X, y, lam, tests)
    nb = lambda x: max(1, int(np.ceil(np.log2(x))))

    # 1. sketched block encoding: alpha = s_r s_c, kappa_s
    A_aug = np.vstack([X, np.sqrt(lam) * np.eye(D)])
    s_r = int((A_aug != 0).sum(1).max())
    s_c = int((A_aug != 0).sum(0).max())
    alpha = s_r * s_c
    s_min = float(np.linalg.svd(A_aug, compute_uv=False).min())
    kappa_s = alpha / s_min
    kappa_poly = int(np.ceil(kappa_s - 1e-6))
    b = int(np.ceil(np.log2(N + D)))

    # 2. degrees: 1/x (ours and with the paper alpha) and sign of the amplification
    deg_inv, scale, fonte_inv = grado_inversa(kappa_poly, eps, gradi, kappa_max_pyqsp)
    alpha_paper = max(s_r, s_c)
    kappa_paper = alpha_paper / s_min
    deg_inv_paper, _, _ = grado_inversa(int(np.ceil(kappa_paper - 1e-6)), eps, gradi, kappa_max_pyqsp)

    sqrt_p = float(scale * alpha * np.linalg.norm(np.asarray(r.w)) / np.linalg.norm(y)) * np.sqrt(rapporto_p)
    deg_sign = None
    if sqrt_p >= sqrt_p_min_pyqsp:
        deg_sign, _ = scegli_deg_sign(sqrt_p, **SIGN_KW)
    fonte_sign = "pyqsp"
    if deg_sign is None:
        deg_sign = int(np.ceil(gradi["a_s"] / sqrt_p + gradi["b_s"]))
        deg_sign += 1 - deg_sign % 2                       # odd
        fonte_sign = "fit (fit_gradi)"

    # 3. shots: half of the error to the statistics, half to the sketch
    m = tests.shape[0]
    K = K_MEDIANA(m, delta)
    eps_th = float(r.epsilon)
    eps_pr = float(np.min(np.abs(np.asarray(r.inner_product)))) / 3
    eps_shot = eps_pr / 2
    T_th = K * int(np.ceil(c_teorico / (eps_th / 2)**2))
    T_pr = K * int(np.ceil(c_pratico / (eps_pr / 2)**2))

    F59 = float(get_exact_num_samples(R_F59, N, int(r.s), float(r.k_reg), float(r.gamma_test), D, m, delta))  # comparison only
    N_eff = max(int(np.count_nonzero(A_aug)), N + D)        # Thm. D.12: p t^2 <= pi^2 N_eff for every block

    # 4. M from the error budget. The error of the sign in O_col per query must stay in the same
    #    eps1 budget as the samples: its degree scales as log(n_row/eps), computed with pyqsp at
    #    EPS_COL and rescaled in log (fixed-point iteration).
    d_col0 = grado_segno_col(s_c, b)[0]
    d_col = d_col0
    for _ in range(10):
        Q_tot = deg_sign * (deg_inv * (2 + d_col * b) + 1)
        eps1 = eps_shot / Q_tot                              # Q_tot * eps1 <= eps_shot
        d_new = int(np.ceil(d_col0 * np.log(b / eps1) / np.log(b / EPS_COL)))
        d_new += 1 - d_new % 2
        if d_new == d_col:
            break
        d_col = d_new
    Q_tot = deg_sign * (deg_inv * (2 + d_col * b) + 1)       # blocks of M samples in one shot
    eps1 = eps_shot / Q_tot                                  # Q_tot * eps1 <= eps_shot
    M = max(1, int(np.ceil(C_M * N_eff / eps1)))             # Thm. D.12: eps1 <= 2 p t^2 / M
    S_tot = T_pr * Q_tot * M

    # 5. qubits and classical memory of the reading
    n_row, n_col = b, nb(D)
    n_U = n_row + n_col + nb(s_r) + nb(s_c) + 2 + 2 * n_row
    qubit = n_U + 1 + 1 + 1
    qubit_riuso = qubit - 2 * (n_row - 1)            # one pair (anc_t, lcu_t) reused in the b steps of the binary search
    mem_lettura_float = K * m                        # running means of the median of means (tests known before the measurement)
    mem_tableau_bit = 2 * qubit * (2 * qubit + 1)    # tableau of the Clifford of the current shot (transient)
    mem_tableau_riuso_bit = 2 * qubit_riuso * (2 * qubit_riuso + 1)
    bit_classici_q = 64 * mem_lettura_float + mem_tableau_bit
    bit_classici_q_riuso = 64 * mem_lettura_float + mem_tableau_riuso_bit

    # gates and depth per shot (controlled: shadow control)
    cx_U = int(cost_UA_query(M, (N + D, D), s_r, M * s_r * n_col // 2, degree=d_col, controlled=True).get("cx", 0))
    cx_y = int(cost(True, int(np.log2(N)) + 1, M // 2).get("cx", 0))
    cx_pr = cx_proiettore(n_U + 1, controlled=True)
    cx_W = deg_inv * cx_U + cx_y + (deg_inv + 1) * cx_pr
    cx_V = deg_sign * cx_W + (deg_sign + 1) * cx_pr
    d_U = depth_UA_query(M, (N + D, D), s_r, M * s_r * n_col // 2, controlled=True, degree=d_col)
    d_y = depth_kick(True, int(np.log2(N)) + 1) * (M // 2)
    d_pr = depth_proiettore(n_U + 1, controlled=True)
    d_W = deg_inv * d_U + d_y + (deg_inv + 1) * d_pr
    d_V = deg_sign * d_W + (deg_sign + 1) * d_pr

    out = dict(N=N, D=D, nnz_X=int(np.count_nonzero(X)), s_r=s_r, s_c=s_c, lam=lam,
               kappa_reg=float(r.k_reg), kappa_s=kappa_s, gamma_test=float(r.gamma_test),
               deg_inv=deg_inv, fonte_deg_inv=fonte_inv, sqrt_p=sqrt_p, deg_sign=deg_sign, fonte_deg_sign=fonte_sign,
               query_U_per_shot=deg_sign * deg_inv, query_Uy_per_shot=deg_sign,
               T_teorico=T_th, T_pratico=T_pr, M=M, campioni_totali=S_tot,
               qubit=qubit, cx_query_U=cx_U, cx_per_shot=cx_V, cx_totali=T_pr * cx_V,
               paper_F59=F59, paper_F58=float(get_exact_num_qubits(D, m, float(r.gamma_test), delta, int(r.s), b_F58, float(r.k_reg))),
               paper_F69=float(compute_F69(kappa_s, eps)), d_col=d_col, mem_tableau_riuso_bit=mem_tableau_riuso_bit, bit_classici_q=bit_classici_q, bit_classici_q_riuso=bit_classici_q_riuso,
               eps1=eps1, C_M=C_M, N_eff=N_eff, Q_tot=Q_tot, S_su_F59=S_tot / F59, rapporto_p=rapporto_p,
               depth_2q_per_shot=float(d_V[0]), depth_1q_per_shot=float(d_V[1]),
               paper_F58_nota="limite superiore, costanti = 1", qubit_riuso=qubit_riuso, mem_lettura_float=mem_lettura_float, mem_tableau_bit=mem_tableau_bit,
               alpha=alpha, alpha_paper=alpha_paper, kappa_paper=kappa_paper, deg_inv_paper=deg_inv_paper)

    # 6. fault-tolerant time and physical time per platform (coherence per shot)
    out.update(stima_fault_tolerant(d_V[0], d_V[1], qubit, delta))
    out["t_totale_FT_s"] = T_pr * out["t_shot_FT_s"]
    for nome, (t1q, t2q, T2) in piattaforme.items():
        t_shot = d_V[0] * t2q + d_V[1] * t1q
        out[f"t_shot_fisico_{nome}_s"] = t_shot
        out[f"t_shot_su_T2_fisico_{nome}"] = t_shot / T2
        out[f"t_totale_fisico_{nome}_s"] = T_pr * t_shot
    return out


def tabella_finale(datasets, gradi, piattaforme=platform, C_M=2 * np.pi**2, rapporto_p=1.0, filename="tabella_finale_mod.json"):
    """Final table: stima_risorse + classical_cost for every dataset, saved as JSON and TXT.

    Args:
        datasets (dict): {name: (X, y, lam, tests)}.
        gradi (dict): output of fit_gradi.
        piattaforme (dict): {name: (t_1q, t_2q, T2)}.
        C_M (float): constant of M (2 pi^2 from Thm. D.12).
        rapporto_p (float): correction p_measured / p_formula.
        filename (str): output JSON (the TXT has the same name).

    Returns:
        list[dict]: one row per dataset.
    """
    righe = []
    for nome, (X, y, lam, tests) in datasets.items():
        riga = dict(dataset=nome)
        riga.update(stima_risorse(X, y, lam, tests, gradi, piattaforme, rapporto_p=rapporto_p, C_M=C_M))
        riga.update(classical_cost(X, y, lam, tests))
        righe.append(riga)
        print(riga)
    with open(filename, "w") as f:
        json.dump(to_jsonable(righe), f, indent=2)
    stampa_risultati(filename, "terminal")
    stampa_risultati(filename, "txt", filename.replace(".json", ".txt"))
    return righe


def valida_lettura(X, y, _lambda, tests, ref, n_campioni, key, delta=0.1, c=65, eps=0.05):
    """Interferometric shadow on the sketched state of W, with ideal amplification (P(sqrt p) = 1).

    Args:
        X (np.ndarray): (N, D) matrix.
        y (np.ndarray): (N,) labels.
        _lambda (float): regularization in (0, 1].
        tests (np.ndarray): (m, D) tests.
        ref (ClassicalReferenceResult): classical reference on the tests.
        n_campioni (int): samples per oracle (M).
        key (jax.Array): PRNG key (simulation and shadow).
        delta (float): failure probability.
        c (float): constant of the shots per group.
        eps (float): precision of 1/x.

    Returns:
        dict: M, T, eps_pr, accordo (fraction of signs equal to the classical ones), errore_max.
    """
    k_sim, k_sh = random.split(key)
    v, info = simula_W_a_pezzi(X, y, _lambda, n_campioni, k_sim, eps)
    D = X.shape[1]
    w = v[(2**info["n_row"]) * np.arange(D)]
    w_hat = w / np.linalg.norm(w)                              # ideal amplification: direction only
    # compact |w~> on ceil(log2 D) + 1 qubits
    n_w = max(1, int(np.ceil(np.log2(D))))
    w_pad = np.zeros(2**n_w, complex)
    w_pad[:D] = w_hat
    w_tilde = np.zeros(2**(n_w + 1), complex)
    w_tilde[0], w_tilde[1::2] = 1 / np.sqrt(2), w_pad / np.sqrt(2)
    x_hat = tests / np.linalg.norm(tests, axis=1, keepdims=True)
    Xp, Xm = embed_tests_rect(x_hat, 0, n_w)
    # median of means with the practical number of shots
    m = tests.shape[0]
    K = K_MEDIANA(m, delta)
    eps_pr = float(np.min(np.abs(np.asarray(ref.inner_product)))) / 3
    N_g = int(np.ceil(c / eps_pr**2))
    rng = np.random.default_rng(int(random.randint(k_sh, (), 0, 2**31 - 1)))
    dati = shadow_measure(None, rng, K * N_g, stato=Statevector(w_tilde))
    est = np.asarray(classical_result(Xp, Xm, n_w + 1, dati)).reshape(K, N_g, m)
    o_hat = np.median(est.mean(axis=1), axis=0)
    return dict(M=n_campioni, T=K * N_g, eps_pr=eps_pr,
                accordo=float(np.mean(np.sign(o_hat) == np.asarray(ref.y_pred))),
                errore_max=float(np.max(np.abs(o_hat - np.real(x_hat @ w_hat)))))


def valida_istanze(key, sizes=((4, 2), (4, 4)), M=100000, n_catena=5, filename="valida_lettura.json"):
    """valida_lettura on the instances of sweep_M (same key chain).

    Args:
        key (jax.Array): PRNG key (the same passed to sweep_M).
        sizes (tuple): (N, D) sizes.
        M (int): samples per oracle.
        n_catena (int): length of the key chain of sweep_M.
        filename (str): output JSON.

    Returns:
        list[dict]: one row per size.
    """
    out = []
    for N, D in sizes:
        key, k_inst = random.split(key)
        for _ in range(n_catena):
            key, _ = random.split(key)
        _, k_x, _ = random.split(k_inst, 3)
        X, y, lam, tests, r = ste.istanza_kappa_piccolo(k_x, N, D)
        out.append(dict(N=N, D=D, **valida_lettura(X, y, lam, tests, r, M, random.fold_in(k_inst, 1))))
        print(out[-1])
    with open(filename, "w") as f:
        json.dump(to_jsonable(out), f, indent=2)
    return out


"""
Plots
"""

def completa_riga(r, gradi):
    """Computes S_tot also when scegli_deg_sign finds no degree (sqrt(p) too small): degree from the fit.

    Args:
        r (dict): row of sweep_M.
        gradi (dict): output of fit_gradi.

    Returns:
        dict: copy of the row with degree_sign (and fonte_degree_sign if fitted) and S_tot.
    """
    r = dict(r)
    if r["degree_sign"] is None:
        d = int(np.ceil(gradi["a_s"] / np.sqrt(r["p"]) + gradi["b_s"]))
        r["degree_sign"] = d + 1 - d % 2                       # odd
        r["fonte_degree_sign"] = "fit (fit_gradi)"
    r["S_tot"] = r["T_pratico"] * r["degree_sign"] * r["campioni_W"]
    return r


def scegli_M_star(rr):
    """First M with all signs correct and infidelity within (overlap/3)^2.

    Args:
        rr (list[dict]): rows of one size sorted by increasing M.

    Returns:
        dict | None: the row of M*, or None.
    """
    for r in rr:
        infid_max = (r.get("overlap_test", 0.45) / 3) ** 2
        if r["correct"] == 1.0 and r["infid"] <= infid_max:
            return r
    return None


def grafico(rows=None, gradi=None, filename="sweep_m_simulazioni.json", png="sweep_m_simulazioni.png"):
    """Infidelity of W vs M for every size, with the fit a M^-p + c, M* and S/F59 at M*.

    Args:
        rows (list | None): rows of sweep_M (read from filename if None).
        gradi (dict | None): output of fit_gradi (needed if some degree_sign is None).
        filename (str): JSON of sweep_M.
        png (str): output figure.

    Returns:
        list[dict]: per size N, D, M_star, rapporto_S_F59, rapporto_p, p_su_p_formula and the fit.
    """
    if rows is None:
        with open(filename) as f:
            rows = json.load(f)
    fits = []
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for N, D in sorted({(r["N"], r["D"]) for r in rows}):
        rr = sorted([r for r in rows if (r["N"], r["D"]) == (N, D)], key=lambda r: r["M"])
        Ms = np.array([r["M"] for r in rr], float)
        inf = np.array([r["infid"] for r in rr], float)
        star = scegli_M_star(rr)
        if star is not None:
            star = completa_riga(star, gradi)
        fit = dict(N=N, D=D, M_star=None if star is None else star["M"],
                   rapporto_S_F59=None if star is None else star["S_tot"] / star["F59_ref"],
                   rapporto_p=None if star is None else star["p"] / star["p_formula"],
                   p_su_p_formula=[r["p"] / r["p_formula"] for r in rr])

        ax.errorbar(Ms, inf, yerr=[r["infid_std"] for r in rr], fmt="o", capsize=3, label=f"({N},{D})")

        if len(Ms) >= 4:
            (a, p, c), cov = curve_fit(func, Ms, np.log(inf), p0=(inf[0] * Ms[0], 1.0, inf[-1] * 0.1), bounds=([0, 0, 0], [np.inf, 3, np.inf]), maxfev=20000)
            print(N, D, a, p, c)
            dp = float(np.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) else float("nan")
            Mf = np.logspace(np.log10(Ms.min()), np.log10(Ms.max()), 200)
            print(f"({N},{D}) fit: p={p:.2f}±{dp:.2f}, c={c:.1e}")
            ax.plot(Mf, np.exp(func(Mf, a, p, c)), "-", label=f"({N},{D}) fit: p={p:.2f}±{dp:.2f}, c={c:.1e}")
            fit.update(a=a, p=p, dp=dp, c=c)

        fits.append(fit)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("M (campioni per query)")
    ax.set_ylabel("1 - |<w_ref|w>|^2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.show()

    rapporti = [f["rapporto_S_F59"] for f in fits if f["rapporto_S_F59"] is not None]
    if rapporti:
        print(f"S/F59 a M*: {rapporti} -> media {np.mean(rapporti):.3e}, dispersione relativa "
              f"{np.std(rapporti) / np.mean(rapporti):.2f}")
    return fits


"""
Tests
"""

def _test_build_V_full(key, degree_int=61, re_min=0.98):
    """Builds V on a small instance (degree of the sign chosen from the simulated sqrt(p)) and prints its operations.

    Args:
        key (jax.Array): PRNG key.
        degree_int (int): first degree of the sign tried (up to 201).
        re_min (float): minimum Re P(sqrt p).

    Returns:
        None
    """
    key, subkey, subkey2 = random.split(key, 3)
    # instance with lambda in (0, 1] and small kappa_s (needed by the element oracle)
    X, y, _lambda, _, _ = ste.istanza_kappa_piccolo(subkey)
    W, info_W = build_W_rect(X, y, _lambda, 1000, subkey2)

    # sqrt(p): norm of w in the J register (I = 0, ancillas = 0)
    v = simulate(W)
    sqrt_p = float(np.linalg.norm(v[(2**info_W["n_row"]) * np.arange(X.shape[1])]))
    deg_sign, thr = scegli_deg_sign(sqrt_p, re_min=re_min, d_min=degree_int, d_max=201, banda=None)
    assert deg_sign is not None, f"nessun grado <= 201 per sqrt(p) = {sqrt_p:.3f}"

    V, info = build_V_full(X, y, _lambda, 1000, subkey2, deg_sign=deg_sign, threshold=thr)
    print("\nInfo:", info)
    print(f"Number of operations: {V.count_ops()}")


def _test_binary_classification(key, T_A=4000, M=50):
    """(A) reading on the rectangular layout; (B) end-to-end run of binary_classification.

    (A) embed_tests_rect + shadow are unbiased for Re<x|v>, with v in the J register (index
        2^n_row * j). Small state built by hand.
    (B) end-to-end run with reduced degrees (W with 3 queries, sign of O_col of degree 3): checks
        that it runs and that the output has the right shape. The signs are NOT checked: with
        T = K shots the noise dominates; correctness is checked by the sweep of singol_instance.

    Args:
        key (jax.Array): PRNG key.
        T_A (int): shots of part (A).
        M (int): samples per oracle of part (B).

    Returns:
        None
    """
    print("Primo test")
    rng = np.random.default_rng(1)
    n_row, n_V, D, m = 2, 4, 2, 3
    v = rng.normal(size=2**n_V) + 1j * rng.normal(size=2**n_V)
    v /= np.linalg.norm(v)
    x = rng.normal(size=(m, D))
    x_hat = x / np.linalg.norm(x, axis=1, keepdims=True)

    Xp, Xm = embed_tests_rect(x_hat, n_row, n_V)
    w_tilde = np.zeros(2**(n_V + 1), dtype=complex)
    w_tilde[0] = 1 / np.sqrt(2)
    w_tilde[1::2] = v / np.sqrt(2)
    target = np.real(x_hat @ v[(2**n_row) * np.arange(D)])

    dati = shadow_measure(None, rng, T_A, stato=Statevector(w_tilde))
    est = np.asarray(classical_result(Xp, Xm, n_V + 1, dati))   # (T_A, m)
    media, errore = est.mean(axis=0), est.std(axis=0) / np.sqrt(T_A)
    assert np.all(np.abs(media - target) < 4 * errore), f"(A) stima distorta: media {media}, atteso {target}, 4*sigma {4 * errore}"
    print(f"(A) lettura ok: |media - target| / sigma = {np.abs(media - target) / errore}")
    """
    (A) lettura ok: |media - target| / sigma = [0.20024518 0.53239041 0.1950713 ]
    """

    print("Secondo test")
    X = y = lam = tests = r = None
    for _ in range(100):
        key, k_x, k_t = random.split(key, 3)
        X, y, lam = ste.small_instance_generator(k_x, 4, 2, kappa_target=1.5)
        lam = float(lam)
        if not (0 < lam <= 1):
            continue
        try:
            tests, r = select_tests(k_t, X, y, lam, 5, 0.45)
            break
        except AssertionError:
            continue
    assert r is not None, "(B) nessuna istanza valida trovata"

    m = tests.shape[0]
    K = int(np.ceil(2 * np.log(2 * m / 0.1)))
    # degree 3 for the sign in O_col: make_UA passes degree_col explicitly, so the default of
    # block_encoding_from_oracles (degree_col) is patched, not the one of col_index_oracle
    old_be = sketch_oracles.block_encoding_from_oracles.__defaults__
    sketch_oracles.block_encoding_from_oracles.__defaults__ = (False, 3)      # (stampa, degree_col)
    try:
        t0 = time.time()
        key, k_run = random.split(key)
        segni = binary_classification(X, y, lam, tests, M, k_run, deg_sign=3, threshold=0.2, gamma=float(r.gamma_test), s=int(r.s), T_max=K, angles_W=np.zeros(4))
    finally:
        sketch_oracles.block_encoding_from_oracles.__defaults__ = old_be
    assert segni.shape == (m,), f"(B) forma {segni.shape}, attesa ({m},)"
    assert set(np.unique(segni)) <= {-1.0, 0.0, 1.0}, f"(B) valori non validi: {np.unique(segni)}"
    accordo = float(np.mean(segni == np.asarray(r.y_pred)))
    print(f"(B) end-to-end ok in {time.time() - t0:.0f} s: {K} shot, segni {segni}, " f"RidgeClassifier {np.asarray(r.y_pred)} (accordo {accordo:.0%}, solo informativo)")
    """
    risultato (with the old patch, which did not reach O_col):
        (B) end-to-end ok in 121 s: 10 shot, segni [-1.  1. -1.  1.  0.], RidgeClassifier [ 1. -1.  1.  1. -1.] (accordo 20%, solo informativo)
    """


if __name__ == "__main__":
    print("---"*10)
    print("binary_classification.py")
    print("---"*10)
    key, k1, k2, k3 = random.split(key, 4)
    #_test_build_V_full(key)
    #_test_binary_classification(key)
    #fit_costi(k2)
    gradi = fit_gradi()
    #taglie = ((2**10, 2**6), (2**12, 2**8), (2**14, 2**10))
    #ks = random.split(k3, len(taglie))
    #datasets = {f"sint_{N}x{D}": dataset_sintetico(k, N=N, D=D, overlap_min=0.4, n_cand=5000)
    #            for k, (N, D) in zip(ks, taglie)}
    #tabella_finale(datasets, gradi)
    # long, not needed for the table:
    #valida_istanze(k1)
    #rows = sweep_M(k1, aggiungi=False, filename="sweep_m_simulazioni_v2.json")
    fits = grafico(gradi=gradi, filename="sweep_m_simulazioni.json", png="sweep_m_simulazioni_v2.png")
