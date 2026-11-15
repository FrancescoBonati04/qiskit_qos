"""
Quantum ridge solver (Lemma F.14): circuit W that prepares |w> with QSVT of 1/x on the
block encoding of A_aug, and amplitude amplification (standard and fixed point with the
sign QSVT) that brings the success probability close to 1.
"""

import json
import gc

import numpy as np
import jax.numpy as jnp
from jax import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator

from utilities import Connect, save_angles, cont_oper
import small_toy_exp as ste
import block_encoding as be
import qsvt_qiskit as qsvt_q
from reference import classical_reference as class_ref, compute_F69

key = Connect()
from utils import infidelity, suppress_stdout_stderr as suppress
from qsvt import get_qsvt_angles_inverse, apply_qsvt, get_qsvt_angles_sign


def sign_poly_value(angles, a):
    """Value P(a) of the QSVT polynomial: repo QSVT on the Halmos dilation of the scalar a.

    Args:
        angles (array): QSVT angles.
        a (float): point in [0, 1].

    Returns:
        complex: P(a).
    """
    U = np.array([[a, np.sqrt(1 - a * a)], [np.sqrt(1 - a * a), -a]])
    return complex(np.asarray(apply_qsvt(jnp.asarray(U), 1, jnp.asarray(angles)))[0, 0])


def scegli_deg_sign(sqrt_p, re_min=0.98, d_min=21, d_max=801, banda=(0.8, 1.2), n_punti=9):
    """Smallest degree of the sign polynomial with Re P >= re_min for the amplification.

    With banda the condition is required on the whole band [banda[0], banda[1]] * sqrt(p)
    (the polynomial oscillates; with sketched oracles sqrt(p) changes from shot to shot).
    With banda=None it is required only at sqrt(p) (version used by binary_classification).

    Args:
        sqrt_p (float): amplitude to amplify.
        re_min (float): minimum Re P.
        d_min (int): first degree tried.
        d_max (int): last degree tried (step 10).
        banda (tuple | None): relative band around sqrt(p), or None for the single point.
        n_punti (int): number of points of the band.

    Returns:
        tuple: (degree or None if no degree works, threshold = 0.7 sqrt(p) with 3 significant digits).
    """
    thr = float(f"{0.7 * sqrt_p:.3g}")
    if banda is None:
        xs = [sqrt_p]
    else:
        xs = np.linspace(banda[0] * sqrt_p, min(banda[1] * sqrt_p, 1.0), n_punti)
    for d in range(d_min, d_max + 1, 10):
        with suppress():
            ang, _ = save_angles(get_qsvt_angles_sign, d, thr, 0.99)
        ang = np.asarray(ang, dtype=float)
        if min(sign_poly_value(ang, x).real for x in xs) >= re_min:
            return d, thr
    return None, thr


def build_W(X, y, _lambda, eps, fast_lcu: bool = True, sketch=None, key=None):
    """Circuit W: prepares |y~> and applies Re P(H/alpha) with P ~ scale/x (QSVT of 1/x).

    Args:
        X (np.ndarray): (N, D) matrix.
        y (np.ndarray): (N,) labels.
        _lambda (float): regularization strength.
        eps (float): precision of the polynomial of 1/x.
        fast_lcu (bool): rzz version of the LCU (True) or Hadamard test with V_Phi, V_-Phi (False).
        sketch (dict | None): if given, sketched oracles: keys "alpha_s", "n_sys",
            "new_Uy"(key) and "new_U"(key) (a fresh block encoding at every query).
        key (jax.Array | None): PRNG key of the sketched oracles (only with sketch).

    Returns:
        tuple: (W, info) with W on n_sys + 2 qubits (block-encoding ancilla, LCU) and info a
            dict with N, D, n_sys, anc, w_slice, alpha, scale, kappa_reg, kappa_poly, degree.
    """
    N, D = X.shape
    if sketch is None:
        be_circ, alpha, H = be.block_encoding(X, _lambda)
        y_prep, y_enc = be.state_prep(X, y)
        n_sys = y_prep.num_qubits
    else:
        key, k_y, k_u = random.split(key, 3)
        alpha, n_sys = sketch["alpha_s"], sketch["n_sys"]
        y_prep, _ = sketch["new_Uy"](k_y)

    W = QuantumCircuit(n_sys + 2)
    W.append(y_prep, range(n_sys))

    # condition number of the block encoding -> degree of 1/x
    A_aug = jnp.vstack([X, jnp.sqrt(_lambda) * jnp.eye(D)])
    S_min = jnp.linalg.svd(A_aug, compute_uv=False).min()
    kappa_reg = alpha / S_min
    kappa_poly = jnp.ceil(kappa_reg - 1e-6)
    angles, scale = save_angles(get_qsvt_angles_inverse, kappa_poly, eps)

    # real part of the QSVT of 1/x
    if sketch is None:
        qc = qsvt_real_fast(be_circ, angles) if fast_lcu else qsvt_q.qsvt_circuit_real(be_circ, angles)
    else:
        U_list = [sketch["new_U"](k) for k in random.split(k_u, len(angles) - 1)]
        qc = qsvt_q.qsvt_circuit_real_fast_list(U_list, angles)
    W.compose(qc, inplace=True)

    info = dict(
        N=N,
        D=D,
        n_sys=n_sys,
        anc=[n_sys, n_sys + 1],
        w_slice=slice(N + D, N + 2 * D),
        alpha=alpha,
        scale=scale,
        kappa_reg=kappa_reg,
        kappa_poly=kappa_poly,
        degree=len(angles) - 1,
    )
    return W, info


def qsvt_real_fast(be_circ, angles):
    """Fast version of qsvt_circuit_real for a single block encoding (rzz trick).

    Args:
        be_circ (QuantumCircuit): block encoding (ancilla = last qubit).
        angles (array): QSVT angles.

    Returns:
        QuantumCircuit: circuit on be_circ.num_qubits + 1 qubits (LCU qubit = last).
    """
    angles = np.asarray(angles, dtype=float)
    n = be_circ.num_qubits
    anc, lcu = n - 1, n
    qc = QuantumCircuit(n + 1)
    qc.h(lcu)
    for phi in angles[::-1][:-1]:
        qc.rzz(-2 * phi, anc, lcu)
        qc.compose(be_circ, range(n), inplace=True)
    qc.rzz(-2 * angles[0], anc, lcu)
    qc.rz(np.pi * len(angles), lcu)
    qc.h(lcu)
    return qc


def general_reflection(qc, qubits):
    """Reflection I - 2|0><0| on the given qubits (up to a global phase).

    Args:
        qc (QuantumCircuit): circuit modified in place.
        qubits (iterable[int]): qubits of the reflection.

    Returns:
        QuantumCircuit: qc.
    """
    qubits = list(qubits)
    qc.x(qubits)
    # phase pi on |1...1>, i.e. on |0...0> before the X gates
    qc.mcp(jnp.pi, qubits[:-1], qubits[-1])
    qc.x(qubits)
    return qc


def amplify_standard(W, info, k: int):
    """Generalized Grover amplification: W (R_anc W^dagger R_0 W)^k.

    Args:
        W (QuantumCircuit): circuit to amplify.
        info (dict): info of build_W (uses info["anc"]).
        k (int): number of Grover iterations.

    Returns:
        QuantumCircuit: amplified circuit.
    """
    W_dagger = W.inverse()
    qc = QuantumCircuit(W.num_qubits)
    qc.compose(W, inplace=True)
    for _ in range(k):
        general_reflection(qc, info['anc'])
        qc.compose(W_dagger, inplace=True)
        general_reflection(qc, range(W.num_qubits))
        qc.compose(W, inplace=True)
        qc.global_phase += jnp.pi
    return qc


def projector_phase_lcu(qc, phi, proj, lcu):
    """Projector phase e^{+-i phi (2 Pi - I)} controlled by the LCU qubit (sign + for lcu=0, - for lcu=1).

    Args:
        qc (QuantumCircuit): circuit modified in place.
        phi (float): phase.
        proj (list[int]): qubits whose all-zero state defines the projector Pi.
        lcu (int): LCU qubit.

    Returns:
        QuantumCircuit: qc.
    """
    phi = float(phi)
    qc.rz(2*phi, lcu)
    qc.x(proj)
    qc.mcp(2 * phi, proj[:-1], proj[-1])
    qc.mcp(-4 * phi, proj[:-1] + [lcu], proj[-1])
    qc.x(proj)
    return qc


def amplify_fixed_point(W, info, degree, threshold, rescale):
    """Fixed-point amplification: real part of the sign QSVT applied to W (alternating W, W^dagger).

    Args:
        W (QuantumCircuit | list[QuantumCircuit]): circuit W, or a list of degree circuits W
            (a fresh sketch at every query).
        info (dict): info of build_W (uses info["anc"]).
        degree (int): degree of the sign polynomial.
        threshold (float): threshold of the sign polynomial.
        rescale (float): rescaling of the sign polynomial.

    Returns:
        tuple: (qc, info_) with qc on W.num_qubits + 1 qubits and info_ = info plus anc (with
            the LCU qubit), degree_sign, threshold, queries_W, angles_sign.
    """
    w_list = W if isinstance(W, list) else None
    W0 = w_list[0] if w_list else W
    num_qubits = W0.num_qubits
    angles, _ = save_angles(get_qsvt_angles_sign, degree, threshold, rescale)
    d = len(angles)
    if w_list:
        assert len(w_list) == d - 1, f"{len(w_list)} W per {d - 1} query"
    angles = jnp.asarray(angles, dtype=float)
    proj_pari = list(range(num_qubits))      # projector |0...0><0...0| on all qubits (input of W)
    proj_dispari = info['anc']               # projector on the ancillas of W (output)

    qc = QuantumCircuit(num_qubits + 1)
    lcu = num_qubits
    qc.h(lcu)
    # alternating sequence: W on even steps, W^dagger on odd steps
    for j, ang in enumerate(angles[::-1][:-1]):
        Wj = w_list[j] if w_list else W
        Wj_dagger = Wj.inverse()
        projector_phase_lcu(qc, ang, proj_pari if j % 2 == 0 else proj_dispari, lcu)
        qc.compose(Wj if j % 2 == 0 else Wj_dagger, inplace=True)
    projector_phase_lcu(qc, angles[0], proj_dispari, lcu)
    qc.rz(jnp.pi * d, lcu)
    qc.h(lcu)

    info_ = dict(
        info,
        anc=proj_dispari + [lcu],
        degree_sign=degree,
        threshold=threshold,
        queries_W=degree,
        angles_sign=angles
    )
    return qc, info_


def create_instance(key):
    """Random instance with exact block encoding, |y~> and classical reference (for the tests).

    Args:
        key (jax.Array): PRNG key.

    Returns:
        tuple: (ref, H, alpha, y_enc, N, D, be_circ).
    """
    key, subkey = random.split(key)
    A, y, _lambda = ste.small_instance_generator(subkey)
    N, D = A.shape
    be_circ, alpha, H = be.block_encoding(A, _lambda)
    __, y_enc = be.state_prep(A, y)
    key, subkey = random.split(key)
    x_prime = ste.sparse_vector_gen(subkey)
    ref = class_ref(A, y, _lambda, x_prime)
    return ref, H, alpha, y_enc, N, D, be_circ


"""
Tests
"""

def _test_theorem_math(key):
    """w = A_aug^+ y~ coincides with the classical ridge solution.

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    key, subkey = random.split(key)
    X, y, _lambda = ste.small_instance_generator(subkey)
    N, D = X.shape
    y_tilde = jnp.append(y, jnp.zeros(D))
    A_aug = jnp.vstack([X, jnp.sqrt(_lambda) * jnp.eye(D)])
    w = jnp.linalg.pinv(A_aug) @ y_tilde

    key, subkey = random.split(key)
    x_prime = ste.sparse_vector_gen(subkey)
    ref = class_ref(X, y, _lambda, x_prime)
    assert jnp.allclose(w, ref.w, atol=1e-8), "Errore nell'implementazione"
    print("w calcolato con successo!")


def _test_inverse_qsvt_on_A(key):
    """Exact 1/x on the Hermitian dilation: f(H)(y~, 0) = (0, w).

    Checks that the entries outside the w register vanish and that the direction of w matches
    the classical reference.

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    ref, H, alpha, y_enc, N, D, be_circ = create_instance(key)

    # pseudo-inverse of H/alpha (eigenvalues below 1e-12 are set to 0)
    w, V = np.linalg.eigh(np.asarray(H / alpha))
    tol = 1e-12
    w_inv = jnp.where(jnp.abs(w) >= tol, 1.0 / jnp.where(jnp.abs(w) >= tol, w, 1.0), 0.0)
    V = (V * w_inv) @ V.conj().T
    ris = V @ y_enc

    outside = np.concatenate([ris[:N + D], ris[N + 2*D:]])
    assert jnp.allclose(outside, 0, atol=1e-8), "Assert error: le componenti fuori da N+D e N+2D non sono nulle!"
    fid = 1 - infidelity(ris[N+D:N+2*D]/jnp.linalg.norm(jnp.asarray(ris[N+D:N+2*D])), ref.w/jnp.linalg.norm(ref.w))
    assert jnp.allclose(fid, 1, atol=1e-8), "Assert error: fidelity not close to 1!"
    print("_test_inverse_qsvt_on_A() concluso con successo")


def _test_k(key):
    """alpha / sigma_min(A_aug) of the exact block encoding equals k_reg of the classical reference.

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    key, subkey = random.split(key)
    A, y, _lambda = ste.small_instance_generator(subkey)
    D = A.shape[1]
    _, alpha, _ = be.block_encoding(A, _lambda)
    A_aug = jnp.vstack([A, jnp.sqrt(_lambda) * jnp.eye(D)])
    kappa = alpha / jnp.linalg.svd(A_aug, compute_uv=False).min()
    key, subkey = random.split(key)
    ref = class_ref(A, y, _lambda, ste.sparse_vector_gen(subkey))
    print("k_ref =", ref.k_reg, " kappa =", float(kappa))
    err = float(jnp.abs(kappa - ref.k_reg))

    print(f"Errore assoluto su kappa: {err:.3e}")
    print(f"kappa calcolato: {float(kappa):.10f}")
    print(f"kappa reference: {float(ref.k_reg):.10f}")
    # SVD vs eigvalsh of X^T X + lambda I: agreement at float32 precision
    assert jnp.allclose(kappa, ref.k_reg, atol=1e-5), "kappa computed not close to classical reference!"
    print("_test_k() concluso con successo")


def _test_poly(key):
    """As _test_inverse_qsvt_on_A, with 1/x approximated by the QSVT polynomial of the repo.

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    ref, H, alpha, y_enc, N, D, be_circ = create_instance(key)
    _, subkey = random.split(key)
    _, y, _ = ste.small_instance_generator(subkey)      # same y as create_instance

    kappa = 4
    epsilon = 0.1
    angles, scale = save_angles(get_qsvt_angles_inverse, kappa, epsilon)
    V = apply_qsvt(Operator(be_circ).data, 1, angles)
    dim = len(y_enc)
    block = jnp.real(V[:dim, :dim])
    ris = block @ y_enc

    # odd polynomial: the output is only in the w register
    outside = np.concatenate([ris[:N + D], ris[N + 2*D:]])
    assert jnp.allclose(outside, 0, atol=1e-8), "Assert error: le componenti fuori da N+D e N+2D non sono nulle!"
    # approximate polynomial: fidelity within the precision epsilon
    fid = 1 - infidelity(ris[N+D:N+2*D]/jnp.linalg.norm(jnp.asarray(ris[N+D:N+2*D])), ref.w/jnp.linalg.norm(ref.w))
    assert fid > 1 - epsilon**2, f"Assert error: fidelity {fid} < 1 - epsilon^2!"
    prob = jnp.sum(np.abs(ris)**2)
    expected = (scale * alpha * jnp.linalg.norm(ref.w) / jnp.linalg.norm(y))**2
    assert abs(prob - expected) < 1e-3, "Assert error: probability not right computed"
    print(f"_test_poly() concluso con successo, errore = {abs(prob-expected):.2e}")


def _test_build_W(key):
    """build_W on 5 instances: fidelity with w and success probability (scale alpha ||w|| / ||y||)^2.

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    for i in tqdm(range(5)):
        key, subkey = random.split(key)
        A, y, _lambda = ste.small_instance_generator(subkey)
        key, subkey = random.split(key)
        x_prime = ste.sparse_vector_gen(subkey)
        ref = class_ref(A, y, _lambda, x_prime)
        eps = 0.1
        with suppress():
            W, info = build_W(A, y, _lambda, eps)

        # part of the state with both ancillas in |0>
        zero_anc = Statevector(W).data[:2**info["n_sys"]]

        w_norm = zero_anc[info["w_slice"]]
        w_norm = w_norm / np.linalg.norm(w_norm)
        fid = abs(np.vdot(ref.w_ket, w_norm))**2
        assert fid > (1 - eps**2), "Assert error: |<w_ref|w_circ>|^2 < 1-epsilon'!"

        prob = float(np.sum(np.abs(zero_anc)**2))
        expected = (info['scale'] * info['alpha'] * jnp.linalg.norm(ref.w) / jnp.linalg.norm(y))**2
        assert abs(prob - expected)/expected < eps, "Assert error: probability not right computed"

    print("_test_build_W() concluso con successo su 5 istanze")


def _test_amplify_standard(key):
    """Grover amplification: success probability sin^2((2k+1) theta) and fidelity preserved.

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    for i in tqdm(range(5)):
        eps = 0.05
        key, subkey = random.split(key)
        X, y, _lambda = ste.small_instance_generator(subkey)
        with suppress():
            W, info = build_W(X, y, _lambda, eps)
        key, subkey = random.split(key)
        ref = class_ref(X, y, _lambda, ste.sparse_vector_gen(subkey))

        # optimal number of iterations from the initial probability
        zero_anc = Statevector(W).data[:2**info["n_sys"]]
        prob = float(np.sum(np.abs(zero_anc)**2))
        k = int(jnp.floor(jnp.pi/(4*(jnp.arcsin(jnp.sqrt(prob))))))

        with suppress():
            qc_amplify = amplify_standard(W, info, k)

        zero_anc = Statevector(qc_amplify).data[:2**info["n_sys"]]
        prob_ = float(np.sum(np.abs(zero_anc)**2))
        w_norm = zero_anc[info["w_slice"]]
        w_norm = w_norm / np.linalg.norm(w_norm)
        fid = abs(np.vdot(ref.w_ket, w_norm))**2
        expected_prob = np.sin((2*k + 1) * (np.arcsin(np.sqrt(prob))))**2

        assert abs(prob_ - expected_prob) < 1e-8, "Asssert error: probability"
        assert fid > 1 - eps**2, "Assert error: fidelity"

    print("_test_amplify_standard() concluso con successo")


def _test_amplify_fixed_point(key, kappa_target=None, eps=0.05, compute_statevectors=True):
    """Fixed-point amplification.

    Without kappa_target: 5 instances, checks that the amplified state is Re P(sqrt p) times the
    state of W, success > 0.95 and fidelity > 1 - eps^2. With kappa_target: one instance with
    degree and threshold chosen from its sqrt(p); returns the query counts (used for F69).

    Args:
        key (jax.Array): PRNG key.
        kappa_target (float | None): target condition number.
        eps (float): precision of 1/x.
        compute_statevectors (bool): run the statevector checks (first mode only).

    Returns:
        list[dict]: query counts of every instance (cont_oper), plus sqrt_p, deg_sign, deg_inv
            in the kappa_target mode.
    """
    results = []
    if kappa_target is None:
        for i in range(5):
            key, subkey = random.split(key)
            X, y, _lambda = ste.small_instance_generator(subkey, kappa_target=kappa_target)
            with suppress():
                W, info = build_W(X, y, _lambda, eps)

            if compute_statevectors:
                zero_anc = Statevector(W).data[:2**info["n_sys"]]
                norm = jnp.linalg.norm(zero_anc)

            key, subkey = random.split(key)
            ref = class_ref(X, y, _lambda, ste.sparse_vector_gen(subkey))
            qc, info_afp = amplify_fixed_point(W, info, 61, 0.1, 0.99)

            if compute_statevectors:
                zero_anc_afp = Statevector(qc).data[:2**info_afp['n_sys']]
                prob = sign_poly_value(info_afp["angles_sign"], norm)

                # the amplified state is Re P(sqrt p) * (state of W) / sqrt p
                assert np.allclose(zero_anc_afp, prob.real * zero_anc / norm, atol=1e-8), "Assert error: 1"
                assert np.abs(zero_anc_afp.imag).max() < 1e-10, "Assert error: 2"

                p_succ = float(np.sum(np.abs(zero_anc_afp)**2))
                w_circ = zero_anc_afp[info["w_slice"]] / np.linalg.norm(zero_anc_afp[info["w_slice"]])
                fid = abs(np.vdot(ref.w_ket, w_circ))**2
                assert p_succ > 0.95, f"istanza {i}: successo {p_succ:.4f}"
                assert fid > 1 - eps**2, f"istanza {i}: fidelity {fid:.6f}"
                print(f"istanza: p = {norm**2:.4f} -> successo {p_succ:.4f}, fidelity {fid:.8f}, " f"P(sqrt p) = {prob.real:+.3f}{prob.imag:+.3f}i")

            results.append(cont_oper(qc))
            del W, info_afp, ref, qc, info
            gc.collect()
    else:
        # F69 check: degree and threshold of the amplification chosen from sqrt(p) of the instance
        key, subkey = random.split(key)
        X, y, _lambda = ste.small_instance_generator(subkey, kappa_target=kappa_target)
        with suppress():
            W, info = build_W(X, y, _lambda, eps)
            sqrt_p = float(np.linalg.norm(Statevector(W).data[:2**info["n_sys"]]))
            deg_sign, thr = scegli_deg_sign(sqrt_p)
        assert deg_sign is not None, f"nessun grado <= 801 per sqrt(p) = {sqrt_p:.3f}"
        qc, info_afp = amplify_fixed_point(W, info, deg_sign, thr, 0.99)
        results.append(dict(cont_oper(qc), sqrt_p=sqrt_p, deg_sign=deg_sign, deg_inv=info["degree"]))
        del W, info_afp, qc, info
        gc.collect()

    return results


def grafico(kappas_target, epss, filename="results.json"):
    """Measured queries vs a* F69(kappa, eps), with a constant a* fitted on every curve.

    Panel 1: one curve per kappa (x = eps); panel 2: one curve per eps (x = kappa).
    a* minimizes sum_i (ln Q_i - ln(a F69_i))^2, i.e. ln a* = mean of ln(Q_i / F69_i).

    Args:
        kappas_target (list[float]): kappa values of the sweep.
        epss (list[float]): eps values of the sweep.
        filename (str): JSON written by _verify_num_queries.

    Returns:
        tuple: (a_kappa, a_eps) dicts of the fitted constants.
    """
    with open(filename, "r", encoding="utf-8") as f:
        results = json.load(f)

    def queries(r):
        return sum(op["U"] + op["U_y"] for op in r["num_queries"])

    def a_star(Q, F):
        Q, F = np.asarray(Q, float), np.asarray(F, float)
        return float(np.exp(np.mean(np.log(Q / F))))

    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    colors1 = plt.get_cmap("viridis", len(kappas_target))
    colors2 = plt.get_cmap("plasma", len(epss))
    k_coord = np.linspace(min(kappas_target), max(kappas_target), 500)
    eps_coord = np.linspace(min(epss), max(epss), 500)

    # panel 1: one constant per kappa
    a_kappa = {}
    for i, k in enumerate(kappas_target):
        dati = sorted([r for r in results if r["kappa_target"] == k], key=lambda r: r["eps"])
        x = np.array([r["eps"] for r in dati], float)
        Q = np.array([queries(r) for r in dati], float)
        F = np.array([float(compute_F69(k, e)) for e in x])
        a_kappa[k] = a_star(Q, F)
        axs[0].scatter(x, Q, marker="o", color=colors1(i), label=f"kappa={k}")
        axs[0].plot(eps_coord, a_kappa[k] * np.asarray(compute_F69(k, eps_coord)), color=colors1(i),
                    linestyle="--", label=f"{k}: a*={a_kappa[k]:.1f}")

    axs[0].set_xlabel("eps")
    axs[0].set_ylabel("Numero di query")
    axs[0].set_title("Dipendenza da epsilon")
    axs[0].legend(fontsize=8)
    axs[0].set_yscale("log")
    axs[0].grid(True, alpha=0.3)

    # panel 2: one constant per eps, plus the power-law exponent in kappa
    a_eps = {}
    for i, eps in enumerate(epss):
        dati = sorted([r for r in results if r["eps"] == eps], key=lambda r: r["kappa_target"])
        x = np.array([r["kappa_target"] for r in dati], float)
        Q = np.array([queries(r) for r in dati], float)
        F = np.array([float(compute_F69(k, eps)) for k in x])
        esp = np.polyfit(np.log(x), np.log(Q), 1)[0]
        print(f"eps = {eps}: Q ~ kappa^{esp:.2f}  (F69: kappa^1 a meno di log)")
        a_eps[eps] = a_star(Q, F)
        axs[1].scatter(x, Q, marker="o", color=colors2(i), label=f"epsilon={eps}")
        axs[1].plot(k_coord, a_eps[eps] * np.asarray(compute_F69(k_coord, eps)), color=colors2(i),
                    linestyle="--", label=f"{eps}: a*={a_eps[eps]:.1f}")

    axs[1].set_xlabel("kappa")
    axs[1].set_ylabel("Numero di query")
    axs[1].set_title("Dipendenza dalla complessità teorica")
    axs[1].legend(fontsize=8)
    axs[1].set_yscale("log")
    axs[1].grid(True, alpha=0.3)

    print("a* per kappa (pannello 1):")
    for k, a in a_kappa.items():
        print(f"  kappa = {k:>4}: a* = {a:.3f}")
    print("a* per eps (pannello 2):")
    for e, a in a_eps.items():
        print(f"  eps = {e:>5}: a* = {a:.3f}")
    tutte = np.array(list(a_kappa.values()) + list(a_eps.values()))
    print(f"intervallo a*: [{tutte.min():.3f}, {tutte.max():.3f}], rapporto max/min = {tutte.max() / tutte.min():.2f}")

    plt.tight_layout()
    plt.show()
    return a_kappa, a_eps


def _verify_num_queries(key, kappas_target, epss):
    """Query counts of the full solver (W + amplification) on a kappa x eps grid, saved in results.json.

    Args:
        key (jax.Array): PRNG key.
        kappas_target (list[float]): target condition numbers.
        epss (list[float]): precisions of 1/x.

    Returns:
        None
    """
    results = []
    for kappa_target in tqdm(kappas_target):
        for eps in tqdm(epss):
            with suppress():
                queries = _test_amplify_fixed_point(key, kappa_target, eps, False)
            results.append({"kappa_target": kappa_target, "eps": eps, "num_queries": queries})

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print("File salvato in memoria")


if __name__ == "__main__":
    kappas_target = [2, 4, 8, 10, 12, 16]
    epss = [1, 0.5, 0.2, 0.1, 0.05, 0.01]
    print("---"*10)
    print("ridge_solver.py")
    print("---"*10)
    #_test_theorem_math(key)
    #_test_inverse_qsvt_on_A(key)
    #_test_k(key)
    #_test_poly(key)
    #_test_build_W(key)
    #_test_amplify_standard(key)
    #r = _test_amplify_fixed_point(key)
    _verify_num_queries(key, kappas_target, epss)
    grafico(kappas_target, epss)
