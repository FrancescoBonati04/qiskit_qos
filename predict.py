"""
Prediction with the ideal oracles: W + fixed-point amplification + interferometric shadow,
choice of the parameters, selection of the test vectors and resource table.
"""

import json
import gc
import time
from collections import Counter
from datetime import timedelta

import numpy as np
import jax.numpy as jnp
from jax import random
from tqdm import tqdm
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from utilities import Connect, cont_oper, save_angles, stampa_risultati
import small_toy_exp as ste
from ridge_solver import build_W, amplify_fixed_point, sign_poly_value
from shadow import shadow_measure, classical_result, embed_tests
from reference import classical_reference as ref

BASIS = ("cx", "rz", "sx", "x")
key = Connect()
from utils import suppress_stdout_stderr as suppress
from qsvt import get_qsvt_angles_sign


def gate_counter(qc: QuantumCircuit, basis, controlled):
    """Gate count and depth of qc, transpiling each distinct top-level operation only once.

    Args:
        qc (QuantumCircuit): circuit (e.g. V).
        basis (tuple[str]): basis gates of the transpilation.
        controlled (bool): count the singly-controlled version of every operation (shadow control).

    Returns:
        tuple: (Counter of the gates, sum of the depths of the operations).
    """
    cost = Counter()
    depth = 0
    seen_gates = {}
    for istr in qc.data:
        op = istr.operation
        op_info = (op.name, op.num_qubits)
        # transpile only the first occurrence of each (name, size)
        if op_info not in seen_gates:
            if controlled:
                op = op.control(1)
            qc_ = QuantumCircuit(op.num_qubits)
            qc_.append(op, range(op.num_qubits))
            qc_t = transpile(qc_, basis_gates=list(basis))
            seen_gates[op_info] = (Counter(qc_t.count_ops()), qc_t.depth())
        c, d = seen_gates[op_info]
        cost += c
        depth += d
    return cost, depth


def select_tests(key, X, y, _lambda, m, overlap_min, n_cand=200):
    """m sparse test vectors (dimension D) with |<x^|w^>| >= overlap_min, rescaled to ||x'||_inf = 1.

    Args:
        key (jax.Array): PRNG key.
        X (np.ndarray): (N, D) matrix.
        y (np.ndarray): (N,) labels.
        _lambda (float): regularization strength.
        m (int): number of tests.
        overlap_min (float): minimum |<x^|w^>|.
        n_cand (int): number of random candidates.

    Returns:
        tuple: (tests (m, D), ClassicalReferenceResult on the tests).

    Raises:
        AssertionError: if fewer than m candidates have the required overlap.
    """
    N, D = X.shape
    keys = random.split(key, n_cand)
    cand = np.array([np.asarray(ste.sparse_vector_gen(k, D))[0] for k in keys])   # (n_cand, D)
    r = ref(X, y, _lambda, cand)
    idx = np.where(np.abs(np.asarray(r.inner_product)) >= overlap_min)[0][:m]
    assert len(idx) == m, f"solo {len(idx)} candidati con overlap >= {overlap_min}"
    tests = cand[idx]
    tests = tests / np.max(np.abs(tests), axis=1, keepdims=True)    # ||x'||_inf = 1 (hypothesis of F.13)
    return tests, ref(X, y, _lambda, tests)


def predict(X, y, _lambda, tests, cfg, sketch=None, key=None, count_gates=True):
    """Signs of <x'_j, w> with V = fixed-point amplification of W and (optionally) the shadow.

    Args:
        X (np.ndarray): (N, D) matrix.
        y (np.ndarray): (N,) labels.
        _lambda (float): regularization strength.
        tests (np.ndarray): (m, D) test vectors.
        cfg (dict): parameters of chose_param_cfg (eps_solver, deg_sign, threshold, shots, K, rng);
            shots=None uses the exact overlaps instead of the shadow.
        sketch (dict | None): sketched oracles for build_W (a fresh W for every query of the amplification).
        key (jax.Array | None): PRNG key of the sketched oracles.
        count_gates (bool): also count gates and depth of V.

    Returns:
        tuple: (signs (m,), report dict with sizes, kappa, degrees, p, target, o_hat, queries,
            qubits, estimates, gates, depth, shots).
    """
    m = tests.shape[0]
    N, D = X.shape
    x_hat = tests / np.linalg.norm(tests, axis=1, keepdims=True)

    with suppress():
        if sketch is None:
            W, info = build_W(X, y, _lambda, cfg['eps_solver'])
        else:
            built = [build_W(X, y, _lambda, cfg['eps_solver'], sketch=sketch, key=k) for k in random.split(key, cfg['deg_sign'])]
            W, info = [b[0] for b in built], built[0][1]
    qc_V, _ = amplify_fixed_point(W, info, cfg['deg_sign'], cfg['threshold'], 0.99)
    queries_dict = cont_oper(qc_V)
    queries = queries_dict['U'] + queries_dict['U_y']
    v = Statevector(qc_V).data
    n_V = qc_V.num_qubits

    # exact overlaps Re<x^|V|0>> (= Re P * <x^|w^>)
    Xp, Xm = embed_tests(x_hat, N, D, n_V)
    target = np.real((np.sqrt(2) * Xp[:, 1::2]).conj() @ v)
    if cfg["shots"] is None:
        o_hat = target
        T = None
        est = None
    else:
        # shadow on |w~> = (|0>|0> + |1>V|0>)/sqrt(2), median of K means of N_g shots
        w_tilde = np.zeros(2 ** (n_V + 1), complex)
        w_tilde[0] = 1 / np.sqrt(2)
        w_tilde[1::2] = v / np.sqrt(2)
        K = cfg["K"]
        N_g = int(cfg["shots"]) // cfg["K"]
        T = K * N_g
        data = shadow_measure(None, cfg["rng"], K * N_g, stato=Statevector(w_tilde))
        est = np.asarray(classical_result(Xp, Xm, n_V + 1, data))
        o_hat = np.median(est.reshape(K, N_g, m).mean(axis=1), axis=0)

    if count_gates:
        gate_cost, depth = gate_counter(qc_V, BASIS, True)   # cost of V, not of the final algorithm
        gates = dict(per_shot=dict(gate_cost), depth_per_shot=depth)
    else:
        gates, depth = None, None

    del qc_V
    gc.collect()

    report = dict(
        N=N,
        D=D,
        n_V=n_V,
        n_sys=info["n_sys"],
        kappa=float(info["kappa_reg"]),
        deg_inv=info["degree"],
        deg_sign=cfg["deg_sign"],
        threshold=cfg['threshold'],
        p=float(np.sum(np.abs(v[:2**info["n_sys"]])**2)),
        target=target,
        o_hat=o_hat,
        queries=queries,
        queries_total=None if T is None else queries * T,
        queries_per_prep=queries_dict,
        qubits=n_V + 1,
        est=est,
        gates=gates,
        depth=depth,
        shots=T
    )
    return np.sign(o_hat), report


def chose_param_cfg(X, y, _lambda, tests, s, exp, delta=float(0.1), c=65, eps_solver=0.05, re_min=0.98, seed=42, sketch=None, key=None, degree_int=61):
    """Parameters of predict: degree and threshold of the amplification, number of shots.

    Args:
        X (np.ndarray): (N, D) matrix.
        y (np.ndarray): (N,) labels.
        _lambda (float): regularization strength.
        tests (np.ndarray): (m, D) test vectors.
        s (int): sparsity (not used, kept for compatibility).
        exp (bool): if True compute the shots K * ceil(c / eps^2) with eps = min overlap / 3,
            otherwise shots = None (exact overlaps).
        delta (float): failure probability (K = ceil(2 ln(2m/delta))).
        c (float): constant of the number of shots per group.
        eps_solver (float): precision of 1/x.
        re_min (float): minimum Re P(sqrt p) of the sign polynomial.
        seed (int): seed of the shadow generator.
        sketch (dict | None): sketched oracles for build_W.
        key (jax.Array | None): PRNG key of the sketched oracles.
        degree_int (int): first degree tried (step 10, up to 201).

    Returns:
        dict: eps_solver, deg_sign, threshold, shots, sqrt_p_pre, K, rng, eps.

    Raises:
        ValueError: if no degree <= 201 reaches re_min.
    """
    with suppress():
        W, info = build_W(X, y, _lambda, eps_solver, sketch=sketch, key=key)
    m = tests.shape[0]
    K = int(np.ceil(2*np.log(2*m/delta)))
    # amplitude of W before the amplification -> threshold 0.7 sqrt(p)
    sqrt_p = float(np.linalg.norm(Statevector(W).data[:2**info["n_sys"]]))
    thr = round(0.7 * sqrt_p, 3)
    deg_sign = None
    for d in range(degree_int, 202, 10):
        with suppress():
            angles, _ = save_angles(get_qsvt_angles_sign, d, thr, 0.99)
        if sign_poly_value(np.asarray(angles, dtype=float), sqrt_p).real >= re_min:
            deg_sign = d
            break
    if deg_sign is None:
        raise ValueError(f"nessun grado <= 201 basta per sqrt(p) = {sqrt_p:.3f}")

    if exp:
        class_ref = ref(X, y, _lambda, tests)
        eps = float(np.min(np.abs(np.asarray(class_ref.inner_product)))) / 3
        shots = K * int(np.ceil(c/(eps**2)))
    else:
        shots = None
        eps = None

    return dict(
        eps_solver=eps_solver,
        deg_sign=deg_sign,
        threshold=thr,
        shots=shots,
        sqrt_p_pre=sqrt_p,
        K=K,
        rng=np.random.default_rng(seed),
        eps=eps,
    )


class ExecutionTimer:
    """Keeps track of the mean and total times of the phases of a test."""

    def __init__(self, name):
        """
        Args:
            name (str): name printed in the report.
        """
        self.name = name
        self.start = time.perf_counter()
        self.stats = {}

    def add(self, phase, elapsed):
        """Records one duration.

        Args:
            phase (str): name of the phase.
            elapsed (float): duration in seconds.

        Returns:
            None
        """
        self.stats.setdefault(phase, []).append(elapsed)

    def report(self):
        """Prints total time and per-phase statistics.

        Returns:
            None
        """
        total = time.perf_counter() - self.start
        print(f"\n{'=' * 65}")
        print(f"REPORT: {self.name}")
        print(f"{'=' * 65}")
        print(f"Tempo totale: {total:.2f} s ({total / 60:.2f} min)\n")
        for phase, values in self.stats.items():
            print(
                f"{phase:30s} | "
                f"n={len(values):4d} | "
                f"totale={sum(values):9.2f} s | "
                f"media={np.mean(values):8.3f} s | "
                f"min={min(values):8.3f} s | "
                f"max={max(values):8.3f} s"
            )
        print(f"{'=' * 65}\n")


def resource_table(key, s=2, out="risorse.json"):
    """Resources of predict on (4,6), (8,8), (16,16): qubits, degrees, shots, queries, gates, depth.

    Args:
        key (jax.Array): PRNG key.
        s (int): sparsity of the instances.
        out (str): output JSON file.

    Returns:
        None
    """
    rows = []
    NS = [4, 8, 16]
    DS = [6, 8, 16]
    s = 2

    for N, D in zip(NS, DS):
        t0 = time.time()
        key, subkey1, subkey2 = random.split(key, 3)
        X, y, _lambda = ste.small_instance_generator(subkey1, N, D, s, 3)
        tests, _ = select_tests(subkey2, X, y, _lambda, 5, 0.45)
        cfg = chose_param_cfg(X, y, _lambda, tests, s, exp=True)
        T = cfg["shots"]
        # shots at the fixed overlap 0.45 (comparable across sizes)
        eps_fix = 0.45 / 3
        T_fix = cfg["K"] * int(np.ceil(65 / eps_fix**2))
        _, rep = predict(X, y, _lambda, tests, {**cfg, "shots": None})

        g = rep["gates"]
        rows.append(dict(
            N=N, D=D, qubits=rep["qubits"], n_sys=rep["n_sys"], kappa=rep["kappa"],
            deg_inv=rep["deg_inv"], deg_sign=rep["deg_sign"],
            sqrt_p_pre=cfg["sqrt_p_pre"], p_post=rep["p"],
            eps_test=cfg["eps"], T_test=int(T), T_fix=T_fix,
            queries_total=int(rep["queries"] * T_fix),
            gates_total=int(sum(g["per_shot"].values()) * T_fix),
            U_per_prep=int(rep["queries_per_prep"]["U"]),
            Uy_per_prep=int(rep["queries_per_prep"]["U_y"]),
            T=int(T),
            cx_per_shot=int(g["per_shot"].get("cx", 0)),
            gates_per_shot=int(sum(g["per_shot"].values())),
            depth_per_shot=int(g["depth_per_shot"]),
            tempo=str(timedelta(seconds=round(time.time() - t0))),
        ))
        print(rows[-1])
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(rows)
    print("\n\n Tabella delle risorse stampata e memorizzata")


"""
Tests
"""

def _test_predict(key):
    """predict with exact overlaps reproduces the classical signs (10 instances per size).

    Args:
        key (jax.Array): PRNG key.

    Returns:
        None
    """
    NS = [4, 8, 16]
    DS = [6, 8, 16]
    s = 2
    n_iter = 10

    timer = ExecutionTimer("_test_predict")
    total = len(NS) * n_iter
    pbar = tqdm(total=total, desc="Test predict", unit="istanza")

    print("\nAvvio _test_predict()")
    print(f"Configurazioni: {list(zip(NS, DS))}")
    print(f"Istanze totali: {total}\n")

    try:
        for N, D in zip(NS, DS):
            print(f"\n--- Configurazione N={N}, D={D} ---")
            config_start = time.perf_counter()

            for i in range(n_iter):
                instance_start = time.perf_counter()

                # instance
                t0 = time.perf_counter()
                key, subkey1, subkey2 = random.split(key, 3)
                X, y, _lambda = ste.small_instance_generator(subkey1, N, D, s, 3)
                timer.add("small_instance_generator", time.perf_counter() - t0)

                # tests
                t0 = time.perf_counter()
                tests, _ = select_tests(key, X, y, _lambda, 5, 0.45)
                timer.add("select_tests", time.perf_counter() - t0)

                # parameters
                t0 = time.perf_counter()
                cfg_dict = chose_param_cfg(X, y, _lambda, tests, s, exp=False)
                timer.add("chose_param_cfg", time.perf_counter() - t0)

                # prediction (block_until_ready: JAX may run asynchronously)
                t0 = time.perf_counter()
                sign, report = predict(X, y, _lambda, tests, cfg_dict)
                if hasattr(sign, "block_until_ready"):
                    sign.block_until_ready()
                timer.add("predict", time.perf_counter() - t0)

                # classical reference
                t0 = time.perf_counter()
                class_ref = ref(X, y, _lambda, tests)
                class_ref.y_pred.block_until_ready()
                timer.add("ref", time.perf_counter() - t0)

                # checks
                t0 = time.perf_counter()
                assert jnp.all(sign == class_ref.y_pred), "Assert error: la predizione del segno non è corretta!"
                assert min(abs(report["target"])) > 0, "Assert error: min(target) < 0!"
                timer.add("assertions", time.perf_counter() - t0)

                elapsed = time.perf_counter() - instance_start
                pbar.update(1)
                pbar.set_postfix({
                    "N": N,
                    "D": D,
                    "istanza": f"{i + 1}/{n_iter}",
                    "ultimo": f"{elapsed:.1f}s",
                    "media": f"{np.mean(timer.stats.get('predict', [0])):.1f}s predict",
                })

            config_elapsed = time.perf_counter() - config_start
            print(f"Configurazione ({N}, {D}) completata in {config_elapsed:.2f} s ({config_elapsed / 60:.2f} min)")

        print("\nPrimo test di _test_predict() concluso con successo!")

    finally:
        pbar.close()
        timer.report()


def _test_predict_sampled(key, sizes=((4, 6), (8, 8)), n_inst=5, R=200, delta=0.1):
    """predict with the shadow: bootstrap success rate >= 1 - delta and error quantile / eps.

    Args:
        key (jax.Array): PRNG key.
        sizes (tuple): (N, D) sizes.
        n_inst (int): instances per size.
        R (int): bootstrap resamplings per instance.
        delta (float): failure probability.

    Returns:
        None
    """
    timer = ExecutionTimer("_test_predict_sampled")
    total = len(sizes) * n_inst
    pbar = tqdm(total=total, desc="Test sampled", unit="istanza")

    print("\nAvvio _test_predict_sampled()")
    print(f"Configurazioni: {sizes}")
    print(f"Istanze per configurazione: {n_inst}")
    print(f"Simulazioni Monte Carlo per istanza: {R}")
    print(f"Simulazioni totali previste: {total * R}\n")

    try:
        for N, D in sizes:
            rates, errs = [], []
            print(f"\n--- Configurazione N={N}, D={D} ---")
            config_start = time.perf_counter()

            for i in range(n_inst):
                instance_start = time.perf_counter()

                # 1. instance
                t0 = time.perf_counter()
                key, subkey1, subkey2 = random.split(key, 3)
                X, y, _lambda = ste.small_instance_generator(subkey1, N, D, kappa_target=3)
                timer.add("small_instance_generator", time.perf_counter() - t0)

                # 2. tests and reference
                t0 = time.perf_counter()
                tests, ref_result = select_tests(subkey2, X, y, _lambda, 5, 0.45)
                timer.add("select_tests", time.perf_counter() - t0)

                # 3. parameters
                t0 = time.perf_counter()
                cfg = chose_param_cfg(X, y, _lambda, tests, 2, exp=True, delta=delta, seed=i)
                timer.add("chose_param_cfg", time.perf_counter() - t0)

                # 4. prediction with the shadow
                t0 = time.perf_counter()
                _, report = predict(X, y, _lambda, tests, cfg)
                est = report["est"]
                if hasattr(est, "block_until_ready"):
                    est.block_until_ready()
                timer.add("predict", time.perf_counter() - t0)

                target = report["target"]
                T = report["shots"]
                K = cfg["K"]
                y_ref = np.asarray(ref_result.y_pred)

                # 5. bootstrap of the T single-shot estimates
                t0 = time.perf_counter()
                rng = np.random.default_rng(1000 + i)
                ok, worst = 0, np.empty(R)
                mc_bar = tqdm(range(R), desc=f"Monte Carlo ({N},{D}) istanza {i + 1}/{n_inst}", leave=False, unit="sim")
                for k in mc_bar:
                    idx = rng.integers(0, T, size=T)
                    o_hat = np.median(est[idx].reshape(K, T // K, -1).mean(axis=1), axis=0)
                    ok += np.all(np.sign(o_hat) == y_ref)
                    worst[k] = np.max(np.abs(o_hat - target)) / cfg["eps"]
                    if (k + 1) % max(1, R // 20) == 0:
                        mc_bar.set_postfix({
                            "successo": f"{ok / (k + 1):.1%}",
                            "tempo": f"{time.perf_counter() - t0:.1f}s",
                        })
                mc_bar.close()
                timer.add("Monte Carlo", time.perf_counter() - t0)

                # 6. statistics of the instance
                rate = ok / R
                err = np.quantile(worst, 1 - delta)
                rates.append(rate)
                errs.append(err)

                elapsed = time.perf_counter() - instance_start
                pbar.update(1)
                pbar.set_postfix({
                    "N": N,
                    "D": D,
                    "istanza": f"{i + 1}/{n_inst}",
                    "successo": f"{rate:.1%}",
                    "tempo": f"{elapsed:.1f}s",
                })
                print(f"  Istanza {i + 1}/{n_inst}: successo={rate:.3f}, quantile={err:.2f}, tempo={elapsed:.2f}s")

            config_elapsed = time.perf_counter() - config_start
            print(f"\nB ({N},{D}): successo min = {min(rates):.3f}, quantile {1 - delta:.0%} di max|o-target|/eps = {max(errs):.2f}")
            print(f"Tempo configurazione ({N},{D}): {config_elapsed:.2f}s ({config_elapsed / 60:.2f} min)")
            assert min(rates) >= 1 - delta, f"B ({N},{D}): successo < 1 - delta"

        print("\n_test_predict_sampled() concluso con successo!")

    finally:
        pbar.close()
        timer.report()


if __name__ == "__main__":
    print("---"*10)
    print("predict.py")
    print("---"*10)
    _test_predict(key)
    _test_predict_sampled(key)
    #resource_table(key)
    #stampa_risultati("risorse.json", "terminal")
    #stampa_risultati("risorse.json", "txt")
    #stampa_risultati("risorse.json", "plot")
