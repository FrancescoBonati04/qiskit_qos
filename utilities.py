"""
Shared helpers: connection to the QOS repository, endianness converters, cache of the
QSVT angles, operation counter and printing of JSON result tables.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jax import random
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator


REPO_NAME = "Exponential Quantum Advantage"


def _find_repo():
    """Locates the folder of the QOS repository.

    The environment variable QOS_REPO has priority; otherwise the folder named REPO_NAME is
    searched in every parent directory of this file (so it works both from codice/ and from
    codice/codice_github/).

    Returns:
        Path: path of the repository folder.

    Raises:
        FileNotFoundError: if the repository cannot be found.
    """
    env = os.environ.get("QOS_REPO")
    if env:
        return Path(env).resolve()
    for parent in Path(__file__).resolve().parents:
        candidate = parent / REPO_NAME
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"cartella '{REPO_NAME}' non trovata: imposta la variabile d'ambiente QOS_REPO")


def Connect():
    """Adds the QOS repository to sys.path and pre-loads its modules.

    After this call the repository modules (utils, qsvt, qos_sampling, data_generation, ...)
    can be imported normally by the caller.

    Returns:
        jax.Array: PRNG key built from the fixed seed 0.
    """
    root = _find_repo()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # pre-load the repository modules (also applies their global configuration, e.g. JAX)
    import qos
    import utils
    import qsvt
    from data_generation import matrix_data, vector_data
    import primitives
    import qos_sampling

    SEED = 0
    return random.PRNGKey(SEED)


def statevector_converter_little_edian_to_MSB(statevector: Statevector):
    """Reorders a statevector from Qiskit little-endian to MSB-first ordering.

    Args:
        statevector (Statevector): state in Qiskit ordering (qubit 0 = least significant bit).

    Returns:
        np.ndarray: amplitudes in MSB-first ordering (qubit 0 = most significant bit).
    """
    n_qubits = statevector.num_qubits
    tensore = statevector.data.reshape([2] * n_qubits)
    # reversing the tensor axes swaps the role of the qubits
    tensore_inverso = np.transpose(tensore, axes=list(reversed(range(n_qubits))))
    return tensore_inverso.flatten()


def operator_converter_little_edian_to_MSB(operator: Operator) -> np.ndarray:
    """Reorders an operator from Qiskit little-endian to MSB-first ordering.

    Args:
        operator (Operator): operator in Qiskit ordering.

    Returns:
        np.ndarray: (2^n, 2^n) matrix in MSB-first ordering.
    """
    n_qubits = operator.num_qubits
    tensore = operator.data.reshape([2] * (2 * n_qubits))
    assi = list(range(2 * n_qubits))
    # reverse separately the row axes and the column axes
    new_axes = assi[:n_qubits][::-1] + assi[n_qubits:][::-1]
    tensore_inverso = np.transpose(tensore, axes=new_axes)
    return tensore_inverso.reshape((2**n_qubits, 2**n_qubits))


def format_statevector_binary(vector: np.ndarray, num_qubits: int, mode_label: str):
    """Prints the non-zero amplitudes of a statevector with binary indices.

    Args:
        vector (np.ndarray): amplitudes.
        num_qubits (int): number of qubits (length of the binary labels).
        mode_label (str): label printed in the header.

    Returns:
        None
    """
    print(f"\n--- Statevector ({mode_label}) ---")
    for idx, amp in enumerate(vector):
        binary_idx = format(idx, f'0{num_qubits}b')
        if abs(amp) > 1e-10:
            print(f"Indice binario [{binary_idx}] (dec {idx}): ampiezza = {amp}")


def format_operator_binary(mat: np.ndarray, n_qubits: int):
    """Prints the non-zero entries of a matrix with binary row/column indices.

    Args:
        mat (np.ndarray): (2^n, 2^n) matrix.
        n_qubits (int): number of qubits.

    Returns:
        None
    """
    dim = 2**n_qubits
    for r in range(dim):
        for c in range(dim):
            val = mat[r, c]
            if abs(val) > 1e-10:
                bin_r = format(r, f'0{n_qubits}b')
                bin_c = format(c, f'0{n_qubits}b')
                print(f"M[{bin_r}, {bin_c}] (dec [{r}, {c}]) = {val}")


def create_directory():
    """Returns the cache folder of the QSVT angles, creating it if needed.

    Returns:
        Path: folder "angoli salvati" next to this file.
    """
    directory = Path(__file__).resolve().parent / "angoli salvati"
    if not directory.exists():
        directory.mkdir()
    return directory


def save_angles(func, *args, **kwargs):
    """Computes QSVT angles with func, caching them in "angoli salvati/<func>_<args>_<kwargs>.npz".

    If the file already exists the angles are read from disk instead of being recomputed.

    Args:
        func (callable): angle generator of the repository (get_qsvt_angles,
            get_qsvt_angles_inverse, get_qsvt_angles_sign).
        *args: positional arguments of func (they also build the file name).
        **kwargs: keyword arguments of func (they also build the file name).

    Returns:
        np.ndarray | tuple[np.ndarray, float]: the angles, or (angles, scale) when func
            returns a scale too.
    """
    path = create_directory()
    # file name: <function>_<args>_<kwargs>
    args_part = "_".join(map(str, args))
    kwargs_part = "_".join(f"{k}{v}" for k, v in kwargs.items())
    parti = [p for p in [func.__name__, args_part, kwargs_part] if p]
    filename = f"{'_'.join(parti)}.npz"

    if (path / filename).exists():
        with np.load(path / filename) as data:
            angoli = data["angoli"]
            if "scala" in data:
                scala = data['scala'].item()
                print("Dati trovati con successo")
                return angoli, scala
            print("Dati trovati con successo")
            return angoli

    # not cached: compute and save
    ris = func(*args, **kwargs)
    if isinstance(ris, tuple):
        angles, scale = ris
        np.savez(path / filename, angoli=angles, scala=scale)
        print(f"Risultati salvati in {filename}")
        return angles, scale
    np.savez(path / filename, angoli=ris)
    print(f"Risultati salvati in {filename}")
    return ris


def cont_oper(qc: QuantumCircuit):
    """Counts the oracle queries in a circuit.

    Args:
        qc (QuantumCircuit): circuit whose top-level operations are inspected.

    Returns:
        dict: {"U": number of "unitary" gates (block encodings),
               "U_y": number of state preparations of y (and their inverses)}.
    """
    oper = qc.count_ops()
    U = oper.get("unitary", 0)
    U_y = sum(oper.get(k, 0) for k in ("state_preparation", "state_preparation_dg", "U_y", "U_y_dg"))
    return dict(U=U, U_y=U_y)


def stampa_risultati(json_path, output="terminal", output_path=None):
    """Reads a JSON file of results and shows it as a table.

    Args:
        json_path (str): path of the JSON file (a list of dicts, or a dict with one list).
        output (str): "terminal" prints it, "txt" saves it as text, "plot" draws it with matplotlib.
        output_path (str | None): output file for "txt" (default "risultati.txt") and "plot".

    Returns:
        pd.DataFrame: the table.

    Raises:
        ValueError: unsupported JSON format or unknown output.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        df = pd.json_normalize(data)
    elif isinstance(data, dict):
        liste = [v for v in data.values() if isinstance(v, list)]
        df = pd.json_normalize(liste[0]) if len(liste) == 1 else pd.json_normalize(data)
    else:
        raise ValueError("Formato JSON non supportato")

    if output == "terminal":
        print(df.to_string(index=False))
    elif output == "txt":
        path = output_path or "risultati.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(df.to_string(index=False))
        print(f"Tabella salvata in: {path}")
    elif output == "plot":
        n_rows, n_cols = df.shape
        fig_width = max(8, min(20, n_cols * 1.5))
        fig_height = max(3, min(15, (n_rows + 1) * 0.4))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")
        tabella = ax.table(
            cellText=df.fillna("").astype(str).values,
            colLabels=df.columns,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1]
        )
        tabella.auto_set_font_size(False)
        tabella.set_fontsize(8)
        tabella.auto_set_column_width(col=list(range(n_cols)))
        tabella.scale(1, 1.2)
        # bold, coloured header
        for (row, col), cell in tabella.get_celld().items():
            cell.set_text_props(wrap=True)
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#D9EAF7")
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.show()
    else:
        raise ValueError("output deve essere 'terminal', 'txt' o 'plot'")
    return df


"""
Tests
"""

def _test_statevector_convert():
    """Prints |001> in Qiskit ordering and in MSB-first ordering.

    Returns:
        None
    """
    num_qubits = 3
    sv_qiskit = Statevector.from_label('001')
    format_statevector_binary(sv_qiskit.data, num_qubits, "Qiskit LSB-first: q2 q1 q0")
    sv_msb = statevector_converter_little_edian_to_MSB(sv_qiskit)
    format_statevector_binary(sv_msb, num_qubits, "Repo MSB-first: q0 q1 q2")


def _test_operator_convert():
    """Prints X on qubit 0 (4 qubits) in Qiskit ordering and in MSB-first ordering.

    Returns:
        None
    """
    n_qubits = 4
    qc = QuantumCircuit(n_qubits)
    qc.x(0)
    op_qiskit = Operator(qc)
    format_operator_binary(op_qiskit.data, n_qubits)
    print("")
    op_mbs = operator_converter_little_edian_to_MSB(op_qiskit)
    format_operator_binary(op_mbs, n_qubits)


if __name__ == "__main__":
    print('---'*10)
    print("utlities.py")
    print('---'*10)
    #_test_statevector_convert()
    #_test_operator_convert()
