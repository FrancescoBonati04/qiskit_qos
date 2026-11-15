# Quantum Oracle Sketching — Theorem F.13 in Qiskit

Qiskit implementation of **Theorem F.13** of *Quantum Oracle Sketching* (arXiv:2604.07639):
binary classification through ridge regression, where every oracle is built from classical
samples (oracle sketching) instead of being given as a black box.

The pipeline is:

1. **Oracles from samples**: flat state sketch `U_y` of the labels, and block encoding `U_A` of
   `A_aug = [X; sqrt(lambda) I]` built from sketched element, row-index and column-index oracles.
2. **Solver `W`**: rectangular QSVT of `1/x` on `U_A`, applied to `|y>` → `|w>` with
   `w = (XᵀX + λI)⁻¹ Xᵀ y`.
3. **Amplification `V`**: fixed-point amplitude amplification (sign QSVT on `W`).
4. **Reading**: interferometric classical shadow (random Cliffords + median of means), which
   estimates `Re<x'_j|w>` for all test vectors at once → predicted signs.
5. **Resources**: number of samples, qubits, gates, depth, physical and fault-tolerant times,
   compared with the paper estimates (F58, F59, F69) and with the classical cost.

Thesis project (Bachelor's in Physics, Università di Milano-Bicocca).

---

## Requirements

- Python ≥ 3.10
- The packages in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```
- The repository of the paper, in a folder named **`Exponential Quantum Advantage`**.
  It provides `utils`, `qsvt`, `qos_sampling`, `data_generation`, ... used throughout the code.

### Where the repository must be

`utilities.Connect()` adds the repository to `sys.path`. It looks for it:

1. in the environment variable `QOS_REPO`, if set;
2. otherwise in every parent folder of this directory (a sibling folder of `codice/` works).

```bash
# Linux / macOS
export QOS_REPO="/path/to/Exponential Quantum Advantage"
# Windows (PowerShell)
$env:QOS_REPO = "C:\path\to\Exponential Quantum Advantage"
```

### Cache of the QSVT angles

Angles computed with pyqsp are cached in `angoli salvati/` (created next to the files) by
`utilities.save_angles`. Copying an existing `angoli salvati/` folder here avoids recomputing them
(the large degrees can take a long time).

---

## Files

Listed in dependency order (each module only imports the ones above it).

| File | Content |
|---|---|
| `utilities.py` | `Connect()` (repository + PRNG key), endianness converters, angle cache `save_angles`, query counter `cont_oper`, table printer `stampa_risultati` |
| `reference.py` | Exact classical ridge solution (`classical_reference`), F58 / F59 / F69, classical time and memory (`classical_cost`) |
| `small_toy_exp.py` | Small random instances, sparse test vectors, instances with small sketched κ (`istanza_kappa_piccolo`) |
| `block_encoding.py` | Exact (dense) block encoding via Hermitian + Halmos dilation, exact state preparation of `y` |
| `qsvt_qiskit.py` | QSVT circuits: projector phases, QSVT sequence, real part via LCU, fast `rzz` version with a list of block encodings |
| `ridge_solver.py` | `build_W` (QSVT of 1/x), standard and fixed-point amplification, choice of the sign degree, F69 check |
| `shadow.py` | Interferometric classical shadow: preparation of `|w~>`, Clifford measurements, estimators |
| `predict.py` | Prediction with ideal oracles (W + amplification + shadow), parameter choice, test selection, resource table |
| `simulate.py` | Pure-NumPy statevector simulator (piece-by-piece evolution, diagonal gates without dense matrices) |
| `sketch_oracles.py` | Sketched oracles (`U_y`, element, row index, column index, cumulative counter), `U_A`, gate/depth costs, sweep over M |
| `data_set_prep.py` | Synthetic dataset satisfying the hypotheses of F.13 |
| `binary_classification.py` | Full algorithm with sketched oracles: rectangular QSVT, `W`, `V`, `binary_classification`, sweep over M, resource estimates and final table |

### Why a custom simulator

`qiskit-aer` crashes at native level on Windows on these circuits (exit code `-1073741819`).
`simulate.py` enters the boxes recursively and applies the large diagonal gates (`mcphase`,
`diagonal`) element-wise, so it never builds matrices larger than `2^8`.

---

## Running

Every module has a `__main__` block. The tests (`_test_*`) are listed there; uncomment the
ones you want.

```bash
python sketch_oracles.py          # oracles vs repo, sweep over M and its fit
python ridge_solver.py            # solver, amplification, F69 check
python shadow.py                  # statistics of the shadow
python predict.py                 # prediction with ideal oracles
python binary_classification.py   # sketched algorithm, sweep over M, final table
```

Main functions of `binary_classification.py`:

- `singol_instance` / `sweep_M`: simulation of the sketched `W` for increasing M (infidelity, signs,
  samples) → `sweep_m_simulazioni.json`, plotted by `grafico`.
- `fit_gradi`: degrees of the 1/x and sign polynomials measured with pyqsp and fitted.
- `stima_risorse` / `tabella_finale`: bottom-up resource estimate on larger synthetic datasets
  (no simulation) → `tabella_finale_mod.json` / `.txt`.
- `binary_classification`: the full algorithm shot by shot. It is very expensive and meant only
  as a demonstration (few shots, tiny instances).

Most functions save JSON files and PNG figures in the working directory.

---

## Conventions

- **Endianness**: Qiskit little-endian (qubit 0 = least significant bit). The repository uses
  MSB-first; the converters are in `utilities.py`.
- **Normalization of `U_A`**: the rank registers are projected, not uncomputed, so
  `U_A` encodes `A / (s_r s_c)` (α = s_r s_c, while the repository uses √(s_r s_c)), and
  κ_s = α / σ_min(A_aug).
- **Rectangular layout**: in `W`, `w_j` is at index `2^n_row · j` (register J, with I and all
  ancillas in |0>).
- **Hypotheses of F.13**: N a power of 2, ‖X‖ ≤ 1, |X_ij| ≤ 1, ‖x'‖∞ ≤ 1, tests at most
  s-sparse, 0 < λ ≤ 1 (needed by the element oracle).
- **Randomness**: all JAX keys descend from `PRNGKey(0)` returned by `Connect()`.

## Known limitations

- Padding indices of the index oracles (`2^b − 1`) are assumed to point to zero entries.
- `grado_inversa` uses the degree of the truncated Childs–Kothari–Somma polynomial for the
  estimates, while the simulations use the full pyqsp polynomial.
- `binary_classification` and the sweeps are limited to very small instances by the cost of
  statevector simulation.
