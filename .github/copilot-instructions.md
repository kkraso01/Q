# Quantum Data Structures (QDS) Research Project

## Project Overview

This is a research project exploring quantum data structures, focusing on achieving preliminary results in 4–6 weeks. The primary goal is to develop quantum alternatives to classical probabilistic data structures (Bloom filters, SimHash, suffix arrays) with demonstrable trade-offs in accuracy, memory, and query performance.




## Phase 5–End: Engineering, Scaling, and Finalization ✓

### Completed Tasks:

1. **✓ State caching in QAM**
   - Implemented circuit and statevector caching in `sim/qam.py` for 20–40% runtime reduction.

2. **✓ Batch query experiments**
   - Implemented `run_batch_query_experiment()` in `experiments/sweeps.py`.
   - Supports batch sizes [1, 16, 64] with amortized cost analysis.
   - Plotting function `plot_batch_query_error_vs_amortized_cost()` added.

3. **✓ Heatmap: shots × noise sweep**
   - Implemented `run_heatmap_sweep()` in `experiments/sweeps.py`.
   - 2D heatmap plotting with `plot_heatmap_shots_noise()`.
   - Visualizes FP/FN rates across shots and noise parameter space.

4. **✓ QAM topology variants**
   - Added topology parameter ('none', 'linear', 'ring', 'all-to-all') to QAM.
   - Implemented `_apply_entanglement_layer()` for configurable entanglement.
   - Topology sweep with `run_topology_sweep()` and plotting.
   - Circuit depth tracking for complexity analysis.

5. **✓ Q-SubSketch evaluation**
   - Implemented `run_q_subsketch_evaluation()` for real/synthetic text corpus.
   - AUC computation vs substring length (L = 4, 8, 16, 32).
   - Plotting with `plot_q_subsketch_auc()`.

6. **✓ Comprehensive figure generation**
   - Created `experiments/generate_all_figures.py` to run all experiments.
   - Generates 8+ reproducible figures:
     1. accuracy_vs_memory.png
     2. accuracy_vs_shots.png
     3. accuracy_vs_noise.png
     4. accuracy_vs_load_factor.png
     5. batch_query_error_vs_amortized_cost.png
     6. heatmap_shots_noise.png
     7. topology_comparison.png
     8. q_subsketch_auc.png

7. **✓ QAM deletion analysis**
   - Empirically validated and documented fundamental limitation.
   - Limitation documented in `theory/qam_deletion_limitations.md`.
   - Integrated into paper with honest reporting.

8. **✓ Classical baselines**
   - Implemented Cuckoo, XOR, and Vacuum filters in `sim/classical_filters.py`.
   - All baselines integrated into parameter sweeps for comparison.

### Remaining Tasks:
- Run `python experiments/generate_all_figures.py` to generate all plots.
- Verify all figures are saved and reproducible.
- Update paper with all experimental results and figures.
- Final polish and submission preparation.



## Repository Structure

```
qds/
  theory/          # Proofs, bounds, theoretical analysis
  sim/             # Core quantum algorithms
    qam.py         # Quantum Approximate Membership
    q_subsketch.py # Quantum Suffix Sketch
    q_simhash.py   # Quantum Similarity Hash
  experiments/     # Experimental harness
    sweeps.py      # Parameter grid search
    plotting.py    # Visualization utilities
  notebooks/       # Jupyter notebooks for prototyping
    qam.ipynb      # Main QAM experiments
  results/         # CSV data + plots (PNG/SVG)
  paper/           # Draft papers/reports
```

## Tech Stack & Dependencies

- **Python 3.11** required
- **Quantum framework**: **Qiskit** (easier to start, better documentation, active community)
  - `pip install qiskit qiskit-aer matplotlib`
  - Use `qiskit.primitives.Sampler` for shot-based measurements
  - Use `qiskit.quantum_info.Statevector` for testing without noise
- **Core libs**: NumPy, SciPy, matplotlib
- **Testing**: pytest with reproducible seeds (`np.random.seed()`, `random.seed()`)
- **Config**: Simple argparse or dataclasses (avoid heavy frameworks for quick prototyping)

## Computational Model (Critical Constraints)

When implementing quantum circuits:

- **Query/Update model**: Unit-cost gate model with unitary circuits + measurements
- **No-cloning constraint**: Respect quantum mechanical limitations
- **Error model**: Depolarizing or Pauli noise with per-2Q-gate error ε; measurement error pᵣ
- **Cost metrics**:
  - Time = gate depth
  - Space = logical qubits
  - Accuracy = (false-positive α, false-negative β)
  - Shot budget at query time

## Algorithm Pattern: Phase-Based Membership

All three QDS approaches follow a common pattern:

1. **Hash to indices**: Use k hash functions to map items to m qubits
2. **Phase encoding**: Apply Rz(θ) rotations or phase flips at hashed positions
3. **Interference test**: Query by re-applying same pattern; measure overlap with reference state
4. **Threshold decision**: Post-process expectation values against threshold τ

Example (QAM insert with Qiskit):
```python
from qiskit import QuantumCircuit
import numpy as np

def qam_insert(circuit, x, hash_functions, m, theta):
    """Insert item x into QAM circuit."""
    for i in range(len(hash_functions)):
        qubit_idx = hash_functions[i](x) % m
        circuit.rz(theta, qubit_idx)
    return circuit

# Usage
m = 32  # number of qubits
k = 3   # number of hash functions
theta = np.pi / 4
qc = QuantumCircuit(m)
qam_insert(qc, b"test_item", hash_functions, m, theta)
```

## Experimental Design Standards

### Required Metrics (Generate These Plots)

1. **Accuracy vs. memory** (compare to classical baselines)
2. **Accuracy vs. shots** (variance analysis)
3. **Accuracy vs. noise** (robustness testing)
4. **Latency vs. accuracy trade-off**

### Parameter Sweeps

- `m` (qubits): {16, 32, 64}
- `k` (hash functions): {2, 3, 4}
- `|S|` (set size): 2⁵ to 2⁷
- Shots `S`: {128, 256, 512, 1024}
- Noise `ε`: {0, 10⁻³, 10⁻²}
- Report 95% CI over ≥10 trials

### Baselines (Always Compare Against)

- Classical Bloom filter (k tuned to optimal for m, |S|)
- Counting Bloom (for deletions)
- SimHash with k hyperplanes (for Q-SimHash)
- Suffix arrays + sketches (for Q-SubSketch)


## Coding Standards
- Use deterministic RNG seeds.
- All new code must have unit tests in `sim/test_*.py`.
- All experiments must be reproducible via scripts in `experiments/`.
- Document all new theory in `theory/`.
- Update the paper in `paper/draft.tex` as new results/theory are added.

## Acceptance Criteria
- All new engineering, scaling, and theory results are in `theory/`, `results/`, and cited in the paper.
- State caching, config sweeps, and Q-SubSketch evaluation are implemented and measured.
- All required figures are generated and reproducible.
- All code, figures, and paper sections are updated and ready for submission.

## When done, report: "Phase 5–End complete. What next?"

## Theoretical Deliverables

Target achievable lemmas within 1–2 weeks:

1. **QAM false-positive bound**: α ≤ exp(-C·k·(1-ρ)) for load factor ρ=|S|/m
2. **Noise perturbation**: Acceptance gap degrades ≤O(kε) under Pauli noise
3. **Amortized batching**: Show variance reduction factor ~B for batch size B

Document proofs in `/theory` with LaTeX or markdown.

## Hash Functions (Use Deterministic Approach)

Use a simple deterministic hash family to avoid randomness bugs:

```python
def splitmix64(x, seed=0):
    """64-bit deterministic hash (splitmix algorithm)."""
    z = (x + seed + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF

def make_hash_functions(k):
    """Generate k independent hash functions."""
    return [lambda x, i=i: splitmix64(hash(x), seed=i) for i in range(k)]
```

## Common Pitfalls & Mitigations

- **High variance**: Start with larger θ (e.g., π/4), fewer entanglers, S≥512 shots
- **No clear advantage**: Frame as *trade-off analysis*, not dominance; emphasize batch amortization
- **Proof complexity**: Begin with restricted models (no-noise, limited-depth), generalize later
- **Hash collisions**: Use deterministic hash family (splitmix64 above) to avoid randomness bugs
- **Qiskit version issues**: Pin to `qiskit>=1.0` in requirements.txt for stability

## Workflow Commands (Windows PowerShell)

```powershell
# Setup environment (first time)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install qiskit qiskit-aer numpy scipy matplotlib pytest

# Daily workflow
.\venv\Scripts\Activate.ps1

# Run QAM experiments
python experiments/sweeps.py --m 32 --k 3 --shots 512

# Generate plots from latest results
python experiments/plotting.py --results results/qam_latest.csv

# Run tests
pytest sim/ -v

# Work in Jupyter (recommended for exploration)
pip install jupyter
jupyter notebook notebooks/qam.ipynb
```

## Paper/Report Structure

Target: 6–8 pages arXiv-ready tech report

1. Introduction (research motivation, QDS overview)
2. Related Work (classical probabilistic DS, existing quantum approaches)
3. Model & Metrics (computational model, cost metrics, baselines)
4. Algorithm(s) (QAM/Q-SubSketch/Q-SimHash with circuit diagrams)
5. Theoretical Results (bounds, lemmas, complexity statements)
6. Experiments (plots, ablations, robustness analysis)
7. Limitations & Future Work

Draft in `/paper/draft.tex` or `/paper/draft.md`.

## Quick Start (First Hour)

From `roadmap.md` Week 1 objectives:

1. **Set up environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install qiskit qiskit-aer numpy matplotlib jupyter
   ```

2. **Create minimal "Hello QDS" notebook**:
   - Encode bitstrings via angle encoding
   - Apply entangler layer (CZ along a line)
   - Measure Z-expectations
   - Plot expectation vs. shots

3. **Build basic experiment structure**:
   - `sim/qam.py` - core QAM circuit builder
   - `experiments/sweeps.py` - parameter grid runner
   - `notebooks/qam.ipynb` - interactive exploration

Reference: See `roadmap.md` "Week 1 — Setup & toy circuits" for detailed tasks.

## Success Criteria (Preliminary Results)

By end of first cycle (4–6 weeks):

✓ Well-defined quantum data-structure model with operations and error handling
✓ One working prototype on simulator with performance plots vs. classical
✓ At least one theorem/bound (even in restricted model)
✓ Reproducible experiment harness with clean repo structure

Reference: See `roadmap.md` for complete 6-week execution plan with milestones.
