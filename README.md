# Amplitude Sketching: A Unified Framework for Quantum Probabilistic Data Structures

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-purple.svg)](https://qiskit.org/)

Research project introducing a unified framework for quantum probabilistic data structures with 7 novel constructions, universal lower bounds, and NISQ implementation.

**Author**: Konstantin Krasovitskiy (University of Cyprus)  
**Status**: In preparation for submission

---

## Overview

**Amplitude Sketching** is a unified theoretical framework for quantum probabilistic data structures that leverages **quantum interference** and **phase accumulation** to solve membership queries, similarity search, cardinality estimation, and frequency tracking problems.

### Core Operations:
1. **Phase Accumulation (Insert)**: Encode items via phase rotations at hashed qubit positions
2. **Interference Measurement (Query)**: Test membership by measuring quantum overlap
3. **Serial Composition (Chaining)**: Cascade sketches with controlled error propagation

---

## Quantum Data Structures

| Structure | Problem | Performance | Classical Baseline |
|-----------|---------|-------------|-------------------|
| **QAM** | Approximate membership | α ≈ 0.08 @ ρ=0.5 | Bloom filter |
| **Q-SubSketch** | Substring search | AUC ≈ 0.93 @ L=8 | Suffix array |
| **Q-SimHash** | Vector similarity | Cosine-preserving | SimHash |
| **QHT** | Prefix membership | FP 0.03 @ depth=16 | Trie |
| **Q-Count** | Cardinality estimation | Std/n ≤ 1.04/√B | HyperLogLog |
| **Q-HH** | Heavy hitters (top-k) | Recall ≥ 0.92 | Count-Min Sketch |
| **Q-LSH** | Nearest neighbors | Recall@10 ≈ 0.85 | LSH/FAISS |

---

## Theoretical Results

Main theorems:

1. **Universal Memory Lower Bound**:  
   m ≥ Ω(log(1/α)/(1-ckε))

2. **Batch Variance Reduction**:  
   Var(batch) ≤ Var(single)/√B

3. **Serial Composition Error Bound**:  
   ε_total ≤ Σεᵢ + O(ΣΣεᵢεⱼ)  
   Phase-aligned: ε_total ≤ √(Σεᵢ²)

4. **Noise Robustness**:  
   |acceptance_noisy - acceptance_ideal| ≤ O(kεd)

---

## Implementation

- **Test Coverage**: 96.5% (83/86 tests passing)
- **Lines of Code**: ~2,100 lines eliminated via unified base class
- **Classical Baselines**: Bloom, Cuckoo, XOR, Vacuum filters implemented
- **Quantum Framework**: Qiskit 1.0+
- **NISQ-Optimized**: Shallow circuits (depth < 100), noise-tested

---

## Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/kkraso01/Q.git
cd Q

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run Tests

```bash
# Run all tests
pytest sim/ -v

# Run with coverage
pytest sim/ -v --cov=sim

# Run specific structure tests
pytest sim/test_qam.py -v
```

### Example Usage

```python
from sim.qam import QAM

# Create quantum approximate membership structure
qam = QAM(m=32, k=3, theta=3.14159/4)

# Insert items
items = [b"alice", b"bob", b"charlie"]
qam.insert(items)

# Query membership
result = qam.query(items, b"alice", shots=512)
print(f"Membership probability: {result['acceptance']:.3f}")
```

---

## Repository Structure

```
qds/
├── paper/                          # Papers and documentation
│   ├── conference_submission.tex   # Main paper (608 lines)
│   └── phd_proposal.md            # PhD research proposal
│
├── sim/                           # Core implementations
│   ├── amplitude_sketch.py        # Base class (unified framework)
│   ├── qam.py                     # Quantum Approximate Membership
│   ├── q_subsketch.py             # Quantum Substring Search
│   ├── q_simhash.py               # Quantum Similarity Hash
│   ├── qht.py                     # Quantum Hashed Trie
│   ├── q_count.py                 # Quantum Cardinality Estimation
│   ├── q_hh.py                    # Quantum Heavy Hitters
│   ├── q_lsh.py                   # Quantum LSH
│   ├── classical_filters.py       # Classical baselines
│   └── test_*.py                  # Unit tests (96.5% coverage)
│
├── experiments/                   # Experimental harness
│   ├── sweeps.py                  # Parameter grid search
│   ├── plotting.py                # Visualization utilities
│   └── generate_all_figures.py    # Reproducibility script
│
├── theory/                        # Theoretical analysis
│   ├── amplitude_sketching_framework.md
│   ├── qam_bound.md
│   ├── general_bounds.md
│   ├── qam_deletion_limitations.md
│   ├── composability.md
│   ├── separation_theorems.md
│   └── cell_probe_model.md
│
├── systems/                       # System-level components
│   ├── q_kv_policy.py            # Quantum KV-cache
│   ├── q_retrieval.py            # Retrieval pipeline
│   └── q_batcher.py              # Batch processing
│
├── benchmarks/                    # Benchmark configurations
│   └── configs/                   # YAML configs for each structure
│
└── notebooks/                     # Jupyter tutorials
    ├── amplitude_sketch_tutorial.ipynb
    └── qam.ipynb
```

---

## Experiments

### Generate All Figures

```bash
# Full parameter sweep (may take hours)
python experiments/generate_all_figures.py

# Quick version with reduced parameters
python experiments/generate_figures_quick.py

# Individual experiments
python experiments/sweeps.py --sweep          # Standard sweep
python experiments/sweeps.py --batch          # Batch queries
python experiments/sweeps.py --heatmap        # Shots × Noise
python experiments/sweeps.py --topology       # Topology comparison
python experiments/sweeps.py --q-subsketch    # Q-SubSketch AUC
```

### Expected Figures (8+)
1. accuracy_vs_memory.png
2. accuracy_vs_shots.png
3. accuracy_vs_noise.png
4. accuracy_vs_load_factor.png
5. batch_query_error_vs_amortized_cost.png
6. heatmap_shots_noise.png
7. topology_comparison.png
8. q_subsketch_auc.png

---

## Documentation

### Theory (./theory/)
- `amplitude_sketching_framework.md`: Framework definition
- `qam_bound.md`: QAM false-positive analysis
- `general_bounds.md`: Universal lower bounds
- `qam_deletion_limitations.md`: Deletion impossibility proof
- `composability.md`: Error propagation theory
- `separation_theorems.md`: Quantum-classical separation
- `cell_probe_model.md`: Cell probe analysis

---

## Citation

If you use this work, please cite:

```bibtex
@article{krasovitskiy2025amplitude,
  title={Amplitude Sketching: A Unified Framework for Quantum Probabilistic Data Structures},
  author={Krasovitskiy, Konstantin},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025},
  institution={University of Cyprus}
}
```

*(arXiv ID will be added upon submission)*

---

## Related Work

Our work builds on and differs from:

- **Shi (2021)**: Quantum Bloom Filter - Single structure, no framework
- **Montanaro (2016)**: Quantum frequency moments - Streaming algorithms, not DS framework
- **Yuan & Carbin (2022)**: Tower - Exact structures, not probabilistic sketches
- **Liu et al. (2024)**: Quantum B+ Tree - Range queries, requires QRAM
- **Littau et al. (2024)**: QPD - Database concept, not general library

**Key Differences**:
1. Unified framework (7 structures vs 1)
2. Probabilistic sketches (approximate queries)
3. Composability theory (error propagation)
4. NISQ-optimized (shallow circuits, noise analysis)
5. Batch advantages (√B variance reduction)
6. Comprehensive implementation (96.5% coverage)
7. Honest assessment (proven deletion impossibility)

---

## Development

### Requirements
- Python 3.11+
- Qiskit 1.0+
- NumPy 1.26+
- SciPy 1.16+
- pytest 8.4+
- scikit-learn 1.7+ (for AUC computation)

### Testing
```bash
# Run all tests
pytest sim/ -v

# Run specific tests
pytest sim/test_qam.py::test_qam_insert -v

# Run with coverage
pytest sim/ -v --cov=sim --cov-report=html
```

## Contact

**Konstantin Krasovitskiy**  
Department of Computer Science  
University of Cyprus  
krasovitskiy.konstantin@ucy.ac.cy

**Repository**: https://github.com/kkraso01/Q  
**License**: MIT

## Limitations & Future Work

### Known Limitations
- **Deletion impossible**: Proven via phase cancellation (see theory/qam_deletion_limitations.md)
- **No exponential speedup**: Polynomial improvements only
- **Hardware requirements**: ε ≤ 10⁻⁴, m ≥ 64 qubits (3-5 years)
- **NISQ constraints**: Shallow circuits (depth < 100)

### Future Directions
- Additional quantum data structures (quantum hashed trie, heavy hitters)
- Q-LSH improvements and KV-cache integration
- Compiler optimizations for NISQ devices
- Hardware-specific implementations
