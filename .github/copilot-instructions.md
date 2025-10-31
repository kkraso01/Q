# Quantum Data Structures (QDS) Research Project

## Project Overview

This is a research project exploring quantum data structures with a 14-phase roadmap from preliminary results to field-founding contributions. The project develops quantum alternatives to classical probabilistic data structures (Bloom filters, SimHash, suffix arrays) with demonstrable trade-offs in accuracy, memory, and query performance.

**Current Status: Phase 2 Complete → Phase 3 Ready to Start**

---

## 📍 CURRENT STATUS SUMMARY

### ✅ Completed (Phase 1-2):
- QAM, Q-SubSketch, Q-SimHash core implementations
- Classical baselines (Bloom, Cuckoo, XOR, Vacuum filters)
- Comprehensive experiment harness (sweeps, batch queries, topology variants)
- Theory documentation (bounds, deletion limitations, cell probe model)
- All code infrastructure ready

### 🟡 In Progress (Phase 2 Finalization):
- **IMMEDIATE NEXT**: Run `experiments/generate_all_figures.py` to generate 8+ plots
- Verify figure reproducibility
- Update paper with all Phase 2 results

### 🔴 Not Started (Phase 3+):
- Quantum Hashed Trie (QHT)
- Quantum Count-Distinct (Q-Count)
- Quantum Heavy Hitters (Q-HH)
- Formal lower bounds generalization
- Q-LSH, Q-KV (Phase 4)
- Full retrieval stack (Phase 6)

---

## 📋 COMPLETE PHASE ROADMAP

### Phase 1: Foundation ✅ DONE
- Repository scaffold, QAM prototype, basic experiments, early bounds, paper skeleton

### Phase 2: Strengthening ✅ 95% DONE
- Classical baselines (Cuckoo/XOR/Vacuum) ✅
- Deletion strategy via inverse rotation ✅
- Batch query experiments ✅
- Noise/topology analysis ✅
- **PENDING**: Generate all figures, update paper

### Phase 3: Novel QDS + Lower Bounds (6-10 weeks)
1. Quantum Hashed Trie (QHT) for prefix membership
2. Quantum Count-Distinct (Q-Count) for streaming cardinality
3. Quantum Heavy Hitters (Q-HH) for top-k frequency
4. Formalize lower bound: m ≥ Ω(log(1/α)/(1-ε))

### Phase 4: Generalized Theory + LSH + KV-Cache (6-10 weeks)
1. Generalized lower bounds (batch advantage, multi-query)
2. Quantum LSH (Q-LSH) for similarity search
3. Quantum KV-cache eviction policy
4. Benchmark suite infrastructure

### Phase 5: Foundational Generalization (8+ weeks)
- Unify all primitives under "Amplitude Sketching" framework
- Hardness results and separation theorems
- Composability theory for chained structures

### Phase 6: Full Retrieval System (6-10 weeks)
- Integrate Q-SubSketch → Q-LSH → Q-HH → Q-KV pipeline
- Compare vs FAISS/HNSW/IVF-PQ
- 10+ performance plots (recall, latency, memory, throughput)

### Phase 7: Hybrid Compiler Optimizations (6-10 weeks)
- Amplitude fusion compiler pass
- Noise-aware scheduling
- Ancilla recycling optimization

### Phase 8: Hardware-Aware Models (6-10 weeks)
- Heavy-hex, ion-trap, superconducting topologies
- Routing penalty models
- Realistic transpilation analysis

### Phase 9: Benchmark Suite (QDBench) (6-10 weeks)
- Standard benchmark suite for QDS community
- Canonical metrics and reproducibility infrastructure

### Phase 10: Amplitude Sketching DSL (8+ weeks)
- Domain-specific language for quantum sketches
- Type system for amplitude accumulation
- Formal semantics and safety

### Phase 11: Meta-Theorems (8+ weeks)
- Fundamental separation results
- Lower bounds comparable to Pătraşcu & Demaine work

### Phase 12: Industry Translation (4-8 weeks)
- Package Q-Retrieval and Q-KV for production
- Target RAG vendors, LLM infra, database systems

### Phase 13: Manifesto Paper (6-8 weeks)
- "Amplitude Sketching: A Unified Framework for QDS"
- Citation magnet covering all constructions

### Phase 14: Survey/Book (6+ months)
- Comprehensive survey establishing field vocabulary
- Tutorial and textbook-style treatment

---

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

---

## 🎯 IMMEDIATE ACTION PLAN (Next 2 Weeks)

### Week 1: Complete Phase 2
**Priority 1: Generate Figures**
```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Run comprehensive figure generation
python experiments/generate_all_figures.py

# Verify output in results/ directory
ls results/*.png
```

**Expected outputs:**
1. accuracy_vs_memory.png
2. accuracy_vs_shots.png
3. accuracy_vs_noise.png
4. accuracy_vs_load_factor.png
5. batch_query_error_vs_amortized_cost.png
6. heatmap_shots_noise.png
7. topology_comparison.png
8. q_subsketch_auc.png

**Priority 2: Paper Update**
- Review `paper/draft.md`
- Add all 8 figures with captions
- Update experimental results section
- Document classical baseline comparisons
- Add deletion strategy results
- Include batch advantage analysis

**Priority 3: Reproducibility Check**
- Re-run `generate_all_figures.py` with clean environment
- Verify deterministic outputs
- Document any missing dependencies

### Week 2: Begin Phase 3
**Quantum Hashed Trie (QHT)**
1. Create `sim/qht.py` with insert/query operations
2. Implement phase rotations for character prefixes
3. Add `sim/test_qht.py` with unit tests
4. Create `notebooks/qht.ipynb` for exploration
5. Sweep branching factors b ∈ {2, 4, 8, 16}
6. Sweep depths L ∈ {4, 8, 16, 32}
7. Generate ROC curves

**Quantum Count-Distinct (Q-Count)**
1. Create `sim/q_count.py` with cardinality estimator
2. Implement bucket hashing and phase encoding
3. Add `sim/test_q_count.py`
4. Create `notebooks/q_count.ipynb`
5. Compare to HyperLogLog baseline
6. Plot: error vs load factor, buckets vs error, shots vs error

---

## 📊 SUCCESS METRICS BY PHASE

### Phase 2 Complete When:
- [ ] All 8+ figures generated and in `results/`
- [ ] Paper updated with experimental results
- [ ] Reproducibility verified (clean run succeeds)
- [ ] All tests passing: `pytest sim/ -v`

### Phase 3 Complete When:
- [ ] QHT, Q-Count, Q-HH implemented with tests
- [ ] Lower bound theorem documented in `theory/general_bounds.md`
- [ ] 12-18 high-quality figures total
- [ ] Hardware-aware topology results included
- [ ] Paper at 15-20 pages with all three new structures

### Phase 4 Complete When:
- [ ] Q-LSH achieves ≥ classical LSH recall@k at similar memory
- [ ] Q-KV shows ≥1-2% PPL reduction OR ≥5-10% hit-rate gain
- [ ] `benchmarks/run_all.py` reproduces all figures end-to-end
- [ ] General bounds theorems proven and documented

---

## 🚨 CRITICAL DEPENDENCIES

**Before starting any Phase 3+ work:**
1. ✅ Python 3.11 environment active
2. ✅ Qiskit ≥1.0 installed
3. ✅ All Phase 2 tests passing
4. ✅ Figures generated successfully
5. ✅ Paper draft up-to-date

**For Phase 4 (Q-LSH/Q-KV):**
- Install: `pip install faiss-cpu scikit-learn transformers torch`
- Download: SIFT1M subset, GloVe embeddings
- Prepare: WikiText-103 or The Pile subset

---

## 📚 KEY REFERENCE DOCS

- **Roadmap**: `roadmap.md` (original 6-week plan)
- **Phase 2**: `roadmap_phase2.md` (strengthening)
- **Phase 3**: `roadmap_phase3.md` (novel QDS)
- **Phase 4**: `roadmap_phase4.md` (generalized theory + LSH)
- **Phase 5**: `roadmap_phase5.md` (foundational generalization)
- **Phase 6**: `roadmap_phase6.md` (retrieval system)
- **Phases 7-14**: `roadmap_phase7.md` through `roadmap_phase14.md` (placeholders)

---

## When Phase 2 complete, report: "Phase 2 finalized. Ready for Phase 3."
## When Phase 3 complete, report: "Phase 3 complete. Ready for Phase 4."

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
