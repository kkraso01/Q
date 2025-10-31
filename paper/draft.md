# Quantum Data Structures: Towards Quantum-Accelerated Probabilistic Membership, Similarity, and Substring Search

## 1. Introduction

Classical probabilistic data structures such as Bloom filters, SimHash, and suffix arrays are foundational for fast approximate membership, similarity, and substring search. As quantum computing matures, it is natural to ask: can quantum algorithms offer new trade-offs in accuracy, memory, and query performance for these core primitives? In this work, we introduce and analyze three quantum data structures (QDS): Quantum Approximate Membership (QAM), Quantum Suffix Sketch (Q-SubSketch), and Quantum Similarity Hash (Q-SimHash). We provide quantum circuit constructions, theoretical bounds, and simulation-based experiments comparing QDS to classical baselines.

## 2. Related Work

We review classical probabilistic data structures (Bloom filters, counting Bloom, SimHash, suffix arrays) and prior quantum approaches to search and hashing. We discuss the quantum query model, Grover's search, and recent work on quantum fingerprinting and quantum Bloom filters.

## 3. Model & Metrics

We adopt the unitary circuit model with noisy measurements. Key metrics: time (gate depth), space (logical qubits), accuracy (false positive/negative rates), and shot budget. We compare QDS to classical baselines under matched memory and accuracy constraints.

## 4. Algorithms

### 4.1 Quantum Approximate Membership (QAM)

QAM encodes set membership via phase rotations on a register of $m$ qubits, using $k$ hash functions. Querying is performed by re-applying the hash pattern and measuring overlap with the reference state. We provide circuit diagrams and pseudocode.

### 4.2 Quantum Suffix Sketch (Q-SubSketch)

Q-SubSketch encodes substrings of a text into a quantum register using phase encoding and stride-based hashing. Substring queries are performed by interference and measurement. We describe the construction and its relation to classical suffix arrays and sketches.

### 4.3 Quantum Similarity Hash (Q-SimHash)

Q-SimHash encodes binary vectors into quantum states using $k$ phase rotations. Similarity queries are performed by measuring the overlap between two encoded states. We relate this to classical SimHash and quantum fingerprinting.

## 5. Theoretical Results

We prove bounds on false positive rates for QAM, analyze noise robustness, and discuss the variance reduction from quantum batching. Lemmas are provided for each QDS, with proofs in the appendix.

## 6. Experiments

We implement all QDS in Qiskit and run parameter sweeps on simulated quantum hardware. Metrics: accuracy vs. memory, accuracy vs. shots, accuracy vs. noise, and latency vs. accuracy. We compare to classical baselines and report 95% confidence intervals.

## 7. Limitations & Future Work

We discuss the limitations of current quantum hardware, simulation bottlenecks, and open questions in quantum data structure design. Future work includes hardware experiments, error mitigation, and extensions to dynamic and streaming settings.

## References

[1] Bloom, B. H. (1970). Space/time trade-offs in hash coding with allowable errors. Communications of the ACM.
[2] SimHash: Charikar, M. S. (2002). Similarity estimation techniques from rounding algorithms. STOC.
[3] Grover, L. K. (1996). A fast quantum mechanical algorithm for database search. STOC.
[4] Buhrman, H., et al. (2001). Quantum fingerprinting. Physical Review Letters.
[5] Qiskit: https://qiskit.org/

## Appendix: Proofs and Additional Figures

See `theory/` for detailed proofs and additional plots.
