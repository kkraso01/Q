# Amplitude Sketching: A Unified Framework for Quantum Data Structures

## Abstract

We introduce **Amplitude Sketching**, a unified framework for quantum probabilistic data structures that generalizes existing constructions (QAM, Q-SubSketch, Q-SimHash, QHT, Q-Count, Q-HH, Q-LSH) under a single theoretical foundation. The framework is based on three core operations: **phase accumulation**, **interference measurement**, and **amplitude composition**.

---

## 1. Framework Definition

### 1.1 Amplitude Sketch

An **amplitude sketch** is a tuple `AS = (m, H, θ, Φ)` where:
- `m`: Number of qubits (memory size)
- `H = {h₁, ..., hₖ}`: Family of k hash functions hᵢ: X → [m]
- `θ`: Phase rotation magnitude (typically π/4 to π)
- `Φ`: Accumulated phase state (complex amplitudes over m qubits)

### 1.2 Core Operations

#### Insert(x)
Accumulate phase rotations at hash locations:
```
For each hᵢ ∈ H:
    qubit_idx = hᵢ(x)
    Apply Rz(θ) at qubit_idx
```

#### Query(y)
Measure interference between accumulated phases and query pattern:
```
1. Build reference circuit with Insert(y)
2. Apply inverse rotations for accumulated state
3. Measure overlap via Z-expectation or |0⟩ⁿ probability
4. Return overlap ∈ [0, 1]
```

#### Compose(AS₁, AS₂)
Chain amplitude sketches for multi-stage processing:
```
Query output from AS₁ feeds into AS₂
Error propagates: ε_total ≤ ε₁ + ε₂ + O(ε₁·ε₂)
```

---

## 2. Unified Taxonomy

All quantum data structures instantiate amplitude sketching with specific parameter choices:

| Structure | m | k | θ | Phase Pattern | Query Test |
|-----------|---|---|---|---------------|------------|
| **QAM** | 32-128 | 3-5 | π/4 | Uniform per item | Overlap > τ |
| **Q-SubSketch** | 32-64 | 3-4 | π/6 | Rolling hash | ROC threshold |
| **Q-SimHash** | 32-128 | 4-8 | π/4 | Sign-based | Hamming distance |
| **QHT** | 64-256 | 3-4 | π/8 | Prefix hierarchy | Depth-wise test |
| **Q-Count** | 32-128 | 2-4 | π/6 | Bucket hashing | Variance estimator |
| **Q-HH** | 64-128 | 3-4 | θ·log(f) | Frequency-weighted | Top-k ranking |
| **Q-LSH** | 64-256 | 4-16 | atan(⟨v,h⟩) | Hyperplane projection | Cosine similarity |

---

## 3. Theoretical Properties

### 3.1 Space-Accuracy Trade-off

**Theorem 1 (Universal Lower Bound)**: Any amplitude sketch achieving false-positive rate α and robustness (1-ε) requires:

```
m ≥ Ω(log(1/α) / (1-ε))
```

**Proof Sketch**:
1. Holevo bound: m qubits store ≤ m bits of classical information
2. No-cloning: Cannot duplicate quantum states for multiple queries
3. Error propagation: Each gate error ε compounds through k hash operations
4. Information-theoretic: Need log(1/α) bits to distinguish with error α

**Corollary**: This bound applies to ALL structures in taxonomy (QAM, QHT, Q-Count, etc.)

### 3.2 Batch Advantage

**Theorem 2 (Variance Reduction)**: For batch size B, query variance satisfies:

```
Var(batch) ≤ Var(single) / √B  (quantum regime)
Var(batch) ≤ Var(single) / B    (classical limit)
```

**Proof**: Amplitude interference provides √B advantage when queries share circuit preparation. Classical batching only amortizes constant overhead.

### 3.3 Composability

**Theorem 3 (Error Propagation)**: Composing N amplitude sketches with error εᵢ yields total error:

```
ε_total ≤ Σᵢ εᵢ + O(Πᵢ εᵢ)  (first-order + cross-terms)
```

**Implication**: Q-Retrieval pipeline (N=4 stages) maintains ε < 0.01 if each stage has ε < 0.0025.

---

## 4. Abstract Interface

All amplitude sketches implement:

```python
class AmplitudeSketch(ABC):
    def __init__(self, m: int, k: int, theta: float):
        """Initialize sketch with memory size m, k hash functions, rotation θ."""
        
    @abstractmethod
    def insert(self, x: bytes) -> None:
        """Accumulate phase for item x."""
        
    @abstractmethod
    def query(self, y: bytes, shots: int) -> float:
        """Measure interference overlap for query y."""
        
    def compose(self, other: 'AmplitudeSketch') -> 'AmplitudeSketch':
        """Chain this sketch with another (optional)."""
        
    def error_bound(self) -> Tuple[float, float]:
        """Return (false_positive_rate, false_negative_rate)."""
```

---

## 5. Comparison to Classical Frameworks

### 5.1 Classical Probabilistic Sketches

**Bloom Filters, Count-Min Sketch, HyperLogLog**:
- **Accumulation**: Bit flips or counter increments
- **Query**: Logical AND or threshold test
- **Composition**: Serial or parallel (no interference)
- **Space**: O(log(1/α)) per Pătraşcu-Demaine lower bounds

**Amplitude Sketching**:
- **Accumulation**: Phase rotations (continuous)
- **Query**: Interference measurement (probability)
- **Composition**: Amplitude chaining (error propagates coherently)
- **Space**: O(log(1/α)/(1-ε)) with noise factor

### 5.2 Trade-offs

| Property | Classical | Quantum (Amplitude) |
|----------|-----------|---------------------|
| **Insert complexity** | O(k) bit ops | O(k) Rz gates |
| **Query complexity** | O(k) lookups | O(k) Rz + shots |
| **Memory** | O(log 1/α) bits | O(log 1/α / (1-ε)) qubits |
| **Batch advantage** | Amortized constant | √B variance reduction |
| **Composability** | Independent errors | Coherent propagation |

---

## 6. Design Patterns

### 6.1 Phase Encoding Strategies

**Uniform encoding** (QAM, Q-SubSketch):
```python
θ = π / 4  # Fixed rotation
for i in hash_indices(x):
    circuit.rz(θ, i)
```

**Frequency-weighted** (Q-HH):
```python
θ = base_theta * log1p(frequency[x])
for i in hash_indices(x):
    circuit.rz(θ, i)
```

**Adaptive encoding** (Q-LSH):
```python
θ = atan(dot(vector, hyperplane))
for i in hash_indices(x):
    circuit.rz(θ, i)
```

### 6.2 Interference Test Patterns

**Overlap test** (QAM, QHT):
```python
overlap = measure_probability(|0⟩^m)
return overlap > threshold
```

**Expectation test** (Q-Count):
```python
z_exp = measure_z_expectations()
variance = var(z_exp)
cardinality = estimate_from_variance(variance)
```

**Ranking test** (Q-HH, Q-LSH):
```python
scores = [overlap(x, candidate) for candidate in database]
return top_k(scores)
```

---

## 7. Extension: Multi-Level Amplitude Sketching

For hierarchical structures (QHT, Q-Retrieval), we define **multi-level sketches**:

```
Level 0: AS₀ (coarse filtering)
Level 1: AS₁ (medium granularity)
Level 2: AS₂ (fine-grained ranking)
...
```

**Query pipeline**:
1. AS₀.query(x) → candidate set C₀
2. For c ∈ C₀: AS₁.query(c) → refined set C₁
3. For c ∈ C₁: AS₂.query(c) → final ranking

**Error accumulation**:
```
P(false positive at level L) ≤ Πᵢ₌₀ᴸ αᵢ
```

---

## 8. Open Problems

1. **Optimal phase allocation**: Given total rotation budget Θ, how to distribute across k hash functions?

2. **Noise-optimal encoding**: Which phase encoding θ(x) minimizes error under noise ε?

3. **Quantum-classical separation**: When does amplitude sketching provide provable advantage?

4. **Hardware-aware compilation**: Map abstract sketch to realistic device topology

5. **Adaptive sketching**: Update θ or k dynamically based on query feedback

---

## 9. Conclusion

Amplitude sketching unifies quantum data structures under a single framework with:
- **Three core operations**: Insert (phase accumulation), Query (interference), Compose (chaining)
- **Universal bounds**: m ≥ Ω(log(1/α)/(1-ε)) space lower bound
- **Batch advantage**: √B variance reduction over classical
- **Composability**: First-order error propagation theory

This framework enables systematic design and analysis of quantum data structures, providing a foundation for future research in quantum algorithms for data-intensive applications.

---

## References

1. Carter & Wegman (1979): Universal hashing families
2. Bloom (1970): Space/time trade-offs in hash coding
3. Pătraşcu & Demaine (2006): Lower bounds for data structures
4. Nielsen & Chuang (2010): Quantum information theory
5. Holevo (1973): Bounds on quantum information capacity

---

**Next Steps**: 
1. Implement `sim/amplitude_sketch.py` base class
2. Refactor all structures to inherit from `AmplitudeSketch`
3. Prove Theorem 3 (composability) rigorously
4. Extend to continuous-variable quantum systems
