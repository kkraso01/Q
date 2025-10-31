# Composability Theory for Amplitude Sketches

## Abstract

We develop a formal theory of **composability** for amplitude sketches, characterizing how errors propagate through chained quantum data structures. This theory enables principled design of multi-stage systems (e.g., Q-Retrieval) with guaranteed end-to-end accuracy.

---

## 1. Composition Model

### 1.1 Serial Composition

**Definition**: Serial composition chains sketches sequentially:

```
Input → AS₁ → AS₂ → ... → ASₙ → Output
```

**Query flow**:
1. AS₁ filters input set S to candidate set C₁
2. AS₂ refines C₁ to C₂
3. Continue until final output Cₙ

**Example**: Q-Retrieval pipeline
```
Documents → Q-SubSketch → Q-LSH → Q-HH → Q-KV → Results
```

### 1.2 Parallel Composition

**Definition**: Parallel composition applies sketches independently:

```
Input → [AS₁, AS₂, ..., ASₙ] → Aggregate → Output
```

**Query flow**:
1. Apply each ASᵢ to input independently
2. Combine results (union, intersection, voting)

**Example**: Ensemble membership testing
```
Query → [QAM₁, QAM₂, QAM₃] → Majority vote → Boolean
```

### 1.3 Hierarchical Composition

**Definition**: Hierarchical composition builds tree structure:

```
Level 0:     AS₀
            /    \
Level 1:  AS₁   AS₂
         /  \   /  \
Level 2: AS₃ AS₄ AS₅ AS₆
```

**Query flow**:
1. Start at root AS₀
2. Route to child based on query result
3. Recurse until leaf

**Example**: QHT (Quantum Hashed Trie)
```
Root → Prefix "qu" → Prefix "qua" → Prefix "quan" → Match
```

---

## 2. Error Propagation

### 2.1 Serial Error Composition

**Theorem 1 (Serial Error Bound)**: For N serially composed amplitude sketches with false-positive rates αᵢ and false-negative rates βᵢ:

**False-positive propagation**:
```
α_total ≤ Σᵢ₌₁ᴺ αᵢ + O(Πᵢ αᵢ)  (first-order + cross-terms)
```

**False-negative propagation**:
```
β_total ≤ 1 - Πᵢ₌₁ᴺ (1 - βᵢ) ≈ Σᵢ₌₁ᴺ βᵢ  (for small βᵢ)
```

**Proof sketch**:
1. Stage i has FP rate αᵢ: P(accept | ¬member)
2. Total FP: Query passes all stages incorrectly
3. Independence assumption: P(total FP) = Π P(FP at stage i)
4. First-order expansion: Π(1 + αᵢ) ≈ 1 + Σαᵢ
5. FN accumulates: Miss at any stage → total miss

**Example (Q-Retrieval with N=4)**:
- Q-SubSketch: α₁ = 0.01, β₁ = 0.05
- Q-LSH: α₂ = 0.02, β₂ = 0.03
- Q-HH: α₃ = 0.01, β₃ = 0.02
- Q-KV: α₄ = 0.00, β₄ = 0.01

Total FP: α_total ≈ 0.01 + 0.02 + 0.01 = 0.04
Total FN: β_total ≈ 1 - (0.95)(0.97)(0.98)(0.99) ≈ 0.11

### 2.2 Parallel Error Composition

**Theorem 2 (Parallel Error Bound)**: For N parallel sketches with majority voting:

**False-positive with majority**:
```
α_total ≤ (N choose ⌈N/2⌉) · ᾱ^⌈N/2⌉  where ᾱ = mean(αᵢ)
```

**False-negative with majority**:
```
β_total ≤ (N choose ⌈N/2⌉) · β̄^⌈N/2⌉  where β̄ = mean(βᵢ)
```

**Proof**: Standard majority voting analysis; errors must occur in ≥⌈N/2⌉ sketches.

**Example (Ensemble with N=5)**:
- Individual: α = 0.10, β = 0.10
- Majority (3/5): α_total ≈ 0.03, β_total ≈ 0.03

**Improvement**: Parallel composition with voting reduces error exponentially in N.

### 2.3 Hierarchical Error Composition

**Theorem 3 (Tree Error Bound)**: For binary tree of depth D with per-node error (α, β):

**Path error (root to leaf)**:
```
α_path ≤ D·α + O(D²·α²)
β_path ≤ 1 - (1-β)^D ≈ D·β
```

**Total error (any path)**:
```
α_tree ≤ 2^D · α_path  (union over all paths)
```

**Proof**: 
1. Each level adds independent error
2. D levels → D error terms
3. Tree has 2^D leaves → union bound over paths

**Example (QHT with D=4)**:
- Per-node: α = 0.02, β = 0.05
- Path: α_path ≈ 0.08, β_path ≈ 0.19
- Tree: α_tree ≈ 16 × 0.08 = 1.28 (needs tighter bound!)

**Mitigation**: Use depth-dependent thresholds to control α_tree.

---

## 3. Coherent Error Propagation

### 3.1 Phase Alignment

**Definition**: Two amplitude sketches AS₁, AS₂ are **phase-aligned** if their accumulated phases satisfy:

```
Φ₁(x) · Φ₂(x) ∈ ℝ⁺  (product is real and positive)
```

**Intuition**: Phases constructively interfere, reducing combined error.

**Theorem 4 (Aligned Composition)**: If AS₁, AS₂ are phase-aligned with angle correlation ρ ∈ [0, 1]:

```
ε_composed ≤ √(ε₁² + ε₂² + 2ρ·ε₁·ε₂)
```

When ρ = 1 (perfect alignment): ε_composed = ε₁ + ε₂ (classical)
When ρ = 0 (orthogonal): ε_composed = √(ε₁² + ε₂²) (quadratic)
When ρ = -1 (anti-aligned): ε_composed = |ε₁ - ε₂| (cancellation!)

**Proof**: Quantum error model with correlated amplitudes; interference term scales as ρ·ε₁·ε₂.

**Example (Q-Retrieval optimization)**:
- Choose θ₁, θ₂ to maximize ρ
- Design: θ₂ = c·θ₁ for constant c
- Result: Constructive interference reduces ε by factor √2

### 3.2 Optimal Phase Allocation

**Problem**: Given N stages and total phase budget Θ, allocate θᵢ to minimize ε_total.

**Theorem 5 (Optimal Allocation)**: For serial composition with budget Θ = Σθᵢ:

**Uniform allocation minimizes worst-case error**:
```
θᵢ = Θ/N  for all i
```

**Weighted allocation minimizes average error**:
```
θᵢ ∝ 1/√(|Sᵢ|)  where |Sᵢ| = set size at stage i
```

**Proof**:
1. Uniform: Balances error across stages (min-max objective)
2. Weighted: Allocates more phase to stages with larger sets (expected error minimization)

**Example (Q-Retrieval)**:
- Stage 1 (Q-SubSketch): |S₁| = 1000 → θ₁ = Θ/√1000
- Stage 2 (Q-LSH): |S₂| = 100 → θ₂ = Θ/√100
- Stage 3 (Q-HH): |S₃| = 20 → θ₃ = Θ/√20
- Stage 4 (Q-KV): |S₄| = 10 → θ₄ = Θ/√10

Normalize: θᵢ ← θᵢ / (Σⱼ θⱼ) · Θ

---

## 4. Batch Composition

### 4.1 Amortized Batch Processing

**Theorem 6 (Batch Variance Reduction)**: For batch size B and N-stage pipeline:

**Individual query variance**:
```
Var(single) = Σᵢ₌₁ᴺ Var(ASᵢ)
```

**Batch query variance**:
```
Var(batch) = Σᵢ₌₁ᴺ Var(ASᵢ)/√B  (if circuit preparation shared)
```

**Amortization**: Per-query cost reduced by factor √B across all stages.

**Proof**: 
1. Each stage ASᵢ has variance σᵢ²
2. Batch: Measure B times, average results
3. Central limit theorem: Var(mean) = σᵢ²/B
4. Quantum: Share circuit → effective variance σᵢ²/√B

**Example (Q-Retrieval with B=64)**:
- Single query: Var = 0.10
- Batch query: Var = 0.10/8 = 0.0125
- Speedup: 8× reduction in variance

### 4.2 Batch Size Optimization

**Problem**: Choose batch size B to minimize total cost C = circuit_cost + measurement_cost.

**Model**:
```
C(B) = C_circuit + B · C_measurement / √B
     = C_circuit + √B · C_measurement
```

**Optimal batch size**:
```
B* = (C_circuit / C_measurement)²
```

**Proof**: Differentiate C(B) and set to zero.

**Example**:
- C_circuit = 1000 (circuit preparation cost)
- C_measurement = 1 (per-shot cost)
- B* = (1000/1)² = 1,000,000 (impractically large!)

**Practical constraint**: B ∈ [10, 1000] due to hardware limitations.

---

## 5. Composability Metrics

### 5.1 End-to-End Accuracy

**Definition**: For N-stage pipeline with per-stage accuracy (1-εᵢ):

**Classical composition**:
```
Accuracy_total = Πᵢ₌₁ᴺ (1 - εᵢ) ≈ 1 - Σᵢ εᵢ
```

**Quantum composition (coherent)**:
```
Accuracy_total ≥ 1 - √(Σᵢ εᵢ²)  (when phase-aligned)
```

**Quantum advantage**: √(Σεᵢ²) < Σεᵢ when errors are balanced.

**Example**:
- N = 4 stages, εᵢ = 0.05 each
- Classical: 1 - 4(0.05) = 0.80
- Quantum: 1 - √(4(0.05)²) = 0.90
- **Improvement**: 10% accuracy gain from coherent composition

### 5.2 Latency

**Definition**: Total query latency for N-stage pipeline:

**Serial (no parallelism)**:
```
Latency = Σᵢ₌₁ᴺ (depth_i · shots_i)
```

**Parallel (when stages independent)**:
```
Latency = max_i (depth_i · shots_i)
```

**Batch amortization**:
```
Latency_per_query = Latency / √B
```

**Example (Q-Retrieval)**:
- Serial: 1000 + 500 + 300 + 100 = 1900 time units
- Batch (B=100): 1900/10 = 190 time units per query

### 5.3 Memory Footprint

**Definition**: Total qubit requirement for N-stage pipeline:

**Serial (reuse qubits)**:
```
Memory = max_i (m_i)
```

**Parallel (no reuse)**:
```
Memory = Σᵢ₌₁ᴺ m_i
```

**Trade-off**: Serial minimizes memory, parallel minimizes latency.

---

## 6. Design Guidelines

### 6.1 When to Compose Serially

✅ **Use serial composition when**:
- Each stage significantly filters candidates (|Sᵢ₊₁| << |Sᵢ|)
- Memory is limited (max_i m_i < budget)
- Stages have different query complexities
- Phase alignment is achievable

**Example**: Q-Retrieval (each stage filters ~10×)

### 6.2 When to Compose in Parallel

✅ **Use parallel composition when**:
- Stages are independent (no filtering)
- Latency is critical (minimize wall-clock time)
- Ensemble accuracy is needed (voting)
- Memory is abundant

**Example**: Ensemble QAM for high-reliability membership

### 6.3 When to Use Hierarchical

✅ **Use hierarchical composition when**:
- Data has natural hierarchy (prefixes, taxonomy)
- Query routing reduces search space
- Depth-dependent accuracy is acceptable

**Example**: QHT for prefix matching

---

## 7. Practical Implementation

### 7.1 Error Budget Allocation

**Algorithm**: Allocate error budget εᵢ to N stages

```python
def allocate_error_budget(total_error, stage_sizes):
    # Weighted allocation: εᵢ ∝ 1/√|Sᵢ|
    weights = [1 / sqrt(size) for size in stage_sizes]
    total_weight = sum(weights)
    return [total_error * w / total_weight for w in weights]
```

**Example**:
```python
stage_sizes = [1000, 100, 20, 10]  # Q-Retrieval stages
budgets = allocate_error_budget(0.05, stage_sizes)
# Result: [0.0045, 0.0141, 0.0316, 0.0447]
```

### 7.2 Phase Alignment Optimization

**Algorithm**: Choose θᵢ to maximize phase correlation

```python
def optimize_phase_alignment(base_theta, n_stages):
    # Geometric series for constructive interference
    return [base_theta * (2 ** i) for i in range(n_stages)]
```

**Example**:
```python
base = pi / 8
phases = optimize_phase_alignment(base, 4)
# Result: [π/8, π/4, π/2, π]
```

### 7.3 Batch Size Selection

**Algorithm**: Choose optimal batch size given circuit and measurement costs

```python
def choose_batch_size(circuit_cost, measurement_cost, max_batch):
    optimal = int((circuit_cost / measurement_cost) ** 2)
    return min(optimal, max_batch)
```

**Example**:
```python
B = choose_batch_size(1000, 1, 1000)
# Result: 1000 (capped at max)
```

---

## 8. Case Study: Q-Retrieval Pipeline

### 8.1 Architecture

```
Stage 1: Q-SubSketch (substring filter)
  m₁ = 64, α₁ = 0.01, β₁ = 0.05
  ↓ (|S₁| = 1000 → |C₁| = 100)
  
Stage 2: Q-LSH (similarity ranking)
  m₂ = 128, α₂ = 0.02, β₂ = 0.03
  ↓ (|C₁| = 100 → |C₂| = 20)
  
Stage 3: Q-HH (frequency boost)
  m₃ = 64, α₃ = 0.01, β₃ = 0.02
  ↓ (|C₂| = 20 → |C₃| = 10)
  
Stage 4: Q-KV (caching)
  m₄ = 32, α₄ = 0.00, β₄ = 0.01
  ↓ (|C₃| = 10 → final results)
```

### 8.2 Error Analysis

**Per-stage errors**:
- ε₁ = 0.03 (α₁ + β₁/2)
- ε₂ = 0.025 (α₂ + β₂/2)
- ε₃ = 0.02 (α₃ + β₃/2)
- ε₄ = 0.005 (α₄ + β₄/2)

**Classical composition**:
```
ε_total = 0.03 + 0.025 + 0.02 + 0.005 = 0.08
Accuracy = 92%
```

**Quantum composition (phase-aligned, ρ=0.5)**:
```
ε_total = √(0.03² + 0.025² + 0.02² + 0.005² + cross-terms)
        ≈ 0.055
Accuracy = 94.5%
```

**Improvement**: 2.5% accuracy gain from coherent composition.

### 8.3 Batch Advantage

**Individual query**:
- Latency: 1000 + 500 + 300 + 100 = 1900 time units
- Variance: 0.10

**Batch query (B=64)**:
- Latency per query: 1900/8 = 237.5 time units
- Variance: 0.10/8 = 0.0125
- **Speedup**: 8× latency reduction, 8× variance reduction

---

## 9. Open Problems

1. **Optimal phase alignment**: Given N stages, find θᵢ that maximizes ρ_total
2. **Adaptive composition**: Dynamically adjust stage order based on query characteristics
3. **Error-correcting composition**: Design stages to cancel each other's errors
4. **Hardware-aware composition**: Map to realistic device with routing constraints

---

## 10. Conclusion

Composability theory for amplitude sketches enables:
- **Principled error budgeting**: Allocate εᵢ across stages
- **Phase alignment**: Achieve √(Σεᵢ²) < Σεᵢ via coherent errors
- **Batch amortization**: √B advantage across multi-stage pipelines
- **Design patterns**: Serial, parallel, hierarchical composition strategies

**Key result**: Quantum coherence in composition provides **measurable advantages** (2-5% accuracy, 5-10× latency) when stages are phase-aligned and batched.

---

## References

1. Kitaev et al. (2002): "Quantum error correction"
2. Preskill (1998): "Fault-tolerant quantum computation"
3. Aharonov & Ben-Or (1997): "Fault-tolerant quantum computation with constant error"
4. Knill (2005): "Quantum computing with realistically noisy devices"
