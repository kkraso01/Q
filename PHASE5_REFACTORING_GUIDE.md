# Phase 5: Refactoring Guide - Amplitude Sketch Inheritance

## Objective

Refactor all existing quantum data structures (QAM, Q-SubSketch, Q-SimHash, QHT, Q-Count, Q-HH, Q-LSH) to inherit from the unified `AmplitudeSketch` base class.

---

## Refactoring Strategy

### Step 1: Import Base Class

```python
from sim.amplitude_sketch import AmplitudeSketch
```

### Step 2: Update Class Declaration

**Before:**
```python
class QAM:
    def __init__(self, m, k, theta=np.pi/4):
        self.m = m
        self.k = k
        self.theta = theta
        # ... custom initialization
```

**After:**
```python
class QAM(AmplitudeSketch):
    def __init__(self, m, k, theta=np.pi/4, topology='none'):
        super().__init__(m, k, theta)
        # ... custom initialization (topology, etc.)
        self.topology = topology
```

### Step 3: Ensure Core Methods Match Interface

Required methods:
- `insert(x: bytes) -> None`
- `query(y: bytes, shots: int, noise_level: float) -> float`
- `_build_insert_circuit(x: bytes) -> QuantumCircuit`

### Step 4: Leverage Base Class Utilities

Use inherited methods:
- `self._hash_to_indices(x)` - instead of custom hashing
- `self._create_noise_model(noise_level)` - standardized noise
- `self._measure_overlap(circuit, shots, noise_level)` - common measurement
- `self.error_bound()` - automatic error estimation
- `self.get_stats()` - unified statistics

### Step 5: Remove Redundant Code

Delete duplicate implementations:
- Hash function initialization (use base class)
- Noise model creation (use `_create_noise_model`)
- Circuit caching logic (inherited)
- Statistics tracking (use `get_stats()`)

---

## Structure-Specific Adaptations

### QAM (Quantum Approximate Membership)

**Key differences:**
- Topology support (entanglement layers)
- Deletion via inverse rotations
- Multiple query modes (overlap, expectation)

**Refactor approach:**
```python
class QAM(AmplitudeSketch):
    def __init__(self, m, k, theta=np.pi/4, topology='none'):
        super().__init__(m, k, theta)
        self.topology = topology
        self.inserted_items = []
        self.deleted_items = []
    
    def insert(self, x: bytes):
        self.inserted_items.append(x)
        self.n_inserts += 1
    
    def delete(self, x: bytes):
        """QAM-specific deletion."""
        self.deleted_items.append(x)
    
    def _build_insert_circuit(self, x: bytes):
        qc = QuantumCircuit(self.m)
        indices = self._hash_to_indices(x)
        for idx in indices:
            qc.rz(self.theta, idx)
        self._apply_entanglement_layer(qc)  # QAM-specific
        return qc
    
    def _apply_entanglement_layer(self, qc):
        """Topology-specific entanglement (QAM feature)."""
        if self.topology == 'linear':
            for i in range(self.m - 1):
                qc.cz(i, i+1)
        # ... other topologies
```

### QHT (Quantum Hashed Trie)

**Key differences:**
- Hierarchical prefix structure
- Depth-wise phase encoding
- Branching factor

**Refactor approach:**
```python
class QHT(AmplitudeSketch):
    def __init__(self, m, k, theta=np.pi/8, branching_factor=4, max_depth=16):
        super().__init__(m, k, theta)
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.trie_data = {}
    
    def _build_insert_circuit(self, x: bytes):
        qc = QuantumCircuit(self.m)
        prefixes = self._extract_prefixes(x)
        for depth, prefix in enumerate(prefixes):
            indices = self._hash_to_indices(prefix)
            for idx in indices:
                # Depth-dependent phase
                qc.rz(self.theta / (depth + 1), idx)
        return qc
```

### Q-Count (Streaming Cardinality)

**Key differences:**
- Bucket-based hashing
- Variance estimator for cardinality
- No membership query

**Refactor approach:**
```python
class QCount(AmplitudeSketch):
    def __init__(self, m, k, theta=np.pi/6, n_buckets=16):
        super().__init__(m, k, theta)
        self.n_buckets = n_buckets
    
    def query(self, y: bytes, shots: int, noise_level: float) -> float:
        """Override: Returns cardinality estimate, not overlap."""
        return self.estimate_cardinality(shots, noise_level)
    
    def estimate_cardinality(self, shots=1024, noise_level=0.0):
        """Q-Count specific: estimate unique count."""
        # Measure variance of Z-expectations
        # ... variance-based estimator
```

### Q-LSH (Locality-Sensitive Hashing)

**Key differences:**
- Vector embeddings instead of bytes
- Hyperplane projections
- Cosine similarity

**Refactor approach:**
```python
class QLSH(AmplitudeSketch):
    def __init__(self, m, k, d, theta=np.pi/4):
        super().__init__(m, k, theta)
        self.d = d  # Vector dimension
        self.hyperplanes = self._generate_hyperplanes()
        self.inserted_vectors = []
    
    def insert(self, vector: np.ndarray):
        """Insert vector (not bytes)."""
        self.inserted_vectors.append(vector)
        self.n_inserts += 1
    
    def _build_insert_circuit(self, vector: np.ndarray):
        """Encode vector via hyperplane projections."""
        qc = QuantumCircuit(self.m)
        for i, hyperplane in enumerate(self.hyperplanes):
            projection = np.dot(vector, hyperplane)
            angle = np.arctan(projection)
            qc.rz(angle, i % self.m)
        return qc
```

---

## Benefits of Refactoring

### 1. Code Reuse
- Eliminate ~200 lines of duplicate code per structure
- Standardized noise handling
- Common circuit caching strategy

### 2. Unified Interface
- All structures implement same methods
- Easy to swap implementations
- Consistent error reporting

### 3. Composability
- `SerialComposition` works with any AmplitudeSketch
- Mix and match structures in pipelines
- Automatic error propagation

### 4. Maintainability
- Bug fixes in base class benefit all structures
- Easier to add new features
- Better documentation

### 5. Testing
- Base class tests cover common functionality
- Structure-specific tests focus on unique features
- Higher test coverage with less code

---

## Migration Checklist

For each structure (QAM, QHT, Q-Count, Q-HH, Q-LSH):

- [ ] Add `from sim.amplitude_sketch import AmplitudeSketch`
- [ ] Change class declaration: `class X(AmplitudeSketch)`
- [ ] Update `__init__` to call `super().__init__(m, k, theta)`
- [ ] Ensure `insert(x: bytes)` signature matches
- [ ] Ensure `query(y: bytes, shots, noise_level)` signature matches
- [ ] Implement `_build_insert_circuit(x: bytes)`
- [ ] Replace custom hashing with `self._hash_to_indices(x)`
- [ ] Replace custom noise with `self._create_noise_model()`
- [ ] Remove redundant cache/tracking code
- [ ] Update tests to verify base class integration
- [ ] Run tests: `pytest sim/test_X.py -v`
- [ ] Update docstrings to reference amplitude sketching

---

## Backward Compatibility

To maintain backward compatibility during migration:

### Option 1: Gradual Migration
Keep old implementations, add new parallel versions:
```python
# Old (deprecated)
from sim.qam import QAM

# New (recommended)
from sim.qam_v2 import QAM as QAMv2
```

### Option 2: Feature Flags
Add flag to switch between implementations:
```python
class QAM(AmplitudeSketch):
    def __init__(self, m, k, legacy_mode=False):
        if legacy_mode:
            # Use old implementation
        else:
            super().__init__(m, k)
```

### Option 3: Direct Migration
Update all at once with comprehensive testing:
- Run full test suite before changes
- Refactor one structure at a time
- Verify all tests pass after each refactor
- Update experiments and notebooks

---

## Timeline Estimate

**Per structure**: 2-4 hours
- Analysis: 30 min
- Refactoring: 1-2 hours
- Testing: 1 hour
- Documentation: 30 min

**Total for 7 structures**: 14-28 hours (~2-3 days)

---

## Priority Order

Refactor in this order (simplest to most complex):

1. **QAM** - Most straightforward, well-tested
2. **Q-SubSketch** - Similar to QAM
3. **Q-SimHash** - Simple similarity hashing
4. **QHT** - Hierarchical but clean structure
5. **Q-Count** - Different query semantics
6. **Q-HH** - Frequency-weighted phases
7. **Q-LSH** - Vector embeddings (most different)

---

## Success Criteria

Phase 5 refactoring complete when:
- [ ] All 7 structures inherit from `AmplitudeSketch`
- [ ] All existing tests pass unchanged
- [ ] `SerialComposition` works with all structures
- [ ] Code coverage ≥ 90%
- [ ] Documentation updated
- [ ] Benchmark suite runs successfully
- [ ] Paper updated with unified framework description

---

## Next Steps After Refactoring

1. **Phase 6**: Full retrieval system with refactored components
2. **Phase 7**: Compiler optimizations leveraging base class
3. **Phase 8**: Hardware-aware mapping using unified interface
4. **Phase 9**: Benchmark suite with standardized metrics
5. **Phase 10**: DSL design building on amplitude sketch primitives

---

## Example: Complete QAM Refactoring

See `notebooks/amplitude_sketch_migration.ipynb` for interactive walkthrough of QAM refactoring with before/after comparison.
