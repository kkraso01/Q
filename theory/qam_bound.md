# QAM False-Positive Bound (Preliminary)

## Model

- **m** qubits as memory
- **k** hash functions (independent, uniform)
- **θ** phase rotation angle per insertion
- **|S|** items inserted
- **ρ = |S|/m** load factor

## Conjecture (To Prove)

For an item **x ∉ S**, the false-positive rate α satisfies:

```
α ≤ exp(-C · k · (1 - ρ))
```

where **C(θ)** is a constant depending on rotation angle θ.

## Intuition

1. Each hash maps to a random qubit index
2. Phase accumulation reduces overlap with |0⟩^⊗m
3. Higher k → more "destructive interference" for non-members
4. Higher ρ → more collisions → degrades performance

## Next Steps

1. Formalize probability calculation using statevector analysis
2. Derive C(θ) explicitly for θ = π/4
3. Add first-order noise correction: α ≤ α₀ + O(kε)
4. Validate empirically with `qam.ipynb` experiments

## References

- Classical Bloom filter bound: (1 - e^(-k·|S|/m))^k
- Target: Show quantum achieves comparable or better trade-off in certain regimes
