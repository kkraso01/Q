# QAM Deletion: Limitations and Empirical Results

## Theoretical Limitation

In phase-based QAM, deletion is implemented by applying the inverse phase rotation (Rz(-θ)) to the hashed qubits of the deleted item. In theory, this should cancel the effect of insertion. However, due to quantum interference and hash collisions, the expectation value for a deleted item often remains close to that of a present item, not an absent one. This is especially true when multiple items share hash indices or when θ is not a multiple of 2π.

## Empirical Results

Extensive simulation sweeps (see `sim/test_qam_deletion_sweep.py`) show that, across a range of θ, m, and k, the expectation for deleted items is not reliably suppressed. In most cases, deleted items are not distinguishable from present items by the acceptance threshold.

## Research Implication

This limitation is fundamental to phase-based QAM and should be reported honestly. It motivates further research into quantum data structures that support robust deletion, possibly using more sophisticated encoding or error-correcting strategies.

## Recommendation

- Report this limitation in the paper and appendices.
- Include empirical plots showing the actual behavior of deletion.
- Frame as an open problem for future quantum data structure research.
