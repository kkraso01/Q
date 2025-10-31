# Quantum Cell Probe Model

We formalize the Quantum Cell Probe Model as follows:

- **Memory:** An array of $m$ logical qubits, each addressable by quantum operations.
- **Query:** A phase-based fingerprinting operation, where $k$ hash functions select qubits and apply $R_z(\theta)$ rotations.
- **Measurement:** The final state is measured, returning an acceptance bit indicating membership or similarity.
- **No-Cloning:** The model respects quantum no-cloning; queries are destructive.
- **Cost Metrics:**
  - Query time: circuit depth
  - Space: number of logical qubits
  - Accuracy: false positive/negative rates
  - Shot budget: number of measurements per query

This model generalizes the classical cell probe model to the quantum setting, capturing the essential features of quantum data structures for membership and similarity search.
