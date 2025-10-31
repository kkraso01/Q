# Reproducibility Checklist for QDS Phase 2 Results

## 1. Environment Setup
- [ ] Python 3.11 is active
- [ ] Qiskit >= 1.0 installed
- [ ] All dependencies in requirements.txt installed
- [ ] Use a clean virtual environment (recommended)

## 2. Figure Generation
- [ ] Run: `python experiments/generate_all_figures.py`
- [ ] Confirm 8+ PNG files in `results/`:
    - accuracy_vs_memory.png
    - accuracy_vs_shots.png
    - accuracy_vs_noise.png
    - accuracy_vs_load_factor.png
    - batch_query_error_vs_amortized_cost.png
    - heatmap_shots_noise.png
    - topology_comparison.png
    - q_subsketch_auc.png
- [ ] Check for corresponding CSVs if generated

## 3. Determinism
- [ ] Re-run `python experiments/generate_all_figures.py` and confirm figures are identical
- [ ] Check that all random seeds are set in code (NumPy, random)

## 4. Dependencies
- [ ] No missing package errors on run
- [ ] Qiskit version matches requirements
- [ ] All matplotlib, numpy, scipy, pytest versions compatible

## 5. Testing
- [ ] Run: `pytest sim/ -v`
- [ ] All tests pass

## 6. Documentation
- [ ] All figures are described in `paper/draft.md` or `draft.tex`
- [ ] Any issues or discrepancies are documented here

---

**If any box is unchecked, document the issue and resolve before submission.**
