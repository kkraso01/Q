# Quantum Data Structures (QDS)

Research project exploring quantum alternatives to classical probabilistic data structures.

## Quick Start

```powershell
# Setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run tests
pytest sim/test_qam.py -v

# Explore interactively
jupyter notebook notebooks/qam.ipynb

# Run experiments
python experiments/sweeps.py --m 32 --k 3 --shots 512

# Full parameter sweep
python experiments/sweeps.py --sweep

# Generate plots
python experiments/plotting.py --results results/qam_sweep_*.csv
```

## Repository Structure

- `sim/` - Core quantum algorithms (QAM, Q-SubSketch, Q-SimHash)
- `experiments/` - Parameter sweeps and plotting utilities
- `notebooks/` - Jupyter notebooks for interactive exploration
- `theory/` - Theoretical proofs and bounds
- `results/` - Experimental data (CSV) and plots (PNG)
- `paper/` - Draft papers and reports

## Current Focus: QAM (Quantum Approximate Membership)

"Quantum Bloom Filter" - supports insert(x) and query(x ∈ S?) with tunable false-positive rate.

Key files:
- `sim/qam.py` - QAM implementation
- `notebooks/qam.ipynb` - Interactive experiments
- `sim/test_qam.py` - Unit tests

See `roadmap.md` for full 6-week execution plan.
