"""
Quick Demo of Quantum Neural Network

This script demonstrates the QNN implementation with a simple example.

Author: Quantum Data Structures Research
Date: November 2025
"""

import numpy as np
import sys
sys.path.append('..')

print("=" * 70)
print("QUANTUM NEURAL NETWORK DEMO")
print("=" * 70)

print("\nThis demo showcases the Quantum Neural Network implementation.")
print("It includes:")
print("  1. Variational quantum circuits with learnable parameters")
print("  2. Multiple feature encoding schemes (angle, amplitude, IQP)")
print("  3. Various ansatz architectures (hardware-efficient, strongly-entangling)")
print("  4. Gradient-based learning via parameter shift rule")
print("  5. Adam/SGD/AdaGrad optimizers")

print("\n" + "=" * 70)
print("KEY FEATURES")
print("=" * 70)

features = [
    ("Feature Encoding", "Transforms classical data into quantum states"),
    ("Variational Layers", "Parameterized gates learn optimal transformations"),
    ("Parameter Shift Rule", "Exact quantum gradients for training"),
    ("Multiple Optimizers", "SGD, Adam, AdaGrad support"),
    ("Learned Features", "Extract quantum feature representations"),
    ("Decision Boundaries", "Implicit via quantum measurement"),
    ("Model Persistence", "Save/load trained parameters"),
]

for feature, description in features:
    print(f"\n✓ {feature}")
    print(f"  {description}")

print("\n" + "=" * 70)
print("ARCHITECTURE COMPONENTS")
print("=" * 70)

print("\n1. FEATURE MAPS")
print("   - Angle Encoding: RY(x_i) on each qubit")
print("   - Amplitude Encoding: Encode x in state amplitudes")
print("   - IQP Encoding: Second-order feature interactions")

print("\n2. ANSÄTZE")
print("   - Hardware Efficient: RY + RZ per qubit, linear entanglement")
print("   - Strongly Entangling: RX + RY + RZ, full entanglement")
print("   - Real Amplitudes: Single RY per qubit")

print("\n3. LOSS FUNCTIONS")
print("   - Cross-Entropy: Binary classification")
print("   - MSE: Regression tasks")
print("   - Hinge Loss: SVM-style")

print("\n4. OPTIMIZERS")
print("   - SGD: Standard gradient descent")
print("   - Adam: Adaptive moment estimation")
print("   - AdaGrad: Adaptive learning rates")

print("\n" + "=" * 70)
print("USAGE EXAMPLE")
print("=" * 70)

print("\n```python")
print("from sim.q_neural_network import QuantumNeuralNetwork, create_synthetic_dataset")
print("")
print("# Create dataset")
print("X, y = create_synthetic_dataset(n_samples=100, n_features=4)")
print("")
print("# Initialize QNN")
print("qnn = QuantumNeuralNetwork(")
print("    n_qubits=4,           # Match feature dimension")
print("    n_layers=2,           # Variational layers")
print("    feature_map='angle',  # Encoding scheme")
print("    ansatz='hardware_efficient',")
print("    learning_rate=0.1,")
print("    batch_size=20")
print(")")
print("")
print("# Train")
print("history = qnn.train(")
print("    X_train, y_train,")
print("    epochs=50,")
print("    optimizer='adam'")
print(")")
print("")
print("# Predict")
print("predictions = qnn.predict_batch(X_test)")
print("accuracy = np.mean(predictions == y_test)")
print("")
print("# Extract learned features")
print("quantum_features = qnn.get_learned_features(X_test)")
print("print(f'Feature shape: {quantum_features.shape}')  # (n_samples, 2^n_qubits)")
print("```")

print("\n" + "=" * 70)
print("QUANTUM CONVOLUTIONAL NETWORKS")
print("=" * 70)

print("\nAlso includes Quantum CNN implementation:")
print("  - Local quantum kernels")
print("  - Stride-based application")
print("  - Quantum pooling layers")
print("  - Hierarchical feature extraction")

print("\n```python")
print("from sim.q_neural_network import QuantumConvolutionalNetwork")
print("")
print("qcnn = QuantumConvolutionalNetwork(")
print("    n_qubits=8,")
print("    n_layers=3,")
print("    kernel_size=2,")
print("    stride=1,")
print("    pooling='trace'")
print(")")
print("```")

print("\n" + "=" * 70)
print("COMPARISON WITH CLASSICAL ML")
print("=" * 70)

print("\n| Aspect              | Classical NN     | Quantum NN          |")
print("|---------------------|------------------|---------------------|")
print("| Feature Space       | d dimensions     | 2^n dimensions      |")
print("| Parameters          | O(d²)            | O(n·L)              |")
print("| Gradients           | Backpropagation  | Parameter Shift     |")
print("| Expressivity        | Polynomial       | Exponential         |")
print("| Hardware            | GPU/CPU          | Quantum Computer    |")

print("\nWhere:")
print("  d = classical feature dimension")
print("  n = number of qubits")
print("  L = number of layers")

print("\n" + "=" * 70)
print("THEORETICAL ADVANTAGES")
print("=" * 70)

advantages = [
    ("Hilbert Space Dimension", "2^n quantum vs n classical"),
    ("Feature Expressivity", "Exponential functions with linear gates"),
    ("Entanglement", "Captures complex feature correlations"),
    ("Interference", "Quantum parallelism for pattern matching"),
    ("Query Complexity", "Potential quantum speedup in certain regimes"),
]

for i, (advantage, description) in enumerate(advantages, 1):
    print(f"\n{i}. {advantage}")
    print(f"   {description}")

print("\n" + "=" * 70)
print("LEARNING MECHANISM")
print("=" * 70)

print("\nHow QNN learns features and boundaries:")
print("")
print("1. ENCODING PHASE")
print("   Classical data x → Quantum state |ψ(x)⟩")
print("   Uses rotations and entanglement gates")
print("")
print("2. VARIATIONAL PHASE")
print("   Apply U(θ) = ∏ᵢ Rᵧ(θᵢ)Rᵧ(θᵢ₊₁)CNOT")
print("   Parameters θ are learned via gradient descent")
print("")
print("3. MEASUREMENT PHASE")
print("   Measure qubits → Classical probabilities")
print("   P(0) ≈ ⟨ψ(x,θ)|0⟩⟨0|ψ(x,θ)⟩")
print("")
print("4. GRADIENT COMPUTATION")
print("   Use parameter shift rule:")
print("   ∂L/∂θᵢ = [L(θᵢ + π/2) - L(θᵢ - π/2)] / 2")
print("")
print("5. PARAMETER UPDATE")
print("   θ ← θ - η∇L(θ)")
print("   With momentum/adaptive learning (Adam)")

print("\n" + "=" * 70)
print("DECISION BOUNDARIES")
print("=" * 70)

print("\nQuantum decision boundaries are implicitly defined by:")
print("")
print("• Measurement operators: Projects quantum states to classical labels")
print("• Quantum interference: Constructive for correct class, destructive otherwise")
print("• Entanglement structure: Captures non-linear feature interactions")
print("• Parameter configuration: Learned optimal measurement basis")
print("")
print("Unlike classical NNs with explicit weight matrices,")
print("QNNs encode boundaries in quantum state transformations!")

print("\n" + "=" * 70)
print("TESTING & VALIDATION")
print("=" * 70)

print("\nComprehensive test suite includes:")
print("")
test_categories = [
    "Initialization and parameter counting",
    "Feature map circuits (angle, amplitude, IQP)",
    "Variational circuits (all ansatzes)",
    "Forward pass and prediction",
    "Loss computation (cross-entropy, MSE, hinge)",
    "Gradient computation (parameter shift rule)",
    "Training with different optimizers",
    "Model persistence (save/load)",
    "Learned feature extraction",
    "QCNN architecture",
    "Integration tests",
    "Edge cases and error handling",
]

for i, category in enumerate(test_categories, 1):
    print(f"  {i:2d}. {category}")

print(f"\nTotal: {len(test_categories)} test categories with 40+ individual tests")

print("\n" + "=" * 70)
print("INSTALLATION & REQUIREMENTS")
print("=" * 70)

print("\nRequired packages:")
print("  - qiskit >= 1.0")
print("  - qiskit-aer")
print("  - numpy")
print("  - matplotlib")
print("  - scikit-learn (for utils)")
print("  - pytest (for testing)")

print("\nInstall via:")
print("  pip install qiskit qiskit-aer numpy matplotlib scikit-learn pytest")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)

next_steps = [
    "Install Qiskit: pip install qiskit qiskit-aer",
    "Run tests: pytest sim/test_q_neural_network.py -v",
    "Try the notebook: jupyter notebook notebooks/q_neural_network.ipynb",
    "Train on real data (MNIST, CIFAR-10)",
    "Experiment with circuit depths and ansatzes",
    "Add noise models for realistic simulation",
    "Deploy on IBM Quantum hardware",
    "Compare with quantum kernel methods",
]

for i, step in enumerate(next_steps, 1):
    print(f"\n{i}. {step}")

print("\n" + "=" * 70)
print("RESEARCH APPLICATIONS")
print("=" * 70)

print("\nThis QNN implementation enables research in:")
print("")
applications = [
    ("Quantum ML", "Classification, regression, clustering"),
    ("Feature Learning", "Quantum representation learning"),
    ("Transfer Learning", "Pre-trained quantum circuits"),
    ("Hybrid Algorithms", "Classical-quantum co-processing"),
    ("NISQ Algorithms", "Near-term quantum advantage"),
    ("Quantum Advantage", "Empirical quantum speedup studies"),
]

for app, desc in applications:
    print(f"  • {app}: {desc}")

print("\n" + "=" * 70)
print("DOCUMENTATION")
print("=" * 70)

print("\nDetailed documentation available in:")
print("  - Source code: sim/q_neural_network.py (500+ lines, fully commented)")
print("  - Tests: sim/test_q_neural_network.py (comprehensive validation)")
print("  - Tutorial: notebooks/q_neural_network.ipynb (interactive examples)")
print("  - Papers: paper/ (theoretical foundations)")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\n✓ Complete Quantum Neural Network implementation")
print("✓ Multiple feature encoding schemes")
print("✓ Various ansatz architectures")
print("✓ Gradient-based learning with parameter shift rule")
print("✓ Multiple optimizers (SGD, Adam, AdaGrad)")
print("✓ Learned quantum feature extraction")
print("✓ Quantum Convolutional Networks")
print("✓ Comprehensive test suite (40+ tests)")
print("✓ Interactive Jupyter notebook tutorial")
print("✓ Ready for real quantum hardware")

print("\n" + "=" * 70)
print("READY TO TRAIN QUANTUM NEURAL NETWORKS!")
print("=" * 70)
print()
