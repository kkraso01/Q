# Quantum Neural Networks (QNN)

A complete implementation of variational quantum neural networks for classification, regression, and feature learning.

## Overview

This module implements **Quantum Neural Networks** that learn optimal feature representations and decision boundaries through gradient-based optimization using quantum circuits.

### Key Features

✅ **Parameterized Quantum Circuits (PQC)** - Variational quantum circuits with learnable parameters  
✅ **Multiple Feature Maps** - Angle encoding, amplitude encoding, IQP-style encoding  
✅ **Various Ansätze** - Hardware-efficient, strongly-entangling, real amplitudes  
✅ **Gradient-Based Learning** - Parameter shift rule for exact quantum gradients  
✅ **Multiple Optimizers** - SGD, Adam, AdaGrad support  
✅ **Quantum Feature Extraction** - Extract learned 2^n dimensional quantum features  
✅ **Quantum CNNs** - Convolutional architecture with local kernels  
✅ **Model Persistence** - Save and load trained models  

## Architecture

```
Classical Data (x)
       ↓
Feature Map: |ψ(x)⟩ = U_φ(x)|0⟩
       ↓
Variational Layers: U(θ₁)...U(θ_L)|ψ(x)⟩
       ↓
Measurement: P(y|x) = ⟨ψ(x,θ)|M_y|ψ(x,θ)⟩
       ↓
Classical Prediction
```

### Components

1. **Feature Map** - Encodes classical data into quantum states
   - **Angle Encoding**: `RY(x_i)` on each qubit
   - **Amplitude Encoding**: Encode data in state amplitudes
   - **IQP Encoding**: Second-order feature interactions

2. **Variational Circuit** - Parameterized gates learn transformations
   - **Hardware Efficient**: RY + RZ per qubit, linear entanglement
   - **Strongly Entangling**: RX + RY + RZ, full entanglement
   - **Real Amplitudes**: Single RY per qubit

3. **Measurement** - Extract predictions from quantum states
   - Binary classification via probability of |0⟩ state
   - Multi-class via multiple measurements

## Installation

```bash
pip install qiskit qiskit-aer numpy matplotlib scikit-learn pytest
```

Or install all project requirements:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from sim.q_neural_network import QuantumNeuralNetwork, create_synthetic_dataset

# Create dataset
X, y = create_synthetic_dataset(n_samples=100, n_features=4)

# Initialize QNN
qnn = QuantumNeuralNetwork(
    n_qubits=4,              # Match feature dimension
    n_layers=2,              # Variational layers
    feature_map='angle',     # Encoding scheme
    ansatz='hardware_efficient',
    learning_rate=0.1,
    batch_size=20,
    shots=1024
)

# Train
history = qnn.train(
    X_train, y_train,
    X_val=X_test, y_val=y_test,
    epochs=50,
    optimizer='adam',
    verbose=True
)

# Predict
predictions = qnn.predict_batch(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Extract learned quantum features
quantum_features = qnn.get_learned_features(X_test)
print(f"Feature shape: {quantum_features.shape}")  # (n_samples, 2^n_qubits)

# Save model
qnn.save_model("trained_qnn.npz")
```

## Advanced Usage

### Different Feature Maps

```python
# Angle encoding (default)
qnn_angle = QuantumNeuralNetwork(n_qubits=4, feature_map='angle')

# Amplitude encoding (exponential feature space)
qnn_amp = QuantumNeuralNetwork(n_qubits=4, feature_map='amplitude')

# IQP encoding (second-order interactions)
qnn_iqp = QuantumNeuralNetwork(n_qubits=4, feature_map='iqp')
```

### Different Ansätze

```python
# Hardware-efficient (fewer gates)
qnn_he = QuantumNeuralNetwork(n_qubits=4, ansatz='hardware_efficient')

# Strongly entangling (more expressive)
qnn_se = QuantumNeuralNetwork(n_qubits=4, ansatz='strongly_entangling')

# Real amplitudes (real-valued amplitudes only)
qnn_ra = QuantumNeuralNetwork(n_qubits=4, ansatz='real_amplitudes')
```

### Different Optimizers

```python
# Adam (recommended)
history_adam = qnn.train(X, y, epochs=50, optimizer='adam')

# SGD
history_sgd = qnn.train(X, y, epochs=50, optimizer='sgd')

# AdaGrad
history_adagrad = qnn.train(X, y, epochs=50, optimizer='adagrad')
```

### Loss Functions

```python
# Cross-entropy (binary classification)
history = qnn.train(X, y, loss_type='cross_entropy')

# Mean squared error (regression)
history = qnn.train(X, y, loss_type='mse')

# Hinge loss (SVM-style)
history = qnn.train(X, y, loss_type='hinge')
```

### Noise Simulation

```python
from qiskit_aer.noise import NoiseModel, depolarizing_error

# Create noise model
noise_model = NoiseModel()
error = depolarizing_error(0.01, 1)  # 1% error rate
noise_model.add_all_qubit_quantum_error(error, ['rx', 'ry', 'rz'])

# Initialize QNN with noise
qnn = QuantumNeuralNetwork(
    n_qubits=4,
    n_layers=2,
    noise_model=noise_model
)
```

## Quantum Convolutional Networks

```python
from sim.q_neural_network import QuantumConvolutionalNetwork

# Initialize QCNN
qcnn = QuantumConvolutionalNetwork(
    n_qubits=8,
    n_layers=3,
    kernel_size=2,
    stride=1,
    pooling='trace'
)

# Forward pass
qc = qcnn.forward(x)
```

## How It Works

### 1. Feature Encoding

Classical data is encoded into quantum states using rotation gates:

```
|0⟩ --RY(x₁)-- ... --CNOT--
|0⟩ --RY(x₂)-- ... --CNOT--
|0⟩ --RY(x₃)-- ... --CNOT--
|0⟩ --RY(x₄)-- ... --CNOT--
```

### 2. Variational Transformation

Parameterized gates learn optimal transformations:

```
For each layer l:
  For each qubit i:
    RY(θ_i^l) --
    RZ(φ_i^l) --
  Entanglement layer (CNOTs)
```

### 3. Measurement

Quantum state is measured to obtain class probabilities:

```
P(y=1|x) = ⟨ψ(x,θ)|0...0⟩⟨0...0|ψ(x,θ)⟩
```

### 4. Gradient Computation

Parameter shift rule computes exact gradients:

```
∂L/∂θᵢ = [L(θᵢ + π/2) - L(θᵢ - π/2)] / 2
```

No approximations needed - this is an exact quantum gradient!

### 5. Parameter Update

Standard gradient descent with momentum:

```
θ ← θ - η∇L(θ)  (SGD)
θ ← θ - η(m / √v)  (Adam)
```

## Learning Features

The QNN learns to:

1. **Transform input space** - Map data to exponentially large 2^n dimensional Hilbert space
2. **Create decision boundaries** - Learn measurement basis that separates classes
3. **Capture correlations** - Entanglement encodes feature interactions
4. **Optimize interference** - Constructive interference for correct class

Unlike k-NN (non-parametric), QNN has learnable parameters that optimize feature representations!

## Decision Boundaries

Quantum decision boundaries are **implicitly** defined by:

- **Measurement operators** - Projects quantum states to classical labels
- **Quantum interference** - Constructive for correct class, destructive otherwise
- **Entanglement structure** - Captures non-linear feature interactions
- **Parameter configuration** - Learned optimal measurement basis

The boundary is not a hyperplane but a complex surface in exponential Hilbert space!

## Comparison: QNN vs Classical NN

| Aspect              | Classical NN     | Quantum NN          |
|---------------------|------------------|---------------------|
| Feature Space       | d dimensions     | 2^n dimensions      |
| Parameters          | O(d²)            | O(n·L)              |
| Gradients           | Backpropagation  | Parameter Shift     |
| Expressivity        | Polynomial       | Exponential         |
| Hardware            | GPU/CPU          | Quantum Computer    |
| Training            | Fast             | Slower (shots)      |
| Inference           | Fast             | Requires shots      |

Where:
- d = classical feature dimension
- n = number of qubits
- L = number of layers

## Theoretical Advantages

1. **Hilbert Space Dimension**: 2^n quantum vs n classical
2. **Feature Expressivity**: Exponential functions with linear gates
3. **Entanglement**: Captures complex feature correlations
4. **Interference**: Quantum parallelism for pattern matching
5. **Query Complexity**: Potential quantum speedup in certain regimes

## Files

- `sim/q_neural_network.py` - Main implementation (600+ lines)
- `sim/test_q_neural_network.py` - Comprehensive tests (40+ tests)
- `notebooks/q_neural_network.ipynb` - Interactive tutorial
- `experiments/qnn_demo.py` - Feature demonstration

## Testing

Run the comprehensive test suite:

```bash
# All tests
pytest sim/test_q_neural_network.py -v

# Specific test class
pytest sim/test_q_neural_network.py::TestQuantumNeuralNetwork -v

# Specific test
pytest sim/test_q_neural_network.py::TestQuantumNeuralNetwork::test_training_adam -v
```

Test coverage includes:
- ✅ Initialization and parameter counting
- ✅ Feature map circuits (all types)
- ✅ Variational circuits (all ansatzes)
- ✅ Forward pass and prediction
- ✅ Loss computation (all types)
- ✅ Gradient computation (parameter shift rule)
- ✅ Training with different optimizers
- ✅ Model persistence (save/load)
- ✅ Learned feature extraction
- ✅ QCNN architecture
- ✅ Integration tests
- ✅ Edge cases and error handling

## Tutorial Notebook

Interactive Jupyter notebook with:
1. Synthetic dataset creation
2. QNN initialization and architecture visualization
3. Training with progress tracking
4. Performance evaluation and confusion matrix
5. Learned quantum feature visualization (t-SNE)
6. Comparison with classical neural networks
7. Experimentation with different architectures
8. Model persistence

Run the notebook:

```bash
jupyter notebook notebooks/q_neural_network.ipynb
```

## Research Applications

This QNN implementation enables research in:

- **Quantum ML** - Classification, regression, clustering
- **Feature Learning** - Quantum representation learning
- **Transfer Learning** - Pre-trained quantum circuits
- **Hybrid Algorithms** - Classical-quantum co-processing
- **NISQ Algorithms** - Near-term quantum advantage
- **Quantum Advantage** - Empirical quantum speedup studies

## Real-World Applications

### Image Classification
```python
# MNIST digit recognition
qnn = QuantumNeuralNetwork(n_qubits=8, n_layers=3)
qnn.train(mnist_train_features, mnist_train_labels, epochs=100)
```

### Financial Prediction
```python
# Stock price movement prediction
qnn = QuantumNeuralNetwork(n_qubits=6, n_layers=2)
qnn.train(market_features, price_direction, optimizer='adam')
```

### Medical Diagnosis
```python
# Disease classification from biomarkers
qnn = QuantumNeuralNetwork(n_qubits=10, n_layers=4)
qnn.train(patient_data, diagnosis_labels, loss_type='cross_entropy')
```

## Performance Tips

1. **Match qubits to features** - Use n_qubits ≥ log₂(n_features) for amplitude encoding
2. **Start with few layers** - 2-3 layers often sufficient, prevents overfitting
3. **Use Adam optimizer** - Better convergence than SGD
4. **Increase shots gradually** - Start with 512, increase to 2048+ for final training
5. **Batch training** - Use mini-batches to speed up gradient computation
6. **Feature normalization** - Normalize inputs to [0, 1] or [-1, 1]
7. **Learning rate** - Start with 0.1, decrease if unstable

## Limitations

1. **Scalability** - Limited by number of qubits (NISQ era: ~100 qubits)
2. **Shot noise** - Requires many measurements for accurate gradients
3. **Training time** - Slower than classical NNs due to parameter shift rule
4. **Barren plateaus** - Deep circuits may have vanishing gradients
5. **Hardware access** - Real quantum computers have limited availability

## Future Directions

- [ ] Multi-class classification (one-vs-rest, softmax)
- [ ] Regression support (continuous outputs)
- [ ] Quantum batch normalization
- [ ] Dropout for regularization
- [ ] Pre-trained models (quantum transfer learning)
- [ ] Hardware-aware compilation
- [ ] Gradient-free optimization (SPSA, genetic algorithms)
- [ ] Quantum GAN integration

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{quantum_neural_networks,
  title = {Quantum Neural Networks: A Complete Implementation},
  author = {Quantum Data Structures Research},
  year = {2025},
  url = {https://github.com/kkraso01/Q}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues, questions, or discussions:
- Open an issue on GitHub
- Email: quantum-research@example.com
- Join our Discord: [link]

## Acknowledgments

Built on top of:
- Qiskit - IBM Quantum framework
- NumPy - Numerical computing
- Scikit-learn - Classical ML utilities

Inspired by:
- Variational Quantum Eigensolver (VQE)
- Quantum Approximate Optimization Algorithm (QAOA)
- Recent quantum machine learning research

---

**Ready to train quantum neural networks and explore quantum advantage in machine learning!** 🚀⚛️
