"""
Quantum Neural Networks (QNN) for Classification and Regression

This module implements variational quantum circuits that learn optimal feature
representations and decision boundaries through gradient-based optimization.

Key Features:
- Parameterized quantum circuits (PQC)
- Gradient-based learning (parameter shift rule)
- Multiple ansatz architectures (hardware-efficient, strongly-entangling)
- Classical-quantum hybrid training
- Support for classification and regression

Author: Quantum Data Structures Research
Date: November 2025
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.primitives import Sampler, Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import logging

logger = logging.getLogger(__name__)


class QuantumNeuralNetwork:
    """
    Variational Quantum Neural Network with learnable parameters.
    
    Architecture:
    1. Feature map: Encode classical data into quantum states
    2. Variational layers: Parameterized gates that learn representations
    3. Measurement: Extract classical predictions from quantum states
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        feature_map: str = "angle",
        ansatz: str = "hardware_efficient",
        learning_rate: float = 0.01,
        batch_size: int = 32,
        shots: int = 1024,
        noise_model: Optional[NoiseModel] = None
    ):
        """
        Initialize Quantum Neural Network.
        
        Args:
            n_qubits: Number of qubits (should match feature dimension)
            n_layers: Number of variational layers
            feature_map: Type of feature encoding ("angle", "amplitude", "iqp")
            ansatz: Variational form ("hardware_efficient", "strongly_entangling", "real_amplitudes")
            learning_rate: Learning rate for parameter updates
            batch_size: Mini-batch size for training
            shots: Number of measurement shots per circuit
            noise_model: Optional noise model for realistic simulation
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_map_type = feature_map
        self.ansatz_type = ansatz
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.shots = shots
        self.noise_model = noise_model
        
        # Initialize parameters
        self.n_params = self._calculate_n_params()
        self.params = np.random.uniform(0, 2*np.pi, self.n_params)
        
        # Training history
        self.loss_history = []
        self.accuracy_history = []
        
        # Create backend
        if noise_model:
            self.backend = AerSimulator(noise_model=noise_model)
        else:
            self.backend = AerSimulator()
        
        logger.info(f"Initialized QNN: {n_qubits} qubits, {n_layers} layers, "
                   f"{self.n_params} parameters")
    
    def _calculate_n_params(self) -> int:
        """Calculate number of trainable parameters based on ansatz."""
        if self.ansatz_type == "hardware_efficient":
            # Each layer: n_qubits rotations + n_qubits-1 CNOTs
            # Rotations: RY and RZ per qubit = 2 * n_qubits per layer
            return 2 * self.n_qubits * self.n_layers
        elif self.ansatz_type == "strongly_entangling":
            # Each layer: 3 rotations per qubit + full entanglement
            return 3 * self.n_qubits * self.n_layers
        elif self.ansatz_type == "real_amplitudes":
            # Each layer: 1 rotation per qubit
            return self.n_qubits * self.n_layers
        else:
            raise ValueError(f"Unknown ansatz: {self.ansatz_type}")
    
    def _feature_map_circuit(self, x: np.ndarray) -> QuantumCircuit:
        """
        Create feature map circuit to encode classical data.
        
        Args:
            x: Input features (shape: [n_features])
            
        Returns:
            Quantum circuit encoding x
        """
        qc = QuantumCircuit(self.n_qubits)
        
        # Normalize input to [0, 2π]
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8) * 2 * np.pi
        
        if self.feature_map_type == "angle":
            # Angle encoding: RY(x_i) on each qubit
            for i in range(min(len(x_norm), self.n_qubits)):
                qc.ry(x_norm[i], i)
            
            # Add entanglement for feature interactions
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
        
        elif self.feature_map_type == "amplitude":
            # Amplitude encoding: encode x in amplitudes
            # Normalize to unit vector
            x_normalized = x / (np.linalg.norm(x) + 1e-8)
            # Pad to 2^n_qubits
            padding = 2**self.n_qubits - len(x_normalized)
            if padding > 0:
                x_padded = np.concatenate([x_normalized, np.zeros(padding)])
            else:
                x_padded = x_normalized[:2**self.n_qubits]
            qc.initialize(x_padded, range(self.n_qubits))
        
        elif self.feature_map_type == "iqp":
            # IQP-style feature map with second-order interactions
            # First layer: Hadamards
            qc.h(range(self.n_qubits))
            
            # Diagonal encoding
            for i in range(min(len(x_norm), self.n_qubits)):
                qc.p(x_norm[i], i)
            
            # Second-order interactions
            for i in range(self.n_qubits - 1):
                for j in range(i + 1, self.n_qubits):
                    if i < len(x_norm) and j < len(x_norm):
                        qc.cp(x_norm[i] * x_norm[j], i, j)
        
        return qc
    
    def _variational_circuit(self, params: np.ndarray) -> QuantumCircuit:
        """
        Create variational circuit with trainable parameters.
        
        Args:
            params: Trainable parameters
            
        Returns:
            Parameterized quantum circuit
        """
        qc = QuantumCircuit(self.n_qubits)
        param_idx = 0
        
        if self.ansatz_type == "hardware_efficient":
            for layer in range(self.n_layers):
                # Rotation layer
                for i in range(self.n_qubits):
                    qc.ry(params[param_idx], i)
                    param_idx += 1
                    qc.rz(params[param_idx], i)
                    param_idx += 1
                
                # Entanglement layer
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i + 1)
                # Wrap-around
                if self.n_qubits > 2:
                    qc.cx(self.n_qubits - 1, 0)
        
        elif self.ansatz_type == "strongly_entangling":
            for layer in range(self.n_layers):
                # Three rotation axes per qubit
                for i in range(self.n_qubits):
                    qc.rx(params[param_idx], i)
                    param_idx += 1
                    qc.ry(params[param_idx], i)
                    param_idx += 1
                    qc.rz(params[param_idx], i)
                    param_idx += 1
                
                # Full entanglement
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        qc.cx(i, j)
        
        elif self.ansatz_type == "real_amplitudes":
            for layer in range(self.n_layers):
                # Single rotation per qubit
                for i in range(self.n_qubits):
                    qc.ry(params[param_idx], i)
                    param_idx += 1
                
                # Linear entanglement
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i + 1)
        
        return qc
    
    def forward(self, x: np.ndarray, params: Optional[np.ndarray] = None) -> QuantumCircuit:
        """
        Forward pass: feature map + variational circuit.
        
        Args:
            x: Input features
            params: Parameters (uses self.params if None)
            
        Returns:
            Complete quantum circuit
        """
        if params is None:
            params = self.params
        
        # Build circuit
        qc = self._feature_map_circuit(x)
        qc = qc.compose(self._variational_circuit(params))
        
        return qc
    
    def predict_proba(self, x: np.ndarray, params: Optional[np.ndarray] = None) -> float:
        """
        Predict probability for binary classification.
        
        Args:
            x: Input features
            params: Parameters (uses self.params if None)
            
        Returns:
            Probability of class 1 (P(y=1|x))
        """
        qc = self.forward(x, params)
        qc.measure_all()
        
        # Execute circuit
        job = self.backend.run(qc, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        
        # Compute probability of measuring |0...0⟩ (class 1)
        zero_state = '0' * self.n_qubits
        prob_1 = counts.get(zero_state, 0) / self.shots
        
        return prob_1
    
    def predict(self, x: np.ndarray, params: Optional[np.ndarray] = None) -> int:
        """
        Predict class label (0 or 1).
        
        Args:
            x: Input features
            params: Parameters
            
        Returns:
            Predicted class (0 or 1)
        """
        prob = self.predict_proba(x, params)
        return 1 if prob > 0.5 else 0
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for batch of inputs.
        
        Args:
            X: Input features (shape: [n_samples, n_features])
            
        Returns:
            Predicted classes (shape: [n_samples])
        """
        predictions = []
        for x in X:
            predictions.append(self.predict(x))
        return np.array(predictions)
    
    def compute_loss(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        params: Optional[np.ndarray] = None,
        loss_type: str = "cross_entropy"
    ) -> float:
        """
        Compute loss function.
        
        Args:
            X: Input features (shape: [n_samples, n_features])
            y: Target labels (shape: [n_samples])
            params: Parameters
            loss_type: Type of loss ("cross_entropy", "mse", "hinge")
            
        Returns:
            Loss value
        """
        if params is None:
            params = self.params
        
        loss = 0.0
        
        for xi, yi in zip(X, y):
            prob = self.predict_proba(xi, params)
            
            if loss_type == "cross_entropy":
                # Binary cross-entropy
                prob = np.clip(prob, 1e-8, 1 - 1e-8)  # Numerical stability
                loss += -yi * np.log(prob) - (1 - yi) * np.log(1 - prob)
            
            elif loss_type == "mse":
                # Mean squared error
                loss += (prob - yi) ** 2
            
            elif loss_type == "hinge":
                # Hinge loss for SVM-style
                margin = yi * (2 * prob - 1)  # Convert prob to [-1, 1]
                loss += max(0, 1 - margin)
        
        return loss / len(X)
    
    def compute_gradient(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_idx: int,
        shift: float = np.pi / 2
    ) -> float:
        """
        Compute gradient using parameter shift rule.
        
        The parameter shift rule states:
        ∂L/∂θᵢ = [L(θᵢ + π/2) - L(θᵢ - π/2)] / 2
        
        Args:
            X: Input features
            y: Target labels
            param_idx: Index of parameter to differentiate
            shift: Shift amount (π/2 for standard gates)
            
        Returns:
            Gradient ∂L/∂θᵢ
        """
        # Shift parameter up
        params_plus = self.params.copy()
        params_plus[param_idx] += shift
        loss_plus = self.compute_loss(X, y, params_plus)
        
        # Shift parameter down
        params_minus = self.params.copy()
        params_minus[param_idx] -= shift
        loss_minus = self.compute_loss(X, y, params_minus)
        
        # Gradient
        gradient = (loss_plus - loss_minus) / 2
        
        return gradient
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        loss_type: str = "cross_entropy",
        optimizer: str = "adam",
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the quantum neural network.
        
        Args:
            X_train: Training features (shape: [n_samples, n_features])
            y_train: Training labels (shape: [n_samples])
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            loss_type: Loss function type
            optimizer: Optimizer ("sgd", "adam", "adagrad")
            verbose: Print training progress
            
        Returns:
            Dictionary with training history
        """
        n_samples = len(X_train)
        
        # Initialize optimizer state
        if optimizer == "adam":
            m = np.zeros(self.n_params)  # First moment
            v = np.zeros(self.n_params)  # Second moment
            beta1, beta2 = 0.9, 0.999
            epsilon = 1e-8
        elif optimizer == "adagrad":
            G = np.zeros(self.n_params)  # Accumulated squared gradients
            epsilon = 1e-8
        
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                
                # Compute gradients for all parameters
                gradients = np.zeros(self.n_params)
                for param_idx in range(self.n_params):
                    gradients[param_idx] = self.compute_gradient(
                        X_batch, y_batch, param_idx
                    )
                
                # Update parameters based on optimizer
                if optimizer == "sgd":
                    self.params -= self.learning_rate * gradients
                
                elif optimizer == "adam":
                    # Adam optimizer
                    t = epoch * (n_samples // self.batch_size) + i // self.batch_size + 1
                    m = beta1 * m + (1 - beta1) * gradients
                    v = beta2 * v + (1 - beta2) * gradients**2
                    m_hat = m / (1 - beta1**t)
                    v_hat = v / (1 - beta2**t)
                    self.params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                
                elif optimizer == "adagrad":
                    # AdaGrad optimizer
                    G += gradients**2
                    self.params -= self.learning_rate * gradients / (np.sqrt(G) + epsilon)
            
            # Compute metrics
            train_loss = self.compute_loss(X_train, y_train, loss_type=loss_type)
            train_predictions = self.predict_batch(X_train)
            train_accuracy = np.mean(train_predictions == y_train)
            
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_accuracy)
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_loss = self.compute_loss(X_val, y_val, loss_type=loss_type)
                val_predictions = self.predict_batch(X_val)
                val_accuracy = np.mean(val_predictions == y_val)
                
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
            
            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                msg = f"Epoch {epoch+1}/{epochs}: "
                msg += f"Loss={train_loss:.4f}, Acc={train_accuracy:.4f}"
                if X_val is not None:
                    msg += f", Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}"
                logger.info(msg)
                print(msg)
        
        self.loss_history = history['train_loss']
        self.accuracy_history = history['train_accuracy']
        
        return history
    
    def get_learned_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract learned feature representations from quantum state.
        
        Args:
            X: Input features (shape: [n_samples, n_features])
            
        Returns:
            Quantum feature representations (shape: [n_samples, 2^n_qubits])
        """
        features = []
        
        for x in X:
            qc = self.forward(x)
            
            # Get statevector (works for small number of qubits)
            if self.n_qubits <= 10:
                from qiskit.quantum_info import Statevector
                state = Statevector(qc)
                # Use amplitudes as features
                features.append(np.abs(state.data))
            else:
                # For larger circuits, use measurement statistics
                qc.measure_all()
                job = self.backend.run(qc, shots=self.shots)
                counts = job.result().get_counts()
                
                # Convert counts to probability distribution
                probs = np.zeros(2**self.n_qubits)
                for bitstring, count in counts.items():
                    idx = int(bitstring, 2)
                    probs[idx] = count / self.shots
                features.append(probs)
        
        return np.array(features)
    
    def visualize_decision_boundary(
        self,
        X: np.ndarray,
        y: np.ndarray,
        resolution: int = 100,
        save_path: Optional[str] = None
    ):
        """
        Visualize decision boundary (for 2D data only).
        
        Args:
            X: Input features (shape: [n_samples, 2])
            y: Target labels
            resolution: Grid resolution for boundary
            save_path: Path to save figure
        """
        if X.shape[1] != 2:
            raise ValueError("Visualization only works for 2D data")
        
        import matplotlib.pyplot as plt
        
        # Create grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        # Predict on grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict_batch(grid_points)
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3, levels=1, cmap='RdBu')
        plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Class 0', edgecolors='k')
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Class 1', edgecolors='k')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Quantum Neural Network Decision Boundary')
        plt.legend()
        plt.colorbar(label='Predicted Class')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath: str):
        """Save trained model parameters."""
        np.savez(
            filepath,
            params=self.params,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            feature_map=self.feature_map_type,
            ansatz=self.ansatz_type,
            loss_history=self.loss_history,
            accuracy_history=self.accuracy_history
        )
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model parameters."""
        data = np.load(filepath)
        self.params = data['params']
        self.loss_history = data['loss_history'].tolist()
        self.accuracy_history = data['accuracy_history'].tolist()
        logger.info(f"Model loaded from {filepath}")


class QuantumConvolutionalNetwork:
    """
    Quantum Convolutional Neural Network with local connectivity.
    
    Applies convolutional-style operations on quantum states.
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        kernel_size: int = 2,
        stride: int = 1,
        pooling: str = "max"
    ):
        """
        Initialize Quantum CNN.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of convolutional layers
            kernel_size: Size of quantum kernel
            stride: Stride for kernel application
            pooling: Pooling strategy ("max", "average", "trace")
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.pooling = pooling
        
        # Calculate number of parameters
        # Each kernel has 2 * kernel_size parameters (RY, RZ per qubit)
        n_kernels = (n_qubits - kernel_size) // stride + 1
        self.n_params = 2 * kernel_size * n_kernels * n_layers
        self.params = np.random.uniform(0, 2*np.pi, self.n_params)
        
        logger.info(f"Initialized QCNN: {n_qubits} qubits, {n_layers} layers, "
                   f"kernel_size={kernel_size}, {self.n_params} parameters")
    
    def _apply_kernel(
        self,
        qc: QuantumCircuit,
        qubits: List[int],
        params: np.ndarray
    ):
        """Apply quantum kernel to specified qubits."""
        param_idx = 0
        for i in qubits:
            qc.ry(params[param_idx], i)
            param_idx += 1
            qc.rz(params[param_idx], i)
            param_idx += 1
        
        # Entanglement within kernel
        for i in range(len(qubits) - 1):
            qc.cx(qubits[i], qubits[i + 1])
    
    def forward(self, x: np.ndarray) -> QuantumCircuit:
        """Forward pass through convolutional layers."""
        qc = QuantumCircuit(self.n_qubits)
        
        # Initialize with data
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8) * 2 * np.pi
        for i in range(min(len(x_norm), self.n_qubits)):
            qc.ry(x_norm[i], i)
        
        param_idx = 0
        for layer in range(self.n_layers):
            # Apply kernels with stride
            for pos in range(0, self.n_qubits - self.kernel_size + 1, self.stride):
                kernel_qubits = list(range(pos, pos + self.kernel_size))
                kernel_params = self.params[
                    param_idx:param_idx + 2 * self.kernel_size
                ]
                self._apply_kernel(qc, kernel_qubits, kernel_params)
                param_idx += 2 * self.kernel_size
            
            # Pooling layer (measure and post-select)
            if self.pooling == "trace" and layer < self.n_layers - 1:
                # Trace out every other qubit
                for i in range(1, self.n_qubits, 2):
                    qc.reset(i)
        
        return qc


# Utility functions

def create_synthetic_dataset(
    n_samples: int = 200,
    n_features: int = 4,
    n_classes: int = 2,
    noise: float = 0.1,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic dataset for testing QNN.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        noise: Noise level
        random_state: Random seed
        
    Returns:
        X, y: Features and labels
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X = []
    y = []
    
    for i in range(n_samples):
        label = i % n_classes
        
        # Create class-dependent features
        if label == 0:
            sample = np.random.randn(n_features) - 1
        else:
            sample = np.random.randn(n_features) + 1
        
        # Add noise
        sample += noise * np.random.randn(n_features)
        
        X.append(sample)
        y.append(label)
    
    return np.array(X), np.array(y)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test sets."""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Quantum Neural Network Demo")
    print("=" * 60)
    
    # Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    X, y = create_synthetic_dataset(n_samples=100, n_features=4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Initialize QNN
    print("\n2. Initializing Quantum Neural Network...")
    qnn = QuantumNeuralNetwork(
        n_qubits=4,
        n_layers=2,
        feature_map="angle",
        ansatz="hardware_efficient",
        learning_rate=0.1,
        batch_size=10,
        shots=1024
    )
    
    # Train
    print("\n3. Training QNN...")
    history = qnn.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=20,
        optimizer="adam",
        verbose=True
    )
    
    # Evaluate
    print("\n4. Evaluating...")
    train_predictions = qnn.predict_batch(X_train)
    test_predictions = qnn.predict_batch(X_test)
    
    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)
    
    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy:  {test_accuracy:.4f}")
    
    # Extract learned features
    print("\n5. Extracting learned quantum features...")
    features = qnn.get_learned_features(X_test[:5])
    print(f"Feature shape: {features.shape}")
    print(f"Sample feature (first 10 dims): {features[0][:10]}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
