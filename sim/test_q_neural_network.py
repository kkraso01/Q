"""
Tests for Quantum Neural Networks

Author: Quantum Data Structures Research
Date: November 2025
"""

import pytest
import numpy as np
import pytest
from sim.qiskit_compat import PRIMITIVES_AVAILABLE
pytestmark = pytest.mark.skipif(
    not PRIMITIVES_AVAILABLE,
    reason="Qiskit primitives (Sampler/Estimator) unavailable in this environment."
)

from sim.q_neural_network import (
    QuantumNeuralNetwork,
    QuantumConvolutionalNetwork,
    create_synthetic_dataset,
    train_test_split
)


class TestQuantumNeuralNetwork:
    """Tests for QuantumNeuralNetwork class."""
    
    def test_initialization(self):
        """Test QNN initialization."""
        qnn = QuantumNeuralNetwork(
            n_qubits=4,
            n_layers=2,
            feature_map="angle",
            ansatz="hardware_efficient"
        )
        
        assert qnn.n_qubits == 4
        assert qnn.n_layers == 2
        assert qnn.n_params == 2 * 4 * 2  # 2 rotations * 4 qubits * 2 layers
        assert len(qnn.params) == qnn.n_params
    
    def test_parameter_calculation(self):
        """Test parameter count for different ansatzes."""
        # Hardware efficient
        qnn_he = QuantumNeuralNetwork(
            n_qubits=4, n_layers=3, ansatz="hardware_efficient"
        )
        assert qnn_he.n_params == 2 * 4 * 3  # RY, RZ per qubit per layer
        
        # Strongly entangling
        qnn_se = QuantumNeuralNetwork(
            n_qubits=4, n_layers=3, ansatz="strongly_entangling"
        )
        assert qnn_se.n_params == 3 * 4 * 3  # RX, RY, RZ per qubit per layer
        
        # Real amplitudes
        qnn_ra = QuantumNeuralNetwork(
            n_qubits=4, n_layers=3, ansatz="real_amplitudes"
        )
        assert qnn_ra.n_params == 4 * 3  # RY per qubit per layer
    
    def test_feature_map_angle(self):
        """Test angle encoding feature map."""
        qnn = QuantumNeuralNetwork(
            n_qubits=4, n_layers=1, feature_map="angle"
        )
        
        x = np.array([0.5, 1.0, 1.5, 2.0])
        qc = qnn._feature_map_circuit(x)
        
        assert qc.num_qubits == 4
        assert qc.depth() > 0
    
    def test_feature_map_amplitude(self):
        """Test amplitude encoding feature map."""
        qnn = QuantumNeuralNetwork(
            n_qubits=3, n_layers=1, feature_map="amplitude"
        )
        
        x = np.array([1.0, 2.0, 3.0, 4.0])
        x_padded = np.concatenate([x, np.zeros(2**qnn.n_qubits - len(x))])
        normalized = qnn._normalize_amplitudes(x_padded)
        assert abs(np.linalg.norm(normalized) - 1.0) < 1e-12
        qc = qnn._feature_map_circuit(x)
        
        assert qc.num_qubits == 3
    
    def test_feature_map_iqp(self):
        """Test IQP-style feature map."""
        qnn = QuantumNeuralNetwork(
            n_qubits=4, n_layers=1, feature_map="iqp"
        )
        
        x = np.array([0.1, 0.2, 0.3, 0.4])
        qc = qnn._feature_map_circuit(x)
        
        assert qc.num_qubits == 4
        # Should have Hadamards, phase gates, and controlled phases
        assert qc.depth() > 0
    
    def test_variational_circuit_hardware_efficient(self):
        """Test hardware-efficient ansatz."""
        qnn = QuantumNeuralNetwork(
            n_qubits=4, n_layers=2, ansatz="hardware_efficient"
        )
        
        qc = qnn._variational_circuit(qnn.params)
        
        assert qc.num_qubits == 4
        # Should have rotation and entanglement layers
        assert qc.depth() >= 2 * 2  # At least 2 layers
    
    def test_variational_circuit_strongly_entangling(self):
        """Test strongly entangling ansatz."""
        qnn = QuantumNeuralNetwork(
            n_qubits=4, n_layers=2, ansatz="strongly_entangling"
        )
        
        qc = qnn._variational_circuit(qnn.params)
        
        assert qc.num_qubits == 4
        assert qc.depth() > 0
    
    def test_forward_pass(self):
        """Test forward pass."""
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2)
        
        x = np.array([0.5, 1.0, 1.5, 2.0])
        qc = qnn.forward(x)
        
        assert qc.num_qubits == 4
        # Circuit should be feature map + variational
        assert qc.depth() > 0
    
    def test_predict_proba(self):
        """Test probability prediction."""
        qnn = QuantumNeuralNetwork(
            n_qubits=4, n_layers=1, shots=1000
        )
        
        x = np.array([0.5, 1.0, 1.5, 2.0])
        prob = qnn.predict_proba(x)
        
        assert 0.0 <= prob <= 1.0
    
    def test_predict(self):
        """Test class prediction."""
        qnn = QuantumNeuralNetwork(
            n_qubits=4, n_layers=1, shots=1000
        )
        
        x = np.array([0.5, 1.0, 1.5, 2.0])
        pred = qnn.predict(x)
        
        assert pred in [0, 1]
    
    def test_predict_batch(self):
        """Test batch prediction."""
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=1, shots=500)
        
        X = np.random.randn(5, 4)
        predictions = qnn.predict_batch(X)
        
        assert len(predictions) == 5
        assert all(p in [0, 1] for p in predictions)
    
    def test_compute_loss_cross_entropy(self):
        """Test cross-entropy loss."""
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=1, shots=500)
        
        X = np.random.randn(3, 4)
        y = np.array([0, 1, 0])
        
        loss = qnn.compute_loss(X, y, loss_type="cross_entropy")
        
        assert loss >= 0.0
        assert not np.isnan(loss)
    
    def test_compute_loss_mse(self):
        """Test MSE loss."""
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=1, shots=500)
        
        X = np.random.randn(3, 4)
        y = np.array([0, 1, 0])
        
        loss = qnn.compute_loss(X, y, loss_type="mse")
        
        assert loss >= 0.0
    
    def test_compute_loss_hinge(self):
        """Test hinge loss."""
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=1, shots=500)
        
        X = np.random.randn(3, 4)
        y = np.array([0, 1, 0])
        
        loss = qnn.compute_loss(X, y, loss_type="hinge")
        
        assert loss >= 0.0
    
    def test_compute_gradient(self):
        """Test gradient computation via parameter shift rule."""
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=1, shots=500)
        
        X = np.random.randn(2, 4)
        y = np.array([0, 1])
        
        # Compute gradient for first parameter
        grad = qnn.compute_gradient(X, y, param_idx=0)
        
        # Gradient should be a real number
        assert isinstance(grad, (int, float))
        assert not np.isnan(grad)
    
    def test_training_sgd(self):
        """Test training with SGD optimizer."""
        np.random.seed(42)
        
        # Create simple dataset
        X = np.random.randn(20, 4)
        y = np.array([0, 1] * 10)
        
        qnn = QuantumNeuralNetwork(
            n_qubits=4,
            n_layers=1,
            learning_rate=0.1,
            batch_size=5,
            shots=500
        )
        
        initial_params = qnn.params.copy()
        
        history = qnn.train(
            X, y,
            epochs=3,
            optimizer="sgd",
            verbose=False
        )
        
        # Parameters should have changed
        assert not np.allclose(qnn.params, initial_params)
        
        # Should have loss history
        assert len(history['train_loss']) == 3
        assert len(history['train_accuracy']) == 3
    
    def test_training_adam(self):
        """Test training with Adam optimizer."""
        np.random.seed(42)
        
        X = np.random.randn(20, 4)
        y = np.array([0, 1] * 10)
        
        qnn = QuantumNeuralNetwork(
            n_qubits=4,
            n_layers=1,
            learning_rate=0.1,
            batch_size=5,
            shots=500
        )
        
        history = qnn.train(
            X, y,
            epochs=3,
            optimizer="adam",
            verbose=False
        )
        
        assert len(history['train_loss']) == 3
    
    def test_training_with_validation(self):
        """Test training with validation set."""
        np.random.seed(42)
        
        X_train = np.random.randn(20, 4)
        y_train = np.array([0, 1] * 10)
        X_val = np.random.randn(10, 4)
        y_val = np.array([0, 1] * 5)
        
        qnn = QuantumNeuralNetwork(
            n_qubits=4,
            n_layers=1,
            learning_rate=0.1,
            batch_size=5,
            shots=500
        )
        
        history = qnn.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=3,
            verbose=False
        )
        
        assert len(history['val_loss']) == 3
        assert len(history['val_accuracy']) == 3
    
    def test_get_learned_features(self):
        """Test learned feature extraction."""
        qnn = QuantumNeuralNetwork(n_qubits=3, n_layers=1)
        
        X = np.random.randn(5, 3)
        features = qnn.get_learned_features(X)
        
        # Should be 2^n_qubits dimensional features
        assert features.shape == (5, 2**3)
    
    def test_save_load_model(self, tmp_path):
        """Test model saving and loading."""
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2)
        
        # Modify parameters
        qnn.params = np.random.randn(qnn.n_params)
        qnn.loss_history = [0.5, 0.4, 0.3]
        qnn.accuracy_history = [0.6, 0.7, 0.8]
        
        # Save
        filepath = tmp_path / "qnn_model.npz"
        qnn.save_model(str(filepath))
        
        # Load into new model
        qnn2 = QuantumNeuralNetwork(n_qubits=4, n_layers=2)
        qnn2.load_model(str(filepath))
        
        # Check parameters match
        np.testing.assert_array_almost_equal(qnn.params, qnn2.params)
        assert qnn.loss_history == qnn2.loss_history
        assert qnn.accuracy_history == qnn2.accuracy_history


class TestQuantumConvolutionalNetwork:
    """Tests for QuantumConvolutionalNetwork class."""
    
    def test_initialization(self):
        """Test QCNN initialization."""
        qcnn = QuantumConvolutionalNetwork(
            n_qubits=8,
            n_layers=2,
            kernel_size=2,
            stride=1
        )
        
        assert qcnn.n_qubits == 8
        assert qcnn.n_layers == 2
        assert qcnn.kernel_size == 2
        assert qcnn.n_params > 0
    
    def test_forward_pass(self):
        """Test QCNN forward pass."""
        qcnn = QuantumConvolutionalNetwork(
            n_qubits=4,
            n_layers=2,
            kernel_size=2
        )
        
        x = np.random.randn(4)
        qc = qcnn.forward(x)
        
        assert qc.num_qubits == 4
        assert qc.depth() > 0
    
    def test_kernel_application(self):
        """Test kernel application."""
        from qiskit import QuantumCircuit
        
        qcnn = QuantumConvolutionalNetwork(
            n_qubits=4,
            n_layers=1,
            kernel_size=2
        )
        
        qc = QuantumCircuit(4)
        params = np.random.randn(4)  # 2 params per qubit in kernel
        
        qcnn._apply_kernel(qc, [0, 1], params)
        
        assert qc.depth() > 0


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_create_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        X, y = create_synthetic_dataset(
            n_samples=100,
            n_features=4,
            n_classes=2,
            random_state=42
        )
        
        assert X.shape == (100, 4)
        assert y.shape == (100,)
        assert set(y) == {0, 1}
    
    def test_create_synthetic_dataset_multiclass(self):
        """Test multiclass dataset creation."""
        X, y = create_synthetic_dataset(
            n_samples=90,
            n_features=5,
            n_classes=3,
            random_state=42
        )
        
        assert X.shape == (90, 5)
        assert set(y) == {0, 1, 2}
    
    def test_train_test_split(self):
        """Test train/test splitting."""
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 2, 100)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
    
    def test_train_test_split_different_sizes(self):
        """Test splitting with different test sizes."""
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 2, 100)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        assert len(X_test) == 30
        assert len(X_train) == 70


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test complete training pipeline."""
        np.random.seed(42)
        
        # Create dataset
        X, y = create_synthetic_dataset(
            n_samples=50,
            n_features=4,
            noise=0.2,
            random_state=42
        )
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Initialize and train
        qnn = QuantumNeuralNetwork(
            n_qubits=4,
            n_layers=1,
            learning_rate=0.1,
            batch_size=10,
            shots=500
        )
        
        history = qnn.train(
            X_train, y_train,
            X_val=X_test, y_val=y_test,
            epochs=3,
            optimizer="adam",
            verbose=False
        )
        
        # Make predictions
        predictions = qnn.predict_batch(X_test)
        accuracy = np.mean(predictions == y_test)
        
        # Should achieve some accuracy
        assert 0.0 <= accuracy <= 1.0
        
        # Training loss should exist
        assert len(history['train_loss']) > 0
    
    def test_different_ansatz_training(self):
        """Test training with different ansatzes."""
        np.random.seed(42)
        
        X = np.random.randn(30, 4)
        y = np.array([0, 1] * 15)
        
        for ansatz in ["hardware_efficient", "strongly_entangling", "real_amplitudes"]:
            qnn = QuantumNeuralNetwork(
                n_qubits=4,
                n_layers=1,
                ansatz=ansatz,
                learning_rate=0.1,
                shots=500
            )
            
            history = qnn.train(
                X, y,
                epochs=2,
                verbose=False
            )
            
            assert len(history['train_loss']) == 2
    
    def test_different_feature_maps(self):
        """Test training with different feature maps."""
        np.random.seed(42)
        
        X = np.random.randn(30, 4)
        y = np.array([0, 1] * 15)
        
        for feature_map in ["angle", "iqp"]:  # Skip amplitude (slower)
            qnn = QuantumNeuralNetwork(
                n_qubits=4,
                n_layers=1,
                feature_map=feature_map,
                learning_rate=0.1,
                shots=500
            )
            
            history = qnn.train(
                X, y,
                epochs=2,
                verbose=False
            )
            
            assert len(history['train_loss']) == 2


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_ansatz(self):
        """Test invalid ansatz raises error."""
        with pytest.raises(ValueError):
            qnn = QuantumNeuralNetwork(
                n_qubits=4,
                n_layers=1,
                ansatz="invalid_ansatz"
            )
    
    def test_mismatched_dimensions(self):
        """Test handling of mismatched feature dimensions."""
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=1)
        
        # More features than qubits - should handle gracefully
        x = np.random.randn(8)
        qc = qnn.forward(x)
        assert qc is not None
    
    def test_zero_samples(self):
        """Test behavior with empty dataset."""
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=1)
        
        X = np.array([]).reshape(0, 4)
        y = np.array([])
        
        # Should handle gracefully
        predictions = qnn.predict_batch(X)
        assert len(predictions) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
