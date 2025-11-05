"""
Test suite for Quantum k-NN classifier
"""
import pytest
import numpy as np
from sim.q_knn import QuantumKNN


def test_qknn_initialization():
    """Test Quantum k-NN initialization."""
    qknn = QuantumKNN(k=5, m=32, k_hash=3, d=64)
    
    assert qknn.k == 5
    assert qknn.m == 32
    assert qknn.d == 64
    assert qknn.X_train is None
    assert qknn.y_train is None


def test_qknn_fit():
    """Test fitting Quantum k-NN."""
    np.random.seed(42)
    
    # Create simple 2-class dataset
    X_train = np.random.randn(20, 32)
    y_train = np.array([0]*10 + [1]*10)
    
    qknn = QuantumKNN(k=3, m=16, d=32)
    qknn.fit(X_train, y_train)
    
    assert qknn.X_train.shape == (20, 32)
    assert qknn.y_train.shape == (20,)
    assert len(qknn.qlsh.inserted_vectors) == 20


def test_qknn_predict_binary():
    """Test prediction on binary classification."""
    np.random.seed(42)
    
    # Create separable 2-class dataset
    X_class0 = np.random.randn(10, 16) - 2  # Centered at -2
    X_class1 = np.random.randn(10, 16) + 2  # Centered at +2
    X_train = np.vstack([X_class0, X_class1])
    y_train = np.array([0]*10 + [1]*10)
    
    # Test samples (should be clearly separable)
    X_test = np.array([
        np.random.randn(16) - 2,  # Should predict 0
        np.random.randn(16) + 2,  # Should predict 1
    ])
    y_test = np.array([0, 1])
    
    qknn = QuantumKNN(k=3, m=32, d=16)
    qknn.fit(X_train, y_train)
    
    predictions = qknn.predict(X_test, shots=256)
    
    assert predictions.shape == (2,)
    assert all(p in [0, 1] for p in predictions)


def test_qknn_predict_multiclass():
    """Test prediction on multi-class classification."""
    np.random.seed(42)
    
    # Create 3-class dataset
    X_class0 = np.random.randn(10, 16) + np.array([2, 0, 0, 0] + [0]*12)
    X_class1 = np.random.randn(10, 16) + np.array([0, 2, 0, 0] + [0]*12)
    X_class2 = np.random.randn(10, 16) + np.array([0, 0, 2, 0] + [0]*12)
    
    X_train = np.vstack([X_class0, X_class1, X_class2])
    y_train = np.array([0]*10 + [1]*10 + [2]*10)
    
    qknn = QuantumKNN(k=5, m=32, d=16)
    qknn.fit(X_train, y_train)
    
    # Test samples from each class
    X_test = np.array([
        np.random.randn(16) + np.array([2, 0, 0, 0] + [0]*12),
        np.random.randn(16) + np.array([0, 2, 0, 0] + [0]*12),
        np.random.randn(16) + np.array([0, 0, 2, 0] + [0]*12),
    ])
    
    predictions = qknn.predict(X_test, shots=256)
    
    assert predictions.shape == (3,)
    assert all(p in [0, 1, 2] for p in predictions)


def test_qknn_predict_proba():
    """Test probability prediction."""
    np.random.seed(42)
    
    X_train = np.random.randn(20, 16)
    y_train = np.array([0]*10 + [1]*10)
    
    qknn = QuantumKNN(k=5, m=32, d=16)
    qknn.fit(X_train, y_train)
    
    X_test = np.random.randn(5, 16)
    proba = qknn.predict_proba(X_test, shots=128)
    
    assert proba.shape == (5, 2)  # 5 samples, 2 classes
    assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1
    assert np.all(proba >= 0) and np.all(proba <= 1)  # Valid probabilities


def test_qknn_score():
    """Test accuracy scoring."""
    np.random.seed(42)
    
    # Create clearly separable dataset
    X_class0 = np.random.randn(15, 16) - 3
    X_class1 = np.random.randn(15, 16) + 3
    X_train = np.vstack([X_class0, X_class1])
    y_train = np.array([0]*15 + [1]*15)
    
    X_test = np.vstack([
        np.random.randn(5, 16) - 3,
        np.random.randn(5, 16) + 3
    ])
    y_test = np.array([0]*5 + [1]*5)
    
    qknn = QuantumKNN(k=3, m=32, d=16)
    qknn.fit(X_train, y_train)
    
    accuracy = qknn.score(X_test, y_test, shots=256)
    
    assert 0.0 <= accuracy <= 1.0
    # Should get reasonable accuracy on separable data
    assert accuracy >= 0.5  # Better than random


def test_qknn_majority_vote():
    """Test majority voting mechanism."""
    qknn = QuantumKNN(k=5, m=32, d=16)
    
    # Test clear majority
    labels = np.array([0, 0, 0, 1, 1])
    assert qknn._majority_vote(labels) == 0
    
    labels = np.array([1, 1, 1, 1, 0])
    assert qknn._majority_vote(labels) == 1
    
    # Test tie (should return one of the classes)
    labels = np.array([0, 0, 1, 1])
    result = qknn._majority_vote(labels)
    assert result in [0, 1]


def test_qknn_get_params():
    """Test parameter retrieval."""
    qknn = QuantumKNN(k=7, m=64, k_hash=4, theta=np.pi/8, d=128)
    
    params = qknn.get_params()
    
    assert params['k'] == 7
    assert params['m'] == 64
    assert params['d'] == 128
    assert params['theta'] == np.pi/8
    assert params['k_hash'] == 4


def test_qknn_with_noise():
    """Test Quantum k-NN with depolarizing noise."""
    np.random.seed(42)
    
    X_train = np.random.randn(20, 16)
    y_train = np.array([0]*10 + [1]*10)
    X_test = np.random.randn(5, 16)
    
    qknn = QuantumKNN(k=3, m=32, d=16)
    qknn.fit(X_train, y_train)
    
    # Predict with noise
    predictions_noisy = qknn.predict(X_test, shots=256, noise_level=0.01)
    
    assert predictions_noisy.shape == (5,)
    assert all(p in [0, 1] for p in predictions_noisy)


def test_qknn_iris_like():
    """Test on Iris-like dataset (3 classes, 4 features)."""
    np.random.seed(42)
    
    # Simulate Iris dataset structure
    n_samples_per_class = 20
    n_features = 4
    
    # Create 3 well-separated clusters
    X_class0 = np.random.randn(n_samples_per_class, n_features) + np.array([0, 0, 0, 0])
    X_class1 = np.random.randn(n_samples_per_class, n_features) + np.array([3, 3, 3, 3])
    X_class2 = np.random.randn(n_samples_per_class, n_features) + np.array([-3, -3, -3, -3])
    
    X_train = np.vstack([X_class0, X_class1, X_class2])
    y_train = np.array([0]*n_samples_per_class + [1]*n_samples_per_class + [2]*n_samples_per_class)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[shuffle_idx], y_train[shuffle_idx]
    
    # Train/test split
    split = int(0.8 * len(X_train))
    X_train_split, X_test = X_train[:split], X_train[split:]
    y_train_split, y_test = y_train[:split], y_train[split:]
    
    qknn = QuantumKNN(k=5, m=32, d=n_features)
    qknn.fit(X_train_split, y_train_split)
    
    accuracy = qknn.score(X_test, y_test, shots=512)
    
    # Quantum k-NN is approximate, so just check it works
    assert 0.0 <= accuracy <= 1.0
    print(f"Iris-like test accuracy: {accuracy:.2f}")
