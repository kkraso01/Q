"""
Quantum k-Nearest Neighbors Classifier

Uses Q-LSH for approximate nearest neighbor search with quantum advantage in batch queries.
"""
import numpy as np
from .q_lsh import QLSH


class QuantumKNN:
    """
    Quantum k-Nearest Neighbors classifier using Q-LSH.
    
    Provides quantum-enhanced classification via approximate nearest neighbor search.
    Achieves √B variance reduction for batch queries.
    """
    
    def __init__(self, k=5, m=64, k_hash=4, theta=np.pi/4, d=128):
        """
        Initialize Quantum k-NN classifier.
        
        Args:
            k: Number of neighbors for voting
            m: Number of qubits (memory size)
            k_hash: Number of hash functions for Q-LSH
            theta: Phase rotation angle
            d: Feature dimensionality
        """
        self.k = k
        self.m = m
        self.d = d
        self.qlsh = QLSH(m=m, k=k_hash, theta=theta, d=d)
        self.X_train = None
        self.y_train = None
        
    def fit(self, X_train, y_train):
        """
        Fit the classifier by inserting training data into Q-LSH.
        
        Args:
            X_train: Training vectors, shape (n_samples, n_features)
            y_train: Training labels, shape (n_samples,)
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        
        # Insert all training vectors into Q-LSH
        for x in X_train:
            self.qlsh.insert(x)
            
        return self
    
    def predict(self, X_test, shots=512, noise_level=0.0):
        """
        Predict labels for test data.
        
        Args:
            X_test: Test vectors, shape (n_test, n_features)
            shots: Number of quantum measurements per query
            noise_level: Depolarizing noise level (0.0 = noiseless)
            
        Returns:
            predictions: Predicted labels, shape (n_test,)
        """
        X_test = np.array(X_test)
        predictions = []
        
        for x in X_test:
            # Query k nearest neighbors using Q-LSH
            # Returns list of (vector, similarity) tuples
            neighbor_results = self.qlsh.query_knn(
                x, 
                k_neighbors=self.k,
                shots=shots,
                noise_level=noise_level
            )
            
            # Find indices of neighbor vectors in training set
            neighbor_indices = []
            for neighbor_vec, _ in neighbor_results:
                # Find matching training vector
                for idx, train_vec in enumerate(self.X_train):
                    if np.allclose(neighbor_vec, train_vec):
                        neighbor_indices.append(idx)
                        break
            
            # Get labels of k nearest neighbors
            neighbor_labels = self.y_train[neighbor_indices]
            
            # Majority vote
            prediction = self._majority_vote(neighbor_labels)
            predictions.append(prediction)
            
        return np.array(predictions)
    
    def predict_proba(self, X_test, shots=512, noise_level=0.0):
        """
        Predict class probabilities for test data.
        
        Args:
            X_test: Test vectors, shape (n_test, n_features)
            shots: Number of quantum measurements
            noise_level: Depolarizing noise level
            
        Returns:
            probabilities: Class probabilities, shape (n_test, n_classes)
        """
        X_test = np.array(X_test)
        probabilities = []
        
        classes = np.unique(self.y_train)
        n_classes = len(classes)
        
        for x in X_test:
            # Query k nearest neighbors
            neighbor_results = self.qlsh.query_knn(
                x,
                k_neighbors=self.k,
                shots=shots,
                noise_level=noise_level
            )
            
            # Find indices of neighbor vectors
            neighbor_indices = []
            for neighbor_vec, _ in neighbor_results:
                for idx, train_vec in enumerate(self.X_train):
                    if np.allclose(neighbor_vec, train_vec):
                        neighbor_indices.append(idx)
                        break
            
            neighbor_labels = self.y_train[neighbor_indices]
            
            # Compute class probabilities
            proba = np.zeros(n_classes)
            for i, cls in enumerate(classes):
                proba[i] = np.sum(neighbor_labels == cls) / len(neighbor_labels) if len(neighbor_labels) > 0 else 0
                
            probabilities.append(proba)
            
        return np.array(probabilities)
    
    def score(self, X_test, y_test, shots=512, noise_level=0.0):
        """
        Compute accuracy on test set.
        
        Args:
            X_test: Test vectors
            y_test: True labels
            shots: Number of quantum measurements
            noise_level: Noise level
            
        Returns:
            accuracy: Fraction of correct predictions
        """
        predictions = self.predict(X_test, shots=shots, noise_level=noise_level)
        accuracy = np.mean(predictions == y_test)
        return accuracy
    
    def _majority_vote(self, labels):
        """Compute majority vote from neighbor labels."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        majority_label = unique_labels[np.argmax(counts)]
        return majority_label
    
    def get_params(self):
        """Get classifier parameters."""
        return {
            'k': self.k,
            'm': self.m,
            'd': self.d,
            'theta': self.qlsh.theta,
            'k_hash': self.qlsh.k
        }
