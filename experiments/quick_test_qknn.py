"""
Quick test of improved Q-kNN on Iris dataset
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from sim.q_knn import QuantumKNN

# Load Iris
iris = load_iris()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris.data)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, iris.target, test_size=0.3, random_state=42, stratify=iris.target
)

print("Iris Dataset Test")
print("=" * 60)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Classical k-NN
classical_knn = KNeighborsClassifier(n_neighbors=5)
classical_knn.fit(X_train, y_train)
classical_acc = classical_knn.score(X_test, y_test)
print(f"\nClassical k-NN: {classical_acc:.4f} ({classical_acc*100:.1f}%)")

# Quantum k-NN
qknn = QuantumKNN(k=5, m=32, d=4)
qknn.fit(X_train, y_train)

# Test on full test set (not subset)
quantum_acc = qknn.score(X_test, y_test, shots=512)
print(f"Quantum k-NN:   {quantum_acc:.4f} ({quantum_acc*100:.1f}%)")

# Ratio
ratio = quantum_acc / classical_acc if classical_acc > 0 else 0
print(f"\nAccuracy Ratio (Q/C): {ratio:.3f}")
print(f"Improvement: {'✅ MUCH BETTER!' if ratio > 0.8 else '⚠️ Still needs work' if ratio > 0.5 else '❌ Poor'}")
