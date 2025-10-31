"""
Base class for Amplitude Sketching framework.

All quantum data structures (QAM, QHT, Q-Count, Q-HH, Q-LSH, etc.) inherit from
this abstract base class and implement the amplitude sketching interface.

Core operations:
    1. insert(x): Accumulate phase rotations for item x
    2. query(y): Measure interference overlap for query y
    3. compose(other): Chain this sketch with another (optional)

Theory: See theory/amplitude_sketching_framework.md
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


class AmplitudeSketch(ABC):
    """
    Abstract base class for amplitude sketching data structures.
    
    All amplitude sketches share:
        - m qubits (memory size)
        - k hash functions H = {h₁, ..., hₖ}
        - θ phase rotation magnitude
        - Accumulated phase state Φ
    
    Subclasses must implement:
        - insert(x): How to accumulate phases for item x
        - query(y): How to measure interference for query y
        - _build_insert_circuit(x): Circuit construction for insert
    """
    
    def __init__(
        self,
        m: int,
        k: int,
        theta: float,
        hash_functions: Optional[List] = None
    ):
        """
        Initialize amplitude sketch.
        
        Args:
            m: Number of qubits (memory size)
            k: Number of hash functions
            theta: Base phase rotation magnitude
            hash_functions: Optional pre-defined hash functions
        """
        self.m = m
        self.k = k
        self.theta = theta
        
        # Hash functions (deterministic)
        if hash_functions is None:
            from sim.utils import make_hash_functions
            self.hash_functions = make_hash_functions(k)
        else:
            self.hash_functions = hash_functions
        
        # Circuit caching for performance
        self._circuit_cache: Dict[bytes, QuantumCircuit] = {}
        
        # Simulator
        self.simulator = AerSimulator()
        
        # Tracking
        self.n_inserts = 0
    
    @abstractmethod
    def insert(self, x: bytes) -> None:
        """
        Insert item x into amplitude sketch.
        
        Accumulates phase rotations at hash locations:
            For each hᵢ ∈ H:
                qubit_idx = hᵢ(x) mod m
                Apply Rz(θ) at qubit_idx
        
        Args:
            x: Item to insert (bytes)
        """
        pass
    
    @abstractmethod
    def query(self, y: bytes, shots: int = 512, noise_level: float = 0.0) -> float:
        """
        Query amplitude sketch for item y.
        
        Measures interference overlap between accumulated phases and query pattern:
            1. Build reference circuit with insert(y)
            2. Apply inverse rotations
            3. Measure overlap via Z-expectation or |0⟩ⁿ probability
        
        Args:
            y: Query item (bytes)
            shots: Number of measurement shots
            noise_level: Depolarizing noise parameter
        
        Returns:
            Overlap score in [0, 1] or estimated quantity
        """
        pass
    
    @abstractmethod
    def _build_insert_circuit(self, x: bytes) -> QuantumCircuit:
        """
        Build quantum circuit for inserting item x.
        
        Args:
            x: Item to insert
        
        Returns:
            QuantumCircuit with phase rotations
        """
        pass
    
    def compose(self, other: 'AmplitudeSketch') -> 'AmplitudeSketch':
        """
        Compose this amplitude sketch with another.
        
        Creates a pipeline: self → other
        Query flow: self.query(x) → filter → other.query(filtered)
        
        Args:
            other: Another amplitude sketch
        
        Returns:
            Composed sketch (or raises NotImplementedError)
        """
        raise NotImplementedError("Composition not supported by this sketch type")
    
    def error_bound(self) -> Tuple[float, float]:
        """
        Theoretical error bounds for this sketch.
        
        Returns:
            (false_positive_rate, false_negative_rate)
        """
        # Default: Compute based on universal lower bound
        # m ≥ Ω(log(1/α)/(1-ε))
        # Solve for α given m
        
        load_factor = self.n_inserts / self.m if self.m > 0 else 0
        
        # Heuristic: α ≈ (ρ)^k where ρ = load factor
        alpha = max(0.01, min(0.5, load_factor ** self.k))
        
        # β typically smaller (fewer false negatives)
        beta = alpha / 2
        
        return (alpha, beta)
    
    def get_memory_size(self) -> int:
        """Return memory size in qubits."""
        return self.m
    
    def get_circuit_depth(self, x: bytes) -> int:
        """
        Estimate circuit depth for inserting item x.
        
        Args:
            x: Item to check
        
        Returns:
            Estimated gate depth
        """
        # Base depth: k Rz rotations
        base_depth = self.k
        
        # Subclasses can override for more accurate estimates
        return base_depth
    
    def clear_cache(self):
        """Clear circuit cache to free memory."""
        self._circuit_cache.clear()
    
    def reset(self):
        """Reset sketch to empty state."""
        self._circuit_cache.clear()
        self.n_inserts = 0
    
    def _hash_to_indices(self, x: bytes) -> List[int]:
        """
        Hash item x to k qubit indices.
        
        Args:
            x: Item to hash
        
        Returns:
            List of k indices in [0, m)
        """
        return [h(x) % self.m for h in self.hash_functions]
    
    def _create_noise_model(self, noise_level: float) -> Optional[NoiseModel]:
        """
        Create noise model for simulation.
        
        Args:
            noise_level: Depolarizing error probability per 2Q gate
        
        Returns:
            NoiseModel or None if noise_level == 0
        """
        if noise_level <= 0:
            return None
        
        noise_model = NoiseModel()
        
        # Add depolarizing noise to 2-qubit gates
        error_2q = depolarizing_error(noise_level, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'cy'])
        
        # Add smaller noise to 1-qubit gates
        error_1q = depolarizing_error(noise_level / 10, 1)
        noise_model.add_all_qubit_quantum_error(error_1q, ['rz', 'rx', 'ry', 'h'])
        
        return noise_model
    
    def _measure_overlap(
        self,
        circuit: QuantumCircuit,
        shots: int,
        noise_level: float = 0.0
    ) -> float:
        """
        Measure overlap probability from circuit.
        
        Standard pattern: Measure |0⟩^m probability as overlap score.
        
        Args:
            circuit: Circuit to measure
            shots: Number of shots
            noise_level: Noise parameter
        
        Returns:
            Overlap in [0, 1]
        """
        # Add measurements
        qc = circuit.copy()
        qc.measure_all()
        
        # Run simulation
        noise_model = self._create_noise_model(noise_level)
        
        if noise_model:
            job = self.simulator.run(qc, shots=shots, noise_model=noise_model)
        else:
            job = self.simulator.run(qc, shots=shots)
        
        counts = job.result().get_counts()
        
        # Overlap = probability of |0⟩^m state
        all_zero = '0' * self.m
        overlap = counts.get(all_zero, 0) / shots
        
        return overlap
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}(m={self.m}, k={self.k}, "
            f"theta={self.theta:.3f}, n_inserts={self.n_inserts})"
        )
    
    def get_stats(self) -> Dict:
        """
        Get statistics about sketch state.
        
        Returns:
            Dictionary with sketch statistics
        """
        alpha, beta = self.error_bound()
        
        return {
            'class': self.__class__.__name__,
            'm': self.m,
            'k': self.k,
            'theta': self.theta,
            'n_inserts': self.n_inserts,
            'load_factor': self.n_inserts / self.m if self.m > 0 else 0,
            'estimated_alpha': alpha,
            'estimated_beta': beta,
            'cache_size': len(self._circuit_cache)
        }


class SerialComposition:
    """
    Serial composition of multiple amplitude sketches.
    
    Pipeline: AS₁ → AS₂ → ... → ASₙ
    """
    
    def __init__(self, sketches: List[AmplitudeSketch]):
        """
        Initialize serial composition.
        
        Args:
            sketches: List of amplitude sketches in order
        """
        self.sketches = sketches
        self.n_stages = len(sketches)
    
    def query(self, x: bytes, shots: int = 512, noise_level: float = 0.0) -> float:
        """
        Query through pipeline.
        
        Args:
            x: Query item
            shots: Shots per stage
            noise_level: Noise per stage
        
        Returns:
            Final overlap score
        """
        # Each stage filters based on query result
        score = 1.0
        
        for sketch in self.sketches:
            stage_score = sketch.query(x, shots=shots, noise_level=noise_level)
            score *= stage_score
            
            # Early termination if score too low
            if score < 0.01:
                break
        
        return score
    
    def error_bound(self) -> Tuple[float, float]:
        """
        Compute composed error bounds.
        
        Returns:
            (α_total, β_total)
        """
        alpha_total = 0.0
        beta_product = 1.0
        
        for sketch in self.sketches:
            alpha, beta = sketch.error_bound()
            alpha_total += alpha  # First-order approximation
            beta_product *= (1 - beta)
        
        beta_total = 1 - beta_product
        
        return (alpha_total, beta_total)
    
    def get_total_memory(self) -> int:
        """Total memory (max across stages if reusing qubits)."""
        return max(sketch.get_memory_size() for sketch in self.sketches)


# Type alias for convenience
Sketch = AmplitudeSketch
