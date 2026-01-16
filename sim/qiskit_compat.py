"""Compatibility layer for Qiskit primitives across versions."""

from __future__ import annotations

from typing import Any

PRIMITIVES_AVAILABLE = False
PRIMITIVES_BACKEND = "unavailable"

_SamplerClass = None
_EstimatorClass = None
_BackendSamplerClass = None
_BackendEstimatorClass = None
_AerSamplerClass = None
_AerEstimatorClass = None


def _try_imports() -> None:
    """Attempt to resolve primitive implementations in priority order."""
    global PRIMITIVES_AVAILABLE
    global PRIMITIVES_BACKEND
    global _SamplerClass
    global _EstimatorClass
    global _BackendSamplerClass
    global _BackendEstimatorClass
    global _AerSamplerClass
    global _AerEstimatorClass

    if PRIMITIVES_AVAILABLE:
        return

    try:
        from qiskit.primitives import Sampler, Estimator

        _SamplerClass = Sampler
        _EstimatorClass = Estimator
        PRIMITIVES_AVAILABLE = True
        PRIMITIVES_BACKEND = "qiskit.primitives"
        return
    except Exception:
        pass

    try:
        from qiskit.primitives import BackendSampler, BackendEstimator

        _BackendSamplerClass = BackendSampler
        _BackendEstimatorClass = BackendEstimator
        PRIMITIVES_AVAILABLE = True
        PRIMITIVES_BACKEND = "qiskit.primitives.backend"
        return
    except Exception:
        pass

    try:
        from qiskit_aer.primitives import Sampler as AerSampler
        from qiskit_aer.primitives import Estimator as AerEstimator

        _AerSamplerClass = AerSampler
        _AerEstimatorClass = AerEstimator
        PRIMITIVES_AVAILABLE = True
        PRIMITIVES_BACKEND = "qiskit_aer.primitives"
    except Exception:
        PRIMITIVES_AVAILABLE = False
        PRIMITIVES_BACKEND = "unavailable"


def _ensure_available() -> None:
    _try_imports()
    if not PRIMITIVES_AVAILABLE:
        raise RuntimeError(
            "Qiskit primitives (Sampler/Estimator) are unavailable. "
            "Install a compatible qiskit/qiskit-aer version or enable "
            "the Aer primitives backend."
        )


def _get_aer_backend() -> Any:
    from qiskit_aer import AerSimulator

    return AerSimulator()


def get_sampler() -> Any:
    """Return a Sampler instance using the best available implementation."""
    _ensure_available()

    if _SamplerClass is not None:
        return _SamplerClass()
    if _BackendSamplerClass is not None:
        return _BackendSamplerClass(backend=_get_aer_backend())
    if _AerSamplerClass is not None:
        return _AerSamplerClass()
    raise RuntimeError("Sampler implementation could not be resolved.")


def get_estimator() -> Any:
    """Return an Estimator instance using the best available implementation."""
    _ensure_available()

    if _EstimatorClass is not None:
        return _EstimatorClass()
    if _BackendEstimatorClass is not None:
        return _BackendEstimatorClass(backend=_get_aer_backend())
    if _AerEstimatorClass is not None:
        return _AerEstimatorClass()
    raise RuntimeError("Estimator implementation could not be resolved.")


_try_imports()
