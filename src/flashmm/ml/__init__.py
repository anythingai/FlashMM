"""
FlashMM Machine Learning Module

Provides ML model loading, inference, and prediction capabilities.
"""

from flashmm.ml.inference.inference_engine import InferenceEngine
from flashmm.ml.models.predictor import Predictor

__all__ = [
    "InferenceEngine",
    "Predictor",
]
