"""
Tensolr - A TensorFlow-like tensor framework with automatic differentiation and MLIR integration
"""

from .tensor import Tensolr
from .global_graph import GLOBAL_GRAPH

__version__ = "0.1.0"
__author__ = "Tensolr Team"

# Expose main tensor class at package level
__all__ = ["Tensolr", "GLOBAL_GRAPH"]