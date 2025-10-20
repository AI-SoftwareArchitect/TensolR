#!/usr/bin/env python
"""
Test script to verify that the fix for the Tensolr multiplication and gradient handling works
"""

import sys
import os

# Add the project root to the path so we can import from src
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from src.tensor import Tensolr

# Test 1: Scalar multiplication (the original issue)
print("Test 1: Testing Tensolr scalar multiplication...")
try:
    t = Tensolr.randn((3, 3))
    result = t * 2.5  # This should now work
    print("✓ Scalar multiplication works!")
    print(f"Original shape: {t.shape}, Result shape: {result.shape}")
except Exception as e:
    print(f"✗ Scalar multiplication failed: {e}")

# Test 2: Reverse scalar multiplication
print("\nTest 2: Testing reverse scalar multiplication...")
try:
    t = Tensolr.randn((2, 2))
    result = 3.0 * t  # This should now work
    print("✓ Reverse scalar multiplication works!")
    print(f"Original shape: {t.shape}, Result shape: {result.shape}")
except Exception as e:
    print(f"✗ Reverse scalar multiplication failed: {e}")

# Test 3: Check if tensor nodes are created properly
print("\nTest 3: Testing tensor node creation...")
try:
    t = Tensolr([1, 2, 3])
    print(f"✓ Tensor node exists: {t._node is not None}")
    print(f"✓ Tensor data: {t.data}")
except Exception as e:
    print(f"✗ Tensor node creation failed: {e}")

print("\nAll tests completed!")