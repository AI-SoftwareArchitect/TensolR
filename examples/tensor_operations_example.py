"""
Example showing various tensor operations with different shapes and data types
"""
import sys
import os

# Add the project root to the path so we can import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from src.tensor import Tensolr

def main():
    print("=== Tensor Operations with Different Shapes and Data Types ===")
    
    # Example 1: Different shapes
    print("\n1. Operations with Different Shapes:")
    
    # Scalar operations
    scalar = Tensolr([[5]])  # 1x1 tensor treated as scalar
    vector = Tensolr([[1, 2, 3]])
    matrix = Tensolr([[1, 2], [3, 4], [5, 6]])  # 3x2 matrix
    
    print(f"Scalar: {scalar.data.flatten()}")
    print(f"Vector: {vector.data.flatten()}")
    print(f"Matrix:\n{matrix.data}")
    
    # Broadcasting operations (simplified for this framework)
    print(f"Vector shape: {vector.shape}")
    print(f"Matrix shape: {matrix.shape}")
    
    # Example 2: Different data types
    print("\n2. Operations with Different Data Types:")
    
    # Integer tensors
    int_tensor = Tensolr([[1, 2], [3, 4]], dtype=np.int32)
    print(f"Integer tensor:\n{int_tensor.data} (dtype: {int_tensor.dtype})")
    
    # Float tensors
    float_tensor = Tensolr([[1.5, 2.7], [3.2, 4.8]])
    print(f"Float tensor:\n{float_tensor.data} (dtype: {float_tensor.dtype})")
    
    # Operations between different types
    result = int_tensor.add(float_tensor)
    print(f"Int + Float = \n{result.data} (dtype: {result.dtype})")
    
    # Example 3: Matrix operations with different dimensions
    print("\n3. Matrix Operations with Different Dimensions:")
    
    # Square matrices
    A = Tensolr([[1, 2], [3, 4]])
    B = Tensolr([[5, 6], [7, 8]])
    print(f"A = \n{A.data}")
    print(f"B = \n{B.data}")
    
    AB = A.matmul(B)
    print(f"A @ B = \n{AB.data}")
    
    # Non-square matrices
    C = Tensolr([[1, 2, 3], [4, 5, 6]])  # 2x3
    D = Tensolr([[7, 8], [9, 10], [11, 12]])  # 3x2
    print(f"\nC (2x3) = \n{C.data}")
    print(f"D (3x2) = \n{D.data}")
    
    CD = C.matmul(D)
    print(f"C @ D (2x2) = \n{CD.data}")
    
    # Example 4: Element-wise operations
    print("\n4. Element-wise Operations:")
    
    E = Tensolr([[1, 4], [9, 16]])
    F = Tensolr([[1, 2], [3, 4]])
    
    print(f"E = \n{E.data}")
    print(f"F = \n{F.data}")
    
    print(f"E + F = \n{E.add(F).data}")
    print(f"E - F = \n{E.sub(F).data}")
    print(f"E * F = \n{E.mul(F).data}")
    print(f"E / F = \n{E.div(F).data}")
    
    # Example 5: Tensor properties
    print("\n5. Tensor Properties:")
    
    test_tensor = Tensolr([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])  # 2x2x3 tensor
    print(f"3D Tensor:\n{test_tensor.data}")
    print(f"Shape: {test_tensor.shape}")
    print(f"Size (total elements): {test_tensor.size}")
    print(f"Dimensions: {test_tensor.ndim}")
    
    # Transpose operations
    print(f"\nOriginal shape: {test_tensor.shape}")
    transposed = test_tensor.transpose()  # This would transpose the last two dimensions in a full implementation
    print(f"After transpose-like operation shape: {transposed.shape}")
    print(f"Transposed data:\n{transposed.data}")
    
    # Example 6: Using factory functions
    print("\n6. Factory Functions:")
    
    zeros_tensor = Tensolr.zeros((2, 3))
    ones_tensor = Tensolr.ones((2, 3))
    rand_tensor = Tensolr.randn((2, 3))
    
    print(f"Zeros (2x3):\n{zeros_tensor.data}")
    print(f"Ones (2x3):\n{ones_tensor.data}")
    print(f"Random normal (2x3):\n{rand_tensor.data}")
    
    print("\nTensor operations with different shapes and data types example completed!")

if __name__ == "__main__":
    main()