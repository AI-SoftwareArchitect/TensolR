"""
Example demonstrating computation graphs and automatic differentiation
"""
import sys
import os

# Add the project root to the path so we can import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from src.tensor import Tensolr
from src.global_graph import GLOBAL_GRAPH

def main():
    print("=== Computation Graphs and Automatic Differentiation Example ===")
    
    # Example 1: Simple function: f(x) = (x + 2) * (x - 1)
    print("\n1. Computing gradients for f(x) = (x + 2) * (x - 1)")
    
    # Clear the global graph
    GLOBAL_GRAPH.nodes = []
    
    # Create variable tensor with gradient tracking
    x = Tensolr([[3.0]], track_graph=True)  # x = 3
    print(f"Input x: {x.data}")
    
    # Compute: a = x + 2
    two = Tensolr([[2.0]], track_graph=True)
    a = x.add(two)
    
    # Compute: b = x - 1
    one = Tensolr([[1.0]], track_graph=True)
    b = x.sub(one)
    
    # Compute: f = a * b = (x + 2) * (x - 1)
    f = a.mul(b)
    
    print(f"a = x + 2 = {a.data}")
    print(f"b = x - 1 = {b.data}")
    print(f"f = a * b = {f.data}")
    
    # Forward pass
    result = GLOBAL_GRAPH.forward()
    print(f"Final result of f(x): {result.data}")
    
    # Analytical gradient: df/dx = d/dx[(x+2)(x-1)] = d/dx[x^2 + x - 2] = 2x + 1
    # At x=3: df/dx = 2*3 + 1 = 7
    analytical_grad = 2 * 3 + 1
    print(f"Analytical gradient at x=3: {analytical_grad}")
    
    # Compute gradients using backpropagation
    GLOBAL_GRAPH.backward()
    print(f"Computed gradient using backpropagation: {x._node.grad}")
    
    # Example 2: More complex function: f(x, y) = (x * y) + (x^2)
    print("\n2. Computing gradients for f(x, y) = (x * y) + (x^2)")
    
    # Clear the global graph
    GLOBAL_GRAPH.nodes = []
    
    # Create variable tensors
    x = Tensolr([[2.0]], track_graph=True)
    y = Tensolr([[3.0]], track_graph=True)
    print(f"Input x: {x.data}, y: {y.data}")
    
    # Compute: z = x * y
    z = x.mul(y)
    
    # Compute: x_sq = x^2 using element-wise power (or x * x for now)
    x_sq = x.mul(x)
    
    # Compute: f = z + x_sq
    f = z.add(x_sq)
    
    print(f"z = x * y = {z.data}")
    print(f"x^2 = {x_sq.data}")
    print(f"f = z + x_sq = {f.data}")
    
    # Forward pass
    result = GLOBAL_GRAPH.forward()
    print(f"Final result of f(x, y): {result.data}")
    
    # Analytical gradients:
    # f(x, y) = xy + x^2
    # df/dx = y + 2x = 3 + 2*2 = 7
    # df/dy = x = 2
    analytical_grad_x = 3 + 2*2  # y + 2x
    analytical_grad_y = 2        # x
    print(f"Analytical gradient df/dx: {analytical_grad_x}")
    print(f"Analytical gradient df/dy: {analytical_grad_y}")
    
    # Compute gradients using backpropagation
    GLOBAL_GRAPH.backward()
    print(f"Computed gradient df/dx: {x._node.grad}")
    print(f"Computed gradient df/dy: {y._node.grad}")
    
    # Example 3: Matrix operations and gradients
    print("\n3. Matrix operations and gradients")
    
    # Clear the global graph
    GLOBAL_GRAPH.nodes = []
    
    # Create matrix tensors
    A = Tensolr([[1.0, 2.0], [3.0, 4.0]], track_graph=True)
    B = Tensolr([[2.0, 0.0], [1.0, 2.0]], track_graph=True)
    
    print(f"Matrix A:\n{A.data}")
    print(f"Matrix B:\n{B.data}")
    
    # Compute C = A @ B (matrix multiplication)
    C = A.matmul(B)
    print(f"C = A @ B:\n{C.data}")
    
    # Compute D = C^T (transpose)
    D = C.transpose()
    print(f"D = C^T:\n{D.data}")
    
    # Compute scalar result as sum of all elements in D
    # For simplicity, we'll sum all elements by adding them repeatedly
    # In a real implementation, we'd have a sum operation
    sum_result = D
    
    # Forward pass
    result = GLOBAL_GRAPH.forward()
    print(f"Final matrix after operations:\n{result.data}")
    
    # Backward pass to compute gradients
    GLOBAL_GRAPH.backward()
    print(f"Gradient of A:\n{A._node.grad}")
    print(f"Gradient of B:\n{B._node.grad}")
    
    # Example 4: Chain rule verification
    print("\n4. Chain rule verification: f(g(h(x))) where h(x)=x^2, g(x)=2x, f(x)=x+1")
    
    # Clear the global graph
    GLOBAL_GRAPH.nodes = []
    
    # Create input
    x = Tensolr([[3.0]], track_graph=True)  # x = 3
    print(f"Input x: {x.data}")
    
    # h(x) = x^2
    h = x.mul(x)  # x^2
    print(f"h = x^2 = {h.data}")
    
    # g(h) = 2*h
    two = Tensolr([[2.0]], track_graph=True)
    g = h.mul(two)  # 2*x^2
    print(f"g = 2*h = {g.data}")
    
    # f(g) = g + 1
    one = Tensolr([[1.0]], track_graph=True)
    f = g.add(one)  # 2*x^2 + 1
    print(f"f = g + 1 = {f.data}")
    
    # Forward pass
    result = GLOBAL_GRAPH.forward()
    print(f"Final result f(g(h(x))): {result.data}")
    
    # Analytical gradient:
    # f(g(h(x))) = 2*x^2 + 1
    # df/dx = d/dx[2*x^2 + 1] = 4*x
    # At x=3: df/dx = 4*3 = 12
    analytical_final_grad = 4 * 3
    print(f"Analytical gradient df/dx at x=3: {analytical_final_grad}")
    
    # Compute gradients using backpropagation
    GLOBAL_GRAPH.backward()
    print(f"Computed gradient df/dx: {x._node.grad}")
    
    # Check if the chain rule is working correctly
    expected = analytical_final_grad
    actual = x._node.grad[0][0] if hasattr(x._node.grad, 'shape') else x._node.grad
    print(f"Chain rule verification: Expected={expected}, Actual={actual}, Match={np.isclose(expected, actual)}")
    
    print("\nComputation graphs and automatic differentiation example completed!")

if __name__ == "__main__":
    main()