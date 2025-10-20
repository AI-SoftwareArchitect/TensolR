"""
Basic example showing Tensolr tensor operations
"""
import sys
import os

# Add the project root to the path so we can import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.tensor import Tensolr
from src.global_graph import GLOBAL_GRAPH

def main():
    print("=== Tensolr Basic Example ===")
    
    # Create tensors
    a = Tensolr([[1, 2], [3, 4]])
    b = Tensolr([[5, 6], [7, 8]])
    
    print(f"Tensor A:\n{a.data}")
    print(f"Tensor B:\n{b.data}")
    
    # Perform operations
    c = a.add(b)
    print(f"A + B:\n{c.data}")
    
    d = a.matmul(b)
    print(f"A @ B:\n{d.data}")
    
    e = a.transpose()
    print(f"A^T:\n{e.data}")
    
    # Operations with graph tracking
    print("\n=== With Graph Tracking ===")
    
    # Clear previous graph
    GLOBAL_GRAPH.nodes = []
    
    x = Tensolr([[1.0, 2.0]], track_graph=True)
    y = Tensolr([[3.0], [4.0]], track_graph=True)
    
    # Compute z = x @ y
    z = x.matmul(y)
    
    print(f"X: {x.data}")
    print(f"Y: {y.data}")
    print(f"Z = X @ Y: {z.data}")
    
    # Forward pass
    result = GLOBAL_GRAPH.forward()
    print(f"Forward pass result: {result.data}")
    
    # Backward pass
    GLOBAL_GRAPH.backward()
    print(f"Gradient of X: {x._node.grad}")
    print(f"Gradient of Y: {y._node.grad}")
    
    print("Basic example completed!")

if __name__ == "__main__":
    main()