"""
Example showing MLIR emission and optimization passes
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
from src.mlir_emitter import graph_to_mlir
from src.passes.dce import DCEPass
from src.passes.fusion import FusionPass
from src.runners.cpu_runner import CPURunner

def main():
    print("=== MLIR Emission and Optimization Passes Example ===")
    
    # Clear the global graph
    GLOBAL_GRAPH.nodes = []
    
    # Create tensors with tracking enabled
    a = Tensolr([[1, 2], [3, 4]], track_graph=True)
    b = Tensolr([[5, 6], [7, 8]], track_graph=True)
    
    # Perform operations to build a computation graph
    c = a.add(b)
    d = c.mul(a)  # This creates a small computation graph
    
    print("Original computation:")
    print(f"A: {a.data}")
    print(f"B: {b.data}")
    print(f"C = A + B: {c.data}")
    print(f"D = C * A: {d.data}")
    
    # Forward pass to compute result
    result = GLOBAL_GRAPH.forward()
    print(f"Final result: {result.data}")
    
    # Show the graph before optimization
    print(f"\nGraph nodes before optimization: {[node.op for node in GLOBAL_GRAPH.nodes]}")
    
    # Apply Dead Code Elimination pass
    dce_pass = DCEPass()
    optimized_graph = dce_pass.run(GLOBAL_GRAPH)
    print(f"Graph nodes after DCE: {[node.op for node in optimized_graph.nodes]}")
    
    # Apply Fusion pass
    fusion_pass = FusionPass()
    # Note: The fusion implementation in our framework is basic, 
    # but we demonstrate how it would be used
    fused_graph = fusion_pass.run(optimized_graph)
    print(f"Graph nodes after Fusion: {[node.op for node in fused_graph.nodes]}")
    
    # Convert to MLIR
    mlir_code = graph_to_mlir(fused_graph)
    print(f"\nMLIR representation:\n{mlir_code}")
    
    # Execute using CPU runner
    cpu_runner = CPURunner()
    execution_result = cpu_runner.execute_graph(fused_graph)
    print(f"\nResult from CPU runner: {execution_result.data}")
    
    print("\nMLIR emission and optimization example completed!")

if __name__ == "__main__":
    main()