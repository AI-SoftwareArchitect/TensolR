"""
Training example with Tensolr neural network API
"""
import sys
import os

# Add the project root to the path so we can import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from src.tensor import Tensolr
from src.api.nn import Linear, ReLU, Sequential, MSE
from src.api.optim import SGD

# Initialize monitoring if the server is running
def initialize_monitoring():
    try:
        from src.monitor import init_monitoring
        # Initialize with the correct server URL (port 8001 as in your example)
        init_monitoring(server_url="http://localhost:8004")
        print("Monitoring initialized")
    except Exception as e:
        print(f"Could not initialize monitoring: {e}. Continuing without monitoring...")


def main():
    print("=== Tensolr Training Example ===")
    
    # Initialize monitoring (will only connect if monitoring server is running)
    initialize_monitoring()
    
    # Generate synthetic data: y = 2*x + 1 + noise
    np.random.seed(42)
    X_data = np.random.randn(100, 1).astype(np.float32)
    y_data = (2 * X_data + 1 + 0.1 * np.random.randn(100, 1)).astype(np.float32)
    
    # Convert to Tensolr tensors
    X = Tensolr(X_data)
    y_true = Tensolr(y_data)
    
    print(f"Input shape: {X.shape}, Output shape: {y_true.shape}")
    
    # Define model: simple linear regression
    model = Sequential(
        Linear(1, 1)  # 1 input, 1 output
    )
    
    # Define loss and optimizer
    loss_fn = MSE()
    optimizer = SGD(model.parameters(), lr=0.01)
    
    # Training loop
    n_epochs = 100
    for epoch in range(n_epochs):
        # Forward pass
        y_pred = model(X)
        
        # Compute loss
        loss = loss_fn(y_pred, y_true)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Backward pass - this will use the global graph
        from src.global_graph import GLOBAL_GRAPH
        GLOBAL_GRAPH.nodes = []  # Clear previous graph
        
        # Add all relevant nodes to the graph
        for param in model.parameters():
            GLOBAL_GRAPH.add_node(param.tensor._node)
        GLOBAL_GRAPH.add_node(X._node)
        GLOBAL_GRAPH.add_node(y_true._node)
        GLOBAL_GRAPH.add_node(y_pred._node)
        GLOBAL_GRAPH.add_node(loss._node)
        
        # Compute gradients
        GLOBAL_GRAPH.backward()
        
        # Update parameters
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.data}")
    
    # Print final parameters
    for i, param in enumerate(model.parameters()):
        print(f"Parameter {i}: {param.tensor.data}")
    
    print("Training example completed!")

if __name__ == "__main__":
    main()