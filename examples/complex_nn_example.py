"""
Example showing complex neural network architectures
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
from src.api.optim import SGD, Adam

def main():
    print("=== Complex Neural Network Architectures Example ===")
    
    # Create synthetic dataset: XOR problem
    # XOR: 0^0=0, 0^1=1, 1^0=1, 1^1=0
    X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    print("XOR Dataset:")
    print("Input -> Output")
    for i in range(len(X_data)):
        print(f"{X_data[i]} -> {y_data[i]}")
    
    # Convert to Tensolr tensors
    X = Tensolr(X_data)
    y_true = Tensolr(y_data)
    
    # Define a more complex model: Multi-layer perceptron
    model = Sequential(
        Linear(2, 4),  # Input: 2 features, Hidden: 4 neurons
        ReLU(),
        Linear(4, 8), # Hidden: 4 to 8 neurons
        ReLU(),
        Linear(8, 1)  # Output: 1 neuron for binary classification
    )
    
    # Define loss and optimizer
    loss_fn = MSE()
    optimizer = Adam(model.parameters(), lr=0.1)
    
    print(f"\nModel architecture:")
    for i, layer in enumerate(model.modules):
        if hasattr(layer, 'weight'):
            print(f"Layer {i}: Linear({layer.in_features} -> {layer.out_features})")
        else:
            print(f"Layer {i}: {layer.__class__.__name__}")
    
    # Training loop
    n_epochs = 1000
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
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.data}")
    
    # Final evaluation
    final_predictions = model(X)
    print(f"\nFinal predictions after {n_epochs} epochs:")
    for i in range(len(X_data)):
        predicted = final_predictions.data[i][0]
        actual = y_data[i][0]
        print(f"Input {X_data[i]} -> Predicted: {predicted:.3f}, Actual: {actual}")
    
    # Calculate accuracy
    correct = 0
    for i in range(len(final_predictions.data)):
        pred = 1 if final_predictions.data[i][0] > 0.5 else 0
        actual = int(y_data[i][0])
        if pred == actual:
            correct += 1
    
    accuracy = correct / len(y_data)
    print(f"\nFinal accuracy: {accuracy * 100:.2f}%")
    
    print("\nComplex neural network example completed!")

if __name__ == "__main__":
    main()