"""
Example comparing different optimizers
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
from src.api.optim import SGD, Adam, RMSprop

def create_dataset():
    """Create a simple regression dataset"""
    np.random.seed(42)
    X_data = np.random.randn(100, 3).astype(np.float32)  # 100 samples, 3 features
    # Create target as a linear combination of features + noise
    y_data = (2*X_data[:, 0:1] - 1.5*X_data[:, 1:2] + 0.5*X_data[:, 2:3] + 
              0.1*np.random.randn(100, 1)).astype(np.float32)
    return X_data, y_data

def train_model(model, X, y_true, optimizer, n_epochs=100, name="Optimizer"):
    """Train a model with a specific optimizer"""
    loss_fn = MSE()
    losses = []
    
    for epoch in range(n_epochs):
        # Forward pass
        y_pred = model(X)
        
        # Compute loss
        loss = loss_fn(y_pred, y_true)
        losses.append(loss.data)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Backward pass
        from src.global_graph import GLOBAL_GRAPH
        GLOBAL_GRAPH.nodes = []
        
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
        
    return losses

def main():
    print("=== Optimizer Comparison Example ===")
    
    # Create dataset
    X_data, y_data = create_dataset()
    X = Tensolr(X_data)
    y_true = Tensolr(y_data)
    
    print(f"Dataset shape: X={X.shape}, y={y_true.shape}")
    
    # Define model architecture
    model_architecture = Sequential(
        Linear(3, 10),
        ReLU(),
        Linear(10, 1)
    )
    
    # Test different optimizers
    optimizers = {
        "SGD": SGD,
        "Adam": Adam,
        "RMSprop": RMSprop
    }
    
    results = {}
    
    for opt_name, opt_class in optimizers.items():
        print(f"\nTraining with {opt_name} optimizer...")
        
        # Create a fresh model for each optimizer
        model = Sequential(
            Linear(3, 10),
            ReLU(),
            Linear(10, 1)
        )
        
        # Create optimizer with appropriate parameters
        if opt_name == "SGD":
            optimizer = opt_class(model.parameters(), lr=0.01)
        elif opt_name == "Adam":
            optimizer = opt_class(model.parameters(), lr=0.01)
        else:  # RMSprop
            optimizer = opt_class(model.parameters(), lr=0.01)
        
        # Train the model
        losses = train_model(model, X, y_true, optimizer, n_epochs=200, name=opt_name)
        results[opt_name] = losses
        
        # Final evaluation
        final_pred = model(X)
        final_loss = MSE()(final_pred, y_true)
        print(f"Final {opt_name} loss: {final_loss.data}")
    
    # Print comparison
    print(f"\nOptimizer Comparison (Final Loss after 200 epochs):")
    for opt_name, losses in results.items():
        print(f"{opt_name}: {losses[-1]}")
    
    # Show loss progression for the first few epochs
    print(f"\nLoss progression (first 10 epochs):")
    print("Epoch\tSGD\t\tAdam\t\tRMSprop")
    for i in range(0, min(10, len(list(results.values())[0])), 1):
        sgd_loss = results["SGD"][i] if "SGD" in results else "N/A"
        adam_loss = results["Adam"][i] if "Adam" in results else "N/A"
        rmsprop_loss = results["RMSprop"][i] if "RMSprop" in results else "N/A"
        print(f"{i}\t{sgd_loss:.4f}\t\t{adam_loss:.4f}\t\t{rmsprop_loss:.4f}")
    
    print("\nOptimizer comparison example completed!")

if __name__ == "__main__":
    main()