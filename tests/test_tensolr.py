"""
Comprehensive tests for Tensolr tensor operations and graph functionality
"""
import numpy as np
import sys
import os

# Add the project root to the path so we can import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.tensor import Tensolr
from src.graph import Graph
from src.global_graph import GLOBAL_GRAPH
from src.api.nn import Linear, ReLU, Sequential, MSE
from src.api.optim import SGD, Adam

def test_basic_operations():
    """Test basic tensor operations"""
    print("Testing basic operations...")
    
    # Test creation
    a = Tensolr([[1, 2], [3, 4]])
    b = Tensolr([[5, 6], [7, 8]])
    
    # Test shapes
    assert a.shape == (2, 2), f"Expected (2, 2), got {a.shape}"
    assert a.size == 4, f"Expected 4, got {a.size}"
    assert a.ndim == 2, f"Expected 2, got {a.ndim}"
    
    # Test addition
    c = a.add(b)
    expected = np.array([[6, 8], [10, 12]])
    assert np.allclose(c.data, expected), f"Add failed: expected {expected}, got {c.data}"
    
    # Test subtraction
    d = b.sub(a)
    expected = np.array([[4, 4], [4, 4]])
    assert np.allclose(d.data, expected), f"Sub failed: expected {expected}, got {d.data}"
    
    # Test multiplication
    e = a.mul(b)
    expected = np.array([[5, 12], [21, 32]])
    assert np.allclose(e.data, expected), f"Mul failed: expected {expected}, got {e.data}"
    
    # Test division
    f = b.div(a)
    expected = np.array([[5.0, 3.0], [7/3, 2.0]])
    assert np.allclose(f.data, expected), f"Div failed: expected {expected}, got {f.data}"
    
    # Test matrix multiplication
    g = a.matmul(b)
    expected = np.array([[19, 22], [43, 50]])
    assert np.allclose(g.data, expected), f"Matmul failed: expected {expected}, got {g.data}"
    
    # Test transpose
    h = a.transpose()
    expected = np.array([[1, 3], [2, 4]])
    assert np.allclose(h.data, expected), f"Transpose failed: expected {expected}, got {h.data}"
    
    print("Basic operations test passed!")

def test_graph_operations():
    """Test graph computation and backpropagation"""
    print("Testing graph operations...")
    
    # Clear the global graph
    GLOBAL_GRAPH.nodes = []
    
    # Create tensors with tracking enabled
    a = Tensolr([[1, 2], [3, 4]], track_graph=True)
    b = Tensolr([[5, 6], [7, 8]], track_graph=True)
    
    # Perform operations
    c = a.add(b)
    d = c.mul(a)
    
    # Forward pass
    result = GLOBAL_GRAPH.forward()
    
    # Expected: (([1,2],[3,4]) + ([5,6],[7,8])) * ([1,2],[3,4]) = ([6,8],[10,12]) * ([1,2],[3,4]) = ([6,16],[30,48])
    expected = np.array([[6, 16], [30, 48]])
    
    assert np.allclose(result.data, expected), f"Forward pass failed: expected {expected}, got {result.data}"
    
    # Backward pass
    GLOBAL_GRAPH.backward()
    
    # Check gradients
    # For d = c * a = (a + b) * a = a*a + b*a
    # d/da = 2*a + b
    # d/db = a
    expected_grad_a = np.array([[2*1+5, 2*2+6], [2*3+7, 2*4+8]])  # [[7, 10], [13, 16]]
    expected_grad_b = np.array([[1, 2], [3, 4]])
    
    assert a._node.grad is not None, "Gradient for 'a' should not be None"
    assert b._node.grad is not None, "Gradient for 'b' should not be None"
    
    assert np.allclose(a._node.grad, expected_grad_a), f"Gradient for 'a' incorrect: expected {expected_grad_a}, got {a._node.grad}"
    assert np.allclose(b._node.grad, expected_grad_b), f"Gradient for 'b' incorrect: expected {expected_grad_b}, got {b._node.grad}"
    
    print("Graph operations test passed!")

def test_neural_network_api():
    """Test neural network API"""
    print("Testing neural network API...")
    
    # Test Linear layer
    linear = Linear(3, 2)
    x = Tensolr([[1, 2, 3]])
    
    # Forward pass
    y = linear(x)
    assert y.shape == (1, 2), f"Linear layer output shape incorrect: expected (1, 2), got {y.shape}"
    
    # Test Sequential
    model = Sequential(
        Linear(3, 4),
        ReLU(),
        Linear(4, 1)
    )
    
    x = Tensolr([[1, 2, 3]])
    y = model(x)
    assert y.shape == (1, 1), f"Sequential model output shape incorrect: expected (1, 1), got {y.shape}"
    
    print("Neural network API test passed!")

def test_optimizers():
    """Test optimizers"""
    print("Testing optimizers...")
    
    # Create a simple model
    linear = Linear(2, 1)
    
    # Create some dummy data
    x = Tensolr([[1, 2], [3, 4]])
    y_true = Tensolr([[1], [0]])
    
    # Forward pass
    y_pred = linear(x)
    
    # Loss function
    loss_fn = MSE()
    loss = loss_fn(y_pred, y_true)
    
    # Backward pass through graph
    GLOBAL_GRAPH.nodes = []  # Clear previous nodes
    # Add nodes for this computation
    GLOBAL_GRAPH.add_node(x._node)
    GLOBAL_GRAPH.add_node(linear.weight.tensor._node)
    if linear.bias is not None:
        GLOBAL_GRAPH.add_node(linear.bias.tensor._node)
    GLOBAL_GRAPH.add_node(y_true._node)
    GLOBAL_GRAPH.add_node(y_pred._node)
    GLOBAL_GRAPH.add_node(loss._node)
    
    GLOBAL_GRAPH.backward()
    
    # Test SGD optimizer
    sgd = SGD(linear.parameters())
    sgd.zero_grad()
    
    # Just make sure the optimizer can be created and step can be called
    sgd.step()
    
    # Test Adam optimizer
    adam = Adam(linear.parameters())
    adam.zero_grad()
    adam.step()
    
    print("Optimizers test passed!")

def test_factory_functions():
    """Test tensor factory functions"""
    print("Testing factory functions...")
    
    # Test zeros
    z = Tensolr.zeros((2, 3))
    expected = np.zeros((2, 3))
    assert np.allclose(z.data, expected), f"Zeros failed: expected {expected}, got {z.data}"
    
    # Test ones
    o = Tensolr.ones((2, 3))
    expected = np.ones((2, 3))
    assert np.allclose(o.data, expected), f"Ones failed: expected {expected}, got {o.data}"
    
    # Test random
    r = Tensolr.randn((2, 3))
    assert r.shape == (2, 3), f"Randn shape failed: expected (2, 3), got {r.shape}"
    
    print("Factory functions test passed!")

def run_all_tests():
    """Run all tests"""
    print("Running comprehensive tests for Tensolr...")
    
    test_basic_operations()
    test_graph_operations()
    test_neural_network_api()
    test_optimizers()
    test_factory_functions()
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    run_all_tests()