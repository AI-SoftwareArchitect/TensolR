"""
Test script for the Tensolr monitoring system
"""
import sys
import os

# Add the project root to the path so we can import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.tensor import Tensolr
from src.monitor.monitor_service import init_monitoring
import time
import numpy as np

def test_monitoring():
    print("Initializing Tensolr monitoring...")
    monitor = init_monitoring("http://localhost:8000")  # This should match where the server runs
    
    print("Creating tensors and performing operations...")
    
    # Create some tensors
    print("Creating tensor A...")
    a = Tensolr([[1, 2], [3, 4]])
    
    print("Creating tensor B...")
    b = Tensolr([[5, 6], [7, 8]])
    
    print("Performing addition...")
    c = a.add(b)
    
    print("Performing multiplication...")
    d = a.mul(b)
    
    print("Performing matrix multiplication...")
    e = a.matmul(b)
    
    print("Creating more tensors to test monitoring...")
    for i in range(5):
        x = Tensolr(np.random.randn(10, 10))
        y = Tensolr(np.random.randn(10, 10))
        z = x.add(y)
        print(f"Operation {i+1} completed")
        time.sleep(0.5)  # Small delay to see changes in monitoring
    
    print("Test operations completed!")
    print("Check the monitoring dashboard at http://localhost:8000 to see the metrics")
    
    # Keep the process alive so monitoring can continue
    try:
        print("Monitoring will continue. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
        monitor.stop_monitoring()

if __name__ == "__main__":
    test_monitoring()