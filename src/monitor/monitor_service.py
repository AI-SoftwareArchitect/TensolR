"""
Tensor Monitoring Service
Integrates with tensor operations to capture resource usage
"""
import psutil
import time
from typing import Optional, Dict, Any
import requests
import threading
import atexit
from functools import wraps

class TensorMonitor:
    """Monitors tensor operations and system resource usage"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.active_tensors = {}
        self.tensor_operations = 0
        self.monitoring = False
        self.session = requests.Session()
        
        # Register cleanup function
        atexit.register(self.cleanup)
    
    def start_monitoring(self):
        """Start the monitoring service"""
        self.monitoring = True
        print(f"Tensor monitoring started at {self.server_url}")
    
    def stop_monitoring(self):
        """Stop the monitoring service"""
        self.monitoring = False
        print("Tensor monitoring stopped")
    
    def cleanup(self):
        """Cleanup function called on exit"""
        self.stop_monitoring()
    
    def register_tensor(self, tensor_id: str, shape: tuple, dtype: str):
        """Register a new tensor for monitoring"""
        self.active_tensors[tensor_id] = {
            'shape': shape,
            'dtype': dtype,
            'created_at': time.time()
        }
        
        # Send to monitoring server
        try:
            response = self.session.post(
                f"{self.server_url}/api/tensor/register",
                json={
                    'tensor_id': tensor_id,
                    'shape': list(shape),
                    'dtype': dtype
                }
            )
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to register tensor with monitoring server: {e}")
    
    def remove_tensor(self, tensor_id: str):
        """Remove a tensor from monitoring"""
        if tensor_id in self.active_tensors:
            del self.active_tensors[tensor_id]
        
        # Send to monitoring server
        try:
            response = self.session.post(
                f"{self.server_url}/api/tensor/remove",
                json={'tensor_id': tensor_id}
            )
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to remove tensor from monitoring server: {e}")
    
    def log_operation(self, op_type: str, duration: float = 0.0):
        """Log a tensor operation"""
        self.tensor_operations += 1
        
        # Send to monitoring server
        try:
            response = self.session.post(
                f"{self.server_url}/api/tensor/operation",
                json={
                    'op_type': op_type,
                    'duration': duration
                }
            )
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to log operation with monitoring server: {e}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        # CPU usage percentage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_used = memory.used
        memory_total = memory.total
        
        # GPU usage (placeholder - would need actual GPU monitoring)
        gpu_percent = 0.0
        gpu_memory = 0.0
        
        # In a real system, we would use libraries like pynvml to get actual GPU metrics
        # For now, we'll just provide placeholders
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_percent = gpu.load * 100
                gpu_memory = gpu.memoryUsed  # in MB
        except ImportError:
            # If GPUtil is not available, we'll use placeholder values
            pass
        except Exception:
            # If GPU monitoring fails, use placeholders
            pass
        
        return {
            'timestamp': time.time(),
            'cpu_usage': cpu_percent,
            'memory_usage': memory_used,
            'memory_total': memory_total,
            'gpu_usage': gpu_percent,
            'gpu_memory': gpu_memory,
            'tensor_operations': self.tensor_operations,
            'active_tensors': len(self.active_tensors)
        }
    
    def monitor_decoration(self):
        """Decorator to monitor tensor operations"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log the operation
                self.log_operation(func.__name__, duration)
                
                return result
            return wrapper
        return decorator

# Global monitor instance
monitor = TensorMonitor()

def init_monitoring(server_url: str = "http://localhost:8001"):
    """Initialize the tensor monitoring service"""
    global monitor
    monitor = TensorMonitor(server_url)
    monitor.start_monitoring()
    return monitor

def get_monitor() -> TensorMonitor:
    """Get the global monitor instance"""
    return monitor