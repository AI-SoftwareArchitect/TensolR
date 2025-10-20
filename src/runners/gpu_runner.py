"""
GPU Runner for Tensolr operations
Executes tensor operations on GPU using CUDA kernels when available
"""
import numpy as np

class GPURunner:
    """Runner for executing tensor operations on GPU"""
    
    def __init__(self):
        self.gpu_available = self._check_gpu_availability()
        if self.gpu_available:
            try:
                import cupy
                self.xp = cupy
            except ImportError:
                import numpy as xp
                self.xp = xp
                self.gpu_available = False
        else:
            import numpy as xp
            self.xp = xp
    
    def _check_gpu_availability(self):
        """Check if GPU is available for computation"""
        try:
            import cupy
            # Check if cupy can access a GPU
            if hasattr(cupy, 'cuda') and cupy.cuda.is_available():
                return True
            return False
        except ImportError:
            return False
    
    def _to_gpu(self, array):
        """Move array to GPU if available"""
        if self.gpu_available:
            import cupy
            return cupy.asarray(array)
        else:
            return np.asarray(array)
    
    def _from_gpu(self, array):
        """Move array from GPU to CPU"""
        if self.gpu_available:
            import cupy
            if isinstance(array, cupy.ndarray):
                return cupy.asnumpy(array)
        return array
    
    def matmul_gpu(self, a, b):
        """GPU matrix multiplication"""
        a_gpu = self._to_gpu(a)
        b_gpu = self._to_gpu(b)
        result_gpu = self.xp.matmul(a_gpu, b_gpu)
        return self._from_gpu(result_gpu)
    
    def add_gpu(self, a, b):
        """GPU element-wise addition"""
        a_gpu = self._to_gpu(a)
        b_gpu = self._to_gpu(b)
        result_gpu = a_gpu + b_gpu
        return self._from_gpu(result_gpu)
    
    def sub_gpu(self, a, b):
        """GPU element-wise subtraction"""
        a_gpu = self._to_gpu(a)
        b_gpu = self._to_gpu(b)
        result_gpu = a_gpu - b_gpu
        return self._from_gpu(result_gpu)
    
    def mul_gpu(self, a, b):
        """GPU element-wise multiplication"""
        a_gpu = self._to_gpu(a)
        b_gpu = self._to_gpu(b)
        result_gpu = a_gpu * b_gpu
        return self._from_gpu(result_gpu)
    
    def div_gpu(self, a, b):
        """GPU element-wise division"""
        a_gpu = self._to_gpu(a)
        b_gpu = self._to_gpu(b)
        result_gpu = a_gpu / b_gpu
        return self._from_gpu(result_gpu)
    
    def pow_gpu(self, a, b):
        """GPU element-wise power"""
        a_gpu = self._to_gpu(a)
        b_gpu = self._to_gpu(b)
        result_gpu = a_gpu ** b_gpu
        return self._from_gpu(result_gpu)
    
    def transpose_gpu(self, a):
        """GPU transpose"""
        a_gpu = self._to_gpu(a)
        result_gpu = a_gpu.T
        return self._from_gpu(result_gpu)
    
    def run_operation(self, op_name, *inputs):
        """Execute a specific operation on GPU"""
        if op_name == "add":
            return self.add_gpu(inputs[0], inputs[1])
        elif op_name == "sub":
            return self.sub_gpu(inputs[0], inputs[1])
        elif op_name == "mul":
            return self.mul_gpu(inputs[0], inputs[1])
        elif op_name == "div":
            return self.div_gpu(inputs[0], inputs[1])
        elif op_name == "pow":
            return self.pow_gpu(inputs[0], inputs[1])
        elif op_name == "matmul":
            return self.matmul_gpu(inputs[0], inputs[1])
        elif op_name == "transpose":
            return self.transpose_gpu(inputs[0])
        else:
            # If operation is not implemented on GPU, fall back to CPU
            raise NotImplementedError(f"Operation {op_name} not implemented on GPU")
    
    def execute_graph(self, graph):
        """Execute an entire computation graph on GPU"""
        if not self.gpu_available:
            print("Warning: GPU not available, falling back to CPU")
            from .cpu_runner import CPURunner
            cpu_runner = CPURunner()
            return cpu_runner.execute_graph(graph)
        
        for node in graph.nodes:
            if node.op is None or node.op == "input":
                # Input nodes, just pass through
                node.output = node.tensor
            else:
                # Get input values from previous computations
                input_values = [self._to_gpu(inp.output.data) for inp in node.inputs]
                
                # Run the operation on GPU
                result_data = self.run_operation(node.op, *input_values)
                
                # Create output tensor
                from ..tensor import Tensolr
                result_tensor = Tensolr(result_data, track_graph=False)
                
                # Update the node
                node.output = result_tensor
        
        # Return the final result
        return graph.nodes[-1].output if graph.nodes else None