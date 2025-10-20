"""
CPU Runner for Tensolr operations
Executes tensor operations on CPU using optimized implementations
"""
import numpy as np
from numba import jit
from ..tensor import Tensolr

class CPURunner:
    """Runner for executing tensor operations on CPU"""
    
    def __init__(self):
        pass
    
    @staticmethod
    @jit(nopython=True)
    def matmul_cpu(a, b):
        """Optimized CPU matrix multiplication using Numba"""
        m, k = a.shape
        k2, n = b.shape
        assert k == k2
        out = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                for l in range(k):
                    out[i, j] += a[i, l] * b[l, j]
        return out
    
    @staticmethod
    def add_cpu(a, b):
        """CPU element-wise addition"""
        return a + b
    
    @staticmethod
    def sub_cpu(a, b):
        """CPU element-wise subtraction"""
        return a - b
    
    @staticmethod
    def mul_cpu(a, b):
        """CPU element-wise multiplication"""
        return a * b
    
    @staticmethod
    def div_cpu(a, b):
        """CPU element-wise division"""
        return a / b
    
    @staticmethod
    def pow_cpu(a, b):
        """CPU element-wise power"""
        return a ** b
    
    @staticmethod
    def transpose_cpu(a):
        """CPU transpose"""
        return a.T
    
    def run_operation(self, op_name, *inputs):
        """Execute a specific operation on CPU"""
        if op_name == "add":
            return self.add_cpu(inputs[0], inputs[1])
        elif op_name == "sub":
            return self.sub_cpu(inputs[0], inputs[1])
        elif op_name == "mul":
            return self.mul_cpu(inputs[0], inputs[1])
        elif op_name == "div":
            return self.div_cpu(inputs[0], inputs[1])
        elif op_name == "pow":
            return self.pow_cpu(inputs[0], inputs[1])
        elif op_name == "matmul":
            return self.matmul_cpu(inputs[0], inputs[1])
        elif op_name == "transpose":
            return self.transpose_cpu(inputs[0])
        else:
            raise NotImplementedError(f"Operation {op_name} not implemented on CPU")
    
    def execute_graph(self, graph):
        """Execute an entire computation graph on CPU"""
        for node in graph.nodes:
            if node.op is None or node.op == "input":
                # Input nodes, just pass through
                node.output = node.tensor
            else:
                # Get input values from previous computations
                input_values = [inp.output.data for inp in node.inputs]
                
                # Run the operation on CPU
                result_data = self.run_operation(node.op, *input_values)
                
                # Create output tensor
                result_tensor = Tensolr(result_data, track_graph=False)
                
                # Update the node
                node.output = result_tensor
        
        # Return the final result
        return graph.nodes[-1].output if graph.nodes else None