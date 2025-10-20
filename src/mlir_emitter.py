"""
MLIR Emitter for Tensolr tensor operations.
This module converts Tensolr computation graphs to MLIR representation.
"""
import numpy as np
from .tensor import Tensolr
from .graph import Graph

class MLIREmitter:
    """Converts Tensolr computation graphs to MLIR representation"""
    
    def __init__(self):
        self.mlir_operations = []
        self.tensor_map = {}  # Map from Tensolr tensors to MLIR tensor names
        self.current_id = 0
    
    def new_tensor_name(self):
        """Generate a new unique tensor name"""
        name = f"tensor_{self.current_id}"
        self.current_id += 1
        return name
    
    def emit_constant(self, tensor, name=None):
        """Emit an MLIR constant operation from a numpy array"""
        if name is None:
            name = self.new_tensor_name()
        
        # Convert tensor data to MLIR constant format
        if isinstance(tensor, Tensolr):
            data = tensor.data
        else:
            data = tensor
            
        # Create MLIR constant operation
        shape_str = "x".join(map(str, data.shape))
        dtype_str = str(data.dtype).replace('float', 'f').replace('int', 'i')
        
        # Convert numpy array to MLIR dense elements format
        if data.ndim == 0:
            # Scalar
            value_str = str(data.item())
        elif data.ndim == 1:
            # 1D vector
            value_str = "[" + ", ".join(map(str, data.tolist())) + "]"
        else:
            # Multi-dimensional tensor - we'll represent as nested brackets
            value_str = self._array_to_mlir_format(data)
        
        mlir_op = f'  {name} = "std.constant"() {{value = dense<{value_str}> : tensor<{shape_str}x{dtype_str}>}} : () -> tensor<{shape_str}x{dtype_str}>'
        self.mlir_operations.append(mlir_op)
        return name
    
    def _array_to_mlir_format(self, arr):
        """Convert numpy array to MLIR dense format"""
        if arr.ndim == 1:
            return "[" + ", ".join(map(str, arr.tolist())) + "]"
        else:
            # For multi-dimensional arrays, recursively convert
            sub_arrays = [self._array_to_mlir_format(arr[i]) for i in range(arr.shape[0])]
            return "[" + ", ".join(sub_arrays) + "]"
    
    def emit_add(self, operand1, operand2, result_name=None):
        """Emit MLIR add operation"""
        if result_name is None:
            result_name = self.new_tensor_name()
            
        mlir_op = f'  {result_name} = "std.addf"({operand1}, {operand2}) : (tensor, tensor) -> tensor'
        self.mlir_operations.append(mlir_op)
        return result_name
    
    def emit_sub(self, operand1, operand2, result_name=None):
        """Emit MLIR subtract operation"""
        if result_name is None:
            result_name = self.new_tensor_name()
            
        mlir_op = f'  {result_name} = "std.subf"({operand1}, {operand2}) : (tensor, tensor) -> tensor'
        self.mlir_operations.append(mlir_op)
        return result_name
    
    def emit_mul(self, operand1, operand2, result_name=None):
        """Emit MLIR multiply operation"""
        if result_name is None:
            result_name = self.new_tensor_name()
            
        mlir_op = f'  {result_name} = "std.mulf"({operand1}, {operand2}) : (tensor, tensor) -> tensor'
        self.mlir_operations.append(mlir_op)
        return result_name
    
    def emit_div(self, operand1, operand2, result_name=None):
        """Emit MLIR divide operation"""
        if result_name is None:
            result_name = self.new_tensor_name()
            
        mlir_op = f'  {result_name} = "std.divf"({operand1}, {operand2}) : (tensor, tensor) -> tensor'
        self.mlir_operations.append(mlir_op)
        return result_name
    
    def emit_matmul(self, operand1, operand2, result_name=None):
        """Emit MLIR matrix multiplication operation"""
        if result_name is None:
            result_name = self.new_tensor_name()
            
        mlir_op = f'  {result_name} = "linalg.matmul"({operand1}, {operand2}) {{}} : (tensor, tensor) -> tensor'
        self.mlir_operations.append(mlir_op)
        return result_name
    
    def emit_transpose(self, operand, result_name=None):
        """Emit MLIR transpose operation"""
        if result_name is None:
            result_name = self.new_tensor_name()
            
        mlir_op = f'  {result_name} = "linalg.transpose"({operand}) {{permutation = [1, 0]}} : (tensor) -> tensor'
        self.mlir_operations.append(mlir_op)
        return result_name
    
    def emit_from_graph(self, graph):
        """Convert a Tensolr computation graph to MLIR"""
        self.mlir_operations = []
        
        # Process nodes in forward order
        for node in graph.nodes:
            if node.op == "input":
                # Create a tensor name for input nodes
                tensor_name = self.new_tensor_name()
                self.tensor_map[node] = tensor_name
            elif node.op in ["add", "sub", "mul", "div", "matmul", "transpose"]:
                # Get input tensors for the operation
                input_names = []
                for input_node in node.inputs:
                    if input_node in self.tensor_map:
                        input_names.append(self.tensor_map[input_node])
                    else:
                        # Input tensor not yet created, create a placeholder
                        input_name = self.emit_constant(input_node.tensor.data)
                        self.tensor_map[input_node] = input_name
                        input_names.append(input_name)
                
                # Emit the appropriate operation
                if node.op == "add":
                    result_name = self.emit_add(input_names[0], input_names[1])
                elif node.op == "sub":
                    result_name = self.emit_sub(input_names[0], input_names[1])
                elif node.op == "mul":
                    result_name = self.emit_mul(input_names[0], input_names[1])
                elif node.op == "div":
                    result_name = self.emit_div(input_names[0], input_names[1])
                elif node.op == "matmul":
                    result_name = self.emit_matmul(input_names[0], input_names[1])
                elif node.op == "transpose":
                    result_name = self.emit_transpose(input_names[0])
                
                # Map the result node to the MLIR tensor name
                self.tensor_map[node] = result_name
        
        # Return the complete MLIR module
        return self.get_mlir_module()
    
    def get_mlir_module(self):
        """Get the complete MLIR module string"""
        module_start = "module {"
        module_end = "\n}"
        
        # Indent operations
        indented_ops = ["  " + op for op in self.mlir_operations]
        
        return module_start + "\n" + "\n".join(indented_ops) + module_end


# Function to convert a graph to MLIR
def graph_to_mlir(graph):
    """Convert a Tensolr graph to MLIR representation"""
    emitter = MLIREmitter()
    return emitter.emit_from_graph(graph)