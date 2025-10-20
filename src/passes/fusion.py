"""
Dead Code Elimination (DCE) Pass
Removes nodes from the computation graph that do not contribute to any output.
"""
from ..graph import Graph

class DCEPass:
    """Dead Code Elimination pass to remove unused nodes from the graph"""
    
    def __init__(self):
        pass
    
    def run(self, graph):
        """
        Run the DCE pass on the given graph
        Removes nodes that are not used by any other node in the graph
        """
        # Find all nodes that are used in the computation
        used_nodes = set()
        
        # Mark all nodes that are used as inputs to other nodes
        for node in graph.nodes:
            for input_node in node.inputs:
                used_nodes.add(input_node)
        
        # Mark the last node (output) as used if it exists
        if graph.nodes:
            used_nodes.add(graph.nodes[-1])
        
        # Keep only nodes that are used
        new_nodes = [node for node in graph.nodes if node in used_nodes]
        graph.nodes = new_nodes
        
        return graph


"""
Operator Fusion Pass
Combines consecutive operations that can be fused for performance.
"""
class FusionPass:
    """Operator fusion pass to combine consecutive operations"""
    
    def __init__(self):
        pass
    
    def run(self, graph):
        """
        Run the fusion pass on the given graph
        Combines compatible consecutive operations
        """
        i = 0
        while i < len(graph.nodes) - 1:
            current_node = graph.nodes[i]
            next_node = graph.nodes[i+1]
            
            # Look for fusion opportunities
            fused_node = self._try_fuse(current_node, next_node)
            
            if fused_node:
                # Replace the two nodes with the fused node
                graph.nodes[i] = fused_node
                # Remove the next node
                graph.nodes.pop(i+1)
                # Don't increment i, so we can check for more fusion opportunities
                # with the new fused node
            else:
                # Move to the next pair
                i += 1
        
        return graph
    
    def _try_fuse(self, node1, node2):
        """
        Try to fuse two consecutive operations
        Returns the fused node if fusion is possible, None otherwise
        """
        # Example fusion opportunities:
        # 1. Mul followed by Add (affine transformation) - not directly supported in current ops
        # 2. Multiple pointwise operations (Add, Mul, etc.)
        
        # For now, implement a simple fusion of two add operations
        if node1.op == "add" and node2.op == "add":
            # Create a fused node that represents (a + b) + c
            fused_node = type('FusedAdd', (), {
                'op': 'fused_add',
                'inputs': node1.inputs + [node2],  # This is a simplification
                'tensor': None,
                'grad': None,
                '__repr__': lambda self: f"FusedAddNode(op='fused_add')"
            })()
            return fused_node
        
        # More fusion opportunities could be added here
        # For example, element-wise operations that can be fused
        return None