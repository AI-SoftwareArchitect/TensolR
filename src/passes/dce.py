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