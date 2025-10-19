# src/graph.py
class Node:
    def __init__(self, tensor, op=None, inputs=None, name=None):
        self.tensor = tensor       # Tensolr Tensor objesi
        self.op = op               # İşlem adı: "add", "matmul" vb.
        self.inputs = inputs or [] # Önceki node’lar (dependencies)
        self.name = name           # Opsiyonel: node ismi
        self.output = None         # İşlem sonucu tensor
        self.grad = None           # Backprop için grad

    def __repr__(self):
        tensor_shape = self.tensor.shape if self.tensor is not None else "None"
        return f"Node(name={self.name}, op={self.op}, shape={tensor_shape})"
