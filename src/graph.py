import numpy as np
from functools import lru_cache

class Graph:
    def __init__(self):
        self.nodes = []  # Tüm node'ları tutar

    def add_node(self, node):
        self.nodes.append(node)

    def forward(self):
        """Tüm node’ları bağımlılık sırasına göre çalıştır"""
        for node in self.nodes:
            if node.op is None or node.op == "input":
                node.output = node.tensor
            else:
                inputs = [inp.output for inp in node.inputs]
                if node.op == "add":
                    node.output = inputs[0].add(inputs[1])
                elif node.op == "sub":
                    node.output = inputs[0].sub(inputs[1])
                elif node.op == "mul":
                    node.output = inputs[0].mul(inputs[1])
                elif node.op == "div":
                    node.output = inputs[0].div(inputs[1])
                elif node.op == "pow":
                    node.output = inputs[0].pow(inputs[1])
                elif node.op == "matmul":
                    node.output = inputs[0].matmul(inputs[1])
                elif node.op == "transpose":
                    node.output = inputs[0].transpose()
                else:
                    raise NotImplementedError(f"Unsupported op: {node.op}")
        return self.nodes[-1].output if self.nodes else None  # son node’un çıktısı

    def backward(self, loss_grad=None):
        """Reverse-mode autodiff optimized"""
        if not self.nodes:
            return

        # Node grad’larını sıfırla
        for node in self.nodes:
            if hasattr(node, 'grad'):
                node.grad = None

        # Son node için grad
        last = self.nodes[-1]
        if loss_grad is None:
            if last.output is not None and last.output.data is not None:
                loss_grad = np.ones_like(last.output.data)
            else:
                # If output is not available, we can't compute gradients
                return
        last.grad = loss_grad

        # Reverse order propagation
        for node in reversed(self.nodes):
            if node.op == "add":
                for inp in node.inputs:
                    self._accumulate_grad(inp, node.grad)
            elif node.op == "sub":
                if len(node.inputs) >= 2:
                    self._accumulate_grad(node.inputs[0], node.grad)
                    if node.grad is not None and node.inputs[1].output is not None:
                        self._accumulate_grad(node.inputs[1], -node.grad)
            elif node.op == "mul":
                if len(node.inputs) >= 2:
                    a, b = node.inputs
                    if node.grad is not None and a.output is not None and b.output is not None:
                        grad_a = node.grad * b.output.data
                        grad_b = node.grad * a.output.data
                        self._accumulate_grad(a, grad_a)
                        self._accumulate_grad(b, grad_b)
            elif node.op == "div":
                if len(node.inputs) >= 2:
                    a, b = node.inputs
                    if node.grad is not None and a.output is not None and b.output is not None:
                        grad_a = node.grad / b.output.data
                        grad_b = -node.grad * a.output.data / (b.output.data ** 2)
                        self._accumulate_grad(a, grad_a)
                        self._accumulate_grad(b, grad_b)
            elif node.op == "pow":
                if len(node.inputs) >= 2:
                    a, b = node.inputs
                    if node.grad is not None and a.output is not None and b.output is not None:
                        grad_a = node.grad * b.output.data * (a.output.data ** (b.output.data - 1))
                        grad_b = node.grad * (a.output.data ** b.output.data) * np.log(np.abs(a.output.data) + 1e-12)  # Safe log
                        self._accumulate_grad(a, grad_a)
                        self._accumulate_grad(b, grad_b)
            elif node.op == "matmul":
                if len(node.inputs) >= 2:
                    a, b = node.inputs
                    if node.grad is not None and a.output is not None and b.output is not None:
                        grad_a = node.grad @ b.output.data.T
                        grad_b = a.output.data.T @ node.grad
                        self._accumulate_grad(a, grad_a)
                        self._accumulate_grad(b, grad_b)
            elif node.op == "transpose":
                if len(node.inputs) >= 1:
                    self._accumulate_grad(node.inputs[0], node.grad.T)

    def _accumulate_grad(self, node, grad):
        """Grad’ı biriktir, direkt numpy array olarak"""
        if hasattr(node, 'grad') and node.grad is not None:
            # Check if shapes match for addition
            if node.grad.shape == grad.shape:
                node.grad += grad
            else:
                # Handle broadcasting cases
                if node.grad.ndim == grad.ndim:
                    node.grad += grad
                else:
                    # Sum over broadcasted dimensions
                    if hasattr(self, '_broadcast_sum'):
                        node.grad += self._broadcast_sum(grad, node.grad.shape)
                    else:
                        # Fallback: try to add with numpy broadcasting
                        node.grad += grad
        else:
            node.grad = grad.copy() if hasattr(grad, 'copy') else np.array(grad)

    def __repr__(self):
        return f"Graph(nodes={[node.op for node in self.nodes]})"
