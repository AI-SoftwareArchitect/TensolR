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
                elif node.op == "matmul":
                    node.output = inputs[0].matmul(inputs[1])
                elif node.op == "transpose":
                    node.output = inputs[0].transpose()
                else:
                    raise NotImplementedError(f"Unsupported op: {node.op}")
        return self.nodes[-1].output  # son node’un çıktısı

    def backward(self, loss_grad=None):
        """
        Reverse-mode autodiff:
        loss_grad: Sonuç (output) için başlangıç gradyanı, genelde np.ones_like(output)
        """
        if not self.nodes:
            return

        # Her node’a grad alanı ekle
        for node in self.nodes:
            node.grad = None

        # Son node’un grad’ı = 1 veya verilen loss_grad
        last = self.nodes[-1]
        if loss_grad is None:
            loss_grad = np.ones_like(last.output.data)
        last.grad = loss_grad

        # Geriye doğru yayılım (reverse order)
        for node in reversed(self.nodes):
            if node.op in ("add", "sub", "matmul", "transpose"):
                self._backprop_node(node)

    def _backprop_node(self, node):
        """Her node tipi için türev kuralları"""
        if node.op == "add":
            # d(a+b)/da = 1, d(a+b)/db = 1
            for inp in node.inputs:
                grad_contrib = node.grad
                self._accumulate_grad(inp, grad_contrib)

        elif node.op == "sub":
            # d(a-b)/da = 1, d(a-b)/db = -1
            self._accumulate_grad(node.inputs[0], node.grad)
            self._accumulate_grad(node.inputs[1], -node.grad)

        elif node.op == "matmul":
            a, b = node.inputs
            # dC/dA = dC/dOut @ B^T
            grad_a = node.grad @ b.output.data.T
            # dC/dB = A^T @ dC/dOut
            grad_b = a.output.data.T @ node.grad
            self._accumulate_grad(a, grad_a)
            self._accumulate_grad(b, grad_b)

        elif node.op == "transpose":
            # d/dA (A^T) = grad^T
            self._accumulate_grad(node.inputs[0], node.grad.T)

    def _accumulate_grad(self, node, grad):
        """Grad'ı biriktir (çünkü aynı tensor birden fazla yerde kullanılabilir)."""
        if getattr(node, "grad", None) is None:
            node.grad = grad
        else:
            node.grad += grad

    def __repr__(self):
        return f"Graph(nodes={[node.op for node in self.nodes]})"
