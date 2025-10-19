import numpy as np
from numba import njit
from global_graph import GLOBAL_GRAPH
from node import Node

class Tensolr:
    def __init__(self, data, track_graph=True):
        self.data = np.array(data)
        self._node = None
        # Tensor oluşturulduğunda graph'a ekle
        if track_graph:
            self._node = Node(tensor=self, op="input", inputs=[])
            GLOBAL_GRAPH.add_node(self._node)

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def ndim(self):
        return self.data.ndim

    # -------------------
    # Statik Numba fonksiyonları (ndarray üzerinde)
    @staticmethod
    @njit
    def _zeros(shape):
        return np.zeros(shape)

    @staticmethod
    @njit
    def _ones(shape):
        return np.ones(shape)

    @staticmethod
    @njit
    def _add(a, b):
        return a + b

    @staticmethod
    @njit
    def _sub(a, b):
        return a - b

    @staticmethod
    @njit
    def _transpose(a):
        return a.T

    @staticmethod
    @njit
    def matmul_jit(a, b):
        m, k = a.shape
        k2, n = b.shape
        assert k == k2
        out = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                for l in range(k):
                    out[i, j] += a[i, l] * b[l, j]
        return out

    def matmul(self, other):
        if isinstance(other, Tensolr):
            result_data = Tensolr.matmul_jit(self.data, other.data)  # ✅ artık var
            result = Tensolr(result_data, track_graph=False)
            result._node = self._add_to_graph(other, "matmul")
            return result
        else:
            raise TypeError("Operand must be Tensolr instance")

    # -------------------
    # Factory metodları
    @classmethod
    def zeros(cls, shape):
        tensor = cls(cls._zeros(tuple(shape)))
        return tensor

    @classmethod
    def ones(cls, shape):
        tensor = cls(cls._ones(tuple(shape)))
        return tensor

    # -------------------
    # Operatorlar
    def add(self, other):
        if isinstance(other, Tensolr):
            result_data = self._add(self.data, other.data)
            result = Tensolr(result_data, track_graph=False)
            result._node = self._add_to_graph(other, "add")
            return result
        else:
            raise TypeError("Operand must be Tensolr instance")
    
    def sub(self, other):
        if isinstance(other, Tensolr):
            result_data = self._sub(self.data, other.data)
            result = Tensolr(result_data, track_graph=False)
            result._node = self._add_to_graph(other, "sub")
            return result
        else:
            raise TypeError("Operand must be Tensolr instance")
    
    def matmul(self, other):
        if isinstance(other, Tensolr):
            result_data = self.matmul_jit(self.data, other.data)
            result = Tensolr(result_data, track_graph=False)
            result._node = self._add_to_graph(other, "matmul")
            return result
        else:
            raise TypeError("Operand must be Tensolr instance")
    
    def transpose(self):
        result_data = self._transpose(self.data)
        result = Tensolr(result_data, track_graph=False)
        result._node = self._add_to_graph(self, "transpose")
        return result

    # -------------------
    # Graph ile bağlantı
    def _add_to_graph(self, other, op_name):
        input_nodes = []
        if isinstance(other, Tensolr):
            if self._node:
                input_nodes.append(self._node)
            else:
                input_nodes.append(Node(self))
            if other._node:
                input_nodes.append(other._node)
            else:
                input_nodes.append(Node(other))
        elif op_name == "transpose":
            # Transpose tek input alır
            if self._node:
                input_nodes.append(self._node)
            else:
                input_nodes.append(Node(self))
        else:
            raise TypeError("Operand must be Tensolr instance")

        new_node = Node(tensor=None, op=op_name, inputs=input_nodes)
        GLOBAL_GRAPH.add_node(new_node)
        return new_node


