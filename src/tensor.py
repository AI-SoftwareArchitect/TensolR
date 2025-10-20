import numpy as np
from numba import njit
import time
from .global_graph import GLOBAL_GRAPH
from .node import Node
try:
    from .monitor.monitor_service import get_monitor
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    get_monitor = None

class Tensolr:
    def __init__(self, data, track_graph=True):
        self.data = np.array(data)
        self._node = None
        # Generate a unique ID for this tensor for monitoring
        import uuid
        self._tensor_id = str(uuid.uuid4())
        
        # Tensor oluşturulduğunda graph'a ekle
        if track_graph:
            self._node = Node(tensor=self, op="input", inputs=[])
            GLOBAL_GRAPH.add_node(self._node)
        
        # Register with monitoring service if available
        if MONITORING_AVAILABLE:
            try:
                monitor = get_monitor()
                if monitor:
                    monitor.register_tensor(self._tensor_id, self.data.shape, str(self.data.dtype))
            except Exception as e:
                # If monitoring fails, continue without it
                pass

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
    def _mul(a, b):
        return a * b

    @staticmethod
    @njit
    def _div(a, b):
        return a / b

    @staticmethod
    @njit
    def _pow(a, b):
        return a ** b

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

    # -------------------
    # Factory metodları
    @classmethod
    def zeros(cls, shape):
        tensor = cls(cls._zeros(tuple(shape)))
        if MONITORING_AVAILABLE:
            try:
                monitor = get_monitor()
                if monitor:
                    monitor.log_operation("zeros", 0.0)
            except Exception:
                pass
        return tensor

    @classmethod
    def ones(cls, shape):
        tensor = cls(cls._ones(tuple(shape)))
        if MONITORING_AVAILABLE:
            try:
                monitor = get_monitor()
                if monitor:
                    monitor.log_operation("ones", 0.0)
            except Exception:
                pass
        return tensor

    @classmethod
    def randn(cls, shape):
        tensor = cls(np.random.randn(*shape))
        if MONITORING_AVAILABLE:
            try:
                monitor = get_monitor()
                if monitor:
                    monitor.log_operation("randn", 0.0)
            except Exception:
                pass
        return tensor

    # -------------------
    # Operatorlar
    def add(self, other):
        start_time = time.time() if MONITORING_AVAILABLE else 0
        if isinstance(other, Tensolr):
            result_data = self._add(self.data, other.data)
            result = Tensolr(result_data, track_graph=False)
            result._node = self._add_to_graph(other, "add")
            
            # Log the operation if monitoring is available
            if MONITORING_AVAILABLE:
                try:
                    duration = time.time() - start_time
                    monitor = get_monitor()
                    if monitor:
                        monitor.log_operation("add", duration)
                except Exception:
                    pass
            return result
        else:
            raise TypeError("Operand must be Tensolr instance")
    
    def sub(self, other):
        start_time = time.time() if MONITORING_AVAILABLE else 0
        if isinstance(other, Tensolr):
            result_data = self._sub(self.data, other.data)
            result = Tensolr(result_data, track_graph=False)
            result._node = self._add_to_graph(other, "sub")
            
            # Log the operation if monitoring is available
            if MONITORING_AVAILABLE:
                try:
                    duration = time.time() - start_time
                    monitor = get_monitor()
                    if monitor:
                        monitor.log_operation("sub", duration)
                except Exception:
                    pass
            return result
        else:
            raise TypeError("Operand must be Tensolr instance")
    
    def mul(self, other):
        start_time = time.time() if MONITORING_AVAILABLE else 0
        if isinstance(other, Tensolr):
            result_data = self._mul(self.data, other.data)
            result = Tensolr(result_data, track_graph=False)
            result._node = self._add_to_graph(other, "mul")
            
            # Log the operation if monitoring is available
            if MONITORING_AVAILABLE:
                try:
                    duration = time.time() - start_time
                    monitor = get_monitor()
                    if monitor:
                        monitor.log_operation("mul", duration)
                except Exception:
                    pass
            return result
        else:
            raise TypeError("Operand must be Tensolr instance")
    
    def div(self, other):
        start_time = time.time() if MONITORING_AVAILABLE else 0
        if isinstance(other, Tensolr):
            result_data = self._div(self.data, other.data)
            result = Tensolr(result_data, track_graph=False)
            result._node = self._add_to_graph(other, "div")
            
            # Log the operation if monitoring is available
            if MONITORING_AVAILABLE:
                try:
                    duration = time.time() - start_time
                    monitor = get_monitor()
                    if monitor:
                        monitor.log_operation("div", duration)
                except Exception:
                    pass
            return result
        else:
            raise TypeError("Operand must be Tensolr instance")
    
    def pow(self, other):
        start_time = time.time() if MONITORING_AVAILABLE else 0
        if isinstance(other, Tensolr):
            result_data = self._pow(self.data, other.data)
            result = Tensolr(result_data, track_graph=False)
            result._node = self._add_to_graph(other, "pow")
            
            # Log the operation if monitoring is available
            if MONITORING_AVAILABLE:
                try:
                    duration = time.time() - start_time
                    monitor = get_monitor()
                    if monitor:
                        monitor.log_operation("pow", duration)
                except Exception:
                    pass
            return result
        else:
            raise TypeError("Operand must be Tensolr instance")
    
    def matmul(self, other):
        start_time = time.time() if MONITORING_AVAILABLE else 0
        if isinstance(other, Tensolr):
            result_data = self.matmul_jit(self.data, other.data)
            result = Tensolr(result_data, track_graph=False)
            result._node = self._add_to_graph(other, "matmul")
            
            # Log the operation if monitoring is available
            if MONITORING_AVAILABLE:
                try:
                    duration = time.time() - start_time
                    monitor = get_monitor()
                    if monitor:
                        monitor.log_operation("matmul", duration)
                except Exception:
                    pass
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
            # Check if this tensor has a node
            if self._node is not None:
                input_nodes.append(self._node)
            else:
                # Create a node for this tensor if it doesn't have one
                self._node = Node(tensor=self, op="input", inputs=[])
                GLOBAL_GRAPH.add_node(self._node)
                input_nodes.append(self._node)
                
            # Check if the other tensor has a node
            if other._node is not None:
                input_nodes.append(other._node)
            else:
                # Create a node for the other tensor if it doesn't have one
                other._node = Node(tensor=other, op="input", inputs=[])
                GLOBAL_GRAPH.add_node(other._node)
                input_nodes.append(other._node)
        elif op_name == "transpose":
            # Transpose takes one input
            if self._node is not None:
                input_nodes.append(self._node)
            else:
                # Create a node for this tensor if it doesn't have one
                self._node = Node(tensor=self, op="input", inputs=[])
                GLOBAL_GRAPH.add_node(self._node)
                input_nodes.append(self._node)
        else:
            raise TypeError("Operand must be Tensolr instance")

        new_node = Node(tensor=None, op=op_name, inputs=input_nodes)
        GLOBAL_GRAPH.add_node(new_node)
        return new_node

    # -------------------
    # Python special methods for operators
    
    def __add__(self, other):
        if isinstance(other, Tensolr):
            return self.add(other)
        else:
            # Handle scalar addition by creating a Tensolr object from the scalar
            other_tensor = Tensolr(np.full(self.shape, other))
            return self.add(other_tensor)
    
    def __radd__(self, other):
        return self.__add__(other)  # Addition is commutative
    
    def __sub__(self, other):
        if isinstance(other, Tensolr):
            return self.sub(other)
        else:
            # Handle scalar subtraction by creating a Tensolr object from the scalar
            other_tensor = Tensolr(np.full(self.shape, other))
            return self.sub(other_tensor)
    
    def __rsub__(self, other):
        # For scalar - tensor, we need to reverse the operation
        other_tensor = Tensolr(np.full(self.shape, other))
        return other_tensor.sub(self)
    
    def __mul__(self, other):
        if isinstance(other, Tensolr):
            return self.mul(other)
        else:
            # Handle scalar multiplication by creating a Tensolr object from the scalar
            other_tensor = Tensolr(np.full(self.shape, other) if np.isscalar(other) else other)
            return self.mul(other_tensor)
    
    def __rmul__(self, other):
        return self.__mul__(other)  # Multiplication is commutative
    
    def __truediv__(self, other):
        if isinstance(other, Tensolr):
            return self.div(other)
        else:
            # Handle scalar division by creating a Tensolr object from the scalar
            other_tensor = Tensolr(np.full(self.shape, other))
            return self.div(other_tensor)
    
    def __rtruediv__(self, other):
        # For scalar / tensor, we need to reverse the operation
        other_tensor = Tensolr(np.full(self.shape, other))
        return other_tensor.div(self)
    
    def __pow__(self, other):
        if isinstance(other, Tensolr):
            return self.pow(other)
        else:
            # Handle scalar power by creating a Tensolr object from the scalar
            other_tensor = Tensolr(np.full(self.shape, other))
            return self.pow(other_tensor)
    
    def __neg__(self):
        return self.mul(Tensolr(-1))