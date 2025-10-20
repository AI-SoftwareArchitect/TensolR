import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # src klasörünü ekle

import pytest
from tensor import Tensolr
from global_graph import GLOBAL_GRAPH
import numpy as np

@pytest.fixture(autouse=True)
def reset_graph():
    """Her testten önce global graph'ı sıfırla."""
    GLOBAL_GRAPH.nodes.clear()
    yield
    GLOBAL_GRAPH.nodes.clear()


def test_forward_add_backward():
    a = Tensolr([[1.0, 2.0]])
    b = Tensolr([[3.0, 4.0]])
    c = a.add(b)

    # Forward pass
    out = GLOBAL_GRAPH.forward()
    np.testing.assert_array_equal(out.data, np.array([[4.0, 6.0]]))

    # Backward pass
    GLOBAL_GRAPH.backward()

    # Gradients kontrolü
    grad_a = GLOBAL_GRAPH.nodes[0].grad
    grad_b = GLOBAL_GRAPH.nodes[1].grad
    np.testing.assert_array_equal(grad_a, np.ones_like(a.data))
    np.testing.assert_array_equal(grad_b, np.ones_like(b.data))


def test_forward_sub_backward():
    a = Tensolr([[5.0, 7.0]])
    b = Tensolr([[2.0, 1.0]])
    c = a.sub(b)

    out = GLOBAL_GRAPH.forward()
    np.testing.assert_array_equal(out.data, np.array([[3.0, 6.0]]))

    GLOBAL_GRAPH.backward()

    grad_a = GLOBAL_GRAPH.nodes[0].grad
    grad_b = GLOBAL_GRAPH.nodes[1].grad
    np.testing.assert_array_equal(grad_a, np.ones_like(a.data))
    np.testing.assert_array_equal(grad_b, -np.ones_like(b.data))


def test_forward_matmul_backward():
    a = Tensolr([[1.0, 2.0, 3.0]])
    b = Tensolr([[4.0], [5.0], [6.0]])
    c = a.matmul(b)  # scalar output: [[32.0]]

    out = GLOBAL_GRAPH.forward()
    assert out.data.shape == (1, 1)
    np.testing.assert_allclose(out.data, np.array([[32.0]]))

    GLOBAL_GRAPH.backward()

    # dC/dA = grad_out @ B^T  → [[4,5,6]]
    grad_a = GLOBAL_GRAPH.nodes[0].grad
    np.testing.assert_allclose(grad_a, np.array([[4.0, 5.0, 6.0]]))

    # dC/dB = A^T @ grad_out  → [[1],[2],[3]]
    grad_b = GLOBAL_GRAPH.nodes[1].grad
    np.testing.assert_allclose(grad_b, np.array([[1.0], [2.0], [3.0]]))


def test_forward_transpose_backward():
    a = Tensolr([[1.0, 2.0, 3.0]])
    t = a.transpose()

    out = GLOBAL_GRAPH.forward()
    np.testing.assert_array_equal(out.data, np.array([[1.0], [2.0], [3.0]]))

    GLOBAL_GRAPH.backward()

    # transpose için grad geri çevrilir (T alınır)
    grad_a = GLOBAL_GRAPH.nodes[0].grad
    np.testing.assert_array_equal(grad_a, np.ones_like(a.data))

@pytest.fixture(autouse=True)
def reset_graph():
    GLOBAL_GRAPH.nodes.clear()
    yield
    GLOBAL_GRAPH.nodes.clear()
