import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # src klasörünü ekle

from tensor import Tensolr
from global_graph import GLOBAL_GRAPH
import numpy as np

def test_tensor_creation():
    a = Tensolr([[1, 2], [3, 4]])
    b = Tensolr([[5, 6], [7, 8]])

    assert np.array_equal(a.data, np.array([[1, 2], [3, 4]]))
    assert np.array_equal(b.data, np.array([[5, 6], [7, 8]]))
    assert a.shape == (2, 2)
    assert b.ndim == 2

def test_add_sub():
    a = Tensolr([[1, 2], [3, 4]])
    b = Tensolr([[5, 6], [7, 8]])

    c = a.add(b)
    d = b.sub(a)

    expected_add = np.array([[6, 8], [10, 12]])
    expected_sub = np.array([[4, 4], [4, 4]])

    assert np.array_equal(c.data, expected_add)
    assert np.array_equal(d.data, expected_sub)

def test_matmul_transpose():
    a = Tensolr([[1, 2], [3, 4]])
    b = Tensolr([[5, 6], [7, 8]])

    c = a.add(b)  # [[6, 8], [10, 12]]
    e = c.matmul(a.transpose())  # a.transpose() = [[1,3],[2,4]]
    f = e.transpose()

    expected_matmul = np.array([[22, 50], [34, 78]])  # düzeltildi
    expected_transpose = expected_matmul.T

    assert np.array_equal(e.data, expected_matmul)
    assert np.array_equal(f.data, expected_transpose)

def test_zeros_ones():
    zeros_tensor = Tensolr.zeros((2, 3))
    ones_tensor = Tensolr.ones((3, 2))

    assert np.array_equal(zeros_tensor.data, np.zeros((2, 3)))
    assert np.array_equal(ones_tensor.data, np.ones((3, 2)))

def test_global_graph_nodes():
    # Reset graph
    GLOBAL_GRAPH.nodes = []

    a = Tensolr([[1, 2], [3, 4]])
    b = Tensolr([[5, 6], [7, 8]])
    c = a.add(b)
    d = c.matmul(a.transpose())
    e = d.transpose()

    # Check that nodes were added
    nodes = GLOBAL_GRAPH.nodes
    assert len(nodes) >= 5  # input a, input b, add, matmul, transpose

    # Optionally check node ops
    ops = [node.op for node in nodes]
    assert "input" in ops
    assert "add" in ops
    assert "matmul" in ops
    assert "transpose" in ops
