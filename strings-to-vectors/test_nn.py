import numpy as np
from numpy.testing import assert_array_equal

import nn


def test_indexes():
    index = nn.Index(["four", "three", "two", "one"])
    objects = ["one", "four", "four"]
    indexes = np.array([3, 0, 0])
    assert_array_equal(index.objects_to_indexes(objects), indexes)
    assert index.indexes_to_objects(indexes) == objects


def test_duplicates_in_vocabulary():
    index = nn.Index([1, 4, 3, 1, 1, 2, 5, 2, 4])
    objects = [1, 2, 3, 4]
    indexes = np.array([0, 3, 2, 1])
    assert_array_equal(index.objects_to_indexes(objects), indexes)
    assert index.indexes_to_objects(indexes) == objects


def test_out_of_vocabulary():
    index = nn.Index("abcd")
    objects = "bcde"
    indexes = np.array([1, 2, 3, -1])
    assert_array_equal(index.objects_to_indexes(objects), indexes)
    assert index.indexes_to_objects(indexes) == ["b", "c", "d"]


def test_start():
    index = nn.Index("abcd", start=1)
    objects = "bcde"
    indexes = np.array([2, 3, 4, 0])
    assert_array_equal(index.objects_to_indexes(objects), indexes)
    assert index.indexes_to_objects(indexes) == ["b", "c", "d"]


def test_binary_vector():
    index = nn.Index("she sells seashells by the seashore".split())
    objects = "the seashells she sells".split()
    sorted_objects = "she sells seashells the".split()
    vector = np.array([1, 1, 1, 0, 1, 0])
    assert_array_equal(index.objects_to_binary_vector(objects), vector)
    assert index.binary_vector_to_objects(vector) == sorted_objects


def test_empty_vector():
    index = nn.Index("she sells seashells by the seashore".split())
    objects = []
    vector = np.array([0, 0, 0, 0, 0, 0])
    assert_array_equal(index.objects_to_binary_vector(objects), vector)
    assert index.binary_vector_to_objects(vector) == objects


def test_index_matrix():
    index = nn.Index("abcdef")
    objects = [["a", "b", "c"],
               ["f", "e", "d"]]
    matrix = np.array([[0, 1, 2],
                       [5, 4, 3]])
    assert_array_equal(index.objects_to_index_matrix(objects), matrix)
    assert index.index_matrix_to_objects(matrix) == objects


def test_ragged_index_matrix():
    index = nn.Index("abcdef", start=1)
    objects = [["a", "b", "c"],
               ["e"]]
    matrix = np.array([[1, 2, 3],
                       [5, 0, 0]])
    assert_array_equal(index.objects_to_index_matrix(objects), matrix)
    assert index.index_matrix_to_objects(matrix) == objects


def test_binary_matrix():
    index = nn.Index("abcdef")
    objects = [["a", "b", "c"],
               ["e"]]
    matrix = np.array([[1, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0]])
    assert_array_equal(index.objects_to_binary_matrix(objects), matrix)
    assert index.binary_matrix_to_objects(matrix) == objects
