from typing import Sequence, Any

import numpy as np


class Index:

    def __init__(self, vocab: Sequence[Any], start=0):
        """
        Assigns an index to each unique word in the `vocab` iterable,
        with indexes starting from `start`.
        """
        seq_of_idxs = {}
        idxs_to_seq = {}
        c = start
        for seq in vocab:
            if seq not in seq_of_idxs:
                seq_of_idxs[seq] = c
                idxs_to_seq[c] = seq
                c+=1

        self.seq_of_idxs = seq_of_idxs
        self.idxs_to_seq = idxs_to_seq
        self.start= start
        self.seq = vocab


    def objects_to_indexes(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a vector of the indexes associated with the input objects.

        For objects not in the vocabulary, use `start-1` as the index.

        :param object_seq: A sequence of objects
        :return: A 1-dimensional array of the object indexes.
        """
        object_indices = []
        for seq in object_seq:
            if seq not in self.seq_of_idxs:
                object_indices.append(self.start -1)
            else:
                object_indices.append(self.seq_of_idxs[seq])
        return np.array(object_indices)

    def objects_to_index_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a matrix of the indexes associated with the input objects.

        :param object_seq_seq: A sequence of sequences of objects
        :return: A 2-dimensional array of the object indexes.
        """
        objects_index = []
        mask_len = len(object_seq_seq[0])
        for i in object_seq_seq:
            temp_idxs = []
            for j in i:
                if j in self.seq_of_idxs:
                    temp_idxs.append(self.seq_of_idxs[j])
                else:
                    temp_idxs.append(self.start -1)   #Due to the comment for the previous method!
            for k in range(len(temp_idxs), mask_len):
                temp_idxs.append(0)
            objects_index.append(np.array(temp_idxs))

        return np.array(objects_index)


    def objects_to_binary_vector(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a binary vector, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq: A sequence of objects
        :return: A 1-dimensional array, with 1s at the indexes of each object,
                 and 0s at all other indexes
        """
        object_indices = [0] * len(self.seq)
        count =0
        for seq in object_seq:
            if seq in self.seq:
                object_indices[self.seq.index(seq)] =1
            count+=1
        return np.array(object_indices)

    def objects_to_binary_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a binary matrix, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq_seq: A sequence of sequences of objects
        :return: A 2-dimensional array, where each row in the array corresponds
                 to a row in the input, with 1s at the indexes of each object,
                 and 0s at all other indexes
        """
        objects_index = []
        for i in object_seq_seq:
            temp_idxs = [0] * len(self.seq)
            for j in i:
                if j in self.seq:
                    temp_idxs[self.seq.index(j)] = 1

            objects_index.append(np.array(temp_idxs))

        return np.array(objects_index)

    def indexes_to_objects(self, index_vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of objects associated with the indexes in the input
        vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_vector: A 1-dimensional array of indexes
        :return: A sequence of objects, one for each index.
        """
        index_objects = []
        for i in index_vector:
            if i in self.idxs_to_seq:
                index_objects.append(self.idxs_to_seq[i])
            #if i in self.seq_of_idxs.values():
                #index_objects.append(list(self.seq_of_idxs.keys())[list(self.seq_of_idxs.values()).index(i)])
        return index_objects

    def index_matrix_to_objects(
            self, index_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects associated with the indexes
        in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_matrix: A 2-dimensional array of indexes
        :return: A sequence of sequences of objects, one for each index.
        """
        seq_seq = []
        for i in index_matrix:
            temp_seq = []
            for j in i:
                if j in self.idxs_to_seq:
                    temp_seq.append(self.idxs_to_seq[j])
            seq_seq.append(temp_seq)

        return seq_seq

    def binary_vector_to_objects(self, vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of the objects identified by the nonzero indexes in
        the input vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param vector: A 1-dimensional binary array
        :return: A sequence of objects, one for each nonzero index.
        """
        index_objects = []
        count = 0
        for i in vector:
            if i == 1:
                index_objects.append(self.seq[count])
            count+=1
        return index_objects

    def binary_matrix_to_objects(
            self, binary_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects identified by the nonzero
        indices in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param binary_matrix: A 2-dimensional binary array
        :return: A sequence of sequences of objects, one for each nonzero index.
        """
        seq_seq = []
        for i in binary_matrix:
            temp_seq = []
            count = 0
            for j in i:
                if j == 1:
                    temp_seq.append(self.seq[count])
                count+=1
            seq_seq.append(temp_seq)

        return seq_seq