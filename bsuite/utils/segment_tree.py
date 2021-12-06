import numpy as np
import operator
from typing import Callable, List

"""
This file implements the SegmentTree class and the SumTree & MinTree

Two sources really helped me to understand these datastructures:
http://blog.varunajayasiri.com/ml/dqn.html
and
https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
"""

class SegmentTree:
    def __init__(self, number_elements: int, merge_operation: Callable, neutral_element: float) -> None:
        """

        Args:
            number_elements: number of datapoints that  can be saved
            merge_operation: operation for merging to elements
            neutral_element: neutral element for the merge operation
        """
        self.number_elements = number_elements
        self.merge_operation = merge_operation
        self.data = np.ones(2 * number_elements) * neutral_element
        self.max_index = -1
        self.current_index = 0
        return

    def add(self, index: int, weight: float) -> None:
        """
        Adds one new element with a weight to data and updates the current index
        Args:
            index: index of the element ot update
            weight: weight of the new point
        Returns:
        """
        self.update(index, weight)
        if self.max_index < self.number_elements:
            self.max_index = self.max_index + 1
        return

    def update(self, index: int, weight: float) -> None:
        """
        Updates the weight of a specific index
        Args:
            index: index of the element ot update
            weight: new weight
        Returns:
        """
        array_index = index + self.number_elements
        self.data[array_index] = weight
        self.traverse(array_index)
        return

    def traverse(self, index: int) -> None:
        """
        Traveres the eleemnts of the segment tree upwards and updates the elements along the way
        Args:
            index: index of the starting point
        Returns:
        """
        data = self.data
        parent_index = self.parent(index)
        left_index = self.left_child(parent_index)
        right_index = self.right_child(parent_index)
        data[parent_index] = self.merge_operation(data[left_index], data[right_index])

        if parent_index == 1:
            return

        self.traverse(parent_index)

        return

    @staticmethod
    def left_child(index):
        return 2 * index

    @staticmethod
    def right_child(index):
        return 2 * index + 1

    @staticmethod
    def parent(index):
        return index // 2

    @property
    def weights(self):
        return self.data

    @property
    def root(self):
        """
        Returns:
            returns root value of the tree, sum over all elements for sum_tree, minimum for min tree
        """
        return self.data[1]


class SumTree(SegmentTree):
    def __init__(self, number_elements: int) -> None:
        """
        The sum tree keeps the cumulative sum and can return the index where this sum is larger than given value
        Args:
            number_elements: number of elements in the data structure
        """
        super().__init__(number_elements, operator.add, 0)
        return

    def get_index(self, value: float) -> int:
        """
        Gets the index where the cumulative sum of the previous elements is smaller than value and the sum up to
        index is larger or equal to value
        Args:
            value: float

        Returns:
            index - integer as defined above
        """
        index = 1
        while index < self.number_elements:
            left_index = self.left_child(index)
            right_index = self.right_child(index)
            if value <= self.data[left_index]:
                index = left_index
            else:
                value -= self.data[left_index]
                index = right_index

        index = index - self.number_elements
        if self.max_index < index:  # Edge case necessary?
            index = self.max_index
        return index

    def get_elements(self, indexes: List[int]) -> float:
        """

        Args:
            indexes(list(int)): indices of values to be returned

        Returns:
            float: priorities at indexes
        """
        number_elements = self.number_elements
        index = list(np.array(indexes) + number_elements)
        return self.data[index]


class MinTree(SegmentTree):

    def __init__(self, number_elements: int) -> None:
        """
        The min tree keeps track of the minimum element in the data structure
        Args:
            number_elements: number of elements in the data structure
        """
        super().__init__(number_elements, min, np.inf)
        return
