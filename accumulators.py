from coffea.processor import AccumulatorABC
import numpy as np

class HistAccumulator(AccumulatorABC):
    '''Holds a value constant-size numpy array
    Parameters
    ----------
        size : int
            size of the array
        initial : np.array, optional
            an initial value, if the np.zeros is not the desired initial value
    '''
    def __init__(self, size, initial=None):
        self.value = np.zeros(size) if initial is None else initial
        self.size = size

    def __repr__(self):
        return "HistAccumulator(%r)" % (self.value)

    def identity(self):
        return HistAccumulator(self.size)

    def add(self, other):
        if isinstance(other, HistAccumulator):
            self.value = self.value + other.value
        else:
            self.value = self.value + other

class ColumnAccumulator(AccumulatorABC):
    '''Like column_accumulator, but better'''
    def __init__(self, dtype):
        self.dtype = dtype
        self.value = np.zeros(dtype=dtype, shape=(0,))

    def __repr__(self):
        return "ColumnAccumulator(%r)" % (self.value)

    def identity(self):
        return ColumnAccumulator(self.dtype)

    def add(self, other):
        if isinstance(other, ColumnAccumulator):
            if other.value.shape[1:] != self.value.shape[1:]:
                raise ValueError(
                    "Cannot add two ColumnAccumulator objects of dissimilar shape (%r vs %r)"
                    % (self.value.shape[1:], other.value.shape[1:])
                )
            if other.value.dtype != self.value.dtype:
                raise ValueError(
                    "Cannot add two ColumnAccumulator objects of dissimilar types (%r vs %r)"
                    % (self.value.dtype, other.value.dtype)
                    )
            self.value = np.concatenate((self.value, other.value))
        elif isinstance(other, np.ndarray):
            if other.shape[1:] != self.value.shape[1:]:
                raise ValueError(
                    "Cannot add two ColumnAccumulator objects of dissimilar shape (%r vs %r)"
                    % (self.value.shape[1:], other.shape[1:])
                )
            if other.dtype != self.value.dtype:
                raise ValueError(
                    "Cannot add two ColumnAccumulator objects of dissimilar types (%r vs %r)"
                    % (self.value.dtype, other.dtype)
                    )
            self.value = np.concatenate((self.value, other))
