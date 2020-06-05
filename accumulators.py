from coffea.processor import AccumulatorABC
import numpy as np
import awkward as awk
from pdb import set_trace

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
    def __init__(self, dtype, jagged = False):
        self.dtype = dtype
        self.jagged = jagged
        if jagged:
            self.value = awk.JaggedArray.fromcounts([], np.array([], dtype=dtype))
        else:
            self.value = np.zeros(dtype=dtype, shape=(0,))

    def __repr__(self):
        return "ColumnAccumulator(%r)" % (self.value)

    @property
    def shape(self):
        return self.value.shape

    def identity(self):
        return ColumnAccumulator(self.dtype, self.jagged)

    def add(self, other):
        if other.shape[0] == 0: # patch for empty stuff
            return
        if isinstance(other, np.ndarray) and isinstance(self.value, awk.JaggedArray):
            raise ValueError('Cannot append a np.array to a JaggedArray')
        elif isinstance(other, ColumnAccumulator):
            if other.value.shape[1:] != self.value.shape[1:]:
                raise ValueError(
                    "Cannot add two ColumnAccumulator objects of dissimilar shape (%r vs %r)"
                    % (self.value.shape[1:], other.value.shape[1:])
                )
            if other.value.dtype != self.value.dtype:
                raise ValueError(
                    "A: Cannot add two ColumnAccumulator objects of dissimilar types (%r vs %r)"
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
                    "B: Cannot add two ColumnAccumulator objects of dissimilar types (%r vs %r)"
                    % (self.value.dtype, other.dtype)
                    )
            self.value = np.concatenate((self.value, other))
        elif isinstance(other,awk.JaggedArray):
            if isinstance(self.value, awk.JaggedArray):
                if other.content.dtype != self.value.content.dtype:
                    raise ValueError(
                        "C: Cannot add two ColumnAccumulator objects of dissimilar types (%r vs %r)"
                        % (self.value.content.dtype, other.content.dtype)
                    )
                self.value = awk.concatenate((self.value, other))
            else:
                raise ValueError(
                    "I need two jagged arrays"
                )
        else:
            raise ValueError(
                "add not implemented for those types"
            )
