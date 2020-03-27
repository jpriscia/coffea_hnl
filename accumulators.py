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

