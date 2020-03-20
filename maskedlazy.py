import numpy as np
from coffea.processor.dataframe import LazyDataFrame
from pdb import set_trace
from copy import copy
import awkward

class MaskedLazyDataFrame(LazyDataFrame):
    def __init__(self):
        self.mask_ = np.array([])
        
    @classmethod
    def from_lazy(self, df):
        ret = MaskedLazyDataFrame()
        ret.__dict__.update(
            copy(df.__dict__)
        )
        ret.mask_ = np.ones(df.size).astype(bool)
        return ret

    @property
    def shape(self):
        return (self.mask_.sum(), len(self.columns))

    def __getitem__(self, k):
        if isinstance(k, (np.ndarray, awkward.IndexedMaskedArray)):
            ret = MaskedLazyDataFrame.from_lazy(self)
            ret.mask_ = self.mask_.copy()
            ret.mask_[ret.mask_] = k            
            return ret
        else:
            #set_trace()
            retval = super().__getitem__(k)
            if isinstance(retval, str):
                # handle special case
                return retval
            return retval[self.mask_]
            ## except Exception as e:
            ##     import sys
            ##     raise type(e)( str(e) + f'\n  Key: {k} \n  ret.shape: {retval.shape} \n df.size: {self.size} \n  mask.shape: {self.mask_.shape}').with_traceback(sys.exc_info()[2])

    def __setitem__(self, k, val):
        if self.mask_.all(): # the mask is not active
            super().__setitem__(k, val)
        else:
            # check that the length is correct:
            if val.shape[0] != self.mask_.sum():
                raise ValueError(
                    f'the size of the value you are trying to attach to the dataframe'
                    f' ({val.shape[0]}) is different from the size of the dataframe ({self.mask_.sum()}).'
                )
            # if it is masked we need to make a masked array
            indexing = np.full(self.mask_.shape[0], -1)
            indexing[self.mask_] = np.arange(np.count_nonzero(self.mask_))
            to_put_in_table = awkward.IndexedMaskedArray(indexing, val)
            super().__setitem__(k, to_put_in_table)

    @property
    def size(self):
        return self.mask_.sum()
