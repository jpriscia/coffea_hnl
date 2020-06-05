import numba
import awkward as awk
import numpy as np
from pdb import set_trace

def zero_pad(jarr, msize):
    '''converts a jagged array into a 2D np.array, filling blanks with zeros
    and trimming arrays that are too long'''
    @numba.njit
    def _zero_pad_jagged(flat_arr, counts, msize):
        ret = np.zeros((counts.shape[0], msize))
        strt = 0
        for idx, size in enumerate(counts):
            trim = min(size, msize)
            left = np.zeros((msize - trim))            
            ret[idx, :] = np.concatenate((flat_arr[strt : strt + trim], left))
            strt += size
        return ret
    
    @numba.jit
    def _zero_pad_ndarr(obj_arr, msize):
        ret = np.zeros((obj_arr.shape[0], msize))
        for idx, sub in enumerate(obj_arr):
            size = sub.shape[0]
            trim = min(size, msize)
            left = np.zeros((msize - trim))            
            ret[idx, :] = np.concatenate((obj_arr[:trim], left))
        return ret
    
    if isinstance(jarr, awk.JaggedArray):
        return _zero_pad_jagged(jarr.flatten(), jarr.counts, msize)
    elif isinstance(jarr, np.ndarray):
        return _zero_pad_ndarr(jarr, msize)
    else:
        raise ValueError('zero pad is not defined for %r' % type(jarr))


from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('jobid', default='2018_preskim', help='jobid to run on')
args = parser.parse_args()

from glob import glob
from coffea.util import load

data, labels, masks = [], [], []
infiles = glob(f'results/{args.jobid}/skimplotnew_*.coffea')
for ifile in infiles[:2]:
    idata = load(ifile)
    for sample, cols in idata['columns'].items():
        print('processing ', sample, '...')
        if sample.startswith('SingleMuon'): continue
        padded = []
        first = True
        for icol in ['sv_tracks_charge', 'sv_tracks_eta',
                     'sv_tracks_phi', 'sv_tracks_pt', 'sv_tracks_p', 'sv_tracks_dxySig',
                     'sv_tracks_dxy', 'sv_tracks_dxyz',]: # TODO CHECK THE ORDER!
            jarr = awk.JaggedArray.fromiter(cols[icol].value)            
            padded.append(zero_pad(jarr, 20)) # TODO CHECK PAD SIZE
            if first:
                first = False
                masks.append(
                    zero_pad(jarr.ones_like(), 20)
                )
        
        data.append(
            np.stack(padded, axis = -1)
        )
        labels.append(
            np.ones(data[-1].shape[0])
            if sample.startswith('M-') else
            np.zeros(data[-1].shape[0])
        )

data = np.concatenate(data)
labels = np.concatenate(labels)
masks = np.concatenate(masks)
# shuffle data
rnd = np.random.permutation(labels.shape[0])
data = data[rnd]
labels = labels[rnd]
masks = masks[rnd]
masks = masks.reshape((-1, masks.shape[1], 1))
signal_weight = labels.sum()/labels.shape[0]

def get_weight(label, weight):
    ret = np.ones(label.shape)
    ret[label == 1] *= weight
    return ret

# train the network
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Dropout
from tensorflow.keras import optimizers

from sklearn.metrics import confusion_matrix 
import itertools 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test, m_train, m_test = train_test_split(data, labels, masks, test_size=0.15, random_state=0)

nn = Input(shape = data.shape[1:])
msk = Input(shape = masks.shape[1:])
conv = Conv1D(32,  1, activation='elu', padding='same', data_format='channels_last', name='conv1')(nn)
conv = Conv1D(32,  1, activation='elu', padding='same', data_format='channels_last', name='conv2')(conv)
conv = Conv1D(32,  1, activation='elu', padding='same', data_format='channels_last', name='conv3')(conv)
conv = Conv1D(32,  1, activation='elu', padding='same', data_format='channels_last', name='conv4')(conv)
masked_conv = conv * msk
summed = keras.backend.sum(masked_conv, axis = 1)
dense = Dense(64, activation='elu', name = 'dense1')(summed)
out = Dense(1, activation = 'sigmoid', name = 'out')(dense)

model = Model(inputs=(nn, msk), outputs=out)
model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=10, verbose=1, mode='auto')

history = model.fit(
    (x_train, m_train),
    y_train,
    batch_size = 100,
    epochs = 5,
    callbacks = [early_stop],
    validation_split = 0.1,
    shuffle = True,
    sample_weight = get_weight(y_train, signal_weight),
    )
