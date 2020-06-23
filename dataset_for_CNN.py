import numba
import awkward as awk
import numpy as np
from pdb import set_trace
import utils
from sklearn import datasets, metrics, model_selection, svm
from scikit_BDT_utils import *
from scikit_Function import *

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

#load weights
years = {
    '2017' : {
        'lumi' : 40e3,
        'xsecs' : '/home/ucl/cp3/jpriscia/CMSSW_10_2_15_patch2/src/HNL/HeavyNeutralLeptonAnalysis/test/input_mc_2017.yml',
    },
    '2018' : {
        'lumi' : 59.7e3,
        'xsecx' : '/home/ucl/cp3/jpriscia/CMSSW_10_2_15_patch2/src/HNL/HeavyNeutralLeptonAnalysis/test/input_mcAll_2018.yml'
    }
}

if '2017' in args.jobid:
    year = years['2017']
elif '2018' in args.jobid:
    year = years['2018']

scaling = utils.compute_weights(
    f'inputs/{args.jobid}/mc.meta.json', year['lumi'],
    year['xsecx']
)

xs_weights = []

data, labels, masks = [], [], []
infiles = glob(f'results/{args.jobid}/skimplotnew_*.coffea')
for ifile in infiles:
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
        xs_weights.append(
            np.ones(data[-1].shape[0]) if sample.startswith('M-') else
             np.ones(data[-1].shape[0])*scaling[sample]

        )

#bkgSumW = sum([x for x in xs_weights if x != 1.0])
#nSigSamples = len([x for x in xs_weights if x == 1.0])
#xs_weights = [bkgSumW/nSigSamples if x==1 else x for x in xs_weights]

data = np.concatenate(data)
labels = np.concatenate(labels)
masks = np.concatenate(masks)
xs_weights = np.concatenate(xs_weights)

#normalise bkg and signal weights. Put the same weight to all signals
mskBkg = (xs_weights!=1.0)
bkgSumW = xs_weights[mskBkg].sum()
nSigSamples = len(mskBkg) - np.count_nonzero(mskBkg)
sig_weight = bkgSumW/nSigSamples
xs_weights[np.invert(mskBkg)] = sig_weight

# shuffle data
rnd = np.random.permutation(labels.shape[0])
data = data[rnd]
labels = labels[rnd]
masks = masks[rnd]
xs_weights = xs_weights[rnd]

masks = masks.reshape((-1, masks.shape[1], 1))

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

####test mohammed CNN###
#channels = 1
#img_size = 160
#data = data.reshape(-1,channels,img_size)
#######
x_train, x_test, y_train, y_test, m_train, m_test, w_train, w_test = train_test_split(data, labels, masks, xs_weights,  test_size=0.15, random_state=0)

set_trace()
#nn = Input(shape = data.shape[1:])
#msk = Input(shape = masks.shape[1:])
#conv = Conv1D(32,  1, activation='elu', padding='same', data_format='channels_last', name='conv1')(nn)
#conv = Conv1D(32,  1, activation='elu', padding='same', data_format='channels_last', name='conv2')(conv)
#conv = Conv1D(32,  1, activation='elu', padding='same', data_format='channels_last', name='conv3')(conv)
#conv = Conv1D(32,  1, activation='elu', padding='same', data_format='channels_last', name='conv4')(conv)
#masked_conv = conv * msk
#summed = keras.backend.sum(masked_conv, axis = 1)
#dense = Dense(64, activation='elu', name = 'dense1')(summed)
#out = Dense(1, activation = 'sigmoid', name = 'out')(dense)

#model = Model(inputs=(nn, msk), outputs=out)
#optimizer = optimizers.Adamax(lr=0.002)
#model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


######test mohammmed CNN######
nn = Input(shape=(channels,img_size), name='input')
conv = Conv1D(32,  3, activation='elu', padding='same', data_format='channels_last', name='conv1')(nn)
conv = Conv1D(64,  3, activation='elu', padding='same', data_format='channels_last', name='conv3')(conv)
conv = Conv1D(128, 3, activation='elu', padding='same', data_format='channels_last', name='conv5')(conv)
conv = Conv1D(256, 3, activation='elu', padding='same', data_format='channels_last', name='conv7')(conv)
dense = Dropout(0.25)(conv)
conv = Conv1D(512, 3, activation='elu', padding='same', data_format='channels_last', name='conv8')(dense)
flat = Flatten()(conv)

dense = Dense(256, activation='elu', name='dense2')(flat)
dense = Dropout(0.5)(dense)
pred = Dense(2, activation='softmax', name='output')(dense)

model = Model(inputs=nn, outputs=pred)
model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
################

early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10, verbose=1, mode='auto')


#history = model.fit(
#    (x_train, m_train),
#    y_train,
#    batch_size = 100,
#    epochs = 30,
#    callbacks = [early_stop],
#    validation_split = 0.1,
#    shuffle = True,
#    sample_weight = w_train,
#    )

######fit mohamed########
history = model.fit(
    (x_train),#, m_train),
    keras.utils.to_categorical(y_train),
    batch_size = 100,
    epochs = 10,
    callbacks = [early_stop],
    validation_split = 0.1,
    shuffle = True,
    sample_weight = w_train,
    )

scores = model.evaluate(x_test, keras.utils.to_categorical(y_test), verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
y_pred = model.predict(x_test)
true = np.argmax(keras.utils.to_categorical(y_test), axis=1)
pred = np.argmax(y_pred, axis=1)
print('True labels: ', true)
print('Predicted labels: ', pred)

cnf_matrix = confusion_matrix(true, pred)
cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
print (cnf_matrix)

##################

set_trace()

for name in ['loss', 'accuracy']:
    epochs = range(len(history.history[name]))
    plt.figure()
    plt.plot(epochs, history.history[name], label='Training')
    plt.plot(epochs, history.history[f'val_{name}'], label='Validation')
    plt.title(f'Training and validation {name}')
    plt.legend()
    plt.savefig(f'plots/{args.jobid}/cnn_{name}.png')

pred_test = model.predict((x_test, m_test))
pred_train = model.predict((x_train, m_train))

#split sig and bkg
pred_train_Sig = model.predict((x_train[y_train > 0.5], m_train[y_train > 0.5]))
pred_train_Bkg = model.predict((x_train[y_train < 0.5], m_train[y_train < 0.5]))
pred_test_Sig  = model.predict((x_test[y_test > 0.5], m_test[y_test > 0.5]))
pred_test_Bkg  = model.predict((x_test[y_test < 0.5], m_test[y_test < 0.5]))

#ROC curve
fpr, tpr, threshold = metrics.roc_curve(y_test, pred_test, sample_weight=w_test)
roc_auc = metrics.auc(fpr, tpr)
plt.figure()
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.grid()
plt.savefig(f'plots/{args.jobid}/ROC.png')

idx_99 = np.argmin(abs(fpr-0.01))
print(fpr[idx_99], tpr[idx_99])
#compare Decision Functions
plt.figure()
compare_train_test_dec(pred_train_Sig.flatten(), pred_train_Bkg.flatten(), pred_test_Sig.flatten(), pred_test_Bkg.flatten())
plt.savefig(f'plots/{args.jobid}/OT.png')
set_trace()

