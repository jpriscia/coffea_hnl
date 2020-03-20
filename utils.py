import numpy as np
import math
import awkward
from pdb import set_trace

def tonp(vv):
    '''helper function to convert an IndexedMaskedArray 
    into a np array, potentially dangerous as it flattens everything'''
    if isinstance(vv, awkward.IndexedMaskedArray):
        return tonp(vv._content[vv._mask]) # there will be multiple nesting
    else:
        return vv

def delta_r(eta1, phi1, eta2, phi2):
    'computed the DR'
    detas = np.abs(eta1 - eta2)
    dphis = (phi1 - phi2 + math.pi) % (2 * math.pi) - math.pi
    return np.sqrt(detas**2 + dphis**2)

def ensure_local(files):
    'convert xrootd address to local ones'
    if isinstance(files, dict):
        ret = {}
        for key in files:
            ret[key] = ensure_local(files[key])
        return ret
    elif isinstance(files, list):
        return list(map(ensure_local, files))
    elif isinstance(files, str):
        rval = files.replace('root://ingrid-se03.cism.ucl.ac.be/', '/storage/data/cms/')
        if not rval.startswith('/storage/data/cms/'):
            raise RuntimeError(f'The file {rval} is not a local path!')
        return rval
    else:
        raise NotImplementedError()

def ensure_xrootd(files):
    'convert xrootd address to local ones'
    if isinstance(files, dict):
        ret = {}
        for key in files:
            ret[key] = ensure_xrootd(files[key])
        return ret
    elif isinstance(files, list):
        return list(map(ensure_xrootd, files))
    elif isinstance(files, str):
        rval = files.replace('/storage/data/cms/', 'root://ingrid-se03.cism.ucl.ac.be/')
        if not rval.startswith('root://ingrid-se03'):
            raise RuntimeError(f'The file {rval} is not a xrootd path!')
        return rval
    else:
        raise NotImplementedError()

import inspect
def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno

import yaml
import json
def compute_weights(mc_meta, target_lumi, xsec):
    doc  = yaml.load(open(xsec))
    meta = json.load(open(mc_meta))
    ret = {}
    for key in meta:
        ret[key] = target_lumi * doc['samples'][key]['xsec'] / float(meta[key])
    return ret
