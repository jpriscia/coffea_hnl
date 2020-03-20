from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor
import uproot
from pdb import set_trace
import os
from argparse import ArgumentParser
import uproot_methods
import numpy as np
import utils
import warnings
from glob import glob
import os
import utils

parser = ArgumentParser()
parser.add_argument('jobid', default='2018_preskim', help='jobid to run on')
parser.add_argument('-f', '--force', action='store_true', help='force re-production of the infos')
args = parser.parse_args()

basedir = f'/home/users/j/p/jpriscia/coffea_hnl/inputs/{args.jobid}/' # HARDCODED FIXME!
if not os.path.isdir(basedir):
    raise RuntimeError('The jobid directory does not exist!')

datasets = [
    i for i in glob(f'{basedir}/*.txt') 
    if not os.path.basename(i).startswith('SingleMuon') # check is not data
    if not os.path.isfile(i.replace('.txt', '.meta.json')) # check is not produced already
]

fileset = {
    os.path.basename(i).replace('.txt', '') : [line.strip() for line in open(i)]
    for i in datasets
}

print('Retrieving meta info for:', fileset.keys())

# Look at ProcessorABC documentation to see the expected methods and what they are supposed to do
class CountNtuplized(processor.ProcessorABC):
    def __init__(self):
        ## make binning for hists        
        self._accumulator = processor.defaultdict_accumulator(int)
    
    @property
    def accumulator(self):
        return self._accumulator
    
    def process(self, df):
        # TODO: PU Reweight, Jet PU ID, check cuts
        accumulator = self.accumulator.identity()
        accumulator[df.dataset] += df.processed.sum()
        accumulator[f'{df.dataset}_weighted'] += df.processedWeighted.sum()
        return accumulator

    def postprocess(self, accumulator):
        return accumulator

print('Starting processing')
import batch
dfk = batch.configure(nodes = 20)
output = processor.run_uproot_job(
    utils.ensure_local(fileset),
    treename = 'metaTree/meta',
    processor_instance = CountNtuplized(),
    ## executor = processor.iterative_executor,
    ## executor_args={'workers': 1, 'flatten' : True},
    # executor=processor.futures_executor,
    # executor_args={'workers': 8, 'flatten' : True},
    executor = processor.parsl_executor,
    executor_args={'flatten' : True, 'xrootdtimeout' : 30},
    #chunksize=500000,
)
print('DONE!')
print(output)

import json
for key in fileset:
    with open(f'inputs/{args.jobid}/{key}.meta.json', 'w') as meta:
        jmap = {}
        jmap['processed'] = float(output[key])
        jmap['processed_weighted'] = float(output[f'{key}_weighted'])
        json.dump(jmap, meta)

from glob import glob
infiles = glob(f'inputs/{args.jobid}/*.meta.json')
with open(f'inputs/{args.jobid}/mc.meta.json', 'w') as meta:
    jmap = {}
    for infile in infiles:        
        key = os.path.basename(infile).split('.')[0]
        if key == 'mc': continue
        jfile = json.load(open(infile))
        jmap[key] = jfile['processed']
    json.dump(jmap, meta)

