import coffea.processor as processor
from accumulators import HistAccumulator
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
import awkward
from fnmatch import fnmatch
import sys

parser = ArgumentParser()
parser.add_argument('jobid', default='2018_preskim', help='jobid to run on')
parser.add_argument('data_pu', help='root file with data PU distribution')
parser.add_argument('data_pu_up', help='root file with data PU distribution, up variation')
parser.add_argument('data_pu_dw', help='root file with data PU distribution, down variation')
parser.add_argument('-d', '--debug', action='store_true', help='run in local mode')
parser.add_argument('-t', '--test', help='test on a single file')
parser.add_argument('-T', '--tag', default='', help='add custon tag in output name')
parser.add_argument('-l', '--limit', default='*', help='limit to specific datasets')
parser.add_argument('-j', '--jobs', type=int, default=1, help='How many processes to run (available only with -d)')
args = parser.parse_args()

basedir = f'/home/users/j/p/jpriscia/coffea_hnl/inputs/{args.jobid}/' # HARDCODED FIXME!
if not os.path.isdir(basedir):
    raise RuntimeError('The jobid directory does not exist!')

datasets = [
    i for i in glob(f'{basedir}/*.txt') 
]
limits = args.limit.split(',')

fileset = {
    os.path.basename(i).replace('.txt', '') : [line.strip() for line in open(i)]
    for i in datasets
    if 'SingleMuon' not in i
}

fileset = {
    i : j for i, j in fileset.items() 
    if any(fnmatch(i, k) for k in limits)
}

if args.test:
    key = 'TEST'
    vals = [args.test] if args.test.endswith('.root') else [line.strip() for line in open(args.test)]
    fileset = {key : vals}
print('processing samples: ', fileset.keys())

# Look at ProcessorABC documentation to see the expected methods and what they are supposed to do
class PUHists(processor.ProcessorABC):
    def __init__(self, samples, binning):
        self.binning = binning
        ## Scale Factors
        self._accumulator = processor.dict_accumulator({
            s : HistAccumulator(binning.shape[0] - 1)
            for s in samples
        })
    
    @property
    def accumulator(self):
        return self._accumulator
    
    def process(self, lazy_df):
        # return self.process_try_except(lazy_df)
        try:
            return self.process_try_except(lazy_df)
        except Exception as e:
            raise type(e)( str(e) + f'\n Exception happens in {lazy_df["filename"]}').with_traceback(sys.exc_info()[2]) 

    def process_try_except(self, lazy_df):
        accumulator = self.accumulator.identity()
        vals, binning = lazy_df['obj'].numpy()
        assert((binning == self.binning).all())
        accumulator[lazy_df['dataset']].value += vals
        return accumulator

    def postprocess(self, accumulator):
        return accumulator

print('Starting processing')
if args.debug:
    fileset = utils.ensure_xrootd(fileset)
    if args.jobs > 1:
        job_kwargs = {
            'executor' : processor.futures_executor,
            'executor_args' : {'workers': 8, 'flatten' : False},
            }
    else:
        job_kwargs = {
            'executor' : processor.iterative_executor,
            'executor_args' :  {'workers': 1, 'flatten' : False},
        }
else:
    import batch
    dfk = batch.configure(nodes = 8, nprocs = 8)
    job_kwargs = {
        'executor' : processor.parsl_executor,
        'executor_args' : {'flatten' : False, 'xrootdtimeout' : 30},
        }
    fileset = utils.ensure_local(fileset)

binning = np.arange(101).astype(float) # yes, hardcoded binning
output = processor.run_uproot_job(
    fileset,
    treename = 'metaTree/PUDistribution',
    processor_instance = PUHists(fileset.keys(), binning),
    chunksize = 500000,
    **job_kwargs
)
print('DONE!')
print(output)

outf = uproot.recreate(f'inputs/{args.jobid}/pu_mc.root')
for key, h in output.items():
    vals = h.value
    vals /= vals.sum()
    outf[key] = (vals, binning)

def norm(h):
  scale = h.allvalues.sum()
  for i in range(len(h)):
    h[i] /= scale
  return h

f1 = uproot.open(args.data_pu)
f2 = uproot.open(args.data_pu_up)
f3 = uproot.open(args.data_pu_dw)
outf['data']     = norm(f1['pileup'])
outf['data_up']  = norm(f2['pileup'])
outf['data_down']= norm(f3['pileup'])

outf.close()
