import coffea.processor as processor
import functools
import uproot
from pdb import set_trace
import os
from argparse import ArgumentParser
import collections
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
parser.add_argument('jobid', help='jobid to run on')
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
}

fileset = {
    i : j for i, j in fileset.items() 
    if any(fnmatch(i, k) for k in limits)
}
#set_trace()
if args.test:
    key = 'TEST'
    vals = [args.test] if args.test.endswith('.root') else [line.strip() for line in open(args.test)]
    fileset = {key : vals}
print('processing samples: ', fileset.keys())

print('Starting processing')
executor = None
executor_args = None
if args.debug:
    fileset = utils.ensure_xrootd(fileset)
    if args.jobs > 1:
        executor = processor.futures_executor
        executor_args = {'workers': 8, 'flatten' : False}
    else:
        executor = processor.iterative_executor
        executor_args = {'workers': 1, 'flatten' : False}
else:
    import batch
    dfk = batch.configure(nodes = 8, nprocs = 8)
    executor = processor.parsl_executor
    executor_args = {'flatten' : False, 'xrootdtimeout' : 30}
    fileset = utils.ensure_local(fileset)

def get_badfiles(item):
    import coffea.processor as processor
    out = processor.set_accumulator()
    # add timeout option according to modified uproot numentries defaults
    xrootdsource = {"timeout": None, "chunkbytes": 32 * 1024, "limitbytes": 1024**2, "parallel": False}
    try:
        # Test ability to open file and read tree
        file = uproot.open(item.filename, xrootdsource=xrootdsource)
        tree = file[item.treename]
        entries = tree.numentries
    except Exception as e:
        out.add((item.dataset, item.filename))
    return out

metadata_cache = processor.executor.DEFAULT_METADATA_CACHE

treename = 'HeavyNeutralLepton/tree_'
fileset = list(processor.executor._normalize_fileset(fileset, treename))
for filemeta in fileset:
    filemeta.maybe_populate(metadata_cache)

badfiles_fetcher = functools.partial(get_badfiles)
out = processor.set_accumulator()

executor_args.update({
    'desc': 'Validating',
    'unit': 'file',
    'compression': None,
    'tailtimeout': None,
    'worker_affinity': False,
})
to_get = set(
    filemeta for filemeta in fileset 
    if not filemeta.populated(clusters=False)
)
executor(to_get, badfiles_fetcher, out, **executor_args)
bad_files = collections.defaultdict(set)
while out:
    dataset, fname = out.pop()
    bad_files[dataset].add(fname)

print('DONE!')
set_trace()

for dataset, files in bad_files.items():
    ans = input(f'I found {len(files)} bad files in the dataset {dataset}, update the inputs? [Yy/Nn]')
    if ans.lower() == 'n':
        continue

    txt = f'{basedir}/{dataset}.txt'
    in_txt = {i.strip() for i in open(txt)}

    with open(f'{basedir}/{dataset}.txt.bak', 'w') as bak:
        bak.write('\n'.join(in_txt))

    good = in_txt - files
    with open(txt, 'w') as out:
        out.write()        


