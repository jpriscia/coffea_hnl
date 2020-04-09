from argparse import ArgumentParser
from glob import glob
import os
import sys
import re
from pdb import set_trace

parser = ArgumentParser()
parser.add_argument('jobid', help='jobid to use')
parser.add_argument('indirs', nargs='+', help='input dirs to harvest from')
args = parser.parse_args()

outdir = f'inputs/{args.jobid}'
if not os.path.isdir(outdir):
    os.makedirs(outdir)

prescaled = re.compile('.*_[0-9]$')
for indir in args.indirs:
    datasets = glob(f'{indir}/*/*')
    
    # group prescales but not extensions
    clean_datasets = set()
    for dataset in datasets:
        if prescaled.match(dataset):
            clean_datasets.add(dataset[:-1]+'*')
        else:
            clean_datasets.add(dataset)
    
    for dataset in clean_datasets:
        files = glob(f'{dataset}/*/*/*.root')
        dname = os.path.basename(dataset).replace('crab_', '').replace('_*', '')
        txt = f'{outdir}/{dname}.txt'
        if os.path.isfile(txt):
            txt_files = [i for i in open(txt)]
            if len(txt_files) != len(files):
                ans = input(f'Dataset {dname} already exists and has a different number of files, update it? [Yy/Nn]')
                if ans.lower() == 'n': 
                    continue
            else:
                continue

        print(f'Saving {dname}...')
        with open(txt, 'w') as outf:
            outf.write('\n'.join(files))
