#! /bin/env python
from coffea.util import save, load
from argparse import ArgumentParser
from pdb import set_trace

parser = ArgumentParser()
parser.add_argument('out', help='output file name')
parser.add_argument('infiles', nargs='+', help='input files')
parser.add_argument('-p', '--prune', help='keys to remove', type=str)

args = parser.parse_args()


reduced = None
for infile in args.infiles:
    accum = load(infile)
    if args.prune:
        for item in args.prune.split(','): del accum[item] 

    if reduced is None:
        reduced = accum.identity()
    reduced += accum

save(reduced, args.out)
