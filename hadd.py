#! /bin/env python
from coffea.util import save, load
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('out', help='output file name')
parser.add_argument('infiles', nargs='+', help='input files')
args = parser.parse_args()

reduced = None
for infile in args.infiles:
    accum = load(infile)
    if reduced is None:
        reduced = accum.identity()
    reduced += accum

save(reduced, args.out)
