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
import objects
from maskedlazy import MaskedLazyDataFrame
from coffea import hist
from coffea import lookup_tools
import uproot_methods
import awkward
from fnmatch import fnmatch
import sys

parser = ArgumentParser()
parser.add_argument('jobid', default='2018_preskim', help='jobid to run on')
parser.add_argument('-d', '--debug', action='store_true', help='run in local mode')
parser.add_argument('-t', '--test', help='test on a single file')
parser.add_argument('-T', '--tag', help='test on a single file')
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

if args.test:
    key = 'SingleMuonTEST' if 'SingleMuon' in args.test else 'TEST'
    vals = [args.test] if args.test.endswith('.root') else [line.strip() for line in open(args.test)]
    fileset = {key : vals}
print('processing samples: ', fileset.keys())

# Look at ProcessorABC documentation to see the expected methods and what they are supposed to do
class SkimPlot(processor.ProcessorABC):
    def __init__(self, jobid):
        ## Scale Factors
        extractor = lookup_tools.extractor()
        extractor.add_weight_sets([f"pu_ * inputs/{jobid}/pu.root"]) 
        extractor.finalize() # after this you cannot load any more weights
        self.weights_lookup = extractor.make_evaluator()

        ## make binning for hists
        sample_axis = hist.Cat("sample", "sample name")
        mass_axis = hist.Bin("mass", r"m [GeV]", 40, 0, 200)
        pt_axis = hist.Bin("pt", r"$p_{T}$ [GeV]", 40, 0, 200)
        dr_axis = hist.Bin("dr", r"$\Delta R$", 40, 0, 8)

        self._accumulator = processor.dict_accumulator({
            'cutflow' : processor.defaultdict_accumulator(float),
            # 'preselection_prompt_pt'   : hist.Hist('prompt_pt'  , sample_axis, pt_axis),
            # 'preselection_diplaced_pt' : hist.Hist('diplaced_pt', sample_axis, pt_axis),
            # 'preselection_di_mu_M'     : hist.Hist('di_mu_M'    , sample_axis, mass_axis),
            # 'preselection_di_mu_DR'    : hist.Hist('di_mu_DR'   , sample_axis, dr_axis),
            'preselection' : processor.dict_accumulator({
                'prompt_pt'   : hist.Hist('prompt_pt'  , sample_axis, pt_axis),
                'diplaced_pt' : hist.Hist('diplaced_pt', sample_axis, pt_axis),
                'di_mu_M'     : hist.Hist('di_mu_M'    , sample_axis, mass_axis),
                'di_mu_DR'    : hist.Hist('di_mu_DR'   , sample_axis, dr_axis),
            }),
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
        df = MaskedLazyDataFrame.from_lazy(lazy_df)
        
        sample_name = df['dataset']
        accumulator['cutflow'][f'{sample_name}_start'] += df.shape[0]

        # compute PU weights
        if sample_name.startswith('Single'):
            df['weight'] = np.ones(df.shape[0])
        else:
            central = self.weights_lookup['pu_data'](df['pu_trueInteraction']) / self.weights_lookup['pu_mc'](df['pu_trueInteraction'])
            up = self.weights_lookup['pu_data_up'](df['pu_trueInteraction']) / self.weights_lookup['pu_mc'](df['pu_trueInteraction'])
            down = self.weights_lookup['pu_data_down'](df['pu_trueInteraction']) / self.weights_lookup['pu_mc'](df['pu_trueInteraction'])
            df['weight'] = central
            df['pileup_up'] = up / central
            df['pileup_down'] = down / central

        # Make objects (electrons, muons, jets, SVs...) here
        df['muons'] = objects.make_muons(df)
        df['svs'] = objects.make_svs(df)
        
        # Get prompt muon and select events
        prompt_mu_mask = (df.muons.p4.pt > 25) & (np.abs(df.muons.p4.eta) < 2.5) & (df.muons.dbIso < 0.15) & df.muons.isTight
        all_prompt_mus = df['muons'][prompt_mu_mask]
        df['prompt_mu'] = all_prompt_mus[:,:1] # pick only the first one
        trig_and_prompt = df['passIsoMu24'] & (all_prompt_mus.counts > 0) 
        accumulator['cutflow'][f'{sample_name}_trigg&promptMu'] += trig_and_prompt.sum()
        
        # Get trailing muon, here we need some gym due to broadcasting issues
        displ_mu_mask = (df['muons'].p4.pt > 5) & (np.abs(df['muons'].p4.eta) < 2.5) & df['muons'].isLoose
        # check that we are not selecting the same muon
        lft, rht = df['muons'].p4.cross(df['prompt_mu'].p4, nested = True).unzip() # make combinations
        min_dr = lft.delta_r(rht).min() # min DR
        displ_mu_mask = trig_and_prompt & displ_mu_mask & (min_dr > 0.0000001)
        all_displ_mu = df['muons'][displ_mu_mask]

        # select SVs
        good_svs = df['svs'][df['svs'].pt > 0] \
                   if (df['svs'].counts > 0).any() else \
                      df['svs']

        # make skimming and attach trailing mu and sv
        presel = trig_and_prompt & (all_displ_mu.counts > 0) & (good_svs.counts > 0) & (df.jet_pt.counts > 0)
        df['second_mu'] = all_displ_mu[:,:1]
        df['goodsv'] = good_svs[:,:1]
        skim = df[presel]
        accumulator['cutflow'][f'{sample_name}_skimming'] += presel.sum()
        
        # flatten objects to avoid messing around
        skim['prompt_mu'] = skim['prompt_mu'].flatten()
        skim['second_mu'] = skim['second_mu'].flatten()
        skim['goodsv'] = skim['goodsv'].flatten()

        # make preselection variables and cuts
        skim['m1_vtx_mass'] = (skim.prompt_mu.p4 + skim.goodsv.p4).mass
        skim['ll_mass'] = (skim.prompt_mu.p4 + skim.second_mu.p4).mass
        skim['ll_dr'] = skim.prompt_mu.p4.delta_r(skim.second_mu.p4)
        
        # make preslection cut
        selection = (skim.prompt_mu.absdxy < 0.005) & (skim.prompt_mu.absdz < 0.1) & \
                    (skim.second_mu.absdxy > 0.02) & \
                    (40 < skim.m1_vtx_mass) & (skim.m1_vtx_mass < 90) & \
                    (20 < skim.ll_mass) & (skim.ll_mass < 85) & \
                    (1 < skim.ll_dr) & (skim.ll_dr < 5) & \
                    (skim.jet_pt.max() > 20)
                    

        preselection = skim[selection]
        accumulator['cutflow'][f'{sample_name}_preselection'] += preselection.shape[0]
        print(preselection.shape[0], preselection['weight'].sum())
        # fill preselection histograms
        accumulator['preselection']['prompt_pt'  ].fill(
            weight = preselection['weight'], sample = sample_name, 
            pt = utils.tonp(preselection.prompt_mu.p4.pt)
        )
        accumulator['preselection']['diplaced_pt'].fill(
            weight = preselection['weight'], sample = sample_name, 
            pt = utils.tonp(preselection.second_mu.p4.pt)
        )
        accumulator['preselection']['di_mu_M'    ].fill(
            weight = preselection['weight'], sample = sample_name, 
            mass = utils.tonp(preselection['ll_mass'])
        )
        accumulator['preselection']['di_mu_DR'   ].fill(
            weight = preselection['weight'], sample = sample_name, 
            dr = utils.tonp(preselection.ll_dr)
        )
        
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

output = processor.run_uproot_job(
    fileset,
    treename = 'HeavyNeutralLepton/tree_',
    processor_instance = SkimPlot(args.jobid),
    # executor = processor.iterative_executor,
    # executor_args={'workers': 1, 'flatten' : False},
    ## executor=processor.futures_executor,
    ## executor_args={'workers': 8, 'flatten' : True},
    # executor = processor.parsl_executor,
    # executor_args={'flatten' : True, 'xrootdtimeout' : 30},
    chunksize = 500000,
    **job_kwargs
)
print('DONE!')
print(output)

from coffea.util import save
if not os.path.isdir(f'results/{args.jobid}'):
    os.makedirs(f'results/{args.jobid}')
save(output, f'results/{args.jobid}/skimplot{args.tag}.coffea')
