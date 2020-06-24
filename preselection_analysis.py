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
import awkward
import math
from fnmatch import fnmatch
import sys
from functools import reduce
from accumulators import ColumnAccumulator

parser = ArgumentParser()
parser.add_argument('jobid', default='2018_preskim', help='jobid to run on')

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


if args.test:
    key = 'SingleMuonTEST' if 'SingleMuon' in args.test else 'TEST'
    vals = [args.test] if args.test.endswith('.root') else [line.strip() for line in open(args.test)]
    fileset = {key : vals}
print('processing samples: ', fileset.keys())

# Look at ProcessorABC documentation to see the expected methods and what they are supposed to do
class SkimPlot(processor.ProcessorABC):
    def __init__(self, jobid, samples):
        # store jobid for steering cuts
        self.jobid = jobid
        
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
        dr_axis_small = hist.Bin("dr", r"$\Delta R$", 40, 0.5, 5)
        mass_axis_small = hist.Bin("mass", r"m [GeV]", 40, 15, 90)
        lxy_axis = hist.Bin("lxy", r"lxy [cm]", 50, 0, 100)
        sv_mass_axis = hist.Bin("mass", r"m [GeV]", 50, 0, 10)

        self._accumulator = processor.dict_accumulator({
            'cutflow' : processor.defaultdict_accumulator(float),
            'columns' : processor.dict_accumulator({
                i : processor.dict_accumulator({
                    'isOS' : ColumnAccumulator(bool),
                    'pu_weight' : ColumnAccumulator(np.float64),
                    'm1_vtx_mass' : ColumnAccumulator(np.float64),
                    'mu2_absdxy' : ColumnAccumulator(np.float32),
                    'mu2_absdz' : ColumnAccumulator(np.float32),
                    'mu2_phi'  : ColumnAccumulator(np.float64),
                    'mu2_ptBT'  : ColumnAccumulator(np.float64),
                    'mu2_etaBT' : ColumnAccumulator(np.float64),
                    'mu2_absdxySig' : ColumnAccumulator(np.float64),
                    'mu2_absdzSig'  : ColumnAccumulator(np.float64),
                    'mu2_deltaBeta' : ColumnAccumulator(np.float64),
                    'mu2_nDof' : ColumnAccumulator(np.float64),
                    'mu2_timeAtIpInOut' : ColumnAccumulator(np.float64),
                    'mu2_timeAtIpInOutErr' : ColumnAccumulator(np.float64),
                    'mu2_timeAtIpOutIn' : ColumnAccumulator(np.float64),
                    'mu2_timeAtIpOutInErr' : ColumnAccumulator(np.float64),
                    'mu2_segmentComp' : ColumnAccumulator(np.float64),
                    'mu2_trkKink' : ColumnAccumulator(np.float64),
                    'mu2_chi2LocalPosition' : ColumnAccumulator(np.float64),
                    'mu2_rhoRelIso' : ColumnAccumulator(np.float64),
                    'sv_mass' : ColumnAccumulator(np.float64),
                    'sv_pt' : ColumnAccumulator(np.float64),
                    'sv_lxySig' : ColumnAccumulator(np.float64),
                    'sv_lxyzSig' : ColumnAccumulator(np.float64),
                    'sv_lxy' : ColumnAccumulator(np.float64),
                    'sv_lxyz' : ColumnAccumulator(np.float64),
                    'sv_angle3D' : ColumnAccumulator(np.float64),
                    'sv_angle2D' : ColumnAccumulator(np.float64),
                    'sv_gamma' : ColumnAccumulator(np.float64),
                    'sv_chi2' : ColumnAccumulator(np.float64),
                    'sv_sum_tracks_dxySig' : ColumnAccumulator(np.float64),
                    'mujet_eta' : ColumnAccumulator(np.float64),
                    'mujet_phi' : ColumnAccumulator(np.float64),
                    'mujet_neutHadEnFrac' : ColumnAccumulator(np.float64),
                    'mujet_neutEmEnFrac' : ColumnAccumulator(np.float64),
                    'mujet_charHadEnFrac' : ColumnAccumulator(np.float64),
                    'mujet_charEmEnFrac' : ColumnAccumulator(np.float64),
                    'mujet_chargedMult' : ColumnAccumulator(np.float64),
                    'mujet_neutMult' : ColumnAccumulator(np.float64),
                    'mujet_smeared_pt' : ColumnAccumulator(np.float64),
                    'mujet_dCsv_bb' : ColumnAccumulator(np.float64),
                    'mujet_charEmEn' : ColumnAccumulator(np.float64),
                    'mujet_charHadEn' : ColumnAccumulator(np.float64),
                    'mujet_charMuEn' : ColumnAccumulator(np.float64),
                    'mujet_charMuEnFrac' : ColumnAccumulator(np.float64),
                    'mujet_muonEn' : ColumnAccumulator(np.float64),
                    'mujet_muonEnFrac' : ColumnAccumulator(np.float64),
                    'mujet_neutEmEn' : ColumnAccumulator(np.float64),
                    'mujet_neutHadEn' : ColumnAccumulator(np.float64),
                    'sv_tM' : ColumnAccumulator(np.float64),
                    'mu1_tM' : ColumnAccumulator(np.float64),
                    'mu2_tM' : ColumnAccumulator(np.float64),
                    'corr_M' : ColumnAccumulator(np.float64),
                    'dimu_deltaphi' : ColumnAccumulator(np.float64),
                    'dimu_mass' : ColumnAccumulator(np.float64),
                    'dimu_dr' : ColumnAccumulator(np.float64),
                    'nLooseMu' : ColumnAccumulator(np.int64),
                    'nDisplacedMu' : ColumnAccumulator(np.int64),
                    'sv_tracks_charge' : ColumnAccumulator(np.int32, True),
                    'sv_tracks_eta' : ColumnAccumulator(np.float32, True),
                    'sv_tracks_phi' : ColumnAccumulator(np.float32, True),
                    'sv_tracks_pt' : ColumnAccumulator(np.float32, True),
                    'sv_tracks_p' : ColumnAccumulator(np.float32, True),
                    'sv_tracks_dxySig' : ColumnAccumulator(np.float32, True),
                    'sv_tracks_dxy' : ColumnAccumulator(np.float32, True),
                    'sv_tracks_dxyz' : ColumnAccumulator(np.float32, True),
                })
                for i in samples
            }),

            # 'preselection_prompt_pt'   : hist.Hist('prompt_pt'  , sample_axis, pt_axis),
            # 'preselection_diplaced_pt' : hist.Hist('diplaced_pt', sample_axis, pt_axis),
            # 'preselection_di_mu_M'     : hist.Hist('di_mu_M'    , sample_axis, mass_axis),
            # 'preselection_di_mu_DR'    : hist.Hist('di_mu_DR'   , sample_axis, dr_axis),
            'preselection_SS' : processor.dict_accumulator({
                'prompt_pt'   : hist.Hist('prompt_pt'  , sample_axis, pt_axis),
                'diplaced_pt' : hist.Hist('diplaced_pt', sample_axis, pt_axis),
                'di_mu_M'     : hist.Hist('di_mu_M'    , sample_axis, mass_axis),
                'di_mu_DR'    : hist.Hist('di_mu_DR'   , sample_axis, dr_axis),
                'sv_tM'       : hist.Hist('sv_tM'   , sample_axis, mass_axis),
                'mu_tM'       : hist.Hist('mu_tM'   , sample_axis, mass_axis),
                'musv_tM'     : hist.Hist('musv_tM'   , sample_axis, mass_axis),
                'corr_M'      : hist.Hist('corr_M'   , sample_axis, mass_axis),
                'sv_lxy'      : hist.Hist('sv_lxy', sample_axis, lxy_axis),
                'sv_mass'     : hist.Hist('sv_mass', sample_axis, sv_mass_axis),
                'm1_vtx_mass' : hist.Hist('m1_vtx_mass', sample_axis, mass_axis),
            }),
            'preselection_OS' : processor.dict_accumulator({
                'prompt_pt'   : hist.Hist('prompt_pt'  , sample_axis, pt_axis),
                'diplaced_pt' : hist.Hist('diplaced_pt', sample_axis, pt_axis),
                'di_mu_M'     : hist.Hist('di_mu_M'    , sample_axis, mass_axis),
                'di_mu_DR'    : hist.Hist('di_mu_DR'   , sample_axis, dr_axis),
                'sv_tM'       : hist.Hist('sv_tM'   , sample_axis, mass_axis),
                'mu_tM'       : hist.Hist('mu_tM'   , sample_axis, mass_axis),
                'musv_tM'     : hist.Hist('musv_tM'   , sample_axis, mass_axis),
                'corr_M'      : hist.Hist('corr_M'   , sample_axis, mass_axis),
                'sv_lxy'      : hist.Hist('sv_lxy', sample_axis, lxy_axis),
                'sv_mass'     : hist.Hist('sv_mass', sample_axis, sv_mass_axis),
                'm1_vtx_mass' : hist.Hist('m1_vtx_mass', sample_axis, mass_axis),
            }),
            'selection_SS' : processor.dict_accumulator({
                'prompt_pt'   : hist.Hist('prompt_pt'  , sample_axis, pt_axis),
                'diplaced_pt' : hist.Hist('diplaced_pt', sample_axis, pt_axis),
                'di_mu_M'     : hist.Hist('di_mu_M'    , sample_axis, mass_axis_small),
                'di_mu_DR'    : hist.Hist('di_mu_DR'   , sample_axis, dr_axis_small),
                'sv_lxy'      : hist.Hist('sv_lxy', sample_axis, lxy_axis),
                'sv_mass'     : hist.Hist('sv_mass', sample_axis, sv_mass_axis),
                'm1_vtx_mass' : hist.Hist('m1_vtx_mass', sample_axis, mass_axis),
            }),
            'selection_OS' : processor.dict_accumulator({
                'prompt_pt'   : hist.Hist('prompt_pt'  , sample_axis, pt_axis),
                'diplaced_pt' : hist.Hist('diplaced_pt', sample_axis, pt_axis),
                'di_mu_M'     : hist.Hist('di_mu_M'    , sample_axis, mass_axis_small),
                'di_mu_DR'    : hist.Hist('di_mu_DR'   , sample_axis, dr_axis_small),
                'sv_lxy'      : hist.Hist('sv_lxy', sample_axis, lxy_axis),
                'sv_mass'     : hist.Hist('sv_mass', sample_axis, sv_mass_axis),
                'm1_vtx_mass' : hist.Hist('m1_vtx_mass', sample_axis, mass_axis),
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
            key = f'pu_{sample_name}'
            if key not in self.weights_lookup:
                key = 'pu_mc'
            if key not in self.weights_lookup:
                print('Probably this is just a test, assigning a random PU weight')
                key = list(self.weights_lookup.keys())[0]
            central = self.weights_lookup['pu_data'](df['pu_trueInteraction']) / self.weights_lookup[key](df['pu_trueInteraction'])
            up = self.weights_lookup['pu_data_up'](df['pu_trueInteraction']) / self.weights_lookup[key](df['pu_trueInteraction'])
            down = self.weights_lookup['pu_data_down'](df['pu_trueInteraction']) / self.weights_lookup[key](df['pu_trueInteraction'])
            df['weight'] = central
            df['pileup_up'] = up / central
            df['pileup_down'] = down / central

        # Make objects (electrons, muons, jets, SVs...) here
        df['muons'] = objects.make_muons(df)
        df['svs'] = objects.make_svs(df)
        df['jets'] = objects.make_jets(df)
        
        #count loose muons (- tight)
        df['nLooseMu'] =  df.muons.isLoose.sum() - 1
        # Get prompt muon and select events
        prompt_mu_mask = (df.muons.p4.pt > 25) & (np.abs(df.muons.p4.eta) < 2.5) & (df.muons.dbIso < 0.1) & df.muons.isTight
        all_prompt_mus = df['muons'][prompt_mu_mask]
        df['prompt_mu'] = all_prompt_mus[:,:1] # pick only the first one
        trig_to_use = 'passIsoMu27All' if '2017' in self.jobid else 'passIsoMu24'
        trig_and_prompt = df[trig_to_use] & (all_prompt_mus.counts > 0) 
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
        presel = trig_and_prompt & (all_displ_mu.counts > 0) & (good_svs.counts > 0) 
        df['n_displaced_mu'] = all_displ_mu.counts
        df['second_mu'] = all_displ_mu[:,:1]
        df['goodsv'] = good_svs[:,:1]
        skim = df[presel]
        accumulator['cutflow'][f'{sample_name}_skimming'] += presel.sum()

        # flatten objects to avoid messing around
        skim['prompt_mu'] = skim['prompt_mu'].flatten()
        skim['second_mu'] = skim['second_mu'].flatten()
        skim['goodsv'] = skim['goodsv'].flatten()

        # select and match jet to the second muon
        jets = skim['jets'] # just a shortcut to avoid writing it constantly
        # jet ID selection by bins, useful given that muons are only up to 2.4? 
        selection_eta_bins = [
            # |eta| < 2.4
            (np.abs(jets.p4.eta) <= 2.4) & \
            (jets.charHadEnFrac > 0.) & (jets.chargedMult > 0) & (jets.charEmEnFrac <= 0.99),
            # |eta| < 2.7
            (np.abs(jets.p4.eta) <= 2.7) & (jets.neutHadEnFrac <= 0.9) & \
            (jets.neutEmEnFrac <= 0.90) & ((jets.chargedMult + jets.neutMult) >= 1),
            # |eta| < 3.0
            (np.abs(jets.p4.eta) <= 2.7) & (jets.neutHadEnFrac <= 0.98) & \
            (jets.neutEmEnFrac <= 0.01) & (jets.neutMult >= 2),
            # any eta \_(-.-)_/
            (jets.neutEmEnFrac <= 0.9) & (jets.neutMult >= 10),
        ]
        # compute the or of the masks
        jet_id = reduce(lambda x, y: x | y, selection_eta_bins)
        
        # compute DR w.r.t. the prompt and second muon
        dr_promt  = utils.tonp( 
            skim['prompt_mu'].p4.delta_r(jets.p4) 
        )
        dr_second = utils.tonp( 
            skim['second_mu'].p4.delta_r(jets.p4) 
        )
        
        selection_dr = (dr_promt >= 0.4) & (dr_second <= 0.7) & \
                    (jets.p4.pt > 20) & jet_id
        
        selected_jets = jets[selection_dr]
        selected_dr_second = dr_second[selection_dr]
        ##skim['n_selected_jets'] = None  TO ADD when we understand what Mohamed did
        
        # pick the closest jet, keep jaggedness to avoid 
        # messing up skim. This BADLY needs awkward v2.0
        # to fix this mess
        min_dr = selected_dr_second.argmin()
        matched_jets = selected_jets[min_dr]
        at_least_one_jet = (selected_dr_second.count() > 0)

        # make preselection variables and cuts
        skim['m1_vtx_mass'] = (skim.prompt_mu.p4 + skim.goodsv.p4).mass
        skim['ll_mass'] = (skim.prompt_mu.p4 + skim.second_mu.p4).mass
        skim['ll_dr'] = skim.prompt_mu.p4.delta_r(skim.second_mu.p4)

        #transverse mass
        skim['tmass_goodsv'] = np.sqrt(pow(skim.goodsv.p4.pt  + skim.pfMet_pt,2) - \
                                       pow(skim.goodsv.p4.x + skim.pfMet_px,2) - \
                                       pow(skim.goodsv.p4.y + skim.pfMet_py,2))
        skim['tmass_promptmu'] = np.sqrt(pow(skim.prompt_mu.p4.pt  + skim.pfMet_pt,2) - \
                                         pow(skim.prompt_mu.p4.x + skim.pfMet_px,2) - \
                                         pow(skim.prompt_mu.p4.y + skim.pfMet_py,2))
        svPlusmu_p4 = skim.goodsv.p4 + skim.prompt_mu.p4
        skim['tmass_svmu'] = np.sqrt(pow(svPlusmu_p4.pt  + skim.pfMet_pt,2) - \
                                     pow(svPlusmu_p4.x + skim.pfMet_px,2) - \
                                     pow(svPlusmu_p4.y + skim.pfMet_py,2))

        skim['dimu_deltaphi'] = np.abs(skim.prompt_mu.p4.delta_phi(skim.second_mu.p4))
        
        goodsv_pt2 = (skim.goodsv.position.cross(skim.goodsv.p3).mag2)/(skim.goodsv.position.mag2)
        skim['mass_corr'] = np.sqrt(skim.goodsv.p4.mass * skim.goodsv.p4.mass + goodsv_pt2) + \
                            np.sqrt(goodsv_pt2)

        # make preslection cut
        preselection_mask = (skim.prompt_mu.absdxy < 0.005) & (skim.prompt_mu.absdz < 0.1) & \
                    (skim.second_mu.absdxy > 0.02) & \
                    (0.3 < skim.ll_dr) & at_least_one_jet
        #(40 < skim.m1_vtx_mass) & (skim.m1_vtx_mass < 90) & \ # removed from preselection_mask

        preselection = skim[preselection_mask]
        matched_jet = matched_jets[preselection_mask][:,0] # cannot attach to preselection for some reason FIXME!
        
        selection_mask = (1 < preselection.ll_dr) & (preselection.ll_dr < 5) & \
                         (preselection.jet_pt.counts > 0) & (preselection.jet_pt.max() > 20)

        #(20 < preselection.ll_mass) & (preselection.ll_mass < 85) & \ # removed from selection_mask

        same_sign = (preselection.prompt_mu.charge * preselection.second_mu.charge) >0.
        opp_sign = np.invert(same_sign)
        preselection['isOS'] = opp_sign
        
        accumulator['cutflow'][f'{sample_name}_preselection'] += preselection.shape[0]
        accumulator['cutflow'][f'{sample_name}_selection'] += selection_mask.sum()

        #print(preselection.shape[0], preselection['weight'].sum())
        # fill preselection histograms

        for category, mask in [
            ('preselection_SS', same_sign),
            ('preselection_OS', opp_sign),
            ('selection_SS', selection_mask & same_sign),
            ('selection_OS', selection_mask & opp_sign),]:

            masked_df = preselection[mask]
            masked_jets = matched_jet[mask]
            accumulator[category]['prompt_pt'  ].fill(
                weight = masked_df['weight'], sample = sample_name, 
                pt = utils.tonp(masked_df.prompt_mu.p4.pt)
            )
            accumulator[category]['diplaced_pt'].fill(
                weight = masked_df['weight'], sample = sample_name, 
                pt = utils.tonp(masked_df.second_mu.p4.pt)
            )
            accumulator[category]['di_mu_M'    ].fill(
                weight = masked_df['weight'], sample = sample_name, 
                mass = utils.tonp(masked_df['ll_mass'])
            )
            accumulator[category]['di_mu_DR'   ].fill(
                weight = masked_df['weight'], sample = sample_name, 
                dr = utils.tonp(masked_df.ll_dr)
            )
            accumulator[category]['sv_lxy'   ].fill(
                weight = masked_df['weight'], sample = sample_name,
                lxy = utils.tonp(masked_df['goodsv']['lxy'])
            )
            accumulator[category]['sv_mass'   ].fill(
                weight = masked_df['weight'], sample = sample_name,
                mass = utils.tonp(masked_df.goodsv.p4.mass)
            )
            accumulator[category]['m1_vtx_mass'   ].fill(
                weight = masked_df['weight'], sample = sample_name,
                mass = utils.tonp(masked_df['m1_vtx_mass'])  
            )
            
            if category == 'preselection_SS' or category == 'preselection_OS':

                accumulator[category]['sv_tM'].fill(
                    weight = masked_df['weight'], sample = sample_name,
                    mass = utils.tonp(masked_df.tmass_goodsv)
                )

                accumulator[category]['mu_tM'].fill(
                    weight = masked_df['weight'], sample = sample_name,
                    mass = utils.tonp(masked_df.tmass_promptmu)
                )

                accumulator[category]['musv_tM'].fill(
                    weight = masked_df['weight'], sample = sample_name,
                    mass = utils.tonp(masked_df.tmass_svmu)
                )

                accumulator[category]['corr_M'].fill(
                    weight = masked_df['weight'], sample = sample_name,
                    mass = utils.tonp(masked_df.mass_corr)
                )

            if category.startswith('selection'):
                # variables for CNN
                accumulator['columns'][sample_name]['isOS'] += utils.tonp(masked_df['isOS'])
                accumulator['columns'][sample_name]['pu_weight'] += utils.tonp(masked_df['weight'])
                accumulator['columns'][sample_name]['m1_vtx_mass'] += utils.tonp(masked_df['m1_vtx_mass'])
                accumulator['columns'][sample_name]['mu2_absdxy'] += utils.tonp(masked_df['second_mu']['absdxy'])
                accumulator['columns'][sample_name]['mu2_absdz'] += utils.tonp(masked_df['second_mu']['absdz'])
                accumulator['columns'][sample_name]['mu2_phi'] += utils.tonp(masked_df['second_mu'].p4.phi)
                accumulator['columns'][sample_name]['mu2_ptBT'] += utils.tonp(masked_df['second_mu']['pt_BT'])
                accumulator['columns'][sample_name]['mu2_etaBT'] += utils.tonp(masked_df['second_mu']['eta_BT'])
                accumulator['columns'][sample_name]['mu2_absdxySig'] += utils.tonp(masked_df['second_mu']['absdxySig'])
                accumulator['columns'][sample_name]['mu2_absdzSig'] += utils.tonp(masked_df['second_mu']['absdzSig'])
                accumulator['columns'][sample_name]['mu2_deltaBeta'] += utils.tonp(masked_df['second_mu']['deltaBeta'])
                accumulator['columns'][sample_name]['mu2_nDof'] += utils.tonp(masked_df['second_mu']['nDof'])
                accumulator['columns'][sample_name]['mu2_timeAtIpInOut'] += utils.tonp(masked_df['second_mu']['timeAtIpInOut'])
                accumulator['columns'][sample_name]['mu2_timeAtIpInOutErr'] += utils.tonp(masked_df['second_mu']['timeAtIpInOutErr'])
                accumulator['columns'][sample_name]['mu2_timeAtIpOutIn'] += utils.tonp(masked_df['second_mu']['timeAtIpOutIn'])
                accumulator['columns'][sample_name]['mu2_timeAtIpOutInErr'] += utils.tonp(masked_df['second_mu']['timeAtIpOutInErr'])
                accumulator['columns'][sample_name]['mu2_segmentComp'] += utils.tonp(masked_df['second_mu']['segmentComp'])
                accumulator['columns'][sample_name]['mu2_trkKink'] += utils.tonp(masked_df['second_mu']['trkKink'])
                accumulator['columns'][sample_name]['mu2_chi2LocalPosition'] += utils.tonp(masked_df['second_mu']['chi2LocalPosition'])
                accumulator['columns'][sample_name]['mu2_rhoRelIso'] += utils.tonp(masked_df['second_mu']['rho_rel_iso'])
                accumulator['columns'][sample_name]['sv_mass'] += utils.tonp(masked_df['goodsv'].p4.mass)
                accumulator['columns'][sample_name]['sv_pt'] += utils.tonp(masked_df['goodsv'].p4.pt)
                accumulator['columns'][sample_name]['sv_lxySig'] += utils.tonp(masked_df['goodsv']['lxySig'])
                accumulator['columns'][sample_name]['sv_lxyzSig'] += utils.tonp(masked_df['goodsv']['lxyzSig'])
                accumulator['columns'][sample_name]['sv_lxy'] += utils.tonp(masked_df['goodsv']['lxy'])
                accumulator['columns'][sample_name]['sv_lxyz'] += utils.tonp(masked_df['goodsv']['lxyz'])
                accumulator['columns'][sample_name]['sv_angle3D'] += utils.tonp(masked_df['goodsv']['angle3D'])
                accumulator['columns'][sample_name]['sv_angle2D'] += utils.tonp(masked_df['goodsv']['angle2D'])
                accumulator['columns'][sample_name]['sv_gamma'] += utils.tonp(masked_df['goodsv']['gamma'])
                accumulator['columns'][sample_name]['sv_chi2'] += utils.tonp(masked_df['goodsv']['chi2'])                
                accumulator['columns'][sample_name]['sv_sum_tracks_dxySig'] += utils.tonp(masked_df['goodsv']['sum_tracks_dxySig']).astype(np.float64)
                accumulator['columns'][sample_name]['mujet_eta'] += utils.tonp(masked_jets.p4.eta)
                accumulator['columns'][sample_name]['mujet_phi'] += utils.tonp(masked_jets.p4.phi)
                accumulator['columns'][sample_name]['mujet_neutHadEnFrac'] += utils.tonp(masked_jets['neutHadEnFrac'])
                accumulator['columns'][sample_name]['mujet_neutEmEnFrac'] += utils.tonp(masked_jets['neutEmEnFrac'])
                accumulator['columns'][sample_name]['mujet_charHadEnFrac'] += utils.tonp(masked_jets['charHadEnFrac'])
                accumulator['columns'][sample_name]['mujet_charEmEnFrac'] += utils.tonp(masked_jets['charEmEnFrac'])
                accumulator['columns'][sample_name]['mujet_neutMult'] += utils.tonp(masked_jets['neutMult'])
                accumulator['columns'][sample_name]['mujet_smeared_pt'] += utils.tonp(masked_jets['smeared_pt'])
                accumulator['columns'][sample_name]['mujet_dCsv_bb'] += utils.tonp(masked_jets['dCsv_bb'])
                accumulator['columns'][sample_name]['mujet_charEmEn'] += utils.tonp(masked_jets['charEmEn'])
                accumulator['columns'][sample_name]['mujet_charHadEn'] += utils.tonp(masked_jets['charHadEn'])
                accumulator['columns'][sample_name]['mujet_charMuEn'] += utils.tonp(masked_jets['charMuEn'])
                accumulator['columns'][sample_name]['mujet_charMuEnFrac'] += utils.tonp(masked_jets['charMuEnFrac'])
                accumulator['columns'][sample_name]['mujet_muonEn'] += utils.tonp(masked_jets['muonEn'])
                accumulator['columns'][sample_name]['mujet_muonEnFrac'] += utils.tonp(masked_jets['muonEnFrac'])
                accumulator['columns'][sample_name]['mujet_muonEn'] += utils.tonp(masked_jets['muonEn'])
                accumulator['columns'][sample_name]['mujet_muonEnFrac'] += utils.tonp(masked_jets['muonEnFrac'])
                accumulator['columns'][sample_name]['mujet_neutEmEn'] += utils.tonp(masked_jets['neutEmEn'])
                accumulator['columns'][sample_name]['mujet_neutHadEn'] += utils.tonp(masked_jets['neutHadEn'])
                accumulator['columns'][sample_name]['sv_tM'] += utils.tonp(masked_df['tmass_goodsv'])
                accumulator['columns'][sample_name]['mu1_tM'] += utils.tonp(masked_df['tmass_promptmu'])
                accumulator['columns'][sample_name]['mu2_tM'] += utils.tonp(masked_df['tmass_svmu'])
                accumulator['columns'][sample_name]['corr_M'] += utils.tonp(masked_df['mass_corr'])
                accumulator['columns'][sample_name]['dimu_deltaphi'] += utils.tonp(masked_df['dimu_deltaphi'])
                accumulator['columns'][sample_name]['dimu_mass'] += utils.tonp(masked_df['ll_mass'])
                accumulator['columns'][sample_name]['dimu_dr'] += utils.tonp(masked_df['ll_dr'])
                accumulator['columns'][sample_name]['nLooseMu'] += utils.tonp(masked_df['nLooseMu'])
                accumulator['columns'][sample_name]['nDisplacedMu'] += utils.tonp(masked_df['n_displaced_mu'])

                #######
                accumulator['columns'][sample_name]['sv_tracks_charge'] += utils.tonp(masked_df['goodsv']['tracks_charge'])
                accumulator['columns'][sample_name]['sv_tracks_eta'] += utils.tonp(masked_df['goodsv']['tracks_eta'] )
                accumulator['columns'][sample_name]['sv_tracks_phi'] += utils.tonp(masked_df['goodsv']['tracks_phi'])
                accumulator['columns'][sample_name]['sv_tracks_pt'] += utils.tonp(masked_df['goodsv']['tracks_pt'])
                accumulator['columns'][sample_name]['sv_tracks_p'] += utils.tonp(masked_df['goodsv']['tracks_p'])
                accumulator['columns'][sample_name]['sv_tracks_dxySig'] += utils.tonp(masked_df['goodsv']['tracks_dxySig'])
                accumulator['columns'][sample_name]['sv_tracks_dxy'] += utils.tonp(masked_df['goodsv']['tracks_dxy'])
                accumulator['columns'][sample_name]['sv_tracks_dxyz'] += utils.tonp(masked_df['goodsv']['tracks_dxyz'])
        
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
    processor_instance = SkimPlot(args.jobid, fileset.keys()),
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

# Save columns that are otherwise not stored
# import pandas as pd
# for sample, columns in output['columns'].items():
#     df = pd.DataFrame({
#         i : j for i, j in columns.items()
#         # TODO: special treatments for conv columns
#     })
#     df.to_hdf(
#         f'results/{args.jobid}/skimplot{args.tag}.coffea', 
#         sample, mode = 'a'
#     )

# remove columns, as they cannot be pickled
# del output['columns']
from coffea.util import save
if not os.path.isdir(f'results/{args.jobid}'):
    os.makedirs(f'results/{args.jobid}')
save(output, f'results/{args.jobid}/skimplot{args.tag}.coffea')

