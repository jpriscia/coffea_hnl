import utils
import os
from coffea import hist as cofplt
from coffea.util import load
from argparse import ArgumentParser
import matplotlib
from pdb import set_trace

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})
import styles
import re
from fnmatch import fnmatch
from collections import OrderedDict
from pdb import set_trace

parser = ArgumentParser()
parser.add_argument('jobid', default='2018_preskim', help='jobid to run on')
parser.add_argument('--isSig' , action='store_true', help='signal or mc/data')

#parser.add_argument
args = parser.parse_args()

hists = load(f'results/{args.jobid}/skimplot.coffea')
if not os.path.isdir(f'plots/{args.jobid}'):
    os.makedirs(f'plots/{args.jobid}')


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
else:
    raise RuntimeError('year not implemented')

exclude = {'DYJetsToLL_M-50_FXFX', 'DYJetsToLL_M-50_FXFX_ext'}
#target_lumi = 59.7e3 # /pb 2018
#target_lumi = 40e3 # /pb  2017
scaling = utils.compute_weights(
    f'inputs/{args.jobid}/mc.meta.json', year['lumi'],
    year['xsecx']
)

plots = [
    ('prompt_pt'  , r'p$_T$(leading $\mu$) [GeV]'),
    ('diplaced_pt', r'p$_T$(sub-leading $\mu$) [GeV]'),
    ('di_mu_M'    , r'm($\mu\mu$) [GeV]'),
    ('di_mu_DR'   , r'$\Delta R(\mu\mu$)'),
    ('sv_tM'      , r'sv tm [GeV]'),
    ('mu_tM'      , r'$\mu$ tm [GeV]'),
    ('musv_tM'    , r'$\mu$sv tm [GeV]'),
    ('corr_M'     , r'corr m [GeV]'),
    ('sv_lxy'     , r'l$_xy$ (sv) [cm]'),
    ('sv_mass'    , r'm(sv) [GeV]'),  
    ('m1_vtx_mass', r'm(pv) [GeV]'),
]

def matches(lst, pat, excl = set()):
    return [i for i in samples if fnmatch(i, pat) and i not in excl] 

mc = re.compile(r'mc_\w+')
sig = re.compile(r'sig_.+')

for key, x_title in plots:
    #for sel in ['preselection_SS','preselection_OS','selection_SS','selection_OS']:
    for sel in ['preselection_SS','preselection_OS']:
        fig, (ax, rax) = plt.subplots(
            2, 1, figsize = (7,7), 
            gridspec_kw = {"height_ratios": (3, 1)}, 
            sharex = True
        )
        fig.subplots_adjust(hspace = .07)

        #set_trace()
        if key not in hists[sel]: continue
        hist = hists[sel][key]
        hist.scale(scaling, axis = 'sample')    

        samples = [i.name for i in hist.axis('sample').identifiers()]
        mapping = OrderedDict()
        # define grouping maps
        mapping['mc_diboson'] = matches(samples, '[WZ][WZG]*')
        mapping['mc_singlet'] = matches(samples, 'ST_*')
        mapping['mc_dy'] = matches(samples, 'DY*', exclude)
        mapping['mc_tt'] = matches(samples, 'TTJets_*')
        mapping['mc_wjets'] = matches(samples, 'WJets*')
        mapping['mc_qcd'] = matches(samples, 'QCD*')
        mapping['data'] = matches(samples, 'Single*')
        mapping['sig_M-10_V-0p0010'] = matches(samples,'M-10_V-0p0010*')
        mapping['sig_M-2_V-0p015'] = matches(samples, 'M-2_V-0p015*')
        mapping['sig_M-2_V-0p035'] = matches(samples, 'M-2_V-0p035*')
        mapping['sig_M-4_V-0p0035'] = matches(samples, 'M-4_V-0p0035*')
        mapping['sig_M-4_V-0p010'] = matches(samples, 'M-4_V-0p010*' )
        mapping['sig_M-6_V-0p0028'] = matches(samples, 'M-6_V-0p0028*')
        mapping['sig_M-8_V-0p002'] = matches(samples, 'M-8_V-0p002*')

        process = cofplt.Cat("process", "Process", sorting='placement')
        grouped = hist.group('sample', process, mapping)
        
        grouped.axis('process').index('mc_diboson').label = r'diboson' 
        grouped.axis('process').index('mc_wjets').label = r'W + jets' 
        grouped.axis('process').index('mc_tt').label = r'$t\bar{t}$' 
        grouped.axis('process').index('mc_singlet').label = r'single top' 
        grouped.axis('process').index('mc_dy').label = r'Drell-Yan' 
        grouped.axis('process').index('mc_qcd').label = r'QCD' 
        grouped.axis('process').index('data').label = r'Observed' 
        grouped.axis('process').index('sig_M-10_V-0p0010').label = r'sig_M10_V0010'
        grouped.axis('process').index('sig_M-2_V-0p015').label = r'sig_M2_V015'
        grouped.axis('process').index('sig_M-2_V-0p035').label = r'sig_M2_V035'
        grouped.axis('process').index('sig_M-4_V-0p0035').label = r'sig_M4_V0035'
        grouped.axis('process').index('sig_M-4_V-0p010').label = r'sig_M4_V010'
        grouped.axis('process').index('sig_M-6_V-0p0028').label = r'sig_M6_V0028'
        grouped.axis('process').index('sig_M-8_V-0p002').label = r'sig_M8_V002'
       
        if args.isSig:
            cofplt.plot1d(
                grouped[sig], 
                overlay = "process", 
                ax = ax,
                clear = False,
                stack = False, 
                line_opts = None,
                fill_opts = styles.fill_opts,
                error_opts = styles.error_opts
            )

        else:       
            cofplt.plot1d(
                grouped[mc],
                overlay = "process",
                ax = ax,
                clear = False,
                stack = True,
                line_opts = None,
                fill_opts = styles.fill_opts,
                error_opts = styles.error_opts
            )
             
            cofplt.plot1d(
                grouped['data'],
                overlay = "process",
                ax = ax,
                clear = False,
                error_opts = styles.data_err_opts
            )
            
            cofplt.plot1d(
                grouped[sig],
                overlay = "process",
                ax = ax,
                clear = False,
                #error_opts = styles.data_err_opts
            )
            

        ax.autoscale(axis='x', tight=True)
        ax.set_ylim(0, None)
        ax.set_xlabel(None)
        ax.set_ylabel('Counts')
        leg = ax.legend()
       
        if not args.isSig:

            cofplt.plotratio(
                grouped['data'].sum("process"), grouped[mc].sum("process"), 
                ax=rax,
                error_opts=styles.data_err_opts, 
                denom_fill_opts={},
                guide_opts={},
                unc='num'
            )

        rax.set_ylabel('Ratio')
        rax.set_xlabel(x_title)
        rax.set_ylim(0,2)
        rax.set_yticks([0, 0.5, 1, 1.5, 2])
        
        lumi = plt.text(
            1., 1., f'{year["lumi"]/1000}'r" fb$^{-1}$ (13 TeV)",
            fontsize=16, 
            horizontalalignment='right', 
            verticalalignment='bottom', 
            transform=ax.transAxes
        )
        
        if not args.isSig:
            #plt.savefig(f'plots/{args.jobid}/{sel}_{key}.pdf')
            plt.savefig(f'plots/{args.jobid}/{sel}_{key}.png')
        
            ax.set_ylim(0.1, None)
            ax.set_yscale('log')
            #plt.savefig(f'plots/{args.jobid}/{sel}_{key}_log.pdf')
            plt.savefig(f'plots/{args.jobid}/{sel}_{key}_log.png')

        else:
            plt.savefig(f'plots/{args.jobid}/{sel}_{key}_Sig.png')
            ax.set_ylim(0.1, None)
            ax.set_yscale('log')
            plt.savefig(f'plots/{args.jobid}/{sel}_{key}_Sig_log.png')
