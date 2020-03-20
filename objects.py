from coffea.analysis_objects import JaggedCandidateArray
import numpy as np
from pdb import set_trace

#
# Always recast p4 elements to float64 to limit rounding errors as much as possible!
#

def make_muons(df):
    nmuons = df['mu_pt'].counts
    return JaggedCandidateArray.candidatesfromcounts(
        nmuons,
        pt = df['mu_pt'].flatten().astype(np.float64),
        eta = df['mu_eta'].flatten().astype(np.float64),
        phi = df['mu_phi'].flatten().astype(np.float64),
        mass = (np.ones(nmuons.sum()) * 0.105).astype(np.float64),
        charge = df['mu_charge'].flatten(),
        isTight = (df['mu_isTightMuon'].flatten() > 0.5),
        isLoose = (df['mu_isLooseMuon'].flatten() > 0.5),
        dbIso = df['mu_recoDeltaBeta'].flatten(),
        absdxy = df['mu_absdxyTunePMuonBestTrack'].flatten(),
        absdz = df['mu_absdzTunePMuonBestTrack'].flatten(),
        # Add here what is needed
        )

def make_electrons(df):
    pass

def make_svs(df):
    nsv = df['sv_pt'].counts
    return JaggedCandidateArray.candidatesfromcounts(
        nsv,        
        mass   = df['sv_mass'].flatten().astype(np.float64),
        # charge = df['sv_charge'].flatten(), # SV CHARGE IS NOT FILLED!
        eta    = df['sv_eta'].flatten().astype(np.float64),
        phi    = df['sv_phi'].flatten().astype(np.float64),
        pt     = df['sv_pt'].flatten().astype(np.float64),
        )

