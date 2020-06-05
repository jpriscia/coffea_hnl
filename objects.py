from coffea.analysis_objects import JaggedCandidateArray
import numpy as np
from pdb import set_trace
import uproot_methods
import awkward as awk
#
# Always recast p4 elements to float64 to limit rounding errors as much as possible!
#

def make_muons(df):
    def eff_area(obj):
        'Returns values of effective area, values may be year dependent'
        areas = obj.charge.ones_like()
        abseta = np.abs(obj.p4.eta)
        areas_dict = {
            (-1, 0.8)  : 0.0735,
            (0.8, 1.3) : 0.0619,
            (1.3, 2.0) : 0.0465,
            (2.0, 2.2) : 0.0433,
            (2.2, 2.4) : 0.0577,
        }
        for mM, val in areas_dict.items():
            m, M = mM
            mask = (abseta > m) & (abseta <= M)
            areas[mask] = val
        return areas
        
    nmuons = df['mu_pt'].counts
    temp = JaggedCandidateArray.candidatesfromcounts(
        nmuons,
        pt = df['mu_pt'].flatten().astype(np.float64),
        eta = df['mu_eta'].flatten().astype(np.float64),
        phi = df['mu_phi'].flatten().astype(np.float64),  #for BDT - second muon - Mohamed does use phiTune blabla.. but I don't have it in the tuples
        mass = (np.ones(nmuons.sum()) * 0.105).astype(np.float64),
        charge = df['mu_charge'].flatten(),
        isTight = (df['mu_isTightMuon'].flatten() > 0.5),
        isLoose = (df['mu_isLooseMuon'].flatten() > 0.5),
        dbIso = df['mu_recoDeltaBeta'].flatten(),
        absdxy = df['mu_absdxyTunePMuonBestTrack'].flatten(),    #for BDT - second muon
        absdz = df['mu_absdzTunePMuonBestTrack'].flatten(),      #for BDT - second muon
        #### Here after variables for BDT ###
        pt_BT = df['mu_ptTunePMuonBestTrack'].flatten().astype(np.float64),   #for second muon 
        eta_BT = df['mu_etaTunePMuonBestTrack'].flatten().astype(np.float64),  #for second muon
        absdxySig = df['mu_absdxySigTunePMuonBestTrack'].flatten().astype(np.float64),  #for second muon
        absdzSig = df['mu_absdzSigTunePMuonBestTrack'].flatten().astype(np.float64),    #for second muon
        deltaBeta= df['mu_recoDeltaBeta'].flatten().astype(np.float64),  #for second muon
        nDof = df['mu_STATofNDof'].flatten().astype(np.float64),   #for second muon
        timeAtIpInOut = df['mu_STATofTimeAtIpInOut'].flatten().astype(np.float64),   #for second muon
        timeAtIpInOutErr = df['mu_STATofTimeAtIpInOutErr'].flatten().astype(np.float64),  #for second muon
        timeAtIpOutIn = df['mu_STATofTimeAtIpOutIn'].flatten().astype(np.float64),    #for second muon
        timeAtIpOutInErr = df['mu_STATofTimeAtIpOutInErr'].flatten().astype(np.float64),  #for second muon
        segmentComp = df['mu_segmentCompatibilityMuonBestTrack'].flatten().astype(np.float64), #for second muon
        trkKink = df['mu_trkKinkMuonBestTrack'].flatten().astype(np.float64), # for second muon
        chi2LocalPosition = df['mu_chi2LocalPositionMuonBestTrack'].flatten().astype(np.float64), #for second muon
        # isolation
        charged_iso = df['mu_pfSumChargedHadronPt'].flatten().astype(np.float64),
        neutral_iso = df['mu_pfSumNeutralHadronEt'].flatten().astype(np.float64),
        pho_iso = df['mu_PFSumPhotonEt'].flatten().astype(np.float64), # ??
        purho_iso = df['mu_rhoIso'].flatten().astype(np.float64),
        )
    
    temp['eff_area'] = eff_area(temp)
    neutral_iso_pu = temp.neutral_iso + temp.pho_iso - temp.purho_iso*temp.eff_area
    neutral_iso_pu[neutral_iso_pu < 0] = 0.
    temp['rho_rel_iso'] = (temp.charged_iso + neutral_iso_pu)/temp.p4.pt
    return temp

def make_electrons(df):
    pass

def make_svs(df):
    nsv = df['sv_pt'].counts
    temp = JaggedCandidateArray.candidatesfromcounts(
        nsv,        
        mass   = df['sv_mass'].flatten().astype(np.float64), #BDT
        # charge = df['sv_charge'].flatten(), # SV CHARGE IS NOT FILLED!
        eta    = df['sv_eta'].flatten().astype(np.float64),
        phi    = df['sv_phi'].flatten().astype(np.float64),
        pt     = df['sv_pt'].flatten().astype(np.float64),    #BDT
        lxySig = df['sv_LxySig'].flatten().astype(np.float64),  #BDT
        lxyzSig = df['sv_LxyzSig'].flatten().astype(np.float64), #BDT
        lxy    =  df['sv_Lxy'].flatten().astype(np.float64),  #BDT
        lxyz   =  df['sv_Lxyz'].flatten().astype(np.float64), #BDT
        angle3D = df['sv_Angle3D'].flatten().astype(np.float64), #BDT
        angle2D = df['sv_Angle2D'].flatten().astype(np.float64), #BDT
        gamma  = df['sv_Gamma'].flatten().astype(np.float64),     #BDT
        chi2   = df['sv_Chi2'].flatten().astype(np.float64),   #BDT
        
        position = uproot_methods.TVector3Array(
            df['sv_Lx'].flatten().astype(np.float64),
            df['sv_Ly'].flatten().astype(np.float64),
            df['sv_Lz'].flatten().astype(np.float64),
        )
    )
    
    # Add CNN variables, they are stored as ObjectArrays instead of
    # doubly JaggedArrays because ROOT. The conversion is expensive
    # ans is probably better to do later on if needed (and probably
    # will be needed) when fewer events are stored.

    # vectors of vectors are a mess to handle, so we need to mem-copy 
    # them. A better solution for the future would be to unroll the 
    # values in each event and then roll them back

    temp['tracks_charge'] = awk.JaggedArray.fromiter(df['sv_tracks_charge'])
    temp['tracks_eta']    = awk.JaggedArray.fromiter(df['sv_tracks_eta'])
    temp['tracks_phi']    = awk.JaggedArray.fromiter(df['sv_tracks_phi'])
    temp['tracks_pt']     = awk.JaggedArray.fromiter(df['sv_tracks_pt'])
    temp['tracks_p']      = awk.JaggedArray.fromiter(df['sv_tracks_p'])
    temp['tracks_dxySig'] = awk.JaggedArray.fromiter(df['sv_tracks_dxySig'])
    temp['tracks_dxy']    = awk.JaggedArray.fromiter(df['sv_tracks_dxy'])
    temp['tracks_dxyz']   = awk.JaggedArray.fromiter(df['sv_tracks_dxyz'])  

    temp['p3'] = uproot_methods.TVector3Array.from_cartesian(temp.p4.x,temp.p4.y,temp.p4.z)
    temp['sum_tracks_dxySig'] = np.abs(temp['tracks_dxySig']).sum()

    return temp


def make_jets(df):
    njets = df['jet_pt'].counts
    temp = JaggedCandidateArray.candidatesfromcounts(
        njets,
        pt = df['jet_pt'].flatten().astype(np.float64),
        eta = df['jet_eta'].flatten().astype(np.float64), #BDT for mujet
        phi = df['jet_phi'].flatten().astype(np.float64),  #BDT for mujet
        energy = df['jet_en'].flatten().astype(np.float64),
        neutHadEnFrac = df['jet_neutralHadronEnergyFraction'].flatten().astype(np.float64), #BDT for mujet
        neutEmEnFrac =df['jet_neutralEmEnergyFraction'].flatten().astype(np.float64),  #BDT for mujet
        charHadEnFrac = df['jet_chargedHadronEnergyFraction'].flatten().astype(np.float64),#BDT for mujet
        charEmEnFrac = df['jet_chargedEmEnergyFraction'].flatten().astype(np.float64), #BDT for mujet
        chargedMult =df['jet_chargedMultiplicity'].flatten().astype(np.float64), #BDT for mujet
        neutMult =df['jet_neutralMultiplicity'].flatten().astype(np.float64),    #BDT for mujet
        smeared_pt = df['jetSmearedPt'].flatten().astype(np.float64), #BDT for mujet
        dCsv_bb = df['jet_deepCSV_bb'].flatten().astype(np.float64), #BDT for mujet
        charEmEn = df['jet_chargedEmEnergy'].flatten().astype(np.float64), #BDT for mujet
        charHadEn = df['jet_chargedHadronEnergy'].flatten().astype(np.float64),##BDT for mujet
        charMuEn = df['jet_chargedMuEnergy'].flatten().astype(np.float64),##BDT for mujet
        charMuEnFrac = df['jet_chargedMuEnergyFraction'].flatten().astype(np.float64),##BDT for mujet
        muonEn  = df['jet_muonEnergy'].flatten().astype(np.float64),##BDT for mujet
        muonEnFrac = df['jet_muonEnergyFraction'].flatten().astype(np.float64),##BDT for mujet
        neutEmEn = df['jet_neutralEmEnergy'].flatten().astype(np.float64),##BDT for mujet
        neutHadEn = df['jet_neutralHadronEnergy'].flatten().astype(np.float64),##BDT for mujet
    )
    temp['p3'] = uproot_methods.TVector3Array.from_cartesian(temp.p4.x,temp.p4.y,temp.p4.z)
    return temp




##########CNN Varables######
#sv_tracks_charge
#sv_tracks_eta
#sv_tracks_phi
#sv_tracks_pt
#sv_tracks_p
#sv_tracks_dxySig
#sv_tracks_dxy
#sv_tracks_dxyz
#################################
####### FOR BDT ################
#'mu_secondPhi',    #mu_phiTunePMuonBestTrack      does not exist , can I use phi
#"mu_second3dIP",  not found
# "mu_second3dIPSig",  not found
# "mu_second2dIP",   not found
# "mu_second2dIPSig",  not found
#"mu_DeltaPhi",  to be added, I think in preselection analysus
#"mu_secondDirection", #STATofDirection  I don't have it
#"mu_secondMatchedStations", i don't have it
#"mu_secondValidPixelHits",  i don't have it
#"mu_secondTrackQuality",  i don't have it
#"mu_secondInrTrackQuality",   i don't have it
#"mu_secondValidMuonHits",   i don't have it
#"mu_secondPixelLayers",      i don't have it
#"mu_secondTrackerLayers",    i don't have it
#"mu_Size",            #to compute                              !!!!!!!!!!!!!!!!!
#"mu_secondInnerTrackFraction",  i don't have it
# "mu_secondGlobalMuon",#end new    does not have any sense to add it
#"secondjet_bjet_L",  to be undestood                             !!!!!!!!!!!!!!!!!
#"mujet_RSecMu" to be undestood                                     !!!!!!!!!!!!!!!!!
