#! /bin/env bash
export PYTHONPATH=$PWD


python preselection_analysis.py 2018_preskimV2  -T new_ST -l ST*
python preselection_analysis.py 2018_preskimV2  -T new_data -l Single*
python preselection_analysis.py 2018_preskimV2  -T new_W -l W*
python preselection_analysis.py 2018_preskimV2  -T new_Z -l Z*
python preselection_analysis.py 2018_preskimV2  -T new_TTJets -l TTJets*
python preselection_analysis.py 2018_preskimV2  -T new_QCD -l QCD*
python preselection_analysis.py 2018_preskimV2  -T new_DY -l DYJetsToLL*
python preselection_analysis.py 2018_preskimV2 -T new_Signal -l M-*
