import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import roc_curve, auc
from scipy.stats import ks_2samp

import itertools

def overtrainDist( train_decsig, train_decbkg, test_decsig, test_decbkg, **kwargs ):
    nbins  = kwargs.get( 'nbins', 20 )
    makeks = kwargs.get( 'ks', True )
    ''' Extract the test and training distributions '''
    lowv = min(min(lst) for lst in (train_decsig, train_decbkg, test_decsig, test_decbkg))
    upv  = max(max(lst) for lst in (train_decsig, train_decbkg, test_decsig, test_decbkg))
    rg   = (lowv, upv)
    sigwts = numpy.ones_like(train_decsig)/float(len(train_decsig))
    bkgwts = numpy.ones_like(train_decbkg)/float(len(train_decbkg))
    trargs  = {'bins': nbins, 'alpha': 0.6, 'histtype': 'stepfilled', 'range': rg}
    ''' Train histograms '''
    plt.hist( train_decsig, color = 'b', weights = sigwts, label = 'Train (B)', **trargs )
    plt.hist( train_decbkg, color = 'r', weights = bkgwts, label = 'Train (S)', **trargs )
    ''' Test histograms '''
    tsig, sig_edges = numpy.histogram( test_decsig, bins = nbins, range = rg )
    tbkg, bkg_edges = numpy.histogram( test_decbkg, bins = nbins, range = rg )
    sig_bc = (sig_edges[:-1] + sig_edges[1:])/2.
    bkg_bc = (bkg_edges[:-1] + bkg_edges[1:])/2.
    sigerr = numpy.sqrt(tsig)/sum(tsig)
    bkgerr = numpy.sqrt(tbkg)/sum(tbkg)
    plt.errorbar( sig_bc, tsig*1./sum(tsig), yerr = sigerr, fmt = 'o', color = 'b', label = 'Test (B)' )
    plt.errorbar( bkg_bc, tbkg*1./sum(tbkg), yerr = bkgerr, fmt = 'o', color = 'r', label = 'Test (S)' )
    plt.legend()
    plt.xlabel('MVA algorithm value')
    plt.ylabel('Nomalized entries')
    plt.title('MVA algorithm distribution and overtraining test')
    ''' Make Kolmogorov-Smirnov test '''
    if makeks:
        print ('--- Kolmogorov-Smirnov tests:')
        print ('---  o Sig =>', ks_2samp( train_decsig, test_decsig ))
        print ('---  o Bkg =>', ks_2samp( train_decbkg, test_decbkg ))

def makeCorrPlot( smp, **kwargs ):
    title     = kwargs.get('title', 'Correlation')
    variables = kwargs.get('variables', smp.columns)
    fig, ax = plt.subplots(ncols=1, figsize=(6,5))
    corrmat = smp.corr( **{el: kwargs[el] for el in kwargs if el not in ['title', 'variables']})
    for cl in list(smp.columns):
        if cl not in variables:
            corrmat.drop( cl, axis = 1, inplace = True )
            corrmat.drop( [cl], inplace = True )
    corrmap = ax.pcolor( corrmat, cmap = plt.get_cmap('jet'), vmin = -1, vmax = +1 )
    plt.colorbar( corrmap, ax=ax )
    labels = corrmat.columns.values
    ax.set_xticks(numpy.arange(len(labels))+0.5, minor=False)
    ax.set_yticks(numpy.arange(len(labels))+0.5, minor=False)
    ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
    ax.set_yticklabels(labels, minor=False)
    plt.tight_layout()

def makeROC( target, dec, **kwargs ):
    ret = kwargs.get('ret', False)
    col = kwargs.get('color', 'b')
    ls  = kwargs.get('linestyle', '-')
    nm  = kwargs.get('name', 'ROC')
    fpr, tpr, thresholds = roc_curve(target, dec)
    bkgrej = []
    for bkg in fpr:
        bkgrej.append( 1. - bkg )
    roc_auc = auc(fpr, tpr)
    plt.plot(tpr, bkgrej,
             col + ls + '-', lw=1, ms=4, markevery=len(tpr)%100,
             label = nm + ' (area = %0.2f)'%(roc_auc))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Signal efficiency')
    plt.ylabel('Background rejection')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    ret = kwargs.get( 'ret', False )
    if ret:
        return tpr, bkgrej

# Plot in the same canvas the ROCs for different MVA algorithms. As in
def compareROCs( bdts ):
    colors = ['b', 'r', 'g', 'c', 'm', 'y']
    linest = ['o', 's', '^', '*', '+', 'd']
    collst = zip( colors, linest )
    linest = numpy.concatenate( list(itertools.permutations( collst )) )
    n = len( linest[:len(bdts)] )
    retdic = {}
    for (col, ls), (kw, bdtlst) in zip( linest[:n], bdts.iteritems()):
        print ('--- BDT type: %s (drawn in < %s > and line style < %s >)' %(kw, col, ls))
        ret = makeROC( *bdtlst, color = col, linestyle = ls, name = kw )
        if ret:
            retdic[ kw ] = ret
    if retdic:
        return ret
