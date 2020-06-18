import os
import random

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rfn

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import learning_curve

#import pydot
#import StringIO

from scipy.stats import ks_2samp

######PLOT VARIABLES FOR SIGNAL AND BKG###############
def signal_bkg(data1, data2, column=None, grid=True,
                                            xlabelsize=None, xrot=None, ylabelsize=None,
                                            yrot=None, ax=None, sharex=False,
                                            sharey=False, figsize=None,layout=None, bins=10,infodis='var', **kwds):

    if 'alpha' not in kwds:
        kwds['alpha'] = 0.5

    if column is not None:
        if not isinstance(column, (list, np.ndarray, Index)):
            column = [column]
        data1 = data1[column]
        data2 = data2[column]

    data1 = data1._get_numeric_data()
    data2 = data2._get_numeric_data()
    naxes = len(data1.columns)
                                                                      
    fig, axes = plotting._subplots(naxes=naxes, ax=ax, squeeze=False, sharex=sharex,sharey=sharey,figsize=figsize,layout=layout)
    _axes = plotting._flatten(axes)

    for i, col in enumerate(com._try_sort(data1.columns)):
        ax = _axes[i]
        low = min(data1[col].min(), data2[col].min())
        high = max(data1[col].max(), data2[col].max())
        ax.hist(data1[col].dropna().values,bins=bins, range=(low,high), **kwds)
        ax.hist(data2[col].dropna().values, bins=bins, range=(low,high), **kwds)
        ax.set_title(col)
        ax.grid(grid)
        plotting._set_ticks_props(axes, xlabelsize=xlabelsize, xrot=xrot,
                                  ylabelsize=ylabelsize, yrot=yrot)
        fig.subplots_adjust(wspace=0.3, hspace=0.7)
        #axes.plot()
        fig.show()
        fig.savefig('distributions'+infodis+str(n_var)+'.pdf')
        return axes
        
        
######## VARIABLE CORRELATIONS ###############
def get_correlations(data,string_fig='unknown', **kwds):
        """Calculate pairwise correlation between features.
            Extra arguments are passed on to DataFrame.corr()
         """
         # simply call df.corr() to get a table of
         # correlation values if you do not need
         # the fancy plotting
        corrmat = data.corr(**kwds)

        fig, ax1 = plt.subplots(ncols=1, figsize=(6,5))

        opts = {'cmap': plt.get_cmap("RdBu"),
                 'vmin': -1, 'vmax': +1}
        heatmap1 = ax1.pcolor(corrmat, **opts)
        plt.colorbar(heatmap1, ax=ax1)

        ax1.set_title("Correlations")

        labels = corrmat.columns.values
        for ax in (ax1,):
        # shift location of ticks to center of the bins
            ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
            ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
            ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
            ax.set_yticklabels(labels, minor=False)

        plt.tight_layout()
        plt.show()
        plt.savefig('coor_matr'+string_fig+str(n_var)+'.pdf')
        # remove the y column from the correlation matrix
        # after using it to select background and signal
        

############## OVERTRAINING ################
        
def compare_train_test(clf, X_train, y_train, X_test, y_test, bins=30,bdt_set='0'):
        decisions = []
        for X,y in ((X_train, y_train), (X_test, y_test)):
            #print 'clf:', clf.decision_function(X[y>0.5]) 
            d1 = clf.decision_function(X[y>0.5]).ravel()
            #print d1
            d2 = clf.decision_function(X[y<0.5]).ravel()
            decisions += [d1, d2]
        print ('type', type(decisions[0]))
        test_KS_signal= ks_2samp(decisions[0], decisions[2])
        test_KS_bkg= ks_2samp(decisions[1], decisions[3])
        print (test_KS_signal)
        print (test_KS_bkg)
        low = min(np.min(d) for d in decisions)
        high = max(np.max(d) for d in decisions)
        low_high = (low,high)
        fig_OT = plt.figure()
        axOT = fig_OT.add_subplot(111)
        plt.hist(decisions[0],
                color='r', alpha=0.5, range=low_high, bins=bins,
                histtype='stepfilled', density=True,
                label='S (train)')
        plt.hist(decisions[1],
                color='b', alpha=0.5, range=low_high, bins=bins,
                histtype='stepfilled', density=True,
                label='B (train)')

        hist, bins = np.histogram(decisions[2],bins=bins, range=low_high, density=True)
        scale = len(decisions[2]) / sum(hist)
        err = np.sqrt(hist * scale) / scale

        width = (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

        hist, bins = np.histogram(decisions[3],bins=bins, range=low_high, density=True)
        scale = len(decisions[2]) / sum(hist)
        err = np.sqrt(hist * scale) / scale
        
        plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

        axOT.set_xlabel("BDT output")
        axOT.set_ylabel("Arbitrary units")
        axOT.legend(loc='best')
        fig_OT.show()
        fig_OT.savefig('OverTraining_'+bdt_set+'.pdf')
        

#####OVERTRAINING WITH DECISION FUNCTION #######

def compare_train_test_dec(decisionTrainSigAll,decisionTrainBkgAll,decisionTestSigAll,decisionTestBkgAll, bins=30,bdt_set='0'):
        decisions = [decisionTrainSigAll,decisionTrainBkgAll,decisionTestSigAll,decisionTestBkgAll]
        print ('type', type(decisions[0]))
        test_KS_signal= ks_2samp(decisions[0], decisions[2])
        test_KS_bkg= ks_2samp(decisions[1], decisions[3])
        print (test_KS_signal)
        print (test_KS_bkg)
        low = min(np.min(d) for d in decisions)
        high = max(np.max(d) for d in decisions)
        low_high = (low,high)
        fig_OT = plt.figure()
        axOT = fig_OT.add_subplot(111)
        plt.hist(decisions[0],
                color='r', alpha=0.5, range=low_high, bins=bins,
                histtype='stepfilled', density=True,
                label='S (train)')
        plt.hist(decisions[1],
                color='b', alpha=0.5, range=low_high, bins=bins,
                histtype='stepfilled', density=True,
                label='B (train)')

        hist, bins = np.histogram(decisions[2],bins=bins, range=low_high, density=True)
        scale = len(decisions[2]) / sum(hist)
        err = np.sqrt(hist * scale) / scale

        width = (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

        hist, bins = np.histogram(decisions[3],bins=bins, range=low_high, density = True)
        scale = len(decisions[2]) / sum(hist)
        err = np.sqrt(hist * scale) / scale
        
        plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

        axOT.set_xlabel("output")
        axOT.set_ylabel("Arbitrary units")
        axOT.legend(loc='best')
        fig_OT.savefig('OverTraining.pdf')
        




# Taken more or less straight from
# http://scikit-learn.org/stable/auto_examples/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5),
                        scoring=None, ax=None, xlabel=True):
    #fig_learn = plt.figure()                    
    if ax is None:
        plt.figure()
        ax.title(title)
    
    if xlabel:
        ax.set_xlabel("Training examples")
        
    ax.set_ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator,
                                                            X, y,
                                                            cv=cv,
                                                            n_jobs=n_jobs,
                                                            train_sizes=train_sizes,
                                                            scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std =  np.std(train_scores, axis=1)
    test_scores_mean =  np.mean(test_scores, axis=1)
    test_scores_std =   np.std(test_scores, axis=1)

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    ax.set_ylim([0.65, 1.0])
    #ax.legend(loc="best")
    ax.set_xlabel("Training examples")
    plt.savefig('plot__learningcurvesbdt1_.pdf')
    plt.show()
    return plt
         


