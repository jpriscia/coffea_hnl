labels = {
    '[WZ][WZG]' : 'diboson',
    'WJets*' : 'W + Jets',
    'TTJets_' : r'$t\bar{t}$',
    'ST_*' : 'single top',
    'DY*' : 'Drell-Yan',
    'Single*' : 'Observed',
}

from fnmatch import fnmatch
def assign_labels(hist):
    for i in hist.axis('sample').identifiers():
        for group in labels:
            if fnmatch(i.name, group):
                hist.axis('sample').index(i).label = labels[group]
    return hist

fill_opts = {
    'edgecolor': (0,0,0,0.3),
    'alpha': 0.8
}

error_opts = {
    'label':'Stat. Unc.',
    'hatch':'///',
    'facecolor':'none',
    'edgecolor':(0,0,0,.5),
    'linewidth': 0
}

data_err_opts = {
    'linestyle':'none',
    'marker': '.',
    'markersize': 10.,
    'color':'k',
    'elinewidth': 1,
}
