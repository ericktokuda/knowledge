#!/usr/bin/env python3
"""Scientific discovery
"""

import argparse
import time
import os
from os.path import join as pjoin
import inspect

import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random, igraph, scipy
import scipy.optimize
from myutils import info, create_readme
import pandas as pd

#############################################################
NONE = 0
NUCLEUS = 1
RESOURCE = 2
#############################################################
UNIFORM = 0
DEGREE = 1
#############################################################
def generate_graph(model, nvertices, avgdegree, rewiringprob,
                   latticethoroidal=False, tmpdir='/tmp/'):
    """Generate graph with given topology """
    # info(inspect.stack()[0][3] + '()')

    if model == 'la':
        mapside = int(np.sqrt(nvertices))
        g = igraph.Graph.Lattice([mapside, mapside], nei=1, circular=latticethoroidal)
    elif model == 'er':
        erdosprob = avgdegree / nvertices
        if erdosprob > 1: erdosprob = 1
        g = igraph.Graph.Erdos_Renyi(nvertices, erdosprob)
    elif model == 'ba':
        m = round(avgdegree/2)
        if m == 0: m = 1
        g = igraph.Graph.Barabasi(nvertices, m)
    elif model == 'gr':
        radius = get_rgg_params(nvertices, avgdegree)
        g = igraph.Graph.GRG(nvertices, radius)
    else:
        msg = 'Please choose a proper topology model'
        raise Exception(msg)

    if model in ['gr', 'wx']:
        aux = np.array([ [g.vs['x'][i], g.vs['y'][i]] for i in range(g.vcount()) ])
        # layoutmodel = 'grid'
    else:
        if model in ['la']:
            layoutmodel = 'grid'
        else:
            layoutmodel = 'random'
        aux = np.array(g.layout(layoutmodel).coords)
    # coords = (aux - np.mean(aux, 0))/np.std(aux, 0) # standardization
    coords = -1 + 2*(aux - np.min(aux, 0))/(np.max(aux, 0)-np.min(aux, 0)) # minmax
    g.vs['type'] = NONE
    g['coords'] = coords
    return g

##########################################################
def weighed_random_sampling(items, weights, return_idx=False):
    n = len(items)
    # weights += 1 # TODO: remove this
    cumsum = np.cumsum(weights)
    cumsumnorm = cumsum / cumsum[-1]
    x = np.random.rand()
    #binary search to find the item

    for i in range(n):
        if x < cumsumnorm[i]:
            if return_idx: return i
            else: return items[i]

    info('Something wrong x:{}'.format(x))

    if return_idx: return -1
    else: return items[-1]

##########################################################
def add_labels(gorig, m, choice, label):
    """Add @nresources to the @g.
    We randomly sample the vertices and change their labels"""
    # info(inspect.stack()[0][3] + '()')
    g = gorig.copy()
    types = np.array(g.vs['type'])
    degrs = np.array(g.degree())
    nones = np.where(types == NONE)[0]
    degrs = degrs[nones]

    if choice == UNIFORM:
        sample = nones.copy()
        random.shuffle(sample)
    elif choice == DEGREE:
        sample = np.zeros(m, dtype=int)
        inds = list(range(len(nones)))
        # nones = list(nones)
        for i in range(m):
            sampleidx = weighed_random_sampling(nones, degrs, return_idx=True)
            sample[i] = nones[sampleidx]
            nones = np.delete(nones, sampleidx)
            degrs = np.delete(degrs, sampleidx)

    for i in range(m):
        idx = sample[i]
        g.vs[idx]['type'] = label
    return g, sorted(sample[:m])


##########################################################
def get_rgg_params(nvertices, avgdegree):
    rggcatalog = {
        '625,6': 0.056865545,
        '10000,6': 0.0139,
        '11132,6': 0.0131495,
        '22500,6': 0.00925,
    }

    if '{},{}'.format(nvertices, avgdegree) in rggcatalog.keys():
        return rggcatalog['{},{}'.format(nvertices, avgdegree)]

    def f(r):
        g = igraph.Graph.GRG(nvertices, r)
        return np.mean(g.degree()) - avgdegree

    return scipy.optimize.brentq(f, 0.0001, 10000)
##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    np.random.seed(args.seed)
    random.seed(args.seed)

    mapside = 500
    plotzoom = 1

    visual = dict(
        bbox = (mapside*10*plotzoom, mapside*10*plotzoom),
        margin = mapside*plotzoom,
        vertex_size = 5*plotzoom,
        vertex_shape = 'circle',
        vertex_frame_width = 0.1*plotzoom,
        edge_width=1.0
    )

    models = ['er', 'ba', 'gr'] # er, ba
    nvertices = 1000
    resoratio = .5
    nresources = int(resoratio * nvertices)
    # nucleiratios = np.arange(0, 1.01, .1)
    nucleiratios = np.arange(0, 1.01, .2)
    # nucleiratios = [0.1]
    rewiringprob = 0.5
    avgdegrees = [6, 20, 100]
    # avgdegrees = [6]
    niter = 3

    weights = [10, 100, 40]

    res = []
    for avgdegree in avgdegrees:
        for model in models:
            plotpath = pjoin(args.outdir, '{}_{}.pdf'.format(model, avgdegree))
            for c in nucleiratios:
                info('{},{},{:.02f}'.format(avgdegree, model, c))
                nnuclei = int(c * nvertices)
                for i in range(niter):
                    if c == 0 or c == 1:
                        res.append([model, avgdegree, c, i, 0, 1.0 - c])
                        continue

                    neighs = []

                    nresources = nvertices - nnuclei
                    g = generate_graph(model, nvertices, avgdegree, rewiringprob)
                    g, nuclids = add_labels(g, nnuclei, DEGREE, NUCLEUS)
                    g, resoids = add_labels(g, nresources, UNIFORM, RESOURCE)

                    if not os.path.exists(plotpath):
                        igraph.plot(g, plotpath, **visual)

                    for nucl in nuclids:
                        neighids = np.array(g.neighbors(nucl))
                        neightypes = np.array(g.vs[neighids.tolist()]['type'])
                        neighresoids = np.where(neightypes == RESOURCE)[0]
                        neighs.extend(neighids[neighresoids])

                    lenunique = len(set(neighs))
                    lenrepeated = len(neighs)

                    r = lenunique / nvertices
                    s = lenunique / lenrepeated if lenunique > 0 else 0
                    res.append([model, avgdegree, c, i, r, s])

    df = pd.DataFrame()
    cols = ['model', 'k', 'c', 'i', 'r', 's']
    for i, col in enumerate(cols):
        df[col] = [x[i] for x in res]

    dfmean = df.groupby(['model', 'k', 'c']).mean()
    dfstd = df.groupby(['model', 'k', 'c']).std()
    cols = list(dfmean.index)

    figscale = 4
    fig, axs = plt.subplots(len(avgdegrees), 2, squeeze=False,
                figsize=(2*figscale, len(avgdegrees)*figscale*.6))

    for i, k in enumerate(avgdegrees):
        for j, meas in enumerate(['r', 's']):
            for model in models:
                cols = [(model, k, r) for r in nucleiratios]
                axs[i, j].errorbar(nucleiratios, dfmean.loc[cols][meas].values,
                                yerr=dfstd.loc[cols][meas].values, label=model)
                axs[i, j].set_xlabel('c')
                axs[i, j].set_ylabel(meas)
                axs[i, j].legend()

    plt.tight_layout(pad=4, h_pad=1)
    dh = 1/ len(avgdegrees)
    for i, k in enumerate(avgdegrees):
        plt.figtext(.5, 1 - i * dh, '<k>:{}'.format(avgdegree),
                    ha='center', va='center')

    outpath = pjoin(args.outdir, 'plot.png')
    plt.savefig(outpath)
    info('For Aiur!')

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
