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
import igraph
from myutils import info, create_readme
import random
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
    elif model == 'ws':
        mapside = int(np.sqrt(nvertices))
        m = round(avgdegree/2)
        g = igraph.Graph.Lattice([mapside, mapside], nei=1, circular=False)
        g.rewire_edges(rewiringprob)
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
        if model in ['la', 'ws']:
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
def add_labels(gorig, m, choice, label):
    """Add @nresources to the @g.
    We randomly sample the vertices and change their labels"""
    # info(inspect.stack()[0][3] + '()')
    # TODO: use @choice
    g = gorig.copy()
    types = np.array(g.vs['type'])
    nones = np.where(types == NONE)[0]
    random.shuffle(nones)

    for i in range(m):
        idx = nones[i]
        g.vs[idx]['type'] = label
    return g, sorted(nones[:m])

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

    models = ['er', 'ba'] # er, ba
    nvertices = 1000
    resoratio = .5
    nresources = int(resoratio * nvertices)
    nucleiratios = np.arange(0, 1.01, .05)
    # nucleiratios = [0.1]
    rewiringprob = 0.5
    avgdegrees = [6, 20, 100]
    niter = 2


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
                    g, resoids = add_labels(g, nresources, UNIFORM, RESOURCE)
                    g, nuclids = add_labels(g, nnuclei, UNIFORM, NUCLEUS)

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
