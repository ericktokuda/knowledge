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
    info(inspect.stack()[0][3] + '()')

    if model == 'la':
        mapside = int(np.sqrt(nvertices))
        g = igraph.Graph.Lattice([mapside, mapside], nei=1, circular=latticethoroidal)
    elif model == 'er':
        erdosprob = avgdegree / nvertices
        if erdosprob > 1: erdosprob = 1
        g = igraph.Graph.Erdos_Renyi(nvertices, erdosprob)
        breakpoint()
        
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

    g = g.clusters().giant()

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
    return g, coords

##########################################################
def add_labels(gorig, m, choice, label):
    """Add @nresources to the @g.
    We randomly sample the vertices and change their labels"""
    info(inspect.stack()[0][3] + '()')
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

    models = ['er'] # ['ER', 'BA']
    nvertices = 100
    resoratio = .5
    nresources = int(resoratio * nvertices)
    nucleiratios = [.5] # np.arange(0, 1.01, .2)
    rewiringprob = 0.5
    avgdegree = 5
    niter = 10

    res = []
    for model in models:
        gorig, coords = generate_graph(model, nvertices, avgdegree, rewiringprob)
        gorig, resoids = add_labels(gorig, nresources, UNIFORM, RESOURCE)

        for c in nucleiratios:
            nnuclei = int(c * nvertices)
            for i in range(niter):
                neighs = []
                g, nuclids = add_labels(gorig, nnuclei, UNIFORM, NUCLEUS)
                for resoid in resoids:
                    # neighids = g.neighbors(resoid)
                    neighids = np.array(g.neighbors(resoid))
                    neightypes = np.array(g.vs[neighids.tolist()]['type'])
                    localids = np.where(neightypes == RESOURCE)[0]
                    neighs.extend(neighids[localids])

                lenunique = len(set(neighs))
                lenrepeated = len(neighs)
                r = lenunique / nvertices
                s = lenunique / lenrepeated
                res.append([model, c, i, r, s])

    print(res)
    breakpoint()

    info('For Aiur!')

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
