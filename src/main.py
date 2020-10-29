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
from myutils import info, create_readme

#############################################################
def generate_graph(model, nvertices, avgdegree, rewiringprob,
                   latticethoroidal=False, tmpdir='/tmp/'):
    """Generate graph with given topology """

    if topologymodel == 'la':
        mapside = int(np.sqrt(nvertices))
        g = igraph.Graph.Lattice([mapside, mapside], nei=1, circular=latticethoroidal)
    elif topologymodel == 'er':
        erdosprob = avgdegree / nvertices
        if erdosprob > 1: erdosprob = 1
        g = igraph.Graph.Erdos_Renyi(nvertices, erdosprob)
    elif topologymodel == 'ba':
        m = round(avgdegree/2)
        if m == 0: m = 1
        g = igraph.Graph.Barabasi(nvertices, m)
    elif topologymodel == 'ws':
        mapside = int(np.sqrt(nvertices))
        m = round(avgdegree/2)
        g = igraph.Graph.Lattice([mapside, mapside], nei=1,
                                 circular=False)
        g.rewire_edges(rewiringprob)
    elif topologymodel == 'gr':
        radius = get_rgg_params(nvertices, avgdegree)
        g = igraph.Graph.GRG(nvertices, radius)

    g = g.clusters().giant()

    if topologymodel in ['gr', 'wx']:
        aux = np.array([ [g.vs['x'][i], g.vs['y'][i]] for i in range(g.vcount()) ])
        # layoutmodel = 'grid'
    else:
        if topologymodel in ['la', 'ws']:
            layoutmodel = 'grid'
        else:
            layoutmodel = 'random'
        aux = np.array(g.layout(layoutmodel).coords)
    # coords = (aux - np.mean(aux, 0))/np.std(aux, 0) # standardization
    coords = -1 + 2*(aux - np.min(aux, 0))/(np.max(aux, 0)-np.min(aux, 0)) # minmax
    return g, coords

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
    models = ['ER'] # ['ER', 'BA']
    nvertices = 100
    nedges = 200
    nresources = nvertices - nnuclei # To reproduce Luc's res., use nvert-nnuclei
    cs = [.25] # np.arange(0, 1.01, .2)
    rewiringprob = 0.5
    avgdegree = 10

    for model in models:
        g = generate_graph(model, nvertices, avgdegree, rewiringprob)
        g = add_resources(g, nresources)
        for c in cs:
            nnuclei = int(c * nvertices)
            for i in range(niter):
                gnew = add_nuclei(g, nnuclei)
                r = neighnvert / nvertices
                s = neighnvert / neighnedges

    info('For Aiur!')

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
