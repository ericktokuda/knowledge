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
from myutils import info, create_readme, append_to_file
import pandas as pd
from multiprocessing import Pool
from itertools import product

#############################################################
NONE = 0
NUCLEUS = 1
RESOURCE = 2

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
        m = int(round(avgdegree/2))
        if m == 0: m = 1
        g = igraph.Graph.Barabasi(nvertices, m)
    elif model == 'gr':
        radius = get_rgg_params(nvertices, avgdegree)
        g = igraph.Graph.GRG(nvertices, radius)
    elif model == 'gm':
        from myutils import graph
        g = graph.simplify_graphml('graph.graphml', directed=False)
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
    # g['coords'] = coords
    return g

##########################################################
def weighted_random_sampling(items, weights, return_idx=False):
    n = len(items)
    cumsum = np.cumsum(weights)
    cumsumnorm = cumsum / cumsum[-1]
    x = np.random.rand()

    for i in range(n):
        if x < cumsumnorm[i]:
            if return_idx: return i
            else: return items[i]

    info('Error when sampling items:{}, weights:{}, x:{:.02f}'. \
         format(items, weights, x))

    if return_idx: return -1
    else: return items[-1]

##########################################################
def weighted_random_sampling_n(items, weights, n):
    """Sample @n items without reposition"""
    item = items.copy(); weights = weights.copy()
    sample = np.zeros(n, dtype=int)
    inds = list(range(len(items)))

    for i in range(n):
        sampleidx = weighted_random_sampling(items, weights, return_idx=True)
        sample[i] = items[sampleidx]
        items = np.delete(items, sampleidx)
        weights = np.delete(weights, sampleidx)

    return sample

##########################################################
def calculate_modified_clucoeff(g):
    """Clustering coefficient as proposed by Luc"""
    # info(inspect.stack()[0][3] + '()')
    adj = np.array(g.get_adjacency().data)

    mult = np.matmul(adj, adj)

    for i in range(mult.shape[0]): mult[i, i] = 0 # Ignoring reaching self

    clucoeffs = np.zeros(g.vcount(), dtype=float)
    for i in range(g.vcount()):
        neighs1 = np.where(adj[i, :] > 0)[0].tolist()
        neighs2 = np.where(mult[i, :] > 0)[0].tolist()
        neighs = list(set([i] + neighs1 + neighs2))
        n = len(neighs)
        induced = adj[neighs, :][:, neighs]
        m = np.sum(induced) / 2
        if n == 1: continue
        clucoeffs[i] = m / ( n * (n-1) / 2)

    return clucoeffs

##########################################################
def add_labels(g, n, choice, label):
    """Add @nresources to graph @g.
    We randomly sample the vertices and change their labels"""
    # info(inspect.stack()[0][3] + '()')
    types = np.array(g.vs['type'])

    validinds = noneinds = np.where(types == NONE)[0]

    if choice == 'un':
        weights = np.ones(len(noneinds), dtype=float)
    elif choice == 'de':
        degrs = np.array(g.degree())
        weights = degrs[noneinds]
    elif choice == 'be':
        betwvs = np.array(g.betweenness())
        weights = betwvs[noneinds]
    elif choice == 'cl':
        clucoeffs = np.array(g.transitivity_local_undirected())
        valid = np.argwhere(~np.isnan(clucoeffs)).flatten()
        validinds = list(set(valid).intersection(set(noneinds)))
        weights = clucoeffs[validinds]
    elif choice == 'cl2':
        clucoeffs = calculate_modified_clucoeff(g)
        valid = np.argwhere(~np.isnan(clucoeffs)).flatten()
        validinds = list(set(valid).intersection(set(noneinds)))
        weights = clucoeffs[validinds]
    else: info('Invalid choice!')

    if np.sum(weights > 0) < n: # When I request @n gt. weights length
        info('Nnuclei is gt. nconnected nodes!')
        return g, []

    sample = np.random.choice(validinds, size=n, replace=False,
                              p=weights/np.sum(weights))

    for i in range(n):
        g.vs[sample[i]]['type'] = label
    return g, sorted(sample[:n])

##########################################################
def get_rgg_params(nvertices, avgdegree):
    rggcatalog = {
        '625,6': 0.056865545,
        '10000,6': 0.0139,
        '11132,6': 0.0131495,
        '22500,6': 0.00925,
         '1000,6': 0.044389839846333226,
        '1000,20': 0.08276843878986143,
       '1000,100': 0.19425867981373568
    }

    if '{},{}'.format(nvertices, avgdegree) in rggcatalog.keys():
        return rggcatalog['{},{}'.format(nvertices, avgdegree)]

    def f(r):
        g = igraph.Graph.GRG(nvertices, r)
        return np.mean(g.degree()) - avgdegree

    return scipy.optimize.brentq(f, 0.0001, 10000)

##########################################################
def run_experiment(params):
    """Sequential runs, given the model. nvertices, avgdegree, nucleipref,
    niter, seed, decayprams, and outdir"""
    model = params['model']
    nvertices = params['nvertices']
    avgdegree = params['avgdegree']
    nucleipref = params['nucleipref']
    niter = params['niter']
    seed = params['seed']
    decayparam1 = params['decayparam1']
    decayparam2 = params['decayparam2']
    graphoutdir = params['graphoutdir']

    info('{},{},{},{},{},{},{}'.format(model, nvertices, avgdegree, nucleipref,
                                 seed, decayparam1, decayparam2))

    np.random.seed(seed)
    random.seed(seed)

    g = generate_graph(model, nvertices, avgdegree, rewiringprob=.5)
    g = g.components(mode='weak').giant() # the largest component
    lens = np.array(g.shortest_paths())

    # prob. based on topological distance
    probfunc = lambda x: decayparam1 * np.exp(- x * decayparam2)

    ret = []
    for i in range(niter):
        r, newg = run_subexperiment(g, nucleipref, i, probfunc, lens)
        ret.extend(r)
        if graphoutdir:
            f = '_'.join(str(x) for x in [model, nvertices, avgdegree,
                                          nucleipref, seed, i])
            outpath = pjoin(graphoutdir, f + '.graphml')
            newg.write_graphml(outpath)
    return ret

##########################################################
def run_subexperiment(gorig, nucleipref, expid, probfunc, lens):
    """Sample nuclei the graph @gorig with preferential location to @nucleipref.
    @outdir defines if we want to store the graph"""
    # info(inspect.stack()[0][3] + '()')

    nvertices = gorig.vcount()
    ret = []

    ret.append([expid, nvertices, 0.0, 0.0, 1.0]) # c = 0

    g = gorig.copy()
    g.vs['type'] = [RESOURCE] * nvertices
    g.vs[np.random.randint(nvertices)]['type'] = NUCLEUS
    maxnuclei = int(.95 * nvertices)

    nuclids = np.where(np.array(g.vs['type']) == NUCLEUS)[0]
    resoids = np.where(np.array(g.vs['type']) == RESOURCE)[0]

    degrees = np.array(g.degree())
    betvs = np.array(g.betweenness())
    eps = .01
    betvs[np.where(betvs == 0)] = eps
    nuclei = - np.ones(nvertices, dtype=int)
    nuclei[0] = nuclids[0]

    maxntries = 1000

    if nucleipref == 'dila':
        dists = g.shortest_paths(nuclids)[0]
        newreso = np.argsort(dists)[1:] # Excluding self

    for nucleusidx in range(1, maxnuclei): #1st stop condition
        if nucleipref == 'betv':
            probs = betvs[resoids]/np.sum(betvs[resoids])
            newnode = np.random.choice(resoids, p=probs)
        elif nucleipref == 'degr':
            probs = degrees[resoids]/np.sum(degrees[resoids])
            newnode = np.random.choice(resoids, p=probs)
        elif nucleipref == 'dist':
            for j in range(maxntries):
                newnode = np.random.choice(resoids)
                mindist = np.min(lens[newnode][nuclids])
                if np.random.rand() < probfunc(mindist): break
        elif nucleipref == 'dila':
            newnode = newreso[nucleusidx]
        elif nucleipref == 'unif':
            newnode = np.random.choice(resoids)

        g.vs[newnode]['type'] = NUCLEUS
        nuclids = np.where(np.array(g.vs['type']) == NUCLEUS)[0]
        resoids = np.where(np.array(g.vs['type']) == RESOURCE)[0]

        neighs = []
        for nucl in nuclids:
            neighids = np.array(g.neighbors(nucl))
            neightypes = np.array(g.vs[neighids.tolist()]['type'])
            neighresoids = np.where(neightypes == RESOURCE)[0]
            neighs.extend(neighids[neighresoids])

        lenunique = len(set(neighs))
        lenrepeated = len(neighs)

        r = lenunique / nvertices
        s = lenunique / lenrepeated if lenunique > 0 else 0
        c = len(nuclids) / nvertices
        ret.append([expid, nvertices, c, r, s])

        nuclei[nucleusidx] = newnode

    nuclei = nuclei[:np.where(nuclei == -1)[0][0]]
    g.vs['nucleiorder'] = -1
    for j, v in enumerate(nuclei):
        g.vs[v]['nucleiorder'] = j

    ret.append([expid, nvertices, 0.0, 0.0, 0.0])
    return ret, g

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nprocs', default=1, type=int,
                        help='Number of parallel processes')
    parser.add_argument('--storegraphs', action='store_true',
                        help='Whether store the graphs')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    models = ['ba', 'er', 'gr'] # ['ba', 'er', 'gm', 'gr']
    nvertices = [100, 300, 500, 700] # [100, 300, 500, 700]
    avgdegrees = [6, 12, 18, 24] # np.arange(4, 21)
    nucleiprefs = ['betv', 'degr', 'dila', 'dist', 'unif'] # ['betv', 'degr', 'dila', 'dist', 'unif']
    niter = 40 # 40
    nseeds = 50 # 50
    decayparam1 = 1
    decayparam2 = 0.5


    append_to_file(readmepath, 'models:{}, nvertices:{}, avgdegrees:{},' \
                   'nucleiprefs:{}, niter:{}, nseeds:{},' \
                   'decayparam1:{}, decayparam2:{}, storegraphs:{}' \
                   .format(models, nvertices, avgdegrees, nucleiprefs,
                           niter, nseeds, decayparam1,
                           decayparam2, args.storegraphs))

    graphoutdir = args.outdir if args.storegraphs else ''
    aux = list(product(models, nvertices, avgdegrees, nucleiprefs,
                       [niter], list(range(nseeds)), [decayparam1],
                       [decayparam2], [graphoutdir])) # Fill here
    params = []
    for i, row in enumerate(aux):
        params.append(dict(model = row[0],
                           nvertices = row[1],
                           avgdegree = row[2],
                           nucleipref = row[3],
                           niter = row[4],
                           seed = row[5],
                           decayparam1 = row[6],
                           decayparam2 = row[7],
                           graphoutdir = row[8],
                           ))

    if args.nprocs == 1:
        info('Running serially (nprocs:{})'.format(args.nprocs))
        ret = [run_experiment(p) for p in params]
    else:
        info('Running in parallel (nprocs:{})'.format(args.nprocs))
        pool = Pool(args.nprocs)
        ret = pool.map(run_experiment, params)

    res = []
    for p, r in zip(params, ret):
        beg = [p['model'], p['nvertices'], p['avgdegree'],
               p['nucleipref'], p['seed']]
        for rr in r: res.append(beg + rr)

    df = pd.DataFrame()
    cols = ['model', 'nverticesfull', 'avgdegree', 'nucleipref', 'seed',
            'i', 'nverticescomp', 'c', 'r', 's']

    for i, col in enumerate(cols):
        df[col] = [x[i] for x in res]

    respath = pjoin(args.outdir, 'results.csv')
    df.to_csv(respath, index=False)
    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
