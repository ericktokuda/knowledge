#!/usr/bin/env python3
"""Plot graphs generated in the simulation with the option --storegraphs
"""

import argparse
import time
import os
from os.path import join as pjoin
import inspect

import sys, random
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import igraph
from myutils import info, create_readme

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphmldir', required=True, help='Graphml directory')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    files = os.listdir(args.graphmldir)
    for f in files:
        if not f.endswith('.graphml'): continue
        np.random.seed(0); random.seed(0) # same visual
        g = igraph.Graph.Read(pjoin(args.graphmldir, f))

        n = g.vcount()
        colours = np.array([[0, 0, 0, 0.5]] * n)

        for i, c in enumerate(g.vs['nucleiorder']):
            if c < 0: continue
            info('c:{}, n:{}'.format(c, n))
            colours[i, 1] = c / n

        outpath = pjoin(args.outdir, f.replace('.graphml', '.png'))
        labels = ['{:03d}'.format(int(x)) for x in g.vs['nucleiorder']]
        colours = np.array([[1, 1, 1, .5]] * n)
        indv0 = np.where(np.array(g.vs['nucleiorder']) == 0)
        colours[indv0, :] = [0, 0, 1, .5]
        igraph.plot(g, outpath, vertex_label=labels, vertex_color=colours.tolist())

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
