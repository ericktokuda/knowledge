#!/usr/bin/env python3
"""Get number of vertices and average degree of the largest component
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
from myutils import info, create_readme, graph

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    graphmldir = './graphml/'
    files = os.listdir(graphmldir)
    for filename in files:
        if not filename.endswith('.graphml'): continue
        f = pjoin(graphmldir, filename)
        g = graph.simplify_graphml(f, directed=False)
        print(f, g.vcount(), np.mean(g.degree()))
    info('For Aiur!')

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
