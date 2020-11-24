#!/usr/bin/env python3
"""Analyze simulation results """

import argparse
import time
import os
from os.path import join as pjoin
import inspect

import sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from myutils import info, create_readme
import src.main

########################################################## KDE
def create_meshgrid(x, y, nx=100, ny=100, relmargin=.1):
    """Create a meshgrid around @x and @y with @nx, @ny tiles and relative
    margins @relmargins"""

    # marginx = (max(x) - min(x)) * relmargin
    # marginy = (max(y) - min(y)) * relmargin
    # xrange = [np.min(x) - marginx, np.max(x) + marginx]
    # yrange = [np.min(y) - marginy - .15, np.max(y) + marginy]
    # dx = (xrange[1] - xrange[0]) / nx
    # dy = (yrange[1] - yrange[0]) / ny
    xx, yy = np.mgrid[0:1:0.05, 0:1:0.05]
    return xx, yy

#############################################################
def plot_surface(f, x, y, xx, yy, outdir):
    info(inspect.stack()[0][3] + '()')
    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1,
            cmap='coolwarm', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('PDF')
    ax.set_title('Surface plot of Gaussian 2D KDE')
    fig.colorbar(surf, shrink=0.5, aspect=5) # add color bar indicating the PDF
    ax.view_init(60, 35)
    plt.savefig(pjoin(outdir, 'surfaceplot.pdf'))

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--res', required=True, help='Results (csv) path')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    nvertices = [100, 500, 1000]
    models = ['ba', 'er', 'gr']
    nucleiprefs = [src.main.UNIFORM, src.main.DEGREE]
    ks = list(np.arange(4, 21))
    cs = list(np.arange(0, 1.0, 0.05))
    niters = list(np.arange(0, 100,1))

    def poly2(x, a, b, c): return a*x*x + b*x + c
    def poly3(x, a, b, c, d): return a*x*x*x + b*x*x + c*x + d
    func = poly3

    df = pd.read_csv(args.res)

    data = []
    for nucleipref in nucleiprefs:
        for model in models:
            for n in nvertices:
                for k in ks:
                    dataiters = []
                    for i in niters:
                        aux = df.loc[(df.nucleipref == nucleipref) & \
                                      (df.model == model) & \
                                      (df.nvertices == n) & \
                                      (df.k == k) &(df.i == i)]
                        rs = aux.r.to_numpy()
                        idxmax = np.argmax(rs)
                        cmax = aux.c.iloc[idxmax]
                        xs = aux.c.to_numpy()[:idxmax + 1]
                        ys = aux.r.to_numpy()[:idxmax + 1]

                        if len(xs) < 4: continue # Insufficient sample for curve_fit

                        p0 = [6, -7, 3, 0]
                        params, _ = curve_fit(func, xs, ys, p0=p0)

                        # Plot
                        xs2 = np.linspace(np.min(xs), np.max(xs), 100)
                        ys2 = func(xs2, *params)
                        plt.scatter(xs, ys, c='blue')
                        plt.xlabel('c')
                        plt.ylabel('r')
                        plt.plot(xs2, ys2, c='red')
                        plt.ylim(0, 1)
                        plt.xlim(0, 1)
                        plt.savefig('/tmp/{:03d}.png'.format(i))
                        plt.close()

                        dataiters.append([cmax, *params])

                    means = np.array(dataiters).mean(axis=0)
                    stds = np.array(dataiters).std(axis=0)

                    data.append([nucleipref,model, n, k, *means, *stds])

    dffinal = pd.DataFrame(data)
    dffinal.to_csv(pjoin(args.outdir, 'out.csv'))

    # xx, yy = np.mgrid[nvertices, 0:1:0.05]
    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
