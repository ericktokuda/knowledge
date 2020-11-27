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
# import matplotlib; matplotlib.use('Agg')
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from myutils import info, create_readme
from myutils import plot as myplot

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
def aggregate_results(df, outdir):
    """Parse results from simulation"""

    aggregpath = pjoin(outdir, 'aggregated.csv')

    if os.path.exists(aggregpath):
        info('Loading existing aggregated results:{}'.format(aggregpath))
        return pd.read_csv(aggregpath)

    models = np.unique(df.model)
    nvertices = np.unique(df.nvertices)
    nucleiprefs = np.unique(df.nucleipref)
    ks = np.unique(df.k)
    niters = np.unique(df.i).astype(int)

    def poly2(x, a, b, c, d): return a*x*x + b*x + c
    def poly3(x, a, b, c, d): return a*x*x*x + b*x*x + c*x + d
    def myexp(x, a, b): return a*(np.exp(b*x) - 1) # Force it to have the (0, 0)
    func = myexp

    data = []
    for nucleipref in nucleiprefs:
        for model in models:
            info('model:{}'.format(model))
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

                        # p2:[-10,4,0,0], p3:[6,-7,3,0], myexp:[-.5, -6]
                        p0 = [-.5, -6] # exp
                        params, _ = curve_fit(func, xs, ys, p0=p0, maxfev=10000)

                        outpath = pjoin(outdir, '{:03d}.png'.format(i))
                        # plot_cxr(xs, ys, outpath, func=func, params=params)
                        dataiters.append([cmax, *params])

                    means = np.array(dataiters).mean(axis=0)
                    stds = np.array(dataiters).std(axis=0)

                    data.append([nucleipref, model, n, k, *means, *stds])

    cols = 'nucleipref,model,nvertices,avgdegree,cmaxmean,' \
        'amean,bmean,cmaxstd,astd,bstd'.split(',')
    dffinal = pd.DataFrame(data, columns=cols)
    dffinal.to_csv(aggregpath, index=False)
    return dffinal

##########################################################
def plot_cxr(cs, rs, outpath, func=None, params=None):
    """Plot C x R"""
    # info(inspect.stack()[0][3] + '()')
    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    ax.scatter(cs, rs)
    if func != None:
        xs = np.linspace(np.min(cs), np.max(cs), 100)
        ys = func(xs, *params)
        ax.plot(xs, ys, c='red')
    ax.set_xlabel('c')
    ax.set_ylabel('r')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.savefig(outpath)
    plt.close()

##########################################################
def plot_origpoints(df, outdir):
    """Plot original points """
    info(inspect.stack()[0][3] + '()')
    models = np.unique(df.model)
    nvertices = np.unique(df.nvertices)
    nucleiprefs = np.unique(df.nucleipref)
    ks = np.unique(df.k)
    niters = np.unique(df.i)

    for nucleipref in nucleiprefs:
        for model in models:
            for n in nvertices:
                for k in ks:
                    for i in niters:

                        aux = df.loc[(df.nucleipref == nucleipref) & \
                                      (df.model == model) & \
                                      (df.nvertices == n) & \
                                      (df.k == k) &(df.i == i)]
                        rs = aux.r.to_numpy()
                        cs = aux.c.to_numpy()
                        f = '{}_{}_{}_{}_{:02d}.png'.format(
                            nucleipref, model, n, k, i)
                        outpath = pjoin(outdir, f)
                        plot_cxr(cs, rs, outpath)

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

    df = pd.read_csv(args.res)
    # plot_origpoints(df, args.outdir)
    dfaggreg = aggregate_results(df, args.outdir)

    # xx, yy = np.mgrid[nvertices, 0:1:0.05]
    from mpl_toolkits.mplot3d import Axes3D
    filtered = dfaggreg.loc[(dfaggreg.nucleipref == 'un') & \
                            (dfaggreg.model == 'er') ]
    x = filtered.nvertices.to_numpy()
    y = filtered.avgdegree.to_numpy()
    z = filtered.cmaxmean.to_numpy()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('nvertices')
    ax.set_ylabel('avgdegree')
    surf = ax.plot_trisurf(x, y, z, color=(0, 0, 0, 0),
                           edgecolor='black')
    # plt.show()
    plt.savefig(pjoin(args.outdir, 'wireframe.png'))

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
