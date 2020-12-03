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
from mpl_toolkits.mplot3d import Axes3D

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
def parse_results(df, outdir):
    """Parse results from simulation"""
    info(inspect.stack()[0][3] + '()')

    parsedpath = pjoin(outdir, 'parsed.csv')

    if os.path.exists(parsedpath):
        info('Loading existing aggregated results:{}'.format(parsedpath))
        return pd.read_csv(parsedpath)

    models = np.unique(df.model)
    nvertices = np.unique(df.nvertices)
    nucleiprefs = np.unique(df.nucleipref)
    ks = np.unique(df.k)
    niters = np.unique(df.i)
    cs = np.unique(df.c)

    data = []

    for nucleipref in nucleiprefs:
        df1 = df.loc[df.nucleipref == nucleipref]
        for model in models:
            df2 = df1.loc[df1.model == model]
            for n in nvertices:
                df3 = df2.loc[df2.nvertices == n]
                for k in ks:
                    df4 = df3.loc[df3.k == k]
                    for c in cs:
                        df5 = df4.loc[df4.c == c]
                        r = df5.r.mean(), df5.r.std()
                        s = df5.s.mean(), df5.s.std()
                        data.append([nucleipref, model, n, k, c, *r, *s])

    cols = 'nucleipref,model,nvertices,avgdegree,c,' \
        'rmean,rstd,smean,sstd'.split(',')
    dffinal = pd.DataFrame(data, columns=cols)
    dffinal.to_csv(parsedpath, index=False)
    return dffinal

##########################################################
def find_coeffs(df, outdir):
    """Parse results from simulation"""
    info(inspect.stack()[0][3] + '()')

    os.makedirs(outdir, exist_ok=True)
    aggregpath = pjoin(outdir, 'aggregated.csv')

    if os.path.exists(aggregpath):
        info('Loading existing aggregated results:{}'.format(aggregpath))
        return pd.read_csv(aggregpath)

    models = np.unique(df.model)
    nvertices = np.unique(df.nvertices)
    nucleiprefs = np.unique(df.nucleipref)
    ks = np.unique(df.avgdegree)

    def poly2(x, a, b, c, d): return a*x*x + b*x + c
    def poly3(x, a, b, c, d): return a*x*x*x + b*x*x + c*x + d
    def myexp(x, a, b): return a*(np.exp(b*x) - 1) # Force it to have the (0, 0)
    func = myexp

    data = []
    for nucleipref in nucleiprefs:
        df1 = df.loc[df.nucleipref == nucleipref]
        for model in models:
            info('model:{}'.format(model))
            df2 = df1.loc[df1.model == model]
            for n in nvertices:
                df3 = df2.loc[df2.nvertices == n]
                for k in ks:
                    df4 = df3.loc[df3.avgdegree == k]

                    idxmax = np.argmax(df4.rmean.to_numpy())
                    aux = df4.iloc[:idxmax + 1]

                    rs = aux.rmean.to_numpy()
                    rmax = np.max(aux.rmean.to_numpy())
                    idxmax = np.argmax(rs)
                    cmax = aux.c.iloc[idxmax]

                    xs = aux.c.to_numpy()[:idxmax + 1]
                    ys = aux.rmean.to_numpy()[:idxmax + 1]

                    if len(xs) < 4: continue # Insufficient sample for curve_fit

                    # p2:[-10,4,0,0], p3:[6,-7,3,0], myexp:[-.5, -6]
                    p0 = [-.5, -6] # exp
                    params, _ = curve_fit(func, xs, ys, p0=p0, maxfev=10000)

                    outpath = pjoin(outdir, '{}_{}_{}_{}.png'.format(
                        nucleipref, model, n, k))
                    scatter_c_vs_r(xs, ys, outpath, func=func, params=params)

                    data.append([nucleipref, model, n, k, cmax, rmax, *params])

    cols = 'nucleipref,model,nvertices,avgdegree,cmax,rmax,a,b'.split(',')
    dffinal = pd.DataFrame(data, columns=cols)
    dffinal.to_csv(aggregpath, index=False)
    return dffinal

##########################################################
def scatter_c_vs_r(cs, rs, outpath, func=None, params=None):
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
                        scatter_c_vs_r(cs, rs, outpath)


##########################################################
def plot_contours(df, outdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    os.makedirs(outdir, exist_ok=True)
    # xx, yy = np.mgrid[nvertices, 0:1:0.05]
    models = np.unique(df.model)
    nucleiprefs = np.unique(df.nucleipref)

    nucleipref = 'de'
    filtered = df.loc[(df.nucleipref == nucleipref)]

    params = ['cmax', 'a', 'b', 'rmax']

    for nucleipref in nucleiprefs:
        for model in models:
            x = filtered.loc[(filtered.model == model)].nvertices.to_numpy()
            y = filtered.loc[(filtered.model == model)].avgdegree.to_numpy()
            for param in params:
                z = filtered.loc[(filtered.model == model)][param].to_numpy()

                f, ax = plt.subplots()
                pp = ax.tricontourf(x, y, z, 20)
                ax.plot(x,y, 'ko ')
                f.colorbar(pp)
                plt.savefig(pjoin(outdir, '{}_{}_{}.png'.format(nucleipref,
                                                                param, model)))
                plt.close()


##########################################################
def plot_triangulations(df, outdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    os.makedirs(outdir, exist_ok=True)
    # xx, yy = np.mgrid[nvertices, 0:1:0.05]
    models = np.unique(df.model)
    nucleiprefs = np.unique(df.nucleipref)

    nucleipref = 'de'
    filtered = df.loc[(df.nucleipref == nucleipref)]

    params = ['cmax', 'a', 'b', 'rmax']

    for nucleipref in nucleiprefs:
        for model in models:
            x = filtered.loc[(filtered.model == model)].nvertices.to_numpy()
            y = filtered.loc[(filtered.model == model)].avgdegree.to_numpy()
            for param in params:
                z = filtered.loc[(filtered.model == model)][param].to_numpy()

                fig = plt.figure()
                ax = Axes3D(fig)
                ax.set_xlabel('nvertices')
                ax.set_ylabel('avgdegree')

                ax.set_zlabel(param)

                # surf = ax.plot_trisurf(x, y, z, color=(0, 0, 0, 0), edgecolor='black')
                surf = ax.plot_trisurf(x, y, z, color=(.2, .2, .2, .8))
                # plt.show()
                plt.savefig(pjoin(outdir, '{}_{}_{}.png'.format(nucleipref,
                                                                param, model)))
                plt.close()
##########################################################
def plot_means(df, outdir):
    """Plot r and s means """
    info(inspect.stack()[0][3] + '()')

    os.makedirs(outdir, exist_ok=True)

    W = 640*2; H = 480
    models = np.unique(df.model)
    nvertices = np.unique(df.nvertices)
    nucleiprefs = np.unique(df.nucleipref)
    ks = np.unique(df.avgdegree)
    cs = np.unique(df.c)

    for nucleipref in nucleiprefs:
        df1 = df.loc[df.nucleipref == nucleipref]
        for model in models:
            df2 = df1.loc[df1.model == model]
            for n in nvertices:
                df3 = df2.loc[df2.nvertices == n]
                for k in ks:
                    df4 = df3.loc[df3.avgdegree == k]
                    fig, ax = plt.subplots(1, 2, figsize=(W*.01, H*.01), dpi=100)
                    ax[0].errorbar(cs, df4.rmean, yerr=df4.rstd)
                    ax[1].errorbar(cs, df4.smean, yerr=df4.sstd)
                    ax[0].set_xlabel('c')
                    ax[0].set_ylabel('r')
                    ax[0].set_xlim(0, 1)
                    ax[0].set_ylim(0, 1)
                    ax[1].set_xlabel('c')
                    ax[1].set_ylabel('s')
                    ax[1].set_xlim(0, 1)
                    ax[1].set_ylim(0, 1)
                    outpath = pjoin(outdir, '{}_{}_{}_{}.png'.format(
                        nucleipref, model, n, k))
                    plt.savefig(outpath)
                    plt.close()

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
    dfparsed = parse_results(df, args.outdir)
    # plot_means(dfparsed, pjoin(args.outdir, 'plots_r_s'))
    dfcoeffs = find_coeffs(dfparsed, pjoin(args.outdir, 'fits'))
    plot_contours(dfcoeffs, pjoin(args.outdir, 'contours'))
    plot_triangulations(dfcoeffs, pjoin(args.outdir, 'surface_tri'))

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
