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
def POLY2(x, a, b, c, d): return a*x*x + b*x + c
def POLY3(x, a, b, c, d): return a*x*x*x + b*x*x + c*x + d
def MYEXP(x, a, b): return a*(np.exp(b*x) - 1) # Force it to have the (0, 0)
FUNC = MYEXP

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
    ks = np.unique(df.avgdegree)
    seeds = np.unique(df.seed)
    cs = np.unique(df.c)
    data = []

    for nucleipref in nucleiprefs:
        df1 = df.loc[df.nucleipref == nucleipref]
        for model in models:
            df2 = df1.loc[df1.model == model]
            for n in nvertices:
                df3 = df2.loc[df2.nvertices == n]
                for k in ks:
                    df4 = df3.loc[df3.avgdegree == k]
                    for seed in seeds:
                        df5 = df4.loc[df4.seed == seed]
                        for c in cs:
                            df6 = df5.loc[df5.c == c]
                            r = df6.r.mean(), df6.r.std()
                            s = df6.s.mean(), df6.s.std()
                            data.append([nucleipref, model, n, k, seed, c, *r, *s])

    cols = 'nucleipref,model,nvertices,avgdegree,seed,c,' \
        'rmean,rstd,smean,sstd'.split(',')
    dffinal = pd.DataFrame(data, columns=cols)
    dffinal.to_csv(parsedpath, index=False)
    return dffinal

##########################################################
def find_coeffs(df, outdir):
    """Parse results from simulation"""
    info(inspect.stack()[0][3] + '()')

    aggregpath = pjoin(outdir, 'aggregated.csv')

    if os.path.exists(aggregpath):
        info('Loading existing aggregated results:{}'.format(aggregpath))
        return pd.read_csv(aggregpath)

    outdir = pjoin(args.outdir, 'fits')

    models = np.unique(df.model)
    nvertices = np.unique(df.nvertices)
    nucleiprefs = np.unique(df.nucleipref)
    ks = np.unique(df.avgdegree)
    seeds = np.unique(df.seed)

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
                    for seed in seeds:
                        df5 = df4.loc[df4.seed == seed]

                        idxmax = np.argmax(df5.rmean.to_numpy())
                        aux = df5.iloc[:idxmax + 1]

                        rs = aux.rmean.to_numpy()
                        rmax = np.max(aux.rmean.to_numpy())
                        idxmax = np.argmax(rs)
                        cmax = aux.c.iloc[idxmax]

                        xs = aux.c.to_numpy()[:idxmax + 1]
                        ys = aux.rmean.to_numpy()[:idxmax + 1]

                        if len(xs) < 3:
                            continue # Insufficient sample for curve_fit

                        # p2:[-10,4,0,0], p3:[6,-7,3,0], myexp:[-.5, -6]
                        p0 = [-.5, -6] # exp
                        params, _ = curve_fit(FUNC, xs, ys, p0=p0, maxfev=10000)

                        outpath = pjoin(outdir, '{}_{}_{}_{}_{}.png'.format(
                            nucleipref, model, n, k, seed))
                        scatter_c_vs_r(xs, ys, outpath, func=func, params=params)

                        data.append([nucleipref, model, n, k, seed, cmax, rmax, *params])

    cols = 'nucleipref,model,nvertices,avgdegree,seed,cmax,rmax,a,b'.split(',')
    dffinal = pd.DataFrame(data, columns=cols)
    dffinal.to_csv(aggregpath, index=False)
    return dffinal

##########################################################
def plot_fits(df, outdir):
    """Plot fits"""
    info(inspect.stack()[0][3] + '()')

    models = np.unique(df.model)
    nvertices = np.unique(df.nvertices)
    nucleiprefs = np.unique(df.nucleipref)
    ks = np.unique(df.avgdegree)
    seeds = np.unique(df.seed)

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
                    scatter_c_vs_r(xs, ys, outpath, func=func, params=params)
                    for seed in seeds:
                        df5 = df4.loc[df4.seed == seed]
                        outpath = pjoin(outdir, '{}_{}_{}_{}_{}.png'.format(
                            nucleipref, model, n, k, seed))
                        scatter_c_vs_r(xs, ys, outpath, func=func, params=params)

                        # data.append([nucleipref, model, n, k, seed, cmax, rmax, *params])

    cols = 'nucleipref,model,nvertices,avgdegree,seed,cmax,rmax,a,b'.split(',')
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
    os.makedirs(outdir, exist_ok=True)
    models = np.unique(df.model)
    nvertices = np.unique(df.nvertices)
    nucleiprefs = np.unique(df.nucleipref)
    ks = np.unique(df.avgdegree)
    iters = np.unique(df.i)

    for nucleipref in nucleiprefs:
        for model in models:
            for n in nvertices:
                for k in ks:
                    for i in iters:

                        aux = df.loc[(df.nucleipref == nucleipref) & \
                                      (df.model == model) & \
                                      (df.nvertices == n) & \
                                      (df.avgdegree == k) &(df.i == i)]
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

    # nucleipref = 'de'
    # filtered = df.loc[(df.nucleipref == nucleipref)]
    params = ['cmax', 'a', 'b', 'rmax']


    for nucleipref in nucleiprefs:
        filtered = df.loc[(df.nucleipref == nucleipref)]
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
def plot_slices(df, outdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    os.makedirs(outdir, exist_ok=True)
    # xx, yy = np.mgrid[nvertices, 0:1:0.05]
    models = np.unique(df.model)
    nucleiprefs = np.unique(df.nucleipref)

    # nucleipref = 'de'

    params = ['cmax', 'a', 'b', 'rmax']
    

    breakpoint()
    
    for nucleipref in nucleiprefs:
        filtered = df.loc[(df.nucleipref == nucleipref)]
        for param in params:
            f, ax = plt.subplots()
            for model in models:
                # z = filtered.loc[(filtered.model == model) & (filtered.nvertices == 500)][param].to_numpy()
                z = filtered.loc[(filtered.model == model) & (filtered.avgdegree == 1)][param].to_numpy()
                ax.plot(np.unique(df.nvertices), z, label=model)

            # ax.set_xlabel('avgdegree')
            ax.set_xlabel('nvertices')
            ax.set_ylabel(param)
            plt.legend()
            plt.savefig(pjoin(outdir, '{}_{}_{}.png'.format(nucleipref,
                                                            param, model)))
            plt.close()

##########################################################
def plot_parameters_pairwise(df, outdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    os.makedirs(outdir, exist_ok=True)
    models = np.unique(df.model)
    avgdegrees = np.unique(df.avgdegree)
    nucleiprefs = np.unique(df.nucleipref)

    L = 16
    combs = [['b', 'rmax'], ['cmax', 'rmax'], ['b', 'cmax']]

    markers = ['o', 's']
    colours = ['green', 'darkorange', 'blue']

    dforig = df.copy()

    for pair in combs:
        param1, param2 = pair
        f, ax = plt.subplots(figsize=(L, L))
        for i, nucleipref in enumerate(nucleiprefs):
            df = dforig[dforig.nucleipref == nucleipref]
            marker = markers[i]
            for j, model in enumerate(models):
                colour = colours[j]
                plt.scatter(np.abs(df[df.model == model][param1]),
                            df[df.model == model][param2],
                            label=model, s=df[df.model == model].nvertices/4,
                            marker=marker, c=colour)
        plt.legend()
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        plt.legend()
        plt.savefig(pjoin(outdir, '{}_{}.png'.format(param1, param2)))
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
    """Plot r and s means for each city"""
    info(inspect.stack()[0][3] + '()')

    os.makedirs(outdir, exist_ok=True)

    W = 640*2; H = 480
    models = np.unique(df.model)
    nvertices = np.unique(df.nvertices)
    nucleiprefs = np.unique(df.nucleipref)
    ks = np.unique(df.avgdegree)
    seeds = np.unique(df.seed)
    cs = np.unique(df.c)

    for nucleipref in nucleiprefs:
        df1 = df.loc[df.nucleipref == nucleipref]
        for model in models:
            df2 = df1.loc[df1.model == model]
            for n in nvertices:
                df3 = df2.loc[df2.nvertices == n]
                for k in ks:
                    df4 = df3.loc[df3.avgdegree == k]
                    for seed in seeds:
                        df5 = df4.loc[df4.seed == seed]
                        
                        fig, ax = plt.subplots(1, 2, figsize=(W*.01, H*.01), dpi=100)
                        ax[0].errorbar(cs, df5.rmean, yerr=df5.rstd)
                        ax[1].errorbar(cs, df5.smean, yerr=df5.sstd)
                        ax[0].set_xlabel('c')
                        ax[0].set_ylabel('r')
                        ax[0].set_xlim(0, 1)
                        ax[0].set_ylim(0, 1)
                        ax[1].set_xlabel('c')
                        ax[1].set_ylabel('s')
                        ax[1].set_xlim(0, 1)
                        ax[1].set_ylim(0, 1)
                        outpath = pjoin(outdir, '{}_{}_{}_{}_{}.png'.format(
                            nucleipref, model, n, k, seed))
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

    # plot_origpoints(df, pjoin(args.outdir, 'origpoints'))
    dfparsed = parse_results(df, args.outdir)
    # plot_means(dfparsed, pjoin(args.outdir, 'plots_r_s'))
    
    dfcoeffs = find_coeffs(dfparsed, args.outdir)
    # plot_fits(dfcoeffs, pjoin(args.outdir, 'fits'))
    plot_parameters_pairwise(dfcoeffs, pjoin(args.outdir, 'params'))
    plot_slices(dfcoeffs, pjoin(args.outdir, 'slices'))
    return 

    # For multiple avgdegrees and nvertices
    plot_contours(dfcoeffs, pjoin(args.outdir, 'contours'))
    plot_triangulations(dfcoeffs, pjoin(args.outdir, 'surface_tri'))

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
