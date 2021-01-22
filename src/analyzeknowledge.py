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
def get_unique_vals(df):
    un = {}
    for c in df.columns:
        un[c] = np.unique(df[c])
    return un

##########################################################
def plot_origpoints(df, un, outdir, sample):
    """Plot original points """
    info(inspect.stack()[0][3] + '()')
    os.makedirs(outdir, exist_ok=True)

    for nucleipref in un['nucleipref']:
        df1 = df.loc[df.nucleipref == nucleipref]
        for model in un['model']:
            df2 = df1.loc[df1.model == model]
            for n in un['nverticescomp']:
                df3 = df2.loc[df2.nverticescomp == n]
                for k in un['avgdegree']:
                    df4 = df3.loc[df3.avgdegree == k]
                    for seed in un['seed'][:sample]:
                        df5 = df4.loc[df4.seed == seed]
                        for i in un['i'][:sample]:
                            aux = df5.loc[df5.i == i]
                            rs = aux.r.to_numpy()
                            cs = aux.c.to_numpy()
                            f = '{}_{}_{}_{}_{:02d}_{:02d}.png'.format(
                                nucleipref, model, n, k, seed, i)
                            outpath = pjoin(outdir, f)
                            scatter_c_vs_r(cs, rs, outpath)

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
def parse_results(df, un, outdir):
    """Parse results from simulation"""
    info(inspect.stack()[0][3] + '()')

    parsedpath = pjoin(outdir, 'parsed.csv')

    if os.path.exists(parsedpath):
        info('Loading existing aggregated results:{}'.format(parsedpath))
        return pd.read_csv(parsedpath)

    data = []

    for nucleipref in un['nucleipref']:
        df1 = df.loc[df.nucleipref == nucleipref]
        for model in un['model']:
            df2 = df1.loc[df1.model == model]
            for n in un['nverticescomp']:
                df3 = df2.loc[df2.nverticescomp == n]
                for k in un['avgdegree']:
                    df4 = df3.loc[df3.avgdegree == k]
                    for seed in un['seed']:
                        df5 = df4.loc[df4.seed == seed]
                        for c in un['c']:
                            df6 = df5.loc[df5.c == c]
                            r = df6.r.mean(), df6.r.std()
                            s = df6.s.mean(), df6.s.std()
                            data.append([nucleipref, model, n, k, seed, c,
                                *r, *s])

    cols = 'nucleipref,model,nverticescomp,avgdegree,seed,c,' \
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

    outdir = pjoin(outdir, 'fits')
    os.makedirs(outdir, exist_ok=True)

    data = []
    for nucleipref in np.unique(df.nucleipref):
        df1 = df.loc[df.nucleipref == nucleipref]
        for model in np.unique(df1.model):
            info('nucleipref:{}, model:{}'.format(nucleipref, model))
            df2 = df1.loc[df1.model == model]
            for n in np.unique(df2.nverticesfull):
                df3 = df2.loc[df2.nverticesfull == n]
                for k in np.unique(df3.avgdegree):
                    df4 = df3.loc[df3.avgdegree == k]
                    for seed in np.unique(df4.seed):
                        df5 = df4.loc[df4.seed == seed]
                        grouped = df5.groupby(['c'])
                        rs = grouped.mean().r.to_numpy()
                        idxmax = np.argmax(rs)

                        rs = rs[:idxmax + 1]
                        ss = grouped.mean().s.to_numpy()[:idxmax + 1]
                        cs = grouped.mean().index.to_numpy()[:idxmax +1]

                        rmax = rs[-1]
                        cmax = cs[-1]

                        xs = cs; ys = rs

                        if len(xs) < 3:
                            continue # Insufficient sample for curve_fit

                        # p2:[-10,4,0,0], p3:[6,-7,3,0], myexp:[-.5, -6]
                        p0 = [-.5, -6] # exp
                        params, _ = curve_fit(FUNC, xs, ys, p0=p0, maxfev=10000)

                        outpath = pjoin(outdir, '{}_{}_{}_{}_{}.png'.format(
                            nucleipref, model, n, k, seed))
                        scatter_c_vs_r(xs, ys, outpath, func=FUNC, params=params)

                        ncomp = np.unique(df5.nverticescomp)[0] # all the same
                        data.append([nucleipref, model, n, ncomp, k, seed, cmax,
                                     rmax, *params])


    cols = 'nucleipref,model,nverticesfull,nverticescomp,avgdegree,seed,cmax,rmax,a,b'.split(',')
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
def plot_contours(df, outdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    os.makedirs(outdir, exist_ok=True)
    # xx, yy = np.mgrid[nverticescomp, 0:1:0.05]
    models = np.unique(df.model)
    nucleiprefs = np.unique(df.nucleipref)

    # nucleipref = 'de'
    # filtered = df.loc[(df.nucleipref == nucleipref)]
    params = ['cmax', 'a', 'b', 'rmax']


    for nucleipref in nucleiprefs:
        filtered = df.loc[(df.nucleipref == nucleipref)]
        for model in models:
            x = filtered.loc[(filtered.model == model)].nverticescomp.to_numpy()
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
def plot_slice(dforig, fixed, fixedparam, outdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    os.makedirs(outdir, exist_ok=True)

    df = dforig.loc[dforig[fixed] == fixedparam]
    varying = 'avgdegree' if fixed == 'nverticescomp' else 'nverticescomp'

    params = ['cmax', 'a', 'b', 'rmax']

    figscale = 4
    ncols = len(np.unique(df.nucleipref))

    for param in params:
        fig, axs = plt.subplots(1, ncols,
                    figsize=(ncols*figscale, 1*figscale))
        for j, nucleipref in enumerate(np.unique(df.nucleipref)):
            df1 = df.loc[(df.nucleipref == nucleipref)]
            for model in np.unique(df1.model):
                df2 = df1.loc[df1.model == model]
                grouped = df2.groupby(['nverticesfull'])
                xmean = grouped.mean()[varying]
                xstd = grouped.std()[varying]
                ymean = grouped.mean()[param]
                ystd = grouped.std()[param]

                axs[j].errorbar(xmean, ymean, xerr=xstd, yerr=ystd,
                        label=model, alpha=0.8)

            axs[j].set_xlabel(varying)
            axs[j].set_ylabel(param)
            axs[j].set_title(nucleipref)

        plt.legend(loc='upper right')
        plotpath = pjoin(outdir, '{}_{}.png'.format(param, model))
        plt.tight_layout()
        plt.savefig(plotpath)
        plt.close()

##########################################################
def plot_parameters_pairwise(df, un, outdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    os.makedirs(outdir, exist_ok=True)

    L = 8
    combs = [['b', 'rmax'], ['cmax', 'rmax'], ['b', 'cmax']]

    markers = ['o', 's']
    colours = ['green', 'darkorange', 'blue']

    dforig = df.copy()

    for pair in combs:
        param1, param2 = pair
        f, ax = plt.subplots(figsize=(L, L))
        for i, nucleipref in enumerate(un['nucleipref']):
            df = dforig[dforig.nucleipref == nucleipref]
            marker = markers[i]
            for j, model in enumerate(un['model']):
                colour = colours[j]
                plt.scatter(np.abs(df[df.model == model][param1]),
                            df[df.model == model][param2],
                            label='{} {}'.format(nucleipref, model),
                            s=np.log(df[df.model == model].nverticescomp),
                            # label=model, s=300,
                            alpha=.7,
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
    # xx, yy = np.mgrid[nverticescomp, 0:1:0.05]
    models = np.unique(df.model)
    nucleiprefs = np.unique(df.nucleipref)

    nucleipref = 'de'
    filtered = df.loc[(df.nucleipref == nucleipref)]

    params = ['cmax', 'a', 'b', 'rmax']

    for nucleipref in nucleiprefs:
        for model in models:
            x = filtered.loc[(filtered.model == model)].nverticescomp.to_numpy()
            y = filtered.loc[(filtered.model == model)].avgdegree.to_numpy()
            for param in params:
                z = filtered.loc[(filtered.model == model)][param].to_numpy()

                fig = plt.figure()
                ax = Axes3D(fig)
                ax.set_xlabel('nverticescomp')
                ax.set_ylabel('avgdegree')

                ax.set_zlabel(param)

                # surf = ax.plot_trisurf(x, y, z, color=(0, 0, 0, 0), edgecolor='black')
                surf = ax.plot_trisurf(x, y, z, color=(.2, .2, .2, .8))
                # plt.show()
                plt.savefig(pjoin(outdir, '{}_{}_{}.png'.format(nucleipref,
                                                                param, model)))
                plt.close()

##########################################################
def plot_r_s(dforig, un, outdir, sample):
    """Plot r and s means for each city"""
    info(inspect.stack()[0][3] + '()')

    os.makedirs(outdir, exist_ok=True)

    W = 640*2; H = 480

    df = dforig.copy()
    del df['nverticesfull']
    aux = df.groupby(['model', 'avgdegree', 'nucleipref',
                      'seed', 'nverticescomp', 'c'])
    means = aux.mean()
    stds = aux.std()
    allrows = np.array(list(means.index)).astype(str)

    # assuming the order of the indices are: (model, k, nucleipref, seed, n, c)
    for model in np.unique(allrows[:, 0]):
        rows1 = allrows[np.where(allrows[:, 0] == model)[0], :]
        for k in np.unique(rows1[:, 1]):
            rows2 = rows1[np.where(rows1[:, 1] == str(k))[0], :]
            for nucleipref in np.unique(rows2[:, 2]):
                rows3 = rows2[np.where(rows2[:, 2] == str(nucleipref))[0], :]
                for seed in np.unique(rows3[:, 3]):
                    rows4 = rows3[np.where(rows3[:, 3] == str(seed))[0], :]
                    for n in np.unique(rows4[:, 4]):
                        rows5 = rows4[np.where(rows4[:, 4] == str(n))[0], :]
                        cs = [float(c) for c in np.unique(rows5[:, 5])]
                        rm = np.ones(len(cs)); rs = rm.copy();
                        sm = rm.copy(); ss = rm.copy();
                        for j, c in enumerate(cs):
                            rm[j] = means.loc[model, int(k), nucleipref,
                                          int(seed), int(n), c].r
                            rs[j] = stds.loc[model, int(k), nucleipref,
                                          int(seed), int(n), c].r
                            sm[j] = means.loc[model, int(k), nucleipref,
                                          int(seed), int(n), c].s
                            ss[j] = stds.loc[model, int(k), nucleipref,
                                          int(seed), int(n), c].s

                        fig, ax = plt.subplots(1, 2, figsize=(W*.01, H*.01),
                                dpi=100)

                        ax[0].errorbar(cs, rm, yerr=rs)
                        ax[0].set_xlabel('c'); ax[0].set_ylabel('r')
                        ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)

                        ax[1].errorbar(cs, sm, yerr=ss)
                        ax[1].set_xlabel('c'); ax[1].set_ylabel('s')
                        ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)

                        outpath = pjoin(outdir, '{}_{}_{}_{}_{}.png'.format(
                            model, k, nucleipref, n, seed))
                        plt.savefig(outpath)
                        plt.close()
                    if int(seed) > 3: break

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
    # un = get_unique_vals(df)

    # plot_r_s(df, un, pjoin(args.outdir, 'plots_r_s'), 1)
    dfcoeffs = find_coeffs(df, args.outdir)
    # plot_parameters_pairwise(dfcoeffs, un, pjoin(args.outdir, 'params'))
    plot_slice(dfcoeffs, 'avgdegree', 8, pjoin(args.outdir, 'slicek8'))
    # return
    # plot_slice(dfcoeffs, un, 'nverticescomp', 300, pjoin(args.outdir, 'slicen300'))

    # For multiple avgdegrees and nverticescomp
    # plot_contours(dfcoeffs, pjoin(args.outdir, 'contours'))
    # plot_triangulations(dfcoeffs, pjoin(args.outdir, 'surface_tri'))

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
