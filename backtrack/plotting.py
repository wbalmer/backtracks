# backtrack plotting functions

# imports
import os
import orbitize
import orbitize.system
import pandas as pd

# sampling
import dynesty

# computations
import numpy as np

# astropy
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, Angle
# astroquery
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # Select Data Release 3, default

# novas
import novas.compat as novas
from novas.compat.eph_manager import ephem_open

# plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import corner
from dynesty import plotting as dyplot
import seaborn as sb
plt.style.use('default')
plt.rcParams['figure.figsize'] = [9., 6.]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'monospace'   # Fonts
plt.rcParams['font.monospace'] = 'DejaVu Sans Mono'
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["ytick.minor.visible"] = True
sb.set_context("talk")

from backtrack.utils import *

def tickform(x, pos):
    # x:  tick value - ie. what you currently see in yticks
    # pos: a position - ie. the index of the tick (from 0 to 9 in this example)
    return '%.4f' % x


def diagnostic(backtrack, fileprefix='./'):
    # Plot results.
    fig, axes = dyplot.runplot(backtrack.results)
    plt.savefig(fileprefix+'{}_evidence_bgstar.png'.format(backtrack.target_name.replace(' ','_')), dpi=300, bbox_inches='tight')
    return fig


def plx_prior(backtrack, fileprefix='./'):
    beta = backtrack.beta
    alpha = backtrack.alpha
    L = backtrack.L

    plt.figure(facecolor='white')
    u = np.arange(0.,1,0.001)
    plt.plot(u, 1000/transform_gengamm(u, L, alpha, beta), label=r'$\alpha=${}, $\beta=${}, L={}'.format(round(alpha,2), round(beta,2), round(L,2)), color='cornflowerblue')
    plt.xlabel('u')
    plt.ylabel('PPF(p(u))')
    plt.legend()
    plt.savefig(fileprefix+"{}_bjprior_bgstar.png".format(backtrack.target_name.replace(' ','_')),dpi=300, bbox_inches='tight')

    return


def posterior(backtrack, fileprefix='./'):
    labels = ["RA (deg, ep=2016)","DEC (deg, ep=2016)","pmra (mas/yr)","pmdec (mas/yr)","parallax (mas)"]
    levels = 1.0 - np.exp(-0.5 * np.arange(1, 3.1, 1) ** 2) # 1,2,3 sigma levels for a 2d gaussian
    # plot extended run (right)
    fg, ax = dyplot.cornerplot(backtrack.results,
                               color='cornflowerblue',
                               # truths=np.zeros(ndim),
                               # span=[(-4.5, 4.5) for i in range(ndim)],
                               labels=labels,
                               hist2d_kwargs={'levels':levels},
                               max_n_ticks=4,
                               # label_kwargs={},
                               show_titles=True,
                               title_kwargs={'fontsize':14}
                               # quantiles=None,
                               # fig=(fig, axes[:, 4:])
                              )
    fg.figsize = (9,9)
    plt.subplots_adjust(wspace=0.3,
                        hspace=0.3)

    tick_formatter = FuncFormatter(tickform)

    for i,axis_row in enumerate(ax):
        if i == 1:
            axis_row[0].yaxis.set_major_formatter(tick_formatter) # make left hand ticks behave
        for axis in axis_row:
            axis.set(xlabel=None, ylabel=None)
    for axis_col in ax[:,-1]:
        axis_col.xaxis.set_major_formatter(tick_formatter)
    ax[-1,-1].set_xscale('log')
    # ax[-1,-1].set_xlim(xmin=1e-2)

    plt.savefig(fileprefix+"{}_corner_bgstar.png".format(backtrack.target_name.replace(' ','_')),dpi=300, bbox_inches='tight')


def trackplot(backtrack, daysback=2600, daysforward=1200, fileprefix='./'):
    """
    """
    fig,axs = plt.subplot_mosaic(
        '''
        AABB
        AACC
        ''',
        figsize=(16,8))

    epochs2=np.arange(daysforward)+backtrack.jd_tt-daysback # 4000 days of epochs to evaluate position at

    times=Time(epochs2,format='jd')
    times0=Time(backtrack.epochs,format='jd')

    flat_samples = backtrack.results.samples
    # sel = np.random.choice(np.arange(np.shape(flat_samples)[0]),10)
    # pars = flat_samples[sel,:]
    best_pars = np.array(backtrack.run_median).T[0]
    pars = np.array(backtrack.run_quant).T

    corr_terms=~np.isnan(backtrack.rho) # corr_terms is everywhere where rho is not a nan.

    # plot stationary bg track at infinity (0 parallax)
    # rasbg,decbg = backtrack.radecdists(epochs2,best_pars[0],best_pars[1],0,0,0) # retrieve coordinates at full range of epochs
    #
    # axs['B'].plot(times.decimalyear,orbitize.system.radec2seppa(rasbg,decbg)[0],color='gray',alpha=1, zorder=3,ls='--')
    # axs['C'].plot(times.decimalyear,orbitize.system.radec2seppa(rasbg,decbg)[1],color='gray',alpha=1, zorder=3,ls='--')
    #
    # axs['A'].plot(rasbg,decbg,color="gray",zorder=3,label="stationary bg",alpha=1,ls='--')

    # plot best moving bg track

    rasbg,decbg = backtrack.radecdists(epochs2,best_pars[0],best_pars[1],best_pars[2],best_pars[3],best_pars[4]) # retrieve coordinates at full range of epochs

    axs['B'].plot(times.decimalyear,orbitize.system.radec2seppa(rasbg,decbg)[0],color='k',alpha=1, zorder=3)
    axs['C'].plot(times.decimalyear,orbitize.system.radec2seppa(rasbg,decbg)[1],color='k',alpha=1, zorder=3)

    axs['A'].plot(rasbg,decbg,color="k",zorder=3,label="best model",alpha=1)

    i = 0
    rasbg1_q1,decbg1_q1 = backtrack.radecdists(epochs2,pars[i,0],pars[i,1],pars[i,2],pars[i,3],pars[i,4]) # retrieve coordinates at full range of epochs
    i = 1
    rasbg2_q1,decbg2_q2 = backtrack.radecdists(epochs2,pars[i,0],pars[i,1],pars[i,2],pars[i,3],pars[i,4]) # retrieve coordinates at full range of epochs

    axs['B'].fill_between(times.decimalyear,orbitize.system.radec2seppa(rasbg1_q1,decbg1_q1)[0], y2=orbitize.system.radec2seppa(rasbg2_q1,decbg2_q2)[0], color='cornflowerblue',alpha=0.5, zorder=0)
    axs['C'].fill_between(times.decimalyear,orbitize.system.radec2seppa(rasbg1_q1,decbg1_q1)[1], y2=orbitize.system.radec2seppa(rasbg2_q1,decbg2_q2)[1], color='cornflowerblue',alpha=0.5, zorder=0)

    axs['A'].fill_between(rasbg1_q1, decbg1_q1, decbg2_q2, color='cornflowerblue',label=r"model $1\sigma$ quantile",alpha=0.5, zorder=0)
    # axs['A'].fill_between(rasbg1_q1, decbg1_q1, decbg2_q2, color="blue",label="model 1-3 sigma quantiles",alpha=0.5, zorder=0)
    # axs['A'].fill_between(rasbg1_q1, decbg1_q1, decbg2_q2, color="blue",label="model 1-3 sigma quantiles",alpha=0.5, zorder=0)


    # plot model @ epochs
    rasbg,decbg = backtrack.radecdists(backtrack.epochs,best_pars[0],best_pars[1],best_pars[2],best_pars[3],best_pars[4]) # retrieve coordinates at observing epochs

    axs['A'].plot(rasbg,decbg,color="red",alpha=1,linestyle="",marker=".", zorder=4)

    for i in range(len(backtrack.ras[~corr_terms])):
        compra = (backtrack.ras[i],rasbg[i])
        compdec = (backtrack.decs[i],decbg[i])
        axs['A'].plot(compra,compdec,color='xkcd:pink', alpha=0.5)

    # plot data in deltaRA, deltaDEC plot
    axs['A'].errorbar(backtrack.ras[~corr_terms],backtrack.decs[~corr_terms],yerr=backtrack.decserr[~corr_terms],xerr=backtrack.raserr[~corr_terms],
                      color="xkcd:orange",label="data",linestyle="", zorder=5)
    axs['A'].errorbar(backtrack.ras[corr_terms],backtrack.decs[corr_terms],yerr=backtrack.decserr[corr_terms],xerr=backtrack.raserr[corr_terms],
                      color="xkcd:orange",linestyle="",marker=".", zorder=5)
    # labels
    # plt.suptitle("fitted model of background star for YSES-2")
    axs['A'].invert_xaxis()
    axs['A'].axis('equal')
    axs['A'].set_xlabel("Delta RA (mas)")
    axs['A'].set_ylabel("Delta DEC (mas)")
    axs['B'].set_xlabel('Date (year)')
    axs['C'].set_xlabel('Date (year)')
    axs['B'].set_ylabel('Separation (mas)')
    axs['C'].set_ylabel('PA (degrees)')

    # plot the datapoints that are not corr_terms in the sep and PA plots, error calculation taken from Orbitize!
    sep,pa=orbitize.system.radec2seppa(backtrack.ras[~corr_terms],backtrack.decs[~corr_terms])
    sep_err=0.5*backtrack.raserr[~corr_terms]+0.5*backtrack.decserr[~corr_terms]
    pa_err=np.degrees(sep_err/sep)
    axs['B'].errorbar(times0.decimalyear[~corr_terms],sep,yerr=sep_err,color="xkcd:orange",linestyle="", zorder=5)
    axs['C'].errorbar(times0.decimalyear[~corr_terms],pa,yerr=pa_err,color="xkcd:orange",linestyle="", zorder=5)

    # plot the corr_terms datapoints with a conversion in the sep and PA plots
    sep, pa = orbitize.system.radec2seppa(backtrack.ras[corr_terms],backtrack.decs[corr_terms])
    for i in np.arange(np.sum(corr_terms)):
        sep_err,pa_err,rho2 = orbitize.system.transform_errors(backtrack.ras[corr_terms][i], backtrack.decs[corr_terms][i],
                                                               backtrack.raserr[corr_terms][i], backtrack.decserr[corr_terms][i], backtrack.rho[corr_terms][i], orbitize.system.radec2seppa)
        axs['B'].errorbar(times0.decimalyear[corr_terms][i], sep[i], yerr=sep_err, color="xkcd:orange", linestyle="", marker='.', zorder=5)
        axs['C'].errorbar(times0.decimalyear[corr_terms][i], pa[i], yerr=pa_err, color="xkcd:orange", linestyle="", marker='.', zorder=5)
    axs['A'].legend()

    plt.tight_layout()
    plt.savefig(fileprefix+"{}_model_bgstar.png".format(backtrack.target_name.replace(' ','_')), dpi=300, bbox_inches='tight')
    return fig


def neighborhood(backtrack, fileprefix='./'):
    """
    """
    # check how unlikely the fit motion is by comparing the distribution to the nearby Gaia distribution

    nearby_table = pd.DataFrame(columns=['pmra','pmdec','parallax'])
    nearby_table['pmra'] = backtrack.nearby['pmra'].data
    nearby_table['pmdec'] = backtrack.nearby['pmdec'].data
    nearby_table['parallax'] = backtrack.nearby['parallax'].data
    truths = np.array(backtrack.run_median).T[0][2:]
    levels = 1.0 - np.exp(-0.5 * np.arange(1, 3.1, 1) ** 2) # 1,2,3 sigma levels for a 2d gaussian
    nearby_corner = corner.corner(nearby_table.dropna(), truths=truths,
                                  truth_color='cornflowerblue',
                                  smooth=1, smooth_1d=1,
                                  quantiles=[0.003,0.997], levels=levels,
                                 )

    plt.savefig(fileprefix+'{}_nearby_gaia_distribution.png'.format(backtrack.target_name.replace(' ','_')), dpi=300, bbox_inches='tight')
    return nearby_corner


def stationtrackplot(backtrack, daysback=2600, daysforward=1200, fileprefix='./'):
    """
    """
    fig,axs = plt.subplot_mosaic(
        '''
        AABB
        AACC
        ''',
        figsize=(16,8))

    epochs2=np.arange(daysforward)+backtrack.jd_tt-daysback # 4000 days of epochs to evaluate position at

    times=Time(epochs2,format='jd')
    times0=Time(backtrack.epochs,format='jd')

    flat_samples = backtrack.results.samples
    # sel = np.random.choice(np.arange(np.shape(flat_samples)[0]),10)
    # pars = flat_samples[sel,:]
    best_pars = np.array(backtrack.run_median).T[0]
    pars = np.array(backtrack.run_quant).T

    corr_terms=~np.isnan(backtrack.rho) # corr_terms is everywhere where rho is not a nan.

    # plot stationary bg track at infinity (0 parallax)
    rasbg,decbg = backtrack.radecdists(epochs2, best_pars[0],best_pars[1],0,0,0) # retrieve coordinates at full range of epochs
    # rasbg,decbg = backtrack.radecdists(epochs2, backtrack.ra0, backtrack.dec0, 0,0,0) # retrieve coordinates at full range of epochs

    axs['B'].plot(times.decimalyear,orbitize.system.radec2seppa(rasbg,decbg)[0],color='gray',alpha=1, zorder=3,ls='--')
    axs['C'].plot(times.decimalyear,orbitize.system.radec2seppa(rasbg,decbg)[1],color='gray',alpha=1, zorder=3,ls='--')

    axs['A'].plot(rasbg,decbg,color="gray",zorder=3,label="stationary bg",alpha=1,ls='--')

    # plot data in deltaRA, deltaDEC plot
    axs['A'].errorbar(backtrack.ras[~corr_terms],backtrack.decs[~corr_terms],yerr=backtrack.decserr[~corr_terms],xerr=backtrack.raserr[~corr_terms],
                      color="xkcd:orange",label="data",linestyle="", zorder=5)
    axs['A'].errorbar(backtrack.ras[corr_terms],backtrack.decs[corr_terms],yerr=backtrack.decserr[corr_terms],xerr=backtrack.raserr[corr_terms],
                      color="xkcd:orange",linestyle="",marker=".", zorder=5)
    # labels
    # plt.suptitle("fitted model of background star for YSES-2")
    axs['A'].invert_xaxis()
    axs['A'].axis('equal')
    axs['A'].set_xlabel("Delta RA (mas)")
    axs['A'].set_ylabel("Delta DEC (mas)")
    axs['B'].set_xlabel('Date (year)')
    axs['C'].set_xlabel('Date (year)')
    axs['B'].set_ylabel('Separation (mas)')
    axs['C'].set_ylabel('PA (degrees)')

    # plot the datapoints that are not corr_terms in the sep and PA plots, error calculation taken from Orbitize!
    sep,pa=orbitize.system.radec2seppa(backtrack.ras[~corr_terms],backtrack.decs[~corr_terms])
    sep_err=0.5*backtrack.raserr[~corr_terms]+0.5*backtrack.decserr[~corr_terms]
    pa_err=np.degrees(sep_err/sep)
    axs['B'].errorbar(times0.decimalyear[~corr_terms],sep,yerr=sep_err,color="xkcd:orange",linestyle="", zorder=5)
    axs['C'].errorbar(times0.decimalyear[~corr_terms],pa,yerr=pa_err,color="xkcd:orange",linestyle="", zorder=5)

    # plot the corr_terms datapoints with a conversion in the sep and PA plots
    sep, pa = orbitize.system.radec2seppa(backtrack.ras[corr_terms],backtrack.decs[corr_terms])
    for i in np.arange(np.sum(corr_terms)):
        sep_err,pa_err,rho2 = orbitize.system.transform_errors(ras[corr_terms][i], backtrack.decs[corr_terms][i],
                                                               backtrack.raserr[corr_terms][i], backtrack.decserr[corr_terms][i], backtrack.rho[corr_terms][i], orbitize.system.radec2seppa)
        axs['B'].errorbar(times0.decimalyear[corr_terms][i], sep[i], yerr=sep_err, color="xkcd:orange", linestyle="", marker='.', zorder=5)
        axs['C'].errorbar(times0.decimalyear[corr_terms][i], pa[i], yerr=pa_err, color="xkcd:orange", linestyle="", marker='.', zorder=5)
    axs['A'].legend()

    plt.tight_layout()
    plt.savefig(fileprefix+"{}_model_stationary_bgstar.png".format(backtrack.target_name.replace(' ','_')), dpi=300, bbox_inches='tight')
    return fig
