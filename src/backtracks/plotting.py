# backtrack plotting functions

# imports
import corner
import matplotlib.pyplot as plt
import numpy as np
import orbitize.system
import pandas as pd
import seaborn as sb

from astropy.time import Time
from dynesty import plotting as dyplot
from matplotlib.ticker import FuncFormatter

from backtracks.utils import transform_gengamm


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


def tickform(x, pos):
    # x:  tick value - ie. what you currently see in yticks
    # pos: a position - ie. the index of the tick (from 0 to 9 in this example)
    return '%.4f' % x


def diagnostic(backtrack, fileprefix='./', filepost='.pdf'):
    # Plot results.
    fig, axes = dyplot.runplot(backtrack.results, color='cornflowerblue')

    target_name = backtrack.target_name.replace(' ','_')
    plt.savefig(f'{fileprefix}{target_name}_evidence_backtrack'+filepost, dpi=300, bbox_inches='tight')

    return fig


def plx_prior(backtrack, fileprefix='./', filepost='.pdf'):
    beta = backtrack.beta
    alpha = backtrack.alpha
    L = backtrack.L

    fig = plt.figure(facecolor='white')
    u = np.arange(0.,1,0.001)
    plt.plot(u, 1000/transform_gengamm(u, L, alpha, beta), label=r'$\alpha=${}, $\beta=${}, L={}'.format(round(alpha,2), round(beta,2), round(L,2)), color='cornflowerblue')
    plt.xlabel('u')
    plt.ylabel('PPF(p(u))')
    plt.legend()

    target_name = backtrack.target_name.replace(' ','_')
    plt.savefig(f"{fileprefix}{target_name}_bjprior_backtrack"+filepost, dpi=300, bbox_inches='tight')

    return fig


def posterior(backtrack, fileprefix='./', filepost='.pdf'):
    labels = ["RA (deg, ep=2016)","DEC (deg, ep=2016)","pmra (mas/yr)","pmdec (mas/yr)","parallax (mas)"]
    levels = 1.0 - np.exp(-0.5 * np.arange(1, 3.1, 1) ** 2) # 1,2,3 sigma levels for a 2d gaussian
    # plot extended run (right)
    fig, ax = dyplot.cornerplot(backtrack.results,
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
    fig.figsize = (9,9)
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

    target_name = backtrack.target_name.replace(' ','_')
    plt.savefig(f"{fileprefix}{target_name}_corner_backtrack"+filepost, dpi=300, bbox_inches='tight')

    return fig

def trackplot(backtrack, days_backward=5.*365., days_forward=5.*365., 
              plot_radec=False, plot_stationary=False, fileprefix='./', filepost='.pdf'):
    """
    """
    fig, axs = plt.subplot_mosaic(
        '''
        AABB
        AACC
        ''',
        figsize=(16, 8))

    # Epochs for the background tracks

    plot_epochs = np.arange(days_backward+days_forward) + backtrack.jd_tt - days_backward

    plot_times = Time(plot_epochs, format='jd')
    data_times = Time(backtrack.epochs, format='jd')

    # Posterior samples, best-fit parameters, 16th and 84th percentile

    post_samples = backtrack.results.samples
    best_pars = np.array(backtrack.run_median).T[0]
    quant_pars = np.array(backtrack.run_quant).T

    # The corr_terms are True where rho is not a nan

    corr_terms =~ np.isnan(backtrack.rho)

    # Plot stationary background track at infinity

    if plot_stationary:
        stat_pars = best_pars.copy()
        stat_pars[2:] = 0.0
        
        ra_stat, dec_stat = backtrack.radecdists(plot_epochs, stat_pars)
        axs['A'].plot(ra_stat, dec_stat, color="lightgray", label="stationary background", ls='--')
        if plot_radec:
            axs['B'].plot(plot_times.decimalyear, ra_stat, color="lightgray", ls='--')
            axs['C'].plot(plot_times.decimalyear, dec_stat, color="lightgray", ls='--')
        else:
            sep_stat, pa_stat = orbitize.system.radec2seppa(ra_stat, dec_stat)
            axs['B'].plot(plot_times.decimalyear, sep_stat, color='lightgray', ls='--')
            axs['C'].plot(plot_times.decimalyear, pa_stat, color='lightgray', ls='--')

    # Plot random samples from the posterior in (deltaRA, deltaDec)

    # random_idx = np.random.choice(post_samples.shape[0], size=3)
    # for i in random_idx:
    #     ra_bg, dec_bg = backtrack.radecdists(plot_epochs, post_samples[i, ])
    #     axs['A'].plot(ra_bg, dec_bg, color='gray', lw=0.3)

    # Get background tracks for 16th and 84th percentile parameters

    ra_bg_q1, dec_bg_q1 = backtrack.radecdists(plot_epochs, quant_pars[0, ])
    ra_bg_q2, dec_bg_q2 = backtrack.radecdists(plot_epochs, quant_pars[1, ])

    # Convert quantile tracks to sep and PA

    sep_q1, pa_q1 = orbitize.system.radec2seppa(ra_bg_q1, dec_bg_q1)
    sep_q2, pa_q2 = orbitize.system.radec2seppa(ra_bg_q2, dec_bg_q2)

    # Plot sep and PA quantile envelopes

    if plot_radec:
        axs['B'].fill_between(plot_times.decimalyear, y1=ra_bg_q1, y2=ra_bg_q2, color='cornflowerblue', alpha=0.4)
        axs['C'].fill_between(plot_times.decimalyear, y1=dec_bg_q1, y2=dec_bg_q2, color='cornflowerblue', alpha=0.4)

    else:
        axs['B'].fill_between(plot_times.decimalyear, y1=sep_q1, y2=sep_q2, color='cornflowerblue', alpha=0.4)
        axs['C'].fill_between(plot_times.decimalyear, y1=pa_q1, y2=pa_q2, color='cornflowerblue', alpha=0.4)

    # Plot background track with best-fit parameters

    ra_bg, dec_bg = backtrack.radecdists(plot_epochs, best_pars)

    axs['A'].plot(ra_bg, dec_bg, color="black", label="best model")

    if plot_radec:
        axs['B'].plot(plot_times.decimalyear, ra_bg, color='black')
        axs['C'].plot(plot_times.decimalyear, dec_bg, color='black')

    else:
        sep_best, pa_best = orbitize.system.radec2seppa(ra_bg, dec_bg)
        axs['B'].plot(plot_times.decimalyear, sep_best, color='black')
        axs['C'].plot(plot_times.decimalyear, pa_best, color='black')

    # Connect data points with best-fit model epochs

    ra_bg_best, dec_bg_best = backtrack.radecdists(backtrack.epochs, best_pars)

    for i in range(len(backtrack.ras)):
        comp_ra = (backtrack.ras[i], ra_bg_best[i])
        comp_dec = (backtrack.decs[i], dec_bg_best[i])
        axs['A'].plot(comp_ra, comp_dec, ls='-', color='tab:gray', lw=1.0)

    # Plot coordinates at observation epochs for best-fit parameters

    axs['A'].plot(ra_bg_best, dec_bg_best, color="tomato", mec='tab:gray',
                  ms=5., mew=1.5, linestyle="none", marker="o")

    # Plot data points (deltaRA, deltaDEC)

    axs['A'].errorbar(backtrack.ras, backtrack.decs,
                      yerr=backtrack.decserr, xerr=backtrack.raserr,
                      color="tab:gray", ecolor='tomato', mec='tomato',
                      label="data", linestyle="none", marker="o", ms=5., mew=1.5)

    # Plot deltaRA/deltaDec or sep/PA as function of date

    if plot_radec:
        # Plot the deltaRA and deltaDec data points

        axs['B'].errorbar(data_times.decimalyear, backtrack.ras,
                          yerr=backtrack.raserr, color="tab:gray",
                          ecolor='tomato', linestyle="none",
                          marker='o', ms=5., mew=1.5, mec='tomato')

        axs['C'].errorbar(data_times.decimalyear, backtrack.decs,
                          yerr=backtrack.decserr, color="tab:gray",
                          ecolor='tomato', linestyle="none",
                          marker='o', ms=5., mew=1.5, mec='tomato')

    else:
        # Plot the sep and PA data points that are not corr_terms
        # The error calculation is adopted from orbitize!

        sep, pa = orbitize.system.radec2seppa(backtrack.ras[~corr_terms], backtrack.decs[~corr_terms])
        sep_err = 0.5*backtrack.raserr[~corr_terms] + 0.5*backtrack.decserr[~corr_terms]
        pa_err = np.degrees(sep_err/sep)

        axs['B'].errorbar(data_times.decimalyear[~corr_terms], sep,
                          yerr=sep_err, color="tomato", linestyle="none")

        axs['C'].errorbar(data_times.decimalyear[~corr_terms], pa,
                          yerr=pa_err, color="tomato", linestyle="none")

        # Plot the sep and PA data points that are corr_terms
        # Transform the uncertainties from RA/Dec to sep/PA

        sep, pa = orbitize.system.radec2seppa(backtrack.ras[corr_terms], backtrack.decs[corr_terms])

        for i in np.arange(np.sum(corr_terms)):
            sep_err, pa_err, rho2 = orbitize.system.transform_errors(
                backtrack.ras[corr_terms][i], backtrack.decs[corr_terms][i],
                backtrack.raserr[corr_terms][i], backtrack.decserr[corr_terms][i],
                backtrack.rho[corr_terms][i], orbitize.system.radec2seppa)

            axs['B'].errorbar(data_times.decimalyear[corr_terms][i], sep[i], yerr=sep_err,
                              color="tomato", linestyle="none", marker='none')

            axs['C'].errorbar(data_times.decimalyear[corr_terms][i], pa[i], yerr=pa_err,
                              color="tomato", linestyle="none", marker='none')

    axs['A'].invert_xaxis()
    axs['A'].axis('equal')
    axs['A'].set_xlabel("Delta RA (mas)")
    axs['A'].set_ylabel("Delta DEC (mas)")
    axs['A'].legend()
    axs['B'].set_xlabel('Date (year)')
    axs['C'].set_xlabel('Date (year)')

    if plot_radec:
        axs['B'].set_ylabel('Delta RA (mas)')
        axs['C'].set_ylabel('Delta DEC (mas)')
    else:
        axs['B'].set_ylabel('Separation (mas)')
        axs['C'].set_ylabel('PA (degrees)')

    plt.tight_layout()

    target_name = backtrack.target_name.replace(' ','_')
    plt.savefig(f"{fileprefix}{target_name}_model_backtrack"+filepost, dpi=300, bbox_inches='tight')

    return fig


def neighborhood(backtrack, fileprefix='./', filepost='.pdf'):
    """
    """
    # check how unlikely the fit motion is by comparing the distribution to the nearby Gaia distribution

    nearby_table = pd.DataFrame(columns=['pmra','pmdec','parallax'])
    nearby_table['pmra'] = backtrack.nearby['pmra'].data
    nearby_table['pmdec'] = backtrack.nearby['pmdec'].data
    nearby_table['parallax'] = backtrack.nearby['parallax'].data

    truths = np.array(backtrack.run_median).T[0][2:]

    # 1,2,3 sigma levels for a 2d gaussian
    levels = 1.0 - np.exp(-0.5 * np.arange(1, 3.1, 1) ** 2)

    nearby_array = nearby_table.to_numpy()
    nan_idx = np.isnan(nearby_array).any(axis=1)

    fig = corner.corner(nearby_array[~nan_idx, ],
                        truths=truths,
                        truth_color='cornflowerblue',
                        smooth=1,
                        smooth_1d=1,
                        quantiles=[0.003,0.997],
                        levels=levels)

    target_name = backtrack.target_name.replace(' ','_')
    plt.savefig(f'{fileprefix}{target_name}_nearby_gaia_distribution'+filepost, dpi=300, bbox_inches='tight')

    return fig


def stationtrackplot(backtrack, daysback=2600, daysforward=1200, fileprefix='./', filepost='.pdf'):
    """
    """
    fig,axs = plt.subplot_mosaic(
        '''
        AABB
        AACC
        ''',
        figsize=(16,8))

    plot_epochs=np.arange(daysforward)+backtrack.jd_tt-daysback # 4000 days of epochs to evaluate position at

    plot_times = Time(plot_epochs, format='jd')
    data_times = Time(backtrack.epochs, format='jd')

    post_samples = backtrack.results.samples
    best_pars = np.array(backtrack.run_median).T[0]
    quant_pars = np.array(backtrack.run_quant).T

    corr_terms=~np.isnan(backtrack.rho) # corr_terms is everywhere where rho is not a nan.

    # plot stationary bg track at infinity (0 parallax)
    rasbg,decbg = backtrack.radecdists(plot_epochs, best_pars[0],best_pars[1],0,0,0) # retrieve coordinates at full range of epochs
    # rasbg,decbg = backtrack.radecdists(plot_epochs, backtrack.ra0, backtrack.dec0, 0,0,0) # retrieve coordinates at full range of epochs

    axs['B'].plot(plot_times.decimalyear,orbitize.system.radec2seppa(rasbg,decbg)[0],color='gray',alpha=1, zorder=3,ls='--')
    axs['C'].plot(plot_times.decimalyear,orbitize.system.radec2seppa(rasbg,decbg)[1],color='gray',alpha=1, zorder=3,ls='--')

    axs['A'].plot(rasbg,decbg,color="gray",zorder=3,label="stationary bg",alpha=1,ls='--')

    # plot data in deltaRA, deltaDEC plot
    axs['A'].errorbar(backtrack.ras[~corr_terms],backtrack.decs[~corr_terms],yerr=backtrack.decserr[~corr_terms],xerr=backtrack.raserr[~corr_terms],
                      color="tomato",label="data",linestyle="", zorder=5)
    axs['A'].errorbar(backtrack.ras[corr_terms],backtrack.decs[corr_terms],yerr=backtrack.decserr[corr_terms],xerr=backtrack.raserr[corr_terms],
                      color="tomato",linestyle="",marker=".", zorder=5)
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
    axs['B'].errorbar(data_times.decimalyear[~corr_terms],sep,yerr=sep_err,color="tomato",linestyle="", zorder=5)
    axs['C'].errorbar(data_times.decimalyear[~corr_terms],pa,yerr=pa_err,color="tomato",linestyle="", zorder=5)

    # plot the corr_terms datapoints with a conversion in the sep and PA plots
    sep, pa = orbitize.system.radec2seppa(backtrack.ras[corr_terms],backtrack.decs[corr_terms])
    for i in np.arange(np.sum(corr_terms)):
        sep_err,pa_err,rho2 = orbitize.system.transform_errors(ras[corr_terms][i], backtrack.decs[corr_terms][i],
                                                               backtrack.raserr[corr_terms][i], backtrack.decserr[corr_terms][i], backtrack.rho[corr_terms][i], orbitize.system.radec2seppa)
        axs['B'].errorbar(data_times.decimalyear[corr_terms][i], sep[i], yerr=sep_err, color="tomato", linestyle="", marker='.', zorder=5)
        axs['C'].errorbar(data_times.decimalyear[corr_terms][i], pa[i], yerr=pa_err, color="tomato", linestyle="", marker='.', zorder=5)
    axs['A'].legend()

    plt.tight_layout()

    target_name = backtrack.target_name.replace(' ','_')
    plt.savefig(f"{fileprefix}{target_name}_model_stationary_backtrack"+filepost, dpi=300, bbox_inches='tight')

    return fig
