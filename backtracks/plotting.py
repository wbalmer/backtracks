# backtracks plotting functions

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from astropy.time import Time
from dynesty import plotting as dyplot
from matplotlib.ticker import FuncFormatter

from backtracks.utils import transform_gengamm, radec2seppa, seppa2radec, transform_errors


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


def diagnostic(backtracks, fileprefix='./', filepost='.pdf'):
    # Plot results.
    fig, axes = dyplot.runplot(backtracks.results, color='cornflowerblue')

    target_name = backtracks.target_name.replace(' ','_')
    plt.savefig(f'{fileprefix}{target_name}_evidence_backtracks'+filepost, dpi=300, bbox_inches='tight')

    return fig


def plx_prior(backtracks, fileprefix='./', filepost='.pdf'):
    u = np.arange(0., 1, 0.001)
    ppf = 1000./transform_gengamm(u, backtracks.L, backtracks.alpha, backtracks.beta)

    fig = plt.figure(facecolor='white')
    label = rf'$\alpha=${backtracks.alpha:.2f}, $\beta=${backtracks.beta:.2f}, L={backtracks.L:.2f}'
    plt.plot(u, ppf, label=label, color='cornflowerblue')
    plt.xlabel('u')
    plt.ylabel('PPF(p(u))')
    plt.legend()

    target_name = backtracks.target_name.replace(' ', '_')
    plt.savefig(f"{fileprefix}{target_name}_bjprior_backtracks"+filepost, dpi=300, bbox_inches='tight')

    return fig


def posterior(backtracks, fileprefix='./', filepost='.pdf'):
    labels = ["RA (deg, ep=2016)", "DEC (deg, ep=2016)", "pmra (mas/yr)", "pmdec (mas/yr)", "parallax (mas)"]
    levels = 1.0 - np.exp(-0.5 * np.arange(1, 3.1, 1) ** 2) # 1,2,3 sigma levels for a 2d gaussian
    # plot extended run (right)
    fig, ax = dyplot.cornerplot(backtracks.results,
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

    target_name = backtracks.target_name.replace(' ', '_')
    plt.savefig(f"{fileprefix}{target_name}_corner_backtracks"+filepost, dpi=300, bbox_inches='tight')

    return fig

def trackplot(
        backtracks,
        ref_epoch,
        days_backward=3.*365.,
        days_forward=3.*365.,
        step_size=10.,
        plot_radec=False,
        plot_stationary=False,
        fileprefix='./',
        filepost='.pdf'
    ):
    """
    """
    fig, axs = plt.subplot_mosaic(
        '''
        AABB
        AACC
        ''',
        figsize=(16, 8))

    # Epochs for the background tracks

    plot_epochs = ref_epoch + np.arange(-days_backward, days_forward, step_size)

    # Create astropy times for observations and background model

    plot_times = Time(plot_epochs, format='jd')
    obs_times = Time(backtracks.epochs, format='jd')

    # The corr_terms are True where rho is not a nan

    corr_terms =~ np.isnan(backtracks.rho)

    # Plot stationary background track at infinity

    if plot_stationary:
        stat_pars = backtracks.run_median.copy()
        stat_pars[2:] = 0.0

        ra_stat, dec_stat = backtracks.radecdists(plot_epochs, stat_pars)
        axs['A'].plot(ra_stat, dec_stat, color="lightgray", label="stationary background", ls='--')

        if plot_radec:
            axs['B'].plot(plot_times.decimalyear, ra_stat, color="lightgray", ls='--')
            axs['C'].plot(plot_times.decimalyear, dec_stat, color="lightgray", ls='--')
        else:
            sep_stat, pa_stat = radec2seppa(ra_stat, dec_stat)
            axs['B'].plot(plot_times.decimalyear, sep_stat, color='lightgray', ls='--')
            axs['C'].plot(plot_times.decimalyear, pa_stat, color='lightgray', ls='--')

    # Create background tracks from posterior samples

    post_samples = backtracks.results.samples_equal()

    n_samples = 200
    random_idx = np.random.choice(post_samples.shape[0], size=n_samples)

    ra_samples = np.zeros((n_samples, plot_epochs.size))
    dec_samples = np.zeros((n_samples, plot_epochs.size))

    for i, idx_item in enumerate(random_idx):
        ra_samples[i, ], dec_samples[i, ] = backtracks.radecdists(plot_epochs, post_samples[idx_item, ])
        axs['A'].plot(ra_samples[i, ], dec_samples[i, ], color='cornflowerblue', lw=0.3, alpha=0.3)

    # Create 1 sigma and 3 sigma percentiles for envelopes

    ra_quant = np.percentile(ra_samples, [0.13, 16.0, 84.0, 99.87], axis=0)
    dec_quant = np.percentile(dec_samples, [0.13, 16.0, 84.0, 99.87], axis=0)

    # Convert quantile tracks to sep and PA

    sep_q1, pa_q1 = radec2seppa(ra_quant[0, ], dec_quant[0, ])
    sep_q2, pa_q2 = radec2seppa(ra_quant[1, ], dec_quant[1, ])
    sep_q3, pa_q3 = radec2seppa(ra_quant[2, ], dec_quant[2, ])
    sep_q4, pa_q4 = radec2seppa(ra_quant[3, ], dec_quant[3, ])

    # Plot quantile envelopes for RA/Dec or sep/PA

    if plot_radec:
        axs['B'].fill_between(plot_times.decimalyear, y1=ra_quant[0, ], y2=ra_quant[3, ], color='cornflowerblue', alpha=0.2, lw=0.)
        axs['C'].fill_between(plot_times.decimalyear, y1=dec_quant[0, ], y2=dec_quant[3, ], color='cornflowerblue', alpha=0.2, lw=0.)
        axs['B'].fill_between(plot_times.decimalyear, y1=ra_quant[1, ], y2=ra_quant[2, ], color='cornflowerblue', alpha=0.4, lw=0.)
        axs['C'].fill_between(plot_times.decimalyear, y1=dec_quant[1, ], y2=dec_quant[2, ], color='cornflowerblue', alpha=0.4, lw=0.)

    else:
        axs['B'].fill_between(plot_times.decimalyear, y1=sep_q1, y2=sep_q4, color='cornflowerblue', alpha=0.2, lw=0.)
        axs['C'].fill_between(plot_times.decimalyear, y1=pa_q1, y2=pa_q4, color='cornflowerblue', alpha=0.2, lw=0.)
        axs['B'].fill_between(plot_times.decimalyear, y1=sep_q2, y2=sep_q3, color='cornflowerblue', alpha=0.4, lw=0.)
        axs['C'].fill_between(plot_times.decimalyear, y1=pa_q2, y2=pa_q3, color='cornflowerblue', alpha=0.4, lw=0.)

    # Plot background track with best-fit parameters

    ra_bg, dec_bg = backtracks.radecdists(plot_epochs, backtracks.run_median)

    axs['A'].plot(ra_bg, dec_bg, color="black", label="best model")

    if plot_radec:
        axs['B'].plot(plot_times.decimalyear, ra_bg, color='black')
        axs['C'].plot(plot_times.decimalyear, dec_bg, color='black')

    else:
        sep_best, pa_best = radec2seppa(ra_bg, dec_bg)
        axs['B'].plot(plot_times.decimalyear, sep_best, color='black')
        axs['C'].plot(plot_times.decimalyear, pa_best, color='black')

    # Connect data points with best-fit model epochs

    ra_bg_best, dec_bg_best = backtracks.radecdists(backtracks.epochs, backtracks.run_median)

    for i in range(len(backtracks.ras)):
        comp_ra = (backtracks.ras[i], ra_bg_best[i])
        comp_dec = (backtracks.decs[i], dec_bg_best[i])
        axs['A'].plot(comp_ra, comp_dec, ls='-', color='tab:gray', lw=1.0)

    # Plot coordinates at observation epochs for best-fit parameters

    axs['A'].plot(ra_bg_best, dec_bg_best, color="tomato", mec='tab:gray',
                  ms=5., mew=1.5, linestyle="none", marker="o")

    # Plot data points (deltaRA, deltaDEC)

    axs['A'].errorbar(backtracks.ras, backtracks.decs,
                      yerr=backtracks.decserr, xerr=backtracks.raserr,
                      color="tab:gray", ecolor='tomato', mec='tomato',
                      label="data", linestyle="none", marker="o", ms=5., mew=1.5)

    # Plot deltaRA/deltaDec or sep/PA as function of date

    if plot_radec:
        # Plot the deltaRA and deltaDec data points

        axs['B'].errorbar(obs_times.decimalyear, backtracks.ras,
                          yerr=backtracks.raserr, color="tab:gray",
                          ecolor='tomato', linestyle="none",
                          marker='o', ms=5., mew=1.5, mec='tomato')

        axs['C'].errorbar(obs_times.decimalyear, backtracks.decs,
                          yerr=backtracks.decserr, color="tab:gray",
                          ecolor='tomato', linestyle="none",
                          marker='o', ms=5., mew=1.5, mec='tomato')

    else:
        # Plot the sep and PA data points
        # The error calculation is adopted from orbitize!

        if np.sum(~corr_terms) > 0:
            # Plot the sep and PA data points that are not corr_terms

            obs_sep, obs_pa = radec2seppa(backtracks.ras[~corr_terms], backtracks.decs[~corr_terms])
            obs_sep_err = 0.5*backtracks.raserr[~corr_terms] + 0.5*backtracks.decserr[~corr_terms]
            obs_pa_err = np.degrees(obs_sep_err/obs_sep)

            axs['B'].errorbar(obs_times[~corr_terms].decimalyear, obs_sep,
                              yerr=obs_sep_err, color="tab:gray",
                              ecolor='tomato', linestyle="none",
                              marker='o', ms=5., mew=1.5, mec='tomato')

            axs['C'].errorbar(obs_times[~corr_terms].decimalyear, obs_pa,
                              yerr=obs_pa_err, color="tab:gray",
                              ecolor='tomato', linestyle="none",
                              marker='o', ms=5., mew=1.5, mec='tomato')

        if np.sum(corr_terms) > 0:
            # Plot the sep and PA data points that are corr_terms

            obs_sep, obs_pa = radec2seppa(backtracks.ras[corr_terms], backtracks.decs[corr_terms])

            obs_sep_err = np.zeros(obs_sep.size)
            obs_pa_err = np.zeros(obs_sep.size)

            for i in np.arange(np.sum(corr_terms)):
                # Transform the uncertainties from RA/Dec to sep/PA
                obs_sep_err[i], obs_pa_err[i], _ = transform_errors(
                    backtracks.ras[corr_terms][i], backtracks.decs[corr_terms][i],
                    backtracks.raserr[corr_terms][i], backtracks.decserr[corr_terms][i],
                    backtracks.rho[corr_terms][i], radec2seppa)

            axs['B'].errorbar(obs_times[corr_terms].decimalyear, obs_sep,
                              yerr=obs_sep_err, color="tab:gray",
                              ecolor='tomato', linestyle="none",
                              marker='o', ms=5., mew=1.5, mec='tomato')

            axs['C'].errorbar(obs_times[corr_terms].decimalyear, obs_pa,
                              yerr=obs_pa_err, color="tab:gray",
                              ecolor='tomato', linestyle="none",
                              marker='o', ms=5., mew=1.5, mec='tomato')

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

    target_name = backtracks.target_name.replace(' ', '_')
    plt.savefig(f"{fileprefix}{target_name}_model_backtracks"+filepost, dpi=300, bbox_inches='tight')

    return fig


def neighborhood(backtracks, fileprefix='./', filepost='.pdf'):
    """
    """
    # check how unlikely the fit motion is by comparing the distribution to the nearby Gaia distribution

    nearby_table = pd.DataFrame(columns=['pmra', 'pmdec', 'parallax'])
    nearby_table['pmra'] = backtracks.nearby['pmra'].data
    nearby_table['pmdec'] = backtracks.nearby['pmdec'].data
    nearby_table['parallax'] = backtracks.nearby['parallax'].data

    truths = backtracks.run_median[2:]

    # 1,2,3 sigma levels for a 2d gaussian
    levels = 1.0 - np.exp(-0.5 * np.arange(1, 3.1, 1) ** 2)

    nearby_array = nearby_table.to_numpy()
    nan_idx = np.isnan(nearby_array).any(axis=1)

    fig = corner.corner(nearby_array[~nan_idx, ],
                        truths=truths,
                        truth_color='cornflowerblue',
                        smooth=1,
                        smooth_1d=1,
                        quantiles=[0.003, 0.997],
                        levels=levels)

    target_name = backtracks.target_name.replace(' ', '_')
    plt.savefig(f'{fileprefix}{target_name}_nearby_gaia_distribution'+filepost, dpi=300, bbox_inches='tight')

    return fig


def stationtrackplot(backtracks, ref_epoch, daysback=2600, daysforward=1200, fileprefix='./', filepost='.pdf'):
    """
    """
    fig,axs = plt.subplot_mosaic(
        '''
        AABB
        AACC
        ''',
        figsize=(16,8))

    plot_epochs=np.arange(daysforward)+ref_epoch-daysback # 4000 days of epochs to evaluate position at

    plot_times = Time(plot_epochs, format='jd')
    obs_times = Time(backtracks.epochs, format='jd')

    post_samples = backtracks.results.samples
    best_pars = np.array(backtracks.run_median).T[0]
    quant_pars = np.array(backtracks.run_quant).T

    corr_terms=~np.isnan(backtracks.rho) # corr_terms is everywhere where rho is not a nan.

    # plot stationary bg track at infinity (0 parallax)
    rasbg,decbg = backtracks.radecdists(plot_epochs, best_pars[0],best_pars[1],0,0,0) # retrieve coordinates at full range of epochs
    # rasbg,decbg = backtracks.radecdists(plot_epochs, backtracks.ra0, backtracks.dec0, 0,0,0) # retrieve coordinates at full range of epochs

    axs['B'].plot(plot_times.decimalyear,radec2seppa(rasbg,decbg)[0],color='gray',alpha=1, zorder=3,ls='--')
    axs['C'].plot(plot_times.decimalyear,radec2seppa(rasbg,decbg)[1],color='gray',alpha=1, zorder=3,ls='--')

    axs['A'].plot(rasbg,decbg,color="gray",zorder=3,label="stationary bg",alpha=1,ls='--')

    # plot data in deltaRA, deltaDEC plot
    axs['A'].errorbar(backtracks.ras[~corr_terms],backtracks.decs[~corr_terms],yerr=backtracks.decserr[~corr_terms],xerr=backtracks.raserr[~corr_terms],
                      color="tomato",label="data",linestyle="", zorder=5)
    axs['A'].errorbar(backtracks.ras[corr_terms],backtracks.decs[corr_terms],yerr=backtracks.decserr[corr_terms],xerr=backtracks.raserr[corr_terms],
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
    sep,pa=radec2seppa(backtracks.ras[~corr_terms],backtracks.decs[~corr_terms])
    sep_err=0.5*backtracks.raserr[~corr_terms]+0.5*backtracks.decserr[~corr_terms]
    pa_err=np.degrees(sep_err/sep)
    axs['B'].errorbar(obs_times.decimalyear[~corr_terms],sep,yerr=sep_err,color="tomato",linestyle="", zorder=5)
    axs['C'].errorbar(obs_times.decimalyear[~corr_terms],pa,yerr=pa_err,color="tomato",linestyle="", zorder=5)

    # plot the corr_terms datapoints with a conversion in the sep and PA plots
    sep, pa = radec2seppa(backtracks.ras[corr_terms],backtracks.decs[corr_terms])
    for i in np.arange(np.sum(corr_terms)):
        sep_err,pa_err,rho2 = transform_errors(backtracks.ras[corr_terms][i], backtracks.decs[corr_terms][i],
                                                               backtracks.raserr[corr_terms][i], backtracks.decserr[corr_terms][i], backtracks.rho[corr_terms][i], radec2seppa)
        axs['B'].errorbar(obs_times.decimalyear[corr_terms][i], sep[i], yerr=sep_err, color="tomato", linestyle="", marker='.', zorder=5)
        axs['C'].errorbar(obs_times.decimalyear[corr_terms][i], pa[i], yerr=pa_err, color="tomato", linestyle="", marker='.', zorder=5)

    axs['A'].legend()

    plt.tight_layout()

    target_name = backtracks.target_name.replace(' ', '_')
    plt.savefig(f"{fileprefix}{target_name}_model_stationary_backtracks"+filepost, dpi=300, bbox_inches='tight')

    return fig
