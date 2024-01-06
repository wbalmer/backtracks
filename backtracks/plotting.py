# backtracks plotting functions

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from astropy.time import Time
from dynesty import plotting as dyplot
from matplotlib.ticker import FuncFormatter

from backtracks.utils import transform_gengamm, radec2seppa, seppa2radec, transform_errors, utc2tt

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
    """
    Function to reformat tick values.
    
    Args:
        x (float): tick value
        pos (int): index of tick    
    Returns:
        Reformatted tick value with 4 decimal places.
    """
    
    return '%.4f' % x


def diagnostic(backtracks, fileprefix='./', filepost='.pdf'):
    """
    Function to plot the dynesty run summary plot.

    Args:
        backtracks (class): backtracks.System class which carries the needed methods and attributes.
        fileprefix (str): Prefix to filename. Default "./" for current folder.
        filepost (str): Postfix to filename. Default ".pdf" for outputting pdf files.

    Returns:
        fig (figure): Figure object corresponding to the saved plot.
    """

    # Plot results.
    fig, axes = dyplot.runplot(backtracks.results, color='cornflowerblue')

    target_name = backtracks.target_name.replace(' ','_')
    plt.savefig(f'{fileprefix}{target_name}_evidence_backtracks'+filepost, dpi=300, bbox_inches='tight')

    return fig


def plx_prior(backtracks, fileprefix='./', filepost='.pdf'):
    """
    Function to plot the parallax prior (in mas).

    Args:
        backtracks (class): backtracks.System class which carries the needed methods and attributes.
        fileprefix (str): Prefix to filename. Default "./" for current folder.
        filepost (str): Postfix to filename. Default ".pdf" for outputting pdf files.

    Returns:
        fig (figure): Figure object corresponding to the saved plot.
    """

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
    """
    Function to plot a cornerplot of the posteriors of the fit to a moving background star scenario.

    Args:
        backtracks (class): backtracks.System class which carries the needed methods and attributes.
        fileprefix (str): Prefix to filename. Default "./" for current folder.
        filepost (str): Postfix to filename. Default ".pdf" for outputting pdf files.

    Returns:
        fig (figure): Figure object corresponding to the saved plot.

    Notes:
        * Only the five parameters of the background star are plotted.
    """

    labels = ["RA (deg, ep=2016)", "DEC (deg, ep=2016)", "pmra (mas/yr)", "pmdec (mas/yr)", "parallax (mas)"]
    levels = 1.0 - np.exp(-0.5 * np.arange(1, 3.1, 1) ** 2) # 1,2,3 sigma levels for a 2d gaussian
    # plot extended run (right)
    fig, ax = dyplot.cornerplot(backtracks.results,
                               color='cornflowerblue',
                               dims=range(5),
                               # truths=np.zeros(ndim),
                               # span=[(-4.5, 4.5) for i in range(ndim)],
                               labels=labels,
                               hist2d_kwargs={'levels':levels},
                               max_n_ticks=4,
                               # label_kwargs={},
                               show_titles=True,
                               title_kwargs={'fontsize':14},
                               quantiles=(0.1587,0.5,0.8413),
                               title_quantiles=(0.1587,0.5,0.8413)
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
    Function to plot a fitted (non-stationary) background star scenario track with the data.

    Args:
        backtracks (class): backtracks.System class which carries the needed methods and attributes.
        ref_epoch (float): Julian Date (UTC) as a reference point for the tracks
        days_backward (float): Days backward from the reference point at which to start tracks
        days_forward (float): Days forward from the reference point at which to end tracks
        step_size (float): Step size with which tracks are generated.
        plot_radec (bool): Option to plot RA vs time and DEC vs time in panels. Default: False for Sep vs time, PA vs time.
        plot_stationary (bool): Option to overplot a track corresponding to an infinitely far stationary background star. Default: False.
        fileprefix (str): Prefix to filename. Default "./" for current folder.
        filepost (str): Postfix to filename. Default ".pdf" for outputting pdf files.

    Returns:
        fig (figure): Figure object corresponding to the saved plot.
    """

    fig, axs = plt.subplot_mosaic(
        '''
        AABB
        AACC
        ''',
        figsize=(16, 8))

    # Epochs for the background tracks

    plot_epochs = ref_epoch + np.arange(-days_backward, days_forward, step_size)
    plot_epochs_tt=utc2tt(plot_epochs)
    # Create astropy times for observations and background model

    plot_times = Time(plot_epochs, format='jd')
    obs_times = Time(backtracks.epochs, format='jd')

    # The corr_terms are True where rho is not a nan

    corr_terms =~ np.isnan(backtracks.rho)

    # Plot stationary background track at infinity

    if plot_stationary:
        stat_pars = backtracks.stationary_params

        ra_stat, dec_stat = backtracks.radecdists(plot_epochs_tt, stat_pars)
        axs['A'].plot(ra_stat, dec_stat, color="lightgray", ls='--',
                      label="Stationary track, $\chi^2_r={}$".format(round(backtracks.stationary_chi2_red,2)))

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
        ra_samples[i, ], dec_samples[i, ] = backtracks.radecdists(plot_epochs_tt, post_samples[idx_item, ])
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

    ra_bg, dec_bg = backtracks.radecdists(plot_epochs_tt, backtracks.run_median)

    axs['A'].plot(ra_bg, dec_bg, color="black", label="Median track, $\chi^2_r={}$".format(round(backtracks.median_chi2_red,2)))

    if plot_radec:
        axs['B'].plot(plot_times.decimalyear, ra_bg, color='black')
        axs['C'].plot(plot_times.decimalyear, dec_bg, color='black')

    else:
        sep_best, pa_best = radec2seppa(ra_bg, dec_bg)
        axs['B'].plot(plot_times.decimalyear, sep_best, color='black')
        axs['C'].plot(plot_times.decimalyear, pa_best, color='black')

    # Connect data points with best-fit model epochs

    ra_bg_best, dec_bg_best = backtracks.radecdists(backtracks.epochs_tt, backtracks.run_median)

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
                      label="Data", linestyle="none", marker="o", ms=5., mew=1.5)

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
    Function to plot the best fitting background star parameters on top of the Gaia priors.

    Args:
        backtracks (class): backtracks.System class which carries the needed methods and attributes.
        fileprefix (str): Prefix to filename. Default "./" for current folder.
        filepost (str): Postfix to filename. Default ".pdf" for outputting pdf files.

    Returns:
        fig (figure): Figure object corresponding to the saved plot.
    """

    # check how unlikely the fit motion is by comparing the distribution to the nearby Gaia distribution

    nearby_table = pd.DataFrame(columns=['pmra', 'pmdec', 'parallax'])
    nearby_table['pmra'] = backtracks.nearby['pmra'].data
    nearby_table['pmdec'] = backtracks.nearby['pmdec'].data
    nearby_table['parallax'] = backtracks.nearby['parallax'].data

    truths = backtracks.run_median[2:5]

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


def stationtrackplot(
        backtracks,
        ref_epoch,
        days_backward=3.*365.,
        days_forward=3.*365.,
        step_size=10.,
        plot_radec=False,
        fileprefix='./',
        filepost='.pdf'
    ):
    """
    Function to plot an infinitely far stationary scenario track with the data.

    Args:
        backtracks (class): backtracks.System class which carries the needed methods and attributes.
        ref_epoch (float): Julian Date (UTC) as a reference point for the tracks
        days_backward (float): Days backward from the reference point at which to start tracks
        days_forward (float): Days forward from the reference point at which to end tracks
        step_size (float): Step size with which tracks are generated.
        plot_radec (bool): Option to plot RA vs time and DEC vs time in panels. Default: False for Sep vs time, PA vs time.
        fileprefix (str): Prefix to filename. Default "./" for current folder.
        filepost (str): Postfix to filename. Default ".pdf" for outputting pdf files.
    
    Returns:
        fig (figure): Figure object corresponding to the saved plot.
    """

    fig, axs = plt.subplot_mosaic(
        '''
        AABB
        AACC
        ''',
        figsize=(16, 8))

    # Epochs for the background tracks

    plot_epochs = ref_epoch + np.arange(-days_backward, days_forward, step_size)
    plot_epochs_tt=utc2tt(plot_epochs)
    # Create astropy times for observations and background model

    plot_times = Time(plot_epochs, format='jd')
    obs_times = Time(backtracks.epochs, format='jd')

    # The corr_terms are True where rho is not a nan

    corr_terms =~ np.isnan(backtracks.rho)

    # Plot stationary background track at infinity

    stat_pars = backtracks.stationary_params

    ra_stat, dec_stat = backtracks.radecdists(plot_epochs_tt, stat_pars)
    axs['A'].plot(ra_stat, dec_stat, color="lightgray", ls='--',
                    label="Stationary track, $\chi^2_r={}$".format(round(backtracks.stationary_chi2_red,2)))

    if plot_radec:
        axs['B'].plot(plot_times.decimalyear, ra_stat, color="lightgray", ls='--')
        axs['C'].plot(plot_times.decimalyear, dec_stat, color="lightgray", ls='--')
    else:
        sep_stat, pa_stat = radec2seppa(ra_stat, dec_stat)
        axs['B'].plot(plot_times.decimalyear, sep_stat, color='lightgray', ls='--')
        axs['C'].plot(plot_times.decimalyear, pa_stat, color='lightgray', ls='--')

    # Connect data points with best-fit model epochs

    ra_bg_best, dec_bg_best = backtracks.radecdists(backtracks.epochs_tt, stat_pars)

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
                      label="Data", linestyle="none", marker="o", ms=5., mew=1.5)

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
    plt.savefig(f"{fileprefix}{target_name}_stationary_backtracks"+filepost, dpi=300, bbox_inches='tight')

    return fig
