# backtrack.py
# authors: Gilles Otten, William Balmer

# special packages needed: astropy, matplotlib, numpy, novas, novas_de405,
# dynesty, emcee, orbitize, corner (potentially cython and tqdm if clean pip install crashes)

# this code does an analysis with a background star model given delta RA and
# delta DEC datapoints of a point source wrt a host star
# comparing Bayes factor or BIC of this analysis and a pure Orbitize! run would
# be informative to see which model fits better for new discoveries.

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
import corner
import seaborn as sb
from matplotlib import patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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


# TODO: CLASS STRUCTURE AND USER INPUTS
# target_name = input('target name? ') # "HD 131399 A", "YSES 2"
# candidate_file = input("candidate file path? ") # 'scorpions1b_orbitizelike.csv'

target_name = "HD 131399 A"
candidate_file = "scorpions1b_orbitizelike.csv"
daysback = 2600
daysforward = 1200

# target_name = "YSES 2"
# candidate_file = "yses2b_orbitizelike.csv"

# resolve target in simbad
target_result_table = Simbad.query_object(target_name)
print('[URASTAR INFO]: Resolved the target star \'{}\' in Simbad!'.format(target_name))
# target_result_table.pprint()
# get gaia ID from simbad
for ID in Simbad.query_objectids(target_name)['ID']:
    if 'Gaia DR3' in ID:
        gaia_id = int(ID.replace('Gaia DR3', ''))
        print('[URASTAR INFO]: Resolved target\'s Gaia ID from Simbad, Gaia DR3',gaia_id)

coord = SkyCoord(ra=target_result_table['RA'][0], dec=target_result_table['DEC'][0], unit=(u.hourangle, u.degree), frame='icrs')
width = u.Quantity(50, u.arcsec)
height = u.Quantity(50, u.arcsec)
Gaia.ROW_LIMIT = -1
target_gaia = Gaia.query_object_async(coordinate=coord, width=width, height=height)
target_gaia = target_gaia[target_gaia['source_id']==gaia_id]
print('[URASTAR INFO]: gathered target Gaia data')

nearby_window = 0.5
# resolve nearby stars
width = u.Quantity(nearby_window, u.deg)
height = u.Quantity(nearby_window, u.deg)

Gaia.ROW_LIMIT = -1
nearby = Gaia.query_object_async(coordinate=coord, width=width, height=height)

print(r'[URASTAR INFO]: gathered {} Gaia objects from the {} sq. deg. nearby {}'.format(len(nearby), nearby_window, target_name))

mu_pmra = np.ma.median(nearby['pmra'].data)
sigma_pmra = np.ma.std(nearby['pmra'].data)

mu_pmdec = np.ma.median(nearby['pmdec'].data)
sigma_pmdec = np.ma.std(nearby['pmdec'].data)

mu_par = np.ma.median(nearby['parallax'].data)
sigma_par = np.ma.std(nearby['parallax'].data)


def pol2car(sep, pa, seperr, paerr, corr=np.nan):
    ra, dec = orbitize.system.seppa2radec(sep, pa)
    raerr, decerr, corr2 = orbitize.system.transform_errors(sep, pa, seperr, paerr, corr, orbitize.system.seppa2radec)
    return ra, dec, decerr, raerr, corr2

transform_uniform = lambda x,a,b : a + (b-a)*x

# planet candidate astrometry
candidate = pd.read_csv(candidate_file)

astrometry = np.zeros((6,len(candidate))) # epoch, ra, dec, raerr, decerr, rho
for i,quant in enumerate(candidate['quant_type']):
    if quant=='radec':
        epoch, _, ra, raerr, dec, decerr, rho2, _ = candidate.iloc[i]
    elif quant=='seppa':
        epoch, _, sep, seperr, pa, paerr, rho, _ = candidate.iloc[i]
        ra, dec, raerr, decerr, rho2 = pol2car(sep,pa,seperr,paerr,corr=rho)
        if np.isnan(rho):
            rho2 = np.nan
    astrometry[0,i] += epoch+2400000.5
    astrometry[1,i] += ra
    astrometry[2,i] += dec
    astrometry[3,i] += raerr
    astrometry[4,i] += decerr
    astrometry[5,i] += rho2

# set variables
# TODO: include gaia errors within prior
# TODO: is there a way to account for gaia correlations?

rao = target_gaia['ra'][0] # deg
deco = target_gaia['dec'][0] # deg
pmrao = target_gaia['pmra'][0] # mas/yr
pmdeco = target_gaia['pmdec'][0] # mas/yr
paro = target_gaia['parallax'][0] # mas
radvelo = target_gaia['radial_velocity'][0]

epochs=astrometry[0]
ras=astrometry[1]
raserr=astrometry[3]
decs=astrometry[2]
decserr=astrometry[4]
rho=astrometry[5]

# initial estimate for background star scenario (best guesses)
# (manually looked at approximate offset from star with cosine compensation for
# dependence of RA on declination (RA is a singularity at the declination poles))
ra0 = rao-(ras[0])/1000/3600/np.cos(deco/180.*np.pi)
dec0 = deco-(decs[0])/1000/3600
pmra0 = pmrao # mas/yr
pmdec0 = pmdeco # mas/yr
par0 = paro/10 # mas
radvel0 = 0 # km/s

jd_tt = novas.julian_date(2022, 3, 20, 12.0) # date used as starting data for plots

jd_start, jd_end, number = ephem_open()
# if the novas_de405 package is installed this will load the ephemerids file,
# this will handle nutation, precession, gravitational lensing by (and barycenter motion induced by?) solar system bodies, etc.
# ephem_open("DE440.bin")
# if the ephemerid files are downloaded from the USNO ftp server the binary's can be directly accessed if placed in the same folder.

# https://ssd.jpl.nasa.gov/planets/eph_export.html https://ssd.jpl.nasa.gov/ftp/eph/planets/
# DE405 : Created May 1997; includes both nutations and librations.
# Referred to the International Celestial Reference Frame.
# Covers JED 2305424.50  (1599 DEC 09)  to  JED 2525008.50  (2201 FEB 20)

# define reference positions for host star (in this case fixed at the median Gaia DR3 parameters)
star1_gaia=novas.make_cat_entry(star_name="star1",catalog="HIP",star_num=1,ra=rao/15.,
                                dec=deco,pm_ra=pmrao,pm_dec= pmdeco,
                                parallax=paro,rad_vel=radvelo)
star1_icrs = novas.transform_cat(option = 1, incat = star1_gaia, date_incat = 2016, date_newcat = 2000,
                                newcat_id = 'HIP')
# this converts the Epoch from 2016 to 2000 following ICRS,
# not sure if app_star needs Epoch 2000 input. In any case we will evaluate targets at the same
# observing epoch and only look at offsets so any difference in coordinate reference should cancel out
# (might be important when including absolute astrometry).

def radecdist(day,ra,dec,pmra,pmdec,par): # for a single epoch, can be merged into radecdists

    star2_gaia=novas.make_cat_entry(star_name="star2",catalog="HIP",star_num=1,ra=ra/15.,
                                dec=dec,pm_ra=pmra,pm_dec=pmdec,
                                parallax=par,rad_vel=radvel0) # radial velocity is only important for nearby stars across large timespans
    star2_icrs = novas.transform_cat(option = 1, incat = star2_gaia, date_incat = 2016, date_newcat = 2000,
                                newcat_id = 'HIP')
    raa,deca=novas.app_star(day,star1_icrs) # evaluates apparent position at certain Epoch, given ICRS coordinates catalog, outputs decimal hours and degrees
    rab,decb=novas.app_star(day,star2_icrs)
    c_a=SkyCoord(raa*15.,deca,unit=("deg","deg"))
    c_b=SkyCoord(rab*15.,decb,unit=("deg","deg"))
    offset=c_a.spherical_offsets_to(c_b) # deltaRA and deltaDEC between targets
    position_x=offset[0].arcsecond*1000 # convert to mas
    position_y=offset[1].arcsecond*1000
    return position_x,position_y

def radecdists(days,ra,dec,pmra,pmdec,par=0): # for multiple epochs

    star2_gaia=novas.make_cat_entry(star_name="star2",catalog="HIP",star_num=1,ra=ra/15.,
                                dec=dec,pm_ra=pmra,pm_dec=pmdec,
                                parallax=par,rad_vel=radvel0)
    star2_icrs = novas.transform_cat(option = 1, incat = star2_gaia, date_incat = 2016, date_newcat = 2000,
                                newcat_id = 'HIP')
    posx=[]
    posy=[]
    for day in days:
        raa,deca=novas.app_star(day,star1_icrs)
        rab,decb=novas.app_star(day,star2_icrs)
        c_a=SkyCoord(raa*15.,deca,unit=("deg","deg"))
        c_b=SkyCoord(rab*15.,decb,unit=("deg","deg"))
        offset=c_a.spherical_offsets_to(c_b)
        position_x=offset[0].arcsecond*1000
        position_y=offset[1].arcsecond*1000
        posx.append(position_x)
        posy.append(position_y)

    return np.array(posx),np.array(posy)

print(radecdist(jd_tt,ra0,dec0,pmra0,pmdec0,par0))

def fmodel(theta):
    if len(theta) == 5:
        ra,dec,pmra,pmdec,par = theta # unpacking parameters
    else:
        ra,dec,pmra,pmdec=theta
        par=0
    xs,ys=radecdists(epochs,ra,dec,pmra,pmdec,par)
    return xs,ys

def loglike(theta):
    lp = log_prior(theta) # start with the log_prior
    if not np.isfinite(lp): # if the log prior is (+-)infinite return log(0)=-inf so it doesn't waste time calculating the log_likelihood
        print('bad lp', lp)
        return -np.inf
    return lp + log_likelihood(theta)

def log_likelihood(theta):
    xs,ys=fmodel(theta)
    gravity=~np.isnan(rho)

    # following equations with correlation terms taken from orbitize! routine _chi2_2x2cov (https://orbitize.readthedocs.io/en/latest/_modules/orbitize/lnlike.html)
    prop=-0.5 * np.sum((ras[~gravity] - xs[~gravity]) ** 2 / (raserr[~gravity]) ** 2  + np.log(2*np.pi*(raserr[~gravity]) ** 2)) -0.5 * np.sum((decs[~gravity] - ys[~gravity]) ** 2 / (decserr[~gravity]) ** 2  + np.log(2*np.pi*(decserr[~gravity]) ** 2))# return the log likelihood

    det_C=(raserr[gravity]**2)*(decserr[gravity]**2)*(1-rho[gravity]**2)
    covs=rho[gravity]*raserr[gravity]*decserr[gravity]
    chi2=((ras[gravity]-xs[gravity])**2*decserr[gravity]**2+(decs[gravity]-ys[gravity])**2*raserr[gravity]**2-2*(ras[gravity]-xs[gravity])*(decs[gravity]-ys[gravity])*covs)/det_C
    chi2=chi2+np.log(det_C)+2*np.log(2*np.pi)
    chi2=np.sum(-0.5*chi2)
    # ^^ block taken/adapted from Orbitize code!
    prop = prop + chi2
    return prop

def log_prior(theta): # parameters go in
    if len(theta) == 5:
        ra,dec,pmra,pmdec,par = theta # unpacking parameters
        if not ((ra0+5e-3> ra>ra0-5e-3) and (dec0+5e-3 > dec > dec0-5e-3) and (0<par) and (-100 < pmra < 100) and (-100 < pmdec < 100)): # if the uniform parameters dont fall within their bounds we return log(0)=-inf anyway to avoid the calculations, if needed only the par positive constraint can be left in.
            print('bad',ra, dec, pmra, pmdec, par)
            return -np.inf
        L=1.35e3 #astraatmadja+ 2016
        r=1./(par/1000.) # distance estimate in parsec for a given parallax is 1 over parallax in arcseconds, par is in mas
        lp_dist=np.log((1.0/(2*L**3))*(r**2))-r/L #bailer-jones https://www2.mpia-hd.mpg.de/~calj/parallax.pdf eq. 17, already normalized to integrate to 1 in linear space
    else:
        ra,dec,pmra,pmdec=theta
        if not ((ra0+5e-3> ra>ra0-5e-3) and (dec0+5e-3 > dec > dec0-5e-3) and (-100 < pmra < 100) and (-100 < pmdec < 100)): # if the uniform parameters dont fall within their bounds we return log(0)=-inf anyway to avoid the calculations
            print('bad',ra, dec, pmra, pmdec)
            return -np.inf
        lp_dist=0

    #lp_dist=np.log(1.0/(np.sqrt(2*np.pi)*sigma_par))-0.5*(par-mu_par)**2/sigma_par**2
    lp_pmra=np.log(1.0/(np.sqrt(2*np.pi)*sigma_pmra))-0.5*(pmra-mu_pmra)**2/sigma_pmra**2 # log probabilities of gaussian prior
    lp_pmdec=np.log(1.0/(np.sqrt(2*np.pi)*sigma_pmdec))-0.5*(pmdec-mu_pmdec)**2/sigma_pmdec**2

    return lp_dist+lp_pmra+lp_pmdec

def prior_transform(u):
    """Transforms samples `u` drawn from the unit cube to samples to those
    from our prior for each variable."""
    theta = np.array(u) # copy u
    if len(theta) == 5:
        ra,dec,pmra,pmdec,par = theta # unpacking parameters
        # par prior
        par *= (par0*10)
    else:
        ra,dec,pmra,pmdec=theta
    # ra prior
    ra = transform_uniform(ra, ra0-5e-3, ra0+5e-3)
    # dec prior
    dec = transform_uniform(dec, dec0-5e-3, dec0+5e-3)
    # pmra prior
    pmra = 2. * pmra - 1.
    pmra *= 100
    # pmdec prior
    pmdec = 2. * pmdec - 1.
    pmdec *= 100

    # TODO INVOLVE THE ACTUAL PRIOR IN THE PRIOR TRANSFORM NOT THE LOG-LIKE

    # L=1.35e3 #astraatmadja+ 2016
    # r=1./(par/1000.) # distance estimate in parsec for a given parallax is 1 over parallax in arcseconds, par is in mas
    # lp_dist=np.log((1.0/(2*L**3))*(r**2))-r/L #bailer-jones https://www2.mpia-hd.mpg.de/~calj/parallax.pdf eq. 17, already normalized to integrate to 1 in linear space

    # mu_pmra=-6.1 # parameters for independent gaussian proper motion priors. real distribution has longer tails and correlations
    # sigma_pmra=2.17
    # mu_pmdec=1.7 # given by Tomas Stolker based on Gaia query within 0.1 deg of target
    # sigma_pmdec=1.3
    #mu_par
    #sigma_par # if known, a Gaussian prior on the parallax for the background source can be given in a similar fashion as the pmra and pmdec
    #lp_dist=np.log(1.0/(np.sqrt(2*np.pi)*sigma_par))-0.5*(par-mu_par)**2/sigma_par**2
    # lp_pmra=np.log(1.0/(np.sqrt(2*np.pi)*sigma_pmra))-0.5*(pmra-mu_pmra)**2/sigma_pmra**2 # log probabilities of gaussian prior
    # lp_pmdec=np.log(1.0/(np.sqrt(2*np.pi)*sigma_pmdec))-0.5*(pmdec-mu_pmdec)**2/sigma_pmdec**2
    if len(theta) == 5:
        theta = ra,dec,pmra,pmdec,par
    else:
        theta = ra,dec,pmra,pmdec
    return theta



### RUN URASTAR ###

if __name__ == '__main__':

    npool = 4
    ndim = 5

    with dynesty.pool.Pool(npool, loglike, prior_transform) as pool: #where sampler is first created
        dsampler = dynesty.DynamicNestedSampler(pool.loglike, pool.prior_transform,
                                                ndim, pool=pool)
        dsampler.run_nested()
        res = dsampler.results

    from dynesty import plotting as dyplot
    # Plot results.
    fig, axes = dyplot.runplot(res)
    plt.savefig('{}_evidence_bgstar.png'.format(target_name.replace(' ','_')), dpi=300, bbox_inches='tight')
    labels = ["RA (deg, ep=2016)","DEC (deg, ep=2016)","pmra (mas/yr)","pmdec (mas/yr)","parallax (mas)"]
    # plot extended run (right)
    fg, ax = dyplot.cornerplot(res,
                               color='blue',
                               # truths=np.zeros(ndim),
                               # span=[(-4.5, 4.5) for i in range(ndim)],
                               labels=labels,
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

    from matplotlib.ticker import FuncFormatter

    def tickform(x, pos):
        # x:  tick value - ie. what you currently see in yticks
        # pos: a position - ie. the index of the tick (from 0 to 9 in this example)
        return '%.4f' % x

    tick_formatter = FuncFormatter(tickform)

    for i,axis_row in enumerate(ax):
        if i == 1:
            axis_row[0].yaxis.set_major_formatter(tick_formatter) # make left hand ticks behave
        for axis in axis_row:
            axis.set(xlabel=None, ylabel=None)
    for axis_col in ax[:,-1]:
        axis_col.xaxis.set_major_formatter(tick_formatter)
    ax[-1,-1].set_xscale('log')

    plt.savefig("{}_corner_bgstar.png".format(target_name.replace(' ','_')),dpi=300, bbox_inches='tight')


    # TODO: save results of a run, check if a run exists before rerunning
    from dynesty import utils as dyfunc

    results = res

    # Extract sampling results.
    samples = results.samples  # samples
    weights = np.exp(results.logwt - results.logz[-1])  # normalized weights

    # Compute 10%-90% quantiles.
    quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
                 for samps in samples.T]

    # Compute weighted mean and covariance.
    mean, cov = dyfunc.mean_and_cov(samples, weights)

    median = [dyfunc.quantile(samps, [0.5], weights=weights)
                 for samps in samples.T]

    # Resample weighted samples.
    samples_equal = dyfunc.resample_equal(samples, weights)

    # Generate a new set of results with sampling uncertainties.
    results_sim = dyfunc.resample_run(results)

    fig,axs = plt.subplot_mosaic(
        '''
        AABB
        AACC
        ''',
        figsize=(16,8))

    epochs2=np.arange(daysforward)+jd_tt-daysback # 4000 days of epochs to evaluate position at

    times=Time(epochs2,format='jd')
    times0=Time(epochs,format='jd')

    # flat_samples = results_sim.samples
    flat_samples = samples_equal
    sel = np.random.choice(np.arange(np.shape(flat_samples)[0]),10)
    pars = flat_samples[sel,:]
    best_pars = np.array(median).T[0]

    gravity=~np.isnan(rho) # gravity is everywhere where rho is not a nan.

    # plot stationary bg track at infinity (0 parallax)
    rasbg,decbg=radecdists(epochs2,best_pars[0],best_pars[1],0,0,0) # retrieve coordinates at full range of epochs

    axs['B'].plot(times.decimalyear,orbitize.system.radec2seppa(rasbg,decbg)[0],color='gray',alpha=1, zorder=3,ls='--')
    axs['C'].plot(times.decimalyear,orbitize.system.radec2seppa(rasbg,decbg)[1],color='gray',alpha=1, zorder=3,ls='--')

    axs['A'].plot(rasbg,decbg,color="gray",zorder=3,label="stationary bg",alpha=1,ls='--')

    # plot best moving bg track

    rasbg,decbg=radecdists(epochs2,best_pars[0],best_pars[1],best_pars[2],best_pars[3],best_pars[4]) # retrieve coordinates at full range of epochs

    axs['B'].plot(times.decimalyear,orbitize.system.radec2seppa(rasbg,decbg)[0],color='k',alpha=1, zorder=3)
    axs['C'].plot(times.decimalyear,orbitize.system.radec2seppa(rasbg,decbg)[1],color='k',alpha=1, zorder=3)

    axs['A'].plot(rasbg,decbg,color="k",zorder=3,label="best model",alpha=1)

    rasbg,decbg=radecdists(epochs,best_pars[0],best_pars[1],best_pars[2],best_pars[3],best_pars[4]) # retrieve coordinates at observing epochs

    axs['A'].plot(rasbg,decbg,color="red",alpha=1,linestyle="",marker=".", zorder=4)

    for i in range(len(ras[~gravity])):
        compra = (ras[~gravity][i],rasbg[~gravity][i])
        compdec = (decs[~gravity][i],decbg[~gravity][i])
        axs['A'].plot(compra,compdec,color='xkcd:pink', alpha=0.5)

    for i in np.arange(10): # for 10 randomly drawn parameter combinations
        rasbg,decbg=radecdists(epochs2,pars[i,0],pars[i,1],pars[i,2],pars[i,3],pars[i,4]) # retrieve coordinates at full range of epochs

        axs['B'].plot(times.decimalyear,orbitize.system.radec2seppa(rasbg,decbg)[0],color='blue',alpha=0.3, zorder=0)
        axs['C'].plot(times.decimalyear,orbitize.system.radec2seppa(rasbg,decbg)[1],color='blue',alpha=0.3, zorder=0)

        if i==0:
            axs['A'].plot(rasbg,decbg,color="blue",label="model",alpha=0.3, zorder=0)
        else:
            axs['A'].plot(rasbg,decbg,color="blue",alpha=0.3, zorder=0)

        rasbg,decbg=radecdists(epochs,pars[i,0],pars[i,1],pars[i,2],pars[i,3],pars[i,4]) # retrieve coordinates at observing epochs

        if i==0:
            axs['A'].plot(rasbg,decbg,color="red",label="model @ epochs",alpha=1,linestyle="",marker=".",zorder=4)
        else:
            axs['A'].plot(rasbg,decbg,color="red",alpha=1,linestyle="",marker=".",zorder=4)

    # plot data in deltaRA, deltaDEC plot
    axs['A'].errorbar(ras[~gravity],decs[~gravity],yerr=decserr[~gravity],xerr=raserr[~gravity],
                      color="xkcd:orange",label="YSES-2b data",linestyle="", zorder=5)
    axs['A'].errorbar(ras[gravity],decs[gravity],yerr=decserr[gravity],xerr=raserr[gravity],
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

    # plot the datapoints that are not gravity in the sep and PA plots, error calculation taken from Orbitize!
    sep,pa=orbitize.system.radec2seppa(ras[~gravity],decs[~gravity])
    sep_err=0.5*raserr[~gravity]+0.5*decserr[~gravity]
    pa_err=np.degrees(sep_err/sep)
    axs['B'].errorbar(times0.decimalyear[~gravity],sep,yerr=sep_err,color="xkcd:orange",linestyle="", zorder=5)
    axs['C'].errorbar(times0.decimalyear[~gravity],pa,yerr=pa_err,color="xkcd:orange",linestyle="", zorder=5)

    # plot the gravity datapoints with a conversion in the sep and PA plots
    sep,pa=orbitize.system.radec2seppa(ras[gravity],decs[gravity])
    for i in np.arange(np.sum(gravity)):
        sep_err,pa_err,rho2=orbitize.system.transform_errors(ras[gravity][i], decs[gravity][i],
                                                             raserr[gravity][i], decserr[gravity][i], rho[gravity][i], orbitize.system.radec2seppa)
        axs['B'].errorbar(times0.decimalyear[gravity][i],sep[i],yerr=sep_err,color="xkcd:orange",linestyle="",marker='.', zorder=5)
        axs['C'].errorbar(times0.decimalyear[gravity][i],pa[i],yerr=pa_err,color="xkcd:orange",linestyle="",marker='.', zorder=5)
    axs['A'].legend()

    plt.tight_layout()
    plt.savefig("{}_model_bgstar.png".format(target_name.replace(' ','_')), dpi=300, bbox_inches='tight')


    # check how unlikely the fit motion is by comparing the distribution to the nearby Gaia distribution

    nearby_table = pd.DataFrame(columns=['pmra','pmdec','parallax'])
    nearby_table['pmra'] = nearby['pmra'].data
    nearby_table['pmdec'] = nearby['pmdec'].data
    nearby_table['parallax'] = nearby['parallax'].data


    nearby_corner = corner.corner(nearby_table.dropna(), truths=[-24.64, 3.85, 1.50],
                                  quantiles=[0.003,0.997], levels=[0.68,0.95,0.997],
                                 )

    plt.savefig('{}_nearby_gaia_distribution.png'.format(target_name.replace(' ','_')), dpi=300, bbox_inches='tight')
