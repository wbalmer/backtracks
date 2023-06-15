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

# target_name = "HD 131399 A"
# candidate_file = "scorpions1b_orbitizelike.csv"

from backtrack.utils import *
from backtrack.plotting import *

# TODO: CLASS STRUCTURE AND USER INPUTS
# TODO: include gaia errors within prior
# TODO: is there a way to account for gaia correlations?

class backtrack():
    """
    """

    def __init__(self,target_name, candidate_file, nearby_window=0.5, **kwargs):
        self.target_name = target_name
        self.candidate_file = candidate_file

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

        self.epochs = astrometry[0]
        self.ras = astrometry[1]
        self.raserr = astrometry[3]
        self.decs = astrometry[2]
        self.decserr = astrometry[4]
        self.rho = astrometry[5]

        self.nearby = self.query_astrometry(nearby_window)

        # initial estimate for background star scenario (best guesses)
        # (manually looked at approximate offset from star with cosine compensation for
        # dependence of RA on declination (RA is a singularity at the declination poles))
        self.ra0 = self.rao-(self.ras[0])/1000/3600/np.cos(self.deco/180.*np.pi)
        self.dec0 = self.deco-(self.decs[0])/1000/3600
        self.pmra0 = self.pmrao # mas/yr
        self.pmdec0 = self.pmdeco # mas/yr
        self.par0 = self.paro/10 # mas
        self.radvel0 = 0 # km/s

        self.jd_tt = novas.julian_date(2022, 3, 20, 12.0) # date used as starting data for plots
        print('[BACKTRACK INFO]: JD_TT set')
        jd_start, jd_end, number = ephem_open()
        print('[BACKTRACK INFO]: Opened ephemrids file')
        # if the novas_de405 package is installed this will load the ephemerids file,
        # this will handle nutation, precession, gravitational lensing by (and barycenter motion induced by?) solar system bodies, etc.
        # ephem_open("DE440.bin")
        # if the ephemerid files are downloaded from the USNO ftp server the binary's can be directly accessed if placed in the same folder.

        # https://ssd.jpl.nasa.gov/planets/eph_export.html https://ssd.jpl.nasa.gov/ftp/eph/planets/
        # DE405 : Created May 1997; includes both nutations and librations.
        # Referred to the International Celestial Reference Frame.
        # Covers JED 2305424.50  (1599 DEC 09)  to  JED 2525008.50  (2201 FEB 20)

        # define reference positions for host star (in this case fixed at the median Gaia DR3 parameters)
        self.host_cat = novas.make_cat_entry(star_name="host",catalog="HIP",star_num=1,ra=self.rao/15.,
                                             dec=self.deco,pm_ra=self.pmrao,pm_dec=self.pmdeco,
                                             parallax=self.paro,rad_vel=self.radvelo)
        print('[BACKTRACK INFO]: made cat entry for host')
        self.host_icrs = novas.transform_cat(option=1, incat=self.host_cat, date_incat=2016., date_newcat=2000.,
                                             newcat_id="HIP")
        print('[BACKTRACK INFO]: transformed cat entry for host')

        # this converts the Epoch from 2016 to 2000 following ICRS,
        # not sure if app_star needs Epoch 2000 input. In any case we will evaluate targets at the same
        # observing epoch and only look at offsets so any difference in coordinate reference should cancel out
        # (might be important when including absolute astrometry).


    def query_astrometry(self,nearby_window=0.5):
        # resolve target in simbad
        self.target_result_table = Simbad.query_object(self.target_name)
        print('[BACKTRACK INFO]: Resolved the target star \'{}\' in Simbad!'.format(self.target_name))
        # target_result_table.pprint()
        # get gaia ID from simbad
        for ID in Simbad.query_objectids(self.target_name)['ID']:
            if 'Gaia DR3' in ID:
                self.gaia_id = int(ID.replace('Gaia DR3', ''))
                print('[BACKTRACK INFO]: Resolved target\'s Gaia ID from Simbad, Gaia DR3',self.gaia_id)

        coord = SkyCoord(ra=self.target_result_table['RA'][0],
                         dec=self.target_result_table['DEC'][0],
                         unit=(u.hourangle, u.degree), frame='icrs')
        width = u.Quantity(50, u.arcsec)
        height = u.Quantity(50, u.arcsec)
        Gaia.ROW_LIMIT = -1
        target_gaia = Gaia.query_object_async(coordinate=coord, width=width, height=height)
        target_gaia = target_gaia[target_gaia['source_id']==self.gaia_id]
        # set variables
        self.rao = target_gaia['ra'][0] # deg
        self.deco = target_gaia['dec'][0] # deg
        self.pmrao = target_gaia['pmra'][0] # mas/yr
        self.pmdeco = target_gaia['pmdec'][0] # mas/yr
        self.paro = target_gaia['parallax'][0] # mas
        self.radvelo = target_gaia['radial_velocity'][0]

        print('[BACKTRACK INFO]: gathered target Gaia data')
        # resolve nearby stars
        width = u.Quantity(nearby_window, u.deg)
        height = u.Quantity(nearby_window, u.deg)
        Gaia.ROW_LIMIT = -1
        nearby = Gaia.query_object_async(coordinate=coord, width=width, height=height)
        print(r'[BACKTRACK INFO]: gathered {} Gaia objects from the {} sq. deg. nearby {}'.format(len(nearby), nearby_window, self.target_name))

        # some statistics
        self.mu_pmra = np.ma.median(nearby['pmra'].data)
        self.sigma_pmra = np.ma.std(nearby['pmra'].data)

        self.mu_pmdec = np.ma.median(nearby['pmdec'].data)
        self.sigma_pmdec = np.ma.std(nearby['pmdec'].data)

        self.mu_par = np.ma.median(nearby['parallax'].data)
        self.sigma_par = np.ma.std(nearby['parallax'].data)
        print('[BACKTRACK INFO]: Finished nearby background gaia statistics')

        # query Bailer-Jones distance parameters
        healpix = np.floor(self.gaia_id / 562949953421312 )
        distance_prior_params = pd.read_csv('bailer-jones_edr3_prior_summary.csv')
        distance_prior_params = distance_prior_params[distance_prior_params['healpix']==healpix]
        self.L = distance_prior_params['GGDrlen'].values[0]
        self.alpha = distance_prior_params['GGDalpha'].values[0]
        self.beta = distance_prior_params['GGDbeta'].values[0]
        print('[BACKTRACK INFO]: Queried distance prior parameters, L={}, alpha={}, beta={}'.format(self.L, self.alpha, self.beta))
        # return table of nearby objects
        return nearby

    def radecdists(self,days,ra,dec,pmra,pmdec,par): # for multiple epochs
        jd_start, jd_end, number = ephem_open()
        star2_gaia = novas.make_cat_entry(star_name="BGR", catalog="HIP", star_num=2,
                                        ra=ra/15., dec=dec, pm_ra=pmra, pm_dec=pmdec,
                                        parallax=par, rad_vel=1)
        star2_icrs = novas.transform_cat(option=1, incat=star2_gaia, date_incat=2016., date_newcat=2000.,
                                         newcat_id="HIP")
        posx=[]
        posy=[]
        for i,day in enumerate(days):
            raa,deca = novas.app_star(day,self.host_icrs)
            rab,decb = novas.app_star(day,star2_icrs)
            c_a=SkyCoord(raa*15.,deca,unit=("deg","deg"))
            c_b=SkyCoord(rab*15.,decb,unit=("deg","deg"))
            offset=c_a.spherical_offsets_to(c_b)
            position_x=offset[0].arcsecond*1000
            position_y=offset[1].arcsecond*1000
            posx.append(position_x)
            posy.append(position_y)
        return np.array(posx),np.array(posy)


    def fmodel(self,theta):
        if len(theta) == 5:
            ra, dec, pmra, pmdec, par = theta # unpacking parameters
        else:
            ra, dec, pmra, pmdec = theta
            par = 0
        xs,ys = self.radecdists(self.epochs, ra, dec, pmra, pmdec, par)
        return xs,ys


    def loglike(self,theta):
        """
        chi2 likelihood function.
        """

        if np.isnan(np.sum(theta)) or np.isinf(np.sum(theta)):
            # if nan or inf, avoid calling fmodel
            return -np.inf

        xs,ys = self.fmodel(theta)

        # separate terms where there is a correlation
        corr_terms = ~np.isnan(self.rho)
        # calculate chi2 for uncorrelated terms
        like = -0.5 * np.sum((self.ras[~corr_terms] - xs[~corr_terms]) ** 2 / (self.raserr[~corr_terms]) ** 2  + np.log(2*np.pi*(self.raserr[~corr_terms]) ** 2)) -0.5 * np.sum((self.decs[~corr_terms] - ys[~corr_terms]) ** 2 / (self.decserr[~corr_terms]) ** 2  + np.log(2*np.pi*(self.decserr[~corr_terms]) ** 2))
        if ~corr_terms.all():
            return like
        else:
            # following equations with correlation terms taken from orbitize! routine
            # _chi2_2x2cov (https://orbitize.readthedocs.io/en/latest/_modules/orbitize/lnlike.html)
            # calculate the chi2 for correlated terms
            det_C = (self.raserr[corr_terms]**2)*(self.decserr[corr_terms]**2)*(1-self.rho[corr_terms]**2)
            covs = self.rho[corr_terms]*self.raserr[corr_terms]*self.decserr[corr_terms]
            chi2 = ((self.ras[corr_terms]-xs[corr_terms])**2*self.decserr[corr_terms]**2+(self.decs[corr_terms]-ys[corr_terms])**2*self.raserr[corr_terms]**2-2*(self.ras[corr_terms]-xs[corr_terms])*(self.decs[corr_terms]-ys[corr_terms])*covs)/det_C
            chi2 = chi2+np.log(det_C)+2*np.log(2*np.pi)
            chi2 = np.sum(-0.5*chi2)
            # add uncorrelated and correlated likelihoods
            like += chi2
            return like


    def prior_transform(self, u):
        """Transforms samples `u` drawn from the unit cube to samples to those
        from our prior for each variable.

        For parallax, we follow Bailer-Jones 2015, eq.17 and Astraatmadja+ 2016

        """
        theta = np.array(u) # copy u
        if len(theta) == 5:
            ra, dec, pmra, pmdec, par = theta # unpacking parameters

            # par prior
            L = self.L # 1.35e3 length scale value from astraatmadja+ 2016
            alpha = self.alpha # 1
            beta = self.alpha # 2
            # the PPF of Bailer-Jones 2015 eq. 17
            par = 1000/transform_gengamm(par, L, alpha, beta) # [units of mas]
            # truncate distribution at 100 kpc (Nielsen+ 2017 do this at 10 kpc)
            if par < 1e-2:
                par = -np.inf

        else:
            ra, dec, pmra, pmdec = theta
        # uniform ra prior
        ra = transform_uniform(ra, self.ra0-5e-3, self.ra0+5e-3)
        # uniform dec prior
        dec = transform_uniform(dec, self.dec0-5e-3, self.dec0+5e-3)
        # uniform pmra prior
        # pmra = 2. * pmra - 1.
        # pmra *= 100
        # # uniform pmdec prior
        # pmdec = 2. * pmdec - 1.
        # pmdec *= 100

        pmra = transform_normal(pmra, self.mu_pmra, self.sigma_pmra)
        pmdec = transform_normal(pmdec, self.mu_pmdec, self.sigma_pmdec)

        if len(theta) == 5:
            theta = ra,dec,pmra,pmdec,par
        else:
            theta = ra,dec,pmra,pmdec
        return theta

    def fit(self,npool=4):
        """
        """
        print('[BACKTRACK INFO]: Beginning sampling, I hope')
        with dynesty.pool.Pool(npool, self.loglike, self.prior_transform) as pool: #where sampler is first created
            dsampler = dynesty.DynamicNestedSampler(pool.loglike, pool.prior_transform,
                                                    5, pool=pool)
            dsampler.run_nested()
            results = dsampler.results

        from dynesty import utils as dyfunc

        # Extract sampling results.
        samples = results.samples  # samples
        weights = np.exp(results.logwt - results.logz[-1])  # normalized weights

        # Compute weighted mean and covariance.
        mean, cov = dyfunc.mean_and_cov(samples, weights)

        self.run_median = [dyfunc.quantile(samps, [0.5], weights=weights)
                     for samps in samples.T]

        self.run_quant = [dyfunc.quantile(samps, [0.32, 0.68], weights=weights)
                     for samps in samples.T]

        # Resample weighted samples.
        samples_equal = dyfunc.resample_equal(samples, weights)

        # Generate a new set of results with sampling uncertainties.
        results_sim = dyfunc.resample_run(results)

        self.results = results_sim
        return results_sim

    def generate_plots(self,daysback=2600,daysforward=1200,fileprefix='./tests/'):
        """
        """
        diagnos = diagnostic(self)
        plx_prior(self)
        post = posterior(self)
        tracks = trackplot(self,daysback=daysback,daysforward=daysforward)
        hood = neighborhood(self)
