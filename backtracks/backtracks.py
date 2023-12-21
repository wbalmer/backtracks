# backtracks.py
# authors: Gilles Otten, William Balmer

# special packages needed: astropy, matplotlib, numpy, novas, novas_de405,
# dynesty, emcee, orbitize, corner (potentially cython and tqdm if clean pip install crashes)

# this code does an analysis with a background star model given delta RA and
# delta DEC datapoints of a point source wrt a host star
# comparing Bayes factor or BIC of this analysis and a pure Orbitize! run would
# be informative to see which model fits better for new discoveries.

import pickle
import sys
import warnings

from pathlib import Path
from typing import Optional, Tuple

import astropy.units as u
import dynesty
import novas.compat as novas
import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.io import fits
from astropy.time import Time
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from matplotlib.pyplot import Figure
from novas.compat.eph_manager import ephem_open

from schwimmbad import MPIPool

from backtracks.utils import pol2car, transform_gengamm, transform_normal, transform_uniform
from backtracks.plotting import diagnostic, neighborhood, plx_prior, posterior, trackplot


# Set the Gaia data release to DR3
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
# Retrieve all rows from a Gaia query
Gaia.ROW_LIMIT = -1

# TODO: is there a way to account for gaia correlations?


class System():
    """
    Class for describing a star system with a companion candidate.
    """

    def __init__(self, target_name: str, candidate_file: str, nearby_window: float = 0.5, fileprefix='./', **kwargs):
        self.target_name = target_name
        self.candidate_file = candidate_file
        self.fileprefix = fileprefix

        if 'unif' in kwargs:
            self.unif = kwargs['unif']
        else:
            self.unif = 5e-3

        if 'jd_tt' in kwargs:
            warnings.warn("The jd_tt parameter has been removed. "
                          "Please use the ref_epoch parameter of "
                          "generate_plots instead.")

        if Gaia.MAIN_GAIA_TABLE == "gaiadr3.gaia_source":
            self.gaia_release = "DR3"
        elif Gaia.MAIN_GAIA_TABLE == "gaiadr4.gaia_source":
            self.gaia_release = "DR4"
        else:
            raise ValueError("The value of the MAIN_GAIA_TABLE "
                             f"({Gaia.MAIN_GAIA_TABLE}) is not valid.")

        self.gaia_epoch = None

        # planet candidate astrometry
        candidate = pd.read_csv(candidate_file)

        astrometry = np.zeros((6, len(candidate))) # epoch, ra, dec, raerr, decerr, rho

        for i,quant in enumerate(candidate['quant_type']):
            if quant=='radec':
                epoch, ra, raerr, dec, decerr, rho2 = \
                    candidate.iloc[i, 0], candidate.iloc[i, 2], candidate.iloc[i, 3], \
                    candidate.iloc[i, 4], candidate.iloc[i, 5], candidate.iloc[i, 6]

            elif quant=='seppa':
                epoch, sep, seperr, pa, paerr, rho = \
                    candidate.iloc[i, 0], candidate.iloc[i, 2], candidate.iloc[i, 3], \
                    candidate.iloc[i, 4], candidate.iloc[i, 5], candidate.iloc[i, 6]

                ra, dec, raerr, decerr, rho2 = pol2car(sep, pa, seperr, paerr, corr=rho)

                if np.isnan(rho):
                    rho2 = np.nan

            astrometry[0, i] += epoch + 2400000.5  # MJD to JD
            astrometry[1, i] += ra
            astrometry[2, i] += dec
            astrometry[3, i] += raerr
            astrometry[4, i] += decerr
            astrometry[5, i] += rho2

        self.epochs = astrometry[0]
        self.ras = astrometry[1]
        self.raserr = astrometry[3]
        self.decs = astrometry[2]
        self.decserr = astrometry[4]
        self.rho = astrometry[5]

        if 'query_file' in kwargs and kwargs['query_file'] is not None:
            self.gaia_id = int(Path(kwargs['query_file']).stem.split('_')[-1])

            with fits.open(kwargs['query_file']) as hdu_list:
                target_gaia = Table(hdu_list[1].data, masked=True)
                self.nearby = Table(hdu_list[2].data, masked=True)

                for col in target_gaia.columns.values():
                    col.mask = np.isnan(col)

                for col in self.nearby.columns.values():
                    col.mask = np.isnan(col)

        else:
            self.nearby, self.gaia_id, target_gaia = self.query_astrometry(nearby_window)

            target_table = fits.BinTableHDU(target_gaia)
            nearby_table = fits.BinTableHDU(self.nearby)
            hdu_list = fits.HDUList([fits.PrimaryHDU(), target_table, nearby_table])
            hdu_list.writeto(f'{self.fileprefix}gaia_query_{self.gaia_id}.fits', overwrite=True)

        self.set_prior_attr()

        # set variables
        self.rao = target_gaia['ra'][0] # deg
        self.deco = target_gaia['dec'][0] # deg
        self.pmrao = target_gaia['pmra'][0] # mas/yr
        self.pmdeco = target_gaia['pmdec'][0] # mas/yr
        self.paro = target_gaia['parallax'][0] # mas
        self.radvelo = target_gaia['radial_velocity'][0] # km/s

        # initial estimate for background star scenario (best guesses)
        # (manually looked at approximate offset from star with cosine compensation for
        # dependence of RA on declination (RA is a singularity at the declination poles))
        self.ra0 = self.rao-(self.ras[0])/1000/3600/np.cos(self.deco/180.*np.pi)
        self.dec0 = self.deco-(self.decs[0])/1000/3600
        self.pmra0 = self.pmrao # mas/yr
        self.pmdec0 = self.pmdeco # mas/yr
        self.par0 = self.paro/10 # mas
        self.radvel0 = 0 # km/s

        jd_start, jd_end, number = ephem_open()
        print('[BACKTRACK INFO]: Opened ephemeris file')
        # if the novas_de405 package is installed this will load the ephemerids file,
        # this will handle nutation, precession, gravitational lensing by (and barycenter motion induced by?) solar system bodies, etc.
        # ephem_open("DE440.bin")
        # if the ephemerid files are downloaded from the USNO ftp server the binary's can be directly accessed if placed in the same folder.

        # https://ssd.jpl.nasa.gov/planets/eph_export.html https://ssd.jpl.nasa.gov/ftp/eph/planets/
        # DE405 : Created May 1997; includes both nutations and librations.
        # Referred to the International Celestial Reference Frame.
        # Covers JED 2305424.50  (1599 DEC 09)  to  JED 2525008.50  (2201 FEB 20)

        # define reference positions for host star (in this case fixed at the median Gaia parameters)
        self.host_cat = novas.make_cat_entry(star_name="host",catalog="HIP",star_num=1,ra=self.rao/15.,
                                             dec=self.deco,pm_ra=self.pmrao,pm_dec=self.pmdeco,
                                             parallax=self.paro,rad_vel=self.radvelo)
        print('[BACKTRACK INFO]: made cat entry for host')
        self.host_icrs = novas.transform_cat(option=1, incat=self.host_cat, date_incat=self.gaia_epoch,
                                             date_newcat=2000., newcat_id="HIP")
        print('[BACKTRACK INFO]: transformed cat entry for host')

        # this converts the Epoch from the Gaia ref_epoch (2016 for DR3) to 2000 following ICRS

    def query_astrometry(self, nearby_window: float = 0.5):
        # resolve target in simbad
        target_result_table = Simbad.query_object(self.target_name)
        print(f'[BACKTRACK INFO]: Resolved the target star \'{self.target_name}\' in Simbad!')
        # target_result_table.pprint()
        # get gaia ID from simbad
        gaia_id = None
        for target_id in Simbad.query_objectids(self.target_name)['ID']:
            if f'Gaia {self.gaia_release}' in target_id:
                gaia_id = int(target_id.replace(f'Gaia {self.gaia_release}', ''))
                print('[BACKTRACK INFO]: Resolved target\'s Gaia ID '
                      f'from Simbad, Gaia {self.gaia_release} {gaia_id}')

        if gaia_id is None:
            raise ValueError(f"The Gaia source ID for {self.target_name} "
                             f"is not found in the selected catalog "
                             f"({Gaia.MAIN_GAIA_TABLE}).")

        coord = SkyCoord(ra=target_result_table['RA'][0],
                         dec=target_result_table['DEC'][0],
                         unit=(u.hourangle, u.degree), frame='icrs')
        width = u.Quantity(50, u.arcsec)
        height = u.Quantity(50, u.arcsec)
        columns = ['source_id', 'ra', 'dec', 'pmra', 'pmdec', 'parallax', 'radial_velocity', 'ref_epoch']
        target_gaia = Gaia.query_object_async(coordinate=coord, width=width, height=height, columns=columns)
        target_gaia = target_gaia[target_gaia['source_id']==gaia_id]
        self.gaia_epoch = target_gaia['ref_epoch'][0]
        print(f'[BACKTRACK INFO]: gathered Gaia {self.gaia_release} data for {self.target_name}')
        print(f'   * Gaia source ID = {gaia_id}')
        print(f'   * Reference epoch = {self.gaia_epoch}')
        print(f'   * RA = {target_gaia["ra"][0]:.4f} deg')
        print(f'   * Dec = {target_gaia["dec"][0]:.4f} deg')
        print(f'   * PM RA = {target_gaia["pmra"][0]:.2f} mas/yr')
        print(f'   * PM Dec = {target_gaia["pmdec"][0]:.2f} mas/yr')
        print(f'   * Parallax = {target_gaia["parallax"][0]:.2f} mas')
        print(f'   * RV = {target_gaia["radial_velocity"][0]:.2f} km/s')

        # resolve nearby stars
        width = u.Quantity(nearby_window, u.deg)
        height = u.Quantity(nearby_window, u.deg)
        nearby = Gaia.query_object_async(coordinate=coord, width=width, height=height, columns=columns)
        print(rf'[BACKTRACK INFO]: gathered {len(nearby)} Gaia objects from the {nearby_window} sq. deg. nearby {self.target_name}')
        print('[BACKTRACK INFO]: Finished nearby background gaia statistics')

        # return table of nearby objects, target's gaia id, and table of target
        return nearby, gaia_id, target_gaia

    def set_prior_attr(self):
        # some statistics
        self.mu_pmra = np.ma.median(self.nearby['pmra'].data)
        self.sigma_pmra = np.ma.std(self.nearby['pmra'].data)

        self.mu_pmdec = np.ma.median(self.nearby['pmdec'].data)
        self.sigma_pmdec = np.ma.std(self.nearby['pmdec'].data)

        self.mu_par = np.ma.median(self.nearby['parallax'].data)
        self.sigma_par = np.ma.std(self.nearby['parallax'].data)

        # query Bailer-Jones distance parameters
        healpix = np.floor(self.gaia_id / 562949953421312)
        data_file = Path(__file__).parent.resolve() / 'bailer-jones_edr3.csv'
        distance_prior_params = pd.read_csv(data_file)
        distance_prior_params = distance_prior_params[distance_prior_params['healpix']==healpix]
        self.L = distance_prior_params['GGDrlen'].values[0]
        self.alpha = distance_prior_params['GGDalpha'].values[0]
        self.beta = distance_prior_params['GGDbeta'].values[0]

        print(f'[BACKTRACK INFO]: Queried distance prior parameters, L={self.L:.2f}, alpha={self.alpha:.2f}, beta={self.beta:.2f}')

    def radecdists(self, days, param): # for multiple epochs
        if len(param) == 4:
            ra, dec, pmra, pmdec = param
        else:
            ra, dec, pmra, pmdec, par = param

        jd_start, jd_end, number = ephem_open()

        star2_gaia = novas.make_cat_entry(star_name="BGR", catalog="HIP", star_num=2,
                                          ra=ra/15., dec=dec, pm_ra=pmra, pm_dec=pmdec,
                                          parallax=par, rad_vel=0)

        star2_icrs = novas.transform_cat(option=1, incat=star2_gaia, date_incat=self.gaia_epoch,
                                         date_newcat=2000., newcat_id="HIP")

        posx=[]
        posy=[]
        for i, day in enumerate(days):
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

    def fmodel(self, param):
        return self.radecdists(self.epochs, param)

    def loglike(self, param):
        """
        chi2 likelihood function.
        """

        if not np.isfinite(np.sum(param)):
            # if nan or inf, avoid calling fmodel
            return -np.inf

        xs, ys = self.fmodel(param)

        # separate terms where there is a correlation
        corr_terms = ~np.isnan(self.rho)

        like = 0.0

        if np.sum(~corr_terms) > 0:
            # calculate chi2 for uncorrelated terms
            like += -0.5 * np.sum((self.ras[~corr_terms] - xs[~corr_terms])**2 / self.raserr[~corr_terms]**2 + np.log(2.*np.pi*self.raserr[~corr_terms]**2))
            like += -0.5 * np.sum((self.decs[~corr_terms] - ys[~corr_terms])**2 / self.decserr[~corr_terms]**2 + np.log(2.*np.pi*self.decserr[~corr_terms]**2))

        if np.sum(corr_terms) > 0:
            # calculate the chi2 for correlated terms
            # following equations with correlation terms taken from orbitize! routine
            # _chi2_2x2cov (https://orbitize.readthedocs.io/en/latest/modules/orbitize/lnlike.html)
            det_C = (self.raserr[corr_terms]**2)*(self.decserr[corr_terms]**2)*(1-self.rho[corr_terms]**2)
            covs = self.rho[corr_terms]*self.raserr[corr_terms]*self.decserr[corr_terms]
            chi2 = ((self.ras[corr_terms]-xs[corr_terms])**2*self.decserr[corr_terms]**2+(self.decs[corr_terms]-ys[corr_terms])**2*self.raserr[corr_terms]**2-2*(self.ras[corr_terms]-xs[corr_terms])*(self.decs[corr_terms]-ys[corr_terms])*covs)/det_C
            chi2 += np.log(det_C)+2*np.log(2*np.pi)
            like += np.sum(-0.5*chi2)

        return like

    def prior_transform(self, u):
        """Transforms samples `u` drawn from the unit cube to samples
        to those from our prior for each variable. For parallax, we
        follow Bailer-Jones 2015, eq.17 and Astraatmadja+ 2016.
        """
        param = np.array(u) # copy u

        if len(param) == 5:
            ra, dec, pmra, pmdec, par = param # unpacking parameters

            # the PPF of Bailer-Jones 2015 eq. 17
            # 1.35e3 length scale value from astraatmadja+ 2016
            par = 1000/transform_gengamm(par, self.L, self.alpha, self.beta) # [units of mas]

            # truncate distribution at 100 kpc (Nielsen+ 2017 do this at 10 kpc)
            if par < 1e-2:
                par = -np.inf

        else:
            ra, dec, pmra, pmdec = param

        # uniform priors for RA and Dec
        ra = transform_uniform(ra, self.ra0-self.unif, self.ra0+self.unif)
        dec = transform_uniform(dec, self.dec0-self.unif, self.dec0+self.unif)

        # normal priors for proper motion
        pmra = transform_normal(pmra, self.mu_pmra, self.sigma_pmra)
        pmdec = transform_normal(pmdec, self.mu_pmdec, self.sigma_pmdec)

        if len(param) == 5:
            param = ra, dec, pmra, pmdec, par
        else:
            param = ra, dec, pmra, pmdec

        return param

    def fit(self, dlogz=0.5, npool=4, dynamic=False, nlive=200, mpi_pool=False, resume=False, sample_method='unif'):
        """
        """
        print('[BACKTRACK INFO]: Beginning sampling, I hope')
        ndim = 5

        if not mpi_pool:
            with dynesty.pool.Pool(npool, self.loglike, self.prior_transform) as pool:
                if dynamic:
                    if resume:
                        dsampler = dynesty.DynamicNestedSampler.restore(
                            fname=f'{self.fileprefix}dynesty.save',
                            pool=pool,
                        )

                    else:
                        dsampler = dynesty.DynamicNestedSampler(
                            loglikelihood=pool.loglike,
                            prior_transform=pool.prior_transform,
                            ndim=ndim,
                            pool=pool,
                            sample=sample_method,
                        )

                    dsampler.run_nested(
                        dlogz_init=dlogz,
                        nlive_init=nlive,
                        checkpoint_file=f'{self.fileprefix}dynesty.save',
                        resume=resume,
                    )

                else:
                    if resume:
                        dsampler = dynesty.NestedSampler.restore(
                            fname=f'{self.fileprefix}dynesty.save',
                            pool=pool,
                        )

                    else:
                        dsampler = dynesty.NestedSampler(
                            loglikelihood=pool.loglike,
                            prior_transform=pool.prior_transform,
                            ndim=ndim,
                            pool=pool,
                            nlive=nlive,
                            sample=sample_method,
                        )

                    dsampler.run_nested(
                        dlogz=dlogz,
                        checkpoint_file=f'{self.fileprefix}dynesty.save',
                        resume=resume,
                    )

        else:
            pool = MPIPool()

            if not pool.is_master():
                pool.wait()
                sys.exit(0)

            if dynamic:
                if resume:
                    dsampler = dynesty.DynamicNestedSampler.restore(
                        fname=f'{self.fileprefix}dynesty.save',
                        pool=pool,
                    )

                else:
                    dsampler = dynesty.DynamicNestedSampler(
                        loglikelihood=self.loglike,
                        prior_transform=self.prior_transform,
                        ndim=ndim,
                        pool=pool,
                        sample=sample_method,
                    )

                dsampler.run_nested(
                    dlogz_init=dlogz,
                    nlive_init=nlive,
                    checkpoint_file=f'{self.fileprefix}dynesty.save',
                    resume=resume,
                )

            else:
                if resume:
                    dsampler = dynesty.NestedSampler.restore(
                        fname=f'{self.fileprefix}dynesty.save',
                        pool=pool,
                    )

                else:
                    dsampler = dynesty.NestedSampler(
                        loglikelihood=self.loglike,
                        prior_transform=self.prior_transform,
                        ndim=ndim,
                        pool=pool,
                        nlive=nlive,
                        sample=sample_method,
                    )

                dsampler.run_nested(
                    dlogz=dlogz,
                    checkpoint_file=f'{self.fileprefix}dynesty.save',
                    resume=resume,
                )

        # Object with sampling results
        self.results = dsampler.results

        # Extract samples with equal weights
        samples = self.results.samples_equal()

        # Compute median, and 16th and 84th percentiles
        self.run_median = np.median(samples, axis=0)
        self.run_quant = np.quantile(samples, [0.16, 0.84], axis=0)

        return self.results

    def save_results(self, fileprefix='./'):
        save_dict = {'med': self.run_median, 'quant': self.run_quant, 'results': self.results}
        target_label = self.target_name.replace(' ','_')
        file_name = f'{fileprefix}{target_label}_dynestyrun_results.pkl'
        pickle.dump(save_dict, open(file_name, "wb"))

    def load_results(self, fileprefix: str = './'):
        target_label = self.target_name.replace(' ', '_')
        file_name = f'{fileprefix}{target_label}_dynestyrun_results.pkl'
        save_dict = pickle.load(open(file_name, "rb"))
        self.run_median = save_dict['med']
        self.run_quant = save_dict['quant']
        self.results = save_dict['results']

    def generate_plots(
            self,
            days_backward: float = 3.*365.,
            days_forward: float = 3.*365.,
            step_size: float = 10.,
            ref_epoch: Optional[Tuple[int, int, int]] = None,
            plot_radec: bool = False,
            plot_stationary: bool = False,
            fileprefix: str = './',
            filepost: str = '.pdf',
        ) -> Tuple[Figure, Figure, Figure, Figure, Figure]:
        """
        """
        if ref_epoch is None:
            mean_epoch = np.floor(np.mean(self.epochs))
            ref_epoch = Time(mean_epoch, format='jd')

        else:
            ref_epoch = Time(f'{ref_epoch[0]}-{ref_epoch[1]}-{ref_epoch[2]}T12:00:00', format='isot')

        fig_track = trackplot(
            self,
            ref_epoch=ref_epoch.jd,
            days_backward=days_backward,
            days_forward=days_forward,
            step_size=step_size,
            plot_radec=plot_radec,
            plot_stationary=plot_stationary,
            fileprefix=fileprefix,
            filepost=filepost
        )

        fig_post = posterior(self, fileprefix=fileprefix, filepost=filepost)
        fig_diag = diagnostic(self, fileprefix=fileprefix, filepost=filepost)
        fig_prior = plx_prior(self, fileprefix=fileprefix, filepost=filepost)
        fig_hood = neighborhood(self, fileprefix=fileprefix, filepost=filepost)

        return fig_track, fig_post, fig_diag, fig_prior, fig_hood
