# backtracks.py
# authors: Gilles Otten, William Balmer, Tomas Stolker

# special packages needed: astropy, matplotlib, numpy, novas, novas_de405,
# dynesty, emcee, corner (potentially cython and tqdm if clean pip install crashes)

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
from scipy.optimize import minimize
import dynesty
import novas.compat as novas
import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord, Distance
from astropy.table import Table
from astropy.io import fits
from astropy.time import Time
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from matplotlib.pyplot import Figure
from novas.compat.eph_manager import ephem_open

from schwimmbad import MPIPool

from backtracks.utils import pol2car, transform_gengamm, transform_normal, transform_uniform, utc2tt, HostStarPriors
from backtracks.plotting import diagnostic, neighborhood, plx_prior, posterior, trackplot, stationtrackplot


# Set the Gaia data release to DR3
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
# Retrieve all rows from a Gaia query
Gaia.ROW_LIMIT = -1

class System():
    """
    Class for describing a star system with a companion candidate.

    """

    def __init__(self, target_name: str, candidate_file: str, nearby_window: float = 0.5, fileprefix = './', ndim = 11, **kwargs):
        """
        Args:
            target_name (str): Target name which will be resolved by SIMBAD into Gaia DR3 ID
            candidate_file (str): .csv file containing the to be test companion coordinates in Orbitize! format
            nearby_window (float): Default 0.5 [degrees]
            fileprefix (str): Prefix to filename. Default "./" for current folder.
            ndim (int): Number of dimensions that need to be fit. Default: 11
            **kwargs: additional keyword arguments
            rv_host_method (str): {'normal','uniform'} Uses Gaia DR3 retrieved RV and normal distribution as Host star RV prior if not defined.
            rv_host_params (tuple of floats): (lower_limit,upper_limit) for uniform rv_host_method, (mu,sigma) for normal rv_host_method. [km/s]
            unif (float): Sets bounds for uniform prior around estimated companion location. Defaults to 5e-3 if not defined. [degrees]
            ref_epoch_idx (int): ID of datapoint at which to pin stationary tracks. Defaults to 0 if not defined. Follows order of candidate_file datapoints.
        """

        self.target_name = target_name
        self.candidate_file = candidate_file
        self.fileprefix = fileprefix
        self.ndim = ndim

        if 'unif' in kwargs:
            self.unif = kwargs['unif']
        else:
            self.unif = 5e-3
        if 'rv_host_method' in kwargs:
             if 'rv_host_params' not in kwargs:
                 raise Exception("'rv_host_method' is set. Please provide (mu,sigma) or (lower,upper) in km/s in 'rv_host_params'")

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
        self.epochs_tt = utc2tt(astrometry[0])
        self.ras = astrometry[1]
        self.raserr = astrometry[3]
        self.decs = astrometry[2]
        self.decserr = astrometry[4]
        self.rho = astrometry[5]

        # choose time of reference observation of relative candidate position (defaults to the first observation)
        if 'ref_epoch_idx' in kwargs:
            self.ref_epoch_idx = kwargs['ref_epoch_idx']
            self.ref_epoch = self.epochs[self.ref_epoch_idx]
        else:
            self.ref_epoch_idx = 0
            self.ref_epoch = self.epochs[self.ref_epoch_idx]

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

        sig_ra = target_gaia['ra_error'][0]*u.mas.to(u.deg) # mas
        sig_dec = target_gaia['dec_error'][0]*u.mas.to(u.deg)
        sig_pmra = target_gaia['pmra_error'][0]
        sig_pmdec = target_gaia['pmdec_error'][0]
        ra_dec_corr = target_gaia['ra_dec_corr'][0]
        ra_parallax_corr = target_gaia['ra_parallax_corr'][0]
        ra_pmra_corr = target_gaia['ra_pmra_corr'][0]
        ra_pmdec_corr = target_gaia['ra_pmdec_corr'][0]
        dec_parallax_corr = target_gaia['dec_parallax_corr'][0]
        dec_pmra_corr = target_gaia['dec_pmra_corr'][0]
        dec_pmdec_corr = target_gaia['dec_pmdec_corr'][0]
        parallax_pmra_corr = target_gaia['parallax_pmra_corr'][0]
        parallax_pmdec_corr = target_gaia['parallax_pmdec_corr'][0]
        pmra_pmdec_corr = target_gaia['pmra_pmdec_corr'][0]
        sig_parallax = target_gaia['parallax_error'][0]
        if 'rv_host_method' in kwargs:
            if kwargs['rv_host_method'].lower() == 'uniform':
                self.rv_host_method='uniform'
                self.rv_lower,self.rv_upper=kwargs['rv_host_params'] # e.g., rv_host_params=(-10,10)
                self.radvelo = np.mean([self.rv_lower,self.rv_upper])
            elif kwargs['rv_host_method'].lower() == 'normal':
                self.rv_host_method='normal'
                self.radvelo,self.sig_rv=kwargs['rv_host_params'] # e.g., rv_host_params=(10,0.5)
        else: # if not set, it is the default gaia
            try:
                self.rv_host_method='gaia'
                self.radvelo = target_gaia['radial_velocity'][0] # km/s
                self.sig_rv = target_gaia['radial_velocity_error'][0]
                if isinstance(self.radvelo, (int, float)) and not isinstance(self.radvelo, bool) == False:
                    raise Exception("No valid Gaia RV, please change rv_host_method to 'normal' or 'uniform' and set rv_host_params to override")
            except:
                raise Exception("No valid Gaia RV, please change rv_host_method to 'normal' or 'uniform' and set rv_host_params to override.")

        self.host_mean = np.r_[self.rao,self.deco,self.pmrao,self.pmdeco,self.paro]
        self.host_cov = np.array([[sig_ra**2,ra_dec_corr*sig_ra*sig_dec,ra_pmra_corr*sig_ra*sig_pmra,sig_ra*sig_pmdec*ra_pmdec_corr,sig_ra*sig_parallax*ra_parallax_corr],
        [ra_dec_corr*sig_ra*sig_dec,sig_dec**2,sig_dec*sig_pmra*dec_pmra_corr,sig_dec*sig_pmdec*dec_pmdec_corr,sig_dec*sig_parallax*dec_parallax_corr],
        [ra_pmra_corr*sig_ra*sig_pmra,dec_pmra_corr*sig_pmra*sig_dec,sig_pmra**2,sig_pmra*sig_pmdec*pmra_pmdec_corr,sig_pmra*sig_parallax*parallax_pmra_corr],[sig_pmdec*sig_ra*ra_pmdec_corr,sig_pmdec*sig_dec*dec_pmdec_corr,sig_pmdec*sig_pmra*pmra_pmdec_corr,sig_pmdec**2,sig_pmdec*sig_parallax*parallax_pmdec_corr],[sig_parallax*sig_ra*ra_parallax_corr,sig_parallax*sig_dec*dec_parallax_corr,sig_parallax*sig_pmra*parallax_pmra_corr,sig_parallax*sig_pmdec*parallax_pmdec_corr,sig_parallax**2]])
        self.HostStarPriors = HostStarPriors(self.host_mean,self.host_cov)

        # set inital guess for position at gaia epoch
        self.set_initial_position()

        # create parameter set for stationary case
        # params = ra, dec, pmra, pmdec, par, ra_host, dec_host, pmra_host, pmdec_host, par_host, rv_host
        self.stationary_params = [self.ra0, self.dec0, 0, 0, 0, self.rao, self.deco, self.pmrao, self.pmdeco, self.paro, self.radvelo]
        # Compute useful chi2 value
        self.stationary_loglike = self.loglike(self.stationary_params)
        self.stationary_chi2_red = -2.*self.stationary_loglike/((2*(len(self.epochs)-1))-self.ndim)

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

        # define reference positions for host star (in this case fixed at the median Gaia parameters), this is superseded in 11-dimension model
        if self.ndim <= 5: #
            self.host_cat = novas.make_cat_entry(star_name="host",catalog="HIP",star_num=1,ra=self.rao/15.,
                                             dec=self.deco,pm_ra=self.pmrao,pm_dec=self.pmdeco,
                                             parallax=self.paro,rad_vel=self.radvelo)
            print('[BACKTRACK INFO]: made cat entry for host')
            self.host_icrs = novas.transform_cat(option=1, incat=self.host_cat, date_incat=self.gaia_epoch,
                                             date_newcat=2000., newcat_id="HIP")
            print('[BACKTRACK INFO]: transformed cat entry for host')

        # this converts the Epoch from the Gaia ref_epoch (2016 for DR3) to 2000 following ICRS

    def set_initial_position(self):
        """
        Calculates and sets attributes for the RA and DEC coordinates at ICRS Epoch 2016.0 assuming the object is a stationary background star.
        """

        # initial estimate for background star scenario (best guesses)
        print(f'[BACKTRACK INFO]: Estimating candidate position if stationary in RA,Dec @ {self.gaia_epoch} from observation #'+str(self.ref_epoch_idx))
        # we'll do a rough estimate using astropy, then minimize the distance between
        # the novas projection and the specified observation using scipy's BFGS with
        # RA,DEC @ Gaia epoch as free parameters.

        # educated guess with SkyCoord transformations
        init_host_coord = SkyCoord(ra=self.rao*u.deg, dec=self.deco*u.deg, distance=Distance(parallax=self.paro*u.mas), pm_ra_cosdec=self.pmrao*u.mas/u.yr, pm_dec=self.pmdeco*u.mas/u.yr, obstime=Time(self.gaia_epoch,format='jyear',scale='tcb'))
        # propagate host position from Gaia epoch (2016.0 for DR3) to reference observation epoch
        init_host_coord_at_obs = init_host_coord.apply_space_motion(Time(self.ref_epoch, format='jd'))
        # apply measurement of relative astrometry to get candidate in absolute coordinates at time of observation
        init_cand_coord_at_obs = init_host_coord_at_obs.spherical_offsets_by(self.ras[self.ref_epoch_idx] * u.mas, self.decs[self.ref_epoch_idx] * u.mas)
        init_cand_coord = SkyCoord(ra=init_cand_coord_at_obs.ra, dec=init_cand_coord_at_obs.dec, distance=Distance(parallax=self.paro*u.mas), pm_ra_cosdec=self.pmrao*u.mas/u.yr, pm_dec=self.pmdeco*u.mas/u.yr, obstime=Time(self.ref_epoch, format='jd'))
        # propagate candidate position from epoch of observation to Gaia epoch
        tdiff = Time(self.ref_epoch, format='jd').decimalyear-self.gaia_epoch
        init_cand_coord_at_gaia = init_cand_coord.apply_space_motion(Time(self.gaia_epoch+tdiff,format='jyear',scale='tcb'))
        ra0 = init_cand_coord_at_gaia.ra.value
        dec0 = init_cand_coord_at_gaia.dec.value

        def dummy_loglike(radec):
            """
            Minimization function for distance to observed offset assuming stationary background scenario.
            Minimization of the distance provides RA and DEC at Epoch 2016.0 assuming stationary background star.

            Returns:
                Distance between observed offset at reference epoch versus predicted offset
            """

            ra, dec = radec
            dummy_param = ra, dec, 0, 0, 0, self.rao, self.deco, self.pmrao, self.pmdeco, self.paro, self.radvelo
            xs, ys = self.radecdists([utc2tt(self.ref_epoch)], dummy_param)

            distance = np.linalg.norm(np.array([self.ras[self.ref_epoch_idx],self.decs[self.ref_epoch_idx]])-np.array([xs[0],ys[0]]))

            return distance

        # minimize distance between these points (bounds are 5e-5 degrees, i.e. 180 mas). The bounds might lead to an issue near the poles.
        bound = 5e-5
        init_result = minimize(dummy_loglike, np.array([ra0,dec0]), tol=1e-15, method='Nelder-Mead', bounds=((ra0-bound, ra0+bound), (dec0-bound, dec0+bound)))

        # not sure which of these is used anymore
        self.ra0 = init_result.x[0] # degrees
        self.dec0 = init_result.x[1] # degrees
        self.pmra0 = self.pmrao # mas/yr
        self.pmdec0 = self.pmdeco # mas/yr
        self.par0 = self.paro/10 # mas
        self.radvel0 = 0 # km/s


    def query_astrometry(self, nearby_window: float = 0.5):
        """
        Queries Simbad for Gaia DR3 stars properties within a certain angular search box around the host star, as well as the properties of the host star itself.

        Args:
            nearby_window (float): Dimensions of search box around host star to find background star population. Default 0.5 [degrees]

        Returns:
            tuple of tables. nearby star properties, host star ID and host star properties.
        """

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
        columns = ['source_id', 'ra', 'dec', 'pmra', 'pmdec', 'parallax', 'radial_velocity', 'ref_epoch','ra_error','dec_error','parallax_error','pmra_error','pmdec_error','radial_velocity_error','ra_dec_corr','ra_parallax_corr','ra_pmra_corr','ra_pmdec_corr','dec_parallax_corr','dec_pmra_corr','dec_pmdec_corr','parallax_pmra_corr','parallax_pmdec_corr','pmra_pmdec_corr']
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
        """
        Gathers and sets attributes for the priors for background star proper motion (from neighbourhood statistics) and distance (from healpix).
        """

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
        """
        Function that calculates the offset between companion and host star at a certain set of Epochs assuming background star tracks.

        Args:
            days (np.array of float): Array of Julian days (Terrestrial Time) at which to calculate the offsets.
            param (np.array of float): Array of host star and background star parameters.

        Returns:
            tuple of arrays: RA and DEC offsets from host star position at Epochs.
        """

        jd_start, jd_end, number = ephem_open() # can't we do this in the System class?

        if len(param) == 4:
            ra, dec, pmra, pmdec = param
            par=0
            host_icrs=self.host_icrs
        elif len(param) == 5:
            ra, dec, pmra, pmdec, par = param
            host_icrs=self.host_icrs

        else:
            ra, dec, pmra, pmdec, par, ra_host, dec_host, pmra_host, pmdec_host, par_host, rv_host = param

            host_gaia= novas.make_cat_entry(star_name="HST", catalog="HIP", star_num=1,
                                          ra=ra_host/15., dec=dec_host, pm_ra=pmra_host, pm_dec=pmdec_host,
                                          parallax=par_host, rad_vel=rv_host)

            host_icrs = novas.transform_cat(option=1, incat=host_gaia, date_incat=self.gaia_epoch,
                                         date_newcat=2000., newcat_id="HIP")

        star2_gaia = novas.make_cat_entry(star_name="BGR", catalog="HIP", star_num=2,
                                          ra=ra/15., dec=dec, pm_ra=pmra, pm_dec=pmdec,
                                          parallax=par, rad_vel=0)

        star2_icrs = novas.transform_cat(option=1, incat=star2_gaia, date_incat=self.gaia_epoch,
                                         date_newcat=2000., newcat_id="HIP")

        posx=[]
        posy=[]
        for i, day in enumerate(days):
            raa,deca = novas.app_star(day,host_icrs)
            rab,decb = novas.app_star(day,star2_icrs)
            c_a=SkyCoord(raa,deca,unit=("hourangle","deg"))
            c_b=SkyCoord(rab,decb,unit=("hourangle","deg"))
            offset=c_a.spherical_offsets_to(c_b)
            position_x=offset[0].mas
            position_y=offset[1].mas
            posx.append(position_x)
            posy.append(position_y)

        return np.array(posx),np.array(posy)

    def fmodel(self, param):
        """
        Function that models the offset of the background star with respect to the host star at the observed epochs.

        Args:
            param (np.array of float): set of parameters describing 5D position of background star and 6D position of host star.

        Returns:
            RA and DEC offsets at given observed epochs.
        """
        return self.radecdists(self.epochs_tt, param)

    def loglike(self, param):
        """
        Function to calculate Log likelihood for correlated (e.g., GRAVITY) and uncorrelated datapoints given certain model parameters.

        Args:
            param (np.array of float): set of parameters describing 5D position of background star and 6D position of host star.

        Returns:
            Loglikelihood values given the observed datapoints and a set of model parameters. Returns -np.inf if a parameter is nan or inf.
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
        """Transforms samples `u` drawn from the unit cube
        to those drawn from our prior for each variable.

        Args:
            u (np.array of float): samples of unit cube (0 to 1).

        Returns:
            tuple of np.array containing samples drawn from all prior distributions
        """

        param = np.array(u) # copy u

        if len(param) == 11:
            ra, dec, pmra, pmdec, par, ra_host, dec_host, pmra_host, pmdec_host, par_host, rv_host = param

            ra_host, dec_host, pmra_host, pmdec_host, par_host = self.HostStarPriors.transform_normal_multivariate(np.c_[ra_host,dec_host,pmra_host,pmdec_host,par_host])
            if self.rv_host_method == 'uniform':
                rv_host = transform_uniform(rv_host, self.rv_lower, self.rv_upper)
            else: # rv_host_method == 'normal' or 'gaia'
                rv_host = transform_normal(rv_host, self.radvelo, self.sig_rv)
            # truncate distribution at 100 kpc (Nielsen+ 2017 do this at 10 kpc)
            par = 1000/transform_gengamm(par, self.L, self.alpha, self.beta) # [units of mas]
            if par < 1e-2:
                par = -np.inf
        elif len(param) == 5:
            ra, dec, pmra, pmdec, par = param # unpacking unit cube samples

            # the PPF of Bailer-Jones 2015 eq. 17
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

        if len(param) == 11:
            param = ra, dec, pmra, pmdec, par, ra_host, dec_host, pmra_host, pmdec_host, par_host, rv_host
        elif len(param) == 5:
            param = ra, dec, pmra, pmdec, par
        else:
            param = ra, dec, pmra, pmdec

        return param

    def fit(self, dlogz=0.5, npool=4, dynamic=False, nlive=200, mpi_pool=False, resume=False, sample_method='unif'):
        """
        Function that fits the data using dynesty.

        Args:
            dlogz (float): Dynesty cumulative log-evidence stop criterium. Default: 0.5
            npool (int): Number of CPU threads to assign to pool. Default: 4
            dynamic (bool): Sets option to dynamically allocate live points to improve posterior estimation. Default: False
            nlive (int): Constant number of live points to set in non-dynamic case. Default: 200
            mpi_pool (bool): Sets option to use MPI multithreading. Default: False
            resume (bool): Sets option to resume from a previous result. Default: False
            sample_method: Default: 'unif'

        Returns:
            results (object): samples of fit
        """

        print('[BACKTRACK INFO]: Beginning sampling')
        ndim = self.ndim

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
        self.run_quant = np.quantile(samples, [0.1587, 0.8413], axis=0)

        # Compute useful chi2 value
        self.median_loglike = self.loglike(self.run_median)
        self.median_chi2_red = -2.*self.median_loglike/((2*len(self.epochs))-self.ndim)

        return self.results

    def save_results(self, fileprefix='./'):
        """
        Function to save fitting results in a dict to a .pkl to allow resuming.

        Args:
            fileprefix (str): Prefix to filename. Default "./" for current folder.
        """

        save_dict = {'med': self.run_median, 'quant': self.run_quant, 'results': self.results}
        target_label = self.target_name.replace(' ','_')
        file_name = f'{fileprefix}{target_label}_dynestyrun_results.pkl'
        print('[BACKTRACK INFO]: Saving results to {}'.format(file_name))
        pickle.dump(save_dict, open(file_name, "wb"))

    def load_results(self, fileprefix: str = './'):
        """
        Function to load fitting results in a dict from a .pkl to allow resuming.

        Args:
            fileprefix (str): Prefix to filename. Default "./" for current folder.
        """
        target_label = self.target_name.replace(' ', '_')
        file_name = f'{fileprefix}{target_label}_dynestyrun_results.pkl'
        print('[BACKTRACK INFO]: Loading results from {}'.format(file_name))
        save_dict = pickle.load(open(file_name, "rb"))
        self.run_median = save_dict['med']
        self.run_quant = save_dict['quant']
        self.results = save_dict['results']

        # recompute useful chi2 value
        self.median_loglike = self.loglike(self.run_median)
        self.median_chi2_red = -2.*self.median_loglike/((2*len(self.epochs))-self.ndim)

    def generate_plots(
            self,
            days_backward: float = 3.*365.,
            days_forward: float = 3.*365.,
            step_size: float = 10.,
            # ref_epoch: Optional[Tuple[int, int, int]] = None,
            plot_radec: bool = False,
            plot_stationary: bool = False,
            fileprefix: str = './',
            filepost: str = '.pdf',
        ) -> Tuple[Figure, Figure, Figure, Figure, Figure]:
        """
        Function to plot various fitting results, corner plot and diagnostic plot.

        Args:
            days_backward (float): Days backward from the reference point at which to start tracks
            days_forward (float): Days forward from the reference point at which to end tracks
            step_size (float): Step size with which tracks are generated.
            plot_radec (bool): Option to plot RA vs time and DEC vs time in panels. Default: False for Sep vs time, PA vs time.
            plot_stationary (bool): Option to overplot a track corresponding to an infinitely far stationary background star. Default: False.
            fileprefix (str): Prefix to filename. Default "./" for current folder.
            filepost (str): Postfix to filename. Default ".pdf" for outputting pdf files.

        Returns:
            tuple of figures: data and model tracks, posterior cornerplot, dynesty summary, parallax prior, gaia neighbourhood
        """

        print('[BACKTRACK INFO]: Generating Plots')
        ref_epoch = Time(self.ref_epoch, format='jd')

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
        print('[BACKTRACK INFO]: Plots saved to {}'.format(fileprefix))
        return fig_track, fig_post, fig_diag, fig_prior, fig_hood

    def generate_stationary_plot(
            self,
            days_backward: float = 3.*365.,
            days_forward: float = 3.*365.,
            step_size: float = 10.,
            # ref_epoch: Optional[Tuple[int, int, int]] = None,
            plot_radec: bool = False,
            fileprefix: str = './',
            filepost: str = '.pdf',
        ) -> Figure:
        """
        Function to plot an infinitely far stationary scenario track with the data.

        Args:
            days_backward (float): Days backward from the reference point at which to start tracks
            days_forward (float): Days forward from the reference point at which to end tracks
            step_size (float): Step size with which tracks are generated.
            plot_radec (bool): Option to plot RA vs time and DEC vs time in panels. Default: False for Sep vs time, PA vs time.
            fileprefix (str): Prefix to filename. Default "./" for current folder.
            filepost (str): Postfix to filename. Default ".pdf" for outputting pdf files.

        Returns:
            fig_track (figure): Figure object corresponding to the saved plot.
        """

        print('[BACKTRACK INFO]: Generating Stationary plot')
        ref_epoch = Time(self.ref_epoch, format='jd')

        fig_track = stationtrackplot(
            self,
            ref_epoch=ref_epoch.jd,
            days_backward=days_backward,
            days_forward=days_forward,
            step_size=step_size,
            plot_radec=plot_radec,
            fileprefix=fileprefix,
            filepost=filepost
        )
        print('[BACKTRACK INFO]: Stationary plot saved to {}'.format(fileprefix))
        return fig_track
