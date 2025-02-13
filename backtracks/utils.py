# backtracks utility functions. stand on the shoulders of giants, its fun!

import numpy as np
from scipy.stats import norm
from scipy.special import gammainc, gammaincc, gammainccinv, gammaincinv
from astropy.time import Time
import novas.compat as novas
from novas.compat.eph_manager import ephem_open
from astropy.coordinates import SkyCoord

def radecdists(backtracks, days, param): # for multiple epochs
        """
        Function that calculates the offset between companion and host star at a certain set of Epochs assuming background star tracks.

        Args:
            backtracks (class): backtracks.System class which carries the needed methods and attributes. 
            days (np.array of float): Array of Julian days (Terrestrial Time) at which to calculate the offsets.
            param (np.array of float): Array of host star and background star parameters.

        Returns:
            tuple of arrays: RA and DEC offsets from host star position at Epochs.
            
        References:
            * Bangert, J., Puatua, W., Kaplan, G., Bartlett, J., Harris, W., Fredericks, A., & Monet, A. (2011) User's Guide to NOVAS Version C3.1 (Washington, DC: USNO).
            * Barron, E. G., Kaplan, G. H., Bangert, J., Bartlett, J. L., Puatua, W., Harris, W., & Barrett, P. (2011) Bull. AAS, 43, 2011.
            * `NOVAS website <https://aa.usno.navy.mil/software/novas_info>`__
            
        Notes:
            * The NOVAS function `app_star` does the heavy lifting here and assumes a geocentric observer location. \
            It takes into account gravitational lensing by solar system bodies, stellar aberration, 3D motion and parallax.
            * For Proxima Centauri you may expect a 33 uas offset throughout a day due to an additional small parallax component \
            (up to 1 Earth radius offset of the observer w.r.t. geocenter). `topo_star` (not implemented in backtracks) would have to be used to account for this effect.
            * The default ephemeris file in `backtracks` is DE405 (valid from 1599 DEC 09 to 2201 FEB 20). To extend time coverage use DE430/DE440.
            * Additional JPL Ephemeris files are found `here <https://ssd.jpl.nasa.gov/planets/eph_export.html>`__ \
            and `here <https://ssd.jpl.nasa.gov/ftp/eph/planets/>`__
        """

        jd_start, jd_end, number = ephem_open() # can't we do this in the System class?

        if len(param) == 2:
            ra, dec = param
            par = 0.0
            pmra = 0.0
            pmdec = 0.0
            host_icrs=backtracks.host_icrs
        elif len(param) == 4:
            ra, dec, pmra, pmdec = param
            par=0
            host_icrs=backtracks.host_icrs
        elif len(param) == 5:
            ra, dec, pmra, pmdec, par = param
            host_icrs=backtracks.host_icrs

        else:
            ra, dec, pmra, pmdec, par, ra_host, dec_host, pmra_host, pmdec_host, par_host, rv_host = param

            host_gaia= novas.make_cat_entry(star_name="HST", catalog="HIP", star_num=1,
                                          ra=ra_host/15., dec=dec_host, pm_ra=pmra_host, pm_dec=pmdec_host,
                                          parallax=par_host, rad_vel=rv_host)

            host_icrs = novas.transform_cat(option=1, incat=host_gaia, date_incat=backtracks.gaia_epoch,
                                         date_newcat=2000., newcat_id="HIP")

        star2_gaia = novas.make_cat_entry(star_name="BGR", catalog="HIP", star_num=2,
                                          ra=ra/15., dec=dec, pm_ra=pmra, pm_dec=pmdec,
                                          parallax=par, rad_vel=0)

        star2_icrs = novas.transform_cat(option=1, incat=star2_gaia, date_incat=backtracks.gaia_epoch,
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

def pol2car(sep, pa, seperr, paerr, corr=np.nan):
    """
    This function converts Separation, PA, and their errors to deltaRA, deltaDEC and their errors.

    Args:
        sep (np.array of float): array of separation values in mas 
        pa (np.array of float): array of position angle values in degrees
        seperr (np.array of float): array of separation error values in mas
        paerr (np.array of float): array of position angle error values in degrees
        corr (np.array of float): array of correlation values [-1,1] default: nan

    Returns:
        tuple of float: (ra [mas], dec [mas], raerr [mas], decerr [mas], corr2 [-])
    """

    ra, dec = seppa2radec(sep, pa)
    raerr, decerr, corr2 = transform_errors(sep, pa, seperr, paerr, corr, seppa2radec)
    return ra, dec, raerr, decerr, corr2

def radec2seppa(ra, dec, mod180=False):
    """
    This function is reproduced here from the orbitize! package, written by S. Blunt et al and distributed under the BSD 3-Clause License
    Convenience function for converting from
    right ascension/declination to separation/
    position angle.

    Args:
        ra (np.array of float): array of RA values, in mas
        dec (np.array of float): array of Dec values, in mas
        mod180 (Bool): if True, output PA values will be given
            in range [180, 540) (useful for plotting short
            arcs with PAs that cross 360 during observations)
            (default: False)

    Returns:
        tuple of float: (separation [mas], position angle [deg])
    """

    sep = np.sqrt((ra**2) + (dec**2))
    pa = np.degrees(np.arctan2(ra, dec)) % 360.

    if mod180:
        pa[pa < 180] += 360

    return sep, pa


def seppa2radec(sep, pa):
    """
    This function is reproduced here from the orbitize! package, written by S. Blunt et al and distributed under the BSD 3-Clause License
    Convenience function to convert sep/pa to ra/dec

    Args:
        sep (np.array of float): array of separation in mas
        pa (np.array of float): array of position angles in degrees

    Returns:
        tuple: (ra [mas], dec [mas])
    """

    ra = sep * np.sin(np.radians(pa))
    dec = sep * np.cos(np.radians(pa))

    return ra, dec


def transform_errors(x1, x2, x1_err, x2_err, x12_corr, transform_func, nsamps=100000):
    """
    This function is reproduced here from the orbitize! package, written by S. Blunt et al and distributed under the BSD 3-Clause License
    Transform errors and covariances from one basis to another using a Monte Carlo
    approach

   Args:
        x1 (float): planet location in first coordinate (e.g., RA, sep) before
            transformation
        x2 (float): planet location in the second coordinate (e.g., Dec, PA)
            before transformation)
        x1_err (float): error in x1
        x2_err (float): error in x2
        x12_corr (float): correlation between x1 and x2
        transform_func (function): function that transforms between (x1, x2)
            and (x1p, x2p) (the transformed coordinates). The function signature
            should look like: `x1p, x2p = transform_func(x1, x2)`
        nsamps (int): number of samples to draw more the Monte Carlo approach.
            More is slower but more accurate.
    Returns:
        tuple (x1p_err, x2p_err, x12p_corr): the errors and correlations for
            x1p,x2p (the transformed coordinates)
    """

    if np.isnan(x12_corr):
        x12_corr = 0.

    # construct covariance matrix from the terms provided
    cov = np.array([[x1_err**2, x1_err*x2_err*x12_corr], [x1_err*x2_err*x12_corr, x2_err**2]])

    samps = np.random.multivariate_normal([x1, x2], cov, size=nsamps)

    x1p, x2p = transform_func(samps[:,0], samps[:, 1])

    x1p_err = np.std(x1p)
    x2p_err = np.std(x2p)
    x12_corr = np.corrcoef([x1p, x2p])[0,1]

    return x1p_err, x2p_err, x12_corr


def transform_uniform(x,a,b):
    """
    This function draws values from the uniform distribution when given values between 0 and 1.
    
    Args:
        x (np.array of float): array of unit size values (0 to 1) used to draw values from the distribution
        a (float): lower limit of the uniform distribution
        b (float): upper limit of the uniform distribution
    Returns:
        Values drawn from uniform distribution
    """

    return a + (b-a)*x


def transform_normal(x, mu, sigma):
    """
    This function draws values from the normal distribution when given values between 0 and 1.
    
    Args:
        x (np.array of float): array of unit size values (0 to 1) used to draw values from the distribution
        mu (float): mean of the normal distribution
        sigma (float): standard deviation of the normal distribution

    Returns:
        Values drawn from normal distribution
    """

    return norm.ppf(x, loc=mu, scale=sigma)


def transform_gengamm(x, L=1.35e3, alpha=1, beta=2):
    """
    This function draws distances in parsec from the generalized gamma distribution (GGD) when given values between 0 and 1.

    Args:
        x (np.array of float): array of unit size values (0 to 1) used to draw values from the distribution
        L (float): scale parameter of the GGD [parsec]
        alpha (float): shape parameter
        beta (float): shape parameter

    Returns:
        Values drawn from generalized gamma distribution in parsec

    References:
        * Bailer-Jones, C. A. L. et al 2021 AJ 161 147 (DR3 distances; The GGD used as basis for this PPF is defined by Equation 3)
        * Bailer-Jones, C. A. L. et al 2018 AJ 156 58 (DR2 distances; Special case of GGD)
        * Stacy, E. W. (1962). A Generalization of the Gamma Distribution. The Annals of Mathematical Statistics, 33(3), 1187–1192. 

    Notes:
        * The exponentially decreasing space density (EDSD) of Bailer-Jones et al 2018 (DR2) is equivalent to the GGD with alpha=1 and beta=2.
        * Parameters `L`, `alpha` and `beta` are equivalent to parameters `a`, `p` and `d-1` from the original paper by Stacy (1962). 
    """

    return L*(gammaincinv((beta+1)/alpha,x)**(1/alpha))

def iso2mjd(yrs):
    """
    This function converts iso times to Modified Julian Dates in UTC
    
    Args:
        yrs (str): array with iso epochs (YYYYMMDD)

    Returns:
        Returns Modified Julian Dates in UTC. 
    """

    return Time(yrs,scale="utc", format='iso').mjd

def utc2tt(jd_utc):
    """
    This function converts Julian Dates in UTC to Julian Dates in TT (Terrestrial Time)
    
    Args:
        jd_utc (float): array with Julian Dates

    Returns:
        Returns Julian Dates that are the TT equivalent of the input UTC JD. 
    """

    return Time(jd_utc,scale="utc",format="jd").tt.jd

class HostStarPriors(): 
    """
    Class to draw values from multivariate normal distribution using pseudoinverse.

    Args:
        mean (np.array of float): `M` array of mean values of multivariate normal
        cov (np.array of float): `M` by `M` array of covariance matrix of multivariate normal

    Notes:
        * This class is a stripped down version of MultivariateGaussianLogPrior in `pints <https://github.com/pints-team/pints/blob/main/pints/_log_priors.py>`__ (BSD3-clause)
    """

    def __init__(self, mean, cov): # setting up distribution, this is done only once so we dont have to bother with a fast Cholesky inversion. Host star parameter distribution doesnt change.
        self._mean = mean
        self._cov = cov
        self._n_parameters = mean.shape[0]
        
        self._sigma12_sigma22_inv_l = []
        self._sigma_bar_l = []
        self._mu1 = []
        self._mu2 = []
        
        for j in range(1, self._n_parameters):
            sigma = self._cov[0:(j + 1), 0:(j + 1)]
            dims = sigma.shape
            sigma11 = sigma[dims[0] - 1, dims[1] - 1]
            sigma22 = sigma[0:(dims[0] - 1), 0:(dims[0] - 1)]
            sigma12 = sigma[dims[0] - 1, 0:(dims[0] - 1)]
            sigma21 = sigma[0:(dims[0] - 1), dims[0] - 1]
            mean = self._mean[0:dims[0]]
            mu2 = mean[0:(dims[0] - 1)]
            mu1 = mean[dims[0] - 1]
            sigma12_sigma22_inv = np.matmul(sigma12,
                                            np.linalg.inv(sigma22))
            sigma_bar = np.sqrt(sigma11 - np.matmul(sigma12_sigma22_inv,
                                                    sigma21))
            self._sigma12_sigma22_inv_l.append(sigma12_sigma22_inv)
            self._sigma_bar_l.append(sigma_bar)
            self._mu1.append(mu1)
            self._mu2.append(mu2)
   
    def transform_normal_multivariate(self, x): # pulling from distribution with random number between 0 and 1
        """
        This function returns values drawn from the multivariate normal when given numbers between 0 and 1.
        
        Args:
            x (np.array of float): `N` by `M` array of values between 0 and 1 used to draw values from the distribution

        Returns:
            tuple: (`M` np.arrays of length `N`)
        """

        n_samples = x.shape[0]
        n_params = x.shape[1]
        
        icdfs = np.zeros((n_samples, n_params))
        for j in range(n_samples):
            for i in range(self._n_parameters):
                if i == 0: # the first axis just takes the default mean and sigma and draws with a ppf
                    mu = self._mean[0]
                    sigma = np.sqrt(self._cov[0, 0])
                else: # the next axis needs to bias the mean and sigma of the ppf based on the previously drawn numbers
                    sigma = self._sigma_bar_l[i - 1]
                    mu = self._mu1[i - 1] + np.matmul(
                        self._sigma12_sigma22_inv_l[i - 1],
                        (np.array(icdfs[j, 0:i]) - self._mu2[i - 1]))
                icdfs[j, i] = norm.ppf(x[j, i], mu, sigma) 
        return np.squeeze(np.array_split(icdfs,n_params,axis=1))
