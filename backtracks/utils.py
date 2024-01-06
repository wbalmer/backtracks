# backtracks utility functions. stand on the shoulders of giants, its fun!

import numpy as np
from scipy.stats import norm
from scipy.special import gammainc, gammaincc, gammainccinv, gammaincinv
from astropy.time import Time

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
        * Parameters L, alpha and beta are equivalent to parameters a, p and d-1 from the original paper by Stacy (1962). 
    """

    return L*(gammaincinv((beta+1)/alpha,x)**(1/alpha))

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
        mean (np.array of float): M array of mean values of multivariate normal
        cov (np.array of float): M by M array of covariance matrix of multivariate normal

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
            x (np.array of float): N by M array of values between 0 and 1 used to draw values from the distribution

        Returns:
            tuple: (M np.arrays of length N)
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
