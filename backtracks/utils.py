# backtracks utility functions. stand on the shoulders of giants, its fun!

import numpy as np
from scipy.stats import norm
from scipy.special import gammainc, gammaincc, gammainccinv, gammaincinv


def pol2car(sep, pa, seperr, paerr, corr=np.nan):
    ra, dec = seppa2radec(sep, pa)
    raerr, decerr, corr2 = transform_errors(sep, pa, seperr, paerr, corr, seppa2radec)
    return ra, dec, decerr, raerr, corr2


def radec2seppa(ra, dec, mod180=False):
    """
    This function is reproduced here from the orbitize! pacakge, written by S. Blunt et al and distributed under the BSD 3-Clause License
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
    This function is reproduced here from the orbitize! pacakge, written by S. Blunt et al and distributed under the BSD 3-Clause License
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
    This function is reproduced here from the orbitize! pacakge, written by S. Blunt et al and distributed under the BSD 3-Clause License
    Transform errors and covariances from one basis to another using a Monte Carlo
    apporach

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
    return a + (b-a)*x


def transform_normal(x, mu, sigma):
    return norm.ppf(x, loc=mu, scale=sigma)


def transform_gengamm(x, L=1.35e3, alpha=1, beta=2):
    return L*(gammaincinv((beta+1)/alpha,x)**(1/alpha))
