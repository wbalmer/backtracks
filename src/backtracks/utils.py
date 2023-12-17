# backtrack utility functions. stand on the shoulders of giants, its fun!

# imports
import orbitize.system
import numpy as np
from scipy.stats import norm
from scipy.special import gammainc, gammaincc, gammainccinv, gammaincinv


def pol2car(sep, pa, seperr, paerr, corr=np.nan):
    ra, dec = orbitize.system.seppa2radec(sep, pa)
    raerr, decerr, corr2 = orbitize.system.transform_errors(sep, pa, seperr, paerr, corr, orbitize.system.seppa2radec)
    return ra, dec, decerr, raerr, corr2


def transform_uniform(x,a,b):
    return a + (b-a)*x


def transform_normal(x, mu, sigma):
    return norm.ppf(x, loc=mu, scale=sigma)


def transform_gengamm(x, L=1.35e3, alpha=1, beta=2):
    return L*(gammaincinv((beta+1)/alpha,x)**(1/alpha))


# plx_pdf = lambda par,L : (par**(-4))*np.exp(-1/(par*L))
#
# plx_cdf = lambda par,L : gammaincc(3, 1/(L*par))*L**3
#
# plx_ppf = lambda x,L : gammainccinv(3, (x/(L**3)))/L
