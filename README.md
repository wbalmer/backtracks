# backtrack
Python package to fit relative astrometry with background star motion tracks.

Written by Gilles Otten (@gotten) and William Balmer (@wbalmer).

Work in progress, as of Jun. 15th, 2023.

eDR3 Distance prior summary file from this [source](https://arxiv.org/pdf/2012.05220.pdf), published in [Bailer-Jones+2021](https://arxiv.org/abs/2012.05220).

Log-likelihood borrowed heavily from `orbitize!` (BSD 3-clause).

Currently requires and python 3.9 ish and `astropy`, `corner`, `dynesty`, `matplotlib`, `numpy`, `novas`, `novas_de405`, `orbitize` and their dependencies.
