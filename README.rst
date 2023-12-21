*backtracks*
============

.. image:: https://img.shields.io/pypi/v/backtracks
   :target: https://pypi.python.org/pypi/backtracks

.. image:: https://github.com/wbalmer/backtracks/actions/workflows/main.yml/badge.svg
   :target: https://github.com/wbalmer/backtracks/actions

.. image:: https://img.shields.io/readthedocs/backtracks
   :target: http://backtracks.readthedocs.io

.. image:: https://img.shields.io/github/license/wbalmer/backtracks
   :target: https://github.com/wbalmer/backtracks/blob/main/LICENSE

:stars: Python package to fit relative astrometry with background helical motion tracks, to discern directly imaged planets :ringed_planet: from contaminant sources.

Written by Gilles Otten (@gotten), William Balmer (@wbalmer), and Tomas Stolker (@tomasstolker).

Documentation
-------------

Documentation can be found at `http://backtracks.readthedocs.io <http://backtracks.readthedocs.io>`_.

Tutorial
--------

A `Jupyter notebook <http://backtracks.readthedocs.io/en/latest/tutorials.html>`_. will show you how to use `backtracks` by reproducing the result in `Nielsen et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017AJ....154..218N/abstract>`_. and `Wagner et al. (2022) <https://ui.adsabs.harvard.edu/abs/2022AJ....163...80W/abstract>`_. for the case of the former exoplanet candidate around HD 131339 A.


Details
-------

eDR3 Distance prior summary file from this [source](https://arxiv.org/pdf/2012.05220.pdf), published in [Bailer-Jones+2021](https://arxiv.org/abs/2012.05220).

Current example (HD131399Ab) uses data from Wagner+22 and Nielsen+17. Thank you to Kevin Wagner for providing the latest astrometry!

Log-likelihood and some utility functions borrowed heavily from `orbitize!` (BSD 3-clause).

