backtracks
==========

.. image:: docs/_static/backtracks-logo-dark.svg
   :align: center



.. image:: https://img.shields.io/pypi/v/backtracks
   :target: https://pypi.python.org/pypi/backtracks

.. image:: https://github.com/wbalmer/backtracks/actions/workflows/main.yml/badge.svg
   :target: https://github.com/wbalmer/backtracks/actions

.. image:: https://img.shields.io/readthedocs/backtracks
   :target: http://backtracks.readthedocs.io

.. image:: https://img.shields.io/github/license/wbalmer/backtracks
   :target: https://github.com/wbalmer/backtracks/blob/main/LICENSE



`backtracks` is a python package to fit relative astrometry with background helical motion tracks, to discern directly imaged planets :ringed_planet: from contaminant sources :dizzy: :star:

The code is written and developed by Gilles Otten (@gotten), William Balmer (@wbalmer), and Tomas Stolker (@tomasstolker).

Documentation
-------------

Documentation can be found at `http://backtracks.readthedocs.io <https://backtracks.readthedocs.io/en/latest/>`_.

Tutorial
--------

A `Jupyter notebook <https://backtracks.readthedocs.io/en/latest/tutorial.html>`_ will show you how to use `backtracks` by reproducing the result in `Nielsen et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017AJ....154..218N/abstract>`_ and `Wagner et al. (2022) <https://ui.adsabs.harvard.edu/abs/2022AJ....163...80W/abstract>`_ for the case of the former exoplanet candidate around HD 131339 A.


Details
-------

* eDR3 Distance prior summary file from `Bailer-Jones et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021AJ....161..147B/abstract>`_.

* Example of HD 131399Ab uses data from `Wagner et al. (2022) <https://ui.adsabs.harvard.edu/abs/2022AJ....163...80W/abstract>`_ and `Nielsen et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017AJ....154..218N/abstract>`_. Thank you to Kevin Wagner for providing the latest astrometry!

* Log-likelihood and some utility functions borrowed heavily from `orbitize! <https://github.com/sblunt/orbitize/>`_ (BSD 3-clause).

* PPF of multivariate normal borrowed from `pints <https://github.com/pints-team/pints>`_ (BSD 3-clause).

Installation
============

Currently requires and python 3.9-3.11 ish and `astropy`, `corner`, `dynesty`, `matplotlib`, `numpy`, `novas`, `novas_de405`, and their dependencies. Note that `novas` is not supported on Windows. You can create a working environment using conda+pip via a few lines of code:

.. code-block:: console

    $ conda create python=3.11 -n backtrack
    $ conda activate backtrack
    $ conda install pip
    $ pip install backtracks

Or, to clone the repo and install in development mode (we recommend this, as the code is a work in progress and you can easily fix bugs you will likely encounter this way):

.. code-block:: console

    $ conda create python=3.11 -n backtrack
    $ conda activate backtrack
    $ conda install pip
    $ git clone https://github.com/wbalmer/backtrack.git
    $ cd backtrack
    $ pip install -e .

Then, test your installation:

.. code-block:: python

    >>> from backtracks import System
