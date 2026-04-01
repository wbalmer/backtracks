.. _index:

Documentation for *backtracks*
==============================

.. image:: https://img.shields.io/pypi/v/backtracks
   :target: https://pypi.python.org/pypi/backtracks

.. image:: https://github.com/wbalmer/backtracks/actions/workflows/main.yml/badge.svg
   :target: https://github.com/wbalmer/backtracks/actions

.. image:: https://img.shields.io/readthedocs/backtracks
   :target: http://backtracks.readthedocs.io

.. image:: https://img.shields.io/github/license/wbalmer/backtracks
   :target: https://github.com/wbalmer/backtracks/blob/main/LICENSE


*backtracks* is a tool to fit relative astrometry with background star motion tracks.

The code is written and developed by Gilles Otten (@gotten), William Balmer (@wbalmer), and Tomas Stolker (@tomasstolker).

Attribution
===========

If you use `backtracks` in your published work, please cite our Zenodo entry (`here <https://doi.org/10.5281/zenodo.14838370>`_), and provide a footnote/acknowledgement linking to our package. An example bibtex citation is included below, but you may wish to cite a specific version of the package via zenodo instead. Thank you!

:: 

   @software{backtracks_code,
        author       = {William O. Balmer and
                        Gilles P. P. L. Otten and
                        Tomas Stolker},
        title        = {backtracks: a python package to compare relative astrometry with background helical motion: v0.6},
        month        = feb,
        year         = 2025,
        publisher    = {Zenodo},
        version      = {v0.6},
        doi          = {10.5281/zenodo.14838369},
        url          = {https://doi.org/10.5281/zenodo.14838369},
      }

Details
=======

* High precision relative astrometry calculations with USNO's `NOVAS` via the `python implementation<https://pypi.org/project/novas/>`. Thanks to Brandon Rhodes for maintaining this python package.

* Example of HD 131399Ab uses data from `Wagner et al. (2022) <https://ui.adsabs.harvard.edu/abs/2022AJ....163...80W/abstract>`_ and `Nielsen et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017AJ....154..218N/abstract>`_. Thank you to Kevin Wagner for providing the latest astrometry!

* eDR3 Distance prior summary file from `Bailer-Jones et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021AJ....161..147B/abstract>`_.

* Log-likelihood and some utility functions borrowed heavily from `orbitize! <https://github.com/sblunt/orbitize/>`_ (BSD 3-clause).

* PPF of multivariate normal borrowed from `pints <https://github.com/pints-team/pints>`_ (BSD 3-clause).


Index
=====

.. toctree::
   :maxdepth: 2

   installation
   tutorial.ipynb
   compare-real-planet.ipynb
   additional-examples.ipynb
   in_the_wild
   modules
   .. about
