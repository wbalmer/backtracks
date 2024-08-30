.. _about:

About
=====

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

Attribution
-----------

If you use `backtracks` in your published work, please cite our Astrophysics Source Code Library entry, https://ascl.net/code/v/3755, and/or provide a footnote/acknowledgement linking to our package. Thank you!

Details
-------

* eDR3 Distance prior summary file from `Bailer-Jones et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021AJ....161..147B/abstract>`_.

* Example of HD 131399Ab uses data from `Wagner et al. (2022) <https://ui.adsabs.harvard.edu/abs/2022AJ....163...80W/abstract>`_ and `Nielsen et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017AJ....154..218N/abstract>`_. Thank you to Kevin Wagner for providing the latest astrometry!

* Log-likelihood and some utility functions borrowed heavily from `orbitize! <https://github.com/sblunt/orbitize/>`_ (BSD 3-clause).

* PPF of multivariate normal borrowed from `pints <https://github.com/pints-team/pints>`_ (BSD 3-clause).
