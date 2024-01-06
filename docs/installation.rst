.. _installation:

Installation
============

Currently requires and python 3.9-3.11 ish and `astropy`, `corner`, `dynesty`, `matplotlib`, `numpy`, `novas`, `novas_de405` and their dependencies. Note that `novas` is not supported on Windows. You can create a working environment using conda+pip via a few lines of code:

.. code-block:: console

    $ conda create python=3.11 -n backtracks
    $ conda activate backtracks
    $ conda install pip
    $ pip install backtracks

Or, to clone the repo and install in development mode (we recommend this, as the code is a work in progress and you can easily fix bugs you will likely encounter this way):

.. code-block:: console

    $ conda create python=3.11 -n backtracks
    $ conda activate backtracks
    $ conda install pip
    $ git clone https://github.com/wbalmer/backtracks.git
    $ cd backtracks
    $ pip install -e .

Then, test your installation:

.. code-block:: python

    >>> from backtracks import System
