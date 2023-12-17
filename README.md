# backtrack
Python package to fit relative astrometry with background star motion tracks.

Written by Gilles Otten (@gotten), William Balmer (@wbalmer), and Tomas Stolker (@tomasstolker).

eDR3 Distance prior summary file from this [source](https://arxiv.org/pdf/2012.05220.pdf), published in [Bailer-Jones+2021](https://arxiv.org/abs/2012.05220).

Current example (HD131399Ab) uses data from Wagner+22 and Nielsen+17. Thank you to Kevin Wagner for providing the latest astrometry!

Log-likelihood borrowed heavily from `orbitize!` (BSD 3-clause).

Currently requires and python 3.9 ish and `astropy`, `corner`, `dynesty`, `matplotlib`, `numpy`, `novas`, `novas_de405`, `orbitize` and their dependencies. Note that `novas` is not supported on Windows. You can create a working environment using conda+pip via a few lines of code:

```
conda create python=3.9 -n backtrack
conda activate backtrack
conda install pip
pip install backtracks
```

Then, download the test data+script and test your installation (takes a while to sample fully):
```
wget https://raw.githubusercontent.com/wbalmer/backtrack/main/tests/scorpions1b_orbitizelike.csv
wget https://raw.githubusercontent.com/wbalmer/backtrack/main/tests/hd131339a.py
python hd131339a.py
```

or, to clone the repo and install in development mode (we recommend this, as the code is a work in progress and you can easily fix bugs you will likely encounter this way):

```
conda create python=3.9 -n backtrack
conda activate backtrack
conda install pip
git clone https://github.com/wbalmer/backtrack.git
cd backtrack
pip install -e .
```

Then, test your installation (takes a while to sample fully):

```
cd tests
python hd131399a.py
```
