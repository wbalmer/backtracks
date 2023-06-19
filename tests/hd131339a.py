import sys
sys.path.append('..')
from backtrack.backtrack import backtrack

if __name__ == '__main__':
    mytrack = backtrack("HD 131399 A", "scorpions1b_orbitizelike.csv", nearby_window=0.5)
    results = mytrack.fit()
    mytrack.generate_plots()
    mytrack.save_results()
    mytrack.load_results()
