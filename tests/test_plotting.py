from backtracks import backtrack

if __name__ == '__main__':
    mytrack = backtrack.system("HD 131399 A", "scorpions1b_orbitizelike.csv", nearby_window=0.5)
    mytrack.load_results()
    mytrack.generate_plots(days_backward=2600, days_forward=0, plot_radec=True, plot_stationary=True)
    
    