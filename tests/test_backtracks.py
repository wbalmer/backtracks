import os
import pytest

import dynesty
from backtracks import System

class TestBacktracks:
    def setup_class(self) -> None:

        self.test_dir = os.path.dirname(__file__) + "/"

        self.bt_system = System(
            "HD 131399 A",
            f"{self.test_dir}scorpions1b_orbitizelike.csv",
            nearby_window=0.5,
            ref_epoch_idx=0,
            fileprefix=self.test_dir,
        )

    def teardown_class(self) -> None:

        file_list = [
            "HD_131399_A_bjprior_backtracks.png",
            "HD_131399_A_corner_backtracks.png",
            "HD_131399_A_evidence_backtracks.png",
            "HD_131399_A_model_backtracks.png",
            "HD_131399A_stationary_backtracks.png",
            "HD_131399_A_nearby_gaia_distribution.png",
            "HD_131399_A_dynestyrun_results.pkl",
            "gaia_query_6204835284262018688.fits",
            "dynesty.save"
        ]

        for file_item in file_list:
            if os.path.exists(self.test_dir + file_item):
                os.remove(self.test_dir + file_item)

    def test_system(self) -> None:

        assert isinstance(self.bt_system, System)

    def test_statplot(self) -> None:

        self.bt_system.generate_stationary_plot(
            days_backward=3.*365.,
            days_forward=3.*365.,
            step_size=50.,
            plot_radec=False,
            fileprefix=self.test_dir,
            filepost='.png',
        )

    def test_fit(self) -> None:

        results = self.bt_system.fit(dlogz=0.5, npool=4, dynamic=False, mpi_pool=False, nlive=100, resume=False)

        assert isinstance(results, dynesty.utils.Results)

    def test_save(self) -> None:

        self.bt_system.save_results(fileprefix=self.test_dir)

    def test_load(self) -> None:

        self.bt_system.load_results(fileprefix=self.test_dir)

    def test_plot(self) -> None:

        self.bt_system.generate_plots(
            days_backward=3.*365.,
            days_forward=3.*365.,
            step_size=50.,
            plot_radec=False,
            plot_stationary=False,
            fileprefix=self.test_dir,
            filepost='.png',
        )
